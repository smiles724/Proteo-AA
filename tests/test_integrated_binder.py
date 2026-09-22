"""The integrated sampler, at the level testable without real donors.

Real-donor execution and binder quality are established on the cluster by the
positive control and the preflight, not here. What these pin are the contracts
that no run would reveal if they were wrong: the churn rule used to pick the
event, the same-state correction, the call accounting, and the tap through
which feedback is the only thing that actually reaches the model.
"""

import torch

import pytest

from pxf.bench.integrated import (DEFAULT_EVENT_SIGMA, EventChoice,
                                  _aligned_rmsd_local, select_event)


# --------------------------------------------------------------- churn rule


def test_churn_doubles_sigma_in_the_useful_range():
    """gamma0=1 => t_hat = 2 * scheduled, which is the whole point.

    A selector that compared the SCHEDULED level against the request would
    place the event where the denoiser sees twice the asked-for noise.
    """
    schedule = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    choice = select_event(schedule, 4.0)
    assert choice.churn_ratio == pytest.approx(2.0)
    assert choice.actual_sigma == pytest.approx(2 * choice.scheduled_sigma)


def test_selection_is_by_actual_not_scheduled_sigma():
    """Requesting 4.0 must pick the step whose CHURNED sigma is 4.0."""
    schedule = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    choice = select_event(schedule, 4.0)
    # scheduled 2.0 -> actual 4.0. The step whose *scheduled* level is 4.0
    # does not exist; the naive reading would have picked scheduled 5.0.
    assert choice.scheduled_sigma == pytest.approx(2.0)
    assert choice.actual_sigma == pytest.approx(4.0)


def test_the_gamma_threshold_is_on_the_next_level():
    """``gamma = gamma0 if c_tau > gamma_min else 0`` -- c_tau, not c_tau_last.

    At the tail this turns churn off, so actual stops being 2x scheduled. A
    selector thresholding on the current level would mis-model the last steps.
    """
    # step 0: c_tau=0.005 <= gamma_min=0.01 -> gamma off -> actual == scheduled
    schedule = torch.tensor([1.0, 0.005])
    choice = select_event(schedule, 1.0, gamma_min=0.01)
    assert choice.churn_ratio == pytest.approx(1.0)
    assert choice.actual_sigma == pytest.approx(1.0)


def test_nearest_actual_sigma_wins():
    schedule = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5])
    # actual levels are 16, 8, 4, 2 -> nearest to 5 is 4 (step 2)
    assert select_event(schedule, 5.0).step == 2
    assert select_event(schedule, 15.0).step == 0


def test_requested_sigma_is_recorded_alongside_the_realised_one():
    """The realised value is what A_BS is conditioned on; both get logged."""
    choice = select_event(torch.tensor([10.0, 5.0, 2.0, 1.0]), DEFAULT_EVENT_SIGMA)
    assert choice.requested_sigma == DEFAULT_EVENT_SIGMA
    assert choice.actual_sigma != choice.requested_sigma  # no exact level here
    assert set(choice.record()) >= {
        "event_step", "scheduled_sigma", "actual_sigma",
        "sigma_churn_ratio", "requested_sigma",
    }


def test_a_degenerate_schedule_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        select_event(torch.tensor([1.0]))


# ------------------------------------------------------- aligned displacement


def test_aligned_rmsd_removes_rigid_motion():
    """Every solver step re-augments, so the raw distance is mostly rotation."""
    x = torch.randn(40, 3)
    theta = torch.tensor(0.7)
    rot = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved = x @ rot.T + torch.tensor([5.0, -2.0, 1.0])
    assert _aligned_rmsd_local(x, moved) == pytest.approx(0.0, abs=1e-6)
    raw = float(torch.sqrt(((x - moved) ** 2).sum(-1).mean()))
    assert raw > 1.0


def test_aligned_rmsd_does_not_fit_a_mirror_image():
    """Without the determinant correction a reflection would score as perfect.

    That would report zero displacement for a structure of the wrong chirality
    and make the transfer assumption look better than it is.
    """
    x = torch.randn(40, 3)
    mirrored = x * torch.tensor([1.0, 1.0, -1.0])
    assert _aligned_rmsd_local(x, mirrored) > 0.1


def test_aligned_rmsd_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        _aligned_rmsd_local(torch.randn(10, 3), torch.randn(11, 3))


# ------------------------------------------------- the event, against doubles


class _StubDiffusion(torch.nn.Module):
    """Enough surface for BackboneTap to attach to."""

    def __init__(self, n_atom):
        super().__init__()
        self.layernorm_a = torch.nn.Identity()
        self.atom_attention_decoder = torch.nn.Identity()
        self.n_atom = n_atom


class _StubDenoiser:
    """Records every call and whether a tap came with it."""

    def __init__(self, n_atom=12, n_levels=5):
        self.model = type("M", (), {"diffusion_module": _StubDiffusion(n_atom)})()
        self._n_atom = n_atom
        self._schedule = torch.logspace(1, -1, n_levels)
        self.device = torch.device("cpu")
        self.calls = 0
        self.injections = 0
        self.saw_tap = []
        self.feedbacks = []

    @property
    def n_atom(self):
        return self._n_atom

    def schedule(self, n_step=None):
        return self._schedule

    def denoise(self, x_noisy, sigma, *, feedback=None, tap=None):
        self.calls += 1
        self.saw_tap.append(tap is not None)
        self.feedbacks.append(feedback)
        if feedback is not None:
            self.injections += 1
        return torch.zeros_like(x_noisy)


def _canned_products(n_tokens=4, n_atom=12, c_h_V=6):
    from pxf.couple.integrated_event import EventProducts
    from pxf.couple.visibility import PackedStructure, Visibility

    binder = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    vis = Visibility(
        available=torch.zeros(1, n_tokens, 37),
        missing_atom_mask=torch.ones(1, n_tokens, 37),
        frame_valid=torch.ones(1, n_tokens, dtype=torch.bool),
        sidechain_visible=torch.zeros(1, n_tokens),
        exists=torch.zeros(1, n_tokens, 37),
        stats={},
    )
    packed = PackedStructure(
        h_packed=torch.ones(1, n_tokens, c_h_V),
        coords37=torch.zeros(1, n_tokens, 37, 3),
        aatype=torch.zeros(1, n_tokens, dtype=torch.long),
        seq_mask=torch.ones(1, n_tokens),
        visibility=vis,
        psce=torch.zeros(1, n_tokens, 33),
    )
    return EventProducts(
        bb0=torch.zeros(1, n_atom, 3),
        a_token=torch.zeros(1, n_tokens, 8),
        residual=torch.zeros(1, n_tokens, c_h_V),
        aatype=torch.zeros(1, n_tokens, dtype=torch.long),
        sequence="AAAA",
        binder_sequence="AA",
        coords_af2=torch.zeros(1, n_tokens, 37, 3),
        atom_mask_af2=torch.zeros(1, n_tokens, 37),
        availability=torch.zeros(1, n_tokens, 37),
        h_base=None,
        h_packed=torch.ones(1, n_tokens, c_h_V),
        packed=packed,
        psce=torch.zeros(1, n_tokens, 33),
        binder_mask=binder,
        sigma=4.0,
        provenance={"binder_length": 2},
    )


def test_event_fires_exactly_once_and_adds_one_provisional_call(monkeypatch):
    """401 evaluations at 400 steps: the solver's, plus one provisional."""
    import pxf.bench.integrated as mod
    from pxf.couple import integrated_event as ev

    products = _canned_products()
    fired = {"n": 0}

    def fake_prepare_event(**kwargs):
        fired["n"] += 1
        kwargs["denoise"](kwargs["x_noisy"], torch.tensor([kwargs["sigma"]]))
        return products

    monkeypatch.setattr(ev, "prepare_event", fake_prepare_event)

    from pxf.bench.integrated import select_event
    from pxf.couple.replay import RngStream, run_trajectory
    from pxf.couple.pxdesign_iface import BackboneTap

    denoiser = _StubDenoiser(n_levels=5)
    schedule = denoiser.schedule()
    choice = select_event(schedule, 4.0)

    with BackboneTap(denoiser.model.diffusion_module) as tap:
        def denoise(x, s, *, feedback=None):
            return denoiser.denoise(x, s, feedback=feedback, tap=tap)

        def feedback(state):
            ev.prepare_event(
                denoise=lambda x, sg, **kw: denoiser.denoise(x, sg, tap=tap),
                x_noisy=state.x_noisy,
                sigma=float(state.sigma.reshape(-1)[0]),
                structure=None, designer=None,
            )
            return None

        run_trajectory(
            denoise=denoise, schedule=schedule, n_atom=denoiser.n_atom,
            device=torch.device("cpu"), n_sample=1, stream=RngStream("t", 0),
            event=choice.key, feedback=feedback,
        )

    n_levels = int(schedule.numel())
    assert fired["n"] == 1, "the event must fire on exactly one invocation"
    # one solver call per level transition, plus the single provisional call
    assert denoiser.calls == (n_levels - 1) + 1


def test_every_denoiser_call_carries_the_tap(monkeypatch):
    """Feedback reaches the model ONLY through a tap.

    ``OfficialDenoiser.denoise`` with ``tap=None`` increments the injection
    counter and discards the residual, so a run could report injections while
    having applied nothing.
    """
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    denoiser = _StubDenoiser(n_levels=4)
    with BackboneTap(denoiser.model.diffusion_module) as tap:
        run_trajectory(
            denoise=lambda x, s, *, feedback=None: denoiser.denoise(
                x, s, feedback=feedback, tap=tap
            ),
            schedule=denoiser.schedule(), n_atom=denoiser.n_atom,
            device=torch.device("cpu"), n_sample=1, stream=RngStream("t", 0),
        )
    assert denoiser.calls > 0
    assert all(denoiser.saw_tap), "a call without a tap would discard feedback"


def test_no_conditioner_means_no_injection():
    """The A_BS-only control takes the genuine no-feedback path."""
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    denoiser = _StubDenoiser(n_levels=4)
    with BackboneTap(denoiser.model.diffusion_module) as tap:
        _x, _r, stats = run_trajectory(
            denoise=lambda x, s, *, feedback=None: denoiser.denoise(
                x, s, feedback=feedback, tap=tap
            ),
            schedule=denoiser.schedule(), n_atom=denoiser.n_atom,
            device=torch.device("cpu"), n_sample=1, stream=RngStream("t", 0),
            event=(0, 0), feedback=lambda state: None,
        )
    assert stats["injections"] == 0
    assert denoiser.injections == 0


def test_fampnn_draws_do_not_move_the_backbone_rng():
    """`stream.protected()` around the callback, or every later step shifts.

    The event runs ~101 encoder calls and a packing rollout, all drawing from
    the global RNG. Without protection the corrected trajectory would diverge
    from the uncorrected one for that reason rather than because of feedback.
    """
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    def trajectory(with_draws):
        denoiser = _StubDenoiser(n_levels=6)

        def feedback(state):
            if with_draws:
                torch.randn(500)  # stand in for FaMPNN's draws
            return None

        with BackboneTap(denoiser.model.diffusion_module) as tap:
            x, _r, _s = run_trajectory(
                denoise=lambda x, s, *, feedback=None: denoiser.denoise(
                    x, s, feedback=feedback, tap=tap
                ),
                schedule=denoiser.schedule(), n_atom=denoiser.n_atom,
                device=torch.device("cpu"), n_sample=1,
                stream=RngStream("t", 0), event=(1, 0), feedback=feedback,
            )
        return x

    assert torch.equal(trajectory(False), trajectory(True))


# ------------------------------------------- contracts a code review caught
#
# Each of these pins a defect found by review at e73f484. They are written as
# the reviewer's own probes so a regression reproduces the original finding
# rather than merely failing somewhere nearby.


def test_packed_coords_reads_the_packers_dict_interface():
    """R8: FaMPNNSideChainPacker.forward returns a DICT.

    The first version accepted tensors, tuples and attribute-bearing objects
    but not that, so the FINAL packing of a complete trajectory raised
    ``TypeError: cannot read coordinates from dict``.
    """
    from pxf.bench.integrated import _packed_coords, _packed_mask

    coords = torch.zeros(1, 4, 37, 3)
    packed = {"coords_af2": coords, "atom_mask_af2": torch.ones(1, 4, 37)}
    assert _packed_coords(packed).shape == (1, 4, 37, 3)
    assert _packed_mask(packed, None).shape == (1, 4, 37)


def test_packed_coords_refuses_a_non_dict_rather_than_guessing():
    from pxf.bench.integrated import _packed_coords

    with pytest.raises(TypeError, match="will not guess"):
        _packed_coords(torch.zeros(1, 4, 37, 3))


def test_packed_psce_refuses_to_substitute_the_events_confidence():
    """Falling back to the event's psCE would describe bb0's packing."""
    from pxf.bench.integrated import _packed_psce

    with pytest.raises(KeyError, match="event's"):
        _packed_psce({}, None)


def test_rng_stream_device_is_supplied_on_cuda():
    """R2: replay only captures CUDA RNG when a device is passed.

    Without it two resumes share coordinates and the CPU stream but not the
    CUDA generator, so they are not paired on GPU.
    """
    from pxf.bench.integrated import _cuda_device

    assert _cuda_device(torch.device("cpu")) is None
    assert _cuda_device("cuda:0") == torch.device("cuda:0")


def test_rng_stream_records_no_cuda_state_without_a_device():
    """The property that made R2 a real defect, stated directly."""
    from pxf.couple.replay import RngStream

    assert RngStream("x", 0)._cuda is None


def test_aligned_rmsd_correction_matrix_follows_the_input_device():
    """R7: the correction was built on CPU while inputs stayed on CUDA."""
    from pxf.bench.integrated import _aligned_rmsd_local

    x = torch.randn(20, 3)
    assert _aligned_rmsd_local(x, x.clone()) == pytest.approx(0.0, abs=1e-6)
