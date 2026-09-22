"""Replay has to be exact, and "exact" has to be pinned against upstream.

Two separate claims, tested separately because one loop serving both the
original trajectory and the replays makes the second true by construction and
the first not at all:

1. ``run_trajectory`` is a faithful transcription of Protenix's
   ``sample_diffusion``. Tested by running both from step 0 on the same seed.
2. Resuming from a recorded state with no feedback reproduces the uninterrupted
   tail. Tested against this loop's own output.

A stub denoiser is enough for both: what is under test is the solver, the churn,
the augmentation and the RNG bookkeeping, none of which depend on the network.
"""

import pytest
import torch

from pxf.couple.replay import FixedTarget, RngStream, run_trajectory

N_ATOM = 64
SCHEDULE = torch.tensor([16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1])
# PXDesign's published sampler settings; gamma_min=0.01 means churn is active
# across this whole schedule, so t_hat = 2 * c_tau_last everywhere.
SAMPLER = dict(gamma0=1.0, gamma_min=0.01, noise_scale_lambda=1.003, step_scale_eta=1.5)


def stub_denoise(scale=0.25):
    """A deterministic, non-trivial x0 predictor. Records what it was called with."""
    seen = []

    def denoise(x_noisy, sigma, *, feedback=None):
        sigma_value = float(torch.as_tensor(sigma).reshape(-1)[0])
        seen.append((sigma_value, float(x_noisy.abs().mean())))
        out = x_noisy * scale + torch.tanh(x_noisy) * 0.1
        if feedback is not None:
            out = out + feedback.reshape(1, 1, -1)[..., :1] * 0.0 + feedback.mean()
        return out

    denoise.seen = seen
    return denoise


def upstream(denoise, seed):
    """Protenix's own sampler, driven with the same stub."""
    from protenix.model.generator import sample_diffusion

    def denoise_net(x_noisy, t_hat_noise_level, **_ignored):
        return denoise(x_noisy, t_hat_noise_level)

    feature = {"atom_to_token_idx": torch.zeros(N_ATOM, dtype=torch.long)}
    s_inputs = torch.zeros(1, N_ATOM, 4)
    # Both generators: the augmentation's translation is torch and its rotation
    # is scipy/numpy, so seeding one reproduces nothing.
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        return sample_diffusion(
            denoise_net=denoise_net,
            input_feature_dict=feature,
            s_inputs=s_inputs,
            s_trunk=s_inputs,
            z_trunk=None,
            pair_z=None,
            p_lm=None,
            c_l=None,
            noise_schedule=SCHEDULE,
            N_sample=1,
            **SAMPLER,
        )


def mine(seed, **overrides):
    kwargs = dict(
        denoise=stub_denoise(),
        schedule=SCHEDULE,
        n_atom=N_ATOM,
        device=torch.device("cpu"),
        # Upstream derives batch_shape from s_inputs.shape[:-2], which is (1,)
        # for a [1, N, 4] tensor. Matching it matters beyond shape: a different
        # element count draws a different amount from randn and the streams part.
        batch_shape=(1,),
        stream=RngStream("bb", seed),
        **SAMPLER,
    )
    kwargs.update(overrides)
    with torch.no_grad():
        return run_trajectory(**kwargs)


# --- claim 1: the transcription is faithful --------------------------------


def test_the_transcription_matches_upstream():
    """Same seed, same schedule, same stub -> same trajectory as Protenix."""
    seed = 1234
    theirs = upstream(stub_denoise(), seed)
    ours, _records, _stats = mine(seed)
    assert ours.shape == theirs.shape, (tuple(ours.shape), tuple(theirs.shape))
    delta = float((ours - theirs).abs().max())
    assert delta < 1e-5, (
        f"the transcription diverges from Protenix's sampler by {delta:.3e}; "
        "replay fidelity would then be measured against the wrong reference"
    )


def test_the_denoiser_sees_the_churned_sigma_not_the_scheduled_one():
    """t_hat = 2 * c_tau_last under PXDesign's gamma0=1.0, gamma_min=0.01."""
    denoise = stub_denoise()
    mine(7, denoise=denoise)
    seen = [sigma for sigma, _ in denoise.seen]
    scheduled = SCHEDULE.tolist()[:-1]
    assert len(seen) == len(scheduled)
    for got, level in zip(seen, scheduled):
        assert got == pytest.approx(level * 2.0, rel=1e-6), (
            "the denoiser was called at the scheduled level; a replay keyed on "
            "that would denoise at half the noise the original saw"
        )


def test_one_denoiser_call_per_step():
    denoise = stub_denoise()
    _x, _r, stats = mine(3, denoise=denoise)
    assert stats["calls"] == SCHEDULE.numel() - 1
    assert stats["augmentations"] == SCHEDULE.numel() - 1


# --- claim 2: replay reproduces the trajectory ------------------------------


@pytest.mark.parametrize("step", [1, 3, 5])
def test_replay_with_feedback_disabled_reproduces_the_trajectory(step):
    """The first acceptance test: resuming changes nothing on its own."""
    reference, records, _stats = mine(99, record_steps=[step])
    assert len(records) == 1 and records[0].step == step
    resumed, _r, stats = mine(99, resume=records[0].to(torch.device("cpu")))
    delta = float((resumed - reference).abs().max())
    assert delta == 0.0, (
        f"resuming from step {step} with no feedback moved the trajectory by "
        f"{delta:.3e}; the arms would then differ by the resume, not the residual"
    )
    assert stats["injections"] == 0


def test_the_rng_state_is_what_makes_replay_exact():
    """Drop the recorded RNG and the tail diverges -- so it is load-bearing."""
    reference, records, _s = mine(5, record_steps=[2])
    state = records[0]
    resumed_ok, _r, _s = mine(5, resume=state)
    assert float((resumed_ok - reference).abs().max()) == 0.0
    # Same state, a stream whose RNG was never restored to the recorded point.
    from dataclasses import replace as dc_replace

    wrong = dc_replace(state, rng=RngStream("bb", 5 + 1).capture())
    resumed_bad, _r, _s = mine(5, resume=wrong)
    assert float((resumed_bad - reference).abs().max()) > 1e-6


def test_exactly_one_injection_at_the_event_key():
    reference, records, _s = mine(11, record_steps=[2])
    seen = []

    def feedback(state):
        seen.append(state.key)
        return torch.full((1, N_ATOM // 4, 8), 0.05)

    corrected, _r, stats = mine(11, resume=records[0], event=(2, 0), feedback=feedback)
    assert stats["injections"] == 1, stats
    assert seen == [(2, 0)]
    assert float((corrected - reference).abs().max()) > 0, "the injection did nothing"


def test_an_event_key_that_never_fires_injects_nothing():
    _reference, records, _s = mine(11, record_steps=[2])
    _x, _r, stats = mine(
        11, resume=records[0], event=(99, 0), feedback=lambda s: torch.ones(1, 4, 8)
    )
    assert stats["injections"] == 0


def test_separate_streams_stop_fampnn_shifting_the_backbone():
    """The trap: FaMPNN drawing from the global RNG between solver steps.

    With one shared stream, running the packer mid-trajectory consumes draws and
    the backbone diverges even with the feedback disabled -- an arm difference
    that has nothing to do with the residual.
    """
    reference, records, _s = mine(21, record_steps=[2])

    def greedy_feedback(state):
        # Stand-in for FaMPNN: draws from the global RNG, as it does.
        torch.randn(256)
        return None

    resumed, _r, stats = mine(21, resume=records[0], event=(2, 0), feedback=greedy_feedback)
    assert stats["injections"] == 0
    assert float((resumed - reference).abs().max()) == 0.0, (
        "a side computation drawing from the global RNG shifted the backbone "
        "trajectory; BB and FaMPNN are not on separate streams"
    )


# --- the fixed-target policy ------------------------------------------------


def test_the_target_stays_fixed_through_augmentation_and_replay():
    """Zero residual is not the mechanism; coordinate enforcement is."""
    mask = torch.zeros(N_ATOM, dtype=torch.bool)
    mask[: N_ATOM // 2] = True
    reference = torch.randn(1, N_ATOM, 3) * 5.0
    fixed = FixedTarget(reference=reference, atom_mask=mask)
    x0, records, stats = mine(31, record_steps=[2], fixed_target=fixed)
    assert stats["target_max_displacement"] < 1e-4, stats
    kept = x0.reshape(-1, 3)[mask]
    assert torch.allclose(kept, reference.reshape(-1, 3)[mask], atol=1e-4)
    # And through a replay.
    resumed, _r, stats2 = mine(31, resume=records[0], fixed_target=fixed)
    assert stats2["target_max_displacement"] < 1e-4
    assert torch.allclose(
        resumed.reshape(-1, 3)[mask], reference.reshape(-1, 3)[mask], atol=1e-4
    )


def test_an_empty_target_mask_is_refused():
    with pytest.raises(ValueError, match="at least one target atom"):
        FixedTarget(reference=torch.zeros(1, N_ATOM, 3), atom_mask=torch.zeros(N_ATOM))


def test_the_denoiser_reads_the_target_at_its_reference():
    """The churn noises the target unless it is pinned after the churn.

    At t_hat = 2*sigma the churn adds sigma*sqrt(3) per coordinate -- ~1.5 A at
    sigma 0.86. Pinning only after the Euler update leaves the denoiser
    conditioning against a smeared target for the whole trajectory, which
    surfaces as nonsense interface geometry rather than as an error.
    """
    mask = torch.zeros(N_ATOM, dtype=torch.bool)
    mask[: N_ATOM // 2] = True
    reference = torch.randn(1, N_ATOM, 3) * 5.0
    fixed = FixedTarget(reference=reference, atom_mask=mask)
    seen = []

    def denoise(x_noisy, sigma, *, feedback=None):
        # What the denoiser actually receives for the fixed atoms.
        got = x_noisy.reshape(-1, 3)[mask]
        seen.append(float((got - reference.reshape(-1, 3)[mask]).norm(dim=-1).max()))
        return x_noisy * 0.25

    mine(41, denoise=denoise, fixed_target=fixed)
    assert seen, "the denoiser was never called"
    worst = max(seen)
    assert worst < 1e-4, (
        f"the denoiser saw the target displaced by up to {worst:.3f} A; the "
        "churn is noising the fixed atoms before the model reads them"
    )


def test_after_event_observes_corrected_estimate_once_and_preserves_rng():
    import numpy as np

    event = (3, 0)
    payload = torch.ones(1, 2, 3) * .1
    plain, _, plain_stats = mine(91, event=event, feedback=lambda _: payload)
    seen = []

    def after(state, estimate):
        seen.append(state.key)
        expected = stub_denoise()(state.x_noisy, state.sigma, feedback=payload)
        assert torch.equal(estimate, expected)
        torch.manual_seed(123456)
        np.random.seed(123456)
        torch.rand(1000)
        np.random.rand(1000)
        return torch.zeros_like(estimate)  # observational callback only

    got, _, stats = mine(91, event=event, feedback=lambda _: payload, after_event=after)
    assert seen == [event]
    assert torch.equal(got, plain)
    assert stats['calls'] == plain_stats['calls']
