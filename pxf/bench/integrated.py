"""One integrated PXDesign trajectory with a single coupling event.

This is the one-stage path. The cached-backbone matrix
(`scripts/design_binder_matrix.py`) generates every backbone first and designs
sequences afterwards, which makes it two-stage by construction. Here the
sequence is designed *inside* the trajectory and the backbone is corrected with
what the designer produced, so backbone and sequence co-determine each other:

    steps 0 .. E-1     stock PXDesign
    step E             bb0, a_token = D(x_noisy, sigma)         provisional
                       seq, sc      = FaMPNN_iter(bb0; A_BS)    100 steps
                       delta        = E1(h_packed, sigma)       feedback
                       bb1          = D(x_noisy, sigma, delta)  SAME state
                       advance the solver once, using bb1
    steps E+1 .. N     stock PXDesign
    finally            repack the event's sequence on the final backbone

With the shipped 400 steps that is **401 denoiser evaluations**: 400 solver
calls plus the one provisional call at the event. No complete baseline
trajectory runs first, and FaMPNN runs its configured decoder once at the
event plus one final packing rollout.

### Why the solver is reused rather than rewritten

`pxf.couple.replay.run_trajectory` is a transcription of Protenix's
``sample_diffusion`` chunk loop, pinned against upstream by
``tests/test_couple_replay.py``. It already has the two things this needs: an
``event``/``feedback`` hook that fires on exactly one invocation, and
``stream.protected()``, which stops the ~101 FaMPNN encoder calls and the
packing draws from advancing the backbone RNG. Writing a second loop would make
"the integrated run matches the stock sampler outside the event" a coincidence
between two implementations instead of a structural fact.

### The event is chosen by ACTUAL sigma

Protenix churns before denoising: ``gamma = gamma0 if c_tau > gamma_min else
0``, then ``t_hat = c_tau_last * (gamma + 1)``. With PXDesign's published
``gamma0 = 1.0, gamma_min = 0.01`` that is ``2 * c_tau_last`` across the useful
range, so the scheduled level is half what the denoiser sees.
:func:`select_event` reproduces that rule, *including* the next-level
threshold, and picks the step whose actual ``t_hat`` is nearest the requested
sigma. The realised value is recorded and is what A_BS is conditioned on.

### Feedback delivery, and a trap in it

``OfficialDenoiser.denoise`` delivers feedback **only through a tap**: with
``tap=None`` it increments the injection counter and discards the residual. So
one tap is installed for the whole trajectory and every denoiser call goes
through it. That also gives ``a_token`` for free on the provisional call.

Early and late feedback are alternatives, never stacked -- ``BackboneTap``
enforces it, routing a ``ConditioningFeedback`` to the conditioning site and
returning early from the ``a_token`` injector.

### What this does not do

One sample per forward, one event, protein-only complexes, canonical known
target identities, ``complex_sc`` context. Multiple samples run sequentially.
Multiple events, wider noise support and cross-pair writes each need their own
validation and are not enabled here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

DEFAULT_N_STEP = 400
DEFAULT_ETA = 2.5
DEFAULT_EVENT_SIGMA = 0.429
GAMMA0 = 1.0
GAMMA_MIN = 0.01


@dataclass
class EventChoice:
    """Which solver invocation the event lands on, and at what real sigma."""

    step: int
    substage: int
    scheduled_sigma: float   # schedule[step], what a naive reading would use
    actual_sigma: float      # t_hat, after churn -- what the denoiser sees
    churn_ratio: float
    requested_sigma: float
    n_levels: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.step, self.substage)

    def record(self) -> dict[str, Any]:
        return {
            "event_step": self.step,
            "event_substage": self.substage,
            "scheduled_sigma": self.scheduled_sigma,
            "actual_sigma": self.actual_sigma,
            "sigma_churn_ratio": self.churn_ratio,
            "requested_sigma": self.requested_sigma,
            "n_levels": self.n_levels,
        }


def select_event(
    schedule: torch.Tensor,
    requested_sigma: float = DEFAULT_EVENT_SIGMA,
    *,
    gamma0: float = GAMMA0,
    gamma_min: float = GAMMA_MIN,
) -> EventChoice:
    """The step whose CHURNED sigma is nearest ``requested_sigma``.

    The threshold is on ``c_tau`` -- the *next* level -- not on the current
    one, which is upstream's rule and matters at the tail where gamma turns
    off and the actual sigma stops being twice the scheduled one.
    """
    levels = schedule.reshape(-1).to(torch.float64)
    n_levels = int(levels.numel())
    if n_levels < 2:
        raise ValueError(f"schedule has {n_levels} level(s); need at least 2")

    best: Optional[EventChoice] = None
    for step in range(n_levels - 1):
        c_tau_last = float(levels[step])
        c_tau = float(levels[step + 1])
        gamma = float(gamma0) if c_tau > gamma_min else 0.0
        t_hat = c_tau_last * (gamma + 1.0)
        candidate = EventChoice(
            step=step,
            substage=0,
            scheduled_sigma=c_tau_last,
            actual_sigma=t_hat,
            churn_ratio=(t_hat / c_tau_last if c_tau_last else float("nan")),
            requested_sigma=float(requested_sigma),
            n_levels=n_levels,
        )
        if best is None or abs(t_hat - requested_sigma) < abs(
            best.actual_sigma - requested_sigma
        ):
            best = candidate
    assert best is not None
    return best


@dataclass
class IntegratedSample:
    """One finished integrated design."""

    x0: torch.Tensor                  # [1, n_atom, 3] the final backbone
    aatype: torch.Tensor              # [1, L] the event's sequence, held
    sequence: str
    binder_sequence: str
    coords_af2: torch.Tensor          # [1, L, 37, 3] repacked on the final bb
    atom_mask_af2: torch.Tensor
    psce: torch.Tensor
    binder_mask: torch.Tensor
    event: EventChoice
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _aligned_rmsd_local(a: torch.Tensor, b: torch.Tensor) -> float:
    """Kabsch RMSD, implemented here so this module has no optional import."""
    x = a.reshape(-1, 3).double()
    y = b.reshape(-1, 3).double()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    u, _s, vt = torch.linalg.svd(x.T @ y)
    d = torch.sign(torch.linalg.det(u @ vt))
    # The determinant correction: without it the optimal orthogonal transform
    # may be a REFLECTION, which fits a mirror image and reports a smaller
    # RMSD than any rotation can achieve.
    correction = torch.diag(torch.tensor([1.0, 1.0, d], dtype=x.dtype))
    rotation = u @ correction @ vt
    return float(torch.sqrt(((x @ rotation - y) ** 2).sum(-1).mean()))


def run_integrated(
    *,
    denoiser,
    structure,
    designer,
    adapters=None,
    conditioner=None,
    event_sigma: float = DEFAULT_EVENT_SIGMA,
    n_step: int = DEFAULT_N_STEP,
    step_scale_eta: float = DEFAULT_ETA,
    context: str = "complex_sc",
    seed: int = 0,
    design_id: str = "integrated",
    target: str = "target",
    pack_seed: Optional[int] = None,
    device=None,
) -> IntegratedSample:
    """One integrated trajectory. ``conditioner`` None = the A_BS-only control."""
    from pxf.bench.backbone_inputs import build_design_inputs, check_design_mask
    from pxf.bench.coupled_design import conditioned
    from pxf.couple.integrated_event import (assert_target_rows_untouched,
                                             mask_feedback, prepare_event)
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    import numpy as np

    device = device or denoiser.device
    schedule = denoiser.schedule(n_step)
    choice = select_event(schedule, event_sigma)

    stats: dict[str, Any] = {
        "event_products": None,
        "feedback_norm": 0.0,
        "feedback_installed": conditioner is not None,
        "provisional_calls": 0,
    }
    started = time.time()

    with BackboneTap(denoiser.model.diffusion_module) as tap:

        def denoise(x_noisy, sigma, *, feedback=None):
            # Every call goes through the tap: `OfficialDenoiser.denoise`
            # delivers feedback ONLY via a tap, and would silently discard it
            # (while counting an injection) if tap were None.
            return denoiser.denoise(x_noisy, sigma, feedback=feedback, tap=tap)

        def feedback(state):
            """The event. Returns the residual for the corrected call."""
            products = prepare_event(
                denoise=lambda x, s, **kw: denoiser.denoise(
                    x, s, feedback=None, tap=tap
                ),
                x_noisy=state.x_noisy,
                sigma=float(state.sigma.reshape(-1)[0]),
                structure=structure,
                designer=designer,
                adapters=adapters,
                context=context,
                seed=seed,
                design_id=design_id,
                target=target,
                tap=tap,
                want_h_base=False,
            )
            stats["event_products"] = products
            stats["provisional_calls"] += 1
            if conditioner is None:
                return None  # the A_BS-only control: no correction at all
            delta = conditioner(
                packed=products.h_packed,
                sigma=torch.full(
                    (1,), products.sigma, device=device, dtype=torch.float32
                ),
            )
            delta = mask_feedback(
                delta, products.binder_mask, zero_bypass=True, name="E1"
            )
            if delta is not None:
                assert_target_rows_untouched(delta, products.binder_mask)
                stats["feedback_norm"] = _payload_norm(delta)
            return delta

        stream = RngStream("integrated", seed)
        x0, _records, solver_stats = run_trajectory(
            denoise=denoise,
            schedule=schedule,
            n_atom=denoiser.n_atom,
            device=device,
            n_sample=1,
            step_scale_eta=step_scale_eta,
            stream=stream,
            event=choice.key,
            feedback=feedback,
            fixed_target=None,  # audit section 9: never in the generation path
            identity={"design_id": design_id, "target": target},
        )
        decode_tap_calls = tap.calls
        conditioning_injections = tap.conditioning_injections
        late_injections = tap.injections

    products = stats["event_products"]
    if products is None:
        raise RuntimeError(
            f"the event at step {choice.step} never fired; the trajectory ran "
            "{solver_stats['calls']} call(s) and no sequence was designed"
        )

    # ---- final packing: the EVENT's sequence, on the FINAL backbone --------
    # Not the event's side-chain coordinates: those were packed for bb0, which
    # this backbone has since moved away from. The A_BS residual IS reused,
    # with the event's actual sigma, rather than re-queried at some arbitrary
    # low noise level -- and that transfer is measured, not assumed, by
    # event_to_final_aligned_rmsd below.
    a2t = np.asarray(structure.topology.atom_to_token_idx.cpu()).astype(int)
    res_names = np.asarray(structure.topology.res_names)
    design = check_design_mask(
        np.asarray(structure.design_mask.cpu()),
        res_names=res_names, atom_to_token=a2t,
        n_tokens=int(structure.num_tokens), what=design_id,
    )
    final_inputs = build_design_inputs(
        x0=x0.reshape(-1, 3),
        a_token=products.a_token.reshape(int(structure.num_tokens), -1),
        sigma=products.sigma,
        atom_names=np.asarray(structure.topology.atom_names),
        res_names=res_names,
        atom_to_token=a2t,
        n_tokens=int(structure.num_tokens),
        design=design,
        residue_index=structure.topology.residue_index,
        asym_id=structure.topology.chain_index,
        design_id=design_id, target=target,
        binder_length=int(design.sum()), context=context, device=device,
    )
    with conditioned(designer.model, products.residual) as pack_hook:
        packed = designer(
            coords_af2=final_inputs.coords_af2,
            aatype=products.aatype,
            atom_mask=final_inputs.atom_mask,
            seq_mask=final_inputs.seq_mask,
            residue_index=final_inputs.residue_index,
            chain_index=final_inputs.chain_index,
            scn_context_mask=final_inputs.sidechain_context_mask,
            seed=(seed if pack_seed is None else pack_seed),
        )

    binder = products.binder_mask.reshape(-1).bool()
    event_bb = products.bb0.reshape(-1, 3)
    diagnostics = {
        **choice.record(),
        **products.provenance,
        "solver_calls": int(solver_stats["calls"]),
        "solver_injections": int(solver_stats["injections"]),
        "augmentations": int(solver_stats["augmentations"]),
        "model_calls_total": int(denoiser.calls),
        "provisional_calls": stats["provisional_calls"],
        "tap_calls": int(decode_tap_calls),
        "conditioning_injections": int(conditioning_injections),
        "late_injections": int(late_injections),
        "feedback_installed": stats["feedback_installed"],
        "feedback_norm": stats["feedback_norm"],
        "n_step": n_step,
        "step_scale_eta": step_scale_eta,
        # The transfer assumption, measured. Rigid motion removed first: every
        # step re-augments, so the raw distance is mostly a random rotation.
        "event_to_final_aligned_rmsd": _aligned_rmsd_local(event_bb, x0.reshape(-1, 3)),
        "event_to_final_raw_rmsd": float(
            torch.sqrt(((event_bb - x0.reshape(-1, 3)) ** 2).sum(-1).mean())
        ),
        "final_pack_hook_calls": pack_hook.calls,
        "seconds": round(time.time() - started, 2),
    }
    return IntegratedSample(
        x0=x0.detach(),
        aatype=products.aatype,
        sequence=products.sequence,
        binder_sequence=products.binder_sequence,
        coords_af2=_packed_coords(packed),
        atom_mask_af2=final_inputs.atom_mask,
        psce=_packed_psce(packed, products),
        binder_mask=products.binder_mask,
        event=choice,
        diagnostics=diagnostics,
    )


def _payload_norm(delta) -> float:
    from pxf.couple.pxdesign_iface import ConditioningFeedback

    if isinstance(delta, ConditioningFeedback):
        parts = [t for t in (delta.delta_single, delta.delta_pair) if t is not None]
        return float(sum(float(t.detach().norm()) for t in parts))
    return float(delta.detach().norm())


def _packed_coords(packed):
    """The packer returns either a tensor or a result object; accept both."""
    for attr in ("coords_af2", "x_denoised"):
        value = getattr(packed, attr, None)
        if value is not None:
            return value if value.dim() == 4 else value.unsqueeze(0)
    if isinstance(packed, (tuple, list)):
        return packed[0]
    if torch.is_tensor(packed):
        return packed if packed.dim() == 4 else packed.unsqueeze(0)
    raise TypeError(f"cannot read coordinates from {type(packed).__name__}")


def _packed_psce(packed, products):
    value = getattr(packed, "psce", None)
    if value is None:
        return products.psce
    return value if value.dim() == 3 else value.unsqueeze(0)
