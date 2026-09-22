"""``integrated_feedback_v1``: train the feedback module and nothing else.

The objective is a binder-design-conditioned reconstruction. For a deposited
complex, the binder backbone is corrupted to the event's noise level, its
identity and side chains are withheld, the target is given as context, and a
sequence is designed under the frozen A_BS. The feedback module then gets one
chance to correct the backbone estimate at that same noisy state, and is scored
against the deposited coordinates:

    no_grad:  bb0, a0      = PXDesign(x_sigma, sigma)
              seq0, sc0    = J03-conditioned iterative FaMPNN(bb0, target ctx)
              packed0      = reencode(bb0, seq0, sc0)        generated masks
    live:     delta        = E1(packed0.detach(), sigma)
              delta        = mask_after_projection(delta)
              bb1          = differentiable_PXDesign(x_sigma, sigma, delta)
              loss         = EDM_weighted_BB_loss(bb1, native, binder_mask)

Only the feedback module trains. PXDesign is frozen and in eval mode but
differentiable *with respect to the feedback*; FaMPNN and A_BS are frozen and
receive no gradient at all.

### What the loss is, precisely

:func:`pxf.couple.losses.backbone_denoising_loss` -- an EDM-weighted coordinate
objective, NOT PXDesign's complete diffusion loss. Its own docstring
enumerates the differences: no rigid alignment of target to prediction, no
SmoothLDDT, no BondLoss, no ``weight_mse``. The absolute value is therefore
not comparable to a PXDesign training curve, and that is recorded here rather
than discovered later.

Not aligning is deliberate. A correction evaluated at a fixed noisy state must
not be credited for re-posing the whole structure, and the sampler consumes
the estimate in the frame it was produced in. It also closes an obvious way to
cheat: independently superposing the binder would lower the loss without
improving anything.

### Where the labels may and may not appear

The noisy coordinates are derived from the native backbone -- that is what
denoising training *is*. What must not happen is native information entering a
CONDITIONING channel: not the binder's identity, not its side chains, not its
atom-existence pattern, and not a cache key the model reads.
:func:`leakage_report` states what was withheld and
``scripts/preflight_integrated_feedback.py`` verifies it by perturbing the
withheld labels and checking the loss does not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from pxf.couple.integrated_event import mask_feedback
from pxf.couple.losses import backbone_denoising_loss

TASK = "integrated_feedback_v1"
SCHEMA = "integrated_feedback/1"
DEFAULT_SIGMA_DATA = 16.0


@dataclass
class FeedbackExample:
    """One cached upstream event. Everything here is detached by construction."""

    example_id: str
    x_noisy: torch.Tensor        # [1, n_atom, 3] the state to correct
    sigma: float                 # the ACTUAL churned sigma
    packed: Any                  # PackedStructure: the readout's whole input
    binder_mask: torch.Tensor    # [1, L] 1 = a designed row
    native_bb: torch.Tensor      # [1, n_atom, 3] deposited coordinates
    supervised: torch.Tensor     # [1, n_atom] 1 = scored (resolved binder)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to(self, device):
        return FeedbackExample(
            example_id=self.example_id,
            x_noisy=self.x_noisy.to(device),
            sigma=self.sigma,
            packed=_packed_to(self.packed, device),
            binder_mask=self.binder_mask.to(device),
            native_bb=self.native_bb.to(device),
            supervised=self.supervised.to(device),
            provenance=self.provenance,
        )


@dataclass
class FeedbackLoss:
    total: torch.Tensor
    delta: Any
    bb1: torch.Tensor
    stats: dict[str, Any]


def feedback_loss(
    example: FeedbackExample,
    conditioner,
    denoise,
    *,
    sigma_data: float = DEFAULT_SIGMA_DATA,
) -> FeedbackLoss:
    """One training step's forward. ``denoise`` must be DIFFERENTIABLE.

    ``denoise`` comes from ``PXDesignBackboneDriver.bind``, which has no
    ``no_grad`` wrapper -- unlike ``OfficialDenoiser.denoise``, which does and
    would silently give a loss with no gradient path to the feedback module.
    """
    sigma = torch.full(
        (1,), float(example.sigma),
        device=example.packed.h_packed.device, dtype=torch.float32,
    )
    # POSITIONAL, and it returns a PAIR. `conditioner(packed=..., sigma=...)`
    # with h_V in place of the PackedStructure fails inside the readout, which
    # reads coords37/aatype/visibility as well as h_packed.
    raw, readout_stats = conditioner(example.packed, sigma)
    # zero_bypass=False: a freshly zero-initialised head emits exactly zero on
    # step one, and dropping the tensor would drop its gradient with it.
    delta = mask_feedback(raw, example.binder_mask, zero_bypass=False, name="E1")
    if delta is None:
        raise AssertionError(
            "mask_feedback returned None during training; the inference "
            "zero-bypass must never be enabled here or the head cannot learn"
        )

    bb1 = denoise(example.x_noisy, sigma, feedback=delta)
    if not bb1.requires_grad:
        raise AssertionError(
            "the corrected backbone carries no gradient. Either the denoiser "
            "is wrapped in no_grad (use the differentiable driver, not the "
            "official inference wrapper) or the feedback never reached it."
        )
    # Returns a CoupledLoss, not a tensor. Its stats already carry the
    # supervised-atom count and the backbone RMSD, so those are read off it
    # rather than recomputed -- a second implementation of the same number is
    # how the reported RMSD and the optimised one drift apart.
    coupled = backbone_denoising_loss(
        bb1, example.native_bb,
        sigma=sigma, sigma_data=sigma_data,
        atom_mask=example.supervised,
    )
    with torch.no_grad():
        stats = {
            **coupled.scalars(),
            "sigma": float(example.sigma),
            "supervised_atoms": int(example.supervised.sum()),
            "delta_norm": _payload_norm(delta),
            **{f"readout/{k}": v for k, v in (readout_stats or {}).items()
               if isinstance(v, (int, float))},
        }
    return FeedbackLoss(total=coupled.total, delta=delta, bb1=bb1, stats=stats)


def _packed_to(packed, device):
    """Move a PackedStructure (and its Visibility) onto ``device``."""
    from dataclasses import fields, replace

    import torch

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if hasattr(value, "__dataclass_fields__"):
            return replace(value, **{
                f.name: move(getattr(value, f.name)) for f in fields(value)
            })
        return value

    return move(packed)


def _payload_norm(delta) -> float:
    from pxf.couple.pxdesign_iface import ConditioningFeedback

    if isinstance(delta, ConditioningFeedback):
        parts = [t for t in (delta.delta_single, delta.delta_pair) if t is not None]
        return float(sum(float(t.detach().norm()) for t in parts))
    return float(delta.detach().norm())


def freeze_everything_but(conditioner, *donors) -> dict[str, Any]:
    """Freeze every donor, leave the conditioner trainable, and report it.

    Returned counts go into the checkpoint. A run whose ``trainable_tensors``
    is not exactly the conditioner's parameter count trained something it
    should not have, and that is checkable after the fact rather than a matter
    of trusting the setup.
    """
    frozen = 0
    for donor in donors:
        if donor is None:
            continue
        donor.eval()
        for parameter in donor.parameters():
            parameter.requires_grad_(False)
            frozen += 1
    trainable = []
    for name, parameter in conditioner.named_parameters():
        parameter.requires_grad_(True)
        trainable.append(name)
    return {
        "frozen_tensors": frozen,
        "trainable_tensors": len(trainable),
        "trainable_parameters": int(
            sum(p.numel() for p in conditioner.parameters() if p.requires_grad)
        ),
        "trainable_names": trainable,
    }


def assert_donors_clean(conditioner, *donors) -> None:
    """No donor may hold a gradient after a backward pass."""
    offenders = []
    for donor in donors:
        if donor is None:
            continue
        for name, parameter in donor.named_parameters():
            if parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
                offenders.append(name)
    if offenders:
        raise AssertionError(
            f"{len(offenders)} donor parameter(s) received a gradient, e.g. "
            f"{offenders[:3]}. The donors must be frozen; a run that trains "
            "them is not this experiment."
        )


def gradient_norms(conditioner) -> dict[str, float]:
    """Per-parameter gradient norms, for the preflight's init check.

    The OUTPUT projection must be non-zero at initialization -- dL/dW_out =
    dL/dh' * h_in^T is non-zero even when W_out is zero, which is what lets a
    zero-initialised head train from step one. Internal readout weights are
    legitimately zero until after the first update, because the zero output
    projection blocks their path.
    """
    out = {}
    for name, parameter in conditioner.named_parameters():
        out[name] = (
            0.0 if parameter.grad is None else float(parameter.grad.norm())
        )
    return out


def output_projection(conditioner):
    """The final linear of the residual head, found STRUCTURALLY.

    Not by name-matching. An earlier version of the init check looked for
    "project_out", matched nothing on ``EarlySingleConditioner`` (whose
    parameters are ``single_head.0/2`` and ``readout.*``), took ``max()`` of an
    empty dict, and reported ``nan`` as "non-zero, as required". A vacuous
    check is worse than no check: it certifies exactly the failure it exists
    to catch.
    """
    for attribute in ("single_head", "head", "out"):
        head = getattr(conditioner, attribute, None)
        if head is None:
            continue
        linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)] \
            if hasattr(head, "modules") else []
        if linears:
            return linears[-1]
    raise AssertionError(
        "cannot locate the conditioner's output projection, so the "
        "initialisation gradient cannot be checked. Refusing to report a "
        f"vacuous pass. Parameters are: {[n for n, _ in conditioner.named_parameters()]}"
    )


def check_initial_gradient(conditioner) -> dict[str, float]:
    """The output projection must carry a NON-ZERO gradient after step one.

    ``dL/dW_out = dL/dh' * h_in^T`` is non-zero even when ``W_out`` is zero,
    which is what lets a zero-initialised head train from the first update. A
    zero here means the feedback never reached the loss -- the inference
    zero-bypass left on, or a ``no_grad`` denoiser -- and 2,000 updates of
    nothing would follow and look like a null result.

    Internal weights ARE legitimately zero at this point: the zero output
    projection blocks their path until it moves.
    """
    final = output_projection(conditioner)
    norms = {
        "weight": 0.0 if final.weight.grad is None else float(final.weight.grad.norm()),
        "bias": (0.0 if final.bias is None or final.bias.grad is None
                 else float(final.bias.grad.norm())),
    }
    if norms["weight"] == 0.0:
        raise AssertionError(
            "the output projection has ZERO gradient after the first backward. "
            "dL/dW_out is non-zero even for a zero W_out, so this means the "
            "feedback never reached the loss. Check that mask_feedback was "
            "called with zero_bypass=False and that the denoiser is the "
            "differentiable driver, not the official no_grad wrapper."
        )
    return norms


def leakage_report(example: FeedbackExample) -> dict[str, Any]:
    """What this example withheld from the model's conditioning channels.

    Descriptive, not a proof. The proof is the preflight's perturbation test:
    change the withheld labels, keep the allowed inputs fixed, and the loss
    must not move.
    """
    return {
        "native_binder_identity_in_conditioning": False,
        "native_binder_sidechains_in_conditioning": False,
        "native_backbone_used_for": "the denoising target and the noisy state "
                                    "only, never as conditioning",
        "sequence_source": "predicted",
        "binder_rows_supervised": int(example.supervised.sum()),
        "cache_keys_exclude_native_labels": True,
    }
