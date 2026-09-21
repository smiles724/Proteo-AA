"""The `bs_seq_sc_v1` joint objective: masked sequence + side-chain diffusion.

    L_J = lambda_seq * L_MLM + lambda_SC * L_diffusion

Both terms read the SAME conditioned representation, which is the entire point
and the reason this is a new task identity rather than a new phase of the
existing one:

    h' = h_V + M_B * A_BS(a_token, sigma_B)
    L_MLM        over  W_out(h')
    L_diffusion  over  D_SC(q_t, t_SC, h', s_GT)

Under the legacy `packing_only` mode the residual lands inside side-chain
diffusion, after that pass's logits, so the same-pass sequence loss has no
gradient into the adapter at all. Here it does.

Nothing is reimplemented. `sequence_mlm_loss`, `diffusion_loss` and their EDM
weighting, noise clones, ghost-slot convention and teacher forcing are used as
they are, so the adapter trains against the objectives the modules already
know. What this module adds is the routing and the mask plumbing.

### One asymmetry that is deliberate and easy to get backwards

The two branches do NOT see the same residue identities:

  * the **encoder** is told ``masks.aatype_encoder`` -- binder identities
    hidden at the masked positions. That is what makes L_MLM a prediction
    problem rather than a copy.
  * the **side-chain** branch is teacher-forced on ``masks.aatype_true``, the
    ground truth, which is what FaMPNN's own diffusion objective does and what
    the existing coupling loss already relies on.

Feeding the masked identities to the SC branch would change the side-chain
objective into something upstream never trained, and feeding the true ones to
the encoder would make the sequence loss trivially zero. Both mistakes produce
a number.

What the two branches do share is that ``L_MLM`` never scores a residue whose
true identity is ``X`` -- see :func:`unknown_identities`. That is the source's
behaviour (``SDLoss`` masks by ``1 - seq_unk_mask``) and a third mistake that
produces a number: the model would be trained to emit its own mask token.

### S03 vs J03

S03 sets ``lambda_seq = 0``. It still builds the same masks and encodes the
same partially masked context -- only the optimisation differs -- so J03 - S03
isolates sequence supervision with routing held fixed. That is why
``lambda_seq`` is a coefficient here rather than a branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from pxf.couple.binder_masks import BinderMaskSet
from pxf.couple.binder_residual import ChainRoles
from pxf.couple.shared_prelogit import conditioned_forward
from pxf.train import losses as loss_fns
from pxf.train import step as train_step

TASK = "bs_seq_sc_v1"
APPLICATION_MODE = "shared_prelogit"


@dataclass
class JointLoss:
    total: torch.Tensor
    sequence: torch.Tensor
    sidechain: torch.Tensor
    lambda_seq: float
    lambda_sc: float
    delta: Optional[torch.Tensor] = None
    stats: dict[str, Any] = field(default_factory=dict)


def batch_from_inputs(coupled_inputs, aatype_true: torch.Tensor) -> dict[str, Any]:
    """The dict `pxf.train.step` needs, teacher-forced on ground truth.

    `aatype_true`, not the masked identities: see the module docstring. The
    keys are exactly `train_step.REQUIRED_KEYS`, and asserting that here means
    a future key addition upstream fails at construction rather than inside
    the loss.
    """
    batch = {
        "x": coupled_inputs.coords_af2,
        "aatype": aatype_true,
        "seq_mask": coupled_inputs.seq_mask,
        "missing_atom_mask": coupled_inputs.missing_atom_mask,
        "residue_index": coupled_inputs.residue_index,
        "chain_index": coupled_inputs.chain_index,
    }
    missing = [k for k in train_step.REQUIRED_KEYS if k not in batch]
    if missing:
        raise KeyError(
            f"pxf.train.step now requires {missing}, which this batch does not "
            "build; add them here rather than letting the loss improvise"
        )
    return batch


def unknown_identities(batch: dict[str, Any], masks: BinderMaskSet) -> torch.Tensor:
    """The ``X`` rows `L_MLM` must not score, derived the way the source does.

    ``SDLoss`` multiplies its sequence mask by ``1 - seq_unk_mask`` before
    scoring anything, so a residue whose true identity is unknown is never a
    label. Training on those teaches the model to emit the very token it uses
    as its own mask. `pxf.train.step`'s single-task path already passes this;
    the joint path has the same obligation, and derives it from the same
    helper rather than restating the predicate.

    The equality check is the point of the function existing at all: the
    labels here come from `masks`, the mask from `batch`, and the two are the
    same tensor only because `batch_from_inputs` was handed `aatype_true`. A
    caller that builds its batch some other way gets an error rather than a
    mask that silently describes different residues than the labels do.
    """
    aatype = batch["aatype"]
    if not torch.equal(aatype.long(), masks.aatype_true.long()):
        raise ValueError(
            "batch['aatype'] is not masks.aatype_true, so the unknown-identity "
            "mask would not line up with the labels. Build the batch with "
            "batch_from_inputs(coupled_inputs, masks.aatype_true)."
        )
    return train_step.batch_masks(batch)["seq_unk_mask"]


def joint_loss(
    model,
    batch: dict[str, Any],
    features: dict[str, Any],
    *,
    adapters,
    a_token: torch.Tensor,
    sigma_b,
    roles: ChainRoles,
    masks: BinderMaskSet,
    lambda_seq: float = 1.0,
    lambda_sc: float = 1.0,
    source: str = "matched",
    gate=None,
    mean=None,
    multiplier: Optional[int] = None,
    self_cond_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    allow_zero: bool = False,
    seq_settings=loss_fns.DEFAULT_LOSS_SETTINGS,
    reduction=loss_fns.DEFAULT_SIDECHAIN_REDUCTION,
) -> JointLoss:
    """One `bs_seq_sc_v1` step. Differentiable into the adapter and nothing else."""
    seq_module = model.denoiser.seq_design_module
    logits, conditioned, delta = conditioned_forward(
        seq_module, features,
        adapters=adapters, a_token=a_token, sigma_b=sigma_b,
        roles=roles, source=source, gate=gate, mean=mean,
        allow_zero=allow_zero,
    )
    if logits is None:
        raise ValueError(
            "the residual was None, so no conditioned logits exist. The "
            "uncoupled arm is U03; it is not this objective with the adapter "
            "switched off."
        )

    sequence, seq_stats = loss_fns.sequence_mlm_loss(
        logits,
        masks.aatype_true,
        masks.seq_mlm_mask,
        masks.seq_mask,
        seq_unk_mask=unknown_identities(batch, masks),
        settings=seq_settings,
    )
    sidechain, sc_stats = train_step.diffusion_loss(
        model, batch, conditioned,
        multiplier=multiplier, self_cond_p=self_cond_p,
        generator=generator, scn_mlm_mask=None, reduction=reduction,
    )

    total = lambda_seq * sequence + lambda_sc * sidechain
    stats = {
        **{f"seq/{k}": v for k, v in seq_stats.items()},
        **{f"sc/{k}": v for k, v in sc_stats.items()},
        "delta_h_norm": delta.detach().norm(dim=-1).mean(),
        "lambda_seq": lambda_seq,
        "lambda_sc": lambda_sc,
    }
    return JointLoss(
        total=total, sequence=sequence, sidechain=sidechain,
        lambda_seq=lambda_seq, lambda_sc=lambda_sc, delta=delta, stats=stats,
    )


def freeze_everything_but(adapters, *modules) -> dict[str, int]:
    """eval() + requires_grad_(False) on the donors; only the adapter trains.

    Returned counts are asserted by the section 7 checks rather than trusted:
    "frozen" that is actually only `eval()` still accumulates gradients and
    still moves under an optimizer that was handed `model.parameters()`.
    """
    frozen = 0
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            frozen += 1
    trainable = 0
    adapters.train()
    for parameter in adapters.parameters():
        parameter.requires_grad_(True)
        trainable += 1
    return {"frozen_tensors": frozen, "trainable_tensors": trainable}


def gradient_norms(loss: torch.Tensor, adapters, *, retain: bool = True) -> float:
    """||dL/dtheta|| over the adapter only, without touching .grad."""
    parameters = [p for p in adapters.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=retain, allow_unused=True,
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach() ** 2).sum())
    return total ** 0.5


def suggest_lambda_seq(
    seq_grad_norm: float, sc_grad_norm: float, *, lambda_sc: float = 1.0,
    lo: float = 1e-3, hi: float = 1e3,
) -> dict[str, Any]:
    """lambda_seq that equalises the two gradient norms at the adapter.

    A starting hypothesis, not a guarantee against conflicting objectives, and
    clamped to a predeclared range: a dead or near-dead gradient should be
    reported and rejected, never compensated for with an enormous weight.
    """
    out: dict[str, Any] = {
        "seq_grad_norm": seq_grad_norm, "sc_grad_norm": sc_grad_norm,
        "lambda_sc": lambda_sc, "range": [lo, hi],
    }
    if not (seq_grad_norm > 0) or not (sc_grad_norm > 0):
        out["lambda_seq"] = None
        out["problem"] = (
            "one of the gradients is zero or non-finite; fix the routing "
            "rather than reweighting it away"
        )
        return out
    raw = lambda_sc * sc_grad_norm / seq_grad_norm
    out["lambda_seq_raw"] = raw
    out["lambda_seq"] = min(max(raw, lo), hi)
    out["clamped"] = not (lo <= raw <= hi)
    if out["clamped"]:
        out["problem"] = (
            f"the balancing coefficient {raw:.4g} falls outside the declared "
            f"range [{lo}, {hi}]; the two objectives are not commensurate and "
            "clamping is a decision, not a fix"
        )
    return out
