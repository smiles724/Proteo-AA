"""Per-residue side-chain loss routing.

STATUS (M4): SKELETON — implemented + unit-tested, but NOT wired into the
training path. `model._train_forward` / `PXDesignLoss` do not call this; they
mask the coordinate loss by `sc_type_match` and apply the physical loss over
all atoms (not mismatch-only). True predicted-mask routing (coord loss on
type-matched residues, physical loss on the PREDICTED atom set of mismatched
residues) is deferred to Stage III. Do not claim "loss routing implemented" in
training until this is called from the model.

Intended behavior: when the predicted residue type matches GT, atom-level
coordinate supervision is valid -> use the coordinate loss on those
residues. When it does not match, the atom sets differ and coordinate MSE is
undefined -> use physical loss only. (SideCraft spec §4; group-chat loss router.)

The two loss callables receive the per-residue boolean mask selecting the
residues they should supervise, so the caller controls exactly how each side is
computed.
"""
from typing import Callable

import torch


def route_sidechain_loss(
    pred_type_logits: torch.Tensor,   # [L, C] (or [..., L, C])
    gt_type: torch.Tensor,            # [L] long (or [..., L])
    coord_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    phys_loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Split residues by type-match and sum coord (matched) + physical (mismatched).

    Args:
        pred_type_logits: residue-type logits; argmax gives predicted type.
        gt_type: ground-truth residue-type indices.
        coord_loss_fn: called with the boolean match mask -> scalar.
        phys_loss_fn:  called with the boolean mismatch mask -> scalar.
    Returns:
        coord_term + phys_term (scalar).
    """
    match = pred_type_logits.argmax(dim=-1) == gt_type   # [..., L] bool
    coord_term = coord_loss_fn(match)
    phys_term = phys_loss_fn(~match)
    return coord_term + phys_term
