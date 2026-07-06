"""Physical regularization losses for side chains.

Used when the predicted residue type != GT type (so atom-level coordinate MSE is
undefined): we fall back to physics — chemical bond lengths, bond angles, steric
clashes, and (stub) rotamer plausibility. Also usable as an auxiliary term when
types match. All terms are differentiable and finite.

This deliverable ships bond/angle/clash; `rotamer` is an intentional stub
(returns 0) per the spec — FangWu flagged the exact rotamer/physical
formulation as not finalized.
"""
from typing import Optional

import torch


def bond_loss(
    coords: torch.Tensor,     # [B, A, 3]
    bond_idx: torch.Tensor,   # [nb, 2] long
    ideal: torch.Tensor,      # [nb] ideal bond lengths (Angstrom)
) -> torch.Tensor:
    """Mean squared deviation of bonded pair distances from ideal lengths."""
    if bond_idx.numel() == 0:
        return coords.sum() * 0.0
    i, j = bond_idx[:, 0], bond_idx[:, 1]
    d = (coords[:, i] - coords[:, j]).norm(dim=-1)   # [B, nb]
    return ((d - ideal) ** 2).mean()


def angle_loss(
    coords: torch.Tensor,      # [B, A, 3]
    angle_idx: torch.Tensor,   # [na, 3] long (i, j-centre, k)
    ideal_cos: torch.Tensor,   # [na] cos of ideal angle
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean squared deviation of cos(angle) at centre atom j from ideal."""
    if angle_idx.numel() == 0:
        return coords.sum() * 0.0
    i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
    v1 = coords[:, i] - coords[:, j]
    v2 = coords[:, k] - coords[:, j]
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)  # [B, na]
    return ((cos - ideal_cos) ** 2).mean()


def clash_loss(
    coords: torch.Tensor,     # [B, A, 3]
    clash_dist: float = 2.0,
    valid_mask: Optional[torch.Tensor] = None,  # [B, A] bool
) -> torch.Tensor:
    """Penalise non-adjacent atom pairs closer than `clash_dist` (relu^2).

    Excludes self-pairs (and, when `valid_mask` given, padded atoms). This is a
    simplified steric term over all i<j pairs — bonded neighbours are tolerated
    because ideal bond lengths (~1.3-1.5 Å) exceed typical clash thresholds set
    below the van-der-Waals sum.
    """
    B, A, _ = coords.shape
    if A < 2:
        return coords.sum() * 0.0
    d = torch.cdist(coords, coords)             # [B, A, A]
    iu = torch.triu_indices(A, A, offset=1, device=coords.device)
    dij = d[:, iu[0], iu[1]]                     # [B, npair]
    pen = torch.relu(clash_dist - dij) ** 2
    if valid_mask is not None:
        pv = valid_mask[:, iu[0]] & valid_mask[:, iu[1]]
        pen = pen * pv.to(pen.dtype)
        denom = pv.sum().clamp_min(1).to(pen.dtype)
        return pen.sum() / denom
    return pen.mean()


def rotamer_loss(coords: torch.Tensor) -> torch.Tensor:
    """Stub (returns 0) — rotamer term not finalized this deliverable."""
    return coords.sum() * 0.0


def physical_loss(
    coords: torch.Tensor,
    bond_idx: Optional[torch.Tensor] = None,
    ideal_bond: Optional[torch.Tensor] = None,
    angle_idx: Optional[torch.Tensor] = None,
    ideal_cos: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    weights: Optional[dict] = None,
) -> dict:
    """Aggregate physical loss. Returns dict with bond/angle/clash/rotamer/total."""
    w = {"bond": 1.0, "angle": 1.0, "clash": 1.0, "rotamer": 1.0}
    if weights:
        w.update(weights)
    zero = coords.sum() * 0.0
    b = bond_loss(coords, bond_idx, ideal_bond) if bond_idx is not None else zero
    a = angle_loss(coords, angle_idx, ideal_cos) if angle_idx is not None else zero
    c = clash_loss(coords, valid_mask=valid_mask)
    r = rotamer_loss(coords)
    total = w["bond"] * b + w["angle"] * a + w["clash"] * c + w["rotamer"] * r
    return {"bond": b, "angle": a, "clash": c, "rotamer": r, "total": total}
