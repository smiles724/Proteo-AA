"""Numerically stable invariant/equivariant metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .geometry import apply_rigid, canonicalize


def _expanded_mask(mask: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    out = mask.bool()
    while out.dim() < value.dim():
        out = out.unsqueeze(-1)
    return out.expand_as(value)


def relative_frobenius(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> float:
    valid = _expanded_mask(mask, reference)
    if valid is not None:
        reference = reference.masked_select(valid)
        candidate = candidate.masked_select(valid)
    numerator = torch.linalg.vector_norm((candidate - reference).float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(eps)
    return float((numerator / denominator).cpu())


def residue_cosine(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Cosine similarity after flattening all non-batch/non-residue features."""
    if reference.dim() < 3:
        return {}
    if residue_mask is not None:
        expected = residue_mask.shape if residue_mask.dim() == 2 else (1, residue_mask.shape[0])
        if tuple(reference.shape[:2]) != tuple(expected):
            return {}
    a = reference.float().reshape(reference.shape[0], reference.shape[1], -1)
    b = candidate.float().reshape(candidate.shape[0], candidate.shape[1], -1)
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    if residue_mask is not None:
        valid = residue_mask.bool()
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        cos = cos[valid]
    else:
        cos = cos.reshape(-1)
    if cos.numel() == 0:
        return {"cos_mean": float("nan"), "cos_min": float("nan")}
    return {
        "cos_mean": float(cos.mean().cpu()),
        "cos_min": float(cos.min().cpu()),
    }


def rmsd(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    sq = (candidate.float() - reference.float()).square().sum(dim=-1)
    if mask is not None:
        valid = mask.bool()
        sq = sq[valid]
    else:
        sq = sq.reshape(-1)
    if sq.numel() == 0:
        return float("nan")
    return float(torch.sqrt(sq.mean()).cpu())


def equivariant_rmsd(
    reference: torch.Tensor,
    transformed_prediction: torch.Tensor,
    q: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    expected = apply_rigid(reference, q, t)
    return {
        "eq_rmsd": rmsd(expected, transformed_prediction, mask),
        "canonical_rmsd": rmsd(reference, canonicalize(transformed_prediction, q, t), mask),
    }


def summarize_invariant(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    residue_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    out = {"relative_frobenius": relative_frobenius(reference, candidate, mask)}
    out.update(residue_cosine(reference, candidate, residue_mask))
    return out


def aggregate_numeric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively aggregate same-shaped numeric dictionaries with mean/max."""
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    out: dict[str, Any] = {}
    for key in keys:
        vals = [row[key] for row in rows]
        if all(isinstance(v, dict) for v in vals):
            out[key] = aggregate_numeric(vals)
        elif all(isinstance(v, (int, float)) for v in vals):
            tensor = torch.tensor(vals, dtype=torch.float64)
            finite = tensor[torch.isfinite(tensor)]
            out[key] = {
                "mean": float(finite.mean()) if finite.numel() else float("nan"),
                "max": float(finite.max()) if finite.numel() else float("nan"),
            }
    return out
