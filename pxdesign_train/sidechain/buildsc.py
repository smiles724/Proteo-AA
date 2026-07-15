"""BuildSC — Cartesian side-chain template from ideal geometry + torsions.

Overleaf 0714 appendix, Step 3::

    { mu_ideal_{ij} }_{j in A_sc(a_hat_i)} = BuildSC( a_hat_i, G_ideal(a_hat_i), chi_i )
    mu_ideal_{ij} = mu_ideal_{a_hat_i, chi_i, j}

i.e. the template depends on the residue identity AND the selected rotamer. That
second dependence is the whole point of the appendix, and it is what the previous
static ``templates.ideal_template`` (one fixed CCD conformer per residue type) did
not have.

How the torsions are applied
----------------------------
``G_ideal`` is realised as the CCD ideal conformer (correct bond lengths and bond
angles, one arbitrary chi) plus the connectivity-derived rigid groups in
``chi_constants``. Setting chi_k is then a rigid rotation of the atoms distal to
the chi_k bond axis about that axis. A rigid rotation about a bond preserves every
bond length and every bond angle in the residue, so the output has exactly the CCD's
ideal covalent geometry and exactly the requested torsions — which is precisely the
(G_ideal, chi) -> Cartesian map the appendix asks for, without needing AlphaFold's
``rigid_group_atom_positions`` table (which Protenix does not ship).

Everything happens in the residue-LOCAL frame (origin CA, ``frames.build_frame``),
so the result drops straight into ``y_T = mu + sigma_T * eps`` and then through the
predicted frame F_hat. chi1 is defined by N-CA-CB-CG, so the ideal *backbone* N/CA/C
must live in that same local frame; that is ``chi_constants.IDEAL_BB_LOCAL``.

PRO is ring-closed (CD bonds back to the backbone N), so no rigid rotation about
CA-CB or CB-CG exists that keeps the pyrrolidine ring intact.
``chi_constants.CHI_ROTATABLE`` reports this, and PRO therefore keeps its CCD
conformer. This is a real geometric limitation, not an oversight; PRO's static
template is only ~0.7 A from real prolines because the ring has so little freedom.
"""
from typing import Optional

import torch

from pxdesign_train.sidechain.chi_constants import (
    CHI_ATOM_IDX,
    CHI_DOWNSTREAM,
    CHI_MASK,
    CHI_ROTATABLE,
    IDEAL_BB_LOCAL,
    MAX_CHI,
    N_BB_FRAME,
)
from pxdesign_train.sidechain.frames import dihedral
from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3
from pxdesign_train.sidechain.templates import IDEAL_SC_LOCAL, IDEAL_SC_MASK


def _rodrigues(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate v [..., A, 3] about unit `axis` [..., 3] by `angle` [...] radians."""
    k = axis[..., None, :]                       # [..., 1, 3]
    a = angle[..., None, None]                   # [..., 1, 1]
    return (
        v * torch.cos(a)
        + torch.cross(k.expand_as(v), v, dim=-1) * torch.sin(a)
        + k * (k * v).sum(-1, keepdim=True) * (1.0 - torch.cos(a))
    )


def build_sidechain_local(
    type_idx: torch.Tensor,
    chi: Optional[torch.Tensor] = None,
):
    """BuildSC(a_hat, G_ideal(a_hat), chi) -> residue-local template.

    Args:
        type_idx: [...] long, residue types in ``STD_AA_3`` order. Out-of-range
            values are treated as GLY (empty side chain), as everywhere else.
        chi: [..., MAX_CHI] float, target torsions in RADIANS. NaN entries (and
            torsions the residue does not have, or that are ring-closed) are left
            at their CCD ideal value. ``None`` returns the CCD conformer unchanged,
            i.e. exactly the old static template.
    Returns:
        coords: [..., MAX_SC, 3] float32, residue-local, zeros at padded slots.
        mask:   [..., MAX_SC] bool.
    """
    dev = type_idx.device
    gly = STD_AA_3.index("GLY")
    valid = (type_idx >= 0) & (type_idx < len(STD_AA_3))
    tix = torch.where(valid, type_idx, torch.full_like(type_idx, gly)).long()

    sc = IDEAL_SC_LOCAL.to(dev)[tix].clone()            # [..., MAX_SC, 3]
    mask = IDEAL_SC_MASK.to(dev)[tix]                   # [..., MAX_SC]
    if chi is None:
        return sc, mask

    bb = IDEAL_BB_LOCAL.to(dev)[tix]                    # [..., 3, 3]  (N, CA, C)
    atom_idx = CHI_ATOM_IDX.to(dev)[tix]                # [..., MAX_CHI, 4]
    kmask = CHI_MASK.to(dev)[tix]                       # [..., MAX_CHI]
    rot_ok = CHI_ROTATABLE.to(dev)[tix]                 # [..., MAX_CHI]
    downstream = CHI_DOWNSTREAM.to(dev)[tix]            # [..., MAX_CHI, MAX_SC]

    chi = chi.to(device=dev, dtype=sc.dtype)

    for k in range(MAX_CHI):
        target = chi[..., k]                                            # [...]
        active = kmask[..., k] & rot_ok[..., k] & torch.isfinite(target)
        if not bool(active.any()):
            continue

        # Recompute the current torsion from the CURRENT coordinates: chi_k's atoms
        # have already been moved by chi_1..chi_{k-1}, so the value in the CCD
        # conformer is stale by the time we get here.
        combined = torch.cat([bb, sc], dim=-2)                          # [..., 3+MAX_SC, 3]
        q = atom_idx[..., k, :]                                         # [..., 4]
        p = torch.gather(combined, -2, q[..., None].expand(*q.shape, 3))
        p0, p1, p2, p3 = p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :]

        cur = dihedral(p0, p1, p2, p3)                                  # [...]
        delta = torch.where(active, target - cur, torch.zeros_like(cur))

        axis = p2 - p1
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        moved = downstream[..., k, :] & active[..., None]               # [..., MAX_SC]
        # Rotate about the line through p2 (p2 lies on the axis and so is fixed).
        rel = sc - p2[..., None, :]
        rotated = _rodrigues(rel, axis, delta) + p2[..., None, :]
        sc = torch.where(moved[..., None], rotated, sc)

    return sc * mask[..., None].to(sc.dtype), mask


def chi_from_local(
    type_idx: torch.Tensor,
    sc_local: torch.Tensor,
    bb_local: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of BuildSC's torsion part: measure chi from local-frame coordinates.

    Used by the evaluation harness (and useful for a rotamer loss). Returns
    [..., MAX_CHI] radians, NaN for torsions the residue does not have.

    Args:
        bb_local: [..., 3, 3] the residue's OWN (N, CA, C) in the same local frame.
            chi1 is N-CA-CB-CG, so measuring chi on a real side chain wants that
            residue's real N, not the ideal one. Defaults to IDEAL_BB_LOCAL.
    """
    dev = type_idx.device
    gly = STD_AA_3.index("GLY")
    valid = (type_idx >= 0) & (type_idx < len(STD_AA_3))
    tix = torch.where(valid, type_idx, torch.full_like(type_idx, gly)).long()

    bb = IDEAL_BB_LOCAL.to(dev)[tix] if bb_local is None else bb_local.to(dev)
    combined = torch.cat([bb, sc_local.to(bb.dtype)], dim=-2)
    atom_idx = CHI_ATOM_IDX.to(dev)[tix]
    kmask = CHI_MASK.to(dev)[tix]

    out = []
    for k in range(MAX_CHI):
        q = atom_idx[..., k, :]
        p = torch.gather(combined, -2, q[..., None].expand(*q.shape, 3))
        ang = dihedral(p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :])
        out.append(torch.where(kmask[..., k], ang, torch.full_like(ang, float("nan"))))
    return torch.stack(out, dim=-1)


__all__ = ["build_sidechain_local", "chi_from_local", "MAX_CHI", "MAX_SC", "N_BB_FRAME"]
