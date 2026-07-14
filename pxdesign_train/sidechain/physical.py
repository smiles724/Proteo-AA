"""Physical regularization losses for side chains.

Used when the predicted residue type != GT type (so atom-level coordinate MSE is
undefined): we fall back to physics — chemical bond lengths, bond angles, steric
clashes, and (stub) rotamer plausibility. Also usable as an auxiliary term when
types match. All terms are differentiable and finite.

Implementation status (keep in sync with README):
  - clash, contact: implemented AND activated in training (the model computes
    them on predicted global side chains, see model._train_forward).
  - bond, angle, rotamer: implemented and unit-tested as differentiable terms,
    but NOT activated in training — they require residue-specific ideal-geometry
    tables (bond_idx/ideal, angle_idx/ideal_cos, torsion_idx/rotamer_targets)
    that are not wired yet, so `physical_loss` only includes them when those
    inputs are passed. `rotamer_loss` below is a real periodic dihedral penalty
    (not a zero stub); it is simply not fed torsion tables at train time.
"""
from typing import Optional

import torch


def select_context_atoms(
    ref_xyz: torch.Tensor,       # [B, L, 3]  anchor points (binder CAs)
    ref_mask: torch.Tensor,      # [B, L] bool
    atom_xyz: torch.Tensor,      # [B, N, 3]  every atom in the complex
    atom_mask: torch.Tensor,     # [B, N] bool
    atom_group: torch.Tensor,    # [B, N] long — token each atom belongs to
    radius: float = 10.0,
    max_atoms: int = 4096,
):
    """Pick the context atoms near the binder, bounded in count.

    The clash/contact terms only look at pairs closer than a few Angstrom, so
    scoring the side chain against EVERY atom of the complex is wasted compute
    and a `cdist` an order of magnitude larger than it needs to be (an 8k-atom
    receptor against 4k side-chain atoms is a ~100 MB tensor per sample, kept
    alive by autograd). We keep the atoms within `radius` of any binder CA, and
    hard-cap at `max_atoms` nearest so the shape can never blow up on a large
    complex.

    Returns (coords [B, M, 3], mask [B, M], group [B, M]) with M <= max_atoms.
    Coordinates are DETACHED: the receptor/motif is fixed conditioning, so the
    side-chain objective must not be able to move it (the same stop-grad rule
    the frames and backbone context already follow).
    """
    B, N, _ = atom_xyz.shape
    big = torch.finfo(atom_xyz.dtype).max
    d = torch.cdist(atom_xyz.detach(), ref_xyz.detach())            # [B, N, L]
    d = d.masked_fill(~ref_mask[:, None, :], big)
    dmin = d.min(dim=-1).values                                      # [B, N]
    dmin = dmin.masked_fill(~atom_mask, big)

    m = min(int(max_atoms), N)
    sel = dmin.topk(m, dim=-1, largest=False).indices                # [B, m]
    coords = atom_xyz.detach().gather(1, sel[..., None].expand(-1, -1, 3))
    mask = dmin.gather(1, sel) <= radius
    group = atom_group.gather(1, sel)
    return coords, mask, group


def build_sidechain_context(
    xyz: torch.Tensor,          # [B, N_atom, 3]  x_denoised (augmented frame)
    center_idx: torch.Tensor,   # [B, L] long — each token's representative (CA) atom, -1 absent
    atom_to_token: torch.Tensor,  # [B, N_atom] long
    bb_atom_idx: torch.Tensor,  # [B, L, >=3] long — binder N/CA/C(/O); -1 on non-binder
    ca: Optional[torch.Tensor] = None,  # [B, L, 3] existing binder CA (frame origin)
    radius: float = 10.0,
    max_atoms: int = 4096,
):
    """Assemble everything S_phi needs to see the receptor / motif / ligand.

    Returns ``(ca_out [B,L,3], ctx_tok [B,L] bool, (coords, mask, group))``:

      * ``ca_out``  — CA for the cross-residue attention. The binder's rows keep
        their existing source (the frame origin: same atom, and unchanged under the
        GT-frame warmup); ONLY the context rows are filled in. Their frame origin is
        garbage — their ``bb_atom_idx`` is -1, which ``frames_from_backbone_index``
        clamps to atom 0 — so without this they would all sit on residue 0's N atom.
      * ``ctx_tok`` — tokens that are context, not binder: they own no S_phi atom, so
        they are attention KEYS only.
      * the context atom set for clash/contact (radius-filtered, capped, stop-grad).

    A token is the binder's iff it has a resolved N/CA/C frame; every other real
    token is context.
    """
    L = center_idx.shape[-1]
    is_binder = (bb_atom_idx[..., :3] >= 0).all(dim=-1)                 # [B, L]
    center_xyz = xyz.gather(1, center_idx.clamp_min(0)[..., None].expand(-1, -1, 3))
    ctx_tok = (center_idx >= 0) & ~is_binder

    if ca is None:
        ca_out = center_xyz.detach()
    else:
        ca_out = torch.where(is_binder[..., None], ca, center_xyz.detach())

    atom_valid = (atom_to_token >= 0) & (atom_to_token < L)
    ctx_atoms = select_context_atoms(
        ref_xyz=center_xyz,
        ref_mask=is_binder,
        atom_xyz=xyz,
        atom_mask=atom_valid,
        atom_group=atom_to_token,
        radius=radius,
        max_atoms=max_atoms,
    )
    return ca_out, ctx_tok, ctx_atoms


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
    group_id: Optional[torch.Tensor] = None,          # [B, A] long — residue of each side-chain atom
    context_coords: Optional[torch.Tensor] = None,    # [B, M, 3]
    context_mask: Optional[torch.Tensor] = None,      # [B, M] bool
    context_group_id: Optional[torch.Tensor] = None,  # [B, M] long — residue of each context atom
) -> torch.Tensor:
    """Steric term over the three pair classes the paper requires
    (§Physical regularization: "side-chain--backbone, side-chain--side-chain,
    and side-chain--context atom pairs"):

      * side-chain <-> side-chain   — all i<j pairs of `coords` (as before);
      * side-chain <-> backbone     — `context_coords` carries the backbone atoms;
      * side-chain <-> context      — ...and the receptor / motif / ligand atoms.

    The latter two are one cross term: `context_coords` is simply every atom the
    side chain must not overlap. Pairs inside the SAME residue are EXCLUDED via
    `group_id` / `context_group_id`, because a side chain is BONDED to its own
    backbone (CB-CA is ~1.53 Å, i.e. *below* `clash_dist`) — scoring that as a
    clash would fight the bond/angle losses instead of preventing an overlap.
    Intra-residue geometry is bond_loss/angle_loss/rotamer_loss's job.

    Returns intra + cross, each normalised over its own valid pairs.
    """
    B, A, _ = coords.shape
    intra = coords.sum() * 0.0
    if A >= 2:
        d = torch.cdist(coords, coords)             # [B, A, A]
        iu = torch.triu_indices(A, A, offset=1, device=coords.device)
        dij = d[:, iu[0], iu[1]]                     # [B, npair]
        pen = torch.relu(clash_dist - dij) ** 2
        if valid_mask is not None:
            pv = valid_mask[:, iu[0]] & valid_mask[:, iu[1]]
            pen = pen * pv.to(pen.dtype)
            intra = pen.sum() / pv.sum().clamp_min(1).to(pen.dtype)
        else:
            intra = pen.mean()

    if context_coords is None or context_coords.shape[-2] == 0:
        return intra

    dc = torch.cdist(coords, context_coords)         # [B, A, M]
    pen_c = torch.relu(clash_dist - dc) ** 2
    m = torch.ones_like(pen_c, dtype=torch.bool)
    if valid_mask is not None:
        m = m & valid_mask[:, :, None]
    if context_mask is not None:
        m = m & context_mask[:, None, :]
    if group_id is not None and context_group_id is not None:
        # Drop same-residue pairs: those atoms are bonded, not clashing.
        m = m & (group_id[:, :, None] != context_group_id[:, None, :])
    pen_c = pen_c * m.to(pen_c.dtype)
    cross = pen_c.sum() / m.sum().clamp_min(1).to(pen_c.dtype)
    return intra + cross


def _dihedral(p0, p1, p2, p3, eps: float = 1e-8) -> torch.Tensor:
    """Signed dihedral angle (radians) of atom quadruples. Each p*: [..., 3]."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def rotamer_loss(
    coords: torch.Tensor,          # [B, A, 3]
    torsion_idx: torch.Tensor,     # [nt, 4] long (i,j,k,l defining a chi torsion)
    targets: torch.Tensor,         # [n_rot] canonical staggered angles (radians)
) -> torch.Tensor:
    """Periodic penalty pulling each side-chain torsion toward the nearest
    canonical staggered rotamer value (min over targets of 1-cos(theta-target))."""
    if torsion_idx.numel() == 0:
        return coords.sum() * 0.0
    i, j, k, l = torsion_idx[:, 0], torsion_idx[:, 1], torsion_idx[:, 2], torsion_idx[:, 3]
    theta = _dihedral(coords[:, i], coords[:, j], coords[:, k], coords[:, l])  # [B, nt]
    # distance to each canonical target, periodic via cosine
    diff = theta.unsqueeze(-1) - targets.view(1, 1, -1)          # [B, nt, n_rot]
    pen = (1.0 - torch.cos(diff)).min(dim=-1).values             # [B, nt]
    return pen.mean()


def contact_loss(
    coords: torch.Tensor,              # [B, A, 3] side-chain atoms
    context_coords: torch.Tensor,      # [B, M, 3] backbone + receptor/motif/ligand atoms
    valid_mask: Optional[torch.Tensor] = None,    # [B, A] bool
    context_mask: Optional[torch.Tensor] = None,  # [B, M] bool
    max_dist: float = 8.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compatibility term: each side-chain atom should stay near some context
    atom (soft hinge on its nearest-context distance beyond `max_dist`) —
    discourages side chains flying away from the fold, and rewards sitting
    against the receptor / motif / ligand it is packing into.

    `context_mask` IS REQUIRED FOR CORRECTNESS whenever `context_coords` has
    padded rows. Callers build it from a per-token table (e.g. `sc_bb_coords`,
    which is all-zero on non-binder tokens, or a gather at `sc_bb_atom_idx`,
    whose -1 rows get clamped to atom 0). Those padded rows are NOT absent — an
    unmasked (0,0,0) row is an atom AT THE ORIGIN, and a clamped row is a
    duplicate of atom 0. Either way `min(dim=-1)` happily selects them, and the
    runaway penalty this loss exists to impose is silently zeroed: any side-chain
    atom within `max_dist` of that phantom scores 0 no matter where the real
    structure is. Masked rows are set to +inf so `min` cannot pick them; an item
    with no valid context atom contributes 0 rather than inf/NaN.
    """
    d = torch.cdist(coords, context_coords)            # [B, A, M]
    if context_mask is not None:
        d = d.masked_fill(~context_mask[:, None, :], float("inf"))
    nearest = d.min(dim=-1).values                     # [B, A]
    finite = torch.isfinite(nearest)
    pen = torch.relu(nearest - max_dist) ** 2
    pen = torch.where(finite, pen, torch.zeros_like(pen))
    m = finite if valid_mask is None else (valid_mask & finite)
    m = m.to(pen.dtype)
    return (pen * m).sum() / (m.sum() + eps)


def physical_loss(
    coords: torch.Tensor,
    bond_idx: Optional[torch.Tensor] = None,
    ideal_bond: Optional[torch.Tensor] = None,
    angle_idx: Optional[torch.Tensor] = None,
    ideal_cos: Optional[torch.Tensor] = None,
    torsion_idx: Optional[torch.Tensor] = None,
    rotamer_targets: Optional[torch.Tensor] = None,
    context_coords: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    context_group_id: Optional[torch.Tensor] = None,
    group_id: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    weights: Optional[dict] = None,
) -> dict:
    """Aggregate physical loss. Terms present only when their inputs are given.

    `context_coords` is every atom the side chain interacts with but does not
    own: its own backbone, other residues' backbones, and the receptor / motif /
    ligand atoms. It feeds BOTH the clash cross term and the contact term. Pass
    `context_mask` whenever it has padded rows (see `contact_loss`), and
    `group_id`/`context_group_id` so the clash term can drop bonded
    same-residue pairs.

    Returns dict with bond/angle/clash/rotamer/contact/total.
    """
    w = {"bond": 1.0, "angle": 1.0, "clash": 1.0, "rotamer": 1.0, "contact": 1.0}
    if weights:
        w.update(weights)
    zero = coords.sum() * 0.0
    b = bond_loss(coords, bond_idx, ideal_bond) if bond_idx is not None else zero
    a = angle_loss(coords, angle_idx, ideal_cos) if angle_idx is not None else zero
    c = clash_loss(
        coords,
        valid_mask=valid_mask,
        group_id=group_id,
        context_coords=context_coords,
        context_mask=context_mask,
        context_group_id=context_group_id,
    )
    r = (rotamer_loss(coords, torsion_idx, rotamer_targets)
         if torsion_idx is not None and rotamer_targets is not None else zero)
    ct = (contact_loss(coords, context_coords, valid_mask, context_mask)
          if context_coords is not None else zero)
    total = (w["bond"] * b + w["angle"] * a + w["clash"] * c
             + w["rotamer"] * r + w["contact"] * ct)
    return {"bond": b, "angle": a, "clash": c, "rotamer": r, "contact": ct, "total": total}
