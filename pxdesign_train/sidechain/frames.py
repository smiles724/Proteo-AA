"""Residue-local frames from backbone atoms (N, CA, C).

Ground-truth side-chain geometry is stored in residue-local coordinates, then
attached to the active backbone frame for global-coordinate S_phi supervision.
This preserves local geometry while keeping model outputs and coordinate losses
in the global frame.

Frame convention (Gram-Schmidt, AF2-style):
  e1 = normalize(C  - CA)
  e2 = normalize((N - CA) orthogonalized against e1)
  e3 = e1 x e2
  R  = [e1 | e2 | e3]  (columns are the local basis; maps local -> global)
  t  = CA (frame origin)

So  x_global = R @ x_local + t   and   x_local = R^T (x_global - t).
"""
import torch
import torch.nn.functional as F


def build_frame(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor):
    """Build per-residue local frames.

    Args:
        n, ca, c: backbone atom coords, each [..., 3].
    Returns:
        R: [..., 3, 3] rotation (columns = local basis, maps local->global).
        t: [..., 3] frame origin (== ca).
    """
    e1 = F.normalize(c - ca, dim=-1)
    u = n - ca
    u = u - (u * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # [..., 3, 3], column k = e_{k+1}
    return R, ca


def valid_ncac(n, ca, c, observed=None):
    """Coordinate validity, separate from atom inventory and observation labels."""
    valid = torch.isfinite(torch.stack((n, ca, c), -2)).all(dim=(-1, -2))
    v, u = torch.nan_to_num(c - ca), torch.nan_to_num(n - ca)
    valid = valid & (v.norm(dim=-1) > 1e-6) & (u.norm(dim=-1) > 1e-6)
    valid = valid & (torch.cross(v, u, dim=-1).norm(dim=-1) > 1e-6)
    if observed is not None:
        valid = valid & observed.bool().all(-1)
    return valid


def valid_rigid_frame(R, t):
    """A local target needs a finite orthonormal frame, not merely atom indices."""
    clean = torch.nan_to_num(R.float())
    eye = torch.eye(3, device=R.device)
    return (torch.isfinite(R).all(dim=(-1, -2)) & torch.isfinite(t).all(-1)
            & ((clean.transpose(-1, -2) @ clean - eye).abs().amax(dim=(-1, -2)) < 1e-4)
            & ((torch.linalg.det(clean) - 1).abs() < 1e-4))


def to_local(x_global: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Global -> local. x_global: [..., A, 3], R: [..., 3, 3], t: [..., 3]."""
    return torch.einsum("...ij,...aj->...ai", R.transpose(-1, -2), x_global - t[..., None, :])


def to_global(x_local: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Local -> global. x_local: [..., A, 3], R: [..., 3, 3], t: [..., 3]."""
    return torch.einsum("...ij,...aj->...ai", R, x_local) + t[..., None, :]


def dihedral(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """Signed IUPAC dihedral p0-p1-p2-p3, in radians. Each p is [..., 3]."""
    b0 = p0 - p1
    b1 = F.normalize(p2 - p1, dim=-1)
    b2 = p3 - p2
    # components of b0 and b2 perpendicular to the p1->p2 axis
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


# A predicted C(i-1)-N(i) further apart than this is not a peptide bond, so phi/psi
# across it are meaningless. Deliberately loose (a real bond is ~1.33 A): this is a
# guard against a garbage backbone, not a geometry check.
MAX_PEPTIDE_BOND = 2.5


def phi_psi_from_ncac(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
    have: torch.Tensor = None,
):
    """Backbone dihedrals (Overleaf 0714 appendix, Step 2).

        phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
        psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})

    phi is undefined at the first residue of a chain (no preceding C) and psi at the
    last (no following N); both are undefined across a chain break. Those positions
    come back as NaN, and the caller falls back to the backbone-independent marginal.

    Args:
        n, ca, c: [..., L, 3] backbone atom coordinates (predicted, or GT in warmup).
        residue_index: [L] long — author residue numbering (gaps mark chain breaks).
        asym_id: [L] long — chain id.
        have: [L] bool — token has all three frame atoms. Default: all True.
    Returns:
        phi, psi: [..., L] float radians, NaN where undefined.
    """
    L = n.shape[-2]
    dev = n.device
    ri = residue_index.to(dev).long()
    ai = asym_id.to(dev).long()
    if have is None:
        have = torch.ones(L, dtype=torch.bool, device=dev)
    have = have.to(dev)
    ri, ai, have = torch.broadcast_tensors(ri, ai, have)

    prev_ok = torch.zeros_like(have, dtype=torch.bool)
    prev_ok[..., 1:] = (ai[..., 1:] == ai[..., :-1]) & (ri[..., 1:] == ri[..., :-1] + 1) & have[..., 1:] & have[..., :-1]
    next_ok = torch.zeros_like(have, dtype=torch.bool)
    next_ok[..., :-1] = (ai[..., :-1] == ai[..., 1:]) & (ri[..., :-1] + 1 == ri[..., 1:]) & have[..., :-1] & have[..., 1:]

    ar = torch.arange(L, device=dev)
    c_prev = c[..., (ar - 1).clamp_min(0), :]
    n_next = n[..., (ar + 1).clamp_max(L - 1), :]

    # A predicted chain can be geometrically broken even where the numbering is contiguous.
    bond_prev = (c_prev - n).norm(dim=-1) <= MAX_PEPTIDE_BOND        # [..., L]
    bond_next = (n_next - c).norm(dim=-1) <= MAX_PEPTIDE_BOND

    phi = dihedral(c_prev, n, ca, c)
    psi = dihedral(n, ca, c, n_next)

    nan = torch.full_like(phi, float("nan"))
    phi = torch.where(prev_ok & bond_prev, phi, nan)
    psi = torch.where(next_ok & bond_next, psi, nan)
    return phi, psi


def backbone_phi_psi(
    coords: torch.Tensor,
    bb_idx: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
):
    """phi/psi of the PREDICTED backbone, gathering N/CA/C out of an atom array.

    Args:
        coords: [..., N_atom, 3] predicted global coordinates (x_hat_0).
        bb_idx: [L, 3] or [L, 4] long — atom indices of (N, CA, C[, O]); -1 = missing.
    Returns:
        phi, psi: [..., L] float radians, NaN where undefined.
    """
    xyz, present = gather_backbone(coords, bb_idx[..., :3])
    n, ca, c = xyz.unbind(-2)
    return phi_psi_from_ncac(n, ca, c, residue_index, asym_id, have=present.all(-1))


def gather_backbone(coords: torch.Tensor, bb_idx: torch.Tensor):
    """Gather [..., L, K, 3] with explicit broadcastable leading axes.

    For [B,S,N,3] coordinates, use [B,1,L,K] per-item indices. Absent atoms
    are zeroed and masked, never a copy of atom zero.
    """
    lead = torch.broadcast_shapes(coords.shape[:-2], bb_idx.shape[:-2])
    n_atom = coords.shape[-2]
    if n_atom == 0:
        raise ValueError("Cannot gather from an empty atom array")
    idx = bb_idx.to(coords.device).long().expand(*lead, *bb_idx.shape[-2:])
    valid = (idx >= 0) & (idx < n_atom)
    xyz = coords.expand(*lead, n_atom, 3).reshape(-1, n_atom, 3)
    gathered = xyz.gather(1, idx.clamp(0, n_atom - 1).reshape(xyz.shape[0], -1, 1).expand(-1, -1, 3))
    gathered = gathered.reshape(*idx.shape, 3)
    return torch.where(valid[..., None], gathered, 0.0), valid


def frames_from_backbone_index(coords: torch.Tensor, bb_idx: torch.Tensor):
    """Frames from N/CA/C with matching leading axes; O is not required.

    Returns R [...,L,3,3], t [...,L,3], valid [...,L]. Invalid/degenerate
    frames have identity rotation, zero translation and valid=False.
    """
    xyz, present = gather_backbone(coords, bb_idx[..., :3])
    n, ca, c = xyz.unbind(-2)
    valid = valid_ncac(n, ca, c, present)
    R, t = build_frame(torch.nan_to_num(n), torch.nan_to_num(ca), torch.nan_to_num(c))
    R = torch.where(valid[..., None, None], R, torch.eye(3, device=R.device, dtype=R.dtype))
    t = torch.where(valid[..., None], t, 0.0)
    return R, t, valid
