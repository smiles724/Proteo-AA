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
    ri = residue_index.to(dev).reshape(-1).long()
    ai = asym_id.to(dev).reshape(-1).long()
    if have is None:
        have = torch.ones(L, dtype=torch.bool, device=dev)
    have = have.to(dev).reshape(-1)

    prev_ok = torch.zeros(L, dtype=torch.bool, device=dev)
    prev_ok[1:] = (ai[1:] == ai[:-1]) & (ri[1:] == ri[:-1] + 1) & have[1:] & have[:-1]
    next_ok = torch.zeros(L, dtype=torch.bool, device=dev)
    next_ok[:-1] = (ai[:-1] == ai[1:]) & (ri[:-1] + 1 == ri[1:]) & have[:-1] & have[1:]

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
    bb = bb_idx[..., :3]
    have = (bb >= 0).all(dim=-1)                          # [L]
    safe = bb.clamp_min(0)
    n = coords[..., safe[:, 0], :]                        # [..., L, 3]
    ca = coords[..., safe[:, 1], :]
    c = coords[..., safe[:, 2], :]
    return phi_psi_from_ncac(n, ca, c, residue_index, asym_id, have=have)


def frames_from_backbone_index(coords: torch.Tensor, bb_idx: torch.Tensor):
    """Build per-residue frames from PREDICTED backbone coords by gathering each
    token's N/CA/C atoms (paper Stage II-B: F_hat = Frame(x_hat_N, x_hat_CA, x_hat_C)).

    Args:
        coords: [..., N_atom, 3] predicted (or any) global coordinates.
        bb_idx: [L, 3] or [L, 4] long — atom indices of (N, CA, C[, O]) per token;
            -1 = invalid. Only the first three columns are used: the featurizer's
            `sc_bb_atom_idx` is (N, CA, C, O) and its O column may be -1 on a token
            whose frame atoms are all present, so validity must NOT be tested over
            all four columns.
    Returns:
        R: [..., L, 3, 3], t: [..., L, 3], valid: [L] bool (False where bb_idx<0).
    """
    bb_idx = bb_idx[..., :3]                             # frame atoms only (N, CA, C)
    valid = (bb_idx >= 0).all(dim=-1)                    # [L]
    safe = bb_idx.clamp_min(0)                           # gather needs non-neg
    n = coords[..., safe[:, 0], :]                       # [..., L, 3]
    ca = coords[..., safe[:, 1], :]
    c = coords[..., safe[:, 2], :]
    R, t = build_frame(n, ca, c)                         # [..., L, 3, 3], [..., L, 3]
    return R, t, valid
