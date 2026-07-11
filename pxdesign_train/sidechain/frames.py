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


def frames_from_backbone_index(coords: torch.Tensor, bb_idx: torch.Tensor):
    """Build per-residue frames from PREDICTED backbone coords by gathering each
    token's N/CA/C atoms (paper Stage II-B: F_hat = Frame(x_hat_N, x_hat_CA, x_hat_C)).

    Args:
        coords: [..., N_atom, 3] predicted (or any) global coordinates.
        bb_idx: [L, 3] long — atom indices of (N, CA, C) per token; -1 = invalid.
    Returns:
        R: [..., L, 3, 3], t: [..., L, 3], valid: [L] bool (False where bb_idx<0).
    """
    valid = (bb_idx >= 0).all(dim=-1)                    # [L]
    safe = bb_idx.clamp_min(0)                           # gather needs non-neg
    n = coords[..., safe[:, 0], :]                       # [..., L, 3]
    ca = coords[..., safe[:, 1], :]
    c = coords[..., safe[:, 2], :]
    R, t = build_frame(n, ca, c)                         # [..., L, 3, 3], [..., L, 3]
    return R, t, valid
