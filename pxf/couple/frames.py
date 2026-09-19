"""Residue-local frames from predicted N, CA, C. One implementation, pinned.

The conditioner expresses every geometric feature in residue ``i``'s own frame,
which is what makes those features invariant to a global rotation or
translation of the structure. Two different frame conventions inside one model
would not fail loudly -- both are orthonormal, both give finite numbers -- so
the convention is written once here and used by the atom encoder, the pair
encoder and the tests alike.

The convention is Proteo-AA's own (``pxdesign_train/sidechain/frames.py``),
reproduced rather than imported so the training path does not depend on an
external checkout being present. ``tests/test_couple_frames.py`` pins the two
equal wherever that checkout is available, so a drift is a test failure rather
than a silent disagreement between what the model reads and what the metrics
report:

    e1 = normalize(C - CA)
    e2 = normalize((N - CA) orthogonalized against e1)
    e3 = e1 x e2
    R  = [e1 | e2 | e3]          columns are the local basis, local -> global
    t  = CA

so ``x_local = R^T (x_global - t)`` and the basis is right-handed by
construction (``det R = +1``).
"""

import torch
import torch.nn.functional as F

# A frame whose defining vectors are shorter than this, or nearly parallel, is
# not a frame: normalizing it would divide by ~0 and produce a rotation made of
# NaNs that silently poisons every feature downstream.
MIN_FRAME_NORM = 1e-6


def build_frame(n, ca, c):
    """``(R, t)`` for backbone atoms ``[..., 3]``. ``R`` maps local -> global."""
    e1 = F.normalize(c - ca, dim=-1)
    u = n - ca
    u = u - (u * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1), ca


def frame_is_valid(n, ca, c):
    """``[...]`` bool: the three atoms actually define a right-handed frame.

    Checked rather than assumed. A predicted backbone can place two of the three
    atoms on top of each other, and the resulting ``R`` is all-NaN -- which
    propagates through the encoder into the conditioning residual and turns one
    bad residue into a whole-structure failure.
    """
    finite = torch.isfinite(torch.stack((n, ca, c), dim=-2)).all(dim=(-1, -2))
    v, u = torch.nan_to_num(c - ca), torch.nan_to_num(n - ca)
    return (
        finite
        & (v.norm(dim=-1) > MIN_FRAME_NORM)
        & (u.norm(dim=-1) > MIN_FRAME_NORM)
        & (torch.cross(v, u, dim=-1).norm(dim=-1) > MIN_FRAME_NORM)
    )


def to_local(x_global, rotation, translation):
    """Global -> local. ``x_global`` ``[..., A, 3]``, ``rotation`` ``[..., 3, 3]``."""
    return torch.einsum(
        "...ij,...aj->...ai",
        rotation.transpose(-1, -2),
        x_global - translation[..., None, :],
    )


def rotate_to_local(vectors, rotation):
    """``R^T v`` for a *difference* vector: rotated, not translated."""
    return torch.einsum("...ij,...aj->...ai", rotation.transpose(-1, -2), vectors)
