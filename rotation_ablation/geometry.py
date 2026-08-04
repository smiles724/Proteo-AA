"""Rigid-transform helpers used by the paired ablations."""

from __future__ import annotations

import math

import torch


def random_rotation(
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw a proper 3-D rotation using a normalized random quaternion."""
    q = torch.randn(4, generator=generator, device=device, dtype=dtype)
    q = q / q.norm().clamp_min(torch.finfo(dtype).eps)
    w, x, y, z = q.unbind()
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def apply_rigid(x: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply x -> Qx+t to row-vector coordinate tensors ending in [..., 3]."""
    return torch.einsum("ij,...j->...i", q, x) + t


def rotate_frames(frame_r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate local-to-global frame matrices: R^Q = Q R."""
    return torch.einsum("ij,...jk->...ik", q, frame_r)


def canonicalize(x_q: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Undo x -> Qx+t."""
    return torch.einsum("ji,...j->...i", q, x_q - t)


def rotate_about_axis(
    coords: torch.Tensor,
    axis_start: torch.Tensor,
    axis_end: torch.Tensor,
    selection: torch.Tensor,
    angle_degrees: float,
) -> torch.Tensor:
    """Rotate selected points around an axis using Rodrigues' formula.

    This is the geometry primitive for the chi positive control. ``coords`` is
    [..., A, 3], ``selection`` is broadcastable to [..., A], and the two axis
    points are broadcastable to [..., 3].
    """
    theta = math.radians(float(angle_degrees))
    axis = axis_end - axis_start
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = coords - axis_start[..., None, :]
    c = math.cos(theta)
    s = math.sin(theta)
    cross = torch.cross(axis[..., None, :].expand_as(v), v, dim=-1)
    dot = (v * axis[..., None, :]).sum(dim=-1, keepdim=True)
    rotated = axis_start[..., None, :] + c * v + s * cross + (1 - c) * dot * axis[..., None, :]
    return torch.where(selection[..., None].bool(), rotated, coords)
