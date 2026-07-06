"""Leakage-free side-chain initialization.

The hard leakage rule (SideCraft spec / Overleaf red text): we do NOT initialize
side-chain denoising by partially noising the ground-truth side-chain
coordinates, because a partially-noised side chain still encodes residue-specific
topology and thereby leaks amino-acid identity. Instead we start from isotropic
Gaussian noise in the residue-local frame, so the module receives only
inference-available information.

By construction this function takes NO ground-truth argument — that is enforced
by a test (`test_no_gt_argument`) so the leakage rule cannot be silently broken.
"""
from typing import Optional

import torch


def gaussian_init_local(
    mask: torch.Tensor,
    sigma: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample local-frame side-chain coordinates from N(0, sigma^2 I).

    Args:
        mask: [..., A] bool valid-atom mask.
        sigma: noise scale.
        generator: optional torch.Generator for reproducibility.
    Returns:
        [..., A, 3] Gaussian, zeroed at invalid (padded) atoms.
    """
    shape = (*mask.shape, 3)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32) * sigma
    return noise * mask[..., None].to(noise.dtype)
