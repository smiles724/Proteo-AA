"""Tests for leakage-free Gaussian side-chain init."""
import inspect
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.init import gaussian_init_local, template_init_local

_GT_NAMES = {"gt", "gt_coords", "x0", "true", "ground_truth"}


def test_no_gt_argument():
    """Leakage guard: the initializer must not be able to see GT coords."""
    params = set(inspect.signature(gaussian_init_local).parameters)
    assert not (params & _GT_NAMES)


def test_template_init_no_gt_argument():
    """Same leakage guard for the template-anchored init (Overleaf para 221).

    Allowed: residue TYPE, the atom mask, and the PREDICTED backbone (par.221 initializes
    "around the predicted backbone frames"; `backbone` exists so a phi/psi-conditioned
    rotamer library can be registered as the mu_ideal provider).
    Never allowed: ground-truth SIDE-CHAIN coordinates -- a partially-noised GT side chain
    still encodes the residue identity through its own geometry.
    """
    params = set(inspect.signature(template_init_local).parameters)
    assert not (params & _GT_NAMES)
    assert params == {"type_idx", "mask", "sigma_T", "generator", "backbone"}


def test_shape_and_masking():
    m = torch.tensor([[True, True, False]])
    y = gaussian_init_local(m, sigma=1.0, generator=torch.Generator().manual_seed(0))
    assert y.shape == (1, 3, 3)
    assert torch.count_nonzero(y[0, 2]) == 0  # padded atom stays zero


def test_reproducible_under_seed():
    m = torch.ones(2, 5, dtype=torch.bool)
    a = gaussian_init_local(m, generator=torch.Generator().manual_seed(7))
    b = gaussian_init_local(m, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a, b)


def test_sigma_scales_std():
    m = torch.ones(1, 2000, dtype=torch.bool)
    y = gaussian_init_local(m, sigma=3.0, generator=torch.Generator().manual_seed(1))
    assert abs(y.std().item() - 3.0) < 0.2
