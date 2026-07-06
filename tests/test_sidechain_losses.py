"""Tests for side-chain losses: local coord, physical, routing."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.losses import sidechain_local_loss
from pxdesign_train.sidechain.physical import (
    bond_loss,
    angle_loss,
    clash_loss,
    physical_loss,
)
from pxdesign_train.sidechain.routing import route_sidechain_loss


# --- local-frame coordinate loss ---
def test_local_loss_zero_when_equal():
    y = torch.randn(2, 5, 3)
    m = torch.ones(2, 5, dtype=torch.bool)
    assert sidechain_local_loss(y, y, m).item() < 1e-8


def test_local_loss_masks_padded_atoms():
    p = torch.zeros(1, 3, 3)
    g = torch.zeros(1, 3, 3)
    g[0, 2] = 100.0  # huge error on a padded atom
    m = torch.tensor([[True, True, False]])
    assert sidechain_local_loss(p, g, m).item() < 1e-8  # padded atom ignored


def test_local_loss_backprop_finite():
    p = torch.randn(2, 5, 3, requires_grad=True)
    g = torch.randn(2, 5, 3)
    m = torch.ones(2, 5, dtype=torch.bool)
    sidechain_local_loss(p, g, m).backward()
    assert torch.isfinite(p.grad).all()


# --- physical losses ---
def test_clash_penalizes_overlap():
    close = torch.zeros(2, 3)
    far = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
    assert clash_loss(close[None]).item() > clash_loss(far[None]).item()


def test_bond_penalizes_wrong_length():
    pos = torch.tensor([[[0.0, 0, 0], [1.53, 0, 0]]])
    bad = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0]]])
    idx = torch.tensor([[0, 1]])
    ideal = torch.tensor([1.53])
    assert bond_loss(bad, idx, ideal).item() > bond_loss(pos, idx, ideal).item()


def test_physical_terms_finite_and_backprop():
    coords = torch.randn(2, 6, 3, requires_grad=True)
    idx = torch.tensor([[0, 1], [1, 2]])
    ideal = torch.tensor([1.5, 1.5])
    aidx = torch.tensor([[0, 1, 2]])
    acos = torch.tensor([-0.5])
    out = physical_loss(coords, bond_idx=idx, ideal_bond=ideal,
                        angle_idx=aidx, ideal_cos=acos)
    for k in ("bond", "angle", "clash", "rotamer", "total"):
        assert torch.isfinite(out[k]).all()
    assert out["rotamer"].item() == 0.0  # stub
    out["total"].backward()
    assert torch.isfinite(coords.grad).all()


# --- routing ---
def test_routing_splits_by_type_match():
    logits = torch.zeros(3, 20)
    logits[0, 5] = 10.0
    logits[1, 2] = 10.0
    logits[2, 7] = 10.0
    gt = torch.tensor([5, 9, 7])  # res0 match, res1 mismatch, res2 match
    got = {}

    def coord_fn(mask):
        got["coord"] = mask.clone()
        return mask.float().sum()

    def phys_fn(mask):
        got["phys"] = mask.clone()
        return mask.float().sum()

    total = route_sidechain_loss(logits, gt, coord_fn, phys_fn)
    assert got["coord"].tolist() == [True, False, True]
    assert got["phys"].tolist() == [False, True, False]
    assert total.item() == 3.0  # 2 matched + 1 mismatched
