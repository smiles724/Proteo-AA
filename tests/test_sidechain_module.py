"""Tests for SideChainModule (S_phi) and h_res feedback."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    sidechain_mask,
    sidechain_atom_name_ids,
)
from pxdesign_train.sidechain.module import SideChainModule
from pxdesign_train.sidechain.feedback import HResFeedback

C_RES = 16


def _toy_batch():
    restypes = ["ALA", "PHE", "LYS"]           # 1, 7, 5 side-chain atoms
    L = len(restypes)
    atom_mask = sidechain_mask(restypes)[None]        # [1, L, MAX_SC]
    atom_ids = sidechain_atom_name_ids(restypes)[None]  # [1, L, MAX_SC]
    h_res = torch.randn(1, L, C_RES, requires_grad=True)
    logits = torch.randn(1, L, 20)
    noisy = torch.randn(1, L, MAX_SC, 3)
    t = torch.tensor([0.5])
    return restypes, atom_mask, atom_ids, h_res, logits, noisy, t


def _module(scale=1.0):
    return SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                           n_heads=4, trunk_grad_scale=scale)


def test_forward_shape_and_padding():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, feats = _module().forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    assert y0.shape == (1, 3, MAX_SC, 3)
    assert feats.shape == (1, 3, MAX_SC, 32)
    assert y0.requires_grad
    # padded atoms (beyond each residue's side-chain count) are exactly zero
    assert torch.count_nonzero(y0[0, 0, 1:]) == 0     # ALA: only CB valid
    assert torch.count_nonzero(y0[0, 2, 5:]) == 0     # LYS: 5 valid


def test_gly_no_nan():
    restypes = ["GLY"]  # zero side-chain atoms -> fully padded residue
    atom_mask = sidechain_mask(restypes)[None]
    atom_ids = sidechain_atom_name_ids(restypes)[None]
    h = torch.randn(1, 1, C_RES)
    y0, _ = _module().forward(h, torch.randn(1, 1, 20), atom_ids, atom_mask,
                              torch.randn(1, 1, MAX_SC, 3), torch.tensor([0.3]))
    assert torch.isfinite(y0).all()
    assert torch.count_nonzero(y0) == 0


def test_grad_reaches_hres_when_coupled():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, _ = _module(scale=1.0).forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    y0.sum().backward()
    assert h_res.grad is not None and torch.count_nonzero(h_res.grad) > 0


def test_grad_cut_when_readonly():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, _ = _module(scale=0.0).forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    y0.sum().backward()
    assert h_res.grad is None or torch.count_nonzero(h_res.grad) == 0


# --- feedback ---
def test_feedback_shape_and_grad_flow():
    fb = HResFeedback(c_atom=32, c_res=C_RES)
    atom_feats = torch.randn(1, 3, MAX_SC, 32, requires_grad=True)
    atom_mask = sidechain_mask(["ALA", "PHE", "LYS"])[None]
    h_res = torch.randn(1, 3, C_RES)
    hp = fb(atom_feats, atom_mask, h_res, detach=False)
    assert hp.shape == (1, 3, C_RES)
    hp.sum().backward()
    assert atom_feats.grad is not None and torch.count_nonzero(atom_feats.grad) > 0


def test_feedback_detach_cuts_sidechain_grad():
    fb = HResFeedback(c_atom=32, c_res=C_RES)
    atom_feats = torch.randn(1, 3, MAX_SC, 32, requires_grad=True)
    atom_mask = sidechain_mask(["ALA", "PHE", "LYS"])[None]
    h_res = torch.randn(1, 3, C_RES)
    hp = fb(atom_feats, atom_mask, h_res, detach=True)
    hp.sum().backward()
    assert atom_feats.grad is None or torch.count_nonzero(atom_feats.grad) == 0
