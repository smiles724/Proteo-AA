"""Tests for the cycle-closure pieces: HResInjector + post-refinement losses."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.sidechain.coevolution import HResInjector
from pxdesign_train.loss import PXDesignLoss


def test_hres_injector_shape_and_grad():
    inj = HResInjector(c_hres=32, c_trunk=16)
    h = torch.randn(2, 5, 32, requires_grad=True)
    out = inj(h)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    assert torch.isfinite(h.grad).all()


def _coord_args(n=6):
    pred = torch.randn(1, n, 3)
    gt = torch.randn(1, n, 3)
    sigma = torch.tensor([10.0])
    cmask = torch.ones(n)
    rep = torch.ones(n, dtype=torch.bool)
    return pred, gt, sigma, cmask, rep


def _loss():
    return PXDesignLoss(align_before_mse=False, weight_lddt=0.0, weight_disto=0.0)


def test_bb_post_contributes():
    pred, gt, sigma, cmask, rep = _coord_args()
    post_pred = torch.randn(1, 6, 3, requires_grad=True)
    post_gt = torch.randn(1, 6, 3)
    base = _loss().forward(pred, gt, sigma, cmask, rep)["loss"]
    out = _loss().forward(pred, gt, sigma, cmask, rep,
                          post_pred_coordinate=post_pred, post_gt_coordinate_aug=post_gt)
    assert out["bb_post"].item() > 0.0
    assert out["loss"].item() > base.item()
    out["loss"].backward()
    assert torch.isfinite(post_pred.grad).all()


def test_aa_post_contributes():
    pred, gt, sigma, cmask, rep = _coord_args()
    post_logits = torch.randn(6, 20, requires_grad=True)  # [N_token, 20], matches aa_clean
    aa_clean = torch.randint(0, 20, (6,))
    aa_mask = torch.ones(6)
    out = _loss().forward(pred, gt, sigma, cmask, rep,
                          post_aa_logits=post_logits, aa_clean=aa_clean, aa_loss_mask=aa_mask)
    assert out["aa_post"].item() > 0.0
    out["loss"].backward()
    assert torch.isfinite(post_logits.grad).all()


def test_backward_compat_no_post_terms():
    pred, gt, sigma, cmask, rep = _coord_args()
    out = _loss().forward(pred, gt, sigma, cmask, rep)
    assert out["bb_post"].item() == 0.0 and out["aa_post"].item() == 0.0
