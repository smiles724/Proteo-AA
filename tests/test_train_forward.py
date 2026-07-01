"""
Synthetic-batch sanity test for PXDesign-d training.

What this validates:
  * the loss module accepts realistic-shape tensors
  * `PXDesignLoss` produces a finite scalar
  * backward through MSE + distogram + smoothLDDT produces finite gradients

What this does NOT validate (deferred to later pieces):
  * `ProtenixDesignTrain.forward(mode="train")` against the real DiffusionModule
    — that needs a full Protenix install and a real feature dict
  * Featurization correctness (binder/target split, hotspot, pair-dist bins)

Run with:
    PYTHONPATH=../PXDesign:../Protenix python -m pytest tests/test_train_forward.py -v
"""
import os
import sys

import pytest
import torch

# Make the sibling repos importable without an editable install.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


def _fake_batch(
    n_token: int = 16,
    atoms_per_token: int = 4,
    n_sample: int = 4,
    seed: int = 0,
):
    """Build a minimal-shape batch.

    Atoms-per-token is fixed at 4 (matches the report's [xpb] N/Cα/C/O backbone).
    Rep-atom mask points to the 2nd atom of each token (the Cα).
    """
    torch.manual_seed(seed)
    n_atom = n_token * atoms_per_token

    gt = torch.randn(n_atom, 3) * 5.0       # roughly protein-scale coords
    gt = gt.unsqueeze(0)                     # batch=1
    coord_mask = torch.ones(1, n_atom)

    rep_atom_mask = torch.zeros(n_atom)
    rep_atom_mask[1::atoms_per_token] = 1    # Cα-style rep atoms

    # Mocked prediction: GT + small per-sample perturbation.
    pred = gt.unsqueeze(1).expand(1, n_sample, n_atom, 3).contiguous()
    pred = pred + 0.1 * torch.randn_like(pred)
    pred.requires_grad_(True)

    gt_aug = gt.unsqueeze(1).expand(1, n_sample, n_atom, 3).contiguous()

    # Sigma sampled from EDM log-normal at sigma_data=16.
    rnd = torch.randn(1, n_sample)
    sigma = (rnd * 1.5 + -1.2).exp() * 16.0

    return {
        "pred": pred,
        "gt_aug": gt_aug,
        "sigma": sigma,
        "coord_mask": coord_mask,
        "rep_atom_mask": rep_atom_mask,
        "n_token": n_token,
    }


def test_loss_finite_and_backprops():
    from pxdesign_train.loss import PXDesignLoss

    batch = _fake_batch()
    loss_mod = PXDesignLoss(align_before_mse=False)  # avoid CUDA-only rigid-align in CPU test

    # Synthetic distogram logits, requires_grad so we exercise the disto branch.
    n_token = batch["n_token"]
    logits = torch.randn(1, n_token, n_token, loss_mod.no_bins, requires_grad=True)

    out = loss_mod(
        pred_coordinate=batch["pred"],
        gt_coordinate_aug=batch["gt_aug"],
        sigma=batch["sigma"],
        coordinate_mask=batch["coord_mask"],
        rep_atom_mask=batch["rep_atom_mask"],
        distogram_logits=logits,
    )

    assert torch.isfinite(out["loss"]), out["loss"]
    assert torch.isfinite(out["mse"]) and out["mse"] >= 0
    assert torch.isfinite(out["lddt"])
    assert torch.isfinite(out["distogram"])

    out["loss"].backward()
    assert batch["pred"].grad is not None and torch.isfinite(batch["pred"].grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_sigma_gate_zeros_lddt_and_disto():
    """When all σ are huge, the gate should fire 0 and only MSE contributes."""
    from pxdesign_train.loss import PXDesignLoss

    batch = _fake_batch()
    loss_mod = PXDesignLoss(align_before_mse=False, sigma_low_threshold=4.0)

    sigma_big = torch.full_like(batch["sigma"], 100.0)
    n_token = batch["n_token"]
    logits = torch.randn(1, n_token, n_token, loss_mod.no_bins)

    out = loss_mod(
        pred_coordinate=batch["pred"],
        gt_coordinate_aug=batch["gt_aug"],
        sigma=sigma_big,
        coordinate_mask=batch["coord_mask"],
        rep_atom_mask=batch["rep_atom_mask"],
        distogram_logits=logits,
    )
    assert torch.isclose(out["lddt"], torch.zeros_like(out["lddt"]))
    assert torch.isclose(out["distogram"], torch.zeros_like(out["distogram"]))
    assert out["sigma_low_frac"].item() == 0.0


def test_training_noise_sampler_lognormal_stats():
    """EDM log-normal sampler: log(σ/sigma_data) should be ~N(p_mean, p_std)."""
    pytest.importorskip("protenix")
    from pxdesign_train.generator import TrainingNoiseSampler

    sampler = TrainingNoiseSampler(p_mean=-1.2, p_std=1.5, sigma_data=16.0)
    s = sampler(size=(10000,), device=torch.device("cpu"))
    log_normalized = torch.log(s / 16.0)
    assert abs(log_normalized.mean().item() - (-1.2)) < 0.1
    assert abs(log_normalized.std().item() - 1.5) < 0.1


def test_design_distogram_head_shape():
    pytest.importorskip("protenix")
    from pxdesign_train.heads import DesignDistogramHead

    head = DesignDistogramHead(c_z=128, no_bins=64)
    z = torch.randn(1, 16, 16, 128)
    out = head(z)
    assert out.shape == (1, 16, 16, 64)
    # Output should be symmetric in the two token axes (we symmetrise inside).
    assert torch.allclose(out, out.transpose(-2, -3), atol=1e-5)


def test_design_diffusion_distogram_head_shape():
    pytest.importorskip("protenix")
    from pxdesign_train.heads import DesignDiffusionDistogramHead

    head = DesignDiffusionDistogramHead(c_token=768, no_bins=64)
    tokens = torch.randn(1, 12, 768)
    out = head(tokens)
    assert out.shape == (1, 12, 12, 64)
    assert torch.allclose(out, out.transpose(-2, -3), atol=1e-5)


def test_design_residue_type_head_shape():
    from pxdesign_train.heads import DesignResidueTypeHead

    head = DesignResidueTypeHead(c_s=384, no_bins=20)
    tokens = torch.randn(2, 12, 384)
    out = head(tokens)
    assert out.shape == (2, 12, 20)


def test_masked_aa_loss_backprops_and_ignores_unmasked():
    from pxdesign_train.loss import PXDesignLoss

    batch = _fake_batch()
    loss_mod = PXDesignLoss(align_before_mse=False, weight_aa=1.0)
    n_token = batch["n_token"]
    aa_logits = torch.randn(1, n_token, 20, requires_grad=True)
    aa_clean = torch.full((1, n_token), -100, dtype=torch.long)
    aa_loss_mask = torch.zeros(1, n_token, dtype=torch.long)
    aa_clean[0, 0] = 7
    aa_clean[0, 3] = 11
    aa_loss_mask[0, 0] = 1
    aa_loss_mask[0, 3] = 1

    out = loss_mod(
        pred_coordinate=batch["pred"],
        gt_coordinate_aug=batch["gt_aug"],
        sigma=batch["sigma"],
        coordinate_mask=batch["coord_mask"],
        rep_atom_mask=batch["rep_atom_mask"],
        aa_logits=aa_logits,
        aa_clean=aa_clean,
        aa_loss_mask=aa_loss_mask,
    )

    assert torch.isfinite(out["loss"])
    assert out["aa_ce"].item() > 0
    assert 0.0 <= out["aa_acc"].item() <= 1.0
    assert torch.isclose(out["aa_mask_frac"], torch.tensor(2.0 / n_token))
    out["loss"].backward()
    assert aa_logits.grad is not None
    assert torch.isfinite(aa_logits.grad).all()


def test_masked_aa_loss_handles_no_valid_mask():
    from pxdesign_train.loss import PXDesignLoss

    batch = _fake_batch()
    loss_mod = PXDesignLoss(align_before_mse=False, weight_aa=1.0)
    n_token = batch["n_token"]
    aa_logits = torch.randn(1, n_token, 20, requires_grad=True)
    aa_clean = torch.full((1, n_token), -100, dtype=torch.long)
    aa_loss_mask = torch.zeros(1, n_token, dtype=torch.long)

    out = loss_mod(
        pred_coordinate=batch["pred"],
        gt_coordinate_aug=batch["gt_aug"],
        sigma=batch["sigma"],
        coordinate_mask=batch["coord_mask"],
        rep_atom_mask=batch["rep_atom_mask"],
        aa_logits=aa_logits,
        aa_clean=aa_clean,
        aa_loss_mask=aa_loss_mask,
    )

    assert torch.isfinite(out["loss"])
    assert out["aa_ce"].item() == 0.0
    assert out["aa_acc"].item() == 0.0
    assert out["aa_mask_frac"].item() == 0.0
    out["loss"].backward()
    assert aa_logits.grad is not None
    assert torch.all(aa_logits.grad == 0)
