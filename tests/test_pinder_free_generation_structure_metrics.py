import importlib.util
from pathlib import Path

import pytest
import torch

from pxdesign_train.loss import PXDesignLoss


def _eval_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/infer_aa_readouts_pinder.py"
    )
    spec = importlib.util.spec_from_file_location("pinder_aa_readout_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_lddt(pred, gt, coordinate_mask, ca_mask):
    mask = coordinate_mask.bool() & ca_mask.bool()
    pred_dist = torch.cdist(pred[0, mask], pred[0, mask])
    gt_dist = torch.cdist(gt[0, mask], gt[0, mask])
    return torch.isclose(pred_dist, gt_dist, atol=1e-5).float().mean()


def test_free_generation_structure_metrics_are_translation_invariant():
    module = _eval_module()
    native = torch.randn(12, 3)
    generated = native + torch.tensor([8.0, -3.0, 1.5])
    coordinate_mask = torch.ones(12, dtype=torch.bool)
    ca_mask = torch.zeros(12, dtype=torch.bool)
    ca_mask[[1, 5, 9]] = True
    backbone_mask = torch.ones(12, dtype=torch.bool)

    metrics = module._free_generation_structure_metrics(
        generated_coordinate=generated,
        native_coordinate=native,
        coordinate_mask=coordinate_mask,
        binder_ca_mask=ca_mask,
        binder_backbone_mask=backbone_mask,
        loss_fn=PXDesignLoss(weight_lddt=0.0, weight_disto=0.0),
        ca_lddt_score=_exact_lddt,
    )

    assert metrics["binder_ca_lddt"] == pytest.approx(1.0)
    assert metrics["binder_ca_rmsd"] < 1e-4
    assert metrics["binder_bb_rmsd"] < 1e-4
    assert metrics["binder_tm_score"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["n_binder_ca"] == 3
    assert metrics["n_binder_backbone_atoms"] == 12


def test_free_generation_structure_metrics_are_rotation_invariant():
    """Kabsch must use the row-vector rotation convention used by coordinates."""
    module = _eval_module()
    torch.manual_seed(7)
    native = torch.randn(12, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.det(rotation)
    generated = native @ rotation + torch.tensor([8.0, -3.0, 1.5])
    coordinate_mask = torch.ones(12, dtype=torch.bool)
    ca_mask = torch.zeros(12, dtype=torch.bool)
    ca_mask[[1, 5, 9]] = True
    backbone_mask = torch.ones(12, dtype=torch.bool)

    metrics = module._free_generation_structure_metrics(
        generated_coordinate=generated,
        native_coordinate=native,
        coordinate_mask=coordinate_mask,
        binder_ca_mask=ca_mask,
        binder_backbone_mask=backbone_mask,
        loss_fn=PXDesignLoss(weight_lddt=0.0, weight_disto=0.0),
        ca_lddt_score=_exact_lddt,
    )

    assert metrics["binder_ca_lddt"] == pytest.approx(1.0)
    assert metrics["binder_ca_rmsd"] < 1e-4
    assert metrics["binder_bb_rmsd"] < 1e-4
    assert metrics["binder_tm_score"] == pytest.approx(1.0, abs=1e-5)


def test_structure_summary_reports_per_complex_mean_and_median():
    module = _eval_module()
    rows = [
        {
            "binder_ca_lddt": 0.4,
            "binder_ca_rmsd": 3.0,
            "binder_bb_rmsd": 3.5,
            "binder_tm_score": 0.6,
            "n_binder_ca": 10,
            "n_binder_backbone_atoms": 40,
        },
        {
            "binder_ca_lddt": 0.8,
            "binder_ca_rmsd": 1.0,
            "binder_bb_rmsd": 1.5,
            "binder_tm_score": 0.9,
            "n_binder_ca": 20,
            "n_binder_backbone_atoms": 80,
        },
    ]

    summary = module._summarize_structure_metrics(rows, "/tmp/checkpoint.pt")

    assert summary["n_completed"] == 2
    assert summary["n_binder_ca_total"] == 30
    assert summary["n_binder_backbone_atoms_total"] == 120
    assert summary["binder_ca_lddt"] == pytest.approx(0.6)
    assert summary["binder_ca_rmsd"] == pytest.approx(2.0)
    assert summary["binder_ca_rmsd_median"] == pytest.approx(2.0)
    assert summary["binder_tm_score"] == pytest.approx(0.75)
    assert summary["pose_metrics_included"] is False
