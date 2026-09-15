"""Staged objectives: weighting, masking, alternation, and the feedback guard."""
import pytest
import torch

from pxf.couple import losses as L


def test_exact_prediction_costs_nothing():
    zeros = torch.zeros(1, 8, 3)
    assert float(L.backbone_denoising_loss(zeros, zeros, sigma=1.0).total) == 0.0


def test_edm_weighting_falls_with_sigma():
    """1/c_out^2 means low-noise errors are penalized far harder."""
    target, predicted = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    low = float(L.backbone_denoising_loss(predicted, target, sigma=1.0).total)
    high = float(L.backbone_denoising_loss(predicted, target, sigma=10.0).total)
    assert low > high > 0


def test_reported_rmsd_is_unweighted():
    target, predicted = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    stats = L.backbone_denoising_loss(predicted, target, sigma=3.0).stats
    assert float(stats["backbone_rmsd_angstrom"]) == pytest.approx(3 ** 0.5)


def test_atom_mask_restricts_the_score():
    target, predicted = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    mask = torch.zeros(1, 8); mask[0, :4] = 1
    stats = L.backbone_denoising_loss(predicted, target, sigma=1.0, atom_mask=mask).stats
    assert float(stats["scored_atoms"]) == 4.0


def test_no_superposition_before_scoring():
    """A rigid translation must cost something: the denoiser predicts in-frame."""
    target = torch.randn(1, 12, 3)
    shifted = target + torch.tensor([5.0, 0.0, 0.0])
    assert float(L.backbone_denoising_loss(shifted, target, sigma=1.0).total) > 0


def test_phase_one_and_two_are_single_objective():
    assert [L.loss_kind_for("bb_to_sc", s) for s in range(3)] == ["sidechain"] * 3
    assert [L.loss_kind_for("sc_to_bb", s) for s in range(3)] == ["backbone"] * 3


def test_joint_alternates_deterministically_by_step():
    """Reproducible from the step number alone, not sampled."""
    assert [L.loss_kind_for("joint", s) for s in range(6)] == [
        "sidechain", "backbone", "sidechain", "backbone", "sidechain", "backbone"]


def test_frozen_and_unknown_phases_are_refused():
    with pytest.raises(ValueError, match="trains nothing"):
        L.loss_kind_for("frozen", 0)
    with pytest.raises(ValueError, match="Unknown phase"):
        L.loss_kind_for("phase9", 0)


def test_feedback_loss_refuses_the_uncorrected_backbone():
    """Scoring bb0 would leave A_SB untrained while still drawing a loss curve."""
    class Cycle:
        bb1_flat = None
        bb0_flat = torch.ones(1, 4, 3)
        delta_a = None
    with pytest.raises(ValueError, match="leave A_SB untrained"):
        L.backbone_feedback_loss(Cycle(), torch.zeros(1, 4, 3), sigma=1.0)


def test_feedback_loss_can_be_told_to_score_bb0_deliberately():
    class Cycle:
        bb1_flat = None
        bb0_flat = torch.ones(1, 4, 3)
        delta_a = None
    loss = L.backbone_feedback_loss(Cycle(), torch.zeros(1, 4, 3), sigma=1.0,
                                    require_feedback=False)
    assert float(loss.stats["used_correction"]) == 0.0


def test_feedback_loss_reports_the_correction_it_used():
    class Cycle:
        bb1_flat = torch.zeros(1, 4, 3)
        bb0_flat = torch.ones(1, 4, 3)
        delta_a = torch.full((1, 4, 8), 0.5)
    loss = L.backbone_feedback_loss(Cycle(), torch.zeros(1, 4, 3), sigma=1.0)
    assert float(loss.total) == 0.0
    assert float(loss.stats["used_correction"]) == 1.0
    assert float(loss.stats["delta_a_norm"]) > 0


def test_scalars_are_json_friendly():
    loss = L.backbone_denoising_loss(torch.ones(1, 2, 3), torch.zeros(1, 2, 3), sigma=1.0)
    scalars = loss.scalars()
    assert scalars["loss_kind"] == "backbone"
    assert isinstance(scalars["loss"], float)
