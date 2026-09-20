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
    assert float(stats["backbone_rmsd_angstrom"]) == pytest.approx(3**0.5)


def test_atom_mask_restricts_the_score():
    target, predicted = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    mask = torch.zeros(1, 8)
    mask[0, :4] = 1
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
        "sidechain",
        "backbone",
        "sidechain",
        "backbone",
        "sidechain",
        "backbone",
    ]


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

    loss = L.backbone_feedback_loss(
        Cycle(), torch.zeros(1, 4, 3), sigma=1.0, require_feedback=False
    )
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


# ---- L_SC is the diffusion term only ---------------------------------------
#
# Phase 1 asks one question: does PXDesign's a_token carry information that
# improves side-chain packing? The sequence is held fixed throughout, so an MLM
# term would be scoring a prediction of something already given, and a change in
# the loss could no longer be read as a change in packing quality. These tests
# pin the objective, because "we called diffusion_loss rather than
# training_forward" is a decision, not an implementation detail.


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    net = SeqDenoiser(bundle["model_cfg"])
    net.load_state_dict(bundle["state_dict"], strict=True)
    net.train()
    net.requires_grad_(False)
    return net


@pytest.fixture(scope="module")
def sidechain_batch():
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    paths = [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")]
    dataset = StructureCropDataset(paths, crop_size=48, noise=0.0, seed=0)
    return collate([dataset[0]])


def _features(model, batch):
    """The encoder feature dict the coupling cycle would hand to L_SC.

    Backbone only, with every side chain masked -- what the cycle actually
    encodes, and what ``sidechain_pack`` encodes at inference.
    """
    from pxf.couple import fampnn_iface as iface

    _, _, features = iface.encode(
        model,
        batch["x"],
        batch["aatype"],
        seq_mask=batch["seq_mask"],
        missing_atom_mask=batch["missing_atom_mask"],
        residue_index=batch["residue_index"],
        chain_index=batch["chain_index"],
    )
    return features


def _full_objective(model, batch):
    from pxf.train import step as train_step

    return train_step.training_forward(model, batch, train_confidence=False)


def test_l_sc_equals_the_diffusion_term_alone(fampnn, sidechain_batch):
    from pxf.train import step as train_step

    features = _features(fampnn, sidechain_batch)
    torch.manual_seed(0)
    coupled = L.sidechain_coupling_loss(fampnn, sidechain_batch, features, delta_h=None)
    torch.manual_seed(0)
    diffusion, _ = train_step.diffusion_loss(fampnn, sidechain_batch, features)
    assert float(coupled.total) == pytest.approx(float(diffusion), rel=1e-6)


def test_l_sc_carries_no_mlm_or_confidence_term(fampnn, sidechain_batch):
    """A cheap structural check: the stats of the two objectives differ."""
    features = _features(fampnn, sidechain_batch)
    torch.manual_seed(0)
    coupled = L.sidechain_coupling_loss(fampnn, sidechain_batch, features, delta_h=None)
    assert "masked_residues" not in coupled.stats  # L_MLM's diagnostic
    assert "confidence_atoms" not in coupled.stats  # the psCE head's

    torch.manual_seed(0)
    full = _full_objective(fampnn, sidechain_batch)
    # The full objective is strictly larger, so they are not interchangeable.
    assert float(full.total) > float(coupled.total)
    assert float(full.mlm) > 0


def test_l_sc_supervises_every_atom_because_packing_hides_them_all(fampnn, sidechain_batch):
    """No ``scn_mlm_mask``, which is both FaMPNN's own objective and this one.

    The cycle packs from scratch, as inference does, so every supervisable atom
    is a target -- and the original training code supervises every one of them
    too, whether or not its side chain was visible to the encoder.
    """
    from pxf.train import step as train_step

    features = _features(fampnn, sidechain_batch)
    _, unrestricted = train_step.sidechain_targets(fampnn, sidechain_batch)
    clones = int(fampnn.denoiser.scn_diffusion_module.cfg.training_batch_size_mult)
    torch.manual_seed(0)
    coupled = L.sidechain_coupling_loss(fampnn, sidechain_batch, features, delta_h=None)
    assert int(coupled.stats["scored_atoms"]) == clones * int(unrestricted.sum())


def test_l_sc_honours_the_reduction_setting(fampnn, sidechain_batch):
    """Both reductions reach L_SC, and the active one is reported.

    On this batch they also *coincide*, which is not a bug: the crop is fully
    resolved and unpadded, so "divide by the supervised components" and "divide
    by the constant example size" are the same division. The padded batch below
    is where they come apart.
    """
    features = _features(fampnn, sidechain_batch)
    losses = {}
    for reduction in ("per_token", "fixed_size"):
        torch.manual_seed(0)
        out = L.sidechain_coupling_loss(
            fampnn, sidechain_batch, features, delta_h=None, reduction=reduction
        )
        assert out.stats["reduction"] == reduction
        assert out.scalars()["reduction"] == reduction  # survives scalars()
        losses[reduction] = float(out.total)
    assert losses["per_token"] == pytest.approx(losses["fixed_size"])


def test_the_two_reductions_differ_once_anything_is_masked_out(fampnn):
    """Padding is enough: fixed_size keeps it in the denominator, per_token does not."""
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    path = str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")  # 95 residues
    padded = collate([StructureCropDataset([path], crop_size=128, seed=0)[0]])
    features = _features(fampnn, padded)
    losses = {}
    for reduction in ("per_token", "fixed_size"):
        torch.manual_seed(0)
        out = L.sidechain_coupling_loss(
            fampnn, padded, features, delta_h=None, reduction=reduction
        )
        losses[reduction] = float(out.total)
    # 95 of 128 positions are real, so fixed_size scores about 95/128 of per_token.
    assert losses["fixed_size"] == pytest.approx(losses["per_token"] * 95 / 128, rel=0.01)
