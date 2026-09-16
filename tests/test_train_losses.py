"""The objectives must match the preprint, including the details easy to get wrong."""

import pytest
import torch

from pxf.train import losses as L


def test_total_is_an_unweighted_sum():
    # Appendix C.1: "L_total = L_MLM + L_diff", no relative weighting.
    a, b = torch.tensor(1.25), torch.tensor(2.5)
    assert float(L.total_loss(a, b)) == pytest.approx(3.75)
    assert float(L.total_loss(a, b, torch.tensor(0.5))) == pytest.approx(4.25)


def test_mlm_scores_the_masked_positions_not_the_kept_ones():
    """Upstream's mask is 1 where a residue was KEPT, so the loss uses 1 - mask."""
    logits = torch.zeros(1, 4, 21)
    logits[0, :, 0] = 10.0  # confidently predicts residue 0 everywhere
    aatype = torch.tensor([[0, 1, 0, 1]])
    seq_mask = torch.ones(1, 4)
    # Mask (0) the positions whose true type is 1 -> the loss should be large.
    keep_wrong = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    loss_wrong, stats = L.sequence_mlm_loss(logits, aatype, keep_wrong, seq_mask)
    assert int(stats["masked_residues"]) == 2
    # Mask the positions the model gets right -> the loss should be near zero.
    keep_right = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    loss_right, _ = L.sequence_mlm_loss(logits, aatype, keep_right, seq_mask)
    assert float(loss_right) < 1e-3 < float(loss_wrong)


def test_mlm_ignores_padding():
    logits = torch.randn(1, 4, 21)
    aatype = torch.zeros(1, 4, dtype=torch.long)
    keep = torch.zeros(1, 4)
    seq_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    _, stats = L.sequence_mlm_loss(logits, aatype, keep, seq_mask)
    assert int(stats["masked_residues"]) == 2


def test_no_scored_positions_gives_a_finite_zero():
    logits = torch.randn(1, 3, 21, requires_grad=True)
    loss, stats = L.sequence_mlm_loss(
        logits, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3), torch.ones(1, 3)
    )
    assert float(loss) == 0.0 and int(stats["masked_residues"]) == 0
    loss.backward()  # must stay differentiable rather than produce NaN
    assert torch.isfinite(logits.grad).all()


def test_diffusion_loss_applies_the_edm_weight_per_example():
    pred = torch.zeros(2, 1, 1, 3)
    target = torch.ones(2, 1, 1, 3)  # squared error 3 per atom
    mask = torch.ones(2, 1, 1)
    loss, stats = L.sidechain_diffusion_loss(pred, target, torch.tensor([1.0, 3.0]), mask)
    assert float(loss) == pytest.approx((3 * 1.0 + 3 * 3.0) / 2)
    assert float(stats["sidechain_mse_local"]) == pytest.approx(3.0)


def test_diffusion_loss_only_scores_masked_atoms():
    pred = torch.zeros(1, 1, 2, 3)
    target = torch.tensor([[[[1.0, 0, 0], [9.0, 0, 0]]]])
    mask = torch.tensor([[[1.0, 0.0]]])  # ignore the huge second atom
    loss, stats = L.sidechain_diffusion_loss(pred, target, torch.ones(1), mask)
    assert float(loss) == pytest.approx(1.0)
    assert int(stats["scored_atoms"]) == 1


def test_psce_bins_match_the_shipped_inference_centres():
    """The head's lower edges are linspace(min,max,n); centres are edge + step/2.

    Rounding to the nearest centre instead of flooring to the edge would put every
    label half a bin low, which is silent -- hence this test.
    """
    lower = torch.linspace(L.PSCE_MIN_BIN, L.PSCE_MAX_BIN, L.PSCE_NUM_BINS)
    centres = lower + (lower[1] - lower[0]) / 2
    for k, centre in enumerate(centres.tolist()):
        delta = torch.tensor([[[[centre, 0.0, 0.0]]]])
        index, _ = L.psce_bin_targets(torch.zeros_like(delta), delta)
        assert int(index) == k, f"centre {centre} landed in bin {int(index)}, want {k}"


def test_psce_bin_edges_and_saturation():
    step = L.PSCE_BIN_WIDTH
    cases = {
        0.0: 0,
        step - 1e-6: 0,
        step: 1,
        2 * step: 2,
        L.PSCE_MAX_BIN: L.PSCE_NUM_BINS - 1,
        99.0: L.PSCE_NUM_BINS - 1,
    }
    for error, expected in cases.items():
        delta = torch.tensor([[[[error, 0.0, 0.0]]]])
        index, _ = L.psce_bin_targets(torch.zeros_like(delta), delta)
        assert int(index) == expected, (error, int(index), expected)


def test_confidence_loss_rewards_the_right_bin():
    delta = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    target_bin, error = L.psce_bin_targets(torch.zeros_like(delta), delta)
    assert float(error) == pytest.approx(1.0)
    logits = torch.zeros(1, 1, 1, L.PSCE_NUM_BINS)
    logits[..., int(target_bin)] = 20.0
    good, stats = L.confidence_loss(
        logits, torch.zeros_like(delta), delta, torch.ones(1, 1, 1)
    )
    wrong = torch.zeros_like(logits)
    wrong[..., 0] = 20.0
    bad, _ = L.confidence_loss(wrong, torch.zeros_like(delta), delta, torch.ones(1, 1, 1))
    assert float(good) < 1e-6 < float(bad)
    assert float(stats["true_sidechain_error"]) == pytest.approx(1.0)


def test_confidence_loss_checks_the_head_width():
    delta = torch.zeros(1, 1, 1, 3)
    with pytest.raises(ValueError, match="bins"):
        L.confidence_loss(torch.zeros(1, 1, 1, 7), delta, delta, torch.ones(1, 1, 1))


def test_bin_spec_prefers_the_modules_own_config():
    class Cfg:
        class sce_bins:
            min_bin, max_bin, n_bins = 0.0, 8.0, 17

    class Module:
        cfg = Cfg

    assert L.psce_bin_spec(Module) == (0.0, 8.0, 17)
    assert L.psce_bin_spec(None) == (L.PSCE_MIN_BIN, L.PSCE_MAX_BIN, L.PSCE_NUM_BINS)


# ---- the reduction, per_residue vs per_atom --------------------------------
#
# FaMPNN released inference only, so neither reduction can be pinned to the
# original training loop. These tests pin what each one *means*, so the choice
# recorded in a checkpoint is interpretable.


def _two_residues():
    """Trp-like (7 atoms) and Ser-like (1 atom), every atom off by the same amount.

    Per-residue this is one bad residue out of two. Per-atom it is seven bad
    atoms out of eight, which is the whole difference between the reductions.
    """
    pred = torch.zeros(1, 2, 7, 3)
    target = torch.zeros(1, 2, 7, 3)
    target[0, 0, :, 0] = 1.0  # squared error 1 per atom, 7 atoms
    target[0, 1, 0, 0] = 1.0  # squared error 1 per atom, 1 atom
    mask = torch.zeros(1, 2, 7)
    mask[0, 0, :] = 1.0
    mask[0, 1, 0] = 1.0
    return pred, target, mask


def test_per_residue_weights_every_residue_once():
    pred, target, mask = _two_residues()
    loss, stats = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_residue"
    )
    # Each residue's own mean is 1.0, so the mean over residues is 1.0 -- the
    # 7-atom residue does not count seven times the 1-atom one.
    assert float(loss) == pytest.approx(1.0)
    assert int(stats["scored_residues"]) == 2
    assert int(stats["scored_atoms"]) == 8
    assert stats["reduction"] == "per_residue"


def test_per_atom_lets_big_sidechains_dominate():
    pred, target, mask = _two_residues()
    # Make only the large residue wrong, so the two reductions must disagree.
    target[0, 1, 0, 0] = 0.0
    per_atom, _ = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_atom"
    )
    per_residue, _ = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_residue"
    )
    assert float(per_atom) == pytest.approx(7 / 8)  # 7 bad atoms of 8
    assert float(per_residue) == pytest.approx(1 / 2)  # 1 bad residue of 2
    assert float(per_atom) > float(per_residue)


def test_per_residue_is_the_default():
    pred, target, mask = _two_residues()
    target[0, 1, 0, 0] = 0.0
    default, stats = L.sidechain_diffusion_loss(pred, target, torch.ones(1), mask)
    explicit, _ = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_residue"
    )
    assert float(default) == pytest.approx(float(explicit))
    assert stats["reduction"] == L.DEFAULT_SIDECHAIN_REDUCTION == "per_residue"


def test_per_residue_still_applies_the_edm_weight_per_example():
    pred = torch.zeros(2, 1, 2, 3)
    target = torch.zeros(2, 1, 2, 3)
    target[:, :, :, 0] = 1.0  # squared error 1 per atom, so L_i = 1 per residue
    mask = torch.ones(2, 1, 2)
    loss, _ = L.sidechain_diffusion_loss(
        pred, target, torch.tensor([1.0, 3.0]), mask, reduction="per_residue"
    )
    assert float(loss) == pytest.approx((1.0 + 3.0) / 2)


def test_residues_with_nothing_to_score_do_not_dilute_the_mean():
    """Glycine, an unresolved side chain, or a target the interpolant masked out.

    Averaging over all residues instead of the scored ones would shrink the loss
    in proportion to how many residues happen to be glycine.
    """
    pred = torch.zeros(1, 3, 2, 3)
    target = torch.zeros(1, 3, 2, 3)
    target[0, 0, :, 0] = 1.0
    mask = torch.zeros(1, 3, 2)
    mask[0, 0, :] = 1.0  # only the first residue is scored
    loss, stats = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_residue"
    )
    assert float(loss) == pytest.approx(1.0)  # not 1/3
    assert int(stats["scored_residues"]) == 1


def test_nothing_scored_gives_a_differentiable_zero_under_both_reductions():
    for reduction in L.SIDECHAIN_REDUCTIONS:
        pred = torch.zeros(1, 2, 2, 3, requires_grad=True)
        target = torch.ones(1, 2, 2, 3)
        loss, stats = L.sidechain_diffusion_loss(
            pred, target, torch.ones(1), torch.zeros(1, 2, 2), reduction=reduction
        )
        assert float(loss) == 0.0
        assert int(stats["scored_residues"]) == 0
        loss.backward()
        assert torch.isfinite(pred.grad).all()


def test_the_unweighted_diagnostic_is_per_atom_under_both_reductions():
    """``sidechain_mse_local`` must stay comparable across runs and reductions."""
    pred, target, mask = _two_residues()
    target[0, 1, 0, 0] = 0.0
    values = {
        reduction: float(
            L.sidechain_diffusion_loss(
                pred, target, torch.tensor([5.0]), mask, reduction=reduction
            )[1]["sidechain_mse_local"]
        )
        for reduction in L.SIDECHAIN_REDUCTIONS
    }
    assert values["per_residue"] == pytest.approx(values["per_atom"])
    assert values["per_atom"] == pytest.approx(7 / 8)  # unweighted by the EDM term


def test_an_unknown_reduction_is_refused():
    pred, target, mask = _two_residues()
    with pytest.raises(ValueError, match="Unknown reduction"):
        L.sidechain_diffusion_loss(pred, target, torch.ones(1), mask, reduction="per_chain")
