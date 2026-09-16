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
