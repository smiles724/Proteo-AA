"""The objectives, against the original training code's ``SDLoss``.

Each test here pins a place where an implementation written from the preprint
alone diverges from the code the released weights were trained by.
"""

import math

import pytest
import torch

from pxf.train import losses as L


def test_total_is_the_weighted_sum():
    # All three weights are 1.0 in the released config, which is what
    # "we did not experiment with relative weightings" means.
    a, b = torch.tensor(1.25), torch.tensor(2.5)
    assert float(L.total_loss(a, b)) == pytest.approx(3.75)
    assert float(L.total_loss(a, b, torch.tensor(0.5))) == pytest.approx(4.25)
    weighted = L.LossSettings(weight_seq=2.0, weight_confidence=0.0)
    assert float(
        L.total_loss(a, b, torch.tensor(0.5), settings=weighted)
    ) == pytest.approx(2 * 1.25 + 2.5)


def test_a_non_finite_term_is_dropped_rather_than_poisoning_the_sum():
    """Upstream skips a NaN term and logs it; one bad example must not end a run."""
    good = torch.tensor(1.0, requires_grad=True)
    bad = torch.tensor(float("nan"), requires_grad=True)
    total = L.total_loss(bad, good)
    assert float(total) == pytest.approx(1.0)
    total.backward()
    assert good.grad is not None


# ---- L_seq -----------------------------------------------------------------


def test_mlm_scores_the_masked_positions_not_the_kept_ones():
    """Upstream's mask is 1 where a residue was KEPT, so the loss uses 1 - mask."""
    logits = torch.zeros(1, 4, 21)
    logits[0, :, 0] = 10.0  # confidently predicts residue 0 everywhere
    aatype = torch.tensor([[0, 1, 0, 1]])
    seq_mask = torch.ones(1, 4)
    keep_wrong = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # mask the positions it gets wrong
    loss_wrong, stats = L.sequence_mlm_loss(logits, aatype, keep_wrong, seq_mask)
    assert int(stats["masked_residues"]) == 2
    keep_right = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    loss_right, _ = L.sequence_mlm_loss(logits, aatype, keep_right, seq_mask)
    assert float(loss_right) < float(loss_wrong)


def test_mlm_is_normalized_by_the_crop_length_not_the_masked_count():
    """``seq_loss.per_token_avg: false``: a sum over masked tokens, divided by L.

    This is the detail the preprint does not give and a plausible implementation
    gets wrong. Dividing by the masked count instead makes the term independent
    of how much the interpolant hid, which rescales every step by a random factor
    and changes its balance against the diffusion term.
    """
    length = 8
    logits = torch.zeros(1, length, 21)
    aatype = torch.zeros(1, length, dtype=torch.long)
    seq_mask = torch.ones(1, length)
    per_token = -math.log(1 / 21)  # uniform logits, ignoring label smoothing

    two_masked = torch.ones(1, length)
    two_masked[0, :2] = 0.0
    four_masked = torch.ones(1, length)
    four_masked[0, :4] = 0.0

    two, _ = L.sequence_mlm_loss(logits, aatype, two_masked, seq_mask)
    four, _ = L.sequence_mlm_loss(logits, aatype, four_masked, seq_mask)
    # Twice the masked tokens, twice the loss -- not the same loss.
    assert float(four) == pytest.approx(2 * float(two))
    assert float(two) == pytest.approx(2 * per_token / length, rel=1e-3)

    # The other normalization is available, and is the one that would be flat.
    per_token_avg = L.LossSettings(seq_per_token_avg=True)
    two_avg, _ = L.sequence_mlm_loss(
        logits, aatype, two_masked, seq_mask, settings=per_token_avg
    )
    four_avg, _ = L.sequence_mlm_loss(
        logits, aatype, four_masked, seq_mask, settings=per_token_avg
    )
    assert float(two_avg) == pytest.approx(float(four_avg))


def test_mlm_excludes_unknown_residues():
    """Training on an ``X`` label teaches the model to predict its own mask token."""
    logits = torch.zeros(1, 3, 21)
    aatype = torch.tensor([[0, 20, 1]])  # 20 == X
    seq_mask = torch.ones(1, 3)
    masked = torch.zeros(1, 3)
    unk = (aatype == 20).float()
    _, with_unk = L.sequence_mlm_loss(logits, aatype, masked, seq_mask)
    _, without = L.sequence_mlm_loss(
        logits, aatype, masked, seq_mask, seq_unk_mask=unk
    )
    assert int(with_unk["masked_residues"]) == 3
    assert int(without["masked_residues"]) == 2


def test_mlm_ignores_padding():
    logits = torch.randn(1, 4, 21)
    aatype = torch.zeros(1, 4, dtype=torch.long)
    keep = torch.zeros(1, 4)
    seq_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    _, stats = L.sequence_mlm_loss(logits, aatype, keep, seq_mask)
    assert int(stats["masked_residues"]) == 2


def test_mlm_smooths_labels():
    """Label smoothing 0.1 spread over 21 classes, added then renormalized."""
    logits = torch.zeros(1, 1, 21)
    logits[0, 0, 0] = 30.0  # an almost one-hot prediction of the true class
    aatype = torch.zeros(1, 1, dtype=torch.long)
    seq_mask = torch.ones(1, 1)
    masked = torch.zeros(1, 1)
    smoothed, _ = L.sequence_mlm_loss(logits, aatype, masked, seq_mask)
    hard, _ = L.sequence_mlm_loss(
        logits, aatype, masked, seq_mask, settings=L.LossSettings(label_smoothing=0.0)
    )
    assert float(hard) == pytest.approx(0.0, abs=1e-6)
    assert float(smoothed) > 0.1, "an overconfident prediction must still be penalized"


def test_mlm_reports_accuracy_over_the_scored_set():
    logits = torch.zeros(1, 4, 21)
    logits[0, :, 0] = 10.0
    aatype = torch.tensor([[0, 1, 0, 1]])
    keep = torch.zeros(1, 4)
    _, stats = L.sequence_mlm_loss(logits, aatype, keep, torch.ones(1, 4))
    assert float(stats["sequence_accuracy"]) == pytest.approx(0.5)


def test_no_scored_positions_gives_a_finite_zero():
    logits = torch.randn(1, 3, 21, requires_grad=True)
    loss, stats = L.sequence_mlm_loss(
        logits, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3), torch.ones(1, 3)
    )
    assert float(loss) == 0.0 and int(stats["masked_residues"]) == 0
    loss.backward()  # must stay differentiable rather than produce NaN
    assert torch.isfinite(logits.grad).all()


# ---- L_scn -----------------------------------------------------------------


def test_diffusion_loss_averages_per_coordinate_then_weights_per_example():
    """``masked_mse(per_token_avg=True)`` divides by components, not atoms.

    The EDM weight multiplies the example's already-reduced error, so an example
    with more resolved atoms does not carry more of the batch.
    """
    pred = torch.zeros(2, 1, 1, 3)
    target = torch.ones(2, 1, 1, 3)  # squared error 3 per atom, 1.0 per component
    mask = torch.ones(2, 1, 1)
    loss, stats = L.sidechain_diffusion_loss(pred, target, torch.tensor([1.0, 3.0]), mask)
    assert float(loss) == pytest.approx((1.0 * 1.0 + 1.0 * 3.0) / 2)
    assert float(stats["sidechain_mse_local"]) == pytest.approx(1.0)
    assert int(stats["scored_atoms"]) == 2


def test_diffusion_loss_takes_an_atom_or_a_coordinate_mask():
    pred = torch.zeros(1, 2, 3, 3)
    target = torch.ones(1, 2, 3, 3)
    atoms = torch.ones(1, 2, 3)
    components = atoms.unsqueeze(-1).expand_as(pred)
    by_atom, _ = L.sidechain_diffusion_loss(pred, target, torch.ones(1), atoms)
    by_component, _ = L.sidechain_diffusion_loss(pred, target, torch.ones(1), components)
    assert float(by_atom) == pytest.approx(float(by_component))


def test_diffusion_loss_only_scores_masked_atoms():
    pred = torch.zeros(1, 1, 2, 3)
    target = torch.tensor([[[[3.0, 0, 0], [9.0, 0, 0]]]])
    mask = torch.tensor([[[1.0, 0.0]]])  # ignore the huge second atom
    loss, stats = L.sidechain_diffusion_loss(pred, target, torch.ones(1), mask)
    assert float(loss) == pytest.approx(9.0 / 3)  # one atom, three components
    assert int(stats["scored_atoms"]) == 1


def test_the_fixed_size_reduction_divides_by_the_constant_example_size():
    """``per_token_avg: false``: a crop with few resolved side chains scores lower."""
    pred = torch.zeros(1, 4, 2, 3)
    target = torch.ones(1, 4, 2, 3)
    mask = torch.zeros(1, 4, 2)
    mask[0, 0] = 1.0  # one residue of four is resolved
    per_token, _ = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_token"
    )
    fixed, stats = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="fixed_size"
    )
    assert float(per_token) == pytest.approx(1.0)
    assert float(fixed) == pytest.approx(6 / (4 * 2 * 3))
    # The diagnostic stays per-component under both, so curves stay comparable.
    assert float(stats["sidechain_mse_local"]) == pytest.approx(1.0)


def test_per_token_is_the_default():
    assert L.DEFAULT_SIDECHAIN_REDUCTION == "per_token"
    pred, target = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)
    mask = torch.ones(1, 2, 2)
    default, stats = L.sidechain_diffusion_loss(pred, target, torch.ones(1), mask)
    explicit, _ = L.sidechain_diffusion_loss(
        pred, target, torch.ones(1), mask, reduction="per_token"
    )
    assert float(default) == pytest.approx(float(explicit))
    assert stats["reduction"] == "per_token"


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


def test_an_unknown_reduction_is_refused():
    pred, target = torch.zeros(1, 1, 1, 3), torch.ones(1, 1, 1, 3)
    with pytest.raises(ValueError, match="Unknown reduction"):
        L.sidechain_diffusion_loss(
            pred, target, torch.ones(1), torch.ones(1, 1, 1), reduction="per_chain"
        )
    with pytest.raises(ValueError, match="Unknown reduction"):
        L.LossSettings(sidechain_reduction="per_residue")


# ---- L_psce ----------------------------------------------------------------


def test_psce_bins_match_the_shipped_inference_centres():
    """The head's lower edges are linspace(min,max,n); centres are edge + step/2.

    Rounding to the nearest centre instead of flooring to the edge would put every
    label half a bin low, which is silent -- hence this test.
    """
    lower = torch.linspace(L.PSCE_MIN_BIN, L.PSCE_MAX_BIN, L.PSCE_NUM_BINS)
    centres = lower + (lower[1] - lower[0]) / 2
    for k, centre in enumerate(centres.tolist()):
        delta = torch.tensor([[[[centre, 0.0, 0.0]]]])
        binned, _ = L.psce_bin_targets(torch.zeros_like(delta), delta)
        assert int(binned.argmax(-1)) == k, f"centre {centre} landed in the wrong bin"


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
        binned, value = L.psce_bin_targets(torch.zeros_like(delta), delta)
        assert float(binned.sum()) == 1.0, error
        assert int(binned.argmax(-1)) == expected, (error, expected)
        assert float(value) == pytest.approx(error)


def test_confidence_loss_rewards_the_right_bin():
    delta = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    binned, error = L.psce_bin_targets(torch.zeros_like(delta), delta)
    assert float(error) == pytest.approx(1.0)
    logits = torch.zeros(1, 1, 1, L.PSCE_NUM_BINS)
    logits[..., int(binned.argmax(-1))] = 20.0
    good, stats = L.confidence_loss(
        logits, torch.zeros_like(delta), delta, torch.ones(1, 1, 1)
    )
    wrong = torch.zeros_like(logits)
    wrong[..., 0] = 20.0
    bad, _ = L.confidence_loss(wrong, torch.zeros_like(delta), delta, torch.ones(1, 1, 1))
    assert float(good) < 1e-6 < float(bad)
    assert float(stats["true_sidechain_error"]) == pytest.approx(1.0)


def test_confidence_loss_averages_within_an_example_then_over_the_batch():
    """Reduced over ``[L, 33]`` per example, so a long example is not worth more."""
    logits = torch.zeros(2, 4, 1, L.PSCE_NUM_BINS)
    pred = torch.zeros(2, 4, 1, 3)
    target = torch.zeros(2, 4, 1, 3)
    mask = torch.zeros(2, 4, 1)
    mask[0, :4] = 1.0  # four scored atoms in the first example
    mask[1, :1] = 1.0  # one in the second
    loss, stats = L.confidence_loss(logits, pred, target, mask)
    # Uniform logits: every atom costs log(n_bins) regardless of the counts.
    assert float(loss) == pytest.approx(math.log(L.PSCE_NUM_BINS), rel=1e-5)
    assert int(stats["confidence_atoms"]) == 5


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
