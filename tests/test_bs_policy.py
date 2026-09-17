"""The gate and the residual sources.

Two properties carry the experiment. At gate zero the coupled arm must be the
*bypass* arm, not an approximation of it, or the clean-packing acceptance
criterion is testing float noise. And at gate one the original 20k adapter must
be reproduced exactly, or the gated arm is not comparable to the ungated one.
"""

import math

import pytest
import torch

from pxf.couple import bs_policy as bp

DIAGNOSTIC_SIGMAS = (0.010, 0.082, 0.429, 1.642, 4.881)


# --- the gate ---------------------------------------------------------------


def test_the_gate_is_exactly_zero_at_and_below_sigma_off():
    gate = bp.GATES["B"]
    assert gate(0.001) == 0.0
    assert gate(0.010) == 0.0
    assert gate(0.082) == 0.0  # sigma_off itself, not just below it


def test_the_gate_is_exactly_one_at_and_above_sigma_on():
    gate = bp.GATES["B"]
    assert gate(0.429) == 1.0
    assert gate(4.881) == 1.0
    assert gate(160.0) == 1.0


def test_the_gate_is_monotone_and_smooth_between_the_knots():
    gate = bp.GATES["C"]
    grid = [0.082 * (1.642 / 0.082) ** (i / 20) for i in range(21)]
    values = [gate(s) for s in grid]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)
    # Smoothstep has zero derivative at both ends, so the first and last steps
    # are much smaller than the middle one -- that is the point of using it.
    assert values[1] - values[0] < values[11] - values[10]
    assert values[-1] - values[-2] < values[11] - values[10]


def test_the_three_candidates_protect_progressively_more():
    a, b, c = bp.GATES["A"], bp.GATES["B"], bp.GATES["C"]
    # At 0.082 A is already partly on, B and C are still off.
    assert a(0.082) > 0.0
    assert b(0.082) == 0.0 and c(0.082) == 0.0
    # At 0.429 A and B are fully on, C is not yet.
    assert a(0.429) == 1.0 and b(0.429) == 1.0
    assert 0.0 < c(0.429) < 1.0


def test_the_gate_matches_on_tensors_and_floats():
    gate = bp.GATES["C"]
    sigmas = torch.tensor(DIAGNOSTIC_SIGMAS, dtype=torch.float64)
    batched = gate(sigmas)
    for value, sigma in zip(batched.tolist(), DIAGNOSTIC_SIGMAS):
        assert value == pytest.approx(gate(sigma), abs=1e-12)


def test_the_ungated_setting_is_one_everywhere_the_schedule_visits():
    for sigma in DIAGNOSTIC_SIGMAS:
        assert bp.GATES["one"](sigma) == 1.0


def test_a_gate_with_inverted_knots_is_refused():
    with pytest.raises(ValueError, match="sigma_off < sigma_on"):
        bp.Gate("bad", 1.0, 0.1)
    with pytest.raises(ValueError, match="sigma_off < sigma_on"):
        bp.Gate("bad", 0.0, 1.0)


def test_gate_lookup_by_name():
    assert bp.gate_by_name("B") is bp.GATES["B"]
    assert bp.gate_by_name(None) is None
    assert bp.gate_by_name("off") is None
    with pytest.raises(ValueError, match="unknown gate"):
        bp.gate_by_name("Z")


# --- the mean residual ------------------------------------------------------


def make_mean():
    return bp.MeanResidual(
        [0.082, 1.642],
        [torch.full((4,), 1.0), torch.full((4,), 3.0)],
        provenance={"n_proteins": 7},
    )


def test_the_mean_returns_its_knots_exactly():
    mean = make_mean()
    assert torch.allclose(mean.at(0.082), torch.full((4,), 1.0))
    assert torch.allclose(mean.at(1.642), torch.full((4,), 3.0))


def test_the_mean_interpolates_in_log_sigma_not_linear_sigma():
    mean = make_mean()
    midpoint = math.sqrt(0.082 * 1.642)  # halfway in log space
    assert torch.allclose(mean.at(midpoint), torch.full((4,), 2.0), atol=1e-5)
    # The linear-space midpoint must land somewhere else, or the test is vacuous.
    assert not torch.allclose(
        mean.at((0.082 + 1.642) / 2), torch.full((4,), 2.0), atol=1e-3
    )


def test_the_mean_clamps_rather_than_extrapolating():
    mean = make_mean()
    assert torch.allclose(mean.at(1e-6), torch.full((4,), 1.0))
    assert torch.allclose(mean.at(1000.0), torch.full((4,), 3.0))


def test_knots_are_sorted_on_construction():
    mean = bp.MeanResidual([4.0, 0.1], [torch.ones(2) * 9, torch.ones(2)])
    assert mean.sigmas == [0.1, 4.0]
    assert torch.allclose(mean.at(0.1), torch.ones(2))


def test_a_mismatched_knot_count_is_refused():
    with pytest.raises(ValueError, match="vectors"):
        bp.MeanResidual([0.1, 1.0], [torch.ones(2)])
    with pytest.raises(ValueError, match="at least one"):
        bp.MeanResidual([], [])


# --- residual sources -------------------------------------------------------


class FakeAdapters:
    """delta_h = a_token.sum(-1, keepdim) broadcast, plus a bias, so A(0) != 0."""

    def __init__(self, width=4, bias=0.25):
        self.width, self.bias = width, bias

    def delta_h(self, a_token, sigma):
        scale = float(sigma.reshape(-1)[0]) if torch.is_tensor(sigma) else float(sigma)
        pooled = a_token.mean(-1, keepdim=True).expand(*a_token.shape[:-1], self.width)
        return pooled * scale + self.bias


def test_bypass_returns_none_rather_than_a_zero_tensor():
    """The uncoupled arm must take the same code path, not an equivalent one."""
    assert bp.residual(FakeAdapters(), "none", sigma=1.0) is None


def test_a_fully_closed_gate_returns_the_bypass_itself():
    adapters, a_token = FakeAdapters(), torch.randn(1, 5, 3)
    out = bp.residual(adapters, "matched", a_token=a_token, sigma=0.010, gate=bp.GATES["B"])
    assert out is None, "gate zero must bypass, not scale to approximately zero"


def test_gate_one_reproduces_the_ungated_residual_exactly():
    adapters, a_token = FakeAdapters(), torch.randn(1, 5, 3)
    plain = bp.residual(adapters, "matched", a_token=a_token, sigma=4.881)
    gated = bp.residual(
        adapters, "matched", a_token=a_token, sigma=4.881, gate=bp.GATES["B"]
    )
    assert torch.equal(plain, gated)


def test_a_partly_open_gate_scales_the_residual():
    adapters, a_token = FakeAdapters(), torch.randn(1, 5, 3)
    plain = bp.residual(adapters, "matched", a_token=a_token, sigma=0.429)
    gated = bp.residual(
        adapters, "matched", a_token=a_token, sigma=0.429, gate=bp.GATES["C"]
    )
    scale = bp.GATES["C"](0.429)
    assert 0.0 < scale < 1.0
    assert torch.allclose(gated, plain * scale)


def test_zero_input_is_not_the_same_as_bypass():
    """Biases and the sigma embedding make A(0, sigma) non-zero; that is the arm."""
    adapters, a_token = FakeAdapters(), torch.randn(1, 5, 3)
    out = bp.residual(adapters, "zero_input", a_token=a_token, sigma=1.642)
    assert out is not None
    assert not torch.allclose(out, torch.zeros_like(out))
    matched = bp.residual(adapters, "matched", a_token=a_token, sigma=1.642)
    assert not torch.allclose(out, matched)


def test_zero_input_ignores_the_token_content_but_not_its_shape():
    adapters = FakeAdapters()
    one = bp.residual(adapters, "zero_input", a_token=torch.randn(1, 6, 3), sigma=1.0)
    two = bp.residual(adapters, "zero_input", a_token=torch.randn(1, 6, 3) * 99, sigma=1.0)
    assert torch.equal(one, two)
    assert one.shape[1] == 6


def test_the_mean_arm_broadcasts_one_vector_over_every_residue():
    out = bp.residual(FakeAdapters(), "mean", sigma=1.642, length=7, mean=make_mean())
    assert out.shape == (1, 7, 4)
    assert torch.allclose(out[0, 0], out[0, 6])
    assert torch.allclose(out[0, 0], torch.full((4,), 3.0))


def test_the_mean_arm_is_gated_like_any_other():
    mean = make_mean()
    assert (
        bp.residual(
            FakeAdapters(), "mean", sigma=0.010, length=3, mean=mean, gate=bp.GATES["B"]
        )
        is None
    )


def test_each_source_reports_what_it_is_missing():
    adapters = FakeAdapters()
    with pytest.raises(ValueError, match="needs a_token"):
        bp.residual(adapters, "matched", sigma=1.0)
    with pytest.raises(ValueError, match="needs a MeanResidual"):
        bp.residual(adapters, "mean", sigma=1.0, length=3)
    with pytest.raises(ValueError, match="residue count"):
        bp.residual(adapters, "mean", sigma=1.0, mean=make_mean())
    with pytest.raises(ValueError, match="unknown residual source"):
        bp.residual(adapters, "telepathy", sigma=1.0)
