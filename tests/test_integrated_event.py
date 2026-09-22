"""The shared event module's contracts, at the level that needs no donors.

The parts requiring real PXDesign/FaMPNN weights are exercised by the preflight
on the cluster, not here. What IS testable on CPU is the part most likely to be
wrong in a way no run would reveal: the masking, and the zero-payload rule.
"""

import pytest
import torch

from pxf.couple.integrated_event import (assert_target_rows_untouched,
                                         mask_feedback)
from pxf.couple.pxdesign_iface import ConditioningFeedback

MASK = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # 2 binder rows, 2 target rows


def test_zero_payload_survives_in_training():
    """The correction this module exists for.

    A freshly zero-initialised output head emits exactly zero on step one.
    Collapsing that to None drops the gradient path with it, so the head would
    never receive a first update and the arm would train to nothing while
    looking like it ran.
    """
    delta = torch.zeros(1, 4, 8, requires_grad=True)
    out = mask_feedback(delta, MASK, zero_bypass=False)
    assert out is not None
    assert out.requires_grad
    out.sum().backward()
    assert delta.grad is not None


def test_zero_payload_bypasses_at_inference():
    """At inference the genuine no-feedback path is better than imitating it."""
    delta = torch.zeros(1, 4, 8)
    assert mask_feedback(delta, MASK, zero_bypass=True) is None


def test_zero_bypass_is_explicit_not_inferred():
    """Same values, opposite results: intent comes from the flag, not the data."""
    delta = torch.zeros(1, 4, 8)
    assert mask_feedback(delta, MASK, zero_bypass=False) is not None
    assert mask_feedback(delta, MASK, zero_bypass=True) is None


def test_target_rows_are_zeroed():
    out = mask_feedback(torch.ones(1, 4, 8), MASK, zero_bypass=True)
    per_row = out.reshape(4, -1).abs().sum(-1)
    assert per_row.tolist() == [8.0, 8.0, 0.0, 0.0]


def test_masking_is_differentiable_through_binder_rows_only():
    delta = torch.ones(1, 4, 8, requires_grad=True)
    mask_feedback(delta, MASK, zero_bypass=False).sum().backward()
    # Gradient reaches binder rows and is exactly zero on target rows, so a
    # target-row parameter cannot be trained by this path.
    assert delta.grad.reshape(4, -1).sum(-1).tolist() == [8.0, 8.0, 0.0, 0.0]


def test_conditioning_feedback_single_is_masked():
    fb = ConditioningFeedback(delta_single=torch.ones(1, 4, 6))
    out = mask_feedback(fb, MASK, zero_bypass=True)
    assert isinstance(out, ConditioningFeedback)
    assert out.delta_single.reshape(4, -1).abs().sum(-1).tolist() == [6, 6, 0, 0]
    assert out.delta_pair is None


def test_pair_block_is_binder_binder_only():
    """E2 writes the binder-BINDER block; cross-pairs are a separate policy."""
    fb = ConditioningFeedback(
        delta_single=torch.ones(1, 4, 6), delta_pair=torch.ones(4, 4, 3)
    )
    out = mask_feedback(fb, MASK, zero_bypass=True)
    kept = out.delta_pair.abs().sum(-1) > 0
    assert kept.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_all_zero_conditioning_feedback_bypasses_only_at_inference():
    fb = ConditioningFeedback(delta_single=torch.zeros(1, 4, 6))
    assert mask_feedback(fb, MASK, zero_bypass=True) is None
    assert mask_feedback(fb, MASK, zero_bypass=False) is not None


def test_target_row_audit_catches_a_bias_added_after_masking():
    """The failure mode the audit exists for: mask, then add a bias."""
    bad = torch.ones(1, 4, 8) * 0.5  # never masked
    with pytest.raises(AssertionError, match="target row"):
        assert_target_rows_untouched(bad, MASK)


def test_target_row_audit_passes_a_correctly_masked_payload():
    good = mask_feedback(torch.ones(1, 4, 8), MASK, zero_bypass=True)
    assert_target_rows_untouched(good, MASK)
    assert_target_rows_untouched(None, MASK)


def test_designer_module_refuses_to_guess():
    from pxf.couple.integrated_event import designer_module

    with pytest.raises(TypeError, match="nothing to attach to"):
        designer_module(lambda *a, **k: None)
