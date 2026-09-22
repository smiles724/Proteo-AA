"""The checkpoint selector must refuse, not shrug.

Both cases below were demonstrated by review to SELECT a checkpoint at
e73f484: the guardrail was skipped whenever either value was missing, and the
chemistry metrics were unimplemented placeholders returning None, so every
checkpoint was eligible -- including one failing chemistry on every complex.
"""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_eval_mod",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_integrated_feedback.py",
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)

SELECTION = {
    "primary": "resolved_binder_bb_rmsd",
    "tie_band_relative": 0.01,
    "guardrails": [
        {"metric": "backbone_chemistry_failure_rate",
         "max_relative_regression": 0.05, "zero_baseline_rule": "require_zero"},
        {"metric": "sidechain_chemistry_failure_rate",
         "max_relative_regression": 0.05, "zero_baseline_rule": "require_zero"},
    ],
}
CLEAN_BASELINE = {"backbone_chemistry_failure_rate": 0.0,
                  "sidechain_chemistry_failure_rate": 0.0}


def record(step, rmsd, backbone=0.0, sidechain=0.0):
    return {"step": step, "metrics": {
        "resolved_binder_bb_rmsd": rmsd,
        "backbone_chemistry_failure_rate": backbone,
        "sidechain_chemistry_failure_rate": sidechain,
    }}


def test_a_missing_guardrail_metric_refuses_selection():
    out = EVAL.select_checkpoint(
        [record(500, 0.2, backbone=None, sidechain=None)],
        SELECTION, no_feedback=CLEAN_BASELINE,
    )
    assert out["selected"] is None
    assert "guardrail" in out["reason"]


def test_total_chemistry_failure_refuses_selection():
    out = EVAL.select_checkpoint(
        [record(500, 0.2, backbone=1.0)], SELECTION, no_feedback=CLEAN_BASELINE
    )
    assert out["selected"] is None
    assert any("require_zero" in r
               for r in out["considered"][0]["ineligible_because"])


def test_a_missing_no_feedback_reference_refuses_selection():
    """The reference must be the matched no-feedback baseline, not the run's
    own earliest checkpoint -- which cannot detect a regression present from
    step 500."""
    out = EVAL.select_checkpoint(
        [record(500, 0.2)], SELECTION, no_feedback=None
    )
    assert out["selected"] is None


def test_a_clean_record_is_still_selectable():
    """The refusals must not be vacuous: a sound checkpoint still wins."""
    out = EVAL.select_checkpoint(
        [record(500, 0.20), record(1000, 0.19), record(2000, 0.189)],
        SELECTION, no_feedback=CLEAN_BASELINE,
    )
    # 0.189 is within 1% of the best, so the EARLIER of the tied pair wins.
    assert out["selected"] == 1000


def test_the_tie_break_prefers_the_earlier_step():
    out = EVAL.select_checkpoint(
        [record(500, 0.2005), record(1000, 0.2)],
        SELECTION, no_feedback=CLEAN_BASELINE,
    )
    assert out["selected"] == 500


def test_non_finite_primary_refuses_selection():
    out = EVAL.select_checkpoint(
        [record(500, float("nan"))], SELECTION, no_feedback=CLEAN_BASELINE
    )
    assert out["selected"] is None
