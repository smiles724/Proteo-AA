"""Tests for the held-out joint-refinement evaluator.

The pure parts -- event identity, the estimand, sign conventions, the verdict
and the JSON contract -- are tested without a donor, because they are where a
wrong answer is least visible. The reconstruction tests need the real donor
and are gated the way the rest of the joint suite is.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from pxf.eval import joint_metrics as M
from pxf.eval import joint_report as R
from pxf.joint import evaluation as E

DONOR = os.environ.get("PXDESIGN_DONOR", "")
CIF = os.environ.get("PXF_TEST_CIF", "")
needs_donor = pytest.mark.skipif(
    not (DONOR and CIF and Path(DONOR).is_file() and Path(CIF).is_file()),
    reason="set PXDESIGN_DONOR and PXF_TEST_CIF to run the reconstruction tests",
)


# ---- event identity ---------------------------------------------------------


def test_a_sigma_key_is_the_value_not_the_sweep_position():
    assert E.sigma_key(0.105) == E.sigma_key(0.10500000001)
    assert E.sigma_key(0.105) != E.sigma_key(0.314)


def test_two_sigmas_that_quantize_together_are_refused():
    with pytest.raises(E.EvaluationError, match="quantize"):
        E.check_distinct_sigmas([0.105, 0.1050000001])


def test_reordering_sigmas_does_not_move_an_event_seed():
    a = E.eval_seed(17, "p", "s1", 0.847, 0, "backbone_noise")
    b = E.eval_seed(17, "p", "s1", 0.847, 0, "backbone_noise")
    assert a == b
    # a different sigma, replicate or purpose is a different draw
    assert a != E.eval_seed(17, "p", "s1", 0.314, 0, "backbone_noise")
    assert a != E.eval_seed(17, "p", "s1", 0.847, 1, "backbone_noise")
    assert a != E.eval_seed(17, "p", "s1", 0.847, 0, "packing")


def test_the_eval_namespace_cannot_collide_with_a_training_seed():
    """Adding evaluation draws must not perturb any training stream."""
    from pxf.joint import randomness as jr

    assert "eval" not in " ".join(jr.STREAMS)
    training = jr.stream_seed(17, "s1", "backbone_noise", occurrence=0)
    evaluation = E.eval_seed(17, "p", "s1", 0.847, 0, "backbone_noise")
    assert training != evaluation


def test_the_same_event_gives_every_model_the_same_noise():
    a = E.replay_backbone_noise((16, 3), 17, "p", "s1", 0.847, 0)
    b = E.replay_backbone_noise((16, 3), 17, "p", "s1", 0.847, 0)
    assert torch.equal(a, b)
    assert E.noise_digest(a) == E.noise_digest(b)
    assert not torch.equal(a, E.replay_backbone_noise((16, 3), 17, "p", "s2", 0.847, 0))


# ---- metric direction and sign ----------------------------------------------


def test_an_undeclared_metric_direction_is_refused_not_guessed():
    with pytest.raises(R.ReportError, match="no declared direction"):
        R.metric_direction("bb_something_new")


def test_two_rates_point_opposite_ways():
    assert R.metric_direction("geom_valid_frame_rate") == +1
    assert R.metric_direction("geom_c_n_bad_rate") == -1


def test_positive_always_means_the_candidate_is_better():
    # RMSD: candidate lower is better
    assert R.signed_improvement("bb_ca_rmsd", 1.0, 1.5) > 0
    assert R.signed_improvement("bb_ca_rmsd", 1.5, 1.0) < 0
    # lDDT: candidate higher is better
    assert R.signed_improvement("bb_lddt", 0.9, 0.8) > 0
    assert R.signed_improvement("bb_lddt", 0.8, 0.9) < 0


# ---- the estimand ------------------------------------------------------------


def _row(model, sample, sigma, value, **extra):
    row = dict(model=model, sample_id=sample, sigma_key=E.sigma_key(sigma),
               bb_ca_rmsd=value, failure=False, panel_id="p",
               backbone_replicate=extra.pop("replicate", 0))
    row.update(extra)
    return row


def test_replicates_collapse_inside_an_event_before_a_target_is_formed():
    rows = [
        _row("B0", "s1", 0.1, 1.0, replicate=0),
        _row("B0", "s1", 0.1, 3.0, replicate=1),
    ]
    values = R.protein_values(rows, "bb_ca_rmsd", sigmas=[E.sigma_key(0.1)])
    assert values[("B0", "s1")] == pytest.approx(2.0)


def test_sigmas_are_weighted_equally_not_by_event_count():
    rows = [
        _row("B0", "s1", 0.1, 1.0, replicate=0),
        _row("B0", "s1", 0.1, 1.0, replicate=1),
        _row("B0", "s1", 0.1, 1.0, replicate=2),
        _row("B0", "s1", 0.9, 5.0, replicate=0),
    ]
    keys = [E.sigma_key(0.1), E.sigma_key(0.9)]
    values = R.protein_values(rows, "bb_ca_rmsd", sigmas=keys)
    # 3.0 (equal sigma weight), not 2.0 (event weight)
    assert values[("B0", "s1")] == pytest.approx(3.0)


def test_a_target_missing_a_sigma_is_dropped_not_reweighted():
    rows = [_row("B0", "s1", 0.1, 1.0), _row("B0", "s2", 0.1, 1.0),
            _row("B0", "s2", 0.9, 2.0)]
    keys = [E.sigma_key(0.1), E.sigma_key(0.9)]
    values = R.protein_values(rows, "bb_ca_rmsd", sigmas=keys)
    assert ("B0", "s1") not in values
    assert ("B0", "s2") in values


def test_a_failed_event_never_enters_the_estimate():
    rows = [_row("B0", "s1", 0.1, 1.0),
            dict(model="B0", sample_id="s2", sigma_key=E.sigma_key(0.1),
                 failure=True, panel_id="p", backbone_replicate=0)]
    values = R.protein_values(rows, "bb_ca_rmsd", sigmas=[E.sigma_key(0.1)])
    assert set(values) == {("B0", "s1")}


# ---- paired contrast and bootstrap ------------------------------------------


def _paired_rows(candidate_values, reference_values, sigma=0.1):
    rows = []
    for sample, value in candidate_values.items():
        rows.append(_row("B1", sample, sigma, value))
    for sample, value in reference_values.items():
        rows.append(_row("B0", sample, sigma, value))
    return rows


def test_a_uniform_improvement_is_positive_with_an_interval_above_zero():
    rows = _paired_rows({f"s{i}": 1.0 for i in range(12)},
                        {f"s{i}": 1.5 for i in range(12)})
    out = R.paired_contrast(rows, candidate="B1", reference="B0",
                            metric="bb_ca_rmsd", sigmas=[E.sigma_key(0.1)],
                            n_resamples=500, seed=3)
    assert out["mean"] == pytest.approx(0.5)
    assert out["ci_low"] > 0
    assert out["n_targets"] == 12


def test_the_bootstrap_unit_is_the_protein_not_the_event():
    rows = _paired_rows({f"s{i}": 1.0 for i in range(6)},
                        {f"s{i}": 1.2 for i in range(6)})
    out = R.paired_contrast(rows, candidate="B1", reference="B0",
                            metric="bb_ca_rmsd", sigmas=[E.sigma_key(0.1)],
                            n_resamples=200, seed=1)
    assert out["n_units"] == 6
    assert out["unit"] == "protein"
    assert out["homology_caveat"]


def test_cluster_labels_shrink_the_number_of_independent_units():
    rows = _paired_rows({f"s{i}": 1.0 for i in range(8)},
                        {f"s{i}": 1.4 for i in range(8)})
    clusters = {f"s{i}": f"c{i // 4}" for i in range(8)}
    out = R.paired_contrast(rows, candidate="B1", reference="B0",
                            metric="bb_ca_rmsd", sigmas=[E.sigma_key(0.1)],
                            clusters=clusters, n_resamples=200, seed=1)
    assert out["n_units"] == 2
    assert out["unit"] == "cluster"
    assert out["homology_caveat"] is None


def test_a_contrast_with_no_shared_target_is_incomplete_not_zero():
    rows = _paired_rows({"s1": 1.0}, {"s2": 1.0})
    out = R.paired_contrast(rows, candidate="B1", reference="B0",
                            metric="bb_ca_rmsd", sigmas=[E.sigma_key(0.1)])
    assert out["status"] == R.STATUS_INCOMPLETE
    assert out["mean"] is None


# ---- gate and verdict --------------------------------------------------------


def test_the_gate_needs_the_threshold_and_an_interval_off_zero():
    good = dict(mean=0.20, ci_low=0.10, ci_high=0.30)
    assert R.backbone_gate(good, 1.0)["status"] == R.STATUS_PASS
    # clears zero but not the 3%-of-baseline floor
    small = dict(mean=0.02, ci_low=0.01, ci_high=0.03)
    assert R.backbone_gate(small, 2.0)["status"] == R.STATUS_FAIL
    # big enough but the interval straddles zero
    noisy = dict(mean=0.20, ci_low=-0.05, ci_high=0.45)
    assert R.backbone_gate(noisy, 1.0)["status"] == R.STATUS_FAIL


def test_the_relative_floor_scales_with_the_matched_baseline():
    contrast = dict(mean=0.08, ci_low=0.02, ci_high=0.14)
    assert R.backbone_gate(contrast, 1.0)["status"] == R.STATUS_PASS   # floor 0.05
    assert R.backbone_gate(contrast, 5.0)["status"] == R.STATUS_FAIL   # floor 0.15


def test_backbone_mode_reports_packing_safety_as_incomplete_not_pass():
    safety = R.safety_status({}, mode="backbone")
    assert safety["status"] == R.STATUS_INCOMPLETE
    assert "unmeasured" in safety["reason"]


def test_without_a_baseline_there_is_no_verdict():
    out = R.verdict(bb_gate=dict(status=R.STATUS_PASS), safety=dict(status=R.STATUS_PASS),
                    have_baseline=False, complete_coverage=True)
    assert out["status"] == R.STATUS_INCOMPLETE
    assert "B0" in out["reason"]


def test_incomplete_coverage_cannot_pass_the_gate():
    out = R.verdict(bb_gate=dict(status=R.STATUS_PASS), safety=dict(status=R.STATUS_PASS),
                    have_baseline=True, complete_coverage=False)
    assert out["status"] == R.STATUS_INCOMPLETE


def test_a_passing_backbone_gate_alone_is_not_a_full_pass():
    out = R.verdict(bb_gate=dict(status=R.STATUS_PASS),
                    safety=R.safety_status({}, mode="backbone"),
                    have_baseline=True, complete_coverage=True)
    assert out["status"] == R.STATUS_INCOMPLETE


# ---- rows, merging and JSON --------------------------------------------------


def test_undefined_values_serialize_as_null_never_nan(tmp_path):
    path = R.write_json(tmp_path / "x.json", dict(value=float("nan"), ok=1.5))
    text = path.read_text()
    assert "NaN" not in text
    assert json.loads(text)["value"] is None


def test_conflicting_duplicate_rows_are_an_error(tmp_path):
    a = dict(model="B0", panel_id="p", sample_id="s1", sigma_key="0.1",
             backbone_replicate=0, bb_ca_rmsd=1.0)
    b = dict(a, bb_ca_rmsd=2.0)
    assert len(R.merge_rows([a], [dict(a)])) == 1
    with pytest.raises(R.ReportError, match="conflicting"):
        R.merge_rows([a], [b])


# ---- metric behaviour --------------------------------------------------------


def test_a_prediction_with_no_finite_atoms_is_a_failure_not_a_small_set():
    coords = torch.full((4, 37, 3), float("nan"))
    supplied = torch.zeros(4, 37)
    supplied[:, 1] = 1
    ok, reason = M.prediction_is_scorable(coords, supplied)
    assert not ok and "non-finite" in reason


def test_a_chain_break_does_not_invent_a_peptide_bond():
    coords = torch.zeros(4, 37, 3)
    coords[:, M.C] = torch.tensor([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    coords[:, M.N] = coords[:, M.C] + 1.329
    mask = torch.zeros(4, 37)
    mask[:, [M.N, M.CA, M.C]] = 1
    joined = M.backbone_geometry(coords, mask, residue_index=torch.tensor([0, 1, 2, 3]))
    broken = M.backbone_geometry(coords, mask, residue_index=torch.tensor([0, 1, 50, 51]))
    assert joined["geom_c_n_count"] == 3
    assert broken["geom_c_n_count"] == 2  # the 1->50 step is not adjacent


def test_a_zero_pair_metric_is_null_with_a_zero_count():
    coords = torch.zeros(3, 37, 3)
    mask = torch.zeros(3, 37)
    out = M.backbone_geometry(coords, mask)
    assert out["geom_c_n_mean"] is None
    assert out["geom_c_n_count"] == 0


def test_the_unimplemented_groups_raise_rather_than_approximate():
    for name in M.NOT_IMPLEMENTED_GROUPS:
        with pytest.raises(NotImplementedError, match="not implemented"):
            M.unimplemented_group(name)


# ---- reconstruction (needs the donor) ---------------------------------------


@needs_donor
def test_the_donor_snapshot_restores_every_tensor():
    model, _configs, record, snapshot = E.load_donor(DONOR)
    name, parameter = next(iter(model.named_parameters()))
    with torch.no_grad():
        parameter.add_(1.0)
    assert not torch.equal(parameter, snapshot[name])
    E._reset_to_donor(model, snapshot)
    assert torch.equal(dict(model.named_parameters())[name], snapshot[name])


@needs_donor
def test_a_checkpoint_trained_on_another_donor_is_refused(tmp_path):
    model, _configs, record, snapshot = E.load_donor(DONOR)
    fake = tmp_path / "ckpt.pt"
    torch.save(dict(
        arm="B0", settings=dict(arm="B0", trainable_blocks=4), step=10,
        examples_seen=80, trainable_state={},
        identity=dict(arm="B0", donor=dict(weights=dict(sha256="0" * 64))),
    ), fake)
    with pytest.raises(E.EvaluationError, match="was trained on donor"):
        E.load_joint_checkpoint(fake, donor_model=model, donor_snapshot=snapshot,
                                donor_record=record, weights="raw")


@needs_donor
def test_a_mislabeled_arm_is_refused(tmp_path):
    model, _configs, record, snapshot = E.load_donor(DONOR)
    fake = tmp_path / "ckpt.pt"
    torch.save(dict(
        arm="B0", settings=dict(arm="B1", trainable_blocks=4), step=1,
        identity=dict(arm="B0", donor=record),
    ), fake)
    with pytest.raises(E.EvaluationError, match="disagrees with itself"):
        E.load_joint_checkpoint(fake, donor_model=model, donor_snapshot=snapshot,
                                donor_record=record, weights="raw")


@needs_donor
def test_requested_ema_is_never_silently_downgraded_to_raw(tmp_path):
    model, _configs, record, snapshot = E.load_donor(DONOR)
    allow = {n: p for n, p in model.named_parameters()}
    from pxf.joint.trainer import select_trainable

    trainable = select_trainable(model, n_blocks=4)
    fake = tmp_path / "ckpt.pt"
    torch.save(dict(
        arm="B0", settings=dict(arm="B0", trainable_blocks=4), step=1,
        examples_seen=8, ema=None,
        trainable_state={n: p.detach().cpu().clone() for n, p in trainable.items()},
        identity=dict(arm="B0", donor=record, trainable_names=list(trainable)),
    ), fake)
    with pytest.raises(E.EvaluationError, match="no EMA payload"):
        E.load_joint_checkpoint(fake, donor_model=model, donor_snapshot=snapshot,
                                donor_record=record, weights="ema")


@needs_donor
def test_raw_reconstruction_restores_the_saved_values_exactly(tmp_path):
    model, _configs, record, snapshot = E.load_donor(DONOR)
    from pxf.joint.trainer import select_trainable

    trainable = select_trainable(model, n_blocks=4)
    saved = {}
    with torch.no_grad():
        for name, parameter in trainable.items():
            saved[name] = (parameter + 0.25).detach().cpu().clone()
    fake = tmp_path / "ckpt.pt"
    torch.save(dict(
        arm="B0", settings=dict(arm="B0", trainable_blocks=4), step=7,
        examples_seen=56, trainable_state=saved,
        identity=dict(arm="B0", donor=record, trainable_names=list(trainable)),
    ), fake)
    loaded = E.load_joint_checkpoint(fake, donor_model=model, donor_snapshot=snapshot,
                                     donor_record=record, weights="raw")
    live = dict(loaded.model.named_parameters())
    for name, value in saved.items():
        assert torch.equal(live[name].detach().cpu(), value)
    assert loaded.step == 7 and loaded.arm == "B0"


@needs_donor
def test_load_order_does_not_leak_weights_between_checkpoints(tmp_path):
    model, _configs, record, snapshot = E.load_donor(DONOR)
    from pxf.joint.trainer import select_trainable

    trainable = select_trainable(model, n_blocks=4)
    shifted = {n: (p + 1.0).detach().cpu().clone() for n, p in trainable.items()}
    same = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    for tag, payload in (("b1", shifted), ("b0", same)):
        torch.save(dict(
            arm="B0", settings=dict(arm="B0", trainable_blocks=4), step=1,
            trainable_state=payload,
            identity=dict(arm="B0", donor=record, trainable_names=list(trainable)),
        ), tmp_path / f"{tag}.pt")

    E.load_joint_checkpoint(tmp_path / "b1.pt", donor_model=model,
                            donor_snapshot=snapshot, donor_record=record, weights="raw")
    E.load_joint_checkpoint(tmp_path / "b0.pt", donor_model=model,
                            donor_snapshot=snapshot, donor_record=record, weights="raw")
    live = dict(model.named_parameters())
    for name, value in same.items():
        assert torch.equal(live[name].detach().cpu(), value), name


class TestReportNumberFormatting:
    """A real effect must not print as an exactly-zero-looking one.

    The report's fixed-point column rounds 1.08e-05 to "+0.0000", which reads
    as "no difference" and makes a narrow interval look degenerate. Scientific
    notation below the rounding floor is the difference between "too small to
    matter" and "too small for this column".
    """

    def test_ordinary_magnitudes_stay_fixed_point(self):
        assert R._fmt(0.1234) == "+0.1234"
        assert R._fmt(-0.5) == "-0.5000"

    def test_below_the_rounding_floor_goes_scientific(self):
        assert R._fmt(1.08e-05) == "+1.080e-05"
        assert R._fmt(-2e-06) == "-2.000e-06"

    def test_exact_zero_is_not_dressed_up(self):
        """Nothing to distinguish from zero: it *is* zero."""
        assert R._fmt(0) == "0"
        assert R._fmt(0.0) == "0"

    def test_missing_is_a_dash_not_a_number(self):
        assert R._fmt(None) == "-"

    def test_the_boundary_is_where_rounding_would_lose_the_sign(self):
        """5e-4 still survives four decimals; below it the digits are gone."""
        assert R._fmt(5e-4) == "+0.0005"
        assert R._fmt(4.9e-4).endswith("e-04")
