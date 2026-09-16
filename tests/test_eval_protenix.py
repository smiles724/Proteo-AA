"""The before/after comparison on the recentPDB eval split.

The whole reason this script exists is to catch a fine-tune that *degrades*
packing, so the regression detector is the part that must not be wrong: a
comparison that silently reports "ok" on a worse model is the failure mode.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

ev = pytest.importorskip("eval_protenix_sidechain")

BASE = {
    "symmetry_rmsd": 0.80,
    "rotamer_recovery": 0.81,
    "chi_recovery_20deg": 0.82,
    "chi_recovery_40deg": 0.88,
    "chi1_accuracy_20deg": 0.91,
    "chi1_chi2_accuracy_20deg": 0.78,
    "lddt_sc_sc": 0.91,
    "lddt_sc_env": 0.93,
    "bad_bond_fraction": 0.003,
    "bond_mae": 0.04,
    "rotamer_outlier_fraction_40deg": 0.19,
    "completeness": 1.0,
    "observed_atoms": 5000,
}


def _run(tmp_path, name, overrides=None):
    """A minimal sidechain_metrics.json, as the script writes it."""
    summary = dict(BASE)
    summary.update(overrides or {})
    record = dict(
        label="recentPDB_low_homology",
        n_scored=1642,
        summary={"supervised": summary, "all_canonical": dict(summary)},
    )
    directory = tmp_path / name
    directory.mkdir()
    (directory / "sidechain_metrics.json").write_text(json.dumps(record))
    return directory


def test_an_unchanged_model_is_not_a_regression(tmp_path, capsys):
    before, after = _run(tmp_path, "b"), _run(tmp_path, "a")
    assert ev.compare(before, after) == 0
    assert "no regression" in capsys.readouterr().out


def test_a_worse_rmsd_is_flagged(tmp_path, capsys):
    """Higher RMSD is worse, and the sign convention is easy to get backwards."""
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a", {"symmetry_rmsd": 0.95})
    assert ev.compare(before, after) == 1
    out = capsys.readouterr().out
    assert "REGRESSION" in out and "symmetry_rmsd: 0.8000 -> 0.9500" in out


def test_a_better_rmsd_is_not_flagged(tmp_path, capsys):
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a", {"symmetry_rmsd": 0.70})
    assert ev.compare(before, after) == 0
    assert "no regression" in capsys.readouterr().out


def test_worse_rotamer_recovery_is_flagged(tmp_path, capsys):
    """Lower recovery is worse -- the opposite convention to RMSD."""
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a", {"rotamer_recovery": 0.75})
    assert ev.compare(before, after) == 1
    assert "rotamer_recovery: 0.8100 -> 0.7500" in capsys.readouterr().out


def test_every_headline_metric_can_trigger_it(tmp_path):
    """Otherwise a metric could be in the table but not in the verdict."""
    for key in ev.HEADLINE:
        worse = BASE[key] * (1.2 if key in ev.LOWER_IS_BETTER else 0.8)
        before = _run(tmp_path, f"b_{key}")
        after = _run(tmp_path, f"a_{key}", {key: worse})
        assert ev.compare(before, after) == 1, f"{key} did not trigger the verdict"


def test_improving_one_metric_does_not_mask_regressing_another(tmp_path, capsys):
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a", {"symmetry_rmsd": 0.60, "rotamer_recovery": 0.70})
    assert ev.compare(before, after) == 1
    out = capsys.readouterr().out
    assert "rotamer_recovery" in out.split("REGRESSION")[1]


def test_a_regression_only_outside_the_supervised_scope_is_not_the_verdict(tmp_path):
    """The verdict is on trustworthy residues; the other scope is context.

    Scoring against zero-occupancy or B>80 side chains measures crystallographic
    noise, and noise moves both ways -- so it is reported, not adjudicated.
    """
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a")
    record = json.loads((after / "sidechain_metrics.json").read_text())
    record["summary"]["all_canonical"]["symmetry_rmsd"] = 2.0
    (after / "sidechain_metrics.json").write_text(json.dumps(record))
    assert ev.compare(before, after) == 0


def test_mismatched_target_counts_are_warned_about(tmp_path, caplog):
    """Comparing deltas computed on different residue sets is meaningless."""
    before = _run(tmp_path, "b")
    after = _run(tmp_path, "a")
    record = json.loads((after / "sidechain_metrics.json").read_text())
    record["n_scored"] = 1200
    (after / "sidechain_metrics.json").write_text(json.dumps(record))
    with caplog.at_level("WARNING"):
        ev.compare(before, after)
    assert "different target counts" in caplog.text


def test_lower_is_better_covers_the_direction_of_every_reported_metric():
    """A metric absent from both conventions would be scored in whichever
    direction the default happened to be."""
    reported = {key for keys in ev.REPORT.values() for key in keys}
    higher_is_better = {
        "rotamer_recovery",
        "chi_recovery_20deg",
        "chi_recovery_40deg",
        "chi1_accuracy_20deg",
        "chi1_chi2_accuracy_20deg",
        "lddt_sc_sc",
        "lddt_sc_env",
        "completeness",
    }
    assert reported == set(ev.LOWER_IS_BETTER) | higher_is_better
    assert set(ev.HEADLINE) <= reported


def test_the_split_is_documented_in_the_module():
    """The disjointness claim should be findable next to the code that relies on it."""
    assert "2021-09-30" in ev.__doc__
    assert "1,818" in ev.__doc__
