"""The folding half of the AF2-IG filter must hand the scorer what it asks for.

`fold_af2ig.py` and `score_af2ig_designability.py` run in different environments
— JAX on one side, torch on the other — so nothing at runtime checks that the
CSV one writes is the CSV the other reads. These tests are that check, plus the
three places where a wrong answer would look like a plausible one:

  * the binder chain. It is not always 'B'; on IL17A, TNFa, VEGFA and H1 the
    target already occupies several letters. Scoring the wrong chain produces a
    complete, believable row.
  * the superposition. An SVD-based Kabsch without the determinant fix accepts
    a mirror image, which would call a left-handed helix a perfect match.
  * resume. The two Table 4 arms score the same `sample_id` twice, once per
    sequence, so a resume key that ignores the variant silently drops one.
"""
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Top-level import, matching how `scripts/evaluation/fold_af2ig.py` does it:
# the folding environment has no Protenix and so cannot execute
# `pxdesign_train/__init__.py`. Importing it the same way here is the point --
# if this stops working, the folder stops working.
BENCHMARKS_DIR = REPO_ROOT / "pxdesign_train" / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from af2ig import (  # noqa: E402
    METRIC_COLUMNS,
    REQUIRED_METRIC_COLUMNS,
    MetricRow,
    MetricsSink,
    bound_unbound_rmsd,
    chain_residue_counts,
    kabsch_rmsd,
    load_prep_sidecars,
    read_designs_csv,
    resolve_binder_chain,
)


def _load_scorer():
    """Import the scorer script by path; it is a script, not a module."""
    path = REPO_ROOT / "scripts" / "evaluation" / "score_af2ig_designability.py"
    spec = importlib.util.spec_from_file_location("score_af2ig_designability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ the CSV contract


def test_metric_columns_are_exactly_what_the_scorer_requires():
    """The first seven columns are a contract between two environments."""
    assert REQUIRED_METRIC_COLUMNS == _load_scorer().REQUIRED


def test_metric_columns_carry_the_variant_the_two_table4_arms_need():
    assert "variant" in METRIC_COLUMNS
    assert set(REQUIRED_METRIC_COLUMNS).issubset(METRIC_COLUMNS)


def test_a_written_metrics_csv_scores_end_to_end(tmp_path):
    """Write rows the way the folder does; read them the way the scorer does."""
    scorer = _load_scorer()
    metrics_csv = tmp_path / "af2ig_metrics.csv"
    with MetricsSink(metrics_csv) as sink:
        # PDL1: one passing design, one failing on ipAE alone.
        sink.write(MetricRow("PDL1_L80_s0000", "PDL1", 80, 5.0, 0.9, 0.95, 1.0))
        sink.write(MetricRow("PDL1_L80_s0001", "PDL1", 80, 20.0, 0.9, 0.95, 1.0))

    rows, dropped = scorer._load_rows(metrics_csv, allow_missing=False)
    assert dropped == 0
    benchmark = scorer.ConditionalBinderDesignBenchmark.load()
    scores = benchmark.designability(rows, variant=None)
    assert scores["PDL1"]["n_samples"] == 2
    assert scores["PDL1"]["n_designable"] == 1
    assert scores["PDL1"]["designability"] == pytest.approx(50.0)


# ------------------------------------------------------------- superposition


def _helix(n: int, handedness: float = 1.0) -> np.ndarray:
    t = np.arange(n) * 100.0 * np.pi / 180.0
    return np.stack([2.3 * np.cos(t), handedness * 2.3 * np.sin(t), 1.5 * np.arange(n)], axis=1)


def test_kabsch_rmsd_is_zero_under_rotation_and_translation():
    coords = _helix(30)
    angle = 0.7
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved = coords @ rotation.T + np.array([10.0, -3.0, 42.0])
    assert kabsch_rmsd(moved, coords) == pytest.approx(0.0, abs=1e-8)


def test_kabsch_rmsd_does_not_accept_a_mirror_image():
    """Without the determinant fix an SVD can return a reflection.

    A left-handed helix would then score 0 A against its right-handed twin, and
    criterion (d) would pass every design whose unbound fold is its own mirror.
    """
    right = _helix(30, handedness=+1.0)
    left = _helix(30, handedness=-1.0)
    assert kabsch_rmsd(left, right) > 1.0


def test_bound_unbound_rmsd_is_the_superimposed_distance():
    bound = _helix(40)
    unbound = bound + np.array([0.0, 0.0, 0.0])
    unbound[20:] += 4.0  # a hinge: half the helix moves
    value = bound_unbound_rmsd(bound, unbound)
    assert 0.5 < value < 10.0
    # symmetric, because it is an optimal superposition of two free frames
    assert value == pytest.approx(bound_unbound_rmsd(unbound, bound), abs=1e-8)


@pytest.mark.parametrize(
    "a, b",
    [
        (np.zeros((5, 3)), np.zeros((6, 3))),   # length mismatch
        (np.zeros((2, 3)), np.zeros((2, 3))),   # too few points to superimpose
    ],
)
def test_kabsch_rmsd_refuses_ill_posed_input(a, b):
    with pytest.raises(ValueError):
        kabsch_rmsd(a, b)


# ------------------------------------------------------------- binder chain


def _write_pdb(path: Path, chains: dict[str, int]) -> Path:
    """A CA-only PDB with `chains` mapping chain id -> residue count."""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    total = sum(chains.values())
    array = struc.AtomArray(total)
    i = 0
    for chain_id, count in chains.items():
        for res in range(count):
            array.coord[i] = [float(i), float(res), 0.0]
            array.chain_id[i] = chain_id
            array.res_id[i] = res + 1
            array.res_name[i] = "GLY"
            array.atom_name[i] = "CA"
            array.element[i] = "C"
            array.hetero[i] = False
            i += 1
    pdb = PDBFile()
    pdb.set_structure(array)
    pdb.write(str(path))
    return path


def test_chain_residue_counts_reads_every_chain(tmp_path):
    pdb = _write_pdb(tmp_path / "d.pdb", {"A": 12, "Z": 8})
    assert chain_residue_counts(pdb) == {"A": 12, "Z": 8}


def test_binder_chain_comes_from_the_sidecar_not_from_a_guess(tmp_path):
    """The multi-chain targets are exactly where guessing goes wrong.

    Here the target is 80 residues and so is the binder — which is not contrived,
    the grid samples 80-130 against targets of 150-440 residues split over up to
    three chains.
    """
    pdb = _write_pdb(tmp_path / "d.pdb", {"A": 80, "Z": 80})
    with pytest.raises(ValueError, match="cannot tell which chain is the binder"):
        resolve_binder_chain(pdb, 80)

    binder, targets = resolve_binder_chain(pdb, 80, {"binder_chain_id": "Z"})
    assert binder == "Z"
    assert targets == ("A",)


def test_binder_chain_is_inferred_when_it_is_unambiguous(tmp_path):
    pdb = _write_pdb(tmp_path / "d.pdb", {"A": 116, "B": 80})
    assert resolve_binder_chain(pdb, 80) == ("B", ("A",))


def test_a_sidecar_that_disagrees_with_the_file_is_an_error(tmp_path):
    pdb = _write_pdb(tmp_path / "d.pdb", {"A": 116, "B": 80})
    with pytest.raises(ValueError, match="designs.csv says"):
        resolve_binder_chain(pdb, 100, {"binder_chain_id": "B"})
    with pytest.raises(ValueError, match="has chains"):
        resolve_binder_chain(pdb, 80, {"binder_chain_id": "Q"})


def test_prep_sidecars_are_keyed_by_task_id(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "PDL1_L80.prep.json").write_text(
        json.dumps({"task_id": "PDL1_L80", "binder_chain_id": "B"})
    )
    (inputs / "TNFa_L90.prep.json").write_text(
        json.dumps({"task_id": "TNFa_L90", "binder_chain_id": "Z"})
    )
    loaded = load_prep_sidecars(inputs)
    assert loaded["PDL1_L80"]["binder_chain_id"] == "B"
    assert loaded["TNFa_L90"]["binder_chain_id"] == "Z"


# -------------------------------------------------------------------- resume


def _row(sample_id: str, variant: str) -> MetricRow:
    return MetricRow(sample_id, "PDL1", 80, 5.0, 0.9, 0.95, 1.0, variant=variant)


def test_sink_resume_distinguishes_the_two_table4_arms(tmp_path):
    path = tmp_path / "m.csv"
    with MetricsSink(path) as sink:
        sink.write(_row("PDL1_L80_s0000", "co_design"))

    with MetricsSink(path) as sink:
        assert sink.has("PDL1_L80_s0000", "co_design")
        # The PMPNN arm scores the same sample under a different sequence; a
        # resume key of sample_id alone would skip it forever.
        assert not sink.has("PDL1_L80_s0000", "pmpnn")
        sink.write(_row("PDL1_L80_s0000", "pmpnn"))

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["variant"] for r in rows] == ["co_design", "pmpnn"]
    assert len(rows) == 2


def test_sink_overwrite_starts_a_fresh_file(tmp_path):
    path = tmp_path / "m.csv"
    with MetricsSink(path) as sink:
        sink.write(_row("PDL1_L80_s0000", "co_design"))
    with MetricsSink(path, overwrite=True) as sink:
        assert not sink.has("PDL1_L80_s0000", "co_design")


def test_sink_refuses_to_append_under_a_foreign_header(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("sample_id,target,plddt\nx,PDL1,0.9\n")
    with pytest.raises(ValueError, match="does not match this harness"):
        MetricsSink(path)


def test_missing_metrics_are_written_blank_not_zero(tmp_path):
    """A failed fold and a failed design are different facts.

    The scorer refuses a blank without `--allow-missing` precisely so a crashed
    job cannot quietly count as a non-designable sample. The sink has to leave
    the cell blank for that to work rather than writing 0.0.
    """
    path = tmp_path / "m.csv"
    with MetricsSink(path) as sink:
        sink.write(MetricRow("PDL1_L80_s0000", "PDL1", 80, None, None, None, None))
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["ipae"] == "" and row["binder_bound_unbound_rmsd"] == ""

    scorer = _load_scorer()
    with pytest.raises(SystemExit, match="has no value for"):
        scorer._load_rows(path, allow_missing=False)
    rows, dropped = scorer._load_rows(path, allow_missing=True)
    assert dropped == 1


# ------------------------------------------------------------- designs.csv


def test_read_designs_csv_requires_the_generation_columns(tmp_path):
    path = tmp_path / "designs.csv"
    path.write_text("sample_id,target\nx,PDL1\n")
    with pytest.raises(ValueError, match="missing column"):
        read_designs_csv(path)


def test_read_designs_csv_rejects_a_header_only_file(tmp_path):
    path = tmp_path / "designs.csv"
    path.write_text("sample_id,target,binder_length,sequence\n")
    with pytest.raises(ValueError, match="no design rows"):
        read_designs_csv(path)


# ------------------------------------------------- ColabDesign unit conversions


def _load_folder():
    """Import the folding driver by path.

    Its module-level imports are numpy + the af2ig helpers only; ColabDesign is
    imported inside `Af2Ig.__init__`, so this works without the AF2 stack.
    """
    path = REPO_ROOT / "scripts" / "evaluation" / "fold_af2ig.py"
    spec = importlib.util.spec_from_file_location("fold_af2ig", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ipae_is_rescaled_out_of_colabdesigns_normalisation():
    """ColabDesign divides PAE by AF2's 31 A cap; the filter is in Angstroms.

    Forgetting the factor is silent and catastrophic in one direction: a raw
    0.30 would read as 0.30 A against a 10.85 A threshold instead of the 9.3 A
    it is, and every design would clear criterion (a).
    """
    folder = _load_folder()
    assert folder.PAE_SCALE == 31.0

    from conditional_binder import AF2IGFilter

    filt = AF2IGFilter()
    others = {"iptm": 0.62, "plddt": 0.88, "binder_bound_unbound_rmsd": 1.0}

    # 0.40 * 31 = 12.4 A, which fails criterion (a) at 10.85 -- but the raw
    # 0.40 clears it comfortably. Every design would pass, in exactly the
    # direction that flatters the model.
    raw = 0.40
    metrics = folder._metrics_from_log(
        {"i_pae": raw, "i_ptm": 0.62, "plddt": 0.88, "ptm": 0.71, "rmsd": 1.4}
    )
    assert metrics["ipae"] == pytest.approx(12.4)
    assert not filt.is_designable({"ipae": metrics["ipae"], **others})
    assert filt.is_designable({"ipae": raw, **others})


def test_plddt_is_taken_on_the_zero_to_one_scale():
    """ColabDesign's log already flips `1 - plddt` back; do not flip it twice."""
    folder = _load_folder()
    metrics = folder._metrics_from_log(
        {"i_pae": 0.1, "i_ptm": 0.8, "plddt": 0.93, "ptm": 0.8, "rmsd": 1.0}
    )
    assert metrics["plddt"] == pytest.approx(0.93)


def test_the_designed_rmsd_is_carried_but_is_not_the_filters_rmsd():
    """`rmsd` from the binder protocol is predicted-vs-designed, not bound/unbound."""
    folder = _load_folder()
    metrics = folder._metrics_from_log(
        {"i_pae": 0.1, "i_ptm": 0.8, "plddt": 0.9, "ptm": 0.8, "rmsd": 2.5}
    )
    assert metrics["binder_designed_rmsd"] == pytest.approx(2.5)
    assert "binder_bound_unbound_rmsd" not in metrics
