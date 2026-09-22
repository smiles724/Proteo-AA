"""The benchmark test set must be exactly what the paper specifies.

Three things can silently turn this benchmark into a different, easier one:

  * a threshold drifting away from A-CODE Appendix C.2;
  * a target quietly running on invented crops or hotspots because nobody
    noticed the manifest entry was a placeholder;
  * a hotspot that does not resolve to a token, so the model conditions on fewer
    hotspots than the paper gives it — which looks like a worse model, not a
    broken harness.

These tests pin all three, plus the arithmetic of the metric (successes pooled
across the length grid, not averaged per length).
"""
import json
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)

from pxdesign_train.benchmarks import (  # noqa: E402
    AF2IGFilter,
    ConditionalBinderDesignBenchmark,
    TargetStatus,
)
from pxdesign_train.benchmarks.conditional_binder import TABLE4_ORDER  # noqa: E402


@pytest.fixture(scope="module")
def benchmark():
    return ConditionalBinderDesignBenchmark.load()


# ----------------------------------------------------------------- the filter


def test_af2ig_thresholds_match_the_paper(benchmark):
    """A-CODE Appendix C.2, verbatim: ipAE < 10.85, ipTM > 0.5, pLDDT > 80%,
    binder bound/unbound RMSD < 3.5 A."""
    f = benchmark.af2ig
    assert f.ipae_max == 10.85
    assert f.iptm_min == 0.5
    assert f.plddt_min == 0.8
    assert f.binder_bound_unbound_rmsd_max == 3.5


def test_filter_is_a_conjunction_and_each_criterion_bites():
    f = AF2IGFilter()
    passing = {"ipae": 5.0, "iptm": 0.9, "plddt": 0.95, "binder_bound_unbound_rmsd": 1.0}
    assert f.is_designable(passing)
    for key, failing_value in (
        ("ipae", 10.85),                      # strict <
        ("iptm", 0.5),                        # strict >
        ("plddt", 0.8),                       # strict >
        ("binder_bound_unbound_rmsd", 3.5),   # strict <
    ):
        assert not f.is_designable({**passing, key: failing_value}), key


def test_plddt_on_the_0_100_scale_is_not_a_free_pass():
    """An AF2 wrapper reporting pLDDT as 0-100 must not make criterion (c) vacuous."""
    f = AF2IGFilter()
    row = {"ipae": 5.0, "iptm": 0.9, "binder_bound_unbound_rmsd": 1.0}
    assert f.is_designable({**row, "plddt": 91.0})
    assert not f.is_designable({**row, "plddt": 62.0})


def test_missing_metric_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        AF2IGFilter().is_designable({"ipae": 1.0, "iptm": 0.9, "plddt": 0.9})


# ----------------------------------------------------------------- the targets


def test_the_test_set_has_the_paper_s_ten_targets(benchmark):
    assert {t.name for t in benchmark.targets} == set(TABLE4_ORDER)


def test_tasks_are_emitted_in_table4_column_order(benchmark):
    tasks = benchmark.tasks(lengths=[80], samples_per_length=1)
    order = [t.target.name for t in tasks]
    assert order == sorted(order, key=TABLE4_ORDER.index)


def test_the_shipped_manifest_is_complete(benchmark):
    """All ten targets are sourced, so a run fills every Table 4 column."""
    assert benchmark.pending_targets() == ()
    assert len(benchmark.runnable_targets()) == 10
    benchmark.tasks(include_pending=True)  # must not raise


def _manifest_with_one_pending(tmp_path):
    manifest = json.loads(
        ConditionalBinderDesignBenchmark.load().manifest_path.read_text()
    )
    for entry in manifest["targets"]:
        if entry["name"] == "PDL1":
            entry.update(
                {"status": "pending_source", "pdb_id": None, "chains": None,
                 "hotspots": None}
            )
    path = tmp_path / "one_pending.json"
    path.write_text(json.dumps(manifest))
    return ConditionalBinderDesignBenchmark.load(path)


def test_a_pending_target_is_skipped_not_guessed(tmp_path):
    """The guard still has to work: an unsourced target never produces a task.

    The shipped manifest is complete, so this drives a copy with one entry
    blanked. Without it, filling the manifest would silently delete the
    protection that kept invented crops and hotspots out of a benchmark run.
    """
    bench = _manifest_with_one_pending(tmp_path)
    pending = bench.pending_targets()
    assert [t.name for t in pending] == ["PDL1"]
    assert "PDL1" not in {t.target.name for t in bench.tasks()}
    assert pending[0].status is TargetStatus.PENDING_SOURCE
    with pytest.raises(ValueError, match="pending_source"):
        pending[0].require_runnable()


def test_include_pending_fails_loudly(tmp_path):
    bench = _manifest_with_one_pending(tmp_path)
    with pytest.raises(ValueError, match="pending_source"):
        bench.tasks(include_pending=True)


def test_every_runnable_target_is_fully_specified(benchmark):
    for target in benchmark.runnable_targets():
        assert target.pdb_id and len(target.pdb_id) == 4
        assert target.chains, target.name
        assert target.hotspots, target.name
        assert target.numbering == "author"
        assert target.source, f"{target.name} has no provenance"
        for chain, ranges in target.chains.items():
            for first, last in ranges:
                assert first <= last


def test_tnfa_follows_alphaproteo_not_pxdesign(benchmark):
    """The one target whose two sources disagree.

    AlphaProteo Table S1 gives A113 + C73; PXDesign Table 3 gives A31, A32, A113,
    C73, C87. A-CODE defines its set as the targets "proposed in Zambaldi et al.",
    so AlphaProteo wins. Pinned because silently drifting to the five-hotspot set
    would condition on a different, more heavily specified interface and nothing
    else would notice.
    """
    tnfa = benchmark.target("TNFa")
    assert tnfa.hotspots == (("A", 113), ("C", 73))
    assert tnfa.source == "alphaproteo_table_s1"
    assert set(tnfa.chains) == {"A", "B", "C"}


def test_h1_chain_b_carries_the_hemagglutinin_numbering_offset(benchmark):
    """H1 is the only entry not transcribed literally from Table S1.

    AlphaProteo quotes HA2 in canonical 1-175 numbering (B1-68, B80-170; hotspots
    B21, B45, B52) while 5VLI deposits HA2 as author residues 501-670. The
    manifest is author-numbered throughout, so chain B carries +500. If this ever
    reverts to the literal values, every H1 hotspot falls outside the crop and the
    conditioning silently empties.
    """
    h1 = benchmark.target("H1")
    assert h1.hotspots == (("B", 521), ("B", 545), ("B", 552))
    assert h1.chains["B"] == [(501, 568), (580, 670)]
    # HA1 needs no shift.
    assert h1.chains["A"][0] == (1, 50)
    assert h1.numbering == "author"


def test_hotspots_outside_the_crop_are_rejected_at_load(tmp_path):
    """A hotspot the crop drops would silently weaken the conditioning."""
    manifest = json.loads(
        ConditionalBinderDesignBenchmark.load().manifest_path.read_text()
    )
    for entry in manifest["targets"]:
        if entry["name"] == "PDL1":
            entry["hotspots"] = [["A", 999]]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside the crop"):
        ConditionalBinderDesignBenchmark.load(path)


def test_manifest_must_have_ten_targets(tmp_path):
    manifest = json.loads(
        ConditionalBinderDesignBenchmark.load().manifest_path.read_text()
    )
    manifest["targets"] = manifest["targets"][:9]
    path = tmp_path / "short.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="has 10"):
        ConditionalBinderDesignBenchmark.load(path)


# ---------------------------------------------------------------- the sampling


def test_length_grid_is_inside_the_papers_envelope(benchmark):
    """"lengths ranging from 80 to 130" and 328-728 samples per target."""
    assert min(benchmark.lengths) == 80
    assert max(benchmark.lengths) == 130
    per_target = len(benchmark.lengths) * benchmark.samples_per_length
    assert 328 <= per_target <= 728, per_target


def test_task_grid_expands_to_the_full_per_target_budget(benchmark):
    tasks = [t for t in benchmark.tasks() if t.target.name == "PDL1"]
    assert len(tasks) == len(benchmark.lengths)
    assert sum(t.n_samples for t in tasks) == (
        len(benchmark.lengths) * benchmark.samples_per_length
    )
    assert len({t.binder_length for t in tasks}) == len(benchmark.lengths)


def test_sample_ids_are_unique_and_stable(benchmark):
    tasks = benchmark.tasks(only=["PDL1"], lengths=[80, 90], samples_per_length=4)
    ids = [sid for task in tasks for sid in task.sample_ids()]
    assert len(ids) == len(set(ids)) == 8
    assert ids[0] == "PDL1_L80_s0000"
    # Regenerating must give byte-identical ids, or a resumed run duplicates work.
    assert ids == [sid for task in tasks for sid in task.sample_ids()]


# ------------------------------------------------------------------ the metric


def _row(target, length, designable, variant="co_design"):
    good = {"ipae": 1.0, "iptm": 0.99, "plddt": 0.95, "binder_bound_unbound_rmsd": 0.5}
    bad = {"ipae": 20.0, "iptm": 0.1, "plddt": 0.5, "binder_bound_unbound_rmsd": 9.0}
    return {"target": target, "binder_length": length, "variant": variant,
            **(good if designable else bad)}


def test_designability_pools_success_counts_across_lengths(benchmark):
    """The paper sums successes over the length grid; it does not average rates.

    With 1/10 at length 80 and 9/10 at 130 the pooled answer is 50%. Averaging the
    per-length rates would also give 50% here — so the case that separates them is
    unequal sample counts, below.
    """
    rows = (
        [_row("PDL1", 80, i == 0) for i in range(10)]
        + [_row("PDL1", 130, i < 9) for i in range(10)]
    )
    scores = benchmark.designability(rows)
    assert scores["PDL1"]["n_samples"] == 20
    assert scores["PDL1"]["n_designable"] == 10
    assert scores["PDL1"]["designability"] == pytest.approx(50.0)


def test_pooling_differs_from_averaging_per_length_rates(benchmark):
    rows = [_row("PDL1", 80, True)] + [_row("PDL1", 130, False) for _ in range(9)]
    scores = benchmark.designability(rows)
    pooled = scores["PDL1"]["designability"]
    per_length = [c["designability"] for c in scores["PDL1"]["per_length"].values()]
    assert pooled == pytest.approx(10.0)
    assert np.mean(per_length) == pytest.approx(50.0)
    assert pooled != pytest.approx(np.mean(per_length))


def test_variants_are_scored_separately(benchmark):
    rows = (
        [_row("PDL1", 80, True, "co_design") for _ in range(4)]
        + [_row("PDL1", 80, False, "pmpnn") for _ in range(4)]
    )
    assert benchmark.designability(rows, variant="co_design")["PDL1"][
        "designability"
    ] == pytest.approx(100.0)
    assert benchmark.designability(rows, variant="pmpnn")["PDL1"][
        "designability"
    ] == pytest.approx(0.0)


def test_summary_table_uses_table4_order(benchmark):
    rows = [_row(name, 80, True) for name in ("VEGFA", "PDL1", "SC2RBD")]
    table = benchmark.summary_table(benchmark.designability(rows))
    header = table.splitlines()[0].split()[1:]
    assert header == ["PDL1", "SC2RBD", "VEGFA"]


# --------------------------------------------------------- input preparation


def test_hotspot_override_refuses_to_union_with_sampled_hotspots():
    """Guard the one way the fixed hotspot set could be contaminated."""
    torch = pytest.importorskip("torch")
    from pxdesign_train.benchmarks.target_prep import apply_hotspots

    feature_dict = {"hotspot": torch.zeros(5)}
    assert apply_hotspots(feature_dict, {("A", 7): 2}, [("A", 7)]) == 1
    assert feature_dict["hotspot"].tolist() == [0, 0, 1, 0, 0]

    feature_dict = {"hotspot": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match="hotspot_force_zero_prob"):
        apply_hotspots(feature_dict, {("A", 7): 2}, [("A", 7)])


def test_hotspot_override_rejects_an_unresolvable_residue():
    torch = pytest.importorskip("torch")
    from pxdesign_train.benchmarks.target_prep import apply_hotspots

    with pytest.raises(KeyError):
        apply_hotspots({"hotspot": torch.zeros(3)}, {("A", 1): 0}, [("A", 42)])


def test_placeholder_binder_is_a_plausible_polymer():
    """Geometry is inert, but parsing still has to accept it."""
    pytest.importorskip("biotite.structure")
    import biotite.structure as struc

    from pxdesign_train.benchmarks.target_prep import make_placeholder_binder

    target = struc.AtomArray(4)
    target.coord = np.array(
        [[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0]], dtype=np.float32
    )
    for i, name in enumerate(("N", "CA", "C", "O")):
        target.chain_id[i] = "A"
        target.res_id[i] = 1
        target.res_name[i] = "GLY"
        target.atom_name[i] = name
        target.element[i] = name[0]

    binder = make_placeholder_binder(target, 30, "Z", anchor=np.array([5.0, 10.0, 0.0]))
    assert binder.array_length() == 30 * 4
    assert set(binder.res_name) == {"GLY"}
    assert list(binder.atom_name[:4]) == ["N", "CA", "C", "O"]

    ca = binder.coord[binder.atom_name == "CA"]
    steps = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    # Consecutive CA within the AF3-style 10 A cutoff, and never coincident.
    assert steps.max() < 10.0
    assert steps.min() > 1.0
    # No two atoms on top of each other, which would break reference geometry.
    distances = np.linalg.norm(binder.coord[:, None] - binder.coord[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 0.5


def test_binder_chain_id_never_collides_with_the_target():
    from pxdesign_train.benchmarks.target_prep import choose_binder_chain_id

    assert choose_binder_chain_id({"A"}) == "B"
    # 1TNF is A/B/C, so the usual 'B' is unavailable.
    assert choose_binder_chain_id({"A", "B", "C"}) not in {"A", "B", "C"}
    assert choose_binder_chain_id({"B"}) != "B"
