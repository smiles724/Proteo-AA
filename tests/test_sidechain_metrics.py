"""The metrics adapter must hand the canonical functions exactly what they expect.

These metrics are Proteo-AA's own implementations, reused so that numbers stay
comparable to the earlier side-chain runs. What is tested here is therefore the
*translation* from AF2 atom37 into their 10-slot local-frame convention: if a
slot were misaddressed the numbers would still look plausible, so the identity
case (native scored against itself) is the load-bearing check.
"""

import pytest
import torch

from pxf import atom37
from pxf.eval import canonical as canonical_module
from pxf.eval.sidechain_metrics import aggregate, score, slot_tables

TARGET = "fampnn/data/casp14/pdbs/T1031.pdb"
SIDECHAIN = list(atom37.SIDECHAIN_SLOTS)


@pytest.fixture(scope="module")
def canonical():
    return canonical_module.load()


@pytest.fixture(scope="module")
def native():
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf.provenance import repo_root

    single = process_single_pdb(load_feats_from_pdb(str(repo_root() / TARGET)))
    return single["x"], single["atom_mask"], single["aatype"].long()


def test_metrics_are_loaded_not_reimplemented(canonical):
    record = canonical.record()
    assert record["reimplemented"] is False
    assert record["source"] == "proteo-aa/pxdesign_train.sidechain"
    assert (canonical.root / "pxdesign_train" / "sidechain" / "metrics.py").is_file()


def test_loading_does_not_pull_in_the_training_model_stack(canonical):
    # pxdesign_train/__init__.py would build the PXDesign/Protenix training model.
    import sys

    assert "pxdesign_train" in sys.modules
    assert not hasattr(sys.modules["pxdesign_train"], "ProtenixDesignTrain")


def test_missing_metrics_root_is_reported(tmp_path):
    with pytest.raises(ValueError, match="No Proteo-AA side-chain metrics"):
        canonical_module.load(tmp_path)


def test_slot_table_addresses_the_right_atoms(canonical):
    table, valid = slot_tables(canonical)
    assert table.shape == (20, canonical.instantiate.MAX_SC)
    for index, name3 in enumerate(canonical.instantiate.STD_AA_3):
        expected = canonical.instantiate.sidechain_atoms(name3)
        got = [
            atom37.ATOM37[slot]
            for slot, ok in zip(table[index].tolist(), valid[index].tolist())
            if ok
        ]
        assert got == expected, name3
    # Glycine has no side chain; tryptophan fills all ten slots.
    assert not valid[atom37.AA_ORDER.index("G")].any()
    assert valid[atom37.AA_ORDER.index("W")].all()


def test_native_scored_against_itself_is_perfect(native, canonical):
    coords, mask, aatype = native
    _, summary = score(coords, mask, coords, mask, aatype, canonical=canonical)
    assert summary["symmetry_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert summary["lddt_sc_sc"] == pytest.approx(1.0, abs=1e-6)
    assert summary["lddt_sc_env"] == pytest.approx(1.0, abs=1e-6)
    assert summary["chi_recovery_20deg"] == pytest.approx(1.0)
    assert summary["chi_recovery_40deg"] == pytest.approx(1.0)
    assert summary["rotamer_recovery"] == pytest.approx(1.0)
    # A crystal structure deviates slightly from ideal covalent geometry, but
    # nothing should cross the 0.2 A "bad bond" threshold.
    assert summary["bad_bond_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert summary["bond_mae"] < 0.05


def test_perturbing_side_chains_moves_every_metric(native, canonical):
    coords, mask, aatype = native
    torch.manual_seed(0)
    noisy = coords.clone()
    noisy[:, SIDECHAIN] += 0.5 * torch.randn_like(noisy[:, SIDECHAIN])
    _, summary = score(noisy, mask, coords, mask, aatype, canonical=canonical)
    assert summary["symmetry_rmsd"] > 0.5
    assert summary["lddt_sc_sc"] < 0.95
    assert summary["lddt_sc_env"] < 0.95
    assert summary["chi_recovery_20deg"] < 0.8
    assert summary["bad_bond_fraction"] > 0.1


def test_perturbing_only_the_backbone_leaves_side_chain_rmsd_alone(native, canonical):
    """The metrics are about side chains; a shared-backbone shift is not their job."""
    coords, mask, aatype = native
    moved = coords.clone() + torch.tensor([10.0, 0.0, 0.0])
    # Reference and prediction are rigidly translated together, so the
    # frame-relative comparison is unchanged.
    _, summary = score(moved, mask, moved, mask, aatype, canonical=canonical)
    assert summary["symmetry_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert summary["lddt_sc_sc"] == pytest.approx(1.0, abs=1e-6)


def test_residue_mask_restricts_scoring(native, canonical):
    coords, mask, aatype = native
    length = coords.shape[0]
    half = torch.zeros(length, dtype=torch.bool)
    half[: length // 2] = True
    _, whole = score(coords, mask, coords, mask, aatype, canonical=canonical)
    _, part = score(
        coords, mask, coords, mask, aatype, canonical=canonical, residue_mask=half
    )
    assert part["scored_residues"] == length // 2 < whole["scored_residues"]
    assert float(part["observed_atoms"]) < float(whole["observed_atoms"])


def test_non_canonical_residues_are_refused(native, canonical):
    coords, mask, aatype = native
    broken = aatype.clone()
    broken[3] = atom37.UNKNOWN_AA_INDEX
    with pytest.raises(ValueError, match="canonical residue types only"):
        score(coords, mask, coords, mask, broken, canonical=canonical)


def test_aggregate_is_atom_weighted_not_a_mean_of_means(native, canonical):
    """A short target must not count as much as a long one."""
    coords, mask, aatype = native
    length = coords.shape[0]
    torch.manual_seed(1)
    noisy = coords.clone()
    noisy[:, SIDECHAIN] += 0.4 * torch.randn_like(noisy[:, SIDECHAIN])

    small = torch.zeros(length, dtype=torch.bool)
    small[:8] = True
    large = torch.zeros(length, dtype=torch.bool)
    large[8:] = True
    counts_small, summary_small = score(
        noisy, mask, coords, mask, aatype, canonical=canonical, residue_mask=small
    )
    counts_large, summary_large = score(
        noisy, mask, coords, mask, aatype, canonical=canonical, residue_mask=large
    )
    combined = aggregate([counts_small, counts_large], canonical=canonical)

    assert combined["n_targets"] == 2
    # The pooled RMSD must sit within the two, and near the larger group.
    low, high = sorted((summary_small["symmetry_rmsd"], summary_large["symmetry_rmsd"]))
    assert low - 1e-6 <= combined["symmetry_rmsd"] <= high + 1e-6
    assert abs(combined["symmetry_rmsd"] - summary_large["symmetry_rmsd"]) < abs(
        combined["symmetry_rmsd"] - summary_small["symmetry_rmsd"]
    )
    # Counts are additive; ratios are formed once at the end.
    assert float(combined["observed_atoms"]) == pytest.approx(
        float(counts_small["observed_atoms"]) + float(counts_large["observed_atoms"])
    )


def test_aggregate_pools_lddt_by_pair_count(native, canonical):
    coords, mask, aatype = native
    counts, summary = score(coords, mask, coords, mask, aatype, canonical=canonical)
    combined = aggregate([counts, counts], canonical=canonical)
    assert combined["lddt_sc_sc"] == pytest.approx(summary["lddt_sc_sc"], abs=1e-6)
    assert combined["n_pairs_sc_sc"] == pytest.approx(2 * float(counts["lddt_sc_sc_pairs"]))


def test_aggregate_needs_at_least_one_target(canonical):
    with pytest.raises(ValueError, match="No targets"):
        aggregate([], canonical=canonical)


# --- lDDT aggregation over degenerate targets -------------------------------


def _lddt_counts(pairs_sc, lddt_sc):
    """One target's lDDT contribution, in the form ``score`` emits."""
    import torch

    from pxf.eval.sidechain_metrics import LDDT_KEYS

    # summarize_metrics needs the reference-free block; the optional
    # observed_atoms branch is left out because these tests are about lDDT.
    counts = {
        "generated_atoms": torch.tensor(10.0),
        "chemical_atoms": torch.tensor(10.0),
        "bond_abs_error_sum": torch.tensor(0.0),
        "bond_count": torch.tensor(10.0),
        "bad_bond_count": torch.tensor(0.0),
    }
    for key in LDDT_KEYS:
        counts[f"{key}_pairs"] = torch.tensor(float(pairs_sc))
        counts[f"{key}_sum"] = torch.tensor(
            float(lddt_sc) * float(pairs_sc) if pairs_sc else 0.0
        )
    return counts


def test_a_target_with_no_pairs_does_not_nan_the_dataset_lddt():
    """One degenerate target must not destroy a headline metric for all of them.

    A single supervised residue has no side-chain neighbour, so its lDDT is
    0/0 = nan. Weighted by its zero pair count that is nan, not 0, and summing
    it turns the dataset figure into nan. On recentPDB 10 of 1,582 targets did
    exactly this and took `lddt_sc_sc` with them.
    """
    import math

    from pxf.eval.sidechain_metrics import aggregate

    canonical = pytest.importorskip("pxf.eval.canonical").load()
    good = [_lddt_counts(1000, 0.90), _lddt_counts(1000, 0.80)]
    degenerate = _lddt_counts(0, float("nan"))
    summary = aggregate(good + [degenerate], canonical=canonical)
    assert not math.isnan(summary["lddt_sc_sc"])
    assert summary["lddt_sc_sc"] == pytest.approx(0.85, abs=1e-6)
    assert summary["n_pairs_sc_sc"] == 2000.0
    assert summary["n_targets_without_pairs_sc_sc"] == 1


def test_pair_weighting_is_by_pair_count_not_target_count():
    from pxf.eval.sidechain_metrics import aggregate

    canonical = pytest.importorskip("pxf.eval.canonical").load()
    summary = aggregate(
        [_lddt_counts(3000, 0.90), _lddt_counts(1000, 0.50)], canonical=canonical
    )
    # (3000*0.90 + 1000*0.50) / 4000 = 0.80 weighted;
    # 0.70 would be the unweighted per-target mean.
    assert summary["lddt_sc_sc"] == pytest.approx(0.80, abs=1e-6)
