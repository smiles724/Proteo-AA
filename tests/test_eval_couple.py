"""The held-out coupling evaluation.

Two things here must not be wrong. The verdict logic, because a comparison that
reports "ok" on an adapter that degrades packing is the failure mode the script
exists to catch. And the atom37 composition, because writing packed side chains
into the wrong slots would produce plausible, entirely meaningless numbers.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pxf import atom37  # noqa: E402
from pxf.eval import couple as ev  # noqa: E402

BASE = {
    "symmetry_rmsd": 1.90,
    "rotamer_recovery": 0.62,
    "chi_recovery_20deg": 0.60,
    "chi1_accuracy_20deg": 0.74,
    "lddt_sc_sc": 0.80,
    "lddt_sc_env": 0.84,
    "bad_bond_fraction": 0.004,
    "rotamer_outlier_fraction_40deg": 0.22,
}


def arms(**coupled_overrides):
    """Two arms that are identical except where the test says otherwise."""
    tuned = dict(BASE)
    tuned.update(coupled_overrides)
    return {"uncoupled": dict(BASE), "coupled": tuned}


# --- the verdict -----------------------------------------------------------


def test_an_inert_adapter_is_neither_a_regression_nor_an_improvement():
    both = arms()
    assert ev.regressions(both) == []
    assert ev.improvements(both) == []


def test_worse_rmsd_is_flagged_even_though_the_number_went_down():
    # symmetry_rmsd is the one headline metric where lower is better, so a
    # naive "delta > 0 means better" would get this exactly backwards.
    assert [k for k, _, _ in ev.regressions(arms(symmetry_rmsd=2.10))] == ["symmetry_rmsd"]
    assert ev.improvements(arms(symmetry_rmsd=2.10)) == []


def test_better_rmsd_is_an_improvement():
    assert [k for k, _, _ in ev.improvements(arms(symmetry_rmsd=1.70))] == ["symmetry_rmsd"]
    assert ev.regressions(arms(symmetry_rmsd=1.70)) == []


def test_worse_recovery_is_flagged():
    assert [k for k, _, _ in ev.regressions(arms(rotamer_recovery=0.55))] == [
        "rotamer_recovery"
    ]


def test_every_headline_metric_can_trigger_the_verdict():
    for key in ev.HEADLINE:
        worse = BASE[key] + (0.5 if key in ev.LOWER_IS_BETTER else -0.5)
        flagged = [k for k, _, _ in ev.regressions(arms(**{key: worse}))]
        assert flagged == [key], f"{key} did not trigger a regression"


def test_improving_one_metric_cannot_mask_regressing_another():
    both = arms(symmetry_rmsd=1.50, rotamer_recovery=0.40)
    assert [k for k, _, _ in ev.regressions(both)] == ["rotamer_recovery"]
    assert [k for k, _, _ in ev.improvements(both)] == ["symmetry_rmsd"]


def test_float_noise_is_not_adjudicated():
    nudge = ev.REGRESSION_TOLERANCE / 10
    both = arms(rotamer_recovery=BASE["rotamer_recovery"] - nudge)
    assert ev.regressions(both) == []


def test_a_non_headline_regression_is_not_the_verdict():
    # Reported in the table, but the verdict is the four headline metrics.
    both = arms(lddt_sc_env=0.10)
    assert ev.regressions(both) == []
    assert "lddt_sc_env" in ev.delta_table(both)


def test_direction_conventions_match_the_protenix_evaluator():
    """The two reports must not disagree about what "better" means."""
    protenix = pytest.importorskip("eval_protenix_sidechain")
    assert set(ev.HEADLINE) == set(protenix.HEADLINE)
    assert ev.REGRESSION_TOLERANCE == protenix.REGRESSION_TOLERANCE
    shared = set(ev.LOWER_IS_BETTER) & {k for g in ev.REPORT.values() for k in g}
    assert shared <= set(protenix.LOWER_IS_BETTER)
    # Every metric this script reports has a stated direction somewhere.
    for group in ev.REPORT.values():
        for key in group:
            assert (
                key in ev.LOWER_IS_BETTER
                or key in protenix.REPORT["rotamer recovery"]
                or key in protenix.REPORT["lddt"]
                or key in protenix.REPORT["rmsd"]
                or key in protenix.REPORT["covalent failures"]
            )


def test_delta_table_skips_metrics_absent_from_one_arm():
    both = arms()
    del both["coupled"]["lddt_sc_env"]
    assert "lddt_sc_env" not in ev.delta_table(both)
    assert "lddt_sc_sc" in ev.delta_table(both)


# --- the sigma sweep -------------------------------------------------------


def make_schedule(**kwargs):
    from pxf.couple import schedule

    return schedule.from_config(
        {"mode": "trajectory", "sigma_min": 0.01, "sigma_max": 5.0, "n_step": 400},
        **kwargs,
    )


def test_the_sweep_is_deterministic():
    s = make_schedule()
    assert ev.sweep_sigmas(s, 5) == ev.sweep_sigmas(s, 5)


def test_the_sweep_stays_inside_the_training_window():
    s = make_schedule()
    values = ev.sweep_sigmas(s, 7)
    assert len(values) == 7
    assert values == sorted(values)
    assert all(s.sigma_min <= v <= s.sigma_max for v in values)


def test_the_sweep_spans_the_window_rather_than_clustering():
    s = make_schedule()
    values = ev.sweep_sigmas(s, 5)
    first, last = s.window_steps()
    window = s.trajectory()[first : last + 1]
    assert values[0] == pytest.approx(float(window.min()), rel=1e-6)
    assert values[-1] == pytest.approx(float(window.max()), rel=1e-6)


def test_a_single_point_is_allowed_and_is_interior():
    s = make_schedule()
    (only,) = ev.sweep_sigmas(s, 1)
    assert s.sigma_min <= only <= s.sigma_max


def test_fixed_mode_sweeps_the_one_value_it_has():
    from pxf.couple import schedule

    s = schedule.from_config({"mode": "fixed", "sigma": 1.5})
    assert ev.sweep_sigmas(s, 4) == [1.5]


def test_zero_sweep_points_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        ev.sweep_sigmas(make_schedule(), 0)


# --- seeding ---------------------------------------------------------------


def test_the_seed_depends_on_target_and_sigma_but_not_on_loop_position():
    a = ev.target_seed(0, "AF-Q9X0-F1", 0.429)
    assert a == ev.target_seed(0, "AF-Q9X0-F1", 0.429)
    assert a != ev.target_seed(0, "AF-Q9X0-F1", 1.642)
    assert a != ev.target_seed(0, "AF-OTHER-F1", 0.429)
    assert a != ev.target_seed(1, "AF-Q9X0-F1", 0.429)
    assert a != ev.target_seed(0, "AF-Q9X0-F1", 0.429, replicate=1)


def test_the_seed_is_stable_across_processes():
    """The property the previous `hash()` implementation did not have.

    Python salts `hash` for str per interpreter, so two jobs -- a run and its
    shuffled control, or a rerun of the same config -- drew different backbone
    noise and different packing trajectories while appearing to share a seed.
    Arms stayed paired inside one process, so the bug was invisible in any
    single run's delta and only corrupted comparisons *between* runs.

    Pinned against literals rather than a second call: a within-process
    comparison cannot detect per-interpreter salting.
    """
    assert ev.target_seed(0, "AF-P81613-F1-model_v4", 0.010) == 1588898999
    assert ev.target_seed(0, "AF-P81613-F1-model_v4", 0.429) == 2130388763
    assert ev.target_seed(0, "AF-P81613-F1-model_v4", 4.881) == 1734635334


def test_the_seed_keys_on_the_sigma_value_not_its_sweep_index():
    """So a 3-point and a 5-point sweep agree wherever they share a sigma."""
    five = ev.sweep_sigmas(make_schedule(), 5)
    three = ev.sweep_sigmas(make_schedule(), 3)
    shared = sorted(set(five) & set(three))
    assert shared, "the sweeps share no sigma; this test proves nothing"
    # At least one shared sigma must sit at a different position in the two
    # sweeps, or keying on the value rather than the index would be untested.
    moved = [s for s in shared if five.index(s) != three.index(s)]
    assert moved, "no shared sigma changed index; the distinction is untested"
    for sigma in moved:
        assert ev.target_seed(0, "t", sigma) == ev.target_seed(0, "t", sigma)
        # An index-keyed seed would differ here; a value-keyed one does not.
        assert ev.target_seed(0, "t", sigma) != ev.target_seed(
            0, "t", float(five.index(sigma))
        )


def test_seeds_are_valid_torch_seeds():
    for name in ("a", "bb", "AF-X-F1"):
        for sigma in (0.01, 0.429, 4.881, 160.0):
            seed = ev.target_seed(7, name, sigma)
            assert 0 <= seed < 2**31 - 1
            torch.Generator().manual_seed(seed)


# --- atom37 composition ----------------------------------------------------


def fake_cycle(length, *, fill_backbone=1.0, fill_sidechain=9.0):
    backbone = torch.full((1, length, 37, 3), float(fill_backbone))
    sidechains = torch.full((1, length, 33, 3), float(fill_sidechain))
    return types.SimpleNamespace(bb0_dense=backbone, sidechains=sidechains)


def test_packed_sidechains_land_in_the_non_backbone_slots():
    rc = pytest.importorskip("fampnn.data.residue_constants")
    length = 6
    aatype = torch.zeros(length, dtype=torch.long)  # all ALA
    pred, _mask = ev.predicted_atom37(fake_cycle(length), aatype, rc)
    for slot in atom37.BACKBONE_SLOTS:
        assert torch.all(pred[:, slot] == 1.0), f"backbone slot {slot} was overwritten"
    for slot in atom37.SIDECHAIN_SLOTS:
        assert torch.all(pred[:, slot] == 9.0), f"side-chain slot {slot} was not written"


def test_the_generated_mask_is_the_residue_type_mask():
    rc = pytest.importorskip("fampnn.data.residue_constants")
    # GLY (7) has no side chain beyond CB; TRP (18) has the most atoms.
    aatype = torch.tensor([7, 18], dtype=torch.long)
    _pred, mask = ev.predicted_atom37(fake_cycle(2), aatype, rc)
    table = torch.as_tensor(rc.restype_atom37_mask)
    assert torch.equal(mask, table[aatype].bool())
    assert int(mask[0].sum()) < int(mask[1].sum())


def test_a_cycle_that_never_packed_is_refused():
    rc = pytest.importorskip("fampnn.data.residue_constants")
    cycle = fake_cycle(3)
    cycle.sidechains = None
    with pytest.raises(ValueError, match="nothing to score"):
        ev.predicted_atom37(cycle, torch.zeros(3, dtype=torch.long), rc)


def test_wrong_sidechain_slot_count_is_refused():
    rc = pytest.importorskip("fampnn.data.residue_constants")
    cycle = fake_cycle(3)
    cycle.sidechains = torch.zeros(1, 3, 32, 3)
    with pytest.raises(ValueError, match="32 slots"):
        ev.predicted_atom37(cycle, torch.zeros(3, dtype=torch.long), rc)


def test_native_masks_drop_atoms_recorded_as_missing():
    rc = pytest.importorskip("fampnn.data.residue_constants")
    aatype = torch.tensor([18], dtype=torch.long)  # TRP: plenty of real slots
    missing = torch.zeros(1, 37)
    present_slot = int(
        torch.nonzero(torch.as_tensor(rc.restype_atom37_mask)[18]).reshape(-1)[-1]
    )
    missing[0, present_slot] = 1.0
    native = {
        "x": torch.zeros(1, 37, 3),
        "aatype": aatype,
        "missing_atom_mask": missing,
    }
    _coords, mask = ev.native_atom37(native, rc)
    assert not bool(mask[0, present_slot])
    assert int(mask.sum()) == int(torch.as_tensor(rc.restype_atom37_mask)[18].sum()) - 1


# --- alignment -------------------------------------------------------------


def test_a_length_disagreement_between_the_two_parsers_is_an_error():
    with pytest.raises(ValueError, match="misaligned"):
        ev.check_alignment(
            "104l", torch.zeros(328, dtype=torch.long), torch.zeros(166, dtype=torch.long)
        )


def test_a_sequence_disagreement_at_equal_length_is_an_error():
    """The count can match while the residues do not; that must not pass."""
    native = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    featurized = torch.tensor([0, 1, 5, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="differs from the file"):
        ev.check_alignment("x", native, featurized)


def test_matching_parses_are_accepted():
    same = torch.tensor([3, 1, 4, 1, 5], dtype=torch.long)
    ev.check_alignment("x", same, same.clone())


# --- frames ----------------------------------------------------------------


def random_structure(length, *, seed=0):
    """A structure with plausible, non-degenerate backbone geometry."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(length, 37, 3, generator=g) * 2.0
    # Spread the residues out so N/CA/C frames are well conditioned.
    coords += torch.arange(length, dtype=torch.float32)[:, None, None] * 4.0
    return coords


def rigid_move(coords, *, seed=1):
    """Rotate and translate a whole structure -- a change of pose, not geometry."""
    g = torch.Generator().manual_seed(seed)
    a = torch.linalg.qr(torch.randn(3, 3, generator=g))[0]
    if torch.det(a) < 0:
        a = a @ torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    return coords @ a.T + torch.tensor([10.0, -5.0, 3.0])


def canonical_or_skip():
    canonical = pytest.importorskip("pxf.eval.canonical")
    try:
        return canonical.load()
    except Exception as error:  # pragma: no cover - depends on the Proteo-AA tree
        pytest.skip(f"Proteo-AA metrics unavailable: {error}")


def test_a_rigidly_moved_prediction_scores_as_its_own_geometry():
    """The whole reason frames are transferred: global pose must not matter.

    A prediction that is the native structure in a different coordinate frame
    has *perfect* side chains. Scoring it against native frames directly would
    call it catastrophically wrong, which is the 20 A artefact this guards.
    """
    canonical = canonical_or_skip()
    native = random_structure(8)
    moved = rigid_move(native)
    placed = ev.place_on_native_backbone(moved, native, canonical)
    slots = list(atom37.SIDECHAIN_SLOTS)
    assert torch.allclose(placed[:, slots, :], native[:, slots, :], atol=1e-4)


def test_transfer_is_a_no_op_when_the_backbones_already_agree():
    canonical = canonical_or_skip()
    native = random_structure(6)
    pred = native.clone()
    pred[:, list(atom37.SIDECHAIN_SLOTS), :] += 0.7  # a genuinely different packing
    placed = ev.place_on_native_backbone(pred, native, canonical)
    assert torch.allclose(placed, pred, atol=1e-4)


def test_transfer_keeps_the_native_backbone():
    canonical = canonical_or_skip()
    native = random_structure(5)
    placed = ev.place_on_native_backbone(rigid_move(native), native, canonical)
    for slot in atom37.BACKBONE_SLOTS:
        assert torch.allclose(placed[:, slot, :], native[:, slot, :], atol=1e-5)


def test_backbone_rmsd_is_zero_for_a_rigidly_moved_copy():
    native = random_structure(10)
    assert ev.backbone_rmsd(rigid_move(native), native) == pytest.approx(0.0, abs=1e-4)


def test_backbone_rmsd_grows_with_real_displacement():
    native = random_structure(10)
    nudged = native.clone()
    nudged[:, list(atom37.BACKBONE_SLOTS), :] += torch.randn(
        10, len(atom37.BACKBONE_SLOTS), 3, generator=torch.Generator().manual_seed(3)
    )
    assert ev.backbone_rmsd(nudged, native) > 0.3


# --- donor reshaping for the shuffled control -------------------------------


def test_a_longer_donor_is_cropped_and_a_shorter_one_tiled():
    source = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    assert ev.donor_a_token(source, 6).shape == (1, 6, 2)
    assert ev.donor_a_token(source, 4).shape == (1, 4, 2)
    assert ev.donor_a_token(source, 10).shape == (1, 10, 2)


def test_an_equal_length_donor_is_returned_untouched():
    source = torch.randn(1, 7, 3)
    assert ev.donor_a_token(source, 7) is source


def test_cropping_keeps_real_donor_values_rather_than_padding():
    """Zero-padding would make the control partly an ablation, not a swap."""
    source = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    out = ev.donor_a_token(source, 3)
    assert torch.equal(out, source[:, :3, :])
    tiled = ev.donor_a_token(source, 8)
    assert torch.equal(tiled[:, :5, :], source)
    assert int((tiled == 0).sum()) == int((source == 0).sum()) * 2  # no new zeros


def test_a_donor_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match=r"\[B, L, C\]"):
        ev.donor_a_token(torch.randn(5, 3), 5)
