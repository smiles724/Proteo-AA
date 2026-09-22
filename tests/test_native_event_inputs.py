"""The binder must reach the model as a generated binder does: backbone only.

This is the fix for the leak the acceptance gate found. A deposited complex's
binder chain carries full side chains, and feeding that as a reconstruction
event hands the denoiser the binder's native side-chain coordinates at the
event's noise level -- at sigma 0.429, its deposited geometry with 0.43 A of
jitter. At inference the binder has four atoms per residue and no identity.

Gated on the calibration CIFs, since the point is the behaviour on real
deposited structures rather than on a synthetic file.
"""

from pathlib import Path

import pytest

MANIFEST = Path(
    "/scratch/m000137-pm06/Proteo-AA/pxf/runs/integrated_feedback_v1/data/"
    "calibration_pdb.parquet"
)
needs_data = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="calibration manifest not on this filesystem"
)

BACKBONE = ("N", "CA", "C", "O")


def _row():
    import pandas as pd

    return pd.read_parquet(MANIFEST).iloc[0]


def _prepared(tmp_path, perturb="none"):
    from pxf.bench.native_event_inputs import prepare

    row = _row()
    return row, prepare(
        row.cif_path, row.converted_binder_chain, tmp_path, perturb=perturb
    )


@needs_data
def test_binder_keeps_only_backbone_atoms(tmp_path):
    import gemmi

    row, stats = _prepared(tmp_path)
    structure = gemmi.read_structure(stats["path"])
    binder = next(c for c in structure[0] if c.name == row.converted_binder_chain)
    names = {a.name for residue in binder for a in residue}
    assert names <= set(BACKBONE), f"binder still carries {names - set(BACKBONE)}"
    # Exactly four atoms per residue is what a generated binder has.
    assert sum(len(r) for r in binder) == 4 * len(binder)
    assert stats["sidechain_atoms_removed"] > 0


@needs_data
def test_target_side_chains_are_untouched(tmp_path):
    """`complex_sc` means the target's resolved side chains ARE context."""
    import gemmi

    row, stats = _prepared(tmp_path)
    structure = gemmi.read_structure(stats["path"])
    for chain in structure[0]:
        if chain.name == row.converted_binder_chain:
            continue
        names = {a.name for residue in chain for a in residue}
        assert names - set(BACKBONE), (
            f"target chain {chain.name} lost its side chains; it is legitimate "
            "context and must not be stripped"
        )


@needs_data
def test_displacing_binder_side_chains_cannot_reach_the_model(tmp_path):
    """The leak, closed structurally rather than by tolerance.

    Displacing every binder side chain by 17 A and then stripping them yields
    a BYTE-IDENTICAL file, so there is no numerical argument to have about
    whether the perturbation propagated.
    """
    _row_a, clean = _prepared(tmp_path, "none")
    _row_b, moved = _prepared(tmp_path, "sidechain")
    assert moved["sidechain_atoms_displaced"] > 0, "the test perturbed nothing"
    assert clean["sha256"] == moved["sha256"]


@needs_data
def test_perturbing_binder_identity_does_change_the_input(tmp_path):
    """The identity direction must remain a LIVE test, not a vacuous one."""
    _row_a, clean = _prepared(tmp_path, "none")
    _row_b, renamed = _prepared(tmp_path, "aatype")
    assert renamed["residues_renamed"] > 0
    assert clean["sha256"] != renamed["sha256"]


@needs_data
def test_an_unknown_perturbation_is_refused(tmp_path):
    from pxf.bench.native_event_inputs import prepare

    row = _row()
    with pytest.raises(ValueError, match="unknown perturbation"):
        prepare(row.cif_path, row.converted_binder_chain, tmp_path,
                perturb="nonsense")


@needs_data
def test_a_wrong_binder_chain_is_refused(tmp_path):
    from pxf.bench.native_event_inputs import prepare

    row = _row()
    with pytest.raises(ValueError, match="no residues"):
        prepare(row.cif_path, "ZZ", tmp_path)
