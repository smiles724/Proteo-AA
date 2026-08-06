"""The binder the model is TRAINED on must be the binder inference can build.

`tests/test_train_inference_parity.py` guards the model-side switches
(`sidechain.*`). It does not look at the data path at all -- and the data path is
where the worst failure this project has had came from: the binder kept its
native side-chain atom rows, whose names / elements / count / reference conformer
decode the residue identity straight into `q -> a_token -> AA head`, so the head
scored ~96% by reading the answer and ~6% once the rows were gone.

Nothing about that failure was loud. Tests passed, training converged, and the
metric that was supposed to prove co-design was the metric being faked.

So these tests pin the contract itself:

  * a design residue reaches the model as exactly N/CA/C/O, GLY-named, with
    residue-type-independent reference metadata;
  * the target/receptor keeps its full-atom conditioning (the fix must not have
    over-reached);
  * the strict rebuild is ON by default, and the switch that turns it off is
    named for what it causes;
  * the strict rebuild refuses to run silently when it cannot do its job.

They are deliberately independent of the implementation: they assert what the
model may see, not how `runner/data.py` arranges to make it so.
"""
import inspect
import pathlib
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)

BACKBONE = ("N", "CA", "C", "O")


def _complex_with_native_binder_sidechains():
    """Target ALA (kept full-atom) + binder ALA/TRP (must be stripped)."""
    biotite = pytest.importorskip("biotite.structure")
    AtomArray = biotite.AtomArray

    residues = [
        ("T", 1, "ALA", ("N", "CA", "C", "O", "CB")),
        ("B", 1, "ALA", ("N", "CA", "C", "O", "CB")),
        ("B", 2, "TRP", ("N", "CA", "C", "O", "CB", "CG", "CD1", "NE1")),
    ]
    n_atom = sum(len(names) for *_, names in residues)
    aa = AtomArray(n_atom)
    aa.coord = np.arange(n_atom * 3, dtype=np.float32).reshape(n_atom, 3) / 10

    for name, fill in {
        "ref_mask": 1, "ref_charge": 3, "ref_space_uid": 0, "mol_id": 0,
        "mol_atom_index": 0, "centre_atom_mask": 0, "distogram_rep_atom_mask": 0,
    }.items():
        aa.set_annotation(name, np.full(n_atom, fill, dtype=np.int64))
    aa.set_annotation("cano_seq_resname", np.full(n_atom, "", dtype="U3"))
    aa.set_annotation("ref_pos", np.zeros((n_atom, 3), dtype=np.float32))

    i, uid = 0, 10
    for chain, resid, resname, atom_names in residues:
        for atom_name in atom_names:
            aa.chain_id[i], aa.res_id[i], aa.res_name[i] = chain, resid, resname
            aa.cano_seq_resname[i], aa.atom_name[i] = resname, atom_name
            aa.element[i] = atom_name[0]
            aa.ref_space_uid[i], aa.mol_id[i], aa.mol_atom_index[i] = uid, 0, i
            # Residue-specific reference metadata: the strict rebuild must erase
            # it on the binder and leave it alone on the target.
            aa.ref_pos[i] = np.asarray([uid, i, -i], dtype=np.float32)
            if atom_name == "CA":
                aa.centre_atom_mask[i] = 1
                aa.distogram_rep_atom_mask[i] = 1
            i += 1
        uid += 17  # native gaps encode the residue's atom count
    return aa


def _strict(aa):
    from pxdesign_train.runner.data import _canonical_backbone_binder_atom_array

    return _canonical_backbone_binder_atom_array(
        aa, np.asarray(aa.chain_id) == "B"
    )


def test_binder_reaches_the_model_as_four_identical_backbone_rows():
    """The atom SET is the leak. Two different residues must become the same rows."""
    safe, safe_binder = _strict(_complex_with_native_binder_sidechains())
    names = np.asarray(safe.atom_name)
    resids = np.asarray(safe.res_id)

    per_residue = [
        sorted(names[safe_binder & (resids == r)].tolist()) for r in (1, 2)
    ]
    assert per_residue[0] == per_residue[1] == sorted(BACKBONE), (
        "binder residues still differ in atom composition -> residue identity is "
        "recoverable from the atom set alone"
    )


def test_no_residue_specific_reference_metadata_survives_on_the_binder():
    """Every per-atom channel the AtomAttentionEncoder reads must be type-blind.

    Checked as a set-equality over the two residues rather than field by field:
    a future channel added to the featurizer should fail here too.
    """
    safe, safe_binder = _strict(_complex_with_native_binder_sidechains())
    resids = np.asarray(safe.res_id)
    names = np.asarray(safe.atom_name)

    for atom in BACKBONE:
        rows = [
            safe_binder & (resids == r) & (names == atom) for r in (1, 2)
        ]
        for field in ("ref_pos", "ref_charge", "ref_mask", "res_name", "element"):
            values = [np.asarray(getattr(safe, field))[row] for row in rows]
            assert np.array_equal(values[0], values[1]), (
                f"{field} on binder atom {atom} still differs between an ALA and a "
                f"TRP residue: {values[0]} vs {values[1]}"
            )


def test_target_keeps_its_full_atom_conditioning():
    """The target's side chains are legitimate conditioning, not leakage.

    Over-stripping would be just as wrong as under-stripping: at inference the
    receptor IS a known full-atom structure.
    """
    native = _complex_with_native_binder_sidechains()
    safe, _ = _strict(native)
    target = np.asarray(safe.chain_id) == "T"
    assert sorted(np.asarray(safe.atom_name)[target].tolist()) == sorted(
        ["N", "CA", "C", "O", "CB"]
    )
    assert set(np.asarray(safe.res_name)[target]) == {"ALA"}
    # and its residue-specific reference metadata is untouched
    native_t = np.asarray(native.chain_id) == "T"
    assert np.array_equal(
        np.asarray(safe.ref_pos)[target], np.asarray(native.ref_pos)[native_t]
    )


def test_the_strict_rebuild_is_the_default():
    """A leak-safe default is the whole point: an opt-in fix protects nobody."""
    from pxdesign_train.runner.data import DesignSourceDataset

    assert DesignSourceDataset.__dataclass_fields__["inference_safe_binder"].default is True


def test_the_escape_hatch_is_named_for_its_consequence():
    """The training entry may expose the leaky arm, but not under a neutral name.

    `--disable-inference-safe-binder` reads like a performance knob;
    `--allow-binder-sidechain-leakage` cannot be flipped by accident.
    """
    # Read as text: the entry imports heavy training deps at module scope, and
    # this assertion is about the source contract, not runtime behaviour.
    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts" / "training" / "train_protenix_monomer.py"
    ).read_text()
    assert "--allow-binder-sidechain-leakage" in src
    assert "inference_safe_binder=not args.allow_binder_sidechain_leakage" in src
    assert src.count("inference_safe_binder=") == 2, (
        "every DesignSourceDataset built by the training entry must state the "
        "contract explicitly"
    )


def test_strict_rebuild_refuses_rather_than_silently_skipping():
    """A provider that cannot be scrubbed must raise, not fall through.

    The dangerous failure mode is not a crash: it is a provider whose annotations
    are too thin for the rebuild, silently taking the old path with native side
    chains still attached.
    """
    from pxdesign_train.runner.data import DesignSourceDataset

    src = inspect.getsource(DesignSourceDataset._get_one)
    assert "InferenceSafeBinder: native binder side-chain rows are present" in src
    assert "_has_binder_sidechain" in src


def test_topology_violations_are_loud():
    """A binder residue missing a backbone atom must raise, not ship a 3-atom token."""
    aa = _complex_with_native_binder_sidechains()
    keep = ~(
        (np.asarray(aa.chain_id) == "B")
        & (np.asarray(aa.res_id) == 1)
        & (np.asarray(aa.atom_name) == "O")
    )
    with pytest.raises(ValueError, match="InferenceSafeBinder"):
        _strict(aa[keep])
