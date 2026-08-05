"""Regression tests for the strict binder train/inference feature contract."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch


# Importing pxdesign_train pulls in Protenix's CUDA-only fused layer norm even
# though these tests exercise data code only.  A name-only stub prevents a
# pointless local CUDA compilation; no fused-layer-norm function is called.
sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)


def _full_atom_complex():
    biotite = pytest.importorskip("biotite.structure")
    AtomArray = biotite.AtomArray

    residues = [
        ("T", 1, "ALA", ("N", "CA", "C", "O", "CB")),
        ("B", 1, "ALA", ("N", "CA", "C", "O", "CB")),
        (
            "B",
            2,
            "TRP",
            ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1"),
        ),
    ]
    n_atom = sum(len(names) for *_prefix, names in residues)
    aa = AtomArray(n_atom)
    aa.coord = np.arange(n_atom * 3, dtype=np.float32).reshape(n_atom, 3) / 10

    annotations = {
        "cano_seq_resname": "",
        "ref_mask": 1,
        "ref_charge": 7,
        "ref_space_uid": 0,
        "mol_id": 0,
        "mol_atom_index": 0,
        "centre_atom_mask": 0,
        "distogram_rep_atom_mask": 0,
    }
    for name, fill in annotations.items():
        dtype = "U3" if name == "cano_seq_resname" else np.int64
        aa.set_annotation(name, np.full(n_atom, fill, dtype=dtype))
    aa.set_annotation("ref_pos", np.zeros((n_atom, 3), dtype=np.float32))

    i = 0
    uid = 10
    for chain, resid, resname, atom_names in residues:
        for atom_name in atom_names:
            aa.chain_id[i] = chain
            aa.res_id[i] = resid
            aa.res_name[i] = resname
            aa.cano_seq_resname[i] = resname
            aa.atom_name[i] = atom_name
            aa.element[i] = (
                "N" if atom_name.startswith("N") else
                "O" if atom_name.startswith("O") else "C"
            )
            aa.ref_space_uid[i] = uid
            aa.mol_id[i] = 0 if chain == "T" else 1
            aa.mol_atom_index[i] = i
            if atom_name == "CA":
                aa.centre_atom_mask[i] = 1
                aa.distogram_rep_atom_mask[i] = 1
            # Deliberately residue-specific native reference data; strict
            # canonicalisation must erase it on binder rows.
            aa.ref_pos[i] = np.asarray([uid, i, -i], dtype=np.float32)
            i += 1
        uid += 17  # native atom-count-dependent gaps must not survive
    return aa


def test_strict_binder_removes_topology_and_canonicalises_metadata():
    from pxdesign_train.runner.data import _canonical_backbone_binder_atom_array

    native = _full_atom_complex()
    binder = np.asarray(native.chain_id) == "B"
    safe, safe_binder = _canonical_backbone_binder_atom_array(native, binder)

    # Target full-atom conditioning is preserved.
    target_names = list(np.asarray(safe.atom_name)[np.asarray(safe.chain_id) == "T"])
    assert target_names == ["N", "CA", "C", "O", "CB"]

    # Every binder residue has exactly the same inference-time topology.
    for resid in (1, 2):
        row = safe_binder & (np.asarray(safe.res_id) == resid)
        assert list(np.asarray(safe.atom_name)[row]) == ["N", "CA", "C", "O"]
        assert set(np.asarray(safe.res_name)[row]) == {"GLY"}
        assert set(np.asarray(safe.cano_seq_resname)[row]) == {"GLY"}
        assert np.all(np.asarray(safe.ref_charge)[row] == 0)

    # ALA and TRP now receive the same canonical metadata for each backbone atom.
    for atom_name in ("N", "CA", "C", "O"):
        row = safe_binder & (np.asarray(safe.atom_name) == atom_name)
        values = np.asarray(safe.ref_pos)[row]
        np.testing.assert_allclose(values[0], values[1])

    # The operation is non-mutating: native side-chain labels remain available
    # for supervision before the strict model input is built.
    assert "CB" in list(np.asarray(native.atom_name)[binder])
    assert "TRP" in set(np.asarray(native.res_name)[binder])


def test_strict_atom_indices_use_the_new_backbone_only_axis():
    from pxdesign_train.runner.data import (
        _canonical_backbone_binder_atom_array,
        _strict_atom_index_features,
    )

    native = _full_atom_complex()
    safe, safe_binder = _canonical_backbone_binder_atom_array(
        native, np.asarray(native.chain_id) == "B"
    )
    # Target has 5 rows; binder residues then have four rows each.
    atom_to_token = torch.tensor([0] * 5 + [1] * 4 + [2] * 4)
    feat = {
        "atom_to_token_idx": atom_to_token,
        "restype": torch.zeros(3, 36),
    }
    idx, centres = _strict_atom_index_features(safe, feat, safe_binder)
    assert idx[0].tolist() == [-1, -1, -1, -1]
    assert idx[1].tolist() == [5, 6, 7, 8]
    assert idx[2].tolist() == [9, 10, 11, 12]
    assert centres.tolist() == [1, 6, 10]

