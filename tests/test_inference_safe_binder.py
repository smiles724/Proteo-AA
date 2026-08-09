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



def test_tokenizer_assertion_becomes_a_retryable_skip(monkeypatch):
    """Regression for the crash that killed jobs 97437 AND 97705, both at step 150.

    Protenix asserts a GLOBAL invariant in `AtomArrayTokenizer`:
    `len(token_array) == centre_atom_mask.sum()`. Our canonical rebuild designates
    exactly one centre per binder residue (the atom named CA), but Protenix emits
    one token PER ATOM for a non-standard residue — so such a residue wants N
    centres and gets 1, and the tokenizer raises AssertionError.

    AssertionError is not ValueError, so `DesignSourceDataset.__getitem__`'s retry
    loop could not skip it and the DataLoader worker killed the whole run. The
    per-residue guard cannot catch this case (the residue does own a CA, so it has
    exactly one centre); only the tokenizer knows the true token count. It must
    therefore be translated into the retryable `InferenceSafeBinder:` ValueError.
    """
    import pxdesign_train.runner.data as data_mod

    native = _full_atom_complex()
    binder = np.asarray(native.chain_id) == "B"

    class _Exploding:
        def __init__(self, *_a, **_k):
            pass

        def get_token_array(self):
            raise AssertionError("Length of values must match the number of tokens")

    import protenix.data.tokenizer as tok_mod
    monkeypatch.setattr(tok_mod, "AtomArrayTokenizer", _Exploding)

    with pytest.raises(ValueError, match="InferenceSafeBinder") as excinfo:
        data_mod._refeaturize_inference_safe_binder(native, binder)

    msg = str(excinfo.value)
    # Must be recognised as retryable by the crop-retry loop's prefix check.
    assert msg.startswith("InferenceSafeBinder:")
    # And must carry the diagnostic that explains WHY it was skipped.
    assert "AtomArrayTokenizer" in msg
    assert "centre atoms" in msg
    # The original assertion is preserved for debugging.
    assert isinstance(excinfo.value.__cause__, AssertionError)


def test_retry_loop_treats_the_tokenizer_skip_as_retryable():
    """The translated error must actually be caught by `__getitem__`, not just raised."""
    from pxdesign_train.runner.data import DesignSourceDataset

    class _Provider:
        def __init__(self):
            self.n = 50
            self.calls = 0

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            self.calls += 1
            return ("aa", "ta", {}, {}, lambda _a: "B")

    provider = _Provider()
    ds = DesignSourceDataset(provider=provider, source_name="s", max_crop_retries=4)

    calls = {"n": 0}

    def _fail_once_then_work(idx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(
                "InferenceSafeBinder: strict re-tokenisation of the canonicalised "
                "binder was rejected by AtomArrayTokenizer (...). 3 centre atoms"
            )
        return {"sample_id": f"p{idx}"}

    ds._get_one = _fail_once_then_work
    assert ds[7]["sample_id"].startswith("p")
    assert calls["n"] == 2, "the tokenizer skip should have been retried exactly once"
