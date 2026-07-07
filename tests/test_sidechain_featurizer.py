"""Test GT side-chain target extraction in DesignFeaturizer (CCD-free)."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


def _ala_atom_array():
    biotite = pytest.importorskip("biotite.structure")
    names = ["N", "CA", "C", "O", "CB"]
    coords = np.array(
        [[-1.0, 0.0, 0.0],   # N
         [0.0, 0.0, 0.0],    # CA (origin)
         [1.0, 0.5, 0.0],    # C
         [1.2, 1.6, 0.0],    # O
         [0.2, -0.8, 0.9]],  # CB (the side-chain atom)
        dtype=np.float32,
    )
    aa = biotite.AtomArray(length=5)
    aa.coord = coords
    aa.chain_id = np.array(["A"] * 5)
    aa.res_id = np.array([1] * 5)
    aa.res_name = np.array(["ALA"] * 5)
    aa.atom_name = np.array(names)
    aa.set_annotation("distogram_rep_atom_mask", np.array([0, 1, 0, 0, 0]))
    return aa, coords


def test_sidechain_targets_extract_cb():
    from pxdesign_train.data import DesignFeaturizer, DesignSelection
    from pxdesign_train.sidechain.instantiate import MAX_SC, ATOM_NAME_TO_ID
    from pxdesign_train.sidechain.frames import build_frame, to_local

    aa, coords = _ala_atom_array()
    binder = np.ones(5, dtype=bool)
    feat = {"distogram_rep_atom_mask": torch.tensor([0, 1, 0, 0, 0])}
    fz = DesignFeaturizer(DesignSelection(binder_atom_mask=binder, compute_sidechain=True))

    out = fz._compute_sidechain_targets(aa, feat, binder)

    assert out["sc_atom_mask"].shape == (1, MAX_SC)
    assert bool(out["sc_atom_mask"][0, 0]) is True          # CB present
    assert int(out["sc_atom_mask"][0, 1:].sum()) == 0        # ALA has only CB
    assert int(out["sc_atom_name_ids"][0, 0]) == ATOM_NAME_TO_ID["CB"]

    # Independent local-frame computation of CB must match the extraction.
    n = torch.from_numpy(coords[0])[None]
    ca = torch.from_numpy(coords[1])[None]
    c = torch.from_numpy(coords[2])[None]
    R, t = build_frame(n, ca, c)
    cb_local = to_local(torch.from_numpy(coords[4])[None, None], R, t)[0, 0]
    assert torch.allclose(out["sc_gt_local"][0, 0], cb_local, atol=1e-5)


def test_target_tokens_get_no_sidechain():
    """Non-binder (target) residues must have an all-False side-chain mask."""
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, _ = _ala_atom_array()
    binder = np.zeros(5, dtype=bool)  # nothing is binder
    # DesignSelection needs a non-empty binder elsewhere; use a mask flag but
    # call the extraction directly with an all-False binder.
    fz = DesignFeaturizer(DesignSelection(binder_atom_mask=np.ones(5, bool),
                                          compute_sidechain=True))
    feat = {"distogram_rep_atom_mask": torch.tensor([0, 1, 0, 0, 0])}
    out = fz._compute_sidechain_targets(aa, feat, binder)
    assert int(out["sc_atom_mask"].sum()) == 0
