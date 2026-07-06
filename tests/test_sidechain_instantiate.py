"""Tests for dynamic side-chain atom instantiation (no ghost atoms)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    sidechain_atoms,
    sidechain_mask,
)


def test_ala_is_cb_only():
    assert sidechain_atoms("ALA") == ["CB"]


def test_gly_has_none():
    assert sidechain_atoms("GLY") == []


def test_phe_ring_no_backbone_no_ghost():
    a = set(sidechain_atoms("PHE"))
    assert {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"} <= a
    assert not ({"N", "CA", "C", "O"} & a)  # no backbone
    assert "DMY" not in a and "OXT" not in a  # no ghost / no terminal-O


def test_lowercase_ok():
    assert sidechain_atoms("phe") == sidechain_atoms("PHE")


def test_max_sc_is_trp():
    assert MAX_SC == len(sidechain_atoms("TRP")) == 10


def test_mask_shape_and_counts():
    m = sidechain_mask(["ALA", "GLY", "PHE"])
    assert m.shape == (3, MAX_SC)
    assert m[0].sum().item() == 1   # ALA -> CB
    assert m[1].sum().item() == 0   # GLY -> none
    assert m[2].sum().item() == 7   # PHE -> 7 side-chain heavy atoms
