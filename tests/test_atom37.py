"""Both modules must agree on atom37, or coordinates scramble silently."""

import pytest
import torch

from pxf import atom37


def test_pinned_vocabulary_is_well_formed():
    assert len(atom37.ATOM37) == atom37.NUM_ATOM37 == 37
    assert len(set(atom37.ATOM37)) == 37
    assert len(atom37.AA_ORDER) == 20 == len(set(atom37.AA_ORDER))


def test_backbone_and_sidechain_slots_partition_atom37():
    assert set(atom37.BACKBONE_SLOTS) | set(atom37.SIDECHAIN_SLOTS) == set(range(37))
    assert not set(atom37.BACKBONE_SLOTS) & set(atom37.SIDECHAIN_SLOTS)
    assert atom37.BACKBONE_ATOMS == ("N", "CA", "C", "O")
    assert len(atom37.SIDECHAIN_SLOTS) == 33


def test_fampnn_agrees_with_the_pinned_af2_contract():
    # This is what makes the boundary a no-op instead of a permutation.
    rc = atom37.assert_upstream_mapping()
    assert tuple(rc.atom_types) == atom37.ATOM37
    assert tuple(rc.restypes) == tuple(atom37.AA_ORDER)
    assert tuple(sorted(rc.non_bb_idxs)) == atom37.SIDECHAIN_SLOTS
    assert rc.restype_order_with_x["X"] == atom37.UNKNOWN_AA_INDEX == 20


def test_a_reordered_upstream_is_rejected():
    class Fake:
        atom_types = ("CA", "N", "C", "O") + atom37.ATOM37[4:]
        restypes = tuple(atom37.AA_ORDER)
        restype_order_with_x = {"X": 20}
        non_bb_idxs = atom37.SIDECHAIN_SLOTS

    with pytest.raises(ValueError, match="atom37 order differs"):
        atom37.assert_upstream_mapping(Fake)


def test_a_reordered_residue_vocabulary_is_rejected():
    class Fake:
        atom_types = atom37.ATOM37
        restypes = tuple("ACDEFGHIKLMNPQRSTVWY")  # alphabetical, not AF2
        restype_order_with_x = {"X": 20}
        non_bb_idxs = atom37.SIDECHAIN_SLOTS

    with pytest.raises(ValueError, match="residue order differs"):
        atom37.assert_upstream_mapping(Fake)


def test_sequence_round_trips_through_aatype():
    sequence = atom37.AA_ORDER * 3
    assert atom37.sequence_from_aatype(atom37.aatype_from_sequence(sequence)) == sequence


def test_aatype_indices_are_the_canonical_order():
    assert atom37.aatype_from_sequence(atom37.AA_ORDER).tolist() == list(range(20))


def test_non_canonical_residues_are_refused():
    with pytest.raises(ValueError, match="non-canonical"):
        atom37.aatype_from_sequence("AAXAA")
    with pytest.raises(ValueError, match="outside the canonical vocabulary"):
        atom37.sequence_from_aatype(torch.tensor([99]))


def test_unknown_index_decodes_to_x():
    assert atom37.sequence_from_aatype(torch.tensor([atom37.UNKNOWN_AA_INDEX])) == "X"


def test_mapping_record_states_no_permutation_and_serializes():
    import json

    record = atom37.mapping_record()
    assert record["permutation_required"] is False
    assert record["mapping_version"] == atom37.MAPPING_VERSION
    json.loads(json.dumps(record))
