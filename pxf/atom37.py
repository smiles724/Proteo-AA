"""The atom37 and residue vocabularies shared by both modules.

PXDesign/Protenix and FaMPNN both use AlphaFold2's 37-slot atom representation
**in the same order**, so coordinates cross the module boundary with no
reordering at all. That is a convenient fact, not a safe assumption: if either
upstream ever renumbered its slots, coordinates would be silently scrambled
rather than rejected. :func:`assert_upstream_mapping` pins the agreement so a
drift fails loudly at import time.
"""
import torch

# AlphaFold2 atom37 order, as used by Protenix/PXDesign and FaMPNN alike.
ATOM37 = ("N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
          "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
          "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
          "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT")
NUM_ATOM37 = 37

# Canonical residue order, shared by both upstreams; index 20 is the unknown token.
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
UNKNOWN_AA_INDEX = 20

# Backbone slots. FaMPNN's ``non_bb_idxs`` is exactly the complement of these,
# which is what makes "keep the backbone, replace the side chains" well defined.
BACKBONE_SLOTS = (0, 1, 2, 4)
SIDECHAIN_SLOTS = tuple(i for i in range(NUM_ATOM37) if i not in BACKBONE_SLOTS)
BACKBONE_ATOMS = tuple(ATOM37[i] for i in BACKBONE_SLOTS)

MAPPING_VERSION = "pxf-pxdesign-fampnn-atom37-v1"


def assert_upstream_mapping(rc=None):
    """Fail unless FaMPNN's vocabularies still match the pinned AF2 contract."""
    if rc is None:
        from fampnn.data import residue_constants as rc
    if tuple(rc.atom_types) != ATOM37:
        raise ValueError(
            "FaMPNN atom37 order differs from the pinned AF2 order shared with "
            f"PXDesign; coordinates would be scrambled. upstream={tuple(rc.atom_types)!r}")
    if tuple(rc.restypes) != tuple(AA_ORDER):
        raise ValueError(f"FaMPNN residue order differs from AF2: {tuple(rc.restypes)!r}")
    if rc.restype_order_with_x["X"] != UNKNOWN_AA_INDEX:
        raise ValueError(f"FaMPNN unknown-residue index is {rc.restype_order_with_x['X']}, "
                         f"expected {UNKNOWN_AA_INDEX}")
    if tuple(sorted(rc.non_bb_idxs)) != SIDECHAIN_SLOTS:
        raise ValueError(
            "FaMPNN's side-chain slot set differs from the complement of the backbone "
            f"slots {BACKBONE_SLOTS}; upstream non_bb_idxs={tuple(sorted(rc.non_bb_idxs))!r}")
    return rc


def aatype_from_sequence(sequence, *, device=None):
    """Encode a one-letter sequence into the shared residue indices."""
    order = {letter: i for i, letter in enumerate(AA_ORDER)}
    unknown = [letter for letter in sequence if letter not in order]
    if unknown:
        raise ValueError(f"Sequence contains non-canonical residues: {sorted(set(unknown))}")
    return torch.tensor([order[letter] for letter in sequence], dtype=torch.long, device=device)


def sequence_from_aatype(aatype):
    """Decode residue indices back to a one-letter sequence."""
    letters = []
    for index in aatype.reshape(-1).tolist():
        if index == UNKNOWN_AA_INDEX:
            letters.append("X")
        elif 0 <= index < len(AA_ORDER):
            letters.append(AA_ORDER[index])
        else:
            raise ValueError(f"Residue index {index} outside the canonical vocabulary")
    return "".join(letters)


def mapping_record():
    """Serializable description of the shared vocabularies, for provenance."""
    return dict(mapping_version=MAPPING_VERSION, atom37=list(ATOM37), aa_order=AA_ORDER,
                backbone_slots=list(BACKBONE_SLOTS), sidechain_slots=list(SIDECHAIN_SLOTS),
                permutation_required=False)
