"""Named atom conversion, retaining autograd and arbitrary leading axes."""
import torch
from pxdesign_train.sidechain.instantiate import ATOM_NAME_TO_ID, STD_AA_3

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
ATOM37 = ("N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
          "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
          "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
          "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT")
BB37 = (0, 1, 2, 4)
MAPPING_VERSION = "proteoaa-fampnn-atom-name-v1"
_ID_TO_37 = torch.full((max(ATOM_NAME_TO_ID.values()) + 1,), -1, dtype=torch.long)
for _name, _id in ATOM_NAME_TO_ID.items():
    _ID_TO_37[_id] = ATOM37.index(_name)


def assert_upstream_mapping(rc):
    if tuple(rc.restypes) != tuple(AA_ORDER) or tuple(rc.atom_types) != ATOM37:
        raise ValueError("FaMPNN canonical AA/Atom37 mapping differs from the pinned contract")
    if [rc.restype_1to3[a] for a in AA_ORDER] != STD_AA_3:
        raise ValueError("Proteo-AA and FaMPNN residue orders differ")


def scatter_named_atoms(xyz, atom_name_ids, mask):
    """Scatter [...,L,A,3] to Atom37 plus mask; padding goes to a trash slot.

    Inactive coordinates are cleared before scatter, including NaN placeholders.
    Duplicate live names are rejected rather than silently added.
    """
    table = _ID_TO_37.to(atom_name_ids.device)
    if ((atom_name_ids < 0) | (atom_name_ids >= table.numel())).any():
        raise ValueError("Unknown side-chain atom-name embedding ID")
    idx = table[atom_name_ids.long()]
    valid = mask.bool() & (idx >= 0)
    idx = torch.where(valid, idx, 37)
    counts = torch.zeros(*idx.shape[:-1], 38, device=idx.device, dtype=torch.long)
    counts = counts.scatter_add(-1, idx, valid.long())
    if (counts[..., :37] > 1).any():
        raise ValueError("Duplicate live atom names in a residue")
    clean = torch.where(valid[..., None], xyz, 0.0)
    out = xyz.new_zeros(*xyz.shape[:-2], 38, 3)
    out = out.scatter_add(-2, idx[..., None].expand_as(xyz), clean)
    return out[..., :37, :], counts[..., :37].bool()
