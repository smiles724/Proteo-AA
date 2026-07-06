"""Dynamic per-residue side-chain atom instantiation.

SideCraft instantiates the *real* side-chain atom set for a residue type — no
virtual / ghost / DMY padding (that is the atom14-style approach the project
explicitly avoids). We read canonical heavy-atom names from Protenix's static
`ATOM14` table (name lists per residue, backbone-first), which needs no CCD /
RDKit file, so this runs on CPU without the components database.

`MAX_SC` is the maximum number of side-chain heavy atoms across the 20 canonical
amino acids (TRP, 10). Masks are padded to `MAX_SC` for batching, but padded
slots are never generated or supervised (mask = False there).
"""
from typing import List, Sequence

import torch
from protenix.data.constants import ATOM14

BACKBONE_ATOMS = ("N", "CA", "C", "O")

STD_AA_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


def sidechain_atoms(restype_3letter: str) -> List[str]:
    """Ordered heavy side-chain atom names for a residue type (no backbone,
    no ghost). Unknown/GLY-like residues return ``[]``."""
    atoms = ATOM14.get(restype_3letter.upper())
    if atoms is None:
        return []
    return [a for a in atoms if a not in BACKBONE_ATOMS]


# Max side-chain heavy-atom count over the 20 canonical AAs (TRP = 10).
MAX_SC = max(len(sidechain_atoms(r)) for r in STD_AA_3)


def sidechain_mask(restypes: Sequence[str]) -> torch.Tensor:
    """Valid-atom mask [L, MAX_SC]: first ``len(sidechain_atoms(r))`` True."""
    L = len(restypes)
    m = torch.zeros(L, MAX_SC, dtype=torch.bool)
    for i, r in enumerate(restypes):
        k = len(sidechain_atoms(r))
        m[i, :k] = True
    return m
