"""Ideal covalent geometry G_ideal(a) and torsion definitions — GENERATED.

Regenerate with::

    python scripts/build_chi_constants.py <components.cif> -o pxdesign_train/sidechain/chi_constants.py
    python scripts/build_chi_constants.py <components.cif> --check

Overleaf 0714 Appendix, Step 1 (`a_hat -> (A_sc, K_i, G_ideal)`). Do not hand-edit.

Index space for CHI_ATOM_IDX: 0=N, 1=CA, 2=C, then 3+j = side-chain column j
in ``instantiate.sidechain_atoms(restype)`` order — the same column order as
``templates.IDEAL_SC_LOCAL``.
"""
import torch

MAX_CHI = 4
N_BB_FRAME = 3  # N, CA, C occupy combined indices 0, 1, 2

# [20, 4, 4] long — the four atoms defining each chi (combined index space).
CHI_ATOM_IDX = torch.tensor([
    [[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # ALA
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 3,  4,  5,  6], [ 4,  5,  6,  7]],  # ARG
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # ASN
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # ASP
    [[ 0,  1,  3,  4], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # CYS
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 3,  4,  5,  6], [ 0,  0,  0,  0]],  # GLN
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 3,  4,  5,  6], [ 0,  0,  0,  0]],  # GLU
    [[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # GLY
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # HIS
    [[ 0,  1,  3,  4], [ 1,  3,  4,  6], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # ILE
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # LEU
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 3,  4,  5,  6], [ 4,  5,  6,  7]],  # LYS
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 3,  4,  5,  6], [ 0,  0,  0,  0]],  # MET
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # PHE
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # PRO
    [[ 0,  1,  3,  4], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # SER
    [[ 0,  1,  3,  4], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # THR
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # TRP
    [[ 0,  1,  3,  4], [ 1,  3,  4,  5], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # TYR
    [[ 0,  1,  3,  4], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],  # VAL
], dtype=torch.long)

# [20, 4] bool — K_i: which chi angles the residue actually has (AF chi mask).
CHI_MASK = torch.tensor([
    [False, False, False, False],  # ALA
    [True , True , True , True ],  # ARG
    [True , True , False, False],  # ASN
    [True , True , False, False],  # ASP
    [True , False, False, False],  # CYS
    [True , True , True , False],  # GLN
    [True , True , True , False],  # GLU
    [False, False, False, False],  # GLY
    [True , True , False, False],  # HIS
    [True , True , False, False],  # ILE
    [True , True , False, False],  # LEU
    [True , True , True , True ],  # LYS
    [True , True , True , False],  # MET
    [True , True , False, False],  # PHE
    [True , True , False, False],  # PRO
    [True , False, False, False],  # SER
    [True , False, False, False],  # THR
    [True , True , False, False],  # TRP
    [True , True , False, False],  # TYR
    [True , False, False, False],  # VAL
], dtype=torch.bool)

# [20, 4] bool — torsion is a genuine rigid rotation (False on ring-closed
# torsions: PRO chi1/chi2, where CD bonds back to N and no rigid rotation exists).
CHI_ROTATABLE = torch.tensor([
    [False, False, False, False],  # ALA
    [True , True , True , True ],  # ARG
    [True , True , False, False],  # ASN
    [True , True , False, False],  # ASP
    [True , False, False, False],  # CYS
    [True , True , True , False],  # GLN
    [True , True , True , False],  # GLU
    [False, False, False, False],  # GLY
    [True , True , False, False],  # HIS
    [True , True , False, False],  # ILE
    [True , True , False, False],  # LEU
    [True , True , True , True ],  # LYS
    [True , True , True , False],  # MET
    [True , True , False, False],  # PHE
    [False, False, False, False],  # PRO
    [True , False, False, False],  # SER
    [True , False, False, False],  # THR
    [True , True , False, False],  # TRP
    [True , True , False, False],  # TYR
    [True , False, False, False],  # VAL
], dtype=torch.bool)

# [20, 4, MAX_SC] bool — side-chain atoms rigidly carried by each chi rotation
# (connected component of the axis's distal atom after cutting the axis bond).
CHI_DOWNSTREAM = torch.tensor([
    [  # ALA: CB
        [False, False, False, False, False, False, False, False, False, False],  # chi1: -
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # ARG: CB, CG, CD, NE, CZ, NH1, NH2
        [True , True , True , True , True , True , True , False, False, False],  # chi1: CB, CG, CD, NE, CZ, NH1, NH2
        [False, True , True , True , True , True , True , False, False, False],  # chi2: CG, CD, NE, CZ, NH1, NH2
        [False, False, True , True , True , True , True , False, False, False],  # chi3: CD, NE, CZ, NH1, NH2
        [False, False, False, True , True , True , True , False, False, False],  # chi4: NE, CZ, NH1, NH2
    ],
    [  # ASN: CB, CG, OD1, ND2
        [True , True , True , True , False, False, False, False, False, False],  # chi1: CB, CG, OD1, ND2
        [False, True , True , True , False, False, False, False, False, False],  # chi2: CG, OD1, ND2
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # ASP: CB, CG, OD1, OD2
        [True , True , True , True , False, False, False, False, False, False],  # chi1: CB, CG, OD1, OD2
        [False, True , True , True , False, False, False, False, False, False],  # chi2: CG, OD1, OD2
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # CYS: CB, SG
        [True , True , False, False, False, False, False, False, False, False],  # chi1: CB, SG
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # GLN: CB, CG, CD, OE1, NE2
        [True , True , True , True , True , False, False, False, False, False],  # chi1: CB, CG, CD, OE1, NE2
        [False, True , True , True , True , False, False, False, False, False],  # chi2: CG, CD, OE1, NE2
        [False, False, True , True , True , False, False, False, False, False],  # chi3: CD, OE1, NE2
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # GLU: CB, CG, CD, OE1, OE2
        [True , True , True , True , True , False, False, False, False, False],  # chi1: CB, CG, CD, OE1, OE2
        [False, True , True , True , True , False, False, False, False, False],  # chi2: CG, CD, OE1, OE2
        [False, False, True , True , True , False, False, False, False, False],  # chi3: CD, OE1, OE2
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # GLY: none
        [False, False, False, False, False, False, False, False, False, False],  # chi1: -
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # HIS: CB, CG, ND1, CD2, CE1, NE2
        [True , True , True , True , True , True , False, False, False, False],  # chi1: CB, CG, ND1, CD2, CE1, NE2
        [False, True , True , True , True , True , False, False, False, False],  # chi2: CG, ND1, CD2, CE1, NE2
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # ILE: CB, CG1, CG2, CD1
        [True , True , True , True , False, False, False, False, False, False],  # chi1: CB, CG1, CG2, CD1
        [False, True , False, True , False, False, False, False, False, False],  # chi2: CG1, CD1
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # LEU: CB, CG, CD1, CD2
        [True , True , True , True , False, False, False, False, False, False],  # chi1: CB, CG, CD1, CD2
        [False, True , True , True , False, False, False, False, False, False],  # chi2: CG, CD1, CD2
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # LYS: CB, CG, CD, CE, NZ
        [True , True , True , True , True , False, False, False, False, False],  # chi1: CB, CG, CD, CE, NZ
        [False, True , True , True , True , False, False, False, False, False],  # chi2: CG, CD, CE, NZ
        [False, False, True , True , True , False, False, False, False, False],  # chi3: CD, CE, NZ
        [False, False, False, True , True , False, False, False, False, False],  # chi4: CE, NZ
    ],
    [  # MET: CB, CG, SD, CE
        [True , True , True , True , False, False, False, False, False, False],  # chi1: CB, CG, SD, CE
        [False, True , True , True , False, False, False, False, False, False],  # chi2: CG, SD, CE
        [False, False, True , True , False, False, False, False, False, False],  # chi3: SD, CE
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # PHE: CB, CG, CD1, CD2, CE1, CE2, CZ
        [True , True , True , True , True , True , True , False, False, False],  # chi1: CB, CG, CD1, CD2, CE1, CE2, CZ
        [False, True , True , True , True , True , True , False, False, False],  # chi2: CG, CD1, CD2, CE1, CE2, CZ
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # PRO: CB, CG, CD
        [False, False, False, False, False, False, False, False, False, False],  # chi1: -
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # SER: CB, OG
        [True , True , False, False, False, False, False, False, False, False],  # chi1: CB, OG
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # THR: CB, OG1, CG2
        [True , True , True , False, False, False, False, False, False, False],  # chi1: CB, OG1, CG2
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # TRP: CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
        [True , True , True , True , True , True , True , True , True , True ],  # chi1: CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
        [False, True , True , True , True , True , True , True , True , True ],  # chi2: CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # TYR: CB, CG, CD1, CD2, CE1, CE2, CZ, OH
        [True , True , True , True , True , True , True , True , False, False],  # chi1: CB, CG, CD1, CD2, CE1, CE2, CZ, OH
        [False, True , True , True , True , True , True , True , False, False],  # chi2: CG, CD1, CD2, CE1, CE2, CZ, OH
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
    [  # VAL: CB, CG1, CG2
        [True , True , True , False, False, False, False, False, False, False],  # chi1: CB, CG1, CG2
        [False, False, False, False, False, False, False, False, False, False],  # chi2: -
        [False, False, False, False, False, False, False, False, False, False],  # chi3: -
        [False, False, False, False, False, False, False, False, False, False],  # chi4: -
    ],
], dtype=torch.bool)

# [20, 3, 3] float32 — ideal N, CA, C in the residue-LOCAL frame (frames.build_frame).
# CA is the frame origin, so its row is exactly zero. chi1's first atom is N, which
# is why the backbone has to be in the same local frame as the side-chain template.
IDEAL_BB_LOCAL = torch.tensor([
    [[ -0.4905,   1.3833,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5055,   0.0000,   0.0000]],  # ALA
    [[ -0.4551,   1.3891,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5181,   0.0000,   0.0000]],  # ARG
    [[ -0.4904,   1.3839,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5072,   0.0000,   0.0000]],  # ASN
    [[ -0.4898,   1.3846,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5069,   0.0000,   0.0000]],  # ASP
    [[ -0.4876,   1.3855,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5064,   0.0000,   0.0000]],  # CYS
    [[ -0.4891,   1.3849,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5063,   0.0000,   0.0000]],  # GLN
    [[ -0.4896,   1.3845,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5078,   0.0000,   0.0000]],  # GLU
    [[ -0.4891,   1.3859,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5069,   0.0000,   0.0000]],  # GLY
    [[ -0.5199,   1.3440,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5218,   0.0000,   0.0000]],  # HIS
    [[ -0.4899,   1.3847,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5061,   0.0000,   0.0000]],  # ILE
    [[ -0.4889,   1.3854,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5061,   0.0000,   0.0000]],  # LEU
    [[ -0.4898,   1.3853,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5066,   0.0000,   0.0000]],  # LYS
    [[ -0.4887,   1.3852,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5062,   0.0000,   0.0000]],  # MET
    [[ -0.4900,   1.3844,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5071,   0.0000,   0.0000]],  # PHE
    [[ -0.5177,   1.3931,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5076,   0.0000,   0.0000]],  # PRO
    [[ -0.4888,   1.3853,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5067,   0.0000,   0.0000]],  # SER
    [[ -0.4883,   1.3859,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5061,   0.0000,   0.0000]],  # THR
    [[ -0.4886,   1.3852,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5074,   0.0000,   0.0000]],  # TRP
    [[ -0.4896,   1.3848,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5072,   0.0000,   0.0000]],  # TYR
    [[ -0.4894,   1.3849,   0.0000], [  0.0000,   0.0000,   0.0000], [  1.5059,   0.0000,   0.0000]],  # VAL
], dtype=torch.float32)

