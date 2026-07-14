"""Ideal side-chain templates ``mu_ideal`` in the residue-local frame.

Paragraph 221 of the SideCraft design ("Template-anchored leakage-free
initialization") starts side-chain denoising from

    y_{T,ij} = mu_ideal[a_i, j] + sigma_T * eps_ij ,   eps ~ N(0, I)
    x_{T,ij} = F_hat_i * y_{T,ij}

The ``mu_ideal`` term is what makes this work. An isotropic Gaussian is
rotation-invariant (R eps ~ eps), so pushing pure noise through the predicted
frame ``F_hat`` carries *no* backbone-orientation information and S_phi cannot
learn where to place atoms in global space. The ideal template is anisotropic:
once rotated by F_hat it encodes the residue's orientation, which is the entire
point of the paragraph.

LEAKAGE: the template depends only on residue TYPE (and hence the atom mask) —
never on ground-truth side-chain coordinates. It is exactly as leakage-free as
the existing atom-mask-only teacher forcing.

Provenance: heavy-atom ``_chem_comp_atom.pdbx_model_Cartn_{x,y,z}_ideal`` from
the wwPDB Chemical Component Dictionary (components.cif), mapped into the
residue-local frame with this package's own ``frames.build_frame`` / ``to_local``
(Gram-Schmidt, origin = CA) using each residue's own ideal N, CA, C. Values are
baked in as a static literal and rounded to 4 decimals, so this module — like
``instantiate.py`` — needs no CCD file and is CPU-testable.

Layout (must stay in lockstep with ``instantiate.py``):
  row order   = ``instantiate.STD_AA_3`` (the type_idx order used by
                ``instantiate_from_type_indices`` and the AA head)
  column order= ``instantiate.sidechain_atoms(restype)``, padded to ``MAX_SC``
  side-chain heavy-atom counts: 1, 7, 4, 4, 2, 5, 5, 0, 6, 4, 4, 5, 4, 7, 3, 2, 3, 10, 8, 3
  GLY has none -> all-False mask row.
"""
import torch

from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3, sidechain_atoms

# [20, MAX_SC, 3] local-frame ideal coordinates; padded slots are exact zeros.
_IDEAL_SC_LOCAL_LIST = [
    # ALA (1 side-chain heavy atom): CB
    [
        [ -0.5092,  -0.7213,  -1.2488],  # CB
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # ARG (7 side-chain heavy atoms): CB, CG, CD, NE, CZ, NH1, NH2
    [
        [ -0.5750,  -0.7822,  -1.1907],  # CB
        [ -0.1182,  -0.2750,  -2.5679],  # CG
        [ -0.6712,  -1.1211,  -3.7129],  # CD
        [ -2.1147,  -1.1081,  -3.7159],  # NE
        [ -2.8848,  -1.8045,  -4.6636],  # CZ
        [ -4.2742,  -1.7632,  -4.6272],  # NH1
        [ -2.2539,  -2.5483,  -5.6556],  # NH2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # ASN (4 side-chain heavy atoms): CB, CG, OD1, ND2
    [
        [ -0.5099,  -0.7216,  -1.2501],  # CB
        [ -2.0114,  -0.8297,  -1.1907],  # CG
        [ -2.6147,  -0.3752,  -0.2413],  # OD1
        [ -2.6840,  -1.4322,  -2.1910],  # ND2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # ASP (4 side-chain heavy atoms): CB, CG, OD1, OD2
    [
        [ -0.5100,  -0.7216,  -1.2491],  # CB
        [ -2.0125,  -0.8298,  -1.1884],  # CG
        [ -2.6122,  -0.3771,  -0.2426],  # OD1
        [ -2.6825,  -1.4296,  -2.1838],  # OD2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # CYS (2 side-chain heavy atoms): CB, SG
    [
        [ -0.5105,  -0.7205,  -1.2476],  # CB
        [ -2.3246,  -0.7186,  -1.2482],  # SG
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # GLN (5 side-chain heavy atoms): CB, CG, CD, OE1, NE2
    [
        [ -0.5114,  -0.7207,  -1.2475],  # CB
        [ -2.0398,  -0.7193,  -1.2482],  # CG
        [ -2.5439,  -1.4297,  -2.4774],  # CD
        [ -1.7564,  -1.8910,  -3.2753],  # OE1
        [ -3.8683,  -1.5521,  -2.6914],  # NE2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # GLU (5 side-chain heavy atoms): CB, CG, CD, OE1, OE2
    [
        [ -0.5090,  -0.7213,  -1.2499],  # CB
        [ -2.0344,  -0.8308,  -1.1880],  # CG
        [ -2.5365,  -1.5414,  -2.4191],  # CD
        [ -1.7549,  -1.9203,  -3.2592],  # OE1
        [ -3.8523,  -1.7521,  -2.5828],  # OE2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # GLY (0 side-chain heavy atoms): none
    [
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # HIS (6 side-chain heavy atoms): CB, CG, ND1, CD2, CE1, NE2
    [
        [ -0.5230,  -0.7950,  -1.2028],  # CB
        [ -0.4595,  -2.2892,  -0.9945],  # CG
        [  0.6429,  -2.9822,  -1.3557],  # ND1
        [ -1.3665,  -3.1226,  -0.4732],  # CD2
        [  0.4330,  -4.2692,  -1.0611],  # CE1
        [ -0.7869,  -4.3673,  -0.5227],  # NE2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # ILE (4 side-chain heavy atoms): CB, CG1, CG2, CD1
    [
        [ -0.5115,  -0.7204,  -1.2477],  # CB
        [ -2.0409,  -0.7211,  -1.2473],  # CG1
        [ -0.0021,  -2.1635,  -1.2479],  # CG2
        [ -2.5524,  -1.4415,  -2.4950],  # CD1
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # LEU (4 side-chain heavy atoms): CB, CG, CD1, CD2
    [
        [ -0.5101,  -0.7189,  -1.2489],  # CB
        [ -2.0404,  -0.7191,  -1.2489],  # CG
        [ -2.5511,  -1.4383,  -2.4990],  # CD1
        [ -2.5505,  -1.4395,  -0.0010],  # CD2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # LYS (5 side-chain heavy atoms): CB, CG, CD, CE, NZ
    [
        [ -0.5096,  -0.7206,  -1.2498],  # CB
        [ -2.0351,  -0.8300,  -1.1877],  # CG
        [ -2.5455,  -1.5509,  -2.4379],  # CD
        [ -4.0695,  -1.6605,  -2.3765],  # CE
        [ -4.5602,  -2.3534,  -3.5757],  # NZ
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # MET (4 side-chain heavy atoms): CB, CG, SD, CE
    [
        [ -0.5116,  -0.7199,  -1.2486],  # CB
        [ -2.0400,  -0.7196,  -1.2488],  # CG
        [ -2.6455,  -1.5739,  -2.7297],  # SD
        [ -4.4343,  -1.4243,  -2.4715],  # CE
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # PHE (7 side-chain heavy atoms): CB, CG, CD1, CD2, CE1, CE2, CZ
    [
        [ -0.5113,  -0.7217,  -1.2472],  # CB
        [ -2.0165,  -0.7214,  -1.2467],  # CG
        [ -2.7086,   0.3142,  -1.8449],  # CD1
        [ -2.7082,  -1.7609,  -0.6516],  # CD2
        [ -4.0906,   0.3137,  -1.8443],  # CE1
        [ -4.0901,  -1.7584,  -0.6486],  # CE2
        [ -4.7812,  -0.7217,  -1.2469],  # CZ
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # PRO (3 side-chain heavy atoms): CB, CG, CD
    [
        [ -0.5383,  -0.6185,  -1.3076],  # CB
        [ -1.7875,   0.2150,  -1.6606],  # CG
        [ -1.8983,   1.2674,  -0.5367],  # CD
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # SER (2 side-chain heavy atoms): CB, OG
    [
        [ -0.5104,  -0.7205,  -1.2479],  # CB
        [ -1.9388,  -0.7194,  -1.2487],  # OG
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # THR (3 side-chain heavy atoms): CB, OG1, CG2
    [
        [ -0.5112,  -0.7189,  -1.2489],  # CB
        [ -0.0345,  -0.0464,  -2.4149],  # OG1
        [ -2.0413,  -0.7174,  -1.2496],  # CG2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # TRP (10 side-chain heavy atoms): CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
    [
        [ -0.5100,  -0.7215,  -1.2474],  # CB
        [ -2.0167,  -0.7207,  -1.2465],  # CG
        [ -2.8151,   0.2136,  -1.7871],  # CD1
        [ -2.8879,  -1.7422,  -0.6628],  # CD2
        [ -4.1269,  -0.1239,  -1.5918],  # NE1
        [ -4.2061,  -1.3144,  -0.9046],  # CE2
        [ -2.6592,  -2.9316,   0.0315],  # CE3
        [ -5.2680,  -2.0917,  -0.4555],  # CZ2
        [ -3.7173,  -3.6799,   0.4627],  # CZ3
        [ -5.0193,  -3.2647,   0.2222],  # CH2
    ],
    # TYR (8 side-chain heavy atoms): CB, CG, CD1, CD2, CE1, CE2, CZ, OH
    [
        [ -0.5105,  -0.7209,  -1.2477],  # CB
        [ -2.0167,  -0.7208,  -1.2475],  # CG
        [ -2.7067,   0.3160,  -1.8472],  # CD1
        [ -2.7070,  -1.7613,  -0.6529],  # CD2
        [ -4.0877,   0.3190,  -1.8481],  # CE1
        [ -4.0879,  -1.7605,  -0.6477],  # CE2
        [ -4.7832,  -0.7202,  -1.2487],  # CZ
        [ -6.1414,  -0.7207,  -1.2484],  # OH
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
    # VAL (3 side-chain heavy atoms): CB, CG1, CG2
    [
        [ -0.5101,  -0.7200,  -1.2483],  # CB
        [ -2.0400,  -0.7198,  -1.2490],  # CG1
        [  0.0001,   0.0003,  -2.4971],  # CG2
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
        [  0.0000,   0.0000,   0.0000],  # pad
    ],
]

IDEAL_SC_LOCAL: torch.Tensor = torch.tensor(_IDEAL_SC_LOCAL_LIST, dtype=torch.float32)

# [20, MAX_SC] bool — True where the residue actually has that side-chain atom.
IDEAL_SC_MASK: torch.Tensor = torch.zeros(len(STD_AA_3), MAX_SC, dtype=torch.bool)
for _i, _r in enumerate(STD_AA_3):
    IDEAL_SC_MASK[_i, : len(sidechain_atoms(_r))] = True

# Padded slots carry no geometry.
IDEAL_SC_LOCAL = IDEAL_SC_LOCAL * IDEAL_SC_MASK[..., None].to(IDEAL_SC_LOCAL.dtype)


def ideal_template(type_idx: torch.Tensor):
    """Look up the ideal local-frame side-chain template for residue types.

    Depends on residue TYPE only — no ground-truth coordinates (leakage rule).

    Args:
        type_idx: [...] long, values in [0, 19] in ``STD_AA_3`` order. Values
            outside that range are treated as GLY (empty side chain), matching
            ``instantiate_from_type_indices``.
    Returns:
        coords: [..., MAX_SC, 3] float32 local-frame template (zeros at pads).
        mask:   [..., MAX_SC] bool valid-atom mask.
    """
    table = IDEAL_SC_LOCAL.to(type_idx.device)
    mask_table = IDEAL_SC_MASK.to(type_idx.device)

    gly = STD_AA_3.index("GLY")
    valid = (type_idx >= 0) & (type_idx < len(STD_AA_3))
    idx = torch.where(valid, type_idx, torch.full_like(type_idx, gly))

    return table[idx], mask_table[idx]


# ---------------------------------------------------------------------------
# Swappable provider
# ---------------------------------------------------------------------------
# The table above is the wwPDB CCD *ideal* geometry: correct bond lengths and bond
# angles, but ONE canonical choice of chi (torsion). Measured against real side chains
# (1cse chain B, residue-local frame): ALA, which has no chi, sits 0.07 A from truth,
# while multi-chi residues are off by 2.4-3.4 A (HIS 3.37, GLU 2.60, ARG 2.42, TYR 2.38).
# So the template is chemically right and conformationally arbitrary.
#
# That is exactly what a rotamer-statistics table would improve, so the lookup is a
# REGISTERABLE PROVIDER rather than a hard-wired call. A replacement drops in with one
# line and NOTHING in model.py / init.py / cogenerate.py has to change:
#
#     from pxdesign_train.sidechain import templates
#     templates.set_ideal_template_provider(my_provider)
#
# The provider contract (all three levels are supported):
#
#     provider(type_idx, *, generator=None, backbone=None) -> (coords, mask)
#
#     type_idx   LongTensor [...]              residue type, STD_AA_3 order
#     generator  Optional[torch.Generator]     for a STOCHASTIC provider -- e.g. sampling a
#                                              rotamer from a distribution instead of always
#                                              returning the same chi. Use it, do not create
#                                              your own RNG, or runs stop being reproducible.
#     backbone   Optional[Tensor [..., 4, 3]]  the residue's own N, CA, C, O in its LOCAL
#                                              frame, for a BACKBONE-DEPENDENT provider
#                                              (phi/psi-conditioned rotamer libraries).
#                                              None when the caller has no backbone to give.
#     returns    coords [..., MAX_SC, 3] float32 in the residue-LOCAL frame,
#                mask   [..., MAX_SC] bool, column order == instantiate.sidechain_atoms.
#
# LEAKAGE RULE, enforced by test: a provider may depend on residue TYPE, on the atom mask,
# and on the (predicted) BACKBONE. It must NEVER receive or consult ground-truth side-chain
# COORDINATES -- that is the whole point of paragraph 221's "leakage-free" initialization.
# There is deliberately no parameter through which they could arrive.


def _default_provider(type_idx, *, generator=None, backbone=None):
    """The shipped CCD ideal table: deterministic, type-only (ignores generator/backbone)."""
    return ideal_template(type_idx)


_PROVIDER = _default_provider


def set_ideal_template_provider(fn=None):
    """Register the mu_ideal provider. Pass None to restore the CCD default."""
    global _PROVIDER
    _PROVIDER = _default_provider if fn is None else fn


def get_ideal_template_provider():
    return _PROVIDER


def is_default_provider() -> bool:
    return _PROVIDER is _default_provider
