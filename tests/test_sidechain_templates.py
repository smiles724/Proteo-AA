"""Tests for the ideal side-chain template table ``mu_ideal`` (Overleaf para 221).

FRAME CONVENTION UNDER TEST
--------------------------
``templates.IDEAL_SC_LOCAL`` stores coordinates in the residue-local frame built by
``frames.build_frame(N, CA, C)``:

    e1 = normalize(C - CA);  e2 = normalize((N - CA) orth. e1);  e3 = e1 x e2
    R  = [e1 | e2 | e3]      (local -> global)
    t  = CA                  <-- THE FRAME ORIGIN IS THE CA ATOM

so a local coordinate is ``R^T (x_global - CA)`` and the origin of the local frame
is exactly the residue's own CA. Therefore the norm of the CB row *is* the CA-CB
bond length.

WHY THIS FILE IS STRUCTURED THE WAY IT IS
-----------------------------------------
An earlier version of this suite was VACUOUS: every geometric assertion was
invariant under rotation, column permutation and reflection, so a table generated
under a different frame convention -- or with scrambled atom columns, or mirrored
into D-amino acids -- passed 37/37. That is precisely the silent-orientation
failure paragraph 221 exists to prevent.

The checks below are therefore written as free functions over a *table argument*
(``check_*(table)``), and ``test_checks_are_non_vacuous`` re-runs each one on a
deliberately corrupted copy of the table and asserts it TRIPS. Three corruptions,
mirroring the three ways the table can silently go wrong:

    MUTATION            what it simulates                        killed by
    ------------------  ---------------------------------------  -----------------------
    permute_basis       table built under another Gram-Schmidt    check_cb_frame_pin
                        convention (axes relabelled)
    scramble_columns    column order != instantiate.sidechain_    check_bond_connectivity
                        atoms(restype)
    flip_z              mirrored chirality (D-amino acids)        check_chirality

Provenance / drift-safety: ``scripts/build_sidechain_templates.py`` regenerates the
literal from a CCD ``components.cif`` using ``frames.build_frame`` / ``frames.to_local``
themselves, so the extraction convention cannot drift from the runtime convention.

BACKBONE ATOMS IN THE LOCAL FRAME
---------------------------------
The table stores side-chain atoms only, but three checks need N and C. They are
NOT free parameters: the frame is *built from* N/CA/C, so their local coordinates
are fully determined by the convention plus ideal backbone geometry --

    CA_local = 0                              (origin)
    C_local  = (|C-CA|, 0, 0)                 (e1 is defined as C - CA)
    N_local  = (|N-CA| cos tau, |N-CA| sin tau, 0)   (e2 spans N's component orth. e1,
                                                      by construction with +y sign;
                                                      e3 is orthogonal to both -> z = 0)

with tau = the N-CA-C angle. ``_ideal_backbone_local()`` below does not hard-code
that algebra: it plants an ideal backbone at a random pose in space, runs
``frames.build_frame`` / ``frames.to_local`` on it, and reads N and C back out --
so it inherits the convention from frames.py rather than restating it.
"""
import math

import pytest
import torch

from pxdesign_train.sidechain.frames import build_frame, to_local
from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    STD_AA_3,
    sidechain_atoms,
    sidechain_mask,
)
from pxdesign_train.sidechain.templates import (
    _IDEAL_SC_LOCAL_LIST,
    IDEAL_SC_LOCAL,
    IDEAL_SC_MASK,
    ideal_template,
)

N_AA = len(STD_AA_3)
LARGE = ["TRP", "PHE", "ARG", "LYS"]
NON_GLY = [r for r in STD_AA_3 if r != "GLY"]

# --- ideal backbone geometry (Engh & Huber); used only to place N and C ------
D_CA_N = 1.458      # A
D_CA_C = 1.525      # A
TAU_N_CA_C = 111.0  # deg

# AF2 / OpenFold virtual-CB constants, fitted on L-amino acids in real PDB
# structures: CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA,
# with b = CA - N, c = C - CA, a = b x c. This is our EXTERNAL reference for
# both the CB direction and the chirality sign -- it is independent of the CCD
# numbers in templates.py, so pinning against it is not circular.
_VCB_A, _VCB_B, _VCB_C = -0.58273431, 0.56802827, -0.54067466


# --------------------------------------------------------------------------
# backbone atoms in frames.py's local frame (derived, never hand-copied)
# --------------------------------------------------------------------------
def _ideal_backbone_local(seed: int = 0):
    """(N_local, CA_local, C_local) for an ideal backbone, via frames.py itself.

    An ideal N/CA/C triangle is placed at an arbitrary random pose in global
    space; ``build_frame`` + ``to_local`` then return its local coordinates. The
    result is pose-independent (asserted in ``test_backbone_local_is_derived_not_assumed``),
    so these ARE the backbone atoms of every residue-local frame.
    """
    g = torch.Generator().manual_seed(seed)
    tau = math.radians(TAU_N_CA_C)
    ca0 = torch.zeros(3, dtype=torch.float64)
    c0 = torch.tensor([D_CA_C, 0.0, 0.0], dtype=torch.float64)
    n0 = torch.tensor([D_CA_N * math.cos(tau), D_CA_N * math.sin(tau), 0.0], dtype=torch.float64)

    # random SE(3) pose so nothing about the *global* placement can leak in
    q = torch.randn(3, 3, generator=g, dtype=torch.float64)
    R_rand, _ = torch.linalg.qr(q)
    if torch.det(R_rand) < 0:
        R_rand[:, 0] *= -1
    shift = torch.randn(3, generator=g, dtype=torch.float64) * 5.0
    n, ca, c = (R_rand @ v + shift for v in (n0, ca0, c0))

    R, t = build_frame(n, ca, c)
    loc = to_local(torch.stack([n, ca, c]), R, t)
    return loc[0], loc[1], loc[2]


def _virtual_cb_local():
    """Analytic L-amino-acid CB, expressed in frames.py's basis (AF2 constants)."""
    n, ca, c = _ideal_backbone_local()
    b = ca - n
    cc = c - ca
    a = torch.linalg.cross(b, cc)
    return _VCB_A * a + _VCB_B * b + _VCB_C * cc + ca


# --------------------------------------------------------------------------
# bond table: intra-side-chain covalent bonds, by ATOM NAME.
# Column indices are resolved through instantiate.sidechain_atoms(restype), so a
# column order that disagrees with instantiate.py breaks these distances.
# --------------------------------------------------------------------------
SIDECHAIN_BONDS = {
    "ALA": [],  # CB only; its CA bond is check_ca_cb_bond
    "ARG": [("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CB", "SG")],
    "GLN": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [],
    "HIS": [("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2")],
    "ILE": [("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ")],                                    # 6-ring closure
    "PRO": [("CB", "CG"), ("CG", "CD")],  # CD-N closes the ring: check_pro_ring_closure
    "SER": [("CB", "OG")],
    "THR": [("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CB", "CG"), ("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"),
            ("CD2", "CG"),                                                    # 5-ring closure
            ("CD2", "CE3"), ("CE3", "CZ3"), ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2")],  # 6-ring
    "TYR": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("CB", "CG1"), ("CB", "CG2")],
}

# Element-pair-dependent covalent bond windows (A). C=O carbonyl is ~1.23, so
# oxygen bonds need a lower floor than C-C/C-N; C-S is much longer.
BOND_RANGE = {"S": (1.70, 1.95), "O": (1.15, 1.50), "CN": (1.28, 1.62)}
CA_CB_RANGE = (1.45, 1.62)
PRO_CD_N_RANGE = (1.38, 1.60)   # pyrrolidine N-CD bond, ~1.48 A
CB_PIN_TOL_DEG = 5.0            # ALA CB vs analytic virtual CB


def _bond_class(a: str, b: str) -> str:
    if a[0] == "S" or b[0] == "S":
        return "S"
    if a[0] == "O" or b[0] == "O":
        return "O"
    return "CN"


# --------------------------------------------------------------------------
# THE CHECKS. Written over a table argument so the non-vacuity test can feed
# them a corrupted table and prove they trip.
# --------------------------------------------------------------------------
def check_bond_connectivity(table: torch.Tensor):
    """Every known intra-side-chain covalent bond is at chemical bond distance."""
    for restype, bonds in SIDECHAIN_BONDS.items():
        i = STD_AA_3.index(restype)
        names = sidechain_atoms(restype)
        for a, b in bonds:
            d = torch.linalg.norm(table[i, names.index(a)] - table[i, names.index(b)]).item()
            lo, hi = BOND_RANGE[_bond_class(a, b)]
            assert lo <= d <= hi, f"{restype} {a}-{b}: {d:.4f} A outside [{lo}, {hi}]"


def check_ca_cb_bond(table: torch.Tensor):
    """CB is column 0 for every non-GLY residue and the origin is CA, so ||CB|| is
    the CA-CB bond."""
    for restype in NON_GLY:
        i = STD_AA_3.index(restype)
        assert sidechain_atoms(restype)[0] == "CB"
        d = torch.linalg.norm(table[i, 0]).item()
        lo, hi = CA_CB_RANGE
        assert lo <= d <= hi, f"{restype}: |CB - CA| = {d:.4f} A outside [{lo}, {hi}]"


def check_pro_ring_closure(table: torch.Tensor):
    """PRO's pyrrolidine ring closes through a BACKBONE atom: CD-N, ~1.48 A.

    N is not in the table -- it is the backbone nitrogen, whose local coordinate is
    fixed by the frame convention (see module docstring / ``_ideal_backbone_local``).
    """
    n_local, _, _ = _ideal_backbone_local()
    i = STD_AA_3.index("PRO")
    cd = table[i, sidechain_atoms("PRO").index("CD")].double()
    d = torch.linalg.norm(cd - n_local).item()
    lo, hi = PRO_CD_N_RANGE
    assert lo <= d <= hi, f"PRO CD-N: {d:.4f} A outside [{lo}, {hi}] -- ring does not close"


def check_chirality(table: torch.Tensor):
    """L-amino acids: det[(N-CA), (C-CA), (CB-CA)] > 0 for all 19 non-GLY rows.

    CA is the origin, so the rows are just N_local, C_local (from the frame
    convention) and the table's CB row. Mirroring the table (z -> -z, or any other
    reflection) flips this determinant's sign: that is a D-amino acid, and it is
    the failure mode a reflection-invariant test suite cannot see.
    """
    n_local, _, c_local = _ideal_backbone_local()
    for restype in NON_GLY:
        i = STD_AA_3.index(restype)
        cb = table[i, 0].double()
        det = torch.linalg.det(torch.stack([n_local, c_local, cb])).item()
        assert det > 0.5, f"{restype}: chirality det = {det:.4f} <= 0 (D-amino acid / mirrored table)"


def check_cb_frame_pin(table: torch.Tensor, restypes=("ALA",), tol_deg: float = CB_PIN_TOL_DEG):
    """Pin CB's DIRECTION (not just its norm) to the analytic L-amino-acid virtual CB
    expressed in frames.py's basis. This is the check that a relabelled/rotated basis
    cannot survive: ||CB|| and eigenvalue ratios are rotation-invariant, a direction is not.
    """
    expected = _virtual_cb_local()
    e_hat = expected / expected.norm()
    for restype in restypes:
        cb = table[STD_AA_3.index(restype), 0].double()
        cos = torch.clamp(cb @ e_hat / cb.norm(), -1.0, 1.0).item()
        ang = math.degrees(math.acos(cos))
        assert ang <= tol_deg, (
            f"{restype}: CB is {ang:.2f} deg from the analytic virtual-CB direction "
            f"{tuple(round(float(v), 4) for v in expected)} (tol {tol_deg} deg) -- "
            f"the table's frame convention does not match frames.py"
        )


# --------------------------------------------------------------------------
# THE MUTATIONS. Each corrupts the table exactly the way it can silently go wrong.
# --------------------------------------------------------------------------
def mutate_permute_basis(table: torch.Tensor) -> torch.Tensor:
    """Table generated under a different Gram-Schmidt convention: e1/e2 swapped
    (a rotation+relabelling of the basis; norms and eigenvalue ratios unchanged)."""
    return table[..., [1, 0, 2]].clone()


def mutate_scramble_columns(table: torch.Tensor, seed: int = 7) -> torch.Tensor:
    """Column order != instantiate.sidechain_atoms(restype).

    CB is deliberately left in column 0: every plausible atom ordering puts CB first,
    so a scramble that moved CB would be caught even by the old ||CB|| check and would
    be a strawman. This permutes columns 1..k-1 (derangement) for every residue with
    >= 3 side-chain atoms -- the realistic failure, and one the old suite cannot see.
    """
    g = torch.Generator().manual_seed(seed)
    out = table.clone()
    for i, restype in enumerate(STD_AA_3):
        k = len(sidechain_atoms(restype))
        if k < 3:
            continue  # nothing to permute once CB is pinned
        tail = torch.arange(1, k)
        perm = tail[torch.randperm(k - 1, generator=g)]
        while bool((perm == tail).all()):
            perm = tail[torch.randperm(k - 1, generator=g)]
        out[i, 1:k] = table[i, perm]
    return out


def mutate_flip_z(table: torch.Tensor) -> torch.Tensor:
    """Mirrored chirality: D-amino acids. Preserves every distance in the table."""
    out = table.clone()
    out[..., 2] *= -1
    return out


# --------------------------------------------------------------------------
# shape / dtype / finiteness
# --------------------------------------------------------------------------
def test_shapes_and_dtypes():
    assert MAX_SC == 10, "MAX_SC is expected to be 10 (TRP)"
    assert IDEAL_SC_LOCAL.shape == (N_AA, MAX_SC, 3)
    assert IDEAL_SC_MASK.shape == (N_AA, MAX_SC)
    assert IDEAL_SC_LOCAL.dtype == torch.float32
    assert IDEAL_SC_MASK.dtype == torch.bool


def test_no_nan_or_inf():
    assert torch.isfinite(IDEAL_SC_LOCAL).all()


def test_raw_literal_padded_slots_are_zero():
    """The RAW literal's pad slots must be zero.

    (The old version of this test asserted on ``IDEAL_SC_LOCAL``, which templates.py
    multiplies by ``IDEAL_SC_MASK`` at import time -- so it was true by construction
    and could never fail no matter what was typed in the literal. Assert on the
    pre-mask source instead.)
    """
    raw = torch.tensor(_IDEAL_SC_LOCAL_LIST, dtype=torch.float64)
    assert raw.shape == (N_AA, MAX_SC, 3), "literal has drifted from [20, MAX_SC, 3]"
    pad = ~IDEAL_SC_MASK
    assert (raw[pad] == 0).all(), (
        "the raw literal carries geometry in slots the mask calls padding -- "
        "the literal and instantiate.sidechain_atoms() disagree on atom counts"
    )
    # and the literal must NOT be zero where the mask says there IS an atom
    assert (raw[IDEAL_SC_MASK].abs().sum(-1) > 1e-6).all(), "a real atom slot is all-zero"


def test_raw_literal_survives_the_masking_step():
    """masking must be a no-op on a correct literal (guards a silently truncated row)."""
    raw = torch.tensor(_IDEAL_SC_LOCAL_LIST, dtype=torch.float32)
    assert torch.equal(raw, IDEAL_SC_LOCAL)


# --------------------------------------------------------------------------
# mask contract — must agree with instantiate.py exactly
# --------------------------------------------------------------------------
def test_gly_row_all_false():
    assert not IDEAL_SC_MASK[STD_AA_3.index("GLY")].any()


def test_ala_has_exactly_one_atom_and_it_is_cb():
    i = STD_AA_3.index("ALA")
    assert int(IDEAL_SC_MASK[i].sum()) == 1
    assert sidechain_atoms("ALA") == ["CB"]


def test_trp_saturates_max_sc():
    i = STD_AA_3.index("TRP")
    assert int(IDEAL_SC_MASK[i].sum()) == MAX_SC == 10


def test_mask_matches_instantiate_for_every_restype():
    expected = sidechain_mask(STD_AA_3)  # [20, MAX_SC], row order == STD_AA_3
    assert torch.equal(IDEAL_SC_MASK, expected)


# --------------------------------------------------------------------------
# the frame convention itself (this tests frames.py, and is a PRECONDITION of
# the table tests below — it says nothing about the table)
# --------------------------------------------------------------------------
def test_backbone_maps_to_canonical_frame_axes():
    """Rebuilding the frame from a residue's own N/CA/C must send CA->origin,
    C onto +e1, and N into the +e2 half of the e1-e2 plane."""
    n = torch.tensor([1.0, 2.0, 3.0])
    ca = torch.tensor([2.0, 2.5, 3.5])
    c = torch.tensor([3.0, 2.0, 3.0])
    R, t = build_frame(n, ca, c)
    local = to_local(torch.stack([n, ca, c]), R, t)
    assert torch.allclose(local[1], torch.zeros(3), atol=1e-5)      # CA at origin
    assert local[2][1].abs() < 1e-5 and local[2][2].abs() < 1e-5    # C along e1
    assert local[2][0] > 0
    assert local[0][2].abs() < 1e-5                                 # N in e1-e2 plane
    assert local[0][1] > 0                                          # ...on the +e2 side
    assert torch.linalg.det(R) > 0                                  # right-handed


def test_backbone_local_is_derived_not_assumed():
    """``_ideal_backbone_local`` must be pose-independent: the local N/C are a
    property of the CONVENTION, not of where we happened to plant the backbone."""
    ref = _ideal_backbone_local(seed=0)
    for seed in range(1, 6):
        got = _ideal_backbone_local(seed=seed)
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=1e-9), "local backbone depends on global pose (!)"
    n, ca, c = ref
    assert torch.allclose(ca, torch.zeros(3, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(c, torch.tensor([D_CA_C, 0.0, 0.0], dtype=torch.float64), atol=1e-6)
    assert n[2].abs() < 1e-6 and n[1] > 0


# --------------------------------------------------------------------------
# 1. BOND CONNECTIVITY  — kills column scrambling
# --------------------------------------------------------------------------
def test_bond_connectivity():
    check_bond_connectivity(IDEAL_SC_LOCAL)


def test_ca_cb_bond():
    check_ca_cb_bond(IDEAL_SC_LOCAL)


def test_pro_ring_closes_through_backbone_n():
    check_pro_ring_closure(IDEAL_SC_LOCAL)


@pytest.mark.parametrize("restype", [r for r in STD_AA_3 if SIDECHAIN_BONDS[r]])
def test_bond_connectivity_per_residue(restype):
    """Per-residue view of the same check, so a failure names the residue."""
    i = STD_AA_3.index(restype)
    names = sidechain_atoms(restype)
    for a, b in SIDECHAIN_BONDS[restype]:
        d = torch.linalg.norm(IDEAL_SC_LOCAL[i, names.index(a)] - IDEAL_SC_LOCAL[i, names.index(b)]).item()
        lo, hi = BOND_RANGE[_bond_class(a, b)]
        assert lo <= d <= hi, f"{restype} {a}-{b}: {d:.4f} A outside [{lo}, {hi}]"


def test_bond_table_covers_every_sidechain_atom():
    """Guard the guard: a bond list that skips atoms would let those columns be
    scrambled undetected. Every non-CB side-chain atom must appear in >= 1 bond,
    and the bond graph must connect the whole side chain to CB."""
    for restype in NON_GLY:
        names = sidechain_atoms(restype)
        adj = {a: set() for a in names}
        for a, b in SIDECHAIN_BONDS[restype]:
            assert a in adj and b in adj, f"{restype}: bond ({a},{b}) names an unknown atom"
            adj[a].add(b)
            adj[b].add(a)
        seen, stack = {"CB"}, ["CB"]
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == set(names), f"{restype}: bond graph does not reach {sorted(set(names) - seen)}"


# --------------------------------------------------------------------------
# 2. CHIRALITY — kills the z-flip / any reflection
# --------------------------------------------------------------------------
def test_chirality_is_l_amino_acid():
    check_chirality(IDEAL_SC_LOCAL)


def test_chirality_sign_agrees_with_the_external_virtual_cb_reference():
    """Sanity of the reference itself: the AF2 virtual-CB constants (fitted on real
    L-amino acids, independent of our CCD extraction) must give det > 0 too."""
    n_local, _, c_local = _ideal_backbone_local()
    cb = _virtual_cb_local()
    det = torch.linalg.det(torch.stack([n_local, c_local, cb])).item()
    assert det > 0.5, f"reference virtual CB has det={det:.4f}; the L/D sign convention is wrong"


# --------------------------------------------------------------------------
# 3. FRAME-CONVENTION PIN — kills a permuted / rotated basis
# --------------------------------------------------------------------------
def test_ala_cb_pinned_to_analytic_virtual_cb_direction():
    """ALA's CB direction, in frames.py's basis, must match the analytic
    L-amino-acid virtual CB to within CB_PIN_TOL_DEG. Derived from the AF2
    virtual-CB constants + ideal backbone geometry, NOT copied from the table."""
    check_cb_frame_pin(IDEAL_SC_LOCAL, restypes=("ALA",))


@pytest.mark.parametrize("restype", [r for r in NON_GLY if r != "PRO"])
def test_every_cb_agrees_with_the_analytic_direction(restype):
    """The same pin for all 18 non-GLY/non-PRO residues (CB is a backbone-determined
    atom, so its local direction is residue-independent). PRO is excluded: its
    pyrrolidine ring pulls CB ~7 deg off the unconstrained ideal."""
    check_cb_frame_pin(IDEAL_SC_LOCAL, restypes=(restype,))


def test_pro_cb_is_ring_strained_but_still_in_the_right_octant():
    check_cb_frame_pin(IDEAL_SC_LOCAL, restypes=("PRO",), tol_deg=10.0)


# --------------------------------------------------------------------------
# NON-VACUITY: every check above must actually FAIL on a corrupted table.
# --------------------------------------------------------------------------
_MUTATIONS = {
    "permute_basis": mutate_permute_basis,
    "scramble_columns": mutate_scramble_columns,
    "flip_z": mutate_flip_z,
}

# which check is *required* to catch which mutation
_KILLS = {
    "permute_basis": check_cb_frame_pin,
    "scramble_columns": check_bond_connectivity,
    "flip_z": check_chirality,
}


@pytest.mark.parametrize("name", sorted(_KILLS))
def test_checks_are_non_vacuous(name):
    """THE POINT OF THIS FILE. Apply the corruption in-process, re-run the check
    that is supposed to own it, and assert it raises. If it does not, the check is
    decoration and the table has no guarantee."""
    bad = _MUTATIONS[name](IDEAL_SC_LOCAL)
    assert not torch.equal(bad, IDEAL_SC_LOCAL), f"{name} did not change the table"
    with pytest.raises(AssertionError):
        _KILLS[name](bad)


def test_mutations_are_invisible_to_the_old_invariant_style_checks():
    """Documents WHY the new checks were needed: norms, eigenvalue ratios and
    finiteness -- the entire old geometric suite -- are blind to all three."""

    def old_style(t):
        for r in NON_GLY:  # ||CB|| in [1.4, 1.7]
            i = STD_AA_3.index(r)
            assert 1.4 <= torch.linalg.norm(t[i, 0]).item() <= 1.7
        for r in LARGE:  # anisotropy eigenvalue ratio > 1.5
            i = STD_AA_3.index(r)
            k = int(IDEAL_SC_MASK[i].sum())
            pts = t[i, :k].double()
            cen = pts - pts.mean(0, keepdim=True)
            ev = torch.linalg.eigvalsh(cen.T @ cen / k).clamp_min(0)
            assert (ev.max() / ev.min().clamp_min(1e-12)).item() > 1.5
        assert torch.isfinite(t).all()

    old_style(IDEAL_SC_LOCAL)  # passes on the good table
    for name, mut in _MUTATIONS.items():
        old_style(mut(IDEAL_SC_LOCAL))  # ...and on every corrupted one, too


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_full_suite_of_new_checks_catches_every_mutation(name):
    """Belt and braces: at least one of the four geometric checks must trip on each
    mutation (not just the one nominally assigned to it)."""
    bad = _MUTATIONS[name](IDEAL_SC_LOCAL)
    tripped = []
    for check in (check_bond_connectivity, check_ca_cb_bond, check_pro_ring_closure,
                  check_chirality, check_cb_frame_pin):
        try:
            check(bad)
        except AssertionError:
            tripped.append(check.__name__)
    assert tripped, f"{name} passed ALL geometric checks -- the suite is still vacuous"


# --------------------------------------------------------------------------
# ANISOTROPY — the entire point of paragraph 221 (necessary, NOT sufficient:
# see test_mutations_are_invisible_to_the_old_invariant_style_checks)
# --------------------------------------------------------------------------
def _eig_ratio(restype: str) -> float:
    i = STD_AA_3.index(restype)
    k = int(IDEAL_SC_MASK[i].sum())
    pts = IDEAL_SC_LOCAL[i, :k].double()
    centered = pts - pts.mean(0, keepdim=True)
    cov = centered.T @ centered / k
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    return (ev.max() / ev.min().clamp_min(1e-12)).item()


@pytest.mark.parametrize("restype", LARGE)
def test_template_is_anisotropic(restype):
    """An isotropic Gaussian is rotation-invariant, so F_hat @ eps carries no
    orientation signal. The ideal template must NOT be isotropic."""
    ratio = _eig_ratio(restype)
    assert ratio > 1.5, f"{restype}: covariance eigenvalue ratio {ratio:.2f} <= 1.5"


def test_template_is_not_the_zero_or_isotropic_baseline():
    """Regression guard against silently reverting to the mu_ideal-deleted path."""
    non_gly = [i for i, r in enumerate(STD_AA_3) if r != "GLY"]
    assert IDEAL_SC_LOCAL[non_gly].abs().sum() > 0


# --------------------------------------------------------------------------
# ideal_template() helper
# --------------------------------------------------------------------------
def test_ideal_template_lookup_matches_table():
    idx = torch.arange(N_AA)
    coords, mask = ideal_template(idx)
    assert torch.equal(coords, IDEAL_SC_LOCAL)
    assert torch.equal(mask, IDEAL_SC_MASK)


def test_ideal_template_batched_shapes():
    idx = torch.randint(0, N_AA, (2, 7))
    coords, mask = ideal_template(idx)
    assert coords.shape == (2, 7, MAX_SC, 3)
    assert mask.shape == (2, 7, MAX_SC)
    assert coords.dtype == torch.float32 and mask.dtype == torch.bool


def test_ideal_template_out_of_range_is_empty_sidechain():
    """Matches instantiate_from_type_indices, which maps out-of-range -> GLY."""
    coords, mask = ideal_template(torch.tensor([-1, 99]))
    assert not mask.any()
    assert (coords == 0).all()


def test_ideal_template_mask_matches_instantiate_masks():
    """Cross-check the helper against the instantiation used by S_phi."""
    from pxdesign_train.sidechain.instantiate import instantiate_from_type_indices

    idx = torch.arange(N_AA)
    _, inst_mask = instantiate_from_type_indices(idx)
    _, tmpl_mask = ideal_template(idx)
    assert torch.equal(inst_mask, tmpl_mask)


def test_no_gt_coordinate_argument():
    """Leakage rule: the template depends on residue TYPE only."""
    import inspect

    params = list(inspect.signature(ideal_template).parameters)
    assert params == ["type_idx"], f"unexpected params {params}"


# --------------------------------------------------------------------------
# provenance: the generator that produced the literal is committed and importable
# --------------------------------------------------------------------------
def test_extraction_generator_exists_and_uses_frames_py():
    """The drift-safety claim rests on the extraction artifact. It must be in-tree,
    and it must build the frame with frames.py rather than re-deriving Gram-Schmidt."""
    from pathlib import Path

    gen = Path(__file__).resolve().parent.parent / "scripts" / "build_sidechain_templates.py"
    assert gen.is_file(), "scripts/build_sidechain_templates.py is missing"
    src = gen.read_text()
    assert "from pxdesign_train.sidechain.frames import build_frame, to_local" in src
    assert "sidechain_atoms(restype)" in src, "generator must use instantiate's column order"
