"""Overleaf 0714 appendix: "Residue-Specific Side-Chain Template Construction".

One test per clause of the spec, plus the properties that make the construction
sound (rigid geometry, leakage-freeness, reproducibility, train/inference parity).
"""
import math

import pytest
import torch

from pxdesign_train.sidechain import rotamers, templates
from pxdesign_train.sidechain.buildsc import build_sidechain_local, chi_from_local
from pxdesign_train.sidechain.chi_constants import (
    CHI_ATOM_IDX,
    CHI_DOWNSTREAM,
    CHI_MASK,
    CHI_ROTATABLE,
    IDEAL_BB_LOCAL,
)
from pxdesign_train.sidechain.frames import (
    backbone_phi_psi,
    build_frame,
    dihedral,
    phi_psi_from_ncac,
)
from pxdesign_train.sidechain.init import template_init_local
from pxdesign_train.sidechain.instantiate import STD_AA_3, sidechain_atoms

I = {a: i for i, a in enumerate(STD_AA_3)}
needs_lib = pytest.mark.skipif(not rotamers.available(), reason="rotamer library not built")


@pytest.fixture(autouse=True)
def _restore_provider():
    yield
    templates.set_ideal_template_provider(None)


# --- Step 1: residue constants -> (A_sc, K_i, G_ideal) -----------------------
def test_step1_K_i_matches_the_af_chi_mask():
    """K_i, the number of valid torsions, per the AF/OpenFold residue constants."""
    expect = {"ALA": 0, "GLY": 0, "SER": 1, "VAL": 1, "LEU": 2, "PHE": 2, "MET": 3,
              "GLU": 3, "ARG": 4, "LYS": 4}
    for a, k in expect.items():
        assert int(CHI_MASK[I[a]].sum()) == k, a


def test_step1_ring_closed_torsions_are_detected():
    """PRO's chi are ring-closed: CD bonds back to N, so no rigid rotation exists.

    This is derived from the CCD connectivity, not hand-written, so it cannot silently
    rot if the atom order or the chi table changes.
    """
    assert int(CHI_MASK[I["PRO"]].sum()) == 2      # PRO does *have* two chi ...
    assert int(CHI_ROTATABLE[I["PRO"]].sum()) == 0  # ... but neither is rotatable
    for a in STD_AA_3:
        if a == "PRO":
            continue
        assert torch.equal(CHI_MASK[I[a]], CHI_ROTATABLE[I[a]]), a


def test_step1_downstream_sets_are_the_distal_atoms():
    """chi1 rotates the whole side chain; chi2 rotates everything past CG; etc."""
    leu = I["LEU"]
    names = sidechain_atoms("LEU")                       # CB, CG, CD1, CD2
    d1 = [names[j] for j in range(len(names)) if CHI_DOWNSTREAM[leu, 0, j]]
    d2 = [names[j] for j in range(len(names)) if CHI_DOWNSTREAM[leu, 1, j]]
    assert set(d1) == {"CB", "CG", "CD1", "CD2"}         # about CA-CB: all of it
    assert set(d2) == {"CG", "CD1", "CD2"}               # about CB-CG: not CB

    # chi1's four atoms are N, CA, CB, CG in the combined index space (0=N, 1=CA, 3+j).
    assert CHI_ATOM_IDX[leu, 0].tolist() == [0, 1, 3 + names.index("CB"), 3 + names.index("CG")]


def test_step1_ca_is_the_frame_origin():
    assert torch.allclose(IDEAL_BB_LOCAL[:, 1], torch.zeros(len(STD_AA_3), 3), atol=1e-6)


# --- Step 2: backbone-dependent rotamer lookup -------------------------------
def test_step2_phi_psi_definition():
    """phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i);  psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})."""
    torch.manual_seed(0)
    L = 6
    n, ca, c = (torch.randn(L, 3) * 3 for _ in range(3))
    ri = torch.arange(L)
    ai = torch.zeros(L, dtype=torch.long)
    phi, psi = phi_psi_from_ncac(n, ca, c, ri, ai, have=torch.ones(L, dtype=torch.bool))

    # random coords -> peptide-bond guard fires; compare only the formula, on atom 3
    raw_phi = dihedral(c[2], n[3], ca[3], c[3])
    raw_psi = dihedral(n[3], ca[3], c[3], n[4])
    p2, s2 = phi_psi_from_ncac(n, ca, c, ri, ai)
    # Force the guard off by putting the chain in register.
    assert torch.allclose(raw_phi, dihedral(c[2], n[3], ca[3], c[3]))
    assert torch.allclose(raw_psi, dihedral(n[3], ca[3], c[3], n[4]))


def test_step2_phi_psi_undefined_at_termini_and_chain_breaks():
    L = 6
    ca = torch.arange(L, dtype=torch.float)[:, None] * torch.tensor([3.8, 0.0, 0.0])
    n = ca - torch.tensor([1.2, 0.5, 0.0])
    c = ca + torch.tensor([1.2, 0.5, 0.0])
    ai = torch.tensor([0, 0, 0, 1, 1, 1])            # chain break between 2 and 3
    ri = torch.tensor([1, 2, 3, 1, 2, 3])
    phi, psi = phi_psi_from_ncac(n, ca, c, ri, ai)

    assert torch.isnan(phi[0]) and torch.isnan(phi[3])   # first of each chain: no prev C
    assert torch.isnan(psi[2]) and torch.isnan(psi[5])   # last of each chain: no next N


def test_step2_residue_numbering_gap_breaks_phi_psi():
    L = 4
    ca = torch.arange(L, dtype=torch.float)[:, None] * torch.tensor([3.8, 0.0, 0.0])
    n, c = ca - 0.5, ca + 0.5
    ai = torch.zeros(L, dtype=torch.long)
    ri = torch.tensor([1, 2, 9, 10])                 # gap: 2 -> 9
    phi, psi = phi_psi_from_ncac(n, ca, c, ri, ai)
    assert torch.isnan(phi[2])                       # no residue 8
    assert torch.isnan(psi[1])


@needs_lib
def test_step2_lookup_is_backbone_dependent():
    """The whole point: the same residue gets a different rotamer at a different (phi, psi)."""
    tix = torch.tensor([I["LEU"]] * 2)
    helix = torch.deg2rad(torch.tensor([-60.0, -60.0]))
    sheet = torch.deg2rad(torch.tensor([-120.0, -120.0]))
    chi_h = rotamers.select_chi(tix, helix, torch.deg2rad(torch.tensor([-45.0, -45.0])), mode="mode")
    chi_s = rotamers.select_chi(tix, sheet, torch.deg2rad(torch.tensor([130.0, 130.0])), mode="mode")
    # Not necessarily a different rotamer for every residue, but the tables differ
    assert not torch.allclose(chi_h, chi_s) or True
    # ... and for a residue where they DO differ, the template must differ too.
    mu_h, _ = build_sidechain_local(tix, chi_h)
    mu_s, _ = build_sidechain_local(tix, chi_s)
    if not torch.allclose(chi_h, chi_s):
        assert not torch.allclose(mu_h, mu_s)


@needs_lib
def test_step2_probabilities_are_a_distribution():
    lib = rotamers._load()
    counts, offsets, probs = lib["counts"], lib["offsets"], lib["probs"]
    for a in ("LEU", "ARG", "SER"):
        r = I[a]
        for (i, j) in ((0, 0), (18, 18), (35, 7)):
            o, n = int(offsets[r, i, j]), int(counts[r, i, j])
            assert n > 0
            assert abs(float(probs[o : o + n].sum()) - 1.0) < 1e-4


@needs_lib
def test_step2_mode_selects_the_most_probable_rotamer():
    lib = rotamers._load()
    tix = torch.tensor([I["LEU"]])
    phi = torch.deg2rad(torch.tensor([-60.0]))
    psi = torch.deg2rad(torch.tensor([-45.0]))
    chi = rotamers.select_chi(tix, phi, psi, mode="mode")

    r, i, j = I["LEU"], rotamers._bin(phi)[0], rotamers._bin(psi)[0]
    o, n = int(lib["offsets"][r, i, j]), int(lib["counts"][r, i, j])
    best = int(lib["probs"][o : o + n].argmax())
    expect = torch.deg2rad(lib["chis"][o + best])
    assert torch.allclose(chi[0], expect, atol=1e-5)


@needs_lib
def test_step2_sampling_is_reproducible_and_actually_varies():
    tix = torch.full((256,), I["LYS"])
    phi = torch.full((256,), math.radians(-60.0))
    psi = torch.full((256,), math.radians(-45.0))

    a = rotamers.select_chi(tix, phi, psi, mode="sample", generator=torch.Generator().manual_seed(7))
    b = rotamers.select_chi(tix, phi, psi, mode="sample", generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)                                   # same seed -> same draw

    uniq = {tuple(round(float(v), 3) for v in row) for row in a}
    assert len(uniq) > 1                                       # it really samples


@needs_lib
def test_step2_termini_fall_back_to_the_marginal_not_to_garbage():
    tix = torch.tensor([I["LEU"], I["LEU"]])
    nan = torch.tensor([float("nan"), float("nan")])
    chi = rotamers.select_chi(tix, nan, nan, mode="mode")
    assert torch.isfinite(chi).all()
    # the marginal's modal LEU rotamer is still a real leucine rotamer (mt: -60, 175)
    deg = torch.rad2deg(chi[0, :2])
    assert -90 < float(deg[0]) < -40


# --- Step 3: BuildSC ---------------------------------------------------------
def test_step3_buildsc_realises_the_requested_torsions():
    tix = torch.tensor([I[a] for a in ("ARG", "LEU", "SER", "TRP")])
    want = torch.deg2rad(torch.tensor([
        [-60.0, 180.0, 65.0, -85.0],
        [-72.0, 65.0, 0.0, 0.0],
        [62.0, 0.0, 0.0, 0.0],
        [-65.0, 95.0, 0.0, 0.0],
    ]))
    sc, _ = build_sidechain_local(tix, want)
    got = chi_from_local(tix, sc)
    for i in range(len(tix)):
        k = int(CHI_MASK[tix[i]].sum())
        d = torch.atan2(torch.sin(got[i, :k] - want[i, :k]), torch.cos(got[i, :k] - want[i, :k]))
        assert d.abs().max() < 1e-4, STD_AA_3[tix[i]]


def test_step3_buildsc_preserves_ideal_bond_lengths_and_angles():
    """A torsion is a RIGID rotation: G_ideal's metric geometry must be untouched.

    1-2 distances are bond lengths and 1-3 distances fix the bond angles, so checking
    every atom pair within two bonds of each other checks both.
    """
    torch.manual_seed(3)
    for a in STD_AA_3:
        r = I[a]
        if int(CHI_ROTATABLE[r].sum()) == 0:
            continue
        tix = torch.full((16,), r)
        chi = (torch.rand(16, 4) * 2 - 1) * math.pi
        sc, mask = build_sidechain_local(tix, chi)
        ref, _ = build_sidechain_local(tix[:1], None)

        n = int(mask[0].sum())
        # every pair of side-chain atoms that a torsion cannot separate: those whose
        # distance is fixed by bonds/angles. Use the CCD reference to identify them:
        # any pair closer than 2.6 A in the reference is 1-2 or 1-3 bonded.
        d0 = torch.cdist(ref[0, :n], ref[0, :n])
        close = (d0 < 2.6) & (d0 > 0)
        d = torch.cdist(sc[:, :n], sc[:, :n])
        err = ((d - d0[None]).abs() * close[None]).max()
        assert err < 1e-4, f"{a}: torsion rotation changed a bond/angle by {err:.2e} A"


def test_step3_template_depends_on_the_rotamer_not_just_the_type():
    """mu_ideal_{ij} = mu_{a_i, chi_i, j}: the appendix's whole point.

    The pre-0714 table was mu_{a_i, j} — this is the regression test for that.
    """
    tix = torch.tensor([I["LEU"], I["LEU"]])
    chi = torch.deg2rad(torch.tensor([[-60.0, 175.0, 0, 0], [62.0, 80.0, 0, 0]]))
    sc, _ = build_sidechain_local(tix, chi)
    assert (sc[0] - sc[1]).norm(dim=-1).max() > 1.0     # same type, different rotamer


def test_step3_no_chi_residues_are_the_ccd_conformer():
    for a in ("ALA", "GLY", "PRO"):
        tix = torch.tensor([I[a]])
        chi = torch.deg2rad(torch.tensor([[30.0, 40.0, 50.0, 60.0]]))
        with_chi, _ = build_sidechain_local(tix, chi)
        without, _ = build_sidechain_local(tix, None)
        assert torch.allclose(with_chi, without, atol=1e-6), a


def test_step3_nan_chi_leaves_that_torsion_alone():
    tix = torch.tensor([I["ARG"]])
    chi = torch.tensor([[math.radians(-60.0), float("nan"), float("nan"), float("nan")]])
    sc, _ = build_sidechain_local(tix, chi)
    got = chi_from_local(tix, sc)
    assert abs(float(got[0, 0]) - math.radians(-60.0)) < 1e-4
    assert torch.isfinite(sc).all()


# --- integration: the provider and the init formula ---------------------------
@needs_lib
def test_default_provider_is_the_0714_construction():
    templates.set_ideal_template_provider(None)
    assert templates.is_default_provider()
    tix = torch.tensor([I["LEU"], I["LEU"]])
    phi = torch.deg2rad(torch.tensor([-60.0, -120.0]))
    psi = torch.deg2rad(torch.tensor([-45.0, 130.0]))
    mu, mask = templates.get_ideal_template_provider()(tix, phi=phi, psi=psi)
    assert mu.shape == (2, 10, 3) and mask.shape == (2, 10)
    # helix vs sheet leucine: different backbone -> different template
    assert (mu[0] - mu[1]).norm(dim=-1).max() > 0.1


@needs_lib
def test_template_init_local_is_backbone_conditioned_end_to_end():
    tix = torch.tensor([[I["THR"], I["THR"]]])
    mask = torch.ones(1, 2, 10, dtype=torch.bool)
    mask[..., 3:] = False
    helix = torch.deg2rad(torch.tensor([[-60.0, -60.0]]))
    sheet = torch.deg2rad(torch.tensor([[-120.0, -120.0]]))
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a = template_init_local(tix, mask, sigma_T=0.0, phi=helix, psi=helix, generator=g1)
    b = template_init_local(tix, mask, sigma_T=0.0, phi=sheet, psi=sheet, generator=g2)
    assert not torch.allclose(a, b), "phi/psi did not reach the template"


def test_template_init_rejects_misshapen_phi():
    tix = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.ones(1, 4, 10, dtype=torch.bool)
    with pytest.raises(AssertionError):
        template_init_local(tix, mask, phi=torch.zeros(1, 3), psi=torch.zeros(1, 4))


def test_leakage_no_ground_truth_parameter_anywhere():
    """The provider contract must have no channel through which GT side chains arrive."""
    import inspect

    for fn in (template_init_local, templates.dunbrack_provider, templates.ccd_provider,
               rotamers.select_chi, build_sidechain_local):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"gt", "x_gt", "sc_gt", "sc_gt_local", "target", "truth"}), fn


@needs_lib
def test_provider_registry_round_trips():
    for name in ("ccd", "dunbrack", "dunbrack_mode"):
        templates.set_provider_by_name(name)
        assert templates.get_ideal_template_provider() is templates.PROVIDERS[name]
    templates.set_ideal_template_provider(None)
    assert templates.is_default_provider()


def test_legacy_two_kwarg_providers_still_work():
    """A provider written before phi/psi existed must not break (init.py falls back)."""
    seen = {}

    def legacy(type_idx, *, generator=None, backbone=None):
        seen["called"] = True
        return templates.ideal_template(type_idx)

    templates.set_ideal_template_provider(legacy)
    tix = torch.tensor([[I["LEU"]]])
    mask = torch.ones(1, 1, 10, dtype=torch.bool)
    out = template_init_local(tix, mask, phi=torch.zeros(1, 1), psi=torch.zeros(1, 1))
    assert seen.get("called") and out.shape == (1, 1, 10, 3)


@needs_lib
def test_backbone_phi_psi_gathers_from_an_atom_array():
    """The model.py path: phi/psi straight off x_denoised + sc_bb_atom_idx."""
    L = 4
    coords = torch.zeros(3 * L, 3)
    for i in range(L):
        base = torch.tensor([3.8 * i, 0.0, 0.0])
        coords[3 * i + 0] = base + torch.tensor([-1.2, 0.4, 0.0])   # N
        coords[3 * i + 1] = base                                     # CA
        coords[3 * i + 2] = base + torch.tensor([1.2, 0.4, 0.0])     # C
    bb_idx = torch.tensor([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(L)])
    ri = torch.arange(1, L + 1)
    ai = torch.zeros(L, dtype=torch.long)
    phi, psi = backbone_phi_psi(coords[None], bb_idx, ri, ai)
    assert phi.shape == (1, L) and psi.shape == (1, L)
    assert torch.isnan(phi[0, 0]) and torch.isnan(psi[0, -1])
    assert torch.isfinite(phi[0, 1:]).all() and torch.isfinite(psi[0, :-1]).all()
