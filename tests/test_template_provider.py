"""The mu_ideal lookup must be genuinely swappable, not swappable-in-principle.

HISTORY. This file was written while the only mu_ideal available was the wwPDB CCD *ideal*
conformer: correct bond lengths and angles, but ONE arbitrary chi, and therefore 2-3 A from
real side chains on the multi-chi residues. The tests below pinned that a better table --
deterministic, STOCHASTIC (sampled from a distribution), or BACKBONE-DEPENDENT (phi/psi
conditioned) -- could be dropped in with a single call, with NOTHING in model.py / init.py /
cogenerate.py having to change.

That table has since landed (Overleaf 0714 appendix; Dunbrack BBDEP2010), it is now the
DEFAULT provider, and the swap hole it was designed for turned out to be one kwarg short --
phi/psi need the neighbouring residues' C and N, which the per-residue `backbone` argument
cannot carry. See tests/test_template_construction.py for the spec-conformance tests.

What survives here, and still matters: the provider really is swappable, the generator really
is threaded through, and the contract really has no channel for GT side-chain coordinates.
A "swappable interface" nobody ever swapped is a claim, not a property. These tests swap it.
"""
import inspect

import pytest
import torch

from pxdesign_train.sidechain import rotamers, templates
from pxdesign_train.sidechain.init import DEFAULT_SIGMA_T, template_init_local
from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3, sidechain_mask


@pytest.fixture(autouse=True)
def _restore_default_provider():
    """A leaked provider would silently corrupt every later test in the process."""
    yield
    templates.set_ideal_template_provider(None)
    assert templates.is_default_provider()


def _chain(restypes=("TRP", "ALA", "GLY", "ARG")):
    tix = torch.tensor([STD_AA_3.index(r) for r in restypes], dtype=torch.long)
    return tix, sidechain_mask(list(restypes))


def test_default_is_the_0714_rotamer_construction_not_the_static_table():
    """Since 2026-07-14 the default mu_ideal is BuildSC(a, G_ideal, chi(a, phi, psi)).

    The static CCD table is still there, and is still what you get with no rotamer
    library on disk, but it is no longer the default: it has one arbitrary chi.
    """
    tix, mask = _chain()
    templates.set_ideal_template_provider(None)
    assert templates.get_ideal_template_provider() is templates.dunbrack_mode_provider

    mu, m = templates.get_ideal_template_provider()(tix)
    assert torch.equal(m, templates.IDEAL_SC_MASK[tix])
    if rotamers.available():
        # ARG is a 4-chi residue: the rotamer template must differ from the CCD conformer.
        arg = list(_chain()[0]).index(STD_AA_3.index("ARG"))
        assert not torch.allclose(mu[arg], templates.IDEAL_SC_LOCAL[tix][arg], atol=1e-3)
    else:
        assert torch.equal(mu, templates.IDEAL_SC_LOCAL[tix])   # graceful CCD fallback


def test_ccd_table_is_still_reachable_as_the_baseline():
    tix, mask = _chain()
    templates.set_provider_by_name("ccd")
    mu, m = templates.get_ideal_template_provider()(tix)
    assert torch.equal(mu, templates.IDEAL_SC_LOCAL[tix])
    assert torch.equal(m, templates.IDEAL_SC_MASK[tix])


# --- L1: swap a different STATIC table ---------------------------------------
def test_a_replacement_static_table_is_actually_used():
    tix, mask = _chain()
    sentinel = 7.0

    def provider(type_idx, *, generator=None, backbone=None):
        mu = torch.full((*type_idx.shape, MAX_SC, 3), sentinel)
        m = templates.IDEAL_SC_MASK[type_idx]
        return mu, m

    templates.set_ideal_template_provider(provider)
    y = template_init_local(tix, mask, sigma_T=0.0)   # sigma_T=0 -> the template itself
    assert torch.allclose(y[mask], torch.full_like(y[mask], sentinel)), (
        "the registered provider was ignored -- the hole is not actually open"
    )


# --- L2: a STOCHASTIC provider (sample a rotamer from a distribution) ---------
def test_a_stochastic_provider_receives_the_generator():
    """Rotamer statistics give a DISTRIBUTION. For a generative model, sampling a
    rotamer is arguably more correct than always starting from the same canonical chi -- so
    the provider must be able to draw, reproducibly, from the caller's generator."""
    tix, mask = _chain()
    seen = []

    def provider(type_idx, *, generator=None, backbone=None):
        seen.append(generator)
        assert generator is not None, "provider got no generator -> cannot sample reproducibly"
        mu = torch.randn(*type_idx.shape, MAX_SC, 3, generator=generator)
        return mu, templates.IDEAL_SC_MASK[type_idx]

    templates.set_ideal_template_provider(provider)
    a = template_init_local(tix, mask, sigma_T=0.0, generator=torch.Generator().manual_seed(0))
    b = template_init_local(tix, mask, sigma_T=0.0, generator=torch.Generator().manual_seed(0))
    c = template_init_local(tix, mask, sigma_T=0.0, generator=torch.Generator().manual_seed(1))

    assert seen and all(g is not None for g in seen)
    assert torch.equal(a, b), "same seed -> same draw; the provider is not reproducible"
    assert not torch.equal(a, c), "different seed -> same draw; the generator is not being used"


# --- L3: a BACKBONE-DEPENDENT provider (phi/psi-conditioned rotamer library) ---
def test_a_backbone_dependent_provider_receives_the_backbone():
    tix, mask = _chain()
    bb = torch.randn(len(tix), 4, 3)          # N, CA, C, O in the residue-local frame
    got = {}

    def provider(type_idx, *, generator=None, backbone=None):
        got["backbone"] = backbone
        assert backbone is not None, "provider got no backbone -> cannot condition on phi/psi"
        # a silly but observable dependence, so we can prove it flows through
        mu = templates.IDEAL_SC_LOCAL[type_idx] + backbone[..., :1, :].mean()
        return mu, templates.IDEAL_SC_MASK[type_idx]

    templates.set_ideal_template_provider(provider)
    y = template_init_local(tix, mask, sigma_T=0.0, backbone=bb)
    assert got["backbone"] is bb
    expected = (templates.IDEAL_SC_LOCAL[tix] + bb[..., :1, :].mean())
    assert torch.allclose(y[mask], expected[mask], atol=1e-5)


def test_backbone_defaults_to_none_so_type_only_providers_still_work():
    """The CCD provider takes type only; callers that have no backbone must still work."""
    tix, mask = _chain()
    templates.set_provider_by_name("ccd")
    y = template_init_local(tix, mask, sigma_T=0.0)          # no backbone passed
    assert torch.allclose(y[mask], templates.IDEAL_SC_LOCAL[tix][mask], atol=1e-6)


# --- the contract itself ------------------------------------------------------
def test_provider_contract_has_no_channel_for_gt_side_chain_coordinates():
    """The leakage rule is structural: there is no parameter GT coords could arrive through.

    phi/psi were added by the 0714 appendix. They are dihedrals of the PREDICTED backbone,
    i.e. inference-available, and carry no side-chain information -- so they widen the
    contract without opening a leak. The set is asserted exactly so that a genuine GT
    channel cannot be slipped in under a harmless-looking name.
    """
    for fn in (templates.ccd_provider, templates.dunbrack_provider, templates.dunbrack_mode_provider):
        params = set(inspect.signature(fn).parameters)
        params.discard("select")                       # dunbrack_provider's mode switch
        assert params == {"type_idx", "generator", "backbone", "phi", "psi"}, fn
        forbidden = {"gt", "gt_coords", "sc_gt_local", "y_gt", "x_sc_gt", "target", "ground_truth"}
        assert not (params & forbidden)


def test_restoring_the_default_really_restores_it():
    templates.set_ideal_template_provider(lambda *a, **k: (None, None))
    assert not templates.is_default_provider()
    templates.set_ideal_template_provider(None)
    assert templates.is_default_provider()
