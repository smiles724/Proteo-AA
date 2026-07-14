"""The mu_ideal lookup must be genuinely swappable, not swappable-in-principle.

The shipped table is the wwPDB CCD *ideal* geometry: correct bond lengths and angles, but
ONE canonical chi. Measured against real side chains (1cse chain B, residue-local frame):

    ALA  (no chi)            0.07 A from truth      <- essentially exact
    PRO  (ring-constrained)  0.68 A
    THR / LEU / SER          1.0 - 1.2 A
    HIS / GLU / ARG / TYR    2.4 - 3.4 A            <- the chi is arbitrary

So the template is chemically right and conformationally arbitrary, and a rotamer-statistics
table is exactly what would fix the part that is wrong. These tests pin that such a table --
deterministic, STOCHASTIC (sampled from a distribution), or BACKBONE-DEPENDENT (phi/psi
conditioned) -- can be dropped in with a single call and NOTHING in model.py / init.py /
cogenerate.py has to change.

A "swappable interface" nobody ever swapped is a claim, not a property. These tests swap it.
"""
import inspect

import pytest
import torch

from pxdesign_train.sidechain import templates
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


def test_default_is_the_ccd_table():
    tix, mask = _chain()
    templates.set_ideal_template_provider(None)
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
    """Jiaming's rotamer statistics give a DISTRIBUTION. For a generative model, sampling a
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
    """The shipped CCD provider takes type only; callers that have no backbone must still work."""
    tix, mask = _chain()
    templates.set_ideal_template_provider(None)
    y = template_init_local(tix, mask, sigma_T=0.0)          # no backbone passed
    assert torch.allclose(y[mask], templates.IDEAL_SC_LOCAL[tix][mask], atol=1e-6)


# --- the contract itself ------------------------------------------------------
def test_provider_contract_has_no_channel_for_gt_side_chain_coordinates():
    """The leakage rule is structural: there is no parameter GT coords could arrive through."""
    sig = inspect.signature(templates._default_provider)
    assert set(sig.parameters) == {"type_idx", "generator", "backbone"}
    forbidden = {"gt", "gt_coords", "sc_gt_local", "y_gt", "x_sc_gt", "target", "ground_truth"}
    assert not (set(sig.parameters) & forbidden)


def test_restoring_the_default_really_restores_it():
    templates.set_ideal_template_provider(lambda *a, **k: (None, None))
    assert not templates.is_default_provider()
    templates.set_ideal_template_provider(None)
    assert templates.is_default_provider()
