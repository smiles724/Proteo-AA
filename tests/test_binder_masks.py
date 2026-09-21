"""Four masks, four meanings. Conflating any two of them leaks the answer.

The sequence head predicts binder identities, so the failure mode is not a
crash: it is an adapter that learns to read a channel that will not exist at
inference, and a designability number nobody can reproduce. Each test below
is one way that channel opens.
"""
import pytest
import torch

from pxf import atom37
from pxf.couple.binder_masks import BinderMaskSet, audit_leakage, build_masks
from pxf.couple.binder_residual import ChainRoles


def _aatype(length, value=7):
    return torch.full((1, length), value, dtype=torch.long)


def _roles(n_target=6, n_binder=6):
    return ChainRoles.from_lengths(n_target, n_binder)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------ construction


def test_the_target_is_never_masked_and_never_supervised():
    roles = _roles()
    m = build_masks(roles, _aatype(12), mask_fraction=1.0, generator=_gen())
    target = roles.target.reshape(1, -1)
    assert bool((m.seq_mlm_mask[target] == 1).all()), "a target identity was hidden"
    assert float(m.seq_supervision[target].sum()) == 0.0


def test_seq_mask_is_all_ones_so_the_target_stays_in_the_encoder():
    """Excluding the target by zeroing seq_mask would remove it from the model.

    That is a different experiment: `complex` context becomes `binder_only`.
    The exclusion must come from seq_mlm_mask.
    """
    m = build_masks(_roles(), _aatype(12), mask_fraction=0.5, generator=_gen())
    assert float(m.seq_mask.min()) == 1.0


def test_supervision_follows_upstreams_kept_convention():
    """seq_mlm_mask is 1 where KEPT; scored is (1 - mask) * seq_mask."""
    roles = _roles(2, 2)
    m = build_masks(roles, _aatype(4), mask_fraction=1.0, generator=_gen())
    assert m.seq_mlm_mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert m.seq_supervision.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_mask_fraction_zero_supervises_nothing_and_the_audit_says_so():
    roles = _roles()
    m = build_masks(roles, _aatype(12), mask_fraction=0.0, generator=_gen())
    report = audit_leakage(m)
    assert report["n_supervised"] == 0
    assert any("identically zero" in p for p in report["problems"])


def test_sc_supervision_covers_every_binder_row_not_only_hidden_ones():
    """Restricting L_SC to sequence-masked rows would silently drop supervision."""
    roles = _roles(6, 6)
    m = build_masks(roles, _aatype(12), mask_fraction=0.25, generator=_gen(3))
    assert float(m.sc_supervision.sum()) == roles.n_binder
    assert float(m.seq_supervision.sum()) < roles.n_binder


def test_a_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="roles describe"):
        build_masks(_roles(6, 6), _aatype(10), mask_fraction=0.5)


# ------------------------------------------------------------- route 1 & 2


def test_hidden_identities_become_unknown_in_the_encoder_input():
    roles = _roles()
    m = build_masks(roles, _aatype(12, value=7), mask_fraction=1.0, generator=_gen())
    hidden = m.seq_supervision.bool()
    assert bool((m.aatype_encoder[hidden] == atom37.UNKNOWN_AA_INDEX).all())
    # ... and the labels are untouched.
    assert bool((m.aatype_true == 7).all())


def test_a_hidden_residue_never_shows_its_side_chain():
    """A side chain identifies the residue almost exactly.

    Hiding the letter while showing the atoms is the leak that looks most
    like a bug fix -- "the model needs context" -- and is not one.
    """
    roles = _roles()
    m = build_masks(
        roles, _aatype(12), mask_fraction=1.0,
        binder_sidechain_dropout=0.0,  # would otherwise keep them visible
        generator=_gen(),
    )
    hidden = m.seq_supervision.bool()
    assert float(m.sidechain_visible[hidden].sum()) == 0.0
    assert audit_leakage(m)["pass"]


def test_the_audit_catches_a_revealed_identity():
    roles = _roles()
    m = build_masks(roles, _aatype(12), mask_fraction=1.0, generator=_gen())
    leaked = m.aatype_encoder.clone()
    leaked[m.seq_supervision.bool()] = 7  # put the answer back
    broken = BinderMaskSet(
        roles=roles, seq_mask=m.seq_mask, seq_mlm_mask=m.seq_mlm_mask,
        sidechain_visible=m.sidechain_visible, aatype_encoder=leaked,
        aatype_true=m.aatype_true,
    )
    report = audit_leakage(broken)
    assert not report["pass"]
    assert any("ROUTE 1" in p for p in report["problems"])


def test_the_audit_catches_a_revealed_side_chain():
    roles = _roles()
    m = build_masks(roles, _aatype(12), mask_fraction=1.0, generator=_gen())
    broken = BinderMaskSet(
        roles=roles, seq_mask=m.seq_mask, seq_mlm_mask=m.seq_mlm_mask,
        sidechain_visible=torch.ones_like(m.sidechain_visible),
        aatype_encoder=m.aatype_encoder, aatype_true=m.aatype_true,
    )
    report = audit_leakage(broken)
    assert not report["pass"]
    assert any("ROUTE 2" in p for p in report["problems"])


def test_the_audit_catches_a_supervised_target_row():
    roles = _roles(6, 6)
    m = build_masks(roles, _aatype(12), mask_fraction=1.0, generator=_gen())
    mlm = m.seq_mlm_mask.clone()
    mlm[0, 0] = 0.0  # hide a TARGET identity -> it becomes supervised
    broken = BinderMaskSet(
        roles=roles, seq_mask=m.seq_mask, seq_mlm_mask=mlm,
        sidechain_visible=m.sidechain_visible,
        aatype_encoder=m.aatype_encoder, aatype_true=m.aatype_true,
    )
    report = audit_leakage(broken)
    assert not report["pass"]
    assert any("TARGET row" in p for p in report["problems"])


# ----------------------------------------------------------------- route 3


def test_pxdesign_conditioning_is_reported_unchecked_when_not_supplied():
    """Silence is not a pass: a_token can leak while FaMPNN's inputs are clean."""
    m = build_masks(_roles(), _aatype(12), mask_fraction=0.5, generator=_gen())
    report = audit_leakage(m)
    assert report["pxdesign_checked"] is False
    assert "a_token" in report["pxdesign_note"]


def test_the_audit_catches_native_identities_in_the_backbone_conditioning():
    roles = _roles()
    m = build_masks(roles, _aatype(12, value=7), mask_fraction=1.0, generator=_gen())
    # The featurizer failed to scrub the design region.
    report = audit_leakage(m, pxdesign_aatype=torch.full((12,), 7, dtype=torch.long))
    assert not report["pass"]
    assert any("ROUTE 3" in p for p in report["problems"])


def test_a_scrubbed_design_region_passes_route_3():
    roles = _roles()
    m = build_masks(roles, _aatype(12, value=7), mask_fraction=1.0, generator=_gen())
    scrubbed = torch.full((12,), 7, dtype=torch.long)
    scrubbed[roles.binder] = atom37.UNKNOWN_AA_INDEX
    report = audit_leakage(m, pxdesign_aatype=scrubbed)
    assert report["pass"], report["problems"]
    assert report["pxdesign_revealed"] == 0


def test_a_design_region_that_disagrees_with_the_binder_role_is_caught():
    roles = _roles(6, 6)
    m = build_masks(roles, _aatype(12), mask_fraction=1.0, generator=_gen())
    scrubbed = torch.full((12,), atom37.UNKNOWN_AA_INDEX, dtype=torch.long)
    wrong = torch.zeros(12, dtype=torch.bool)
    wrong[:6] = True  # the TARGET marked as the design region
    report = audit_leakage(
        m, pxdesign_aatype=scrubbed, pxdesign_design_mask=wrong
    )
    assert not report["pass"]
    assert any("disagree" in p for p in report["problems"])
