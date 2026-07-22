"""The mismatched-residue regularizer: scope, and the ablation arms.

Two things are pinned here.

SCOPE (0722). The term applies ONLY to residues whose
predicted aa type differs from GT. Where the type is right, `L_sc^coord` already
supervises the full local geometry and the paper explicitly wants no additional
geometric term. Before 2026-07-22 the code applied clash+contact to every residue;
these tests exist so that cannot come back silently.

ARMS. Nobody has decided whether wrong-type residues get a physical term or nothing
at all, so all arms must actually be reachable and actually differ.
"""
import pytest
import torch

from pxdesign_train.model import ProtenixDesignTrain
from pxdesign_train.sidechain.physical import (
    MISMATCH_ARMS,
    clash_loss,
    contact_loss,
    physical_loss,
)


def _two_residues_one_clashing():
    """Residue 0 (atoms 0,1) sits ON the context atom; residue 1 (atoms 2,3) is far.

    So all the steric penalty belongs to residue 0, and none to residue 1.
    """
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],   # res 0 atom a  — right on top of the context atom
        [0.5, 0.0, 0.0],   # res 0 atom b
        [50.0, 0.0, 0.0],  # res 1 atom a  — nowhere near anything
        [50.5, 0.0, 0.0],  # res 1 atom b
    ]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    group = torch.tensor([[0, 0, 1, 1]])
    ctx = torch.tensor([[[0.1, 0.0, 0.0]]])          # one context atom, near res 0
    ctx_mask = torch.ones(1, 1, dtype=torch.bool)
    ctx_group = torch.tensor([[9]])                  # a different residue -> real clash
    return coords, valid, group, ctx, ctx_mask, ctx_group


# --- scope ------------------------------------------------------------------
def test_subject_mask_restricts_the_penalty_to_mismatched_residues():
    coords, valid, group, ctx, ctx_mask, ctx_group = _two_residues_one_clashing()

    everything = clash_loss(coords, valid_mask=valid, group_id=group,
                            context_coords=ctx, context_mask=ctx_mask,
                            context_group_id=ctx_group)
    # only residue 1 (the far, non-clashing one) is "mismatched"
    only_far = clash_loss(coords, valid_mask=valid, group_id=group,
                          context_coords=ctx, context_mask=ctx_mask,
                          context_group_id=ctx_group,
                          subject_mask=torch.tensor([[False, False, True, True]]))

    assert everything > 0, "the clashing residue must be penalised at all"
    assert float(only_far) == pytest.approx(0.0, abs=1e-6), (
        "residue 1 does not clash; scoping the term to it must give 0"
    )


def test_an_empty_subject_set_gives_exactly_zero():
    """Stage II/III: teacher forcing => nothing mismatches => the term must vanish.

    The Stage III objective in 0722 has no L_compat term at all, so anything other
    than 0 here would be training against an equation the paper does not have.
    """
    coords, valid, group, ctx, ctx_mask, ctx_group = _two_residues_one_clashing()
    none_subject = torch.zeros(1, 4, dtype=torch.bool)

    c = clash_loss(coords, valid_mask=valid, group_id=group, context_coords=ctx,
                   context_mask=ctx_mask, context_group_id=ctx_group,
                   subject_mask=none_subject)
    ct = contact_loss(coords, ctx, valid, ctx_mask, subject_mask=none_subject)
    assert float(c) == pytest.approx(0.0, abs=1e-8)
    assert float(ct) == pytest.approx(0.0, abs=1e-8)


def test_mismatched_residue_is_still_penalised_against_a_MATCHED_neighbour():
    """The 0722 term is ASYMMETRIC: subject = mismatched, partner = everything.

    A wrong-type residue growing into a correctly-typed neighbour's side chain is
    exactly the overlap we care about. Gating with valid_mask alone would drop it,
    because that requires BOTH ends to be in the subject set.
    """
    # two side-chain atoms, overlapping each other; no context atoms at all
    coords = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    subject = torch.tensor([[True, False]])          # only atom 0 is mismatched

    scoped = clash_loss(coords, valid_mask=valid, subject_mask=subject)
    assert scoped > 0, (
        "a mismatched atom overlapping a matched atom must still be penalised"
    )


def test_subject_mask_defaults_to_valid_mask():
    """Omitting it must not change behaviour — Stage II warmup has no predicted type."""
    coords, valid, group, ctx, ctx_mask, ctx_group = _two_residues_one_clashing()
    kw = dict(valid_mask=valid, group_id=group, context_coords=ctx,
              context_mask=ctx_mask, context_group_id=ctx_group)
    assert torch.allclose(clash_loss(coords, **kw),
                          clash_loss(coords, subject_mask=valid, **kw))


def test_a_residues_own_bonded_atoms_are_not_a_clash():
    """Regression: the intra term used to score same-residue pairs.

    A side chain's own atoms are covalently bonded at ~1.5 A, below clash_dist, so
    every correctly-built residue was paying a steric penalty for existing — pulling
    against the coordinate loss. Only the CROSS term excluded same-residue pairs,
    even though the docstring claimed the exclusion applied generally.
    """
    bonded = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])   # CB-CG, one residue
    valid = torch.ones(1, 2, dtype=torch.bool)
    same = torch.tensor([[7, 7]])
    assert float(clash_loss(bonded, valid_mask=valid, group_id=same)) == pytest.approx(0.0)
    # ...but the same geometry across two residues IS an overlap
    other = torch.tensor([[7, 8]])
    assert clash_loss(bonded, valid_mask=valid, group_id=other) > 0


# --- arms -------------------------------------------------------------------
def _phys(arm, **kw):
    coords, valid, group, ctx, ctx_mask, ctx_group = _two_residues_one_clashing()
    return physical_loss(
        coords, context_coords=ctx, context_mask=ctx_mask, context_group_id=ctx_group,
        group_id=group, valid_mask=valid, arm=arm, **kw
    )


def test_arm_none_is_identically_zero_but_still_differentiable():
    out = _phys("none")
    assert float(out["total"]) == 0.0
    assert float(out["clash"]) == 0.0
    # must stay in the graph: the trainer adds it to a loss that gets .backward()
    assert out["total"].requires_grad is False or out["total"].grad_fn is not None


def test_arm_clash_scores_steric_only_no_contact_term():
    out = _phys("clash")
    assert out["clash"] > 0
    assert float(out["contact"]) == 0.0, "0722 replaced contact with pack+hbond"
    assert torch.allclose(out["total"], out["clash"])


def test_arm_legacy_keeps_the_pre_0722_contact_hinge():
    legacy, clash_only = _phys("legacy"), _phys("clash")
    assert legacy["contact"] > 0, "legacy must still include the old contact term"
    assert legacy["total"] > clash_only["total"]


def test_arm_compat_refuses_rather_than_silently_running_something_else():
    """pack + hbond do not exist yet. Quietly falling back to clash would make an
    ablation report an arm it never ran."""
    with pytest.raises(NotImplementedError, match="pack"):
        _phys("compat")


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown mismatch_loss arm"):
        _phys("physical")          # the pre-0722 name, a plausible typo


def test_all_declared_arms_are_reachable():
    for arm in MISMATCH_ARMS:
        if arm == "compat":
            continue                                  # asserted above
        _phys(arm)


# --- deprecated terms -------------------------------------------------------
def test_bond_angle_rotamer_stay_off_unless_explicitly_fed():
    """0722 removed them from the objective. They must never switch themselves on."""
    for arm in ("clash", "legacy"):
        out = _phys(arm)
        assert float(out["bond"]) == 0.0
        assert float(out["angle"]) == 0.0
        assert float(out["rotamer"]) == 0.0


# --- the teacher-forced stages must contribute nothing ----------------------
def test_no_type_match_mask_means_empty_subject_set_not_everything():
    """Stage II/III have no predicted type, so NOTHING is mismatched — the term is 0.

    Regression, caught by GPU smoke: the fallback used to be "score every residue",
    which silently resurrected the pre-0722 behaviour in exactly the stage whose
    objective (0722, L_joint) has no L_compat term at all. A teacher-forced run was
    reporting sc_phys = 3e-3 where the paper says there is no such term.
    """
    class _M:
        sc_predicted_mask = False

        _warn_once = staticmethod(lambda *a, **k: None)
        _mismatch_subject_mask = ProtenixDesignTrain._mismatch_subject_mask

    valid = torch.ones(2, 6, dtype=torch.bool)
    subj = _M._mismatch_subject_mask(_M(), {}, valid, 2, 3, 2)
    assert subj is not None and subj.shape == valid.shape
    assert not subj.any(), "teacher forcing must give an EMPTY mismatch set"

    # and an empty subject set really does zero the loss
    coords = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    v = torch.ones(1, 2, dtype=torch.bool)
    assert float(clash_loss(coords, valid_mask=v,
                            subject_mask=torch.zeros(1, 2, dtype=torch.bool))) == 0.0


# --- config wiring ----------------------------------------------------------
def test_config_default_arm():
    from pxdesign_train.configs.configs_train import training_configs

    assert training_configs["sidechain"]["mismatch_loss"] == "clash"


def test_model_rejects_a_bad_arm_at_construction():
    """A typo must fail loudly at build time, not silently train the wrong ablation."""
    from pxdesign_train.sidechain.physical import MISMATCH_ARMS

    assert "none" in MISMATCH_ARMS and "clash" in MISMATCH_ARMS
