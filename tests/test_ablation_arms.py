"""The six ablation arms must be distinct, explicit, and mean what they say.

The trap this pins: before `sidechain.hres_inject` existed, the "no feedback" arm was
NOT no-feedback -- HResInjector (h_res' -> s_trunk) is on by default whenever
co-evolution runs, so every arm silently contained the indirect token channel. That
would have contaminated the DEFINITION of every later ablation: an apparent "q helps"
could have been "q helps on top of a channel we forgot we had".

`hres_inject=False` is NOT the same as `enable_coevolution=False`:
  * enable_coevolution=False -> no refinement pass at all.
  * hres_inject=False       -> the refinement pass STILL RUNS (B_theta is called a second
                               time) but carries NO side-chain information.
Only the second one is a valid control for "does the feedback channel buy anything",
because it holds the second pass fixed and ablates only the channel.

These tests assert the switch SEMANTICS. They deliberately do not compare loss values:
a single-structure memorization run cannot rank these arms (the arms with feedback add
freshly-initialised parameters, so they memorize more slowly regardless of whether the
channel helps). Ranking waits for real data.
"""
from types import SimpleNamespace

import pytest
import torch

from pxdesign_train.configs.configs_train import (
    SC_ABLATION_ARMS as ARMS,
    apply_sidechain_ablation_arm,
    training_configs,
)


def _sc_cfg(**overrides):
    base = dict(training_configs["sidechain"])
    base.update(overrides)
    return SimpleNamespace(**base)


# Arms that REPLACE the default indirect channel (hres_inject=False): they answer
# "which single channel is best". Arms prefixed "+" KEEP it and add one channel on
# top: they answer "does adding this channel buy anything". Reporting one as
# evidence for the other is the confusion these two sets exist to prevent.
REPLACEMENT_ARMS = {
    "no", "a-indirect", "a-direct", "a-direct-pre", "q", "a-direct+q",
    "a-bs", "q-bs",
}
# The single 10-slot arm. Every other arm runs the 14-slot atom axis so they can
# share one Stage II checkpoint; this one answers "does the axis itself help" and
# pays for it with its own Stage II run.
TEN_SLOT_ARMS = {"no-bbctx"}
INCREMENTAL_ARMS = {
    "+a-direct", "+a-direct-pre", "+q", "+a-bs", "+q-bs",
}
# Leave-one-out from the shipped default. Now that the default is the full
# bidirectional wiring, this is the PRIMARY ablation: each arm drops exactly one
# channel, so a regression is that channel's marginal contribution.
LEAVE_ONE_OUT_ARMS = {
    "full", "full-a", "full-q", "full-a-bs", "full-q-bs", "full-a-post",
}
# `hres_inject` is the one channel the default leaves OFF -- it shares a cell of
# the 2x2 with a_direct_pre (residue level, S->B) and predates it. So it gets an
# ADD-back arm rather than a leave-one-out one.
ADD_BACK_ARMS = {"full+hres"}


def test_all_arms_are_defined():
    assert set(ARMS) == (REPLACEMENT_ARMS | INCREMENTAL_ARMS | LEAVE_ONE_OUT_ARMS
                         | ADD_BACK_ARMS | TEN_SLOT_ARMS)


def test_every_arm_specifies_every_switch():
    """An arm is a FULL specification, never a delta on the current defaults.

    A partially-specified arm silently inherits whatever the defaults happen to
    be, so the same arm name would mean different wiring before and after a
    default changes — and past runs would stop being comparable.
    """
    keys = {frozenset(cfg) for cfg in ARMS.values()}
    assert len(keys) == 1, f"arms specify different switch sets: {keys}"


def test_incremental_arms_keep_the_default_channel_and_add_exactly_one_thing():
    """Each "+x" arm must be a-indirect plus the channel(s) its name promises."""
    base = ARMS["a-indirect"]
    for name in INCREMENTAL_ARMS:
        arm = ARMS[name]
        assert arm["hres_inject"] is True, f"{name} dropped the default channel"
        added = {k for k in arm if arm[k] != base[k]}
        assert added, f"{name} adds nothing — it is just a-indirect"
        counterpart = ARMS[name[1:]]
        if True:
            expected = {k for k in counterpart if counterpart[k] != base[k]} - {"hres_inject"}
            assert added == expected, (
                f"{name} must add exactly what {name[1:]} switches on; "
                f"got {added}, expected {expected}"
            )


def test_arms_are_pairwise_distinct():
    """No two arms may resolve to the same configuration, or the ablation proves nothing."""
    seen = {}
    for name, cfg in ARMS.items():
        key = tuple(sorted(cfg.items()))
        assert key not in seen, f"{name} is identical to {seen[key]}"
        seen[key] = name


def test_defaults_reproduce_exactly_one_named_arm():
    """The shipped default must BE a named arm, so a run can cite the arm it used
    instead of "whatever the defaults were that week".

    Today that arm is `full`: the four explicit a/q channels on, `hres_inject`
    off. The four form a clean 2x2 (residue/atom level x B->S/S->B direction);
    hres_inject would be a fifth sharing a cell with a_direct_pre, which is why
    it is off and gets an add-back arm instead of a leave-one-out one.
    """
    d = training_configs["sidechain"]
    assert d["hres_inject"] is False         # <- the fifth channel, off by default
    assert d["a_direct_pre"] is True         # <- it covers the same cell of the 2x2
    assert d["bb_context"] is True           # <- one Stage II checkpoint for all arms
    matches = [
        name for name, cfg in ARMS.items() if {k: d[k] for k in cfg} == cfg
    ]
    assert len(matches) == 1, (
        f"defaults must reproduce exactly one arm, matched {matches}"
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_config_helper_applies_named_arm(arm):
    cfg = {"sidechain": dict(training_configs["sidechain"])}
    apply_sidechain_ablation_arm(cfg, arm)
    for key, value in ARMS[arm].items():
        assert cfg["sidechain"][key] is value


def test_no_arm_really_has_no_feedback_channel():
    """The 'no' arm must switch off EVERY side-chain -> backbone channel."""
    cfg = ARMS["no"]
    assert cfg["hres_inject"] is False, "indirect token channel still on -> not a control"
    assert cfg["a_direct"] is False
    assert cfg["a_direct_pre"] is False
    assert cfg["q_direct"] is False


def test_q_arm_and_its_control_differ_only_in_the_q_channel():
    """`q` minus `no` must isolate the atom channel.

    `bbctx` used to be this control ("14 slots, no feedback"). Now that every arm
    runs 14 slots, `no` IS that configuration, so `bbctx` was removed rather than
    left as a duplicate row that silently makes two arms the same experiment.
    """
    q, ctrl = ARMS["q"], ARMS["no"]
    diff = {k for k in q if q[k] != ctrl[k]}
    assert diff == {"q_direct"}, f"q vs no differ in {diff}, not just the q channel"


def test_the_axis_question_has_exactly_one_arm_and_it_is_flagged():
    """`no` vs `no-bbctx` is the only pair that differs in the atom axis alone."""
    from pxdesign_train.configs.configs_train import ARMS_NEEDING_14_SLOT_STAGE2

    diff = {k for k in ARMS["no"] if ARMS["no-bbctx"][k] != ARMS["no"][k]}
    assert diff == {"bb_context"}
    assert "no-bbctx" not in ARMS_NEEDING_14_SLOT_STAGE2
    assert set(ARMS) - ARMS_NEEDING_14_SLOT_STAGE2 == {"no-bbctx"}, (
        "exactly one arm may need a separate Stage II checkpoint"
    )


def test_q_direct_implies_bb_context():
    """S_phi cannot produce backbone-atom features without the 14-slot axis."""
    from pxdesign_train.model import ProtenixDesignTrain  # noqa: F401  (import guard)
    for name in ("q", "a-direct+q"):
        assert ARMS[name]["bb_context"] is True, f"{name} needs the 14-slot axis"


def test_hres_inject_flag_controls_the_injection_branch():
    """The switch must gate the HResInjector call, not the refinement pass itself."""
    import inspect

    from pxdesign_train.model import ProtenixDesignTrain

    src = inspect.getsource(ProtenixDesignTrain._train_forward)
    # the refinement pass is gated on enable_coevolution ...
    assert 'getattr(self, "enable_coevolution", False)' in src
    # ... and the INJECTION is gated separately, on sc_hres_inject.
    assert 'getattr(self, "sc_hres_inject", True)' in src
    i_pass = src.index('getattr(self, "enable_coevolution", False)')
    i_inject = src.index('getattr(self, "sc_hres_inject", True)')
    assert i_inject > i_pass, (
        "hres_inject must gate the injection INSIDE the refinement pass — if it gated the "
        "pass itself, 'no' would confound 'no second pass' with 'no feedback'."
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_model_attributes_match_the_arm(arm):
    """Each arm's config must land on the model as the attributes the forward reads."""
    cfg = _sc_cfg(**ARMS[arm])
    # Emulate the resolution model.__init__ performs (q_direct implies bb_context).
    hres = bool(getattr(cfg, "hres_inject", True))
    a_d = bool(getattr(cfg, "a_direct", False))
    q_d = bool(getattr(cfg, "q_direct", False))
    bb = bool(getattr(cfg, "bb_context", False)) or q_d

    assert hres == ARMS[arm]["hres_inject"]
    assert a_d == ARMS[arm]["a_direct"]
    assert q_d == ARMS[arm]["q_direct"]
    if q_d:
        assert bb, "q_direct must imply bb_context"
    # Every old feedback channel off <=> this has no old feedback (but may have new B->S channels).
    a_dp = bool(getattr(cfg, "a_direct_pre", False))
    assert a_dp == ARMS[arm]["a_direct_pre"]
    # Every S->B channel off <=> this arm has no side-chain -> backbone feedback
    # (it may still have the B->S input channels a_bs_concat / q_bs).
    no_feedback = not hres and not a_d and not a_dp and not q_d
    assert no_feedback == (arm in ("no", "no-bbctx", "a-bs", "q-bs"))


def test_bs_channels_ship_on_and_each_has_a_removal_arm():
    """Both B->S channels are part of the default wiring, and both are ablatable."""
    from pxdesign_train.configs.configs_train import SC_ABLATION_ARMS, training_configs
    assert training_configs["sidechain"]["a_bs_concat"] is True
    assert training_configs["sidechain"]["q_bs"] is True
    assert SC_ABLATION_ARMS["full-a-bs"]["a_bs_concat"] is False
    assert SC_ABLATION_ARMS["full-q-bs"]["q_bs"] is False


def test_leave_one_out_arms_remove_exactly_one_channel():
    full = ARMS["full"]
    for name in LEAVE_ONE_OUT_ARMS - {"full", "full-a-post"}:
        removed = {k for k in full if ARMS[name][k] != full[k]}
        assert len(removed) == 1, f"{name} changes {removed}, not exactly one channel"
        assert all(ARMS[name][k] is False for k in removed)


def test_the_add_back_arm_only_restores_the_fifth_channel():
    """full+hres must be `full` plus hres_inject and nothing else."""
    full, add = ARMS["full"], ARMS["full+hres"]
    assert {k for k in full if add[k] != full[k]} == {"hres_inject"}
    assert add["hres_inject"] is True and full["hres_inject"] is False


def test_the_injection_point_arm_swaps_rather_than_removes():
    """full-a-post holds everything fixed and only moves where `a` is injected."""
    full, post = ARMS["full"], ARMS["full-a-post"]
    assert {k for k in full if post[k] != full[k]} == {"a_direct", "a_direct_pre"}
    assert post["a_direct"] is True and post["a_direct_pre"] is False


def test_stage2_checkpoint_groups_are_derived_and_non_trivial():
    """An arm's Stage II warm-start requirement must follow from its own config.

    bb_context decides whether S_phi's atom axis is 10 slots or 14. Two arms on
    opposite sides of that flip cannot share a Stage II checkpoint, so a run that
    compares them against one warm-start is measuring the layout change on top of
    the channel it meant to isolate. The grouping is derived from the table (not
    hand-maintained) so a newly added arm lands in a group automatically; this
    test just pins that both groups are actually populated, i.e. that the trap is
    real and a reader is not lulled by an empty set.
    """
    from pxdesign_train.configs.configs_train import (
        ARMS_NEEDING_14_SLOT_STAGE2,
        stage2_checkpoint_group,
    )

    ten = {a for a in ARMS if a not in ARMS_NEEDING_14_SLOT_STAGE2}
    assert ARMS_NEEDING_14_SLOT_STAGE2 and ten, (
        "both Stage II groups must be non-empty, or the documented trap is stale"
    )
    for arm in ARMS:
        expected = "14-slot" if ARMS[arm]["bb_context"] else "10-slot"
        assert stage2_checkpoint_group(arm) == expected

    # q_direct / q_bs force the 14-slot axis, so every q arm is in that group.
    for arm, cfg in ARMS.items():
        if cfg["q_direct"] or cfg["q_bs"]:
            assert arm in ARMS_NEEDING_14_SLOT_STAGE2, (
                f"{arm} uses a q channel but is not marked as needing 14 slots"
            )


def test_unknown_arm_is_rejected_rather_than_defaulted():
    from pxdesign_train.configs.configs_train import stage2_checkpoint_group

    with pytest.raises(ValueError, match="unknown arm"):
        stage2_checkpoint_group("not-an-arm")


def test_a_pre_record_sidechain_checkpoint_is_refused_not_silently_loaded():
    """A checkpoint with S_phi weights but no layout record must not load quietly.

    bb_context adds no parameters, so a 10-slot S_phi loads into a 14-slot model
    with no shape error at all -- it just silently runs under a layout it never
    trained on. A newly enabled fusion is equally quiet: load_strict=False leaves
    it at its zero init, wired in but untrained. Both look like "the channel
    didn't help".
    """
    import types

    from pxdesign_train.runner.trainer import PXDesignTrainer

    stub = types.SimpleNamespace(
        configs=types.SimpleNamespace(
            enable_sidechain=True,
            sidechain=types.SimpleNamespace(
                bb_context=True, local_coord_input=False,
                frame_aware_head=False, a_bs_concat=True, q_bs=False,
            ),
        )
    )
    stub.SIDECHAIN_ARCH_KEYS = PXDesignTrainer.SIDECHAIN_ARCH_KEYS
    stub._sidechain_arch = PXDesignTrainer._sidechain_arch.__get__(stub, type(stub))
    check = PXDesignTrainer._check_sidechain_arch.__get__(stub, type(stub))

    # a Stage I checkpoint has no side-chain weights -> nothing to mismatch
    check({"model": {"diffusion_module.x": 1}})

    # a pre-record Stage II checkpoint DOES -> refuse
    with pytest.raises(ValueError, match="records no sidechain_arch"):
        check({"model": {"sidechain_module.atom_embed.weight": 1}})



def test_layout_and_additive_switches_are_guarded_differently():
    """Crossing a LAYOUT switch must fail; crossing an ADDITIVE one must not.

    bb_context re-shapes the axis every pretrained intra-residue block attends
    over, so a warm-start across it degrades silently -- refuse. a_bs_concat /
    q_bs merely decide whether an extra zero-initialised fusion exists, and
    enabling one at a later stage is exactly how the curriculum is meant to work
    (Stage II deliberately runs without q_bs). Blocking that would have made the
    intended Stage II -> Stage III hand-off impossible.
    """
    import types

    from pxdesign_train.runner.trainer import PXDesignTrainer

    def _checker(**current):
        base = dict(bb_context=True, local_coord_input=False,
                    frame_aware_head=False, a_bs_concat=True, q_bs=True)
        base.update(current)
        stub = types.SimpleNamespace(
            configs=types.SimpleNamespace(
                enable_sidechain=True, sidechain=types.SimpleNamespace(**base)
            ),
            _log=lambda *a, **k: None,
        )
        stub.SIDECHAIN_ARCH_KEYS = PXDesignTrainer.SIDECHAIN_ARCH_KEYS
        stub.SIDECHAIN_LAYOUT_KEYS = PXDesignTrainer.SIDECHAIN_LAYOUT_KEYS
        stub.SIDECHAIN_ADDITIVE_KEYS = PXDesignTrainer.SIDECHAIN_ADDITIVE_KEYS
        stub._sidechain_arch = PXDesignTrainer._sidechain_arch.__get__(stub, type(stub))
        return PXDesignTrainer._check_sidechain_arch.__get__(stub, type(stub))

    saved = dict(bb_context=True, local_coord_input=False, frame_aware_head=False,
                 a_bs_concat=True, q_bs=False)          # a Stage II checkpoint

    # Stage II -> Stage III turns q_bs on. This is the intended hand-off.
    _checker(q_bs=True)({"sidechain_arch": saved, "model": {}})

    # Flipping the atom axis is not.
    with pytest.raises(ValueError, match="LAYOUT changed"):
        _checker(bb_context=False)({"sidechain_arch": saved, "model": {}})
