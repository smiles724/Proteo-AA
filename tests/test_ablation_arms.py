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
    "+a-direct", "+a-direct-pre", "+q", "+a-bs", "+q-bs", "+all",
}


def test_all_arms_are_defined():
    assert set(ARMS) == REPLACEMENT_ARMS | INCREMENTAL_ARMS | TEN_SLOT_ARMS


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
        if name != "+all":
            counterpart = ARMS[name[1:]]
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

    Today that arm is `+a-bs`: the indirect channel (the paper's formula) plus the
    14-slot axis plus the residue-level B->S concat. bb_context is on by default
    even though no q channel is -- it adds no parameters but fixes S_phi's input
    layout, so defaulting it off would force a separate Stage II run for every q
    ablation arm.
    """
    d = training_configs["sidechain"]
    assert d["hres_inject"] is True          # <- the channel that used to be invisible
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


def test_bs_arms_present_and_default_off():
    from pxdesign_train.configs.configs_train import SC_ABLATION_ARMS, training_configs
    assert training_configs["sidechain"]["a_bs_concat"] is False
    assert training_configs["sidechain"]["q_bs"] is False
    assert "a-bs" in SC_ABLATION_ARMS and SC_ABLATION_ARMS["a-bs"]["a_bs_concat"] is True
    assert "q-bs" in SC_ABLATION_ARMS and SC_ABLATION_ARMS["q-bs"]["q_bs"] is True


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
