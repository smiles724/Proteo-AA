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
    "no", "a-indirect", "a-direct", "a-direct-pre", "bbctx", "q", "a-direct+q",
    "a-bs", "q-bs",
}
INCREMENTAL_ARMS = {
    "+a-direct", "+a-direct-pre", "+q", "+a-bs", "+q-bs",
}
# Leave-one-out from the shipped default. This is the PRIMARY ablation now that
# the default is the full bidirectional wiring: each arm removes exactly one
# channel, so a drop measures that channel's contribution.
LEAVE_ONE_OUT_ARMS = {
    "full", "full-hres", "full-a", "full-q", "full-a-bs", "full-q-bs",
    "full-a-post",
}


def test_all_arms_are_defined():
    assert set(ARMS) == REPLACEMENT_ARMS | INCREMENTAL_ARMS | LEAVE_ONE_OUT_ARMS


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


def test_defaults_reproduce_the_full_arm():
    """The shipped default must BE a named arm, so a run can cite the arm it used.

    It used to be `a-indirect`: every explicit a/q channel shipped off, and the
    only live feedback was the one channel that appears on neither the slide nor
    the paper. The default is now the full bidirectional wiring, and the primary
    ablation is leave-one-out from it.
    """
    d = training_configs["sidechain"]
    assert {k: d[k] for k in ARMS["full"]} == ARMS["full"], (
        "the shipped defaults must reproduce exactly one named arm"
    )
    # the injection point that can reach spatial neighbours is the default one
    assert d["a_direct_pre"] is True
    assert d["a_direct"] is False


def test_leave_one_out_arms_remove_exactly_one_channel():
    full = ARMS["full"]
    for name in LEAVE_ONE_OUT_ARMS - {"full", "full-a-post"}:
        removed = {k for k in full if ARMS[name][k] != full[k]}
        assert len(removed) == 1, f"{name} changes {removed}, not exactly one channel"
        assert all(ARMS[name][k] is False for k in removed), (
            f"{name} must switch its channel OFF"
        )


def test_the_injection_point_arm_swaps_rather_than_removes():
    """full-a-post must hold everything fixed and only move where `a` is injected."""
    full, post = ARMS["full"], ARMS["full-a-post"]
    changed = {k for k in full if post[k] != full[k]}
    assert changed == {"a_direct", "a_direct_pre"}
    assert post["a_direct"] is True and post["a_direct_pre"] is False


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
    """q - bbctx must isolate the atom channel: the two arms differ in q_direct ONLY."""
    q, ctrl = ARMS["q"], ARMS["bbctx"]
    diff = {k for k in q if q[k] != ctrl[k]}
    assert diff == {"q_direct"}, f"q vs bbctx differ in {diff}, not just the q channel"


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
    assert no_feedback == (arm in ("no", "bbctx", "a-bs", "q-bs"))


def test_bs_channels_are_on_by_default_and_ablatable():
    """Both B->S channels ship ON (the slide's wiring) and each has a removal arm."""
    from pxdesign_train.configs.configs_train import SC_ABLATION_ARMS, training_configs
    assert training_configs["sidechain"]["a_bs_concat"] is True
    assert training_configs["sidechain"]["q_bs"] is True
    assert SC_ABLATION_ARMS["full-a-bs"]["a_bs_concat"] is False
    assert SC_ABLATION_ARMS["full-q-bs"]["q_bs"] is False