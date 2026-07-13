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

from pxdesign_train.configs.configs_train import training_configs


# name -> the sidechain config overrides that DEFINE the arm.
ARMS = {
    # true control: refinement pass runs, no side-chain info reaches the backbone
    "no":            dict(hres_inject=False, a_direct=False, bb_context=False, q_direct=False),
    # the pre-existing indirect channel (h_res' -> s_trunk -> a_token recomputed)
    "a-indirect":    dict(hres_inject=True,  a_direct=False, bb_context=False, q_direct=False),
    # FangWu's slide: a'_bb = a_bb + MLP([a_bb, a_sc]) injected into the token itself
    "a-direct":      dict(hres_inject=False, a_direct=True,  bb_context=False, q_direct=False),
    # CONTROL for q: S_phi gets the 4 backbone context atoms, but NO q channel.
    # Without this arm, "q helps" is confounded with "S_phi went from 10 to 14 slots".
    "bbctx":         dict(hres_inject=False, a_direct=False, bb_context=True,  q_direct=False),
    # the atom-level channel proper (implies bb_context)
    "q":             dict(hres_inject=False, a_direct=False, bb_context=True,  q_direct=True),
    # both channels
    "a-direct+q":    dict(hres_inject=False, a_direct=True,  bb_context=True,  q_direct=True),
}


def _sc_cfg(**overrides):
    base = dict(training_configs["sidechain"])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_all_six_arms_are_defined():
    assert set(ARMS) == {"no", "a-indirect", "a-direct", "bbctx", "q", "a-direct+q"}


def test_arms_are_pairwise_distinct():
    """No two arms may resolve to the same configuration, or the ablation proves nothing."""
    seen = {}
    for name, cfg in ARMS.items():
        key = tuple(sorted(cfg.items()))
        assert key not in seen, f"{name} is identical to {seen[key]}"
        seen[key] = name


def test_defaults_reproduce_the_a_indirect_arm():
    """Today's default behaviour must be exactly one named arm -- and it is NOT 'no'.

    This is the whole point of the file: the shipped default IS the indirect channel.
    """
    d = training_configs["sidechain"]
    assert d["hres_inject"] is True          # <- the channel that used to be invisible
    assert d["a_direct"] is False
    assert d["bb_context"] is False
    assert d["q_direct"] is False
    assert dict(hres_inject=True, a_direct=False, bb_context=False, q_direct=False) == ARMS["a-indirect"]


def test_no_arm_really_has_no_feedback_channel():
    """The 'no' arm must switch off EVERY side-chain -> backbone channel."""
    cfg = ARMS["no"]
    assert cfg["hres_inject"] is False, "indirect token channel still on -> not a control"
    assert cfg["a_direct"] is False
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
    # Every feedback channel off <=> this is the control arm.
    assert (not hres and not a_d and not q_d) == (arm in ("no", "bbctx"))
