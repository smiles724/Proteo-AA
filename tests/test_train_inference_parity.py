"""Every side-chain switch the TRAINING path reads, the SAMPLING path must read too.

This is a meta-test, and it exists because the same bug was shipped three separate times:

  * `sidechain.template_init` — training initialised S_phi from the ideal template
    (Overleaf par.221) while cogenerate still handed it isotropic Gaussian noise.
  * `sidechain.a_direct`      — the trained ATokenFusion was never armed at sampling, so
    every one of its parameters was dead weight at inference.
  * `sidechain.hres_inject`   — training could switch the indirect h_res' channel OFF (the
    true no-feedback control arm), but sampling switched it back ON, which would have
    contaminated the information-flow ablation for FOUR of the six arms.

Each time the symptom is the same and it is silent: the model is *sampled* under a
configuration it was never *trained* under. Unit tests pass, the run does not crash, and
the numbers are quietly meaningless. So instead of trusting the next person to remember,
this test enumerates the switches from the source and fails when one is not mirrored.

A switch that is genuinely training-only belongs in TRAIN_ONLY below, with a reason.
"""
import inspect
import re

from pxdesign_train import cogenerate as cg
from pxdesign_train.configs.configs_train import training_configs
from pxdesign_train.model import ProtenixDesignTrain

# Switches that legitimately have no sampling counterpart. Each needs a REASON.
TRAIN_ONLY = {
    # loss-side only: there is no coordinate loss at sampling time.
    "trunk_grad_scale": "gradient routing; no backward at inference",
    "detach_feedback": "gradient routing; no backward at inference",
    "weight_bb_post": "loss weight",
    "weight_aa_post": "loss weight",
    "weight_sc_global": "loss weight",
    "backbone_only_binder": "featurizer/label-side (which atoms enter L_bb)",
    "route_by_type": "loss routing (which residues get the coordinate loss)",
    "predicted_mask": "gates whether post_aa is SUPERVISED; sampling always uses "
                      "the predicted type anyway",
    "predicted_frame": "training-side choice between GT and predicted frames; "
                       "sampling only ever has the predicted backbone",
    "per_sigma": "consulted at sampling — see sc_per_sigma in the sampler",
    "c_atom": "architecture dim, not a behavioural switch",
    "init_sigma": "consulted at sampling (the Gaussian fallback)",
    "q_direct_zero_init": "initialisation-time only",
    "a_direct_zero_init": "initialisation-time only",
    # context_aware itself IS mirrored (cogenerate appends the key-only context rows).
    # These two only shape the ATOM set the clash/contact terms score against, and there
    # is no coordinate loss at sampling. The cross-residue context KEYS — the part that
    # changes S_phi's forward — are not radius-filtered in either path.
    "context_radius": "loss-side only: bounds the clash/contact atom set, no loss at inference",
    "context_max_atoms": "loss-side only: memory cap on the clash/contact atom set",
}


def _model_switches() -> set:
    """The sidechain.* keys the training forward actually branches on."""
    src = inspect.getsource(ProtenixDesignTrain)
    found = set()
    for key in training_configs["sidechain"]:
        # model.py reads them as getattr(sc_cfg, "<key>", ...) at construction time
        if re.search(r'getattr\(\s*sc_cfg\s*,\s*"%s"' % re.escape(key), src):
            found.add(key)
    return found


def _sampler_source() -> str:
    return inspect.getsource(cg)


def test_every_behavioural_switch_is_mirrored_in_the_sampler():
    switches = _model_switches()
    assert switches, "no sidechain switches discovered — the scraper is broken"

    sampler = _sampler_source()
    missing = []
    for key in sorted(switches):
        if key in TRAIN_ONLY:
            continue
        # the sampler reads them off the model as model.sc_<key>
        if f"sc_{key}" not in sampler:
            missing.append(key)

    assert not missing, (
        "These side-chain switches change TRAINING behaviour but the sampler never reads "
        f"them: {missing}. The model would be sampled under a configuration it was never "
        "trained under — silently. Mirror them in cogenerate(), or add them to TRAIN_ONLY "
        "with a reason."
    )


def test_train_only_entries_are_real_switches():
    """Guard the whitelist: it must not accumulate stale or invented keys."""
    known = set(training_configs["sidechain"])
    stale = set(TRAIN_ONLY) - known
    assert not stale, f"TRAIN_ONLY lists keys that are not sidechain config at all: {stale}"


def test_the_three_switches_that_were_missed_are_now_mirrored():
    """Explicitly pin the three regressions this test exists to prevent."""
    sampler = _sampler_source()
    for key in ("sc_template_init", "sc_a_direct", "sc_hres_inject"):
        assert key in sampler, f"{key} is not consulted by the sampler"
