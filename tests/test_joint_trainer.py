"""The training loop's contracts, against the real donor.

The experiment is a comparison, so most of what matters here is that the arms
differ in exactly one thing. The load-bearing test is that B0 and BS -- which
run different amounts of computation and optimize the same objective -- take
bit-identical optimizer steps. If they do not, every later difference between
a candidate and its baseline is confounded by the seeding or the arm switching
rather than by the objective.
"""

import os
from pathlib import Path

import pytest
import torch

from pxf.joint import trainer as T

DONOR = os.environ.get(
    "PXDESIGN_DONOR",
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt",
)
CIF = os.environ.get("PXF_TEST_CIF", "/hai/scratch/yfsun/casp14/cif/T1031.cif")

needs_donor = pytest.mark.skipif(
    not (Path(DONOR).is_file() and Path(CIF).is_file()),
    reason="set PXDESIGN_DONOR and PXF_TEST_CIF to run the real training loop",
)


# ---- the arm table ----------------------------------------------------------


def test_every_arm_is_defined_in_one_place():
    assert set(T.ARMS) == {"R0", "B0", "B1", "B2", "BF", "BS"}
    # B0 runs no side-chain branch; BS runs it and discards the gradient.
    assert T.arm_spec("B0")[4] is False
    assert T.arm_spec("BS")[4] is True and T.arm_spec("BS")[3] is True
    # Only B2 carries both auxiliary terms.
    assert T.arm_spec("B2")[:3] == (True, True, False)
    assert T.arm_spec("BF")[:3] == (False, False, True)


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="Unknown arm"):
        T.arm_spec("B9")
    with pytest.raises(ValueError, match="Unknown arm"):
        T.JointSettings(arm="candidate")


def test_the_auxiliary_coefficients_ramp_then_hold():
    settings = T.JointSettings(arm="B1", warmup_auxiliary_steps=100)
    assert settings.auxiliary_scale(0) == pytest.approx(0.01)
    assert settings.auxiliary_scale(49) == pytest.approx(0.5)
    assert settings.auxiliary_scale(99) == pytest.approx(1.0)
    assert settings.auxiliary_scale(10_000) == pytest.approx(1.0)
    assert T.JointSettings(arm="B1", warmup_auxiliary_steps=0).auxiliary_scale(0) == 1.0


# ---- the cache --------------------------------------------------------------


def test_the_cache_evicts_and_reports():
    cache = T.ExampleCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # evicts b, the least recently used
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.stats()["held"] == 2


# ---- the real loop ----------------------------------------------------------


@pytest.fixture(scope="module")
def rig():
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.provenance import fampnn_checkpoint

    backbone, _bundle, record = load_backbone_model(DONOR)
    driver = PXDesignBackboneDriver(backbone)
    weights = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(weights["model_cfg"])
    fampnn.load_state_dict(weights["state_dict"], strict=True)
    return driver, fampnn, record


def _entries(n=4):
    return [
        dict(sample_id=Path(CIF).stem, path=CIF, crop_size=256, split="train")
        for _ in range(n)
    ]


def _trainer(rig, tmp_path, **overrides):
    driver, fampnn, record = rig
    defaults = dict(
        max_steps=1,
        grad_accum_steps=1,
        multiplier=2,
        log_every=1,
        checkpoint_every=0,
        ema_relative_length=None,
        warmup_steps=0,
        lr=1e-4,  # large enough that one step is visible in float32
    )
    defaults.update(overrides)
    settings = T.JointSettings(**defaults)
    return T.JointTrainer(
        driver,
        fampnn,
        iter(_entries()),
        out_dir=tmp_path,
        settings=settings,
        donor_record=record,
    )


@needs_donor
def test_the_optimizer_trains_the_allowlist_and_nothing_else(rig, tmp_path):
    trainer = _trainer(rig, tmp_path, arm="B0")
    in_optimizer = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert in_optimizer == {id(p) for p in trainer.trainable.values()}
    live = [n for n, p in trainer.backbone.named_parameters() if p.requires_grad]
    assert set(live) == set(trainer.trainable)
    assert all(not p.requires_grad for p in trainer.fampnn.parameters())
    assert trainer.identity()["trainable_parameters"] > 0


@needs_donor
def test_a_trainable_fampnn_is_frozen_and_the_fact_recorded(rig, tmp_path):
    """This stage freezes the side-chain model, and says how much it froze.

    Silently disabling what a caller asked to train is the failure mode; the
    count goes in the run record so "FaMPNN was frozen" is evidence rather than
    an assumption.
    """
    _driver, fampnn, _record = rig
    fampnn.denoiser.scn_diffusion_module.scn_denoiser.requires_grad_(True)
    try:
        trainer = _trainer(rig, tmp_path, arm="B1", lambda_local=1.0)
        assert trainer.froze_sidechain > 0
        assert trainer.identity()["froze_sidechain_parameters"] > 0
        assert all(not p.requires_grad for p in fampnn.parameters())
    finally:
        fampnn.requires_grad_(False)


@needs_donor
def test_asking_for_a_trainable_sidechain_is_refused_until_j0_exists(rig, tmp_path):
    """freeze_sidechain=False has no optimizer group behind it yet.

    Those parameters would accumulate gradient and never update, which is a
    run that looks like joint training and is not.
    """
    _driver, fampnn, _record = rig
    fampnn.denoiser.scn_diffusion_module.scn_denoiser.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="not implemented yet"):
            _trainer(rig, tmp_path, arm="B1", lambda_local=1.0, freeze_sidechain=False)
    finally:
        fampnn.requires_grad_(False)


@needs_donor
def test_a_scope_the_donor_does_not_have_is_refused(rig, tmp_path):
    with pytest.raises(ValueError, match="fewer than the"):
        _trainer(rig, tmp_path, arm="B0", trainable_blocks=999)


@needs_donor
def test_r0_has_no_training_run(rig, tmp_path):
    trainer = _trainer(rig, tmp_path, arm="R0")
    with pytest.raises(ValueError, match="untrained donor reference"):
        trainer.train(progress=None)


def _snapshot(trainer):
    return {n: p.detach().clone() for n, p in trainer.trainable.items()}


def _restore(trainer, snapshot):
    with torch.no_grad():
        for name, parameter in trainer.trainable.items():
            parameter.copy_(snapshot[name])


@needs_donor
def test_b0_and_bs_take_the_same_step(rig, tmp_path):
    """The control's whole purpose: same objective, same step, more compute.

    BS runs the side-chain branch with the backbone detached at both
    entrances, so its gradient is exactly B0's. A difference here would mean
    the arms are not paired -- the noise draws, the example order or the
    optimizer state diverged -- and every later B1-vs-B0 comparison would
    inherit that.

    **Single-threaded, and that is not incidental.** With torch's default 16
    intra-op threads, two runs of B0 from the same weights and the same seed
    diverge by 4.2e-05 on a weight after two steps -- roughly 40% of an
    optimizer step at lr 1e-4 -- purely from reduction order. At one thread B0
    reproduces itself exactly, which is what makes a bitwise assertion about
    the arms meaningful rather than a coin flip. Pilot runs happen on a GPU,
    where the equivalent statement needs a declared tolerance instead.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        b0 = _trainer(rig, tmp_path / "b0", arm="B0", max_steps=2)
        start = _snapshot(b0)
        b0.train(progress=None)
        after_b0 = _snapshot(b0)
        assert any(
            not torch.equal(start[n], after_b0[n]) for n in start
        ), "B0 did not move at all, so the comparison is vacuous"

        # B0 reproduces itself, or the comparison below means nothing.
        _restore(b0, start)
        again = _trainer(rig, tmp_path / "b0again", arm="B0", max_steps=2)
        again.train(progress=None)
        for name in start:
            assert torch.equal(after_b0[name], _snapshot(again)[name]), (
                f"B0 is not reproducible at one thread ({name}), so this test "
                "cannot distinguish an arm difference from run-to-run noise"
            )

        _restore(b0, start)
        bs = _trainer(rig, tmp_path / "bs", arm="BS", max_steps=2, lambda_local=1.0)
        bs.train(progress=None)
        after_bs = _snapshot(bs)
        for name in start:
            assert torch.equal(after_b0[name], after_bs[name]), name
    finally:
        torch.set_num_threads(threads)


@needs_donor
def test_a_backbone_only_arm_does_not_run_the_sidechain_branch(rig, tmp_path):
    """B0 skips it; BS runs it and discards the gradient. That is the difference.

    Running it anyway would cost an encoder pass and m denoiser passes B0 never
    scores, and -- because the branch draws its own noise when none is supplied
    -- would leave B0 and B1 on different global RNG streams from that point on.
    """
    from pxf.joint import model as joint_model

    trainer = _trainer(rig, tmp_path, arm="B0")
    entry = _entries(1)[0]
    batch, conditioning = trainer.prepare(entry)
    calls = {}
    original = joint_model.joint_forward

    def spy(*args, **kwargs):
        calls["run_sidechain"] = kwargs.get("run_sidechain")
        return original(*args, **kwargs)

    joint_model.joint_forward = spy
    try:
        _combined, forward = trainer.forward_losses(entry, occurrence=0)
    finally:
        joint_model.joint_forward = original
    assert calls["run_sidechain"] is False
    assert not forward.ran_sidechain
    assert forward.prediction is None and forward.placed is None


@needs_donor
def test_disabled_clipping_does_not_zero_the_gradients(rig, tmp_path):
    """``max_grad_norm = 0`` means "do not clip", not "scale everything to 0"."""
    trainer = _trainer(rig, tmp_path, arm="B0", max_grad_norm=0.0, max_steps=1)
    start = _snapshot(trainer)
    trainer.train(progress=None)
    assert any(
        not torch.equal(start[n], p.detach()) for n, p in trainer.trainable.items()
    )


@needs_donor
def test_a_non_finite_loss_names_the_example_and_stops(rig, tmp_path):
    """Refused, not zeroed: a dropped term turns the candidate into the baseline."""
    trainer = _trainer(rig, tmp_path, arm="B1", lambda_local=1.0, max_steps=1)
    entry = _entries(1)[0]
    batch, conditioning = trainer.prepare(entry)
    batch.backbone_target[0, 0] = float("nan")
    try:
        with pytest.raises(ValueError, match=entry["sample_id"]):
            trainer.forward_losses(entry, occurrence=0)
    finally:
        trainer.cache = T.ExampleCache(4)  # the poisoned batch must not persist


@needs_donor
def test_resume_restores_the_step_optimizer_and_weights(rig, tmp_path):
    trainer = _trainer(rig, tmp_path / "run", arm="B0", max_steps=1, checkpoint_every=1)
    trainer.train(progress=None)
    saved = trainer.save(tag="probe")
    weights = _snapshot(trainer)

    fresh = _trainer(rig, tmp_path / "resumed", arm="B0", max_steps=1)
    assert fresh.resume(saved) == trainer.step
    for name, parameter in fresh.trainable.items():
        assert torch.equal(parameter.detach(), weights[name]), name
    assert fresh.optimizer.state_dict()["state"], "optimizer moments should be restored"


@needs_donor
def test_resuming_across_arms_is_refused(rig, tmp_path):
    trainer = _trainer(rig, tmp_path / "b0", arm="B0", max_steps=1)
    saved = trainer.save(tag="b0")
    other = _trainer(rig, tmp_path / "b1", arm="B1", lambda_local=1.0)
    with pytest.raises(ValueError, match="resuming across arms"):
        other.resume(saved)


@needs_donor
def test_the_checkpoint_carries_what_a_rerun_needs(rig, tmp_path):
    trainer = _trainer(rig, tmp_path, arm="B0", max_steps=1)
    state = torch.load(trainer.save(tag="probe"), map_location="cpu", weights_only=False)
    for key in ("step", "arm", "settings", "trainable_state", "optimizer", "identity", "rng"):
        assert key in state, key
    identity = state["identity"]
    assert identity["donor"] and identity["randomness"]["base_seed"] == 0
    # The donor itself is pinned by digest rather than copied into every arm's
    # checkpoint; only the trained slice is stored.
    assert set(state["trainable_state"]) == set(trainer.trainable)


@needs_donor
def test_the_cache_holds_no_model_derived_tensor(rig, tmp_path):
    """Static parses and frozen conditioning only -- never B_hat, h_V or frames.

    Anything derived from the current weights changes every update, so caching
    it would train against a stale structure that still looks plausible.
    """
    trainer = _trainer(rig, tmp_path, arm="B1", lambda_local=1.0)
    entry = _entries(1)[0]
    trainer.prepare(entry)
    assert trainer.cache.stats()["held"] == 1
    trainer.prepare(entry)
    assert trainer.cache.stats()["hits"] == 1

    batch, conditioning = trainer.cache.get(entry["sample_id"])
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            assert not value.requires_grad, name
    for field in ("s_inputs", "s_trunk", "z_trunk"):
        held = getattr(conditioning, field, None)
        if torch.is_tensor(held):
            assert not held.requires_grad, field
