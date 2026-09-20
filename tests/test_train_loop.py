"""Weight averaging, the schedule, and checkpoint/resume semantics."""

import pytest
import torch
from torch import nn

from pxf.train.ema import EMA
from pxf.train.trainer import OptimSettings, Trainer, TrainSettings, learning_rate


def tiny():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    return model


# ---- EMA -------------------------------------------------------------------


def test_constant_decay_averages_as_expected():
    model = tiny()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    assert float(ema.shadow["weight"].flatten()[0]) == pytest.approx(0.5)
    ema.update(model)
    assert float(ema.shadow["weight"].flatten()[0]) == pytest.approx(0.75)


def test_relative_length_decay_rises_toward_one():
    """ "EMA length of 25%" means a window that grows with training."""
    ema = EMA(tiny(), relative_length=0.25)
    decays = []
    for step in (10, 100, 1000, 10000):
        ema.step = step
        decays.append(ema.current_decay())
    assert decays == sorted(decays)
    assert decays[-1] > 0.999
    assert all(0.0 <= d < 1.0 for d in decays)


def test_exactly_one_of_decay_or_relative_length():
    with pytest.raises(ValueError, match="exactly one"):
        EMA(tiny(), decay=0.9, relative_length=0.25)
    with pytest.raises(ValueError, match="exactly one"):
        EMA(tiny())


def test_relative_length_is_range_checked():
    with pytest.raises(ValueError, match="relative_length"):
        EMA(tiny(), relative_length=1.5)


def test_swap_restores_the_live_weights():
    model = tiny()
    ema = EMA(model, decay=0.0)  # shadow tracks the model exactly
    with torch.no_grad():
        model.weight.fill_(7.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    with ema.swapped_into(model):
        assert float(model.weight.flatten()[0]) == pytest.approx(7.0)
    assert float(model.weight.flatten()[0]) == pytest.approx(1.0)


def test_swap_restores_even_when_the_body_raises():
    model = tiny()
    ema = EMA(model, decay=0.0)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    with pytest.raises(RuntimeError):
        with ema.swapped_into(model):
            raise RuntimeError("boom")
    assert float(model.weight.flatten()[0]) == pytest.approx(2.0)


def test_ema_state_round_trips():
    model = tiny()
    first = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    first.update(model)
    second = EMA(model, decay=0.1)
    second.load_state_dict(first.state_dict())
    assert second.step == first.step and second.decay == first.decay
    assert torch.equal(second.shadow["weight"], first.shadow["weight"])


def test_snapshot_is_plain_cpu_weights():
    ema = EMA(tiny(), decay=0.9)
    snapshot = ema.snapshot_state()
    assert set(snapshot) == set(ema.shadow)
    assert all(v.device.type == "cpu" and not v.requires_grad for v in snapshot.values())


# ---- schedule --------------------------------------------------------------


def test_noam_is_the_default_and_matches_the_original_scheduler():
    """``NoamLR(model_size=128, factor=2, warmup=4000)``, stepped once per step.

    The preprint gives no optimizer; the original training code does, and this
    is it -- peak at the end of warmup, inverse-sqrt decay after.
    """
    settings = OptimSettings()
    assert settings.optimizer == "noam"
    assert settings.betas == (0.9, 0.98) and settings.eps == pytest.approx(1e-9)
    assert settings.max_grad_norm == 0.0, "the original clips nothing"

    def noam(step):
        n = max(step, 1)
        return 2.0 * (128**-0.5 * min(n**-0.5, n * 4000**-1.5))

    for step in (0, 1, 100, 4000, 40000):
        assert learning_rate(step, settings, 100000) == pytest.approx(noam(step))
    peak = learning_rate(4000, settings, 100000)
    assert learning_rate(100, settings, 100000) < peak
    assert learning_rate(40000, settings, 100000) < peak


def test_adamw_warmup_is_linear_and_reaches_the_target():
    settings = OptimSettings(
        optimizer="adamw", lr=1e-3, warmup_steps=100, schedule="constant"
    )
    assert settings.betas == (0.9, 0.999) and settings.eps == pytest.approx(1e-8)
    assert learning_rate(0, settings, 1000) == pytest.approx(1e-5)
    assert learning_rate(49, settings, 1000) == pytest.approx(5e-4)
    assert learning_rate(99, settings, 1000) == pytest.approx(1e-3)
    assert learning_rate(500, settings, 1000) == pytest.approx(1e-3)


def test_cosine_decays_to_the_floor():
    settings = OptimSettings(
        optimizer="adamw", lr=1e-3, warmup_steps=0, schedule="cosine", min_lr_ratio=0.1
    )
    assert learning_rate(0, settings, 1000) == pytest.approx(1e-3)
    assert learning_rate(999, settings, 1000) == pytest.approx(1e-4, rel=1e-2)
    mid = learning_rate(500, settings, 1000)
    assert 1e-4 < mid < 1e-3


def test_unknown_schedule_or_optimizer_is_rejected():
    with pytest.raises(ValueError, match="Unknown schedule"):
        learning_rate(
            0, OptimSettings(optimizer="adamw", schedule="magic", warmup_steps=0), 10
        )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        OptimSettings(optimizer="lion")


def test_the_departure_from_the_original_optimizer_is_recorded():
    """A checkpoint must say whether it used the original schedule or ours."""
    assert "allatom_design" in OptimSettings().source
    assert "not the original" in OptimSettings(optimizer="adamw").source


# ---- checkpoints -----------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A very short real run, so checkpoint semantics are tested on real state."""
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint, repo_root
    from pxf.train.data import build_loader

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    paths = [
        str(repo_root() / f"fampnn/data/casp14/pdbs/{n}.pdb") for n in ("T1031", "T1033")
    ]
    dataset, loader = build_loader(paths, batch_size=2, crop_size=48, shuffle=False)
    out = tmp_path_factory.mktemp("run")
    trainer = Trainer(
        model,
        bundle["model_cfg"],
        loader,
        out_dir=out,
        dataset=dataset,
        optim=OptimSettings(optimizer="adamw", lr=3e-4, warmup_steps=2),
        train=TrainSettings(
            max_steps=4,
            log_every=2,
            checkpoint_every=0,
            ema_relative_length=0.25,
            train_confidence=False,
        ),
    )
    result = trainer.train(progress=None)
    return trainer, result, bundle


def test_a_short_run_completes_and_logs(trained):
    trainer, result, _ = trained
    assert result["steps"] == 4
    log = trainer.out_dir / "train_log.jsonl"
    assert log.is_file() and log.read_text().strip()
    import json

    record = json.loads(log.read_text().splitlines()[0])
    assert "loss_main" in record and "grad_norm" in record and "window_steps" in record


def test_checkpoint_serves_inference_and_resume(trained):
    trainer, result, _ = trained
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    # The pair upstream inference needs...
    assert "state_dict" in state and "model_cfg" in state
    # ...and the training state the released checkpoints lack.
    assert state["step"] == 4
    assert "optimizer" in state and state["ema"] is not None
    # Settings are recorded because the paper does not specify them, and this
    # run deliberately departs from what the original training code used.
    assert state["optim_settings"]["lr"] == pytest.approx(3e-4)
    assert "not the original" in state["optim_settings"]["source"]
    # The objective's own settings travel with the weights too.
    assert state["train_settings"]["loss"]["sidechain_reduction"] == "per_token"
    assert state["augment_eps"] == pytest.approx(0.0)


def test_trained_checkpoint_loads_into_upstream_inference(trained):
    from fampnn.model.sd_model import SeqDenoiser

    _, result, _ = trained
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    model = SeqDenoiser(state["model_cfg"])
    model.load_state_dict(state["state_dict"], strict=True)  # must be strict
    assert hasattr(model, "sidechain_pack")


def test_resume_restores_step_and_optimizer(trained):
    trainer, result, bundle = trained
    from fampnn.model.sd_model import SeqDenoiser

    fresh = SeqDenoiser(bundle["model_cfg"])
    twin = Trainer(
        fresh,
        bundle["model_cfg"],
        trainer.loader,
        out_dir=trainer.out_dir,
        train=TrainSettings(ema_relative_length=0.25),
    )
    assert twin.resume(result["checkpoint"]) == 4
    assert twin.step == 4
    assert twin.optimizer.state_dict()["state"], "optimizer moments should be restored"


def test_released_weights_cannot_be_resumed(trained):
    """They carry only state_dict + model_cfg, so say so instead of half-resuming."""
    trainer, _, bundle = trained
    from pxf.provenance import fampnn_checkpoint

    with pytest.raises(ValueError, match="cannot be resumed"):
        trainer.resume(fampnn_checkpoint("0.0"))


def test_training_actually_reduces_the_loss(tmp_path):
    """The end-to-end check: overfit two structures and require the error to fall.

    Every other test here checks a contract. This one checks that the objective,
    the gradients and the optimizer are wired to each other -- a mis-signed loss
    or a detached target would pass all the contract tests and fail this one.

    It is read off the *normalized* diagnostics, not off ``loss_main``. With the
    original's normalization the sequence term is a sum over masked tokens
    divided by the crop length, so its value tracks how much the interpolant
    happened to hide in that window (``keep_fraction`` swings between 0.5 and
    0.95 here) and a 10-step window is dominated by that, not by learning.
    """
    import json

    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint, repo_root
    from pxf.train.data import build_loader

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    paths = [
        str(repo_root() / f"fampnn/data/casp14/pdbs/{n}.pdb") for n in ("T1031", "T1033")
    ]
    dataset, loader = build_loader(paths, batch_size=2, crop_size=64, shuffle=False, seed=0)
    trainer = Trainer(
        model,
        bundle["model_cfg"],
        loader,
        out_dir=tmp_path,
        dataset=dataset,
        optim=OptimSettings(optimizer="adamw", lr=1e-3, warmup_steps=5),
        train=TrainSettings(
            max_steps=120, log_every=10, checkpoint_every=0, train_confidence=False, seed=0
        ),
    )
    trainer.train(progress=None)

    records = [
        json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()
    ]
    assert len(records) >= 8

    def window(key, records):
        return sum(r[key] for r in records) / len(records)

    # Both objectives must be learning, not just one carrying the sum. The
    # thresholds differ only because 120 clipped steps move them at different
    # rates, not because one matters more.
    for key, factor in (("sidechain_mse_local", 1.8), ("mlm_per_token", 1.4)):
        first, last = window(key, records[:2]), window(key, records[-2:])
        assert last < first / factor, f"{key} only moved {first:.4f} -> {last:.4f}"
    # Memorizing two crops should leave masked-token accuracy high. Averaged over
    # the second half, not the last window: at a 0.9 keep rate a window can hold
    # only a couple of masked tokens, so a single miss reads as 0.5.
    assert window("sequence_accuracy", records[len(records) // 2 :]) > 0.85
