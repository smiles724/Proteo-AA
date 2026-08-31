"""Validation must report BOTH the per-protein losses and the set-wide mean.

The averaged number alone hides which proteins a checkpoint is actually failing
on: a side-chain run whose mean sc_local looks flat can be one where a handful of
large residues got much worse while everything else improved. So `evaluate()`
keeps returning the mean (its callers serialise that straight to JSON) and
additionally leaves the rows it averaged on `last_eval_per_protein`.

These tests pin the three properties that make those rows trustworthy:
  * one row per eval item, in loader order;
  * the reported mean is exactly the mean of the rows (no double counting, no
    silently dropped batch);
  * each row is named, and the name survives a provider that cannot name itself.
"""
import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))

# Reuse the established fake-model / synthetic-provider harness rather than
# building a second one that could drift from it.
from test_trainer_integration import (  # noqa: E402
    _SyntheticProvider,
    _make_trainer,
)


def _eval_loader(n_items, named=True, source_name="val"):
    """A batch_size=1 eval loader over `n_items` synthetic complexes.

    Mirrors `build_eval_dataloader`: batch_size=1, shuffle=False, identity
    collate — i.e. one batch is exactly one protein, in a stable order.
    """
    from pxdesign_train.runner import DesignSourceDataset
    from pxdesign_train.runner.trainer import _identity_collate

    provider = _SyntheticProvider(n_items=n_items)
    if named:
        # Stand in for ProtenixComplexProvider.sample_id.
        provider.sample_id = lambda idx: f"prot{idx}"
    ds = DesignSourceDataset(
        provider, source_name=source_name, crop_size=20, hotspot_force_zero_prob=0.0,
    )
    return DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_identity_collate,
    )


def test_evaluate_reports_one_row_per_protein_and_the_mean(monkeypatch):
    n = 3
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(n)

    metrics = trainer.evaluate()
    rows = trainer.last_eval_per_protein

    assert len(rows) == n, f"expected one row per protein, got {len(rows)}"
    assert [r["index"] for r in rows] == list(range(n)), "rows must be in loader order"
    assert [r["sample_id"] for r in rows] == [f"prot{i}" for i in range(n)]

    # Every loss component in the mean must also be present per protein.
    assert metrics, "expected non-empty averaged metrics"
    for key in metrics:
        for r in rows:
            assert key in r, f"{key} missing from per-protein row {r['sample_id']}"

    # The mean must be exactly the mean of the reported rows.
    for key, avg in metrics.items():
        expected = sum(r[key] for r in rows) / n
        assert avg == pytest.approx(expected, rel=1e-6, abs=1e-9), (
            f"{key}: reported mean {avg} != mean of per-protein rows {expected}"
        )


def test_per_protein_values_are_finite_scalars(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2)
    trainer.evaluate()
    for r in trainer.last_eval_per_protein:
        for k, v in r.items():
            if k in ("index", "sample_id"):
                continue
            assert isinstance(v, float), f"{k} should be a plain float, got {type(v)}"
            assert torch.isfinite(torch.tensor(v)), f"{k} not finite for {r['sample_id']}"


def test_unnamed_provider_falls_back_to_positional_label(monkeypatch):
    """A provider with no `sample_id` must not break per-protein reporting.

    The CIF provider and every test double are in this category, so the fallback
    is the common path, not an edge case.
    """
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2, named=False, source_name="src")
    trainer.evaluate()
    ids = [r["sample_id"] for r in trainer.last_eval_per_protein]
    assert ids == ["src#0", "src#1"], ids


def test_no_eval_loader_yields_empty_rows(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = None
    assert trainer.evaluate() == {}
    assert trainer.last_eval_per_protein == []


def test_last_eval_per_protein_exists_before_any_eval(monkeypatch):
    """Callers (e.g. the eval script's JSON dump) read it unconditionally."""
    trainer = _make_trainer(monkeypatch)
    assert trainer.last_eval_per_protein == []


def test_rows_are_replaced_not_appended_across_evals(monkeypatch):
    """Two evals must not accumulate — the run loop evaluates every eval_interval."""
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2)
    trainer.evaluate()
    trainer.evaluate()
    assert len(trainer.last_eval_per_protein) == 2


def test_run_loop_logs_per_protein_and_mean(monkeypatch):
    """The numbers must reach the log, not just the attribute."""
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2)
    trainer.configs.training.eval_interval = 1

    lines = []
    monkeypatch.setattr(type(trainer), "_log", lambda self, msg: lines.append(msg))
    trainer.run(max_steps=1)

    per_protein = [ln for ln in lines if "val_protein=" in ln]
    means = [ln for ln in lines if "val_n=" in ln]
    assert len(per_protein) == 2, f"expected 2 per-protein lines, got {per_protein}"
    assert len(means) == 1, f"expected exactly 1 mean line, got {means}"
    assert "val_loss=" in means[0]
    assert "val_n=2" in means[0]
    for ln in per_protein:
        assert "val_index=" in ln and "val_loss=" in ln


def test_per_protein_metrics_are_all_val_prefixed(monkeypatch):
    """No bare metric name may appear on a per-protein validation line.

    The logs are parsed by KEY. A bare `loss=`/`sc_local=` on a validation line is
    indistinguishable from a training row, and both plotting scripts then fold the
    ~491 per-protein rows (~40x the training value) into the TRAINING curve —
    `plot_training_metrics.py` keeps the last row per (job, step) so it replaces
    the real point outright. Guard the prefix, not the plots.
    """
    import re

    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2)
    trainer.configs.training.eval_interval = 1

    lines = []
    monkeypatch.setattr(type(trainer), "_log", lambda self, msg: lines.append(msg))
    trainer.run(max_steps=1)

    kv = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(?=[-+]?(?:\d|\.\d))")
    for ln in (l for l in lines if "val_protein=" in l):
        # `step` is the shared x-coordinate of every log line, not a metric.
        for key in (k for k in kv.findall(ln) if k != "step"):
            assert key.startswith("val_"), (
                f"metric {key!r} on a per-protein line lacks the val_ prefix; "
                f"log parsers would read it as a training metric: {ln}"
            )


def test_eval_and_checkpoint_fire_once_per_step_not_per_batch(monkeypatch):
    """Regression: with grad accumulation these fired once per BATCH.

    `self.step` advances only on the accumulation boundary, so a naive
    `step % interval == 0` guard is true for all `iters_to_accumulate` batches of
    that step. Job 95909 ran the full 491-protein validation 8x per eval point
    (160 val lines / 20 eval points, all repeats identical) and rewrote the same
    checkpoint 8x. With per-protein logging that is ~3.9k redundant lines each
    eval, so this must stay fixed.
    """
    trainer = _make_trainer(monkeypatch)
    trainer.eval_dl = _eval_loader(2)
    trainer.configs.training.iters_to_accumulate = 4
    trainer.iters_to_accumulate = 4
    trainer.configs.training.eval_interval = 1
    trainer.configs.training.checkpoint_interval = 1

    evals = []
    saves = []
    monkeypatch.setattr(
        type(trainer), "evaluate",
        lambda self: (evals.append(self.step), {"loss": 1.0})[1],
    )
    monkeypatch.setattr(type(trainer), "save_checkpoint", lambda self, tag=None: saves.append(self.step))
    monkeypatch.setattr(type(trainer), "_log", lambda self, msg: None)

    trainer.run(max_steps=2)

    # Two optimizer steps -> two evals and two saves, despite 4 batches each.
    assert evals == [1, 2], f"expected one eval per step, got {evals}"
    assert saves == [1, 2], f"expected one checkpoint per step, got {saves}"


def test_training_log_averages_accumulation_window_once(monkeypatch):
    """One log row represents one optimizer update's effective batch.

    With gradient accumulation, loss components are emitted per micro-batch.
    Logging an arbitrary last micro-batch is noisy, while guarding only on
    ``step % interval`` prints duplicate rows because ``step`` remains fixed
    between optimizer updates.  Average the window and emit exactly once.
    """
    trainer = _make_trainer(monkeypatch)
    trainer.configs.training.iters_to_accumulate = 4
    trainer.iters_to_accumulate = 4
    trainer.configs.training.log_interval = 1
    trainer.configs.training.eval_interval = 0
    trainer.configs.training.checkpoint_interval = 0

    values = iter((1.0, 2.0, 3.0, 4.0))

    def _fake_train_step(self, batch):
        value = next(values)
        self.global_step += 1
        if self.global_step % self.iters_to_accumulate == 0:
            self.step += 1
        return {
            "loss": torch.tensor(value),
            "loss_bb": torch.tensor(2.0 * value),
        }

    lines = []
    monkeypatch.setattr(type(trainer), "train_step", _fake_train_step)
    monkeypatch.setattr(type(trainer), "_log", lambda self, msg: lines.append(msg))

    trainer.run(max_steps=1)

    training_lines = [line for line in lines if " loss=" in f" {line}"]
    assert training_lines == ["step=1 loss=2.5 loss_bb=5"]


def test_training_log_splits_aa_metrics_by_source(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    trainer.configs.training.iters_to_accumulate = 4
    trainer.iters_to_accumulate = 4
    trainer.configs.training.log_interval = 1
    trainer.configs.training.eval_interval = 0
    trainer.configs.training.checkpoint_interval = 0
    trainer.train_dl = [
        {"source_name": "protenix_monomer"},
        {"source_name": "pinder_ppi_complex"},
        {"source_name": "pinder_ppi_complex"},
        {"source_name": "protenix_ppi_complex"},
    ]

    values = iter((1.0, 2.0, 3.0, 4.0))

    def _fake_train_step(self, batch):
        value = next(values)
        self.global_step += 1
        if self.global_step % self.iters_to_accumulate == 0:
            self.step += 1
        return {
            "loss": torch.tensor(value),
            "aa_ce": torch.tensor(value),
            "aa_acc": torch.tensor(value / 10.0),
        }

    lines = []
    monkeypatch.setattr(type(trainer), "train_step", _fake_train_step)
    monkeypatch.setattr(type(trainer), "_log", lambda self, msg: lines.append(msg))

    trainer.run(max_steps=1)

    assert len(lines) == 1
    line = lines[0]
    assert "protenix_monomer_aa_ce=1" in line
    assert "pinder_ppi_complex_aa_ce=2.5" in line
    assert "pinder_ppi_complex_aa_acc=0.25" in line
    assert "protenix_ppi_complex_aa_ce=4" in line


def test_sample_id_is_the_complex_actually_returned(monkeypatch):
    """A crop retry shifts which complex is returned; the label must follow it.

    `DesignSourceDataset.__getitem__` walks forward on crop failure, so naming a
    row by the requested index would mislabel exactly the retried rows. Here item
    0 always fails to crop, so index 0 must report item 1's name.
    """
    from pxdesign_train.runner import DesignSourceDataset

    provider = _SyntheticProvider(n_items=3)
    provider.sample_id = lambda idx: f"prot{idx}"
    ds = DesignSourceDataset(
        provider, source_name="s", crop_size=20, hotspot_force_zero_prob=0.0,
    )

    real_get_one = ds._get_one
    used: list[int] = []

    def _fail_first(idx):
        if idx == 0:
            raise ValueError("DesignCropper: forced failure")
        used.append(idx)
        return real_get_one(idx)

    monkeypatch.setattr(ds, "_get_one", _fail_first)
    item = ds[0]

    # Assert against the index actually used, not a hard-coded one: which index
    # the retry lands on is `_probe_index`'s business (it deliberately jumps
    # rather than stepping to idx+1), and this test is about the LABEL following
    # the returned complex.
    assert used, "expected the retry to reach a working index"
    assert item["sample_id"] == f"prot{used[-1]}"
    assert item["sample_id"] != "prot0", "must not report the index that failed"
