"""Training loop for FaMPNN.

Implements the preprint's schedule (Appendix B.2) and fills the gaps it leaves.

What the paper specifies, and this follows:

* CATH: batch 64, fixed size 256 residues, 100k steps, one GPU.
* PDB: batch 8 per GPU on 4 GPUs with 4 gradient accumulation steps, i.e.
  effective batch 128, fixed size 1024 residues, 300k steps.
* Post-hoc EMA for the PDB models (see :mod:`pxf.train.ema` for what is and is
  not reproduced).

What the paper does **not** specify -- optimizer, learning rate, schedule, weight
decay, gradient clipping -- and is therefore chosen here and recorded in every
checkpoint so a run is never ambiguous about it. The defaults are tuned for
*continuing* training from the released weights rather than training from
scratch: a low constant learning rate after a short warmup.

Checkpoints deliberately carry ``state_dict`` and ``model_cfg`` alongside the
optimizer, EMA and step. That means a checkpoint from this loop loads directly
into upstream's inference path *and* can resume -- the released weights carry
only the first pair, which is why they cannot be resumed from.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import math
import time
import torch

from pxf.train.ema import EMA
from pxf.train.step import training_forward


@dataclass
class OptimSettings:
    """Optimization settings. The paper does not state these; these are ours."""
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    schedule: str = "constant"        # constant | cosine
    min_lr_ratio: float = 0.1         # cosine floor, as a fraction of lr
    source: str = "not specified in the preprint; chosen for fine-tuning"


@dataclass
class TrainSettings:
    """Loop settings; the paper's presets are in configs/train_*.yaml."""
    max_steps: int = 100_000
    grad_accum_steps: int = 1
    ema_relative_length: Optional[float] = None
    ema_decay: Optional[float] = None
    log_every: int = 50
    checkpoint_every: int = 5_000
    snapshot_every: int = 0           # >0 also writes plain EMA snapshots
    seed: int = 0
    train_confidence: Optional[bool] = None   # None = the paper's 1-in-8 sampling
    amp_dtype: Optional[str] = None   # None | bf16 | fp16


def learning_rate(step, settings: OptimSettings, max_steps):
    """Warmup then constant or cosine decay."""
    if settings.warmup_steps and step < settings.warmup_steps:
        return settings.lr * (step + 1) / settings.warmup_steps
    if settings.schedule == "constant":
        return settings.lr
    if settings.schedule == "cosine":
        span = max(1, max_steps - settings.warmup_steps)
        progress = min(1.0, (step - settings.warmup_steps) / span)
        floor = settings.lr * settings.min_lr_ratio
        return floor + 0.5 * (settings.lr - floor) * (1 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule {settings.schedule!r}")


class Trainer:
    """Drives :func:`pxf.train.step.training_forward` over a data loader."""

    def __init__(self, model, model_cfg, loader, *, out_dir, optim=None, train=None,
                 device=None, dataset=None):
        self.model = model
        self.model_cfg = model_cfg
        self.loader = loader
        self.dataset = dataset
        self.optim_settings = optim or OptimSettings()
        self.settings = train or TrainSettings()
        self.out_dir = Path(out_dir)
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.optim_settings.lr, betas=tuple(self.optim_settings.betas),
            eps=self.optim_settings.eps, weight_decay=self.optim_settings.weight_decay)
        self.ema = None
        if self.settings.ema_relative_length or self.settings.ema_decay:
            self.ema = EMA(model, decay=self.settings.ema_decay,
                           relative_length=self.settings.ema_relative_length)
        self.step = 0
        # MAR's masking and EDM's timestep draws use the *global* generator, so
        # seed it as well -- otherwise a run silently depends on whatever ambient
        # RNG state the process was left in, and is not reproducible.
        torch.manual_seed(self.settings.seed)
        self.generator = torch.Generator().manual_seed(self.settings.seed)
        self._log_path = self.out_dir / "train_log.jsonl"
        self._amp = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.settings.amp_dtype]

    # ---- persistence -----------------------------------------------------

    def checkpoint_state(self, *, include_training=True):
        """A checkpoint that upstream inference can load, and this loop can resume."""
        state = dict(state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                     model_cfg=self.model_cfg)
        if include_training:
            state.update(step=self.step,
                         optimizer=self.optimizer.state_dict(),
                         ema=self.ema.state_dict() if self.ema else None,
                         optim_settings=asdict(self.optim_settings),
                         train_settings=asdict(self.settings),
                         generator=self.generator.get_state())
        return state

    def save(self, tag=None, *, include_training=True):
        name = f"step{self.step:08d}" if tag is None else tag
        path = self.out_dir / "checkpoints" / f"{name}.pt"
        torch.save(self.checkpoint_state(include_training=include_training), path)
        return path

    def save_snapshot(self):
        """Plain EMA weights, for post-hoc EMA reconstruction after training."""
        if not self.ema:
            return None
        path = self.out_dir / "checkpoints" / f"ema_snapshot_step{self.step:08d}.pt"
        torch.save(dict(state_dict=self.ema.snapshot_state(), model_cfg=self.model_cfg,
                        step=self.step), path)
        return path

    def resume(self, path):
        """Restore weights, optimizer, EMA, step and RNG from a checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device)
        if "optimizer" not in state:
            raise ValueError(
                f"{path} has no optimizer state, so it cannot be resumed -- only "
                "warm-started. Load it as the initial weights instead (the released "
                "FaMPNN checkpoints are in this category).")
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.step = int(state.get("step", 0))
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"].cpu())
        return self.step

    # ---- the loop --------------------------------------------------------

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _log(self, record):
        with self._log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def train(self, *, max_steps=None, progress=print):
        """Run until ``max_steps``, accumulating gradients as configured."""
        target = int(max_steps or self.settings.max_steps)
        accum = max(1, int(self.settings.grad_accum_steps))
        self.model.train()
        started = time.time()
        epoch = 0
        pending, running, counts = 0, {}, {}

        while self.step < target:
            if self.dataset is not None:
                self.dataset.set_epoch(epoch)
            for batch in self.loader:
                if self.step >= target:
                    break
                batch = self._to_device(batch)
                context = (torch.autocast(device_type=self.device.type, dtype=self._amp)
                           if self._amp else torch.enable_grad())
                with context:
                    out = training_forward(self.model, batch,
                                           train_confidence=self.settings.train_confidence,
                                           generator=self.generator)
                (out.total / accum).backward()
                pending += 1
                for key, value in out.scalars().items():
                    running[key] = running.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

                if pending < accum:
                    continue

                lr = learning_rate(self.step, self.optim_settings, target)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.optim_settings.max_grad_norm))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)
                self.step += 1

                if self.step % max(1, self.settings.log_every) == 0:
                    # Divide each key by how many steps reported it, so the
                    # intermittent confidence loss is not diluted by the window.
                    record = {k: v / max(1, counts[k]) for k, v in running.items()}
                    record.update(step=self.step, lr=lr, grad_norm=grad_norm,
                                  confidence_steps=counts.get('loss_confidence', 0),
                                  window_steps=pending,
                                  seconds=round(time.time() - started, 1))
                    self._log(record)
                    if progress:
                        progress(f"step {self.step:>7d}  main {record['loss_main']:.4f}  "
                                 f"mlm {record['loss_mlm']:.4f}  "
                                 f"diff {record['loss_diffusion']:.4f}  "
                                 f"seq_acc {record.get('sequence_accuracy', float('nan')):.3f}  "
                                 f"lr {lr:.2e}  |g| {grad_norm:.2f}")
                pending, running, counts = 0, {}, {}

                if self.settings.checkpoint_every and self.step % self.settings.checkpoint_every == 0:
                    self.save()
                if self.settings.snapshot_every and self.step % self.settings.snapshot_every == 0:
                    self.save_snapshot()
            epoch += 1

        final = self.save(tag="final")
        return dict(steps=self.step, checkpoint=str(final),
                    seconds=round(time.time() - started, 1))
