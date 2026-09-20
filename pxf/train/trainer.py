"""Training loop for FaMPNN.

The schedule comes from the preprint (Appendix B.2); the optimizer comes from
the original training code, ``allatom_design`` (commit ``51c9d53``), which the
paper does not describe:

    optim.optimizer: noam
    Adam(lr=0, betas=(0.9, 0.98), eps=1e-9)
    NoamLR(model_size=128, factor=2, warmup=4000)
    trainer.gradient_clip_val: 0.0        # nothing is clipped
    trainer.precision: bf16-mixed
                       -- allatom_design/configs/seq_denoiser/seq_denoiser.yaml
                          allatom_design/model/seq_denoiser/lit_sd_model.py

so ``optimizer="noam"`` is the default here and reproduces it. ``"adamw"`` keeps
the low-constant-lr setup that suits *continuing* training from the released
weights on a small set, which is what this loop is usually used for; it is a
deliberate departure and is recorded as one.

From the paper, unchanged:

* CATH: batch 64, fixed size 256 residues, 100k steps, one GPU.
* PDB: batch 8 per GPU on 4 GPUs with 4 gradient accumulation steps, i.e.
  effective batch 128, fixed size 1024 residues, 300k steps.

**Structural noise is a model setting, not a data setting.** "The 0.3 A model"
is ``ProteinFeatures.augment_eps = 0.3``, applied to the encoder's atom14 input
in train mode and never to the diffusion target, so ``TrainSettings.
structural_noise`` writes it onto the model rather than perturbing the dataset.

Checkpoints deliberately carry ``state_dict`` and ``model_cfg`` alongside the
optimizer, EMA and step. That means a checkpoint from this loop loads directly
into upstream's inference path *and* can resume -- the released weights carry
only the first pair, which is why they cannot be resumed from.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from pxf.train.ema import EMA
from pxf.train.losses import LossSettings
from pxf.train.step import training_forward

OPTIMIZERS = ("noam", "adamw")

ORIGINAL_SOURCE = "allatom_design configs/seq_denoiser/seq_denoiser.yaml (optim.noam)"
FINETUNE_SOURCE = "not the original; chosen here for fine-tuning from released weights"


@dataclass
class OptimSettings:
    """Optimizer and schedule.

    ``noam`` is what the released weights were trained with, down to the betas
    and epsilon. ``adamw`` is the fine-tuning alternative: a low constant rate
    after a short warmup, which is a departure and says so in ``source``.
    """

    optimizer: str = "noam"
    # -- noam (the original) --
    noam_factor: float = 2.0
    noam_warmup_steps: int = 4_000
    noam_model_size: int = 128  # the MPNN hidden dim, hardcoded upstream too
    # -- adamw --
    lr: float = 1e-4
    warmup_steps: int = 1_000
    schedule: str = "constant"  # constant | cosine
    min_lr_ratio: float = 0.1  # cosine floor, as a fraction of lr
    weight_decay: float = 0.0
    # -- shared --
    betas: tuple | None = None  # (0.9, 0.98) for noam, (0.9, 0.999) for adamw
    eps: float | None = None  # 1e-9 for noam, 1e-8 for adamw
    # 0 disables clipping, which is what the original does
    # (trainer.gradient_clip_val: 0.0); the fine-tuning setup clips at 1.0.
    max_grad_norm: float | None = None
    source: str | None = None

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}; choose from {list(OPTIMIZERS)}"
            )
        if self.betas is None:
            self.betas = (0.9, 0.98) if self.optimizer == "noam" else (0.9, 0.999)
        self.betas = tuple(self.betas)
        if self.eps is None:
            self.eps = 1e-9 if self.optimizer == "noam" else 1e-8
        if self.max_grad_norm is None:
            self.max_grad_norm = 0.0 if self.optimizer == "noam" else 1.0
        if self.source is None:
            self.source = (
                ORIGINAL_SOURCE if self.optimizer == "noam" else FINETUNE_SOURCE
            )


@dataclass
class TrainSettings:
    """Loop settings; the paper's presets are in configs/train_*.yaml."""

    max_steps: int = 100_000
    grad_accum_steps: int = 1
    ema_relative_length: float | None = None
    ema_decay: float | None = None
    log_every: int = 50
    checkpoint_every: int = 5_000
    snapshot_every: int = 0  # >0 also writes plain EMA snapshots
    seed: int = 0
    train_confidence: bool | None = None  # None = the original's 1-in-8 sampling
    amp_dtype: str | None = None  # None | bf16 | fp16
    # sigma of the encoder-input coordinate noise: ProteinFeatures.augment_eps,
    # 0.0 and 0.3 being the two released models. None leaves the checkpoint's
    # own value alone.
    structural_noise: float | None = None
    # The objective itself: weights, label smoothing, the two normalizations.
    loss: LossSettings = field(default_factory=LossSettings)
    # Ablation only: score just the side chains the interpolant hid. The
    # original scores every resolved one -- see pxf.train.step.
    hidden_sidechains_only: bool = False

    def __post_init__(self):
        if isinstance(self.loss, dict):
            self.loss = LossSettings(**self.loss)


def learning_rate(step, settings: OptimSettings, max_steps):
    """The learning rate for ``step`` (0-based), under either optimizer."""
    if settings.optimizer == "noam":
        # Vaswani et al.'s schedule, as NoamLR computes it: the scheduler is
        # stepped once per optimizer step and clamps its own counter at 1, so
        # step 0 here is step 1 there.
        n = max(step, 1)
        return settings.noam_factor * (
            settings.noam_model_size**-0.5
            * min(n**-0.5, n * settings.noam_warmup_steps**-1.5)
        )
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


def set_structural_noise(model, sigma):
    """Write ``augment_eps`` onto the encoder's ProteinFeatures.

    This is where the paper's structural noise lives: applied to the atom14
    input inside the encoder, in train mode only, leaving the diffusion target
    clean. Returns the value that is now in effect.
    """
    features = model.denoiser.seq_design_module.features
    if sigma is not None:
        features.augment_eps = float(sigma)
        model.denoiser.seq_design_module.augment_eps = float(sigma)
    return float(features.augment_eps)


class Trainer:
    """Drives :func:`pxf.train.step.training_forward` over a data loader."""

    def __init__(
        self,
        model,
        model_cfg,
        loader,
        *,
        out_dir,
        optim=None,
        train=None,
        device=None,
        dataset=None,
    ):
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

        trainable = [p for p in model.parameters() if p.requires_grad]
        if self.optim_settings.optimizer == "noam":
            # Adam, not AdamW: the original applies no weight decay at all, and
            # the rate is supplied per step by the Noam schedule.
            self.optimizer = torch.optim.Adam(
                trainable,
                lr=0.0,
                betas=self.optim_settings.betas,
                eps=self.optim_settings.eps,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=self.optim_settings.lr,
                betas=self.optim_settings.betas,
                eps=self.optim_settings.eps,
                weight_decay=self.optim_settings.weight_decay,
            )
        self.augment_eps = set_structural_noise(model, self.settings.structural_noise)
        self.ema = None
        if self.settings.ema_relative_length or self.settings.ema_decay:
            self.ema = EMA(
                model,
                decay=self.settings.ema_decay,
                relative_length=self.settings.ema_relative_length,
            )
        self.step = 0
        # MAR's masking and EDM's timestep draws use the *global* generator, so
        # seed it as well -- otherwise a run silently depends on whatever ambient
        # RNG state the process was left in, and is not reproducible.
        torch.manual_seed(self.settings.seed)
        self.generator = torch.Generator().manual_seed(self.settings.seed)
        self._log_path = self.out_dir / "train_log.jsonl"
        self._amp = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.settings.amp_dtype
        ]

    # ---- persistence -----------------------------------------------------

    def checkpoint_state(self, *, include_training=True):
        """A checkpoint that upstream inference can load, and this loop can resume."""
        state = dict(
            state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            model_cfg=self.model_cfg,
        )
        if include_training:
            state.update(
                step=self.step,
                optimizer=self.optimizer.state_dict(),
                ema=self.ema.state_dict() if self.ema else None,
                optim_settings=asdict(self.optim_settings),
                train_settings=asdict(self.settings),
                # The value actually in force, which is the checkpoint's own
                # when structural_noise was left unset.
                augment_eps=self.augment_eps,
                generator=self.generator.get_state(),
            )
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
        torch.save(
            dict(
                state_dict=self.ema.snapshot_state(),
                model_cfg=self.model_cfg,
                step=self.step,
            ),
            path,
        )
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
                "FaMPNN checkpoints are in this category)."
            )
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.step = int(state.get("step", 0))
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"].cpu())
        return self.step

    # ---- the loop --------------------------------------------------------

    def _to_device(self, batch):
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

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
                context = (
                    torch.autocast(device_type=self.device.type, dtype=self._amp)
                    if self._amp
                    else torch.enable_grad()
                )
                with context:
                    out = training_forward(
                        self.model,
                        batch,
                        train_confidence=self.settings.train_confidence,
                        generator=self.generator,
                        settings=self.settings.loss,
                        hidden_sidechains_only=self.settings.hidden_sidechains_only,
                    )
                (out.total / accum).backward()
                pending += 1
                for key, value in out.scalars().items():
                    # Some stats are labels (the active reduction), not numbers.
                    if not isinstance(value, (int, float)):
                        continue
                    running[key] = running.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

                if pending < accum:
                    continue

                lr = learning_rate(self.step, self.optim_settings, target)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                # The original clips nothing (gradient_clip_val: 0.0), but the
                # norm is still worth logging, so compute it either way with an
                # infinite threshold when clipping is off.
                limit = self.optim_settings.max_grad_norm or float("inf")
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], limit
                    )
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)
                self.step += 1

                if self.step % max(1, self.settings.log_every) == 0:
                    # Divide each key by how many steps reported it, so the
                    # intermittent confidence loss is not diluted by the window.
                    record = {k: v / max(1, counts[k]) for k, v in running.items()}
                    record.update(
                        step=self.step,
                        lr=lr,
                        grad_norm=grad_norm,
                        confidence_steps=counts.get("loss_confidence", 0),
                        window_steps=pending,
                        seconds=round(time.time() - started, 1),
                    )
                    self._log(record)
                    if progress:
                        # loss_mlm is normalized by the crop length, so it moves
                        # with the masking rate; ce/tok and scn_mse are the two
                        # numbers a curve should actually be read off.
                        progress(
                            f"step {self.step:>7d}  main {record['loss_main']:.4f}  "
                            f"mlm {record['loss_mlm']:.4f}  "
                            f"diff {record['loss_diffusion']:.4f}  "
                            f"ce/tok {record.get('mlm_per_token', float('nan')):.4f}  "
                            f"scn_mse {record.get('sidechain_mse_local', float('nan')):.4f}  "
                            f"seq_acc {record.get('sequence_accuracy', float('nan')):.3f}  "
                            f"lr {lr:.2e}  |g| {grad_norm:.2f}"
                        )
                pending, running, counts = 0, {}, {}

                if (
                    self.settings.checkpoint_every
                    and self.step % self.settings.checkpoint_every == 0
                ):
                    self.save()
                if (
                    self.settings.snapshot_every
                    and self.step % self.settings.snapshot_every == 0
                ):
                    self.save_snapshot()
            epoch += 1

        final = self.save(tag="final")
        return dict(
            steps=self.step, checkpoint=str(final), seconds=round(time.time() - started, 1)
        )
