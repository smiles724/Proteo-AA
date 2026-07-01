"""
PXDesignTrainer — the training loop for PXDesign-d.

Adapted from `Protenix/runner/train.py` but stripped down to what PXDesign-d
actually needs:

  - swap the Protenix model for `ProtenixDesignTrain`
  - swap `ProtenixLoss` for `PXDesignLoss` (eq. 4 in the technical report)
  - swap Protenix's data loader for the curriculum-aware setup we built in
    pieces 4(a)–4(c): `DesignSourceDataset` + `CurriculumMultiDataset` +
    `CurriculumSampler` (or `CurriculumDistributedSampler` under DDP)
  - drop Protenix's confidence-head machinery (`SymmetricPermutation`,
    `LDDTMetrics`, `mc_dropout_apply_rate`, `label_full_dict`) — PXDesign-d
    has no confidence head, so these are dead weight

What's still inherited from Protenix unchanged:

  - `EMAWrapper`, `seed_everything`, `DIST_WRAPPER`, `to_device`
  - the train-step layout (AMP autocast → forward → backward → optimizer step,
    grad accumulation, NaN-loss skip, grad clip)
  - the run loop layout (eval/log/save-checkpoint intervals)

The trainer exposes `train_step()` / `evaluate()` / `run()` so tests can drive
it step-by-step without needing the full `run()` loop.
"""
from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from pxdesign_train.data.curriculum import (
    CurriculumDistributedSampler,
    CurriculumMultiDataset,
    CurriculumSampler,
    CurriculumSchedule,
)
from pxdesign_train.loss import PXDesignLoss
from pxdesign_train.model import ProtenixDesignTrain

logger = logging.getLogger(__name__)


def _identity_collate(batch):
    """We use batch_size=1; the DataLoader still hands us a list of length 1.
    Pull the lone item out so the trainer sees the raw dict.

    This matches Protenix's `collate_fn_first` semantics.
    """
    assert len(batch) == 1, f"expected single-item batches, got len={len(batch)}"
    return batch[0]


@dataclass
class TrainerComponents:
    """Plumbing the trainer needs that we *cannot* generate from configs alone.

    Callers build these from their own data layout and pass them in. This
    keeps the trainer agnostic to where PDB / AFDB / MGnify shards live.

    Args:
        train_dataset: a `CurriculumMultiDataset` over your sources.
        schedule: the curriculum schedule (per-source stage1/stage2 weights).
        train_samples_per_epoch: how many samples one `__iter__` draws.
        eval_dataloader: optional, can be a plain DataLoader over a small
            held-out set. Trainer only reads `loss` from eval — no metrics yet.
    """

    train_dataset: CurriculumMultiDataset
    schedule: CurriculumSchedule
    train_samples_per_epoch: int = 1000
    eval_dataloader: Optional[DataLoader] = None


class PXDesignTrainer:
    """Training driver for PXDesign-d.

    Args:
        configs: a parsed `pxdesign_train.configs.configs_train` object.
            Must have nested `.training.{lr, max_steps, warmup_steps,
            ema_decay, checkpoint_interval, log_interval, eval_interval,
            diffusion_batch_size}`, `.loss.{weight_mse, ...}`, plus the
            model-side fields consumed by `ProtenixDesignTrain`.
        components: data-side plumbing — see `TrainerComponents`.
        device: CPU is fine for tests; CUDA preferred for real training.
        rank / world_size: distributed-training topology. Defaults to single-GPU.
        checkpoint_dir: where checkpoints go. Created on rank 0.
        load_checkpoint_path: optional warm-start. With load_strict=False
            this is how you fine-tune from the released `pxdesign_v0.1.0.pt`.
    """

    def __init__(
        self,
        configs: Any,
        components: TrainerComponents,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: Optional[str] = None,
        load_checkpoint_path: Optional[str] = None,
    ) -> None:
        self.configs = configs
        self.components = components
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = world_size > 1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None and self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state.
        self.step = 0
        self.global_step = 0  # increments every micro-batch; step = global_step // iters_to_accumulate
        self.iters_to_accumulate = int(getattr(configs.training, "iters_to_accumulate", 1))

        self._init_model()
        self._init_loss()
        self._init_optimizer()
        self._init_dataloader()

        if load_checkpoint_path:
            self.load_checkpoint(load_checkpoint_path, params_only=True)

    # ----- init helpers -----

    def _init_model(self) -> None:
        self.raw_model = ProtenixDesignTrain(self.configs).to(self.device)
        if self.use_ddp:
            self.model = DDP(
                self.raw_model,
                device_ids=[self.rank] if self.device.type == "cuda" else None,
                find_unused_parameters=False,
                static_graph=False,
            )
        else:
            self.model = self.raw_model

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self._log(f"Model has {n_params:.2f}M parameters")

        ema_decay = float(getattr(self.configs.training, "ema_decay", 0.0))
        if ema_decay > 0:
            from runner.ema import EMAWrapper  # protenix's EMAWrapper

            ema_keywords = list(getattr(self.configs, "ema_mutable_param_keywords", [""]))
            self.ema_wrapper = EMAWrapper(self.model, ema_decay, ema_keywords)
            self.ema_wrapper.register()
        else:
            self.ema_wrapper = None

    def _init_loss(self) -> None:
        loss_cfg = self.configs.loss
        self.loss_fn = PXDesignLoss(
            weight_mse=loss_cfg.weight_mse,
            weight_lddt=loss_cfg.weight_lddt,
            weight_disto=loss_cfg.weight_disto,
            weight_aa=getattr(loss_cfg, "weight_aa", 0.0),
            aa_ignore_index=getattr(
                getattr(self.configs, "residue_type", object()),
                "ignore_index",
                -100,
            ),
            aa_time_weighting=bool(getattr(loss_cfg, "aa_time_weighting", False)),
            sigma_low_threshold=loss_cfg.sigma_low_threshold,
            no_bins=loss_cfg.no_bins,
            min_bin=loss_cfg.min_bin,
            max_bin=loss_cfg.max_bin,
            lddt_radius=loss_cfg.lddt_radius,
            # On CPU the rigid-align uses CUDA autocast; disable in tests.
            align_before_mse=loss_cfg.align_before_mse and torch.cuda.is_available(),
        )

    def _init_optimizer(self) -> None:
        cfg = self.configs.training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg.lr),
            betas=(0.9, 0.95),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )
        # Linear warmup then constant lr. Matches Protenix's demo style; if
        # callers want cosine they can swap this out post-hoc on `self.scheduler`.
        warmup = int(getattr(cfg, "warmup_steps", 0))

        def _lr_lambda(step):
            return min(1.0, (step + 1) / max(1, warmup)) if warmup > 0 else 1.0

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

    def _init_dataloader(self) -> None:
        c = self.components
        if self.use_ddp:
            sampler: torch.utils.data.Sampler = CurriculumDistributedSampler(
                dataset=c.train_dataset,
                schedule=c.schedule,
                num_samples=c.train_samples_per_epoch,
                num_replicas=self.world_size,
                rank=self.rank,
            )
        else:
            sampler = CurriculumSampler(
                dataset=c.train_dataset,
                schedule=c.schedule,
                num_samples=c.train_samples_per_epoch,
                seed=int(getattr(self.configs, "seed", 0)),
            )
        self.train_sampler = sampler
        self.train_dl = DataLoader(
            c.train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=int(getattr(self.configs.training, "num_workers", 0)),
            collate_fn=_identity_collate,
        )
        self.eval_dl = c.eval_dataloader

    # ----- core compute -----

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        def _move(v):
            if isinstance(v, torch.Tensor):
                return v.to(self.device)
            if isinstance(v, dict):
                return {k: _move(x) for k, x in v.items()}
            return v

        return _move(batch)

    def _maybe_add_batch_dim(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Datasets return per-example tensors; the model expects a leading
        batch dim. We add it here rather than in the dataset so the dataset
        layer stays simple.
        """
        def _add(v):
            if isinstance(v, torch.Tensor):
                return v.unsqueeze(0)
            if isinstance(v, dict):
                return {k: _add(x) for k, x in v.items()}
            return v

        return {
            "input_feature_dict": _add(batch["input_feature_dict"]),
            "label_dict": _add(batch["label_dict"]),
            **{k: v for k, v in batch.items() if k not in ("input_feature_dict", "label_dict")},
        }

    def forward_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """One forward pass + composite loss; returns the loss-component dict.

        Public for tests; production code calls `train_step` which wraps this.
        """
        batch = self._to_device(batch)
        out = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            mode="train",
        )
        rep_atom_mask = batch["input_feature_dict"]["distogram_rep_atom_mask"]
        loss_out = self.loss_fn(
            pred_coordinate=out["x_denoised"],
            gt_coordinate_aug=out["x_gt_aug"],
            sigma=out["sigma"],
            coordinate_mask=batch["label_dict"]["coordinate_mask"],
            rep_atom_mask=rep_atom_mask,
            distogram_logits=out.get("distogram_logits"),
            aa_logits=out.get("aa_logits"),
            aa_clean=batch["input_feature_dict"].get("aa_clean"),
            aa_loss_mask=batch["input_feature_dict"].get("aa_loss_mask"),
            aa_t=batch["input_feature_dict"].get("aa_t"),
        )
        return loss_out

    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Single training step. Returns the loss-component dict for logging."""
        self.model.train()
        dtype = self._train_precision()
        ctx = (
            torch.autocast("cuda", dtype=dtype, cache_enabled=False)
            if self.device.type == "cuda" else nullcontext()
        )
        with ctx:
            loss_out = self.forward_loss(batch)
            loss = loss_out["loss"]
        if not torch.isfinite(loss):
            self._log(f"Skip step {self.step}: non-finite loss {loss.item()}")
            return {k: torch.zeros_like(v) for k, v in loss_out.items()}

        (loss / self.iters_to_accumulate).backward()

        is_update = (self.global_step + 1) % self.iters_to_accumulate == 0
        if is_update:
            grad_clip = float(getattr(self.configs.training, "grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            if self.ema_wrapper is not None:
                self.ema_wrapper.update()
            self.step += 1
            if isinstance(self.train_sampler, (CurriculumSampler, CurriculumDistributedSampler)):
                self.train_sampler.set_step(self.step)
        self.global_step += 1
        return loss_out

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_dl is None:
            return {}
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        for batch in self.eval_dl:
            loss_out = self.forward_loss(batch)
            for k, v in loss_out.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            count += 1
        return {k: v / max(1, count) for k, v in sums.items()}

    # ----- run loop -----

    def run(self, max_steps: Optional[int] = None) -> None:
        """Main loop. Stops at `max_steps` (or `configs.training.max_steps`).

        At step boundaries: log, eval, checkpoint, per the configured intervals.
        """
        target_steps = int(max_steps if max_steps is not None else self.configs.training.max_steps)
        cfg = self.configs.training
        log_int = int(cfg.log_interval)
        eval_int = int(cfg.eval_interval)
        ckpt_int = int(cfg.checkpoint_interval)

        while self.step < target_steps:
            for batch in self.train_dl:
                loss_out = self.train_step(batch)

                if self.step > 0 and self.step % log_int == 0:
                    self._log(
                        f"step={self.step} "
                        + " ".join(f"{k}={v.detach().item():.4g}" if isinstance(v, torch.Tensor) else f"{k}={v:.4g}" for k, v in loss_out.items())
                    )
                if eval_int > 0 and self.step > 0 and self.step % eval_int == 0:
                    metrics = self.evaluate()
                    if metrics:
                        self._log(f"step={self.step} eval: {metrics}")
                if ckpt_int > 0 and self.step > 0 and self.step % ckpt_int == 0:
                    self.save_checkpoint()

                if self.step >= target_steps:
                    break

    # ----- checkpointing -----

    def save_checkpoint(self, tag: Optional[str] = None) -> Optional[str]:
        if self.rank != 0 or self.checkpoint_dir is None:
            return None
        name = f"step{self.step}{('_' + tag) if tag else ''}.pt"
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
                "step": self.step,
                "global_step": self.global_step,
            },
            path,
        )
        self._log(f"Saved checkpoint -> {path}")
        return path

    def load_checkpoint(self, path: str, params_only: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        load_strict = bool(getattr(self.configs, "load_strict", False))
        # Strip DDP prefix if loading a DDP checkpoint into a single-GPU model.
        state = ckpt["model"]
        if not self.use_ddp and any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=load_strict)
        self._log(f"Loaded {path} (missing={len(missing)}, unexpected={len(unexpected)})")
        if not params_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])
            self.step = int(ckpt.get("step", 0))
            self.global_step = int(ckpt.get("global_step", self.step))

    # ----- misc -----

    def _train_precision(self) -> torch.dtype:
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            str(getattr(self.configs, "dtype", "bf16"))
        ]

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            logger.info(msg)
            # Also print, since not every caller configures logging.
            print(msg, flush=True)
