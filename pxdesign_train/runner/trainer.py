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
            held-out set. Trainer averages every scalar returned by `PXDesignLoss`.
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
        checkpoint_params_only: bool = True,
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
            self.load_checkpoint(load_checkpoint_path, params_only=checkpoint_params_only)

    # ----- init helpers -----

    def _init_model(self) -> None:
        self.raw_model = ProtenixDesignTrain(self.configs).to(self.device)
        self._apply_trainable_filter()
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
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        self._log(f"Model has {n_params:.2f}M parameters ({n_trainable:.2f}M trainable)")

        ema_decay = float(getattr(self.configs.training, "ema_decay", 0.0))
        if ema_decay > 0:
            from runner.ema import EMAWrapper  # protenix's EMAWrapper

            ema_keywords = list(getattr(self.configs, "ema_mutable_param_keywords", [""]))
            self.ema_wrapper = EMAWrapper(self.model, ema_decay, ema_keywords)
            self.ema_wrapper.register()
        else:
            self.ema_wrapper = None

    def _apply_trainable_filter(self) -> None:
        keywords = list(getattr(self.configs.training, "trainable_param_keywords", []) or [])
        if not keywords:
            return
        n_trainable = 0
        for name, param in self.raw_model.named_parameters():
            keep = any(str(k) in name for k in keywords)
            param.requires_grad_(keep)
            if keep:
                n_trainable += param.numel()
        if n_trainable == 0:
            raise ValueError(
                "trainable_param_keywords matched no parameters: "
                + ", ".join(map(str, keywords))
            )
        self._log(
            "Trainable parameter filter: "
            + ", ".join(map(str, keywords))
        )

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
            # M5: side-chain loss weights透传 from config (previously stuck at
            # PXDesignLoss defaults regardless of config).
            weight_sc_local=float(getattr(loss_cfg, "weight_sc_local", 1.0)),
            weight_sc_phys=float(getattr(loss_cfg, "weight_sc_phys", 0.1)),
            weight_sc_global=float(getattr(loss_cfg, "weight_sc_global", 0.5)),
        )
        # Post-refinement weights are passed per-call in forward_loss.
        self._weight_bb_post = float(getattr(loss_cfg, "weight_bb_post", 1.0))
        self._weight_aa_post = float(getattr(loss_cfg, "weight_aa_post", 1.0))

    def _init_optimizer(self) -> None:
        cfg = self.configs.training
        self.train_mode = str(getattr(cfg, "train_mode", "joint"))
        warmup = int(getattr(cfg, "warmup_steps", 0))

        def _make_adam(params):
            return torch.optim.Adam(
                params,
                lr=float(cfg.lr),
                betas=(0.9, 0.95),
                weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
            )

        # Linear warmup then constant lr. Matches Protenix's demo style; if
        # callers want cosine they can swap this out post-hoc on `self.scheduler`.
        def _make_sched(opt):
            def _lr_lambda(step):
                return min(1.0, (step + 1) / max(1, warmup)) if warmup > 0 else 1.0

            return torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

        # Alternating is a Stage-III concept: it only applies when BOTH modules
        # are actively trained. Stage I is backbone-only (no side-chain params)
        # and Stage II warms up S_phi with a frozen backbone (no trainable
        # backbone params); in those cases we transparently fall back to joint
        # (normal single-optimizer training) with a log line rather than
        # erroring, so 'alternating' is a safe default across all stages. Multi-
        # GPU IS supported: the two phases' grads are all-reduced manually before
        # the optimizer step (see `_allreduce_alternating_grads`).
        sc_params, bb_params = [], []
        if self.train_mode == "alternating":
            fallback_reason = None
            sc_params, bb_params = self._split_sc_bb_params()
            if not sc_params:
                fallback_reason = (
                    "no Side-Chain Module parameters (Stage I backbone-only "
                    "or side-chain disabled)"
                )
            elif not bb_params:
                fallback_reason = (
                    "no trainable backbone-group parameters (Stage II "
                    "side-chain warmup with a frozen backbone)"
                )
            if fallback_reason is not None:
                # Uses the single-optimizer code path (internally the "joint"
                # branch), but with only one module trainable it just trains that
                # one module — there is no two-module 'joint' happening here.
                self._log(
                    "train_mode='alternating' not applicable "
                    f"({fallback_reason}) -> single-optimizer training."
                )
                self.train_mode = "joint"

        if self.train_mode == "alternating":
            self.sc_params = sc_params
            self.bb_params = bb_params
            self.sc_optimizer = _make_adam(sc_params)
            self.bb_optimizer = _make_adam(bb_params)
            self.sc_lr_scheduler = _make_sched(self.sc_optimizer)
            self.bb_lr_scheduler = _make_sched(self.bb_optimizer)
            # Aliases so generic code (checkpoint/log) that reaches for a single
            # optimizer still finds one; the step loop uses `_optimizers`.
            self.optimizer = self.bb_optimizer
            self.lr_scheduler = self.bb_lr_scheduler
            self._optimizers = [self.sc_optimizer, self.bb_optimizer]
            self._schedulers = [self.sc_lr_scheduler, self.bb_lr_scheduler]
            self._log(
                f"Alternating training: "
                f"{sum(p.numel() for p in sc_params) / 1e6:.2f}M side-chain params, "
                f"{sum(p.numel() for p in bb_params) / 1e6:.2f}M backbone-group params"
            )
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise ValueError("No trainable parameters")
            self.optimizer = _make_adam(params)
            self.lr_scheduler = _make_sched(self.optimizer)
            self._optimizers = [self.optimizer]
            self._schedulers = [self.lr_scheduler]

    def _split_sc_bb_params(self):
        """Partition trainable params into the Side-Chain group and the Backbone
        group for alternating training.

        The Side-Chain group is exactly the ``SideChainModule`` (parameter names
        containing ``sidechain_module``); everything else trainable — the
        DiffusionModule, the AA head, the h_res generation, and the a/q
        fusion / feedback modules — is the Backbone group. Putting the fusion
        modules in the backbone group is deliberate: they consume the side
        chain's output and are supervised only by the post-refinement loss (part
        of ``loss_bb``), so they must update in the bb phase, not the sc phase.
        """
        sc, bb = [], []
        for name, p in self.raw_model.named_parameters():
            if not p.requires_grad:
                continue
            (sc if "sidechain_module" in name else bb).append(p)
        return sc, bb

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
        # Per-protein breakdown from the most recent `evaluate()`. Defined here so
        # callers can read it unconditionally, before any eval has run.
        self.last_eval_per_protein: list[dict[str, Any]] = []

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
        # In alternating mode we fill grads via torch.autograd.grad, which does
        # NOT engage DDP's reducer; forward through the raw model so DDP's
        # backward-expecting hooks are never armed, and sync grads manually
        # (see `_allreduce_alternating_grads`). Single-GPU: raw_model IS model.
        fwd_model = self.raw_model if getattr(self, "train_mode", "joint") == "alternating" else self.model
        out = fwd_model(
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
            # S_phi emits GLOBAL side-chain coords plus the predicted frame it used.
            # loss.py's `has_global_sc` needs all of sc_pred_global / sc_gt_local /
            # sc_frame_R / sc_frame_t / sc_atom_mask; without these three the whole
            # side-chain coordinate loss block is skipped and sc_local stays 0.0.
            # sc_pred_local is kept for backward compat with older model outputs.
            sc_pred_local=out.get("sc_pred_local"),
            sc_pred_global=out.get("sc_pred_global"),
            sc_frame_R=out.get("sc_frame_R"),
            sc_frame_t=out.get("sc_frame_t"),
            sc_gt_local=batch["input_feature_dict"].get("sc_gt_local"),
            sc_atom_mask=out.get("sc_atom_mask"),
            sc_type_match=out.get("sc_type_match"),
            sc_phys=out.get("sc_phys_val"),
            sc_global=out.get("sc_global_aux"),
            post_pred_coordinate=out.get("post_pred_coordinate"),
            post_gt_coordinate_aug=out.get("post_gt_coordinate_aug"),
            post_aa_logits=out.get("post_aa_logits"),
            eval_ca_atom_mask=batch["input_feature_dict"].get("eval_ca_atom_mask"),
            eval_backbone_atom_mask=batch["input_feature_dict"].get("eval_backbone_atom_mask"),
            weight_bb_post=getattr(self, "_weight_bb_post", 1.0),
            weight_aa_post=getattr(self, "_weight_aa_post", 1.0),
            backbone_atom_mask=batch["input_feature_dict"].get("backbone_loss_mask"),
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

        # --- Accumulate gradients (mode-specific) ---
        if self.train_mode == "alternating":
            self._alternating_accumulate(loss_out)
        else:
            (loss / self.iters_to_accumulate).backward()

        # --- Optimizer step on the accumulation boundary (shared) ---
        is_update = (self.global_step + 1) % self.iters_to_accumulate == 0
        if is_update:
            # Sync the alternating phases' grads across ranks before clipping/step
            # (autograd.grad bypassed DDP's own all-reduce). No-op single-GPU.
            if self.train_mode == "alternating":
                self._allreduce_alternating_grads()
            grad_clip = float(getattr(self.configs.training, "grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            for opt in self._optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in self._schedulers:
                sched.step()
            if self.ema_wrapper is not None:
                self.ema_wrapper.update()
            self.step += 1
            if isinstance(self.train_sampler, (CurriculumSampler, CurriculumDistributedSampler)):
                self.train_sampler.set_step(self.step)
        self.global_step += 1
        return loss_out

    def _alternating_accumulate(self, loss_out: dict[str, torch.Tensor]) -> None:
        """One forward's worth of gradient for the alternating (Stage III) scheme.

        From a SINGLE forward we take two independent gradient sets via
        ``torch.autograd.grad`` so the phases never contaminate each other:

          - Phase A (side chain): ``d loss_sc / d(sc params)`` — the Side-Chain
            Module trained against the backbone's fixed output this step.
          - Phase B (backbone group): ``d loss_bb / d(bb params)`` — backbone +
            AA head + h_res generation + a/q fusion, with the gradient flowing
            THROUGH the (un-stepped) Side-Chain Module. Because we differentiate
            ``loss_bb`` w.r.t. the backbone params directly, the side chain acts
            as a conduit (never a ``no_grad`` barrier) yet its own params are not
            updated here — matching the "freeze via requires_grad=False, not
            no_grad" requirement without touching any requires_grad flag.

        Gradients are written into ``.grad`` (scaled by ``iters_to_accumulate``)
        so grad accumulation and the shared optimizer-step boundary keep working.
        """
        acc = self.iters_to_accumulate

        # Phase A — skip cleanly if this batch produced no side-chain output
        # (loss_sc is then a detached zero, so requires_grad is False).
        loss_sc = loss_out["loss_sc"]
        if loss_sc.requires_grad:
            sc_grads = torch.autograd.grad(
                loss_sc, self.sc_params, retain_graph=True, allow_unused=True
            )
            self._accumulate_into_grad(self.sc_params, sc_grads, acc)

        # Phase B — last consumer of the shared graph, so no retain.
        bb_grads = torch.autograd.grad(
            loss_out["loss_bb"], self.bb_params, retain_graph=False, allow_unused=True
        )
        self._accumulate_into_grad(self.bb_params, bb_grads, acc)

    @staticmethod
    def _accumulate_into_grad(params, grads, acc: int) -> None:
        """Add freshly computed grads into ``param.grad`` (creating it if None),
        mirroring what ``loss.backward()`` would do under grad accumulation."""
        for p, g in zip(params, grads):
            if g is None:
                continue  # param not in this loss's graph (allow_unused)
            g = (g / acc).detach()
            p.grad = g if p.grad is None else (p.grad + g)

    def _allreduce_alternating_grads(self) -> None:
        """Average the alternating phases' gradients across DDP ranks.

        The alternating path fills ``param.grad`` via ``torch.autograd.grad``,
        which does NOT trigger DDP's gradient all-reduce hooks (those fire only
        on ``loss.backward()``). So after each rank has accumulated its local
        grads we average them across ranks manually — exactly what DDP would do —
        once at the accumulation boundary, right before the optimizer step.

        A param can legitimately have ``grad is None`` on one rank but not
        another (e.g. a batch with no side-chain atoms skips Phase A, so its
        ``loss_sc`` is a detached zero and the side-chain params get no grad). We
        materialise a zero grad in that case so every rank participates in the
        collective with matching shapes; averaging by ``world_size`` then matches
        DDP semantics (a rank with no contribution counts as zero).
        """
        if not (self.use_ddp and dist.is_available() and dist.is_initialized()):
            return
        world = float(self.world_size)
        for p in (self.sc_params + self.bb_params):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Score the whole validation set.

        Returns the SET-WIDE MEAN of every loss component. That return contract is
        deliberately unchanged — `scripts/evaluation/eval_protenix_monomer.py`
        serialises it straight to JSON as "metrics".

        The per-protein rows the mean was computed from are left on
        `self.last_eval_per_protein`: one dict per eval item, carrying its
        position, its `sample_id`, and every loss component for that single
        protein. The eval loader is built with `batch_size=1` and `shuffle=False`
        (see `build_eval_dataloader`), so one batch IS one protein and the order
        is stable across evals — which is what makes the rows comparable
        step-to-step.
        """
        if self.eval_dl is None:
            self.last_eval_per_protein = []
            return {}
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        per_protein: list[dict[str, Any]] = []
        for i, batch in enumerate(self.eval_dl):
            loss_out = self.forward_loss(batch)
            row = {k: float(v.detach()) for k, v in loss_out.items()}
            for k, v in row.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            # `index` is kept alongside `sample_id` on purpose: a crop retry can
            # make two eval positions resolve to the same complex, so sample_id
            # alone is not guaranteed unique within a pass.
            per_protein.append(
                {"index": i, "sample_id": str(batch.get("sample_id", f"idx{i}")), **row}
            )
        self.last_eval_per_protein = per_protein
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
                # `self.step` counts OPTIMIZER steps, so it only advances on the
                # accumulation boundary — it holds the same value for
                # `iters_to_accumulate` consecutive batches. Anything gated on
                # `self.step % interval == 0` therefore fires once per BATCH, not
                # once per step: with iters_to_accumulate=8 that ran the whole
                # validation set 8x per eval point (measured: 160 val log lines
                # over 20 eval points in job 95909, all 8 repeats numerically
                # identical) and rewrote the same checkpoint file 8x. Gate on the
                # step boundary so each eval/checkpoint happens exactly once.
                step_before = self.step
                loss_out = self.train_step(batch)
                stepped = self.step != step_before

                if self.step > 0 and self.step % log_int == 0:
                    self._log(
                        f"step={self.step} "
                        + " ".join(f"{k}={v.detach().item():.4g}" if isinstance(v, torch.Tensor) else f"{k}={v:.4g}" for k, v in loss_out.items())
                    )
                if stepped and eval_int > 0 and self.step > 0 and self.step % eval_int == 0:
                    metrics = self.evaluate()
                    # Per-protein first, then the set-wide mean, so the mean reads
                    # as the summary line of the block above it.
                    # Every metric on a validation line carries the `val_` prefix,
                    # including the per-protein ones. The log is parsed by key, and
                    # a bare `loss=`/`sc_local=` here would be indistinguishable
                    # from a training row: `plot_training_metrics.py` keeps the LAST
                    # row per (job, step) and would replace the real training point
                    # at every eval step, while `plot_sidechain_warmup_report.py`
                    # branches on `val_sc_local` vs `sc_local` and would file all
                    # ~491 rows as training samples ~40x above the true curve.
                    for row in self.last_eval_per_protein:
                        self._log(
                            f"step={self.step} val_protein={row['sample_id']} "
                            f"val_index={row['index']} "
                            + " ".join(
                                f"val_{k}={v:.4g}"
                                for k, v in row.items()
                                if k not in ("index", "sample_id")
                            )
                        )
                    if metrics:
                        self._log(
                            f"step={self.step} val_n={len(self.last_eval_per_protein)} "
                            + " ".join(f"val_{k}={v:.4g}" for k, v in metrics.items())
                        )
                if stepped and ckpt_int > 0 and self.step > 0 and self.step % ckpt_int == 0:
                    self.save_checkpoint()

                if self.step >= target_steps:
                    break

    # ----- checkpointing -----

    def save_checkpoint(self, tag: Optional[str] = None) -> Optional[str]:
        if self.rank != 0 or self.checkpoint_dir is None:
            return None
        name = f"step{self.step}{('_' + tag) if tag else ''}.pt"
        path = os.path.join(self.checkpoint_dir, name)
        state = {
            "model": self.model.state_dict(),
            "step": self.step,
            "global_step": self.global_step,
            "train_mode": self.train_mode,
            # Structural switches that change WHAT S_phi is fed, not just how it
            # is optimized. Recorded so the next stage can refuse to warm-start
            # across a change in them (see `_check_sidechain_arch`).
            "sidechain_arch": self._sidechain_arch(),
        }
        if self.train_mode == "alternating":
            state["sc_optimizer"] = self.sc_optimizer.state_dict()
            state["bb_optimizer"] = self.bb_optimizer.state_dict()
            state["sc_scheduler"] = self.sc_lr_scheduler.state_dict()
            state["bb_scheduler"] = self.bb_lr_scheduler.state_dict()
        else:
            state["optimizer"] = self.optimizer.state_dict()
            state["scheduler"] = self.lr_scheduler.state_dict()
        torch.save(state, path)
        self._log(f"Saved checkpoint -> {path}")
        return path

    # Switches that change S_phi's INPUT STRUCTURE rather than its optimization.
    # Flipping one of these between stages means the warm-started module meets an
    # input shape/composition it was never trained on -- e.g. `bb_context` turns
    # S_phi's atom axis from 10 side-chain slots into 4 backbone context slots +
    # 10, which changes the intra-residue attention's key set for every residue.
    # Loading anyway does not crash (the parameters are shape-compatible), it just
    # silently degrades, which is exactly the failure mode this project keeps
    # hitting.
    # Two kinds of side-chain switch, and they deserve different treatment.
    #
    # LAYOUT keys change how the EXISTING weights are used. bb_context turns
    # S_phi's atom axis from 10 side-chain slots into 4 backbone context slots +
    # 10, so every pretrained intra-residue block suddenly attends over a
    # different key set. Nothing errors -- the parameters are shape-compatible --
    # it just quietly degrades, which is indistinguishable from "this channel
    # didn't help". Crossing one of these is refused.
    #
    # ADDITIVE keys only decide whether an extra, zero-initialised fusion module
    # exists. Enabling one at a later stage is a normal curriculum move, not a
    # fault: the fusion starts as an exact no-op and learns from there. Warn so
    # the transition is visible in the log, but do not block it.
    SIDECHAIN_LAYOUT_KEYS = ("bb_context", "local_coord_input", "frame_aware_head")
    SIDECHAIN_ADDITIVE_KEYS = ("a_bs_concat", "q_bs")
    SIDECHAIN_ARCH_KEYS = SIDECHAIN_LAYOUT_KEYS + SIDECHAIN_ADDITIVE_KEYS

    def _sidechain_arch(self) -> dict:
        sc = getattr(self.configs, "sidechain", None)
        if sc is None or not getattr(self.configs, "enable_sidechain", False):
            return {}
        return {k: bool(getattr(sc, k, False)) for k in self.SIDECHAIN_ARCH_KEYS}

    def _check_sidechain_arch(self, ckpt: dict) -> None:
        saved = ckpt.get("sidechain_arch")
        current = self._sidechain_arch()
        if not current:
            return                      # this run has no S_phi -- nothing to match
        if not saved:
            # No record in the checkpoint. Two very different cases:
            #   * it has no S_phi weights at all (a Stage I backbone checkpoint) --
            #     there is nothing to mismatch, carry on;
            #   * it DOES have S_phi weights but predates this record, so we cannot
            #     tell which layout they were trained under. Silence would be the
            #     worst outcome: bb_context adds no parameters, so a 10-slot S_phi
            #     loads into a 14-slot model with no shape error and simply degrades,
            #     and a newly enabled fusion sits at its zero init, wired in but
            #     untrained. Refuse to guess.
            has_sc_weights = any(k.startswith("sidechain_module.") for k in ckpt.get("model", {}))
            if has_sc_weights:
                raise ValueError(
                    "Checkpoint contains side-chain weights but records no "
                    f"sidechain_arch, so the layout they were trained under is "
                    f"unknown (this run wants {current}). Retrain the side-chain "
                    "module, or pass --warm-start-params-only with a prefix filter "
                    "that excludes 'sidechain_module.' if you meant to start it "
                    "from scratch."
                )
            return
        mismatched = {
            k: (saved.get(k), current[k])
            for k in current
            if k in saved and saved[k] != current[k]
        }
        def _fmt(keys):
            return ", ".join(
                f"{k}: checkpoint={mismatched[k][0]} current={mismatched[k][1]}"
                for k in keys
            )

        layout = [k for k in self.SIDECHAIN_LAYOUT_KEYS if k in mismatched]
        additive = [k for k in self.SIDECHAIN_ADDITIVE_KEYS if k in mismatched]
        if layout:
            raise ValueError(
                "Side-chain input LAYOUT changed between the checkpoint and this "
                f"run ({_fmt(layout)}). S_phi would be warm-started onto an atom "
                "axis it never saw -- the parameters still fit, so this would not "
                "fail anywhere later, it would just silently underperform. Keep "
                "these identical across stages, or train the side-chain module "
                "from scratch."
            )
        if additive:
            self._log(
                f"Side-chain fusion channels changed ({_fmt(additive)}). A channel "
                "switched ON has no weights in this checkpoint and starts from its "
                "zero init -- an exact no-op at step 0, then it learns. A channel "
                "switched OFF leaves its trained weights unused. Both are legal "
                "curriculum moves; logging so the change is visible."
            )

    def _migrate_atom_name_vocab(self, state: dict) -> dict:
        """Zero-pad S_phi's atom-name embedding when an OLDER checkpoint is loaded.

        The atom-name vocab grew when the 4 backbone names (N, CA, C, O) were appended
        for the 14-slot S_phi axis: ATOM_VOCAB_SIZE 33 -> 37. The 32 side-chain ids
        (1..32) are FROZEN — append-only — so rows 0..32 keep their exact meaning and the
        new rows are simply absent from old checkpoints.

        Without this, every existing side-chain checkpoint becomes unloadable: strict=False
        does NOT tolerate a shape mismatch, it still raises
        RuntimeError("size mismatch for sidechain_module.atom_embed.weight ...").
        """
        out = dict(state)
        for k, v in state.items():
            if not k.endswith("atom_embed.weight") or not torch.is_tensor(v):
                continue
            cur = self.model.state_dict().get(k)
            if cur is None or cur.shape == v.shape:
                continue
            if cur.dim() != 2 or v.dim() != 2 or cur.shape[1] != v.shape[1]:
                continue                       # not the vocab axis — leave it to load_state_dict
            if cur.shape[0] <= v.shape[0]:
                continue                       # shrinking is not a migration we understand
            pad = cur.new_zeros(cur.shape[0] - v.shape[0], cur.shape[1])
            out[k] = torch.cat([v.to(cur.dtype).to(cur.device), pad], dim=0)
            self._log(
                f"Migrated {k}: {tuple(v.shape)} -> {tuple(out[k].shape)} "
                f"(zero-padded {pad.shape[0]} new atom-name rows; existing ids unchanged)"
            )
        return out

    def load_checkpoint(self, path: str, params_only: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        load_strict = bool(getattr(self.configs, "load_strict", False))
        # Strip DDP prefix if loading a DDP checkpoint into a single-GPU model.
        self._check_sidechain_arch(ckpt)
        state = ckpt["model"]
        if not self.use_ddp and any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        state = self._migrate_atom_name_vocab(state)
        if params_only and not load_strict:
            include_prefixes = getattr(
                getattr(self.configs, "training", object()),
                "checkpoint_include_prefixes",
                None,
            )
            if include_prefixes:
                include_prefixes = tuple(str(p) for p in include_prefixes)
                before = len(state)
                state = {
                    k: v for k, v in state.items()
                    if k.startswith(include_prefixes)
                }
                self._log(
                    "Filtered params-only checkpoint to prefixes "
                    f"{include_prefixes}: kept {len(state)}/{before} tensors"
                )
            model_state = self.model.state_dict()
            skipped = []
            filtered = {}
            for k, v in state.items():
                cur = model_state.get(k)
                if cur is not None and torch.is_tensor(v) and cur.shape != v.shape:
                    skipped.append((k, tuple(v.shape), tuple(cur.shape)))
                    continue
                filtered[k] = v
            if skipped:
                preview = ", ".join(
                    f"{k}: {old}->{new}" for k, old, new in skipped[:5]
                )
                more = "" if len(skipped) <= 5 else f", ... +{len(skipped) - 5} more"
                self._log(
                    "Skipped shape-incompatible checkpoint tensors during "
                    f"params-only warm start ({len(skipped)}): {preview}{more}"
                )
            state = filtered
        missing, unexpected = self.model.load_state_dict(state, strict=load_strict)
        self._log(f"Loaded {path} (missing={len(missing)}, unexpected={len(unexpected)})")
        if not params_only:
            ckpt_mode = str(ckpt.get("train_mode", "joint"))
            if ckpt_mode != self.train_mode:
                self._log(
                    f"Checkpoint train_mode='{ckpt_mode}' != current "
                    f"'{self.train_mode}'; skipping optimizer/scheduler restore "
                    f"(model weights + step counters still loaded)."
                )
            elif self.train_mode == "alternating":
                self.sc_optimizer.load_state_dict(ckpt["sc_optimizer"])
                self.bb_optimizer.load_state_dict(ckpt["bb_optimizer"])
                self.sc_lr_scheduler.load_state_dict(ckpt["sc_scheduler"])
                self.bb_lr_scheduler.load_state_dict(ckpt["bb_scheduler"])
            else:
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
