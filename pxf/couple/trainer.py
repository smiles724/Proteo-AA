"""Training loop for the coupling adapters.

Only the adapters train; PXDesign and FaMPNN stay frozen. That is the whole
point of the staged plan -- the coupled system starts exactly equal to the two
pretrained models (adapters are zero-initialized) and improves from there, so
any gain is attributable to the coupling rather than to fine-tuning either
component.

    phase "bb_to_sc"   train A_BS on L_SC
    phase "sc_to_bb"   train A_SB on L_BB, packing detached
    phase "joint"      both, alternating one objective per step

**No data source is assumed.** The loop consumes an iterable of
:class:`CoupledBatch` supplied by the caller; there is no default dataset, path
or index anywhere in this module. What a batch must carry is stated once, in
:class:`CoupledBatch`, and validated on arrival.

Checkpoints hold the adapters, optimizer, EMA and step, plus the identity of the
frozen components, so a resumed run cannot silently pair adapters with different
donor weights than they were trained against.
"""

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pxf.couple import losses as couple_losses
from pxf.train.ema import EMA
from pxf.train.trainer import OptimSettings, learning_rate


@dataclass
class CoupledBatch:
    """One training example for the coupled cycle.

    Everything the cycle needs and nothing about where it came from:

    ``topology``      :class:`pxf.couple.controller.Topology` for this structure
    ``x_noisy``       noised coordinates on the backbone module's flat atom axis
    ``sigma``         backbone noise level, ``[B]`` or scalar
    ``aatype``        residue identities, ``[L]`` (X = 20 where undetermined)
    ``sidechain_batch`` the dict :mod:`pxf.train.step` needs for ``L_SC``
                      (keys of ``pxf.train.step.REQUIRED_KEYS``)
    ``backbone_target`` GT coordinates aligned with ``x_noisy``, for ``L_BB``
    ``backbone_atom_mask`` optional mask over that axis
    """

    topology: Any
    x_noisy: torch.Tensor
    sigma: torch.Tensor
    aatype: torch.Tensor
    sidechain_batch: dict | None = None
    backbone_target: torch.Tensor | None = None
    backbone_atom_mask: torch.Tensor | None = None
    name: str | None = None

    def require(self, kind):
        """Fail loudly when a batch cannot serve the objective being asked of it."""
        if kind == "sidechain" and self.sidechain_batch is None:
            raise ValueError(
                f"batch {self.name!r} carries no sidechain_batch, so "
                "L_SC cannot be computed for it"
            )
        if kind == "backbone" and self.backbone_target is None:
            raise ValueError(
                f"batch {self.name!r} carries no backbone_target, so "
                "L_BB cannot be computed for it"
            )


@dataclass
class CoupleSettings:
    """Loop settings specific to coupling."""

    phase: str = "bb_to_sc"
    max_steps: int = 10_000
    grad_accum_steps: int = 1
    pack_steps: int | None = None  # side-chain rollout length in the cycle
    sigma_data_backbone: float = 16.0
    log_every: int = 25
    checkpoint_every: int = 1_000
    ema_decay: float | None = None
    ema_relative_length: float | None = None
    seed: int = 0


class CoupledTrainer:
    """Trains the adapters around a frozen PXDesign and a frozen FaMPNN."""

    def __init__(
        self,
        controller,
        *,
        out_dir,
        optim=None,
        settings=None,
        device=None,
        frozen_identity=None,
    ):
        self.controller = controller
        self.adapters = controller.adapters
        self.settings = settings or CoupleSettings()
        self.optim_settings = optim or OptimSettings()
        self.out_dir = Path(out_dir)
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.frozen_identity = frozen_identity or {}

        self._assert_only_adapters_train()
        record = self.controller.set_phase(self.settings.phase)
        trainable = [p for p in self.adapters.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"Phase {self.settings.phase!r} leaves no adapter "
                "parameter trainable; there is nothing to optimize"
            )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.optim_settings.lr,
            betas=tuple(self.optim_settings.betas),
            eps=self.optim_settings.eps,
            weight_decay=self.optim_settings.weight_decay,
        )
        self.ema = None
        if self.settings.ema_decay or self.settings.ema_relative_length:
            self.ema = EMA(
                self.adapters,
                decay=self.settings.ema_decay,
                relative_length=self.settings.ema_relative_length,
            )
        self.step = 0
        self.phase_record = record
        torch.manual_seed(self.settings.seed)
        self.generator = torch.Generator().manual_seed(self.settings.seed)
        self._log_path = self.out_dir / "train_log.jsonl"

    def _assert_only_adapters_train(self):
        """The donors must be frozen, or a 'coupling' gain is really fine-tuning."""
        leaks = []
        for name, module in (("fampnn", self.controller.fampnn),):
            if module is None:
                continue
            leaks += [
                f"{name}.{n}" for n, p in module.named_parameters() if p.requires_grad
            ]
        if leaks:
            raise ValueError(
                f"{len(leaks)} frozen-component parameter(s) still require grad "
                f"(e.g. {leaks[:3]}). Freeze the donors, or any improvement cannot "
                "be attributed to the adapters."
            )

    # ---- one step --------------------------------------------------------

    def loss_for(self, batch, kind):
        """Run the cycle and compute the objective this step owns."""
        batch.require(kind)
        run_feedback = kind == "backbone"
        cycle = self.controller.forward(
            batch.topology,
            batch.x_noisy,
            batch.sigma,
            batch.aatype,
            run_feedback=run_feedback,
        )
        if kind == "sidechain":
            features = cycle.aux.get("features")
            if features is None:
                raise ValueError("the cycle recorded no encoder features for L_SC")
            return couple_losses.sidechain_coupling_loss(
                self.controller.fampnn,
                batch.sidechain_batch,
                features,
                delta_h=cycle.delta_h,
                generator=self.generator,
            ), cycle
        return couple_losses.backbone_feedback_loss(
            cycle,
            batch.backbone_target,
            sigma=batch.sigma,
            sigma_data=self.settings.sigma_data_backbone,
            atom_mask=batch.backbone_atom_mask,
        ), cycle

    # ---- persistence -----------------------------------------------------

    def checkpoint_state(self):
        return dict(
            adapters=self.adapters.state_dict(),
            step=self.step,
            optimizer=self.optimizer.state_dict(),
            ema=self.ema.state_dict() if self.ema else None,
            settings=asdict(self.settings),
            optim_settings=asdict(self.optim_settings),
            frozen=self.frozen_identity,
            controller=self.controller.identity(),
            generator=self.generator.get_state(),
        )

    def save(self, tag=None):
        name = f"step{self.step:08d}" if tag is None else tag
        path = self.out_dir / "checkpoints" / f"{name}.pt"
        torch.save(self.checkpoint_state(), path)
        return path

    def resume(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "adapters" not in state:
            raise ValueError(f"{path} is not a coupling checkpoint")
        frozen = state.get("frozen") or {}
        if self.frozen_identity and frozen and frozen != self.frozen_identity:
            raise ValueError(
                "This checkpoint's adapters were trained against different frozen "
                "components than the ones loaded now; pairing them would be "
                "meaningless. Load the matching donors or start fresh."
            )
        self.adapters.load_state_dict(state["adapters"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.step = int(state.get("step", 0))
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"].cpu())
        return self.step

    # ---- the loop --------------------------------------------------------

    def _log(self, record):
        with self._log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def train(self, batches: Iterable[CoupledBatch], *, max_steps=None, progress=print):
        """Consume ``batches`` until ``max_steps``. Yields nothing; logs and checkpoints."""
        target = int(max_steps or self.settings.max_steps)
        accum = max(1, int(self.settings.grad_accum_steps))
        started = time.time()
        pending = 0
        running = {"sidechain": [], "backbone": []}

        for batch in batches:
            if self.step >= target:
                break
            kind = couple_losses.loss_kind_for(self.settings.phase, self.step)
            loss, cycle = self.loss_for(batch, kind)
            (loss.total / accum).backward()
            running[kind].append(loss.scalars())
            pending += 1
            if pending < accum:
                continue

            lr = learning_rate(self.step, self.optim_settings, target)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.adapters.parameters() if p.requires_grad],
                    self.optim_settings.max_grad_norm,
                )
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema:
                self.ema.update(self.adapters)
            self.step += 1
            pending = 0

            if self.step % max(1, self.settings.log_every) == 0:
                record = dict(
                    step=self.step,
                    lr=lr,
                    grad_norm=grad_norm,
                    phase=self.settings.phase,
                    seconds=round(time.time() - started, 1),
                )
                # The two objectives live on different scales, so never average
                # them together -- report each over the steps that used it.
                for name, rows in running.items():
                    if rows:
                        record[f"{name}_loss"] = sum(r["loss"] for r in rows) / len(rows)
                        record[f"{name}_steps"] = len(rows)
                self._log(record)
                if progress:
                    parts = [f"step {self.step:>7d}", f"phase {self.settings.phase}"]
                    for name in ("sidechain", "backbone"):
                        if f"{name}_loss" in record:
                            parts.append(f"{name} {record[f'{name}_loss']:.4f}")
                    parts += [f"lr {lr:.2e}", f"|g| {grad_norm:.2f}"]
                    progress("  ".join(parts))
                running = {"sidechain": [], "backbone": []}

            if (
                self.settings.checkpoint_every
                and self.step % self.settings.checkpoint_every == 0
            ):
                self.save()

        final = self.save(tag="final")
        return dict(
            steps=self.step,
            checkpoint=str(final),
            seconds=round(time.time() - started, 1),
            phase=self.settings.phase,
        )
