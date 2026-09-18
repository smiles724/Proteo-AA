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
    ``backbone_atom_mask`` which atoms on that axis may be supervised. Required
                      by the SC->BB pilot: the monomer configuration happens to
                      emit a backbone-only axis, but only by configuration, and
                      the featurizer collapses design-region side chains onto
                      their CA rather than dropping them. See
                      :func:`pxf.couple.pilot.backbone_supervision_mask`.
    ``upstream``      a cached, detached :class:`pxf.couple.pilot.UpstreamState`,
                      so a pilot pays for the packing once per example rather
                      than once per step
    """

    topology: Any
    x_noisy: torch.Tensor
    sigma: torch.Tensor
    aatype: torch.Tensor
    sidechain_batch: dict | None = None
    backbone_target: torch.Tensor | None = None
    backbone_atom_mask: torch.Tensor | None = None
    upstream: Any = None
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
    # Use the explicit-boundary corrective-event path for backbone steps: the
    # frozen half under no_grad and cacheable, the correction outside it. The
    # older path runs the whole cycle in one call, which is equivalent here
    # because everything upstream is frozen, but says so nowhere.
    corrective_event: bool = False
    # The selected Phase-1 BB->SC policy, held FIXED across every SC->BB
    # comparison -- that is what makes the A_SB arms comparable to each other.
    # "bypass" runs no BB->SC residual at all, which the plan prescribes while
    # Phase 1 is unresolved and which does not prevent testing A_SB. "matched"
    # applies the loaded A_BS to this structure's own a_token. Recorded in every
    # checkpoint, because an A_SB trained under one policy is not deployable
    # under the other.
    bs_policy: str = "bypass"
    # Steps at which to score the held-out set. Step 0 is always included, so
    # "at initialization" is a measurement rather than an assumption.
    eval_steps: tuple = ()
    # Reduction for L_SC; see pxf.train.losses.SIDECHAIN_REDUCTIONS.
    sidechain_reduction: str = "per_residue"
    # The sigma_B distribution the batches were drawn from, as
    # CouplingNoiseSchedule.identity(). Recorded, not used: the schedule lives in
    # the batch generator, but a checkpoint that does not say which noise range
    # its adapter was trained on cannot be deployed correctly.
    sigma_schedule: dict | None = None
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
        fampnn_finetune=False,
        fampnn_model_cfg=None,
    ):
        self.controller = controller
        self.adapters = controller.adapters
        # The FaMPNN fine-tune CONTROL: train the donor's own weights on the
        # same data, for the same steps, with the adapters switched off. This is
        # deliberately not the default and not reachable by accident -- the
        # caller has to ask for it -- because a "coupling" number produced with
        # an unfrozen donor would be a fine-tune wearing the coupling's name.
        # See _assert_only_adapters_train.
        self.fampnn_finetune = bool(fampnn_finetune)
        self.tuned = controller.fampnn if self.fampnn_finetune else self.adapters
        self.fampnn_model_cfg = fampnn_model_cfg
        if self.fampnn_finetune and fampnn_model_cfg is None:
            raise ValueError(
                "fampnn_finetune needs fampnn_model_cfg: the checkpoint has to "
                "carry the config so --fampnn-checkpoint can rebuild SeqDenoiser"
            )
        self.settings = settings or CoupleSettings()
        self.optim_settings = optim or OptimSettings()
        self.out_dir = Path(out_dir)
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.frozen_identity = frozen_identity or {}

        record = self.controller.set_phase(self.settings.phase)
        if self.fampnn_finetune:
            # set_phase unfreezes this phase's adapter branch -- that is its job
            # -- so the control arm has to neutralise it afterwards, not before.
            # Freezing in the caller is silently undone by the line above.
            self.adapters.requires_grad_(False)
            self.adapters.enable_bb_to_sc = False
            self.adapters.enable_sc_to_bb = False
            self._assert_adapters_are_inert()
        else:
            self._assert_only_adapters_train()
        trainable = [p for p in self.tuned.parameters() if p.requires_grad]
        if not trainable:
            which = "FaMPNN" if self.fampnn_finetune else "adapter"
            raise ValueError(
                f"Phase {self.settings.phase!r} leaves no {which} "
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
                self.tuned,
                decay=self.settings.ema_decay,
                relative_length=self.settings.ema_relative_length,
            )
        self.step = 0
        self.phase_record = record
        from pxf.couple.controller import UNSET

        policies = {"bypass": None, "matched": UNSET}
        if self.settings.bs_policy not in policies:
            raise ValueError(
                f"unknown bs_policy {self.settings.bs_policy!r}; choose from "
                f"{sorted(policies)}"
            )
        self.bs_delta_h = policies[self.settings.bs_policy]
        # Held-out scoring, installed by the driver. Called at step 0 and at
        # every step in `settings.eval_steps`.
        self.validate_fn = None
        torch.manual_seed(self.settings.seed)
        self.generator = torch.Generator().manual_seed(self.settings.seed)
        self._log_path = self.out_dir / "train_log.jsonl"

    def _assert_adapters_are_inert(self):
        """In the control arm no adapter may train or be applied.

        The point of the control is to attribute the gain: if an adapter were
        live here, the run would measure coupling plus fine-tuning together and
        the comparison against the adapter run would be meaningless.
        """
        live = [n for n, p in self.adapters.named_parameters() if p.requires_grad]
        if live:
            raise ValueError(
                f"{len(live)} adapter parameter(s) still require grad (e.g. "
                f"{live[:3]}); the FaMPNN fine-tune control must train the donor "
                "only. Freeze the adapters."
            )
        applied = [
            name
            for name, flag in (
                ("bb_to_sc", getattr(self.adapters, "enable_bb_to_sc", False)),
                ("sc_to_bb", getattr(self.adapters, "enable_sc_to_bb", False)),
            )
            if flag
        ]
        if applied:
            raise ValueError(
                f"adapter branch(es) {applied} are still applied; the control "
                "must run no adapter at all (set enable_* False)"
            )

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
        if run_feedback and self.settings.corrective_event:
            cycle = self.controller.corrective_event(
                batch.topology,
                batch.x_noisy,
                batch.sigma,
                batch.aatype,
                bs_delta_h=self.bs_delta_h,
                upstream=batch.upstream,
            )
        else:
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
                reduction=self.settings.sidechain_reduction,
            ), cycle
        if batch.backbone_atom_mask is None and self.settings.corrective_event:
            raise ValueError(
                f"batch {batch.name!r} carries no backbone_atom_mask, so the "
                "supervised atom set is whatever the featurizer happened to put "
                "on the flat axis. Under the monomer configuration that is "
                "backbone-only and harmless, but the featurizer collapses "
                "design-region side chains onto their CA rather than dropping "
                "them, so any other configuration would supervise those. Build "
                "the mask with pxf.couple.pilot.backbone_supervision_mask."
            )
        loss = couple_losses.backbone_feedback_loss(
            cycle,
            batch.backbone_target,
            sigma=batch.sigma,
            sigma_data=self.settings.sigma_data_backbone,
            atom_mask=batch.backbone_atom_mask,
        )
        loss.stats.update(
            {
                f"fb_{k}": v
                for k, v in cycle.feedback_stats.items()
                if not isinstance(v, list)
            }
        )
        return loss, cycle

    # ---- persistence -----------------------------------------------------

    def checkpoint_state(self):
        extra = {}
        if self.fampnn_finetune:
            # Saved in the shape --fampnn-checkpoint expects (model_cfg +
            # state_dict), so every existing evaluator can load the control
            # without special-casing it.
            # The adapter arm is evaluated on EMA weights, so the control has
            # to be too or the comparison is not like-for-like. `state_dict` is
            # therefore the EMA view (what --fampnn-checkpoint will load) and
            # the raw weights are kept alongside it.
            raw = {
                k: v.detach().cpu().clone()
                for k, v in self.controller.fampnn.state_dict().items()
            }
            if self.ema is not None:
                with self.ema.swapped_into(self.controller.fampnn):
                    tuned = {
                        k: v.detach().cpu().clone()
                        for k, v in self.controller.fampnn.state_dict().items()
                    }
            else:
                tuned = raw
            extra = dict(
                fampnn_finetune=True,
                fampnn_state_is_ema=self.ema is not None,
                model_cfg=self.fampnn_model_cfg,
                state_dict=tuned,
                fampnn_state_raw=raw,
            )
        return dict(
            **extra,
            adapters=self.adapters.state_dict(),
            step=self.step,
            optimizer=self.optimizer.state_dict(),
            ema=self.ema.state_dict() if self.ema else None,
            settings=asdict(self.settings),
            optim_settings=asdict(self.optim_settings),
            initialized_from=getattr(self, "initialized_from", None),
            frozen=self.frozen_identity,
            controller=self.controller.identity(),
            generator=self.generator.get_state(),
        )

    def save(self, tag=None):
        name = f"step{self.step:08d}" if tag is None else tag
        path = self.out_dir / "checkpoints" / f"{name}.pt"
        torch.save(self.checkpoint_state(), path)
        return path

    # Which adapter direction each phase trains, i.e. which one a later phase
    # should inherit. Keyed by the phase that WROTE the checkpoint.
    PHASE_TRAINS = {
        "bb_to_sc": ("bb_to_sc.",),
        "sc_to_bb": ("sc_to_bb.",),
        "joint": ("bb_to_sc.", "sc_to_bb."),
        "frozen": (),
    }

    def initialize_from(self, path, *, prefix=None):
        """Start this phase from a previous phase's *weights*. Nothing else.

        Distinct from :meth:`resume`, which continues one experiment and
        restores the step counter, the optimizer moments, the EMA and the RNG.
        None of those transfer across phases, and the reasons are separate
        failures rather than one:

        * the step counter makes the new phase a no-op whenever the two phases
          share a ``max_steps`` -- the loop breaks on its first batch and still
          writes a "final" checkpoint;
        * the optimizer moments load into an optimizer that owns *different*
          parameters, and because both directions are symmetric six-parameter
          adapters ``load_state_dict`` matches them by position and succeeds
          silently, so the previous direction's AdamW state drives the new one.

        So this resets all of it: step to zero, a fresh optimizer over whatever
        ``set_phase`` left trainable, a fresh EMA, and a reseeded generator.
        ``prefix`` defaults to whichever direction the *source* checkpoint's
        phase trained.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "adapters" not in state:
            raise ValueError(f"{path} is not a coupling checkpoint")
        source_phase = (state.get("settings") or {}).get("phase")
        if prefix is None:
            prefixes = self.PHASE_TRAINS.get(source_phase)
            if not prefixes:
                raise ValueError(
                    f"{path} was written by phase {source_phase!r}, which trains "
                    "no adapter direction, so there is nothing in it to inherit. "
                    "Pass prefix= explicitly if you meant something else."
                )
        else:
            prefixes = (prefix,)

        source = state["adapters"]
        if state.get("ema"):
            # The previous phase's own evaluation used its EMA weights, so the
            # policy being held fixed is the EMA one; loading the raw weights
            # would hold a different policy fixed than the one selected.
            shadow = state["ema"].get("shadow") or {}
            source = {**source, **{k: v for k, v in shadow.items() if k in source}}
        wanted = {k: v for k, v in source.items() if k.startswith(tuple(prefixes))}
        if not wanted:
            raise ValueError(
                f"{path} carries no {list(prefixes)} parameters, so there is no "
                f"phase {source_phase!r} policy in it to hold fixed"
            )
        missing, unexpected = self.adapters.load_state_dict(wanted, strict=False)
        unexpected = [k for k in unexpected if k.startswith(tuple(prefixes))]
        if unexpected:
            raise ValueError(
                f"{path} has parameters the current adapters do not: {unexpected[:5]}"
            )
        still_missing = [k for k in missing if k.startswith(tuple(prefixes))]
        if still_missing:
            raise ValueError(f"{path} is missing parameters: {still_missing[:5]}")

        # Everything that belonged to the previous phase's optimization is
        # discarded, not carried. Rebuilt rather than assumed fresh: relying on
        # this only ever being called on a new trainer is what made the claim
        # true by luck instead of by construction.
        self.step = 0
        trainable = [p for p in self.tuned.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.optim_settings.lr,
            betas=tuple(self.optim_settings.betas),
            eps=self.optim_settings.eps,
            weight_decay=self.optim_settings.weight_decay,
        )
        if self.settings.ema_decay or self.settings.ema_relative_length:
            self.ema = EMA(
                self.tuned,
                decay=self.settings.ema_decay,
                relative_length=self.settings.ema_relative_length,
            )
        self.generator = torch.Generator().manual_seed(self.settings.seed)

        record = dict(
            source=str(path),
            source_phase=source_phase,
            source_step=int(state.get("step", 0)),
            source_is_ema=bool(state.get("ema")),
            prefixes=list(prefixes),
            loaded=len(wanted),
            step=self.step,
            optimizer_reset=True,
            ema_reset=self.ema is not None,
            trainable_parameters=sum(p.numel() for p in trainable),
        )
        self.initialized_from = record
        return record

    def resume(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "adapters" not in state:
            raise ValueError(f"{path} is not a coupling checkpoint")
        source_phase = (state.get("settings") or {}).get("phase")
        if source_phase is not None and source_phase != self.settings.phase:
            # Resuming across phases restores the *previous* phase's optimizer
            # into an optimizer that owns different parameters. Both directions
            # are symmetric six-parameter adapters, so `load_state_dict` matches
            # them by position and succeeds without complaint: A_BS's AdamW
            # moments end up driving A_SB. Measured -- identical exp_avg sums
            # before and after. Together with the step counter this makes the
            # documented phase chain a no-op that reports success.
            raise ValueError(
                f"{path} was written by phase {source_phase!r} but this run is "
                f"phase {self.settings.phase!r}. --resume continues one "
                "experiment: it restores the step counter, the optimizer moments "
                "and the EMA, none of which transfer across phases (the two "
                "directions have the same parameter shapes, so the moments load "
                "silently into the wrong adapter). To start this phase from that "
                "one's weights, use --init-from."
            )
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

    def _maybe_validate(self, wanted, progress):
        """Score the held-out set if this step is one of the evaluation points.

        Evaluating at initialization as well as at the end is what makes the
        pilot's claim falsifiable: a zero-initialized A_SB must reproduce the
        uncorrected proposal exactly, so step 0 is the equality check and every
        later point is measured against it rather than against a remembered
        number.
        """
        if self.validate_fn is None or self.step not in wanted:
            return None
        record = self.validate_fn(self.step)
        if record is None:
            return None
        record = dict(eval=True, step=self.step, **record)
        self._log(record)
        if progress:
            summary = "  ".join(
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in record.items()
                if k not in ("eval", "step")
            )
            progress(f"[eval step {self.step:>7d}]  {summary}")
        return record

    def train(self, batches: Iterable[CoupledBatch], *, max_steps=None, progress=print):
        """Consume ``batches`` until ``max_steps``. Yields nothing; logs and checkpoints."""
        target = int(max_steps or self.settings.max_steps)
        if self.step >= target:
            # The failure this guards is silent and looks like success: the loop
            # below breaks on its first batch, `save(tag="final")` still writes a
            # checkpoint, and the returned record still says `steps: <target>`.
            # The only outward sign is an empty train_log.jsonl. It is reachable
            # whenever a checkpoint at or past `max_steps` is loaded -- which is
            # exactly what `--resume <previous phase's final.pt>` does when both
            # phases share a step budget.
            raise ValueError(
                f"the step counter is already at {self.step} and max_steps is "
                f"{target}, so this run would perform zero optimizer updates "
                "and still write a 'final' checkpoint. If you meant to continue "
                "this experiment, raise max_steps above the loaded step; if you "
                "meant to start a new phase from a previous one's weights, use "
                "initialize_from() (--init-from), which resets the step "
                "counter, the optimizer and the EMA."
            )
        accum = max(1, int(self.settings.grad_accum_steps))
        started = time.time()
        pending = 0
        running = {"sidechain": [], "backbone": []}
        wanted_evals = sorted({0, *(int(s) for s in self.settings.eval_steps)})
        self._maybe_validate(wanted_evals, progress)

        for batch in batches:
            if self.step >= target:
                break
            kind = couple_losses.loss_kind_for(self.settings.phase, self.step)
            loss, cycle = self.loss_for(batch, kind)
            (loss.total / accum).backward()
            scalars = loss.scalars()
            # sigma_B now varies per batch, so the log has to say which noise
            # levels a window actually covered; a mean alone would hide a
            # collapsed range.
            scalars["sigma_b"] = float(torch.as_tensor(batch.sigma).float().mean())
            running[kind].append(scalars)
            pending += 1
            if pending < accum:
                continue

            lr = learning_rate(self.step, self.optim_settings, target)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.tuned.parameters() if p.requires_grad],
                    self.optim_settings.max_grad_norm,
                )
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema:
                self.ema.update(self.tuned)
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
                        sigmas = [r["sigma_b"] for r in rows]
                        record[f"{name}_loss"] = sum(r["loss"] for r in rows) / len(rows)
                        record[f"{name}_steps"] = len(rows)
                        record[f"{name}_sigma_b_mean"] = sum(sigmas) / len(sigmas)
                        record[f"{name}_sigma_b_range"] = [min(sigmas), max(sigmas)]
                        # The feedback diagnostics the readout reports -- how big
                        # the residual is, absolutely and relative to the token
                        # features it is added to. Computed per step all along
                        # but previously dropped before the log, so the only
                        # place to see them was the held-out evaluation.
                        for key in sorted(
                            {k for r in rows for k in r if k.startswith("fb_")}
                        ):
                            values = [r[key] for r in rows if key in r]
                            record[key] = sum(values) / len(values)
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
            self._maybe_validate(wanted_evals, progress)

        final = self.save(tag="final")
        return dict(
            steps=self.step,
            checkpoint=str(final),
            seconds=round(time.time() - started, 1),
            phase=self.settings.phase,
        )
