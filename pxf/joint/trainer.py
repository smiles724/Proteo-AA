"""The training loop for side-chain-supervised backbone fine-tuning.

The experiment is a comparison, not a single run, so the loop's job is mostly
to make the arms differ in exactly one thing. Everything else -- the examples,
their order, the backbone noise, the side-chain noise, the initialization, the
trainable set -- is shared by construction:

    R0  nothing trainable                  the donor, unchanged
    B0  L_BB                               the matched baseline
    B1  L_BB + l_local L_local             encoder-mediated supervision
    B2  L_BB + l_local L_local + l_place L_place    the candidate
    BF  L_BB + l_place L_frame             geometric supervision, no FaMPNN prediction
    BS  L_BB, side chains computed with the backbone detached   the negative control

BS exists to answer "did the side-chain branch do anything, or did merely
*running* it change the result?" It does the same work as B1 and throws the
gradient away, so it must take bit-identical optimizer steps to B0 under a
replayed draw. :func:`arm_spec` is the single place the arms are defined;
adding one anywhere else is how two arms quietly stop being comparable.

Three things this loop refuses to do quietly:

**Train a different parameter set than it says.** The allowlist is resolved
from the loaded donor by name, and the optimizer's membership is asserted
against it. ``requires_grad=True`` does not put a parameter in an optimizer and
being in an optimizer does not mean it receives gradient; both are checked.

**Continue past a non-finite loss.** :func:`pxf.joint.losses.combined_loss`
raises, and the step is reported with the sample and noise level that produced
it. The generic parity loss zeroes a NaN term instead, which is right for an
unattended run and wrong for a controlled comparison -- it would turn the
candidate arm into the baseline for that step.

**Clip to zero.** ``max_grad_norm = 0`` means "do not clip"; passed literally to
``clip_grad_norm_`` it scales every gradient to zero, which looks like a run
that trains nothing for no visible reason.
"""

import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pxf.joint import data as joint_data
from pxf.joint import losses as joint_losses
from pxf.joint import model as joint_model
from pxf.joint import randomness as joint_random
from pxf.train.ema import EMA

TRAINABLE_BLOCKS = 4

# What each arm optimizes, and whether the side-chain branch may reach the
# backbone at all. (local, place, frame, detach_backbone, runs_sidechain)
ARMS = {
    "R0": (False, False, False, False, False),
    "B0": (False, False, False, False, False),
    "B1": (True, False, False, False, True),
    "B2": (True, True, False, False, True),
    "BF": (False, False, True, False, True),
    "BS": (True, False, False, True, True),
}


def arm_spec(arm):
    """``(local, place, frame, detach_backbone, runs_sidechain)`` for one arm."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return ARMS[arm]


@dataclass
class JointSettings:
    """One arm's configuration. Recorded whole in every checkpoint."""

    arm: str = "B0"
    max_steps: int = 2_000
    grad_accum_steps: int = 8
    # The auxiliary coefficients, from scripts/preflight_joint.py. Left at 0.0
    # so an unconfigured run is a loud baseline rather than a quiet candidate.
    lambda_local: float = 0.0
    lambda_place: float = 0.0
    warmup_auxiliary_steps: int = 200
    # Side-chain clones per example, and the self-conditioning probability.
    multiplier: int = 8
    self_cond_p: float = 0.0
    # The backbone noise window the refinement is trained in.
    sigma_min: float = 0.1
    sigma_max: float = 2.0
    sigma_mode: str = "trajectory"
    trainable_blocks: int = TRAINABLE_BLOCKS
    lr: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    ema_relative_length: float | None = 0.25
    seed: int = 0
    # This stage freezes the side-chain model entirely. The later joint arms
    # (J0/J1) train its denoiser, which needs a second optimizer group that
    # does not exist yet -- so False is refused rather than quietly training
    # nothing, and the flag marks where that work goes.
    freeze_sidechain: bool = True
    log_every: int = 25
    checkpoint_every: int = 500
    amp_dtype: str | None = None
    cache_size: int = 32

    def __post_init__(self):
        arm_spec(self.arm)
        self.betas = tuple(self.betas)
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

    def auxiliary_scale(self, step):
        """Linear ramp of the auxiliary coefficients over the first steps."""
        if self.warmup_auxiliary_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / float(self.warmup_auxiliary_steps))


def select_trainable(backbone, *, n_blocks=TRAINABLE_BLOCKS):
    """The plan's initial scope, resolved against the loaded donor.

    Raises rather than silently training a different set: "the last four
    blocks" only means something if there are four, and a renamed module would
    otherwise produce a run that trains nothing and reports a falling loss
    because ``L_BB`` still moves with the frozen donor's own predictions.
    """
    backbone.requires_grad_(False)
    blocks = backbone.diffusion_module.diffusion_transformer.blocks
    if len(blocks) < n_blocks:
        raise ValueError(
            f"the donor has {len(blocks)} diffusion blocks, fewer than the "
            f"{n_blocks} the trainable scope names"
        )
    tail = {
        f"diffusion_transformer.blocks.{i}."
        for i in range(len(blocks) - n_blocks, len(blocks))
    }
    trainable = OrderedDict()
    for name, parameter in backbone.named_parameters():
        if (
            any(key in name for key in tail)
            or "diffusion_module.layernorm_a" in name
            or "atom_attention_decoder" in name
        ):
            parameter.requires_grad_(True)
            trainable[name] = parameter
    if not trainable:
        raise ValueError("none of the named backbone modules exist in this donor")
    return trainable


class ExampleCache:
    """Featurized structures and their frozen conditioning, by sample id.

    Only genuinely static things live here. The featurization and the joint
    batch are functions of the file; the conditioning is a function of the
    input features *and the design-condition embedder*, which this experiment
    freezes. Nothing derived from the current weights -- B_hat, h_V, the
    side-chain prediction, the predicted frames -- is ever cached, because all
    of them change every update.
    """

    def __init__(self, capacity=32):
        self.capacity = int(capacity)
        self._items = OrderedDict()
        self.hits = self.misses = 0

    def get(self, sample_id):
        if sample_id in self._items:
            self._items.move_to_end(sample_id)
            self.hits += 1
            return self._items[sample_id]
        self.misses += 1
        return None

    def put(self, sample_id, value):
        self._items[sample_id] = value
        self._items.move_to_end(sample_id)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, held=len(self._items))


class JointTrainer:
    """Drives one arm over a stream of structures."""

    def __init__(
        self,
        driver,
        fampnn,
        source,
        *,
        out_dir,
        settings=None,
        donor_record=None,
        device=None,
    ):
        self.driver = driver
        self.backbone = driver.model
        self.fampnn = fampnn
        self.source = source
        self.settings = settings or JointSettings()
        self.donor_record = donor_record or {}
        self.out_dir = Path(out_dir)
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device) if device else next(
            self.backbone.parameters()
        ).device

        # FaMPNN is frozen for every arm in this stage. Frozen by requires_grad,
        # not by no_grad: the input derivatives are the mechanism under test.
        # Checked before freezing, so a caller that meant to train the
        # side-chain model finds out rather than having it silently disabled.
        live = [n for n, p in self.fampnn.named_parameters() if p.requires_grad]
        if live and not self.settings.freeze_sidechain:
            raise ValueError(
                f"{len(live)} FaMPNN parameters require grad and "
                "freeze_sidechain=False, but this trainer has no optimizer group "
                "for them -- they would accumulate gradient and never update. "
                "The joint arms that train the side-chain denoiser (J0/J1) are "
                f"not implemented yet (e.g. {live[:3]})"
            )
        self.froze_sidechain = len(live)
        self.fampnn.requires_grad_(False)
        self.fampnn.eval()
        # Deterministic dropout in the donor too, with the selected parameters
        # still receiving gradient. eval() does not disable autograd; this is a
        # fine-tuning choice, recorded rather than implied.
        self.backbone.eval()

        self.trainable = select_trainable(
            self.backbone, n_blocks=self.settings.trainable_blocks
        )
        self.optimizer = torch.optim.AdamW(
            list(self.trainable.values()),
            lr=self.settings.lr,
            betas=self.settings.betas,
            eps=self.settings.eps,
            weight_decay=self.settings.weight_decay,
        )
        self._assert_optimizer_matches_allowlist()

        self.ema = (
            EMA(self.backbone, relative_length=self.settings.ema_relative_length)
            if self.settings.ema_relative_length
            else None
        )
        self.cache = ExampleCache(self.settings.cache_size)
        self.step = 0
        self.examples_seen = 0
        self._log_path = self.out_dir / "train_log.jsonl"
        self._amp = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.settings.amp_dtype
        ]
        torch.manual_seed(self.settings.seed)

    # ---- contracts -------------------------------------------------------

    def _assert_optimizer_matches_allowlist(self):
        """The optimizer trains the allowlist, and nothing else trains at all."""
        in_optimizer = {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }
        allowed = {id(p) for p in self.trainable.values()}
        if in_optimizer != allowed:
            raise ValueError(
                f"the optimizer holds {len(in_optimizer)} tensors but the allowlist "
                f"names {len(allowed)}"
            )
        stray = [
            name
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and name not in self.trainable
        ]
        if stray:
            raise ValueError(f"{len(stray)} backbone parameters are live but unlisted: {stray[:4]}")
        leaking = [n for n, p in self.fampnn.named_parameters() if p.requires_grad]
        if leaking:  # unreachable after the freeze above; a guard, not a filter
            raise ValueError(
                f"{len(leaking)} FaMPNN parameters require grad; this stage freezes "
                f"the side-chain model entirely (e.g. {leaking[:3]})"
            )

    def identity(self):
        """What the run is, for the record written beside every checkpoint."""
        from pxf import provenance

        return dict(
            arm=self.settings.arm,
            settings=asdict(self.settings),
            trainable_parameters=int(
                sum(p.numel() for p in self.trainable.values())
            ),
            trainable_names=list(self.trainable),
            donor=self.donor_record,
            driver=self.driver.identity(),
            augment_eps=float(
                self.fampnn.denoiser.seq_design_module.features.augment_eps
            ),
            # How many side-chain parameters this run had to freeze on entry.
            # Nonzero means the caller handed over a trainable FaMPNN.
            froze_sidechain_parameters=int(self.froze_sidechain),
            randomness=joint_random.identity(base=self.settings.seed),
            upstream=provenance.runtime_sources(components=("fampnn",)),
        )

    # ---- one example -----------------------------------------------------

    def prepare(self, entry):
        """Featurize, align and cache one structure. Returns ``(batch, conditioning)``."""
        sample_id = entry["sample_id"]
        cached = self.cache.get(sample_id)
        if cached is not None:
            return cached
        from fampnn.data.data import load_feats_from_pdb, process_single_pdb

        from pxf.backbone.driver import featurize_structures, to_featurized

        _id, source = featurize_structures([entry["path"]], crop_size=entry.get("crop_size", 1024))[0]
        structure = to_featurized(sample_id, source[0]).to(self.device)
        native = process_single_pdb(load_feats_from_pdb(entry["path"]))
        batch = joint_data.build_joint_batch(
            self.fampnn, structure, native, split=entry.get("split")
        )
        conditioning = self.driver.conditioning(structure.feature_dict)
        self.cache.put(sample_id, (batch, conditioning))
        return batch, conditioning

    def forward_losses(self, entry, *, occurrence):
        """One arm's objective on one example, with its named noise draw."""
        from pxf.couple.losses import backbone_denoising_loss
        from pxf.couple.schedule import CouplingNoiseSchedule
        from pxf.train import losses as loss_fns

        use_local, use_place, use_frame, detach, runs_sidechain = arm_spec(
            self.settings.arm
        )
        batch, conditioning = self.prepare(entry)
        sample_id = entry["sample_id"]

        schedule = CouplingNoiseSchedule(
            mode=self.settings.sigma_mode,
            sigma_min=self.settings.sigma_min,
            sigma_max=self.settings.sigma_max,
            sigma_data=self.driver.sigma_data,
        )
        sigma = float(
            schedule.sample(
                1,
                generator=joint_random.generator_for(
                    self.settings.seed, sample_id, "backbone_noise",
                    occurrence=occurrence,
                ),
            )
        )
        backbone_noise = joint_random.draw_backbone_noise(
            batch.backbone_target.shape,
            joint_random.generator_for(
                self.settings.seed, sample_id, "backbone_noise",
                occurrence=occurrence, sigma=sigma,
            ),
        )
        sidechain_noise = None
        if runs_sidechain:
            from fampnn.data import residue_constants as rc

            sidechain_noise = joint_random.sidechain_noise_for(
                self.fampnn.denoiser.scn_diffusion_module.scn_interpolant,
                (
                    self.settings.multiplier,
                    batch.length,
                    len(rc.non_bb_idxs),
                    3,
                ),
                base=self.settings.seed,
                sample_id=sample_id,
                occurrence=occurrence,
                sigma=sigma,
                device=self.device,
            )

        forward = joint_model.joint_forward(
            self.driver,
            conditioning,
            self.fampnn,
            batch,
            sigma_b=sigma,
            backbone_noise=backbone_noise,
            sidechain_noise=sidechain_noise,
            multiplier=self.settings.multiplier if runs_sidechain else 1,
            self_cond_p=self.settings.self_cond_p,
            detach_backbone=detach,
            run_sidechain=runs_sidechain,
        )

        anchor = backbone_denoising_loss(
            forward.bb_pred,
            forward.bb_target,
            sigma=forward.sigma_b,
            sigma_data=self.driver.sigma_data,
            atom_mask=forward.bb_mask,
        ).total

        local = placement = None
        stats = dict(sigma_b=sigma, length=batch.length, **batch.counts)
        if use_local:
            local, local_stats = loss_fns.sidechain_diffusion_loss(
                forward.prediction.q_pred,
                forward.prediction.q_target,
                forward.prediction.weight,
                forward.prediction.loss_mask,
            )
            stats["sidechain_mse_local"] = float(local_stats["sidechain_mse_local"])
        if use_place or use_frame:
            if use_frame:
                placement, place_stats = joint_losses.frame_only_placement(forward, batch)
            else:
                from fampnn.data import residue_constants as rc

                native_scn = forward.prediction.clone(
                    batch.native_batch["x"][..., rc.non_bb_idxs, :]
                )
                placement, place_stats = joint_losses.placement_loss(
                    forward.placed,
                    native_scn,
                    forward.prediction.clone(batch.physical_mask),
                    aatype=forward.prediction.aatype,
                )
            stats["placement_rmsd"] = float(place_stats["placement_rmsd"])

        scale = self.settings.auxiliary_scale(self.step)
        combined = joint_losses.combined_loss(
            anchor,
            local=local,
            placement=placement,
            lambda_local=self.settings.lambda_local * scale,
            lambda_place=self.settings.lambda_place * scale,
            stats=stats,
            sample_id=f"{sample_id} at sigma_B={sigma:.4g}",
        )
        return combined, forward

    # ---- the loop --------------------------------------------------------

    def _log(self, record):
        with self._log_path.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    def train(self, *, max_steps=None, progress=print):
        target = int(max_steps or self.settings.max_steps)
        accum = max(1, self.settings.grad_accum_steps)
        if self.settings.arm == "R0":
            raise ValueError(
                "R0 is the untrained donor reference; it has no training run. "
                "Evaluate the donor directly instead."
            )
        started = time.time()
        pending, running = 0, {}
        self.optimizer.zero_grad(set_to_none=True)

        for entry in self.source:
            if self.step >= target:
                break
            context = (
                torch.autocast(device_type=self.device.type, dtype=self._amp)
                if self._amp
                else torch.enable_grad()
            )
            with context:
                combined, _forward = self.forward_losses(
                    entry, occurrence=self.examples_seen
                )
            (combined.total / accum).backward()
            self.examples_seen += 1
            pending += 1
            for key, value in combined.scalars().items():
                if isinstance(value, (int, float)):
                    running[key] = running.get(key, 0.0) + value
            if pending < accum:
                continue

            lr = self._learning_rate()
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            # 0 means "do not clip", not "clip to zero".
            limit = self.settings.max_grad_norm or float("inf")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(list(self.trainable.values()), limit)
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema:
                self.ema.update(self.backbone)
            self.step += 1

            if self.step % max(1, self.settings.log_every) == 0:
                record = {k: v / pending for k, v in running.items()}
                record.update(
                    step=self.step,
                    arm=self.settings.arm,
                    lr=lr,
                    grad_norm=grad_norm,
                    examples=self.examples_seen,
                    seconds=round(time.time() - started, 1),
                    cache=self.cache.stats(),
                )
                self._log(record)
                if progress:
                    progress(
                        f"[{self.settings.arm}] step {self.step:>6d}  "
                        f"loss {record['loss']:.4f}  bb {record['loss_bb']:.4f}  "
                        + (f"local {record.get('loss_local', float('nan')):.4f}  ")
                        + (f"place {record.get('loss_place', float('nan')):.4f}  ")
                        + f"|g| {grad_norm:.2f}  lr {lr:.2e}"
                    )
            pending, running = 0, {}
            if (
                self.settings.checkpoint_every
                and self.step % self.settings.checkpoint_every == 0
            ):
                self.save()

        final = self.save(tag="final")
        return dict(
            arm=self.settings.arm,
            steps=self.step,
            examples=self.examples_seen,
            checkpoint=str(final),
            seconds=round(time.time() - started, 1),
        )

    def _learning_rate(self):
        if self.settings.warmup_steps and self.step < self.settings.warmup_steps:
            return self.settings.lr * (self.step + 1) / self.settings.warmup_steps
        return self.settings.lr

    # ---- persistence -----------------------------------------------------

    def state(self):
        return dict(
            step=self.step,
            examples_seen=self.examples_seen,
            arm=self.settings.arm,
            settings=asdict(self.settings),
            # Only the trainable slice: the rest is the donor, pinned by digest.
            trainable_state={
                name: parameter.detach().cpu().clone()
                for name, parameter in self.trainable.items()
            },
            optimizer=self.optimizer.state_dict(),
            ema=self.ema.state_dict() if self.ema else None,
            identity=self.identity(),
            rng=dict(
                torch=torch.get_rng_state(),
                cuda=(
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            ),
        )

    def save(self, tag=None):
        name = f"step{self.step:08d}" if tag is None else tag
        path = self.out_dir / "checkpoints" / f"{name}.pt"
        torch.save(self.state(), path)
        return path

    def resume(self, path):
        """Restore the trainable slice, the optimizer, EMA, counters and RNG."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["arm"] != self.settings.arm:
            raise ValueError(
                f"{path} is arm {state['arm']!r} and this run is {self.settings.arm!r}; "
                "resuming across arms would silently mix two experiments"
            )
        missing = set(self.trainable) - set(state["trainable_state"])
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} trainable tensors, so the scope "
                f"differs from this run's (e.g. {sorted(missing)[:3]})"
            )
        with torch.no_grad():
            for name, parameter in self.trainable.items():
                parameter.copy_(state["trainable_state"][name].to(parameter.device))
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.step = int(state["step"])
        self.examples_seen = int(state["examples_seen"])
        rng = state.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        return self.step
