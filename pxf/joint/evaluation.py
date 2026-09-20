"""Reconstruct a trained joint-refinement model and run held-out predictions.

Nothing here trains and nothing here scores. The module answers the two
questions an evaluator must not get wrong on its own:

*Which weights is this checkpoint, exactly?* A checkpoint stores only the
trainable slice, as absolute values, over a donor pinned by digest -- so the
model is the donor with those keys overwritten, never the donor plus a delta,
and never a partially loaded donor that would still emit plausible
coordinates. The EMA payload is a different object again: it shadows *every*
floating parameter and buffer, so restoring it is a whole-model operation.

*What did the model predict for an event every arm saw identically?* The noisy
backbone is drawn once per event from a replayable stream and handed to every
checkpoint unchanged; each produces its own ``B_hat``. Arm identity decides how
a number is interpreted afterwards, never which forward path runs. At inference
every current arm is one backbone denoise with the side-chain branch off.
"""

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
from pxf.joint import model as joint_model
from pxf.joint.trainer import ARMS, TRAINABLE_BLOCKS, select_trainable

# Evaluation draws live in their own namespace. They are deliberately NOT
# registered in pxf.joint.randomness.STREAMS: that tuple is read by running
# training jobs, and an evaluator has no business editing a file a training run
# may be importing. The prefix below cannot collide with a training seed
# because the stream component of the hash differs, which is the same property
# STREAMS relies on.
EVAL_NAMESPACE = "eval/v1"
WEIGHT_CHOICES = ("raw", "ema")
DONOR_LABEL = "R0"


class EvaluationError(RuntimeError):
    """A checkpoint, donor or event that cannot be evaluated as asked."""


# ---- event identity ---------------------------------------------------------


def sigma_key(value):
    """Canonical text for a sigma, so a key survives a round trip through JSON.

    Keyed on the value rather than its position in the sweep: two runs that
    request the same sigmas in a different order must produce the same keys,
    and a sweep that is later extended must not renumber the events already
    scored.
    """
    return f"{float(value):.6g}"


def check_distinct_sigmas(sigmas):
    """Refuse two requested sigmas that quantize onto one key."""
    seen = {}
    for value in sigmas:
        key = sigma_key(value)
        if key in seen:
            raise EvaluationError(
                f"sigmas {seen[key]!r} and {value!r} both quantize to {key!r}, so "
                "their events would collide and one would silently overwrite the "
                "other; separate them or drop one"
            )
        seen[key] = value
    return [sigma_key(v) for v in sigmas]


def event_key(panel_id, sample_id, sigma, replicate):
    """The identity of one backbone event, shared by every model."""
    return (str(panel_id), str(sample_id), sigma_key(sigma), int(replicate))


def eval_seed(base, panel_id, sample_id, sigma, replicate, purpose):
    """A stable 63-bit seed for one evaluation draw.

    Hashed over the whole event key, so adding a model, reordering sigmas or
    sharding the run cannot move an existing event's noise. ``purpose`` keeps
    the backbone draw and any later packing draw independent.
    """
    parts = [
        EVAL_NAMESPACE,
        str(int(base)),
        str(panel_id),
        str(sample_id),
        sigma_key(sigma),
        str(int(replicate)),
        str(purpose),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def eval_generator(base, panel_id, sample_id, sigma, replicate, purpose):
    """A CPU generator for one evaluation draw; CPU so a seed is device-free."""
    return torch.Generator().manual_seed(
        eval_seed(base, panel_id, sample_id, sigma, replicate, purpose)
    )


def replay_backbone_noise(shape, base, panel_id, sample_id, sigma, replicate):
    """Unit backbone noise for one event, drawn on the CPU.

    Drawn once and handed to every model. The corruption itself is applied by
    :func:`pxf.joint.model.joint_forward`, so the evaluator cannot drift from
    the training rule by reimplementing it.
    """
    from pxf.joint import randomness as joint_random

    generator = eval_generator(
        base, panel_id, sample_id, sigma, replicate, "backbone_noise"
    )
    return joint_random.draw_backbone_noise(shape, generator)


def noise_digest(tensor):
    """Content digest of a drawn noise tensor, for the row's provenance."""
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


# ---- checkpoint reconstruction ----------------------------------------------


@dataclass
class LoadedModel:
    """One reconstructed model, with everything a row must record about it."""

    label: str
    arm: str
    weights: str
    model: object
    driver: object
    step: int
    examples_seen: int
    checkpoint_path: str
    checkpoint_sha256: str
    donor_sha256: str
    trainable_names: list = field(default_factory=list)
    trainable_parameters: int = 0
    ema_step: int = None
    identity: dict = field(default_factory=dict)
    is_donor: bool = False

    def content_identity(self):
        """What makes two rows comparable: the weights, not the file path."""
        return dict(
            label=self.label,
            arm=self.arm,
            weights=self.weights,
            step=self.step,
            checkpoint_sha256=self.checkpoint_sha256,
            donor_sha256=self.donor_sha256,
            ema_step=self.ema_step,
            is_donor=self.is_donor,
        )


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise EvaluationError(message)


def _checkpoint_arm(state):
    """The arm this checkpoint claims, with its three records agreeing."""
    top = state.get("arm")
    settings = (state.get("settings") or {}).get("arm")
    identity = (state.get("identity") or {}).get("arm")
    _require(top is not None, "checkpoint has no 'arm' field")
    _require(
        top == settings == identity,
        f"checkpoint disagrees with itself about the arm: top={top!r} "
        f"settings.arm={settings!r} identity.arm={identity!r}",
    )
    _require(
        top in ARMS,
        f"checkpoint arm {top!r} is not in the trainer registry {sorted(ARMS)}; a "
        "new label is not an alias for an existing arm",
    )
    return top


def _validate_tensor(name, value, reference):
    _require(torch.is_tensor(value), f"{name}: expected a tensor, got {type(value)}")
    _require(
        value.shape == reference.shape,
        f"{name}: shape {tuple(value.shape)} does not match the model's "
        f"{tuple(reference.shape)}",
    )
    _require(
        value.dtype == reference.dtype,
        f"{name}: dtype {value.dtype} does not match the model's {reference.dtype}",
    )
    _require(torch.isfinite(value).all(), f"{name}: contains non-finite values")


def load_donor(donor_path, *, device=None, proteoaa_root=None):
    """The unmodified donor, plus an immutable snapshot of its weights.

    The snapshot is what makes a later reconstruction independent of load
    order: every checkpoint is overlaid on a model reset to this state, so a
    B1 -> B0 -> R0 sequence cannot leak B1's weights into B0 or R0.
    """
    model, configs, record = load_backbone_model(
        donor_path, device=device, proteoaa_root=proteoaa_root
    )
    snapshot = {
        name: value.detach().clone()
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
    }
    return model, configs, record, snapshot


def _reset_to_donor(model, snapshot):
    """Restore every parameter and buffer to the donor snapshot, in place."""
    with torch.no_grad():
        live = dict(list(model.named_parameters()) + list(model.named_buffers()))
        missing = set(snapshot) ^ set(live)
        _require(
            not missing,
            f"donor snapshot and model disagree on {sorted(missing)[:6]}; the "
            "snapshot cannot be restored safely",
        )
        for name, value in live.items():
            value.copy_(snapshot[name].to(value.device))


def load_joint_checkpoint(
    path,
    *,
    donor_model,
    donor_snapshot,
    donor_record,
    weights="ema",
    expected_arm=None,
    label=None,
    trainable_blocks=None,
):
    """Reconstruct one trained model onto a donor reset to its snapshot.

    ``donor_model`` is mutated in place and returned inside the result, so a
    caller evaluating several checkpoints must finish with one before loading
    the next -- or pass a fresh donor per checkpoint. Resetting first is what
    makes the order irrelevant.
    """
    _require(
        weights in WEIGHT_CHOICES,
        f"weights must be one of {list(WEIGHT_CHOICES)}, got {weights!r}",
    )
    path = Path(path)
    _require(path.is_file(), f"checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    arm = _checkpoint_arm(state)
    if expected_arm is not None:
        _require(
            arm == expected_arm,
            f"{path.name} records arm {arm!r} but was configured as "
            f"{expected_arm!r}; an independent label check exists so a "
            "mislabeled checkpoint cannot enter a paired comparison",
        )

    identity = state.get("identity") or {}
    recorded_donor = ((identity.get("donor") or {}).get("weights") or {}).get("sha256")
    actual_donor = (donor_record.get("weights") or {}).get("sha256")
    _require(
        recorded_donor is not None,
        f"{path.name} records no donor digest, so the donor it was trained on "
        "cannot be confirmed; supply an archived manifest or re-train",
    )
    _require(
        recorded_donor == actual_donor,
        f"{path.name} was trained on donor {recorded_donor[:12]} but the supplied "
        f"donor is {actual_donor[:12]}; content identity is compared rather than "
        "path because the same file moves between clusters",
    )

    _reset_to_donor(donor_model, donor_snapshot)

    settings = state.get("settings") or {}
    n_blocks = int(
        trainable_blocks
        if trainable_blocks is not None
        else settings.get("trainable_blocks", TRAINABLE_BLOCKS)
    )
    # Resolve the allowlist against THIS architecture, then check it against
    # what the run recorded. A renamed module would otherwise silently shift
    # which parameters get overwritten.
    allowlist = select_trainable(donor_model, n_blocks=n_blocks)
    recorded_names = identity.get("trainable_names")
    if recorded_names is not None:
        _require(
            sorted(recorded_names) == sorted(allowlist),
            f"{path.name}: the recorded trainable set and the one this "
            f"architecture resolves differ "
            f"(recorded {len(recorded_names)}, resolved {len(allowlist)})",
        )

    ema_step = None
    if weights == "raw":
        trainable_state = state.get("trainable_state")
        _require(trainable_state is not None, f"{path.name} has no trainable_state")
        _require(
            set(trainable_state) == set(allowlist),
            f"{path.name}: trainable_state keys do not equal the allowlist "
            f"(missing {sorted(set(allowlist) - set(trainable_state))[:4]}, "
            f"extra {sorted(set(trainable_state) - set(allowlist))[:4]})",
        )
        with torch.no_grad():
            for name, parameter in allowlist.items():
                saved = trainable_state[name]
                _validate_tensor(name, saved, parameter)
                # Absolute values. Adding them to the donor would be a
                # different model that still trains and still scores.
                parameter.copy_(saved.to(parameter.device))
    else:
        ema = state.get("ema")
        _require(
            ema is not None,
            f"{path.name} carries no EMA payload but --weights ema was asked "
            "for; falling back to raw would compare different objects across "
            "arms without saying so",
        )
        shadow = ema.get("shadow")
        _require(shadow is not None, f"{path.name}: EMA payload has no shadow")
        floats = {
            name: value
            for name, value in list(donor_model.named_parameters())
            + list(donor_model.named_buffers())
            if value.dtype.is_floating_point
        }
        _require(
            set(shadow) == set(floats),
            f"{path.name}: the EMA shadow does not cover exactly the model's "
            f"floating tensors (missing {sorted(set(floats) - set(shadow))[:4]}, "
            f"extra {sorted(set(shadow) - set(floats))[:4]}). EMA.load_state_dict "
            "ignores both cases, which would leave donor values in place and "
            "report them as a trained model",
        )
        with torch.no_grad():
            for name, value in floats.items():
                saved = shadow[name]
                _validate_tensor(name, saved, value)
                value.copy_(saved.to(value.device))
        ema_step = int(ema.get("step", 0))

    donor_model.eval()
    donor_model.requires_grad_(False)
    driver = PXDesignBackboneDriver(donor_model)
    _assert_no_feedback(driver)

    return LoadedModel(
        label=label or arm,
        arm=arm,
        weights=weights,
        model=donor_model,
        driver=driver,
        step=int(state.get("step", 0)),
        examples_seen=int(state.get("examples_seen", 0)),
        checkpoint_path=str(path.resolve()),
        checkpoint_sha256=file_sha256(path),
        donor_sha256=actual_donor,
        trainable_names=sorted(allowlist),
        trainable_parameters=int(sum(p.numel() for p in allowlist.values())),
        ema_step=ema_step,
        identity=identity,
    )


def load_donor_as_model(
    donor_model, donor_snapshot, donor_record, *, label=DONOR_LABEL
):
    """The untrained donor, as a model row. R0 invents no EMA state."""
    _reset_to_donor(donor_model, donor_snapshot)
    donor_model.eval()
    donor_model.requires_grad_(False)
    driver = PXDesignBackboneDriver(donor_model)
    _assert_no_feedback(driver)
    return LoadedModel(
        label=label,
        arm="R0",
        weights="raw",
        model=donor_model,
        driver=driver,
        step=0,
        examples_seen=0,
        checkpoint_path="",
        checkpoint_sha256="",
        donor_sha256=(donor_record.get("weights") or {}).get("sha256", ""),
        identity=dict(arm="R0", donor=donor_record),
        is_donor=True,
    )


def _assert_no_feedback(driver):
    """Refuse a driver carrying feedback hooks or a payload.

    Evaluation runs every arm through the same plain denoise. A stray hook from
    an adapter experiment would change one model's predictions and nothing in
    the numbers would say so.
    """
    for attribute in ("feedback", "_feedback", "payload", "_payload"):
        value = getattr(driver, attribute, None)
        if value:
            raise EvaluationError(
                f"the driver carries {attribute!r}; evaluation requires a plain "
                "hook-free backbone so every arm runs the identical forward"
            )


# ---- prediction --------------------------------------------------------------


@torch.no_grad()
def predict_backbone(loaded, conditioning, batch, *, sigma, backbone_noise):
    """One denoise for one event; no side-chain branch, no gradients.

    Routed through ``joint_forward(run_sidechain=False)`` rather than a private
    call so the corruption, the masking and the densification are the same code
    training used. ``run_sidechain=False`` also keeps the branch from consuming
    the global RNG, which is what lets two arms see the identical noisy input.
    """
    forward = joint_model.joint_forward(
        loaded.driver,
        conditioning,
        None,
        batch,
        sigma_b=sigma,
        backbone_noise=backbone_noise,
        run_sidechain=False,
    )
    dense, supplied = joint_model.densify_prediction(
        forward.bb_pred, batch.topology, num_tokens=batch.length
    )
    return forward, dense, supplied


def conditioning_for(loaded, batch):
    """Per-target conditioning for the active weights.

    Recomputed per model on purpose. Conditioning is a function of the
    parameters, and an EMA reconstruction differs from the donor in *every*
    floating tensor, so a cache shared across checkpoints would hand one model
    another's conditioning.
    """
    return loaded.driver.conditioning(batch.feature_dict)
