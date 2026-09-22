"""Load a feedback checkpoint, or refuse to.

A feedback module is a function of the state the *upstream* pipeline produced:
which A_BS weights conditioned the decode, whether they were EMA or raw, which
donors, which context, how many decode steps, at what temperature. Run it
against a different upstream and it is being asked to correct a distribution it
never saw, silently and with plausible-looking output.

So a checkpoint must carry ``integrated_policy`` describing the training that
actually happened, and this module refuses by default when it is missing or
disagrees with the run being set up.

### Two failures that are not the same

**A donor mismatch is never allowed.** A feedback module trained against
FaMPNN 0.0 cannot run with J03 on FaMPNN 0.3: the representation it reads is a
different function. No flag overrides this.

**A policy difference can be declared.** Decode settings, EMA-vs-raw, context:
a deliberate ablation may legitimately differ, so
``allow_transfer=True`` (the CLI's ``--allow-feedback-policy-transfer``)
records every difference in the run's provenance and proceeds. It is a
diagnostic, not a way to make an incompatible pair load.

### On retrofitting

``integrated_policy`` must describe real training. Adding it by hand to an old
0.0 or bypass checkpoint so the loader accepts it converts a loud refusal into
a quiet claim that the checkpoint was trained under a policy it was not.
:func:`check_policy` cannot detect that -- nothing can -- which is exactly why
it is written here as a prohibition rather than a validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

TASK = "integrated_feedback_v1"

#: Keys ``integrated_policy`` must carry. A checkpoint missing any of them
#: cannot be checked, which is the same risk as being wrong.
REQUIRED_POLICY = (
    "bs_checkpoint_sha256",
    "bs_weights",
    "application_mode",
    "sequence_source",
    "context",
    "feedback_scope",
    "seq_steps",
    "pack_steps",
    "temperature",
)

#: Differences that are a hard refusal: the module reads a different function.
DONOR_KEYS = ("bs_checkpoint_sha256", "application_mode", "sequence_source")

#: Differences that ``allow_transfer`` may record and proceed past.
TRANSFERABLE = ("bs_weights", "context", "feedback_scope", "seq_steps",
                "pack_steps", "temperature")


@dataclass
class PolicyReport:
    """What matched, what differed, and whether the run may proceed."""

    matched: list[str] = field(default_factory=list)
    donor_mismatches: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    policy_differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    transferred: bool = False

    @property
    def ok(self) -> bool:
        return not self.donor_mismatches and (
            not self.policy_differences or self.transferred
        )

    def record(self) -> dict[str, Any]:
        return {
            "matched": sorted(self.matched),
            "donor_mismatches": {k: list(v) for k, v in self.donor_mismatches.items()},
            "policy_differences": {
                k: list(v) for k, v in self.policy_differences.items()
            },
            "policy_transferred": self.transferred,
        }


def file_sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_policy(
    recorded: Optional[dict[str, Any]],
    expected: dict[str, Any],
    *,
    allow_transfer: bool = False,
    path=None,
) -> PolicyReport:
    """Compare a checkpoint's ``integrated_policy`` against this run's.

    Raises unless the donors match and either the policy matches or
    ``allow_transfer`` is set.
    """
    where = f"{path}: " if path else ""
    if not recorded:
        raise SystemExit(
            f"{where}no integrated_policy in this checkpoint. A feedback module "
            "is a function of the upstream that produced its training states, "
            f"so it cannot be run without one. Expected keys: "
            f"{list(REQUIRED_POLICY)}. Do NOT add them by hand to an older "
            "checkpoint -- that turns this refusal into a false claim about "
            "how it was trained."
        )
    missing = [k for k in REQUIRED_POLICY if k not in recorded]
    if missing:
        raise SystemExit(
            f"{where}integrated_policy is missing {missing}, so the run cannot "
            "be checked against it"
        )

    report = PolicyReport()
    for key in REQUIRED_POLICY:
        if key not in expected:
            continue
        want, got = expected[key], recorded[key]
        if _same(want, got):
            report.matched.append(key)
        elif key in DONOR_KEYS:
            report.donor_mismatches[key] = (got, want)
        else:
            report.policy_differences[key] = (got, want)

    if report.donor_mismatches:
        lines = "\n".join(
            f"    {k}: checkpoint {g!r} != run {w!r}"
            for k, (g, w) in sorted(report.donor_mismatches.items())
        )
        raise SystemExit(
            f"{where}DONOR MISMATCH -- refused, and not overridable:\n{lines}\n"
            "The feedback module reads the representation its upstream "
            "produced. A module trained against one A_BS/donor/sequence source "
            "is a different function from one trained against another; loading "
            "it anyway would produce plausible output from a model being asked "
            "a question it was never trained on."
        )
    if report.policy_differences:
        lines = "\n".join(
            f"    {k}: checkpoint {g!r} != run {w!r}"
            for k, (g, w) in sorted(report.policy_differences.items())
        )
        if not allow_transfer:
            raise SystemExit(
                f"{where}integrated_policy differs from this run:\n{lines}\n"
                "Pass --allow-feedback-policy-transfer to record the "
                "differences and proceed. That is a diagnostic: it does not "
                "make the numbers comparable to a matched run."
            )
        report.transferred = True
    return report


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


def expected_policy(
    *,
    bs_checkpoint,
    bs_weights: str,
    context: str,
    seq_steps: int,
    pack_steps: int,
    temperature: float,
    feedback_scope: str = "binder_single_binder_binder_pair",
    application_mode: str = "shared_prelogit",
    sequence_source: str = "predicted",
) -> dict[str, Any]:
    """The policy THIS run implements, for comparison against a checkpoint."""
    return {
        "bs_checkpoint_sha256": (
            None if bs_checkpoint is None else file_sha256(bs_checkpoint)
        ),
        "bs_weights": bs_weights,
        "application_mode": application_mode,
        "sequence_source": sequence_source,
        "context": context,
        "feedback_scope": feedback_scope,
        "seq_steps": int(seq_steps),
        "pack_steps": int(pack_steps),
        "temperature": float(temperature),
    }


def load_feedback(
    path,
    *,
    expected: dict[str, Any],
    expected_arm: str,
    c_h_V: int,
    c_s: int,
    c_z: int,
    c_token: int,
    device=None,
    allow_transfer: bool = False,
    weights: str = "ema",
):
    """``(module, report, identity)`` for a trained feedback checkpoint.

    ``expected_arm`` is the arm the CALLER asked for, e.g. from
    ``--feedback-arm``. It matters that it comes from outside the checkpoint:
    E1's ``full`` and ``bb_only`` variants have identical parameter shapes by
    construction -- that is what makes them a matched control -- so the wrong
    one loads cleanly under ``strict=True`` and the run reports the wrong
    variant. The metadata is the only thing that distinguishes them, and
    comparing it against a request is the only version of that check which is
    not a record compared with itself.
    """
    import torch

    from pxf.couple.conditioning import (build_conditioner,
                                         check_is_a_known_arm,
                                         check_is_the_expected_arm)

    path = Path(path)
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    if identity.get("task") not in (None, TASK):
        raise SystemExit(
            f"{path}: records task={identity.get('task')!r}, expected {TASK!r}. "
            "A checkpoint from another task is a different experiment."
        )
    report = check_policy(
        state.get("integrated_policy"), expected,
        allow_transfer=allow_transfer, path=path,
    )

    # The arm the checkpoint IS, from its recorded architecture triple...
    arm = check_is_a_known_arm(identity, path=path)
    # ...checked against the arm the caller ASKED for. Building from the
    # recorded identity and then comparing the two would compare a record with
    # itself; `check_compatible` says so in as many words and is the wrong
    # function here.
    check_is_the_expected_arm(identity, expected_arm, path=path)
    module = build_conditioner(
        arm, c_h_V=c_h_V, c_token=c_token, c_s=c_s, c_z=c_z,
        **_reconstructed(identity),
    )
    module.load_state_dict(state["conditioner"])
    if state.get("ema") and weights == "ema":
        from pxf.train.ema import EMA

        ema = EMA(module, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(module)
    module.eval().requires_grad_(False)
    if device is not None:
        module.to(device)
    return module, report, identity


def _reconstructed(identity: dict[str, Any]) -> dict[str, Any]:
    """Constructor arguments a checkpoint may legitimately differ on."""
    from pxf.couple.conditioning import reconstruct_kwargs

    return reconstruct_kwargs(identity)
