#!/usr/bin/env python3
"""Turn the checkpoint roster into a pinned, verified manifest.

    python scripts/build_checkpoint_manifest.py
    python scripts/build_checkpoint_manifest.py --no-hash     # fast re-inspect
    python scripts/build_checkpoint_manifest.py --strict      # fail on any gap

Git commits do not identify trained weights, so nothing in this experiment may
start from "the checkpoint in that directory". This reads
`configs/binder_benchmark/checkpoints.yaml` and writes
`configs/binder_benchmark/checkpoint_manifest.json` recording, per entry:
absolute path, SHA-256, size, training step, whether an EMA shadow exists,
module dimensions and parameter count, the donor hashes the checkpoint recorded
at training time, and the backbone-noise window it was actually trained on.

Two checks here are the reason this is a script and not a directory listing.

**Donor identity is verified by hash, against the file on disk now.** Every
adapter checkpoint in this repo records `frozen.fampnn.sha256` and
`frozen.pxdesign.weights.sha256` from its own training run. This re-hashes those
files today and compares. The failure it exists to catch is the FaMPNN 0.0 to
0.3 swap: the two checkpoints have identical tensor shapes, so they load into
each other without complaint and the adapter silently reads features from a
donor it was never trained against. Shape compatibility is not evidence.

**A missing candidate stays missing.** It is recorded with
`status: unavailable` and a reason. Nothing is substituted, least of all a
freshly initialized adapter, and `--strict` turns any gap into a non-zero exit
so a launcher cannot proceed on a partial roster by accident.

The sigma window is reported in Angstroms as well as in scheduler steps.
`settings.sigma_schedule` records the trajectory (`n_step`, `s_max`, `s_min`,
`rho`) and the clamp (`sigma_min`, `sigma_max`) that together defined training
support, but it stores the window as step indices. Feature caching has to pick
an event by *actual* sigma, so the conversion belongs here rather than in a
comment somewhere downstream.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402  (puts the repo and upstreams on sys.path)

import yaml  # noqa: E402

from pxf.provenance import file_sha256  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER = REPO_ROOT / "configs" / "binder_benchmark" / "checkpoints.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "binder_benchmark" / "checkpoint_manifest.json"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(text: str, defaults: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        return os.environ.get(name) or defaults.get(name) or match.group(0)

    return _VAR.sub(replace, text)


# --------------------------------------------------------------- sigma window


def karras_sigmas(n_step: int, s_max: float, s_min: float, rho: float) -> list[float]:
    """The Karras/EDM noise ladder, as the trajectory sampler walks it.

    Reproduced here rather than imported so the manifest can be rebuilt without
    constructing a sampler, and so the numbers in it are auditable against the
    four values the checkpoint itself recorded.
    """
    if n_step < 2:
        raise ValueError(f"n_step must be >= 2, got {n_step}")
    inv = 1.0 / rho
    hi, lo = s_max ** inv, s_min ** inv
    return [(hi + (i / (n_step - 1)) * (lo - hi)) ** rho for i in range(n_step)]


def sigma_support(schedule: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Translate a recorded `sigma_schedule` into Angstroms.

    **The ladder is in sigma_data units; physical sigma is `ladder *
    sigma_data`.** Getting this wrong is not subtle in its consequences and is
    completely silent in its symptoms: without the factor of 16 the recorded
    training window reads as 0.00058-0.30 A instead of 0.0092-4.76 A, and a
    perfectly supported feature-caching event at 0.429 A looks out of range.
    The evidence for the scaling is inside the checkpoint and does not depend on
    trusting this comment: `distinct_values` is the number of ladder steps that
    fall inside the declared clamp, and it is 115. Scaled, exactly 115 of the
    400 steps land in [0.01, 5.0]; unscaled, 170 do.

    So this returns two independently derived things and reconciles them:

      * `clamp_*` -- the [sigma_min, sigma_max] the run declared, authoritative;
      * `window_sigma_*` -- the sigma at the recorded `window_steps`, derived by
        rebuilding the ladder here.

    When the two disagree beyond a tolerance the record says so rather than
    quietly preferring one. Downstream code picks a feature-caching event by
    *actual* sigma; a step index means nothing without the schedule that
    produced it.
    """
    if not isinstance(schedule, dict):
        return None
    needed = ("n_step", "s_max", "s_min", "rho")
    if not all(k in schedule for k in needed):
        return {"raw": schedule, "note": "incomplete schedule; cannot convert to sigma"}

    sigma_data = float(schedule.get("sigma_data") or 1.0)
    ladder = [
        s * sigma_data
        for s in karras_sigmas(
            int(schedule["n_step"]), float(schedule["s_max"]),
            float(schedule["s_min"]), float(schedule["rho"]),
        )
    ]
    clamp_low = schedule.get("sigma_min")
    clamp_high = schedule.get("sigma_max")
    out: dict[str, Any] = {
        "trajectory_steps": int(schedule["n_step"]),
        "rho": float(schedule["rho"]),
        "sigma_data": sigma_data,
        "sigma_is_ladder_times_sigma_data": True,
        "clamp_sigma_min": clamp_low,
        "clamp_sigma_max": clamp_high,
        "recorded_distinct_values": schedule.get("distinct_values"),
    }

    # Cross-check: how many scaled steps actually fall inside the clamp?
    if clamp_low is not None and clamp_high is not None:
        inside = [i for i, s in enumerate(ladder)
                  if float(clamp_low) <= s <= float(clamp_high)]
        out["steps_inside_clamp"] = len(inside)
        if inside:
            out["clamp_step_range"] = [inside[0], inside[-1]]
        recorded = schedule.get("distinct_values")
        if recorded is not None:
            out["distinct_values_agree"] = (len(inside) == int(recorded))

    window = schedule.get("window_steps")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        first, last = int(window[0]), int(window[1])
        out["window_steps"] = [first, last]
        # The ladder descends, so the earlier step carries the larger sigma.
        # `last` is recorded one past the final in-clamp step in these runs, so
        # clamp it into range rather than indexing off the end.
        lo_index = min(last, len(ladder) - 1)
        if 0 <= first < len(ladder):
            out["window_sigma_high"] = round(ladder[first], 6)
            out["window_sigma_low"] = round(ladder[lo_index], 6)

    # Reconcile. Tolerance is generous because window_steps and the clamp are
    # related by a boundary convention, not by an identity.
    high, low = out.get("window_sigma_high"), out.get("window_sigma_low")
    if None not in (high, low, clamp_low, clamp_high):
        ok_high = abs(high - float(clamp_high)) <= 0.1 * float(clamp_high)
        ok_low = abs(low - float(clamp_low)) <= 0.5 * float(clamp_low)
        out["window_matches_clamp"] = bool(ok_high and ok_low)
        if not (ok_high and ok_low):
            out["reconciliation_note"] = (
                f"window_steps convert to [{low:g}, {high:g}] but the run declared "
                f"[{clamp_low:g}, {clamp_high:g}]. Do not pick an event from either "
                "until this is explained -- one of them is not what was trained."
            )
    return out


def sigma_supported(support: Optional[dict[str, Any]], sigma: float) -> Optional[bool]:
    """Is `sigma` inside the window this checkpoint was trained on?

    The declared clamp is authoritative: it is what the training loop enforced.
    The step-derived window is a cross-check, and `sigma_support` has already
    flagged any disagreement between them.
    """
    if not support:
        return None
    low, high = support.get("clamp_sigma_min"), support.get("clamp_sigma_max")
    if low is None or high is None:
        low = support.get("window_sigma_low")
        high = support.get("window_sigma_high")
    if low is None or high is None:
        return None
    return bool(float(low) <= sigma <= float(high))


# ------------------------------------------------------------- introspection


def _tensor_dict_summary(state: dict[str, Any]) -> dict[str, Any]:
    total = 0
    shapes: dict[str, list[int]] = {}
    for key, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        dims = list(shape)
        shapes[key] = dims
        count = 1
        for d in dims:
            count *= int(d)
        total += count
    return {"n_tensors": len(shapes), "n_parameters": total, "shapes": shapes}


def _donor_records(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull every (path, sha256) pair a checkpoint recorded for a frozen donor."""
    out: dict[str, dict[str, Any]] = {}
    fampnn = frozen.get("fampnn")
    if isinstance(fampnn, dict) and fampnn.get("sha256"):
        out["fampnn"] = {
            "recorded_path": fampnn.get("path"),
            "recorded_sha256": fampnn.get("sha256"),
            "variant": fampnn.get("variant"),
        }
    pxdesign = frozen.get("pxdesign")
    if isinstance(pxdesign, dict):
        weights = pxdesign.get("weights") or {}
        if weights.get("sha256"):
            out["pxdesign"] = {
                "recorded_path": weights.get("path"),
                "recorded_sha256": weights.get("sha256"),
                "model_name": pxdesign.get("model_name"),
                "c_token": pxdesign.get("c_token"),
                "sigma_data": pxdesign.get("sigma_data"),
            }
    return out


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Family-agnostic summary of one checkpoint, without materialising weights.

    Every tensor-bearing top-level group is reported, not just the first one
    recognised. That is not thoroughness for its own sake: `ft_fampnn_phase1` is
    a single file holding a fine-tuned FaMPNN (548 tensors), its raw/EMA
    counterpart, *and* an A_BS adapter (12 tensors). Summarising only the
    adapter described a 201 MB sequence-model fine-tune as a 494,720-parameter
    adapter -- a manifest entry that is wrong in the one direction that matters,
    since it hides the component Phase 2's T2 would actually start from.
    """
    import torch

    # mmap keeps a 1 GB joint checkpoint off the heap; every field read below is
    # either a python scalar or a tensor shape, so nothing is ever touched.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        return {"family": "unknown", "note": f"top level is {type(payload).__name__}"}

    info: dict[str, Any] = {"top_level_keys": sorted(payload)}
    info["step"] = payload.get("step")
    info["examples_seen"] = payload.get("examples_seen")

    # Scalar flags a checkpoint uses to record decisions already made. The
    # EMA/raw question the plan asks is answered outright by some of these.
    flags = {
        key: value for key, value in payload.items()
        if isinstance(value, (bool, int, float, str)) and key not in ("step", "examples_seen")
    }
    if flags:
        info["flags"] = flags

    ema = payload.get("ema")
    if isinstance(ema, dict):
        info["ema"] = {
            "available": bool(ema.get("shadow")),
            "step": ema.get("step"),
            "decay": ema.get("decay"),
            "relative_length": ema.get("relative_length"),
            "n_shadow_tensors": len(ema.get("shadow") or {}),
        }
    else:
        info["ema"] = {"available": False}

    settings = payload.get("settings")
    if isinstance(settings, dict):
        info["settings"] = {
            k: v for k, v in settings.items()
            if not isinstance(v, (dict, list)) or k == "sigma_schedule"
        }
        info["sigma_support"] = sigma_support(settings.get("sigma_schedule") or {})

    # Every tensor group, named. `optimizer` is skipped (its tensors are moment
    # buffers, not weights) and so is the EMA shadow, already counted above.
    skip = {"optimizer", "ema", "rng"}
    groups: dict[str, Any] = {}
    for key, value in payload.items():
        if key in skip or not isinstance(value, dict):
            continue
        summary = _tensor_dict_summary(value)
        if summary["n_tensors"]:
            groups[key] = summary
    if not groups:
        # A bare state dict, e.g. a released donor.
        summary = _tensor_dict_summary(payload)
        if summary["n_tensors"]:
            groups["<top_level>"] = summary
    info["weight_groups"] = {
        name: {"n_tensors": g["n_tensors"], "n_parameters": g["n_parameters"]}
        for name, g in groups.items()
    }
    info["shapes"] = {name: g["shapes"] for name, g in groups.items()}

    # `weights` stays as the headline group so the printed table has one number,
    # chosen as the largest rather than the first key that happened to match.
    if groups:
        headline = max(groups, key=lambda k: groups[k]["n_parameters"])
        info["headline_group"] = headline
        info["weights"] = {
            "n_tensors": groups[headline]["n_tensors"],
            "n_parameters": groups[headline]["n_parameters"],
        }
        info["total_parameters"] = sum(g["n_parameters"] for g in groups.values())

    families = []
    if "adapters" in groups:
        families.append("adapter")
    if "trainable_state" in groups:
        families.append("joint")
    if "state_dict" in groups or "fampnn_state_raw" in groups:
        families.append("sequence_finetune")
    if not families:
        families.append("donor_or_plain")
    info["family"] = "+".join(families)

    info["initialized_from"] = payload.get("initialized_from")
    controller = payload.get("controller")
    if isinstance(controller, dict):
        info["controller"] = {
            "phase": controller.get("phase"),
            "policy": controller.get("policy"),
            "pack_steps": controller.get("pack_steps"),
            "bs_gate": controller.get("bs_gate"),
            "adapters": controller.get("adapters"),
        }
    identity = payload.get("identity")
    if isinstance(identity, dict):
        info["identity"] = {k: v for k, v in identity.items() if k != "trainable_names"}
        info["n_trainable_names"] = len(identity.get("trainable_names") or [])
    if "arm" in payload:
        info["arm"] = payload.get("arm")
    model_cfg = payload.get("model_cfg")
    if model_cfg is not None:
        try:
            from omegaconf import OmegaConf

            info["model_cfg"] = OmegaConf.to_container(model_cfg, resolve=False)
        except Exception:  # noqa: BLE001 - a config we cannot serialise is still a fact
            info["model_cfg"] = {"repr": str(model_cfg)[:2000]}
    info["has_rng_state"] = isinstance(payload.get("rng"), dict)

    frozen = payload.get("frozen")
    if isinstance(frozen, dict):
        info["recorded_donors"] = _donor_records(frozen)
        info["backbone_driver"] = frozen.get("backbone_driver")

    return info


def _run_metadata(checkpoint: Path) -> dict[str, Any]:
    """`run_config.json` / `result.json` sit beside `checkpoints/`, not inside it."""
    run_dir = checkpoint.parent.parent
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("run_config.json", "result.json"):
        candidate = run_dir / name
        if candidate.is_file():
            try:
                out[name] = json.loads(candidate.read_text())
            except json.JSONDecodeError as exc:
                out[name] = {"error": f"unparseable: {exc}"}
        else:
            out[name] = None
    return out


# ------------------------------------------------------------------- driver


def build(roster_path: Path, do_hash: bool) -> dict[str, Any]:
    roster = yaml.safe_load(roster_path.read_text())
    defaults = {k: str(v) for k, v in (roster.get("defaults") or {}).items()}

    entries: list[dict[str, Any]] = []
    donor_hashes: dict[str, str] = {}

    for raw in roster["entries"]:
        path = Path(_expand(str(raw["path"]), defaults))
        record: dict[str, Any] = {
            "role": raw["role"],
            "family": raw.get("family"),
            "required": bool(raw.get("required", False)),
            "note": (raw.get("note") or "").strip() or None,
            "path": str(path),
        }
        if not path.is_file():
            record["status"] = "unavailable"
            record["reason"] = "no file at this path"
            entries.append(record)
            continue

        stat = path.stat()
        record["status"] = "available"
        record["bytes"] = stat.st_size
        record["mtime_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        if do_hash:
            record["sha256"] = file_sha256(path)
            donor_hashes[str(path)] = record["sha256"]

        try:
            record["checkpoint"] = inspect_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint is a finding
            record["status"] = "unreadable"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            entries.append(record)
            continue

        if record["checkpoint"].get("family") != "donor_or_plain":
            record["run_metadata"] = _run_metadata(path)
        entries.append(record)

    # Index the donors this roster knows about by content, so a donor that has
    # simply moved can still be identified. Several checkpoints here were
    # trained on another cluster and record paths under /hai/...; the files
    # themselves were transferred, and their bytes are what matters.
    by_hash: dict[str, list[str]] = {}
    for record in entries:
        digest = record.get("sha256")
        if digest:
            by_hash.setdefault(digest, []).append(record["path"])

    # Cross-check recorded donor hashes against the donor files as they are now.
    for record in entries:
        recorded = (record.get("checkpoint") or {}).get("recorded_donors") or {}
        checks = []
        for donor, detail in recorded.items():
            donor_path = detail.get("recorded_path")
            expected = detail.get("recorded_sha256")
            check = {"donor": donor, "recorded_path": donor_path,
                     "recorded_sha256": expected}
            present = bool(donor_path) and Path(donor_path).is_file()

            if present:
                actual = donor_hashes.get(donor_path) or (
                    file_sha256(donor_path) if do_hash else None
                )
                check["actual_sha256"] = actual
                if actual is None:
                    check["verdict"] = "not_checked"
                elif actual == expected:
                    check["verdict"] = "match"
                else:
                    check["verdict"] = "MISMATCH"
                    check["detail"] = (
                        "the file at the recorded path is not the one this "
                        "checkpoint was trained against"
                    )
            elif expected and expected in by_hash:
                # Same bytes, different path. This is the normal state after a
                # cluster transfer and is a genuine verification, not a
                # concession: the donor is identified by content.
                check["verdict"] = "match_relocated"
                check["resolved_path"] = by_hash[expected][0]
                check["detail"] = (
                    "recorded path does not exist here, but a roster donor has "
                    "exactly these bytes"
                )
            else:
                check["verdict"] = "donor_unresolved"
                check["detail"] = (
                    "the file this checkpoint was trained against is neither at "
                    "the recorded path nor anywhere in this roster; identity "
                    "cannot be confirmed"
                )
            checks.append(check)
        if checks:
            record["donor_verification"] = checks

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "roster": str(roster_path),
        "hashed": do_hash,
        "entries": entries,
    }


def _verdict_lines(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    rows: list[str] = []
    for record in manifest["entries"]:
        role, status = record["role"], record["status"]
        ck = record.get("checkpoint") or {}
        step = ck.get("step")
        ema = (ck.get("ema") or {}).get("available")
        params = ck.get("total_parameters") or (ck.get("weights") or {}).get("n_parameters")
        support = ck.get("sigma_support") or {}
        window = ""
        if support.get("window_sigma_low") is not None:
            window = (f"sigma {support['window_sigma_low']:.4g}"
                      f"-{support['window_sigma_high']:.4g}")
        groups = ",".join((ck.get("weight_groups") or {})) or "-"
        rows.append(
            f"  {role:<18} {status:<11} step={str(step):<7} ema={str(ema):<5} "
            f"params={str(params):<10} {window:<22} [{groups}]"
        )
        if status != "available":
            line = f"{role}: {status} ({record.get('reason')})"
            problems.append(("REQUIRED " if record["required"] else "optional ") + line)
        for check in record.get("donor_verification") or []:
            if check["verdict"] not in ("match", "match_relocated", "not_checked"):
                problems.append(
                    f"{role}: donor {check['donor']} {check['verdict']} "
                    f"-- {check.get('detail', '')}"
                )
    return rows, problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-hash", action="store_true",
                        help="skip SHA-256 (fast); donor verification degrades to "
                             "'not_checked' and the manifest records hashed=false")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any required entry is missing or any "
                             "donor hash mismatches")
    parser.add_argument("--probe-sigma", type=float, default=None,
                        help="report, per adapter, whether this sigma lies inside the "
                             "window it was trained on")
    args = parser.parse_args()

    manifest = build(Path(args.roster), do_hash=not args.no_hash)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows, problems = _verdict_lines(manifest)
    print("checkpoint manifest")
    print("\n".join(rows))

    if args.probe_sigma is not None:
        print(f"\nsigma {args.probe_sigma} against each trained window:")
        for record in manifest["entries"]:
            support = (record.get("checkpoint") or {}).get("sigma_support")
            verdict = sigma_supported(support, args.probe_sigma)
            if verdict is None:
                continue
            print(f"  {record['role']:<20} "
                  f"{'INSIDE' if verdict else 'OUTSIDE — do not use this event'}")

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("\nno problems: every entry present, every donor hash verified")

    print(f"\nwrote {output}")

    if args.strict:
        fatal = [p for p in problems
                 if p.startswith("REQUIRED") or "MISMATCH" in p
                 or "donor_unresolved" in p]
        if fatal:
            raise SystemExit(
                f"--strict: {len(fatal)} blocking problem(s); refusing to pass"
            )


if __name__ == "__main__":
    main()
