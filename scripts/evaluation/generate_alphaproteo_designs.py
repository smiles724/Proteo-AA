#!/usr/bin/env python3
"""Generate paired AlphaProteo10 designs for designability evaluation.

The official PXDesign checkpoint is evaluated as a backbone generator followed
by ProteinMPNN.  A Proteo-AA checkpoint additionally exports AA sequences from
multiple readouts on the *same sampled backbone*.  This separates backbone
quality from sequence-head quality without paying for duplicate diffusion runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "PXDesign", REPO_ROOT / "Protenix"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TARGETS = (
    "bhrf1", "h1", "il17a", "il7ra", "ir",
    "pdl1", "sc2rbd", "tnfa", "trka", "vegfa",
)
READOUTS = ("final", "target_sigma", "confidence_best")
logger = logging.getLogger(__name__)


def _load_single_target_module():
    path = REPO_ROOT / "scripts/evaluation/design_binder_from_target.py"
    spec = importlib.util.spec_from_file_location("design_binder_from_target", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def length_schedule(
    *, seed: int, target_index: int, count: int,
    length_min: int, length_max: int, fixed_length: int | None,
) -> list[int]:
    """Return a reproducible uniform inclusive length schedule.

    It depends only on benchmark seed and target index, not checkpoint or method,
    so two model jobs receive exactly the same lengths.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if fixed_length is not None:
        if fixed_length < 1:
            raise ValueError("fixed_length must be positive")
        return [fixed_length] * count
    if length_min < 1 or length_max < length_min:
        raise ValueError("invalid length range")
    rng = np.random.default_rng(np.random.SeedSequence([seed, target_index, 105]))
    return rng.integers(length_min, length_max + 1, size=count).tolist()


def design_seed(base_seed: int, target_index: int, sample_index: int) -> int:
    return int(base_seed + target_index * 100_000 + sample_index)


def _csv_values(raw: str, allowed: tuple[str, ...]) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"unsupported values {invalid}; choose from {allowed}")
    return values


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--mode", choices=("pxdesign", "proteoaa"), required=True)
    p.add_argument(
        "--targets-dir",
        default=str(REPO_ROOT / "benchmarks/alphaproteo10/targets"),
    )
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--output-root", required=True)
    p.add_argument("--num-designs-per-target", type=int, default=1)
    p.add_argument("--fixed-length", type=int, default=None)
    p.add_argument("--length-min", type=int, default=80)
    p.add_argument("--length-max", type=int, default=130)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-step", type=int, default=400)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--sampler-mode",
        choices=("pxdesign_native", "minimal_euler"),
        default="pxdesign_native",
    )
    p.add_argument("--aa-readout-sigma", type=float, default=0.4)
    p.add_argument("--aa-readouts", default=",".join(READOUTS))
    p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.model_label):
        raise ValueError("--model-label must contain only letters, digits, ._- ")
    targets = _csv_values(args.targets, TARGETS)
    readouts = _csv_values(args.aa_readouts, READOUTS)
    if args.mode == "proteoaa" and not readouts:
        raise ValueError("Proteo-AA mode requires at least one AA readout")
    if args.num_designs_per_target < 1:
        raise ValueError("--num-designs-per-target must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    targets_dir = Path(args.targets_dir).expanduser().resolve()
    model_root = Path(args.output_root).expanduser().resolve() / args.model_label
    model_root.mkdir(parents=True, exist_ok=True)

    single = _load_single_target_module()
    model = single.build_model(str(checkpoint), args.device)
    if getattr(model, "aa_backend", "mlp") == "fampnn":
        readouts = ["fampnn"]
    manifest_rows: list[dict[str, object]] = []
    commit = _git_commit()

    for target_index, target_name in enumerate(TARGETS):
        if target_name not in targets:
            continue
        cfg_path = targets_dir / f"{target_name}.yaml"
        cfg = single.load_target_config(cfg_path)
        target_atoms, hotspot = single.parse_and_crop(cfg["file"], cfg["chains"])
        lengths = length_schedule(
            seed=args.seed,
            target_index=target_index,
            count=args.num_designs_per_target,
            length_min=args.length_min,
            length_max=args.length_max,
            fixed_length=args.fixed_length,
        )

        arms = ["proteinmpnn"]
        if args.mode == "proteoaa":
            arms.extend(readouts)
        for sample_index, binder_length in enumerate(lengths):
            seed = design_seed(args.seed, target_index, sample_index)
            stem = f"{target_name}_L{binder_length}_seed{seed}"
            paths = {
                arm: model_root / target_name / arm / f"{stem}.cif"
                for arm in arms
            }
            need_generation = not (
                args.skip_existing and all(path.is_file() for path in paths.values())
            )
            if need_generation:
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32 - 1))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                binder = single.fabricate_binder(target_atoms, binder_length)
                feat, _, combined, is_binder = single.build_features(
                    target_atoms, binder, hotspot
                )
                generated = single.design(
                    model,
                    feat,
                    n_step=args.n_step,
                    device=args.device,
                    sampler_mode=args.sampler_mode,
                    seq_mode="complete_unmask",
                    sidechain_cycle=False,
                    seed=seed,
                    refinement_steps=0,
                    aa_readout_mode=(readouts[0] if readouts else "final"),
                    aa_readout_sigma=args.aa_readout_sigma,
                )
                coords = generated["coordinate"].squeeze(0).float().cpu().numpy()
                if not np.isfinite(coords).all():
                    raise FloatingPointError(f"non-finite coordinates for {stem}")
                single.write_cif(
                    paths["proteinmpnn"], combined, coords, is_binder, None
                )
                for readout in readouts if args.mode == "proteoaa" else ():
                    sequence = (
                        (generated["sequence"] if readout == "fampnn" else generated["aa_readouts"][readout]["sequence"])
                        .detach().cpu().numpy()
                    )
                    single.write_cif(
                        paths[readout], combined, coords, is_binder, sequence
                    )
                logger.info(
                    "generated model=%s target=%s sample=%d/%d length=%d seed=%d",
                    args.model_label, target_name, sample_index + 1,
                    len(lengths), binder_length, seed,
                )
            else:
                logger.info("skip existing %s", stem)

            for arm, cif_path in paths.items():
                manifest_rows.append(
                    {
                        "model_label": args.model_label,
                        "model_mode": args.mode,
                        "target": target_name,
                        "sequence_arm": arm,
                        "sample_name": stem,
                        "binder_length": binder_length,
                        "seed": seed,
                        "cif_path": str(cif_path.resolve()),
                        "checkpoint": str(checkpoint),
                        "git_commit": commit,
                        "sampler_mode": args.sampler_mode,
                        "n_step": args.n_step,
                        "aa_readout_sigma": args.aa_readout_sigma,
                    }
                )

    manifest = model_root / "manifest.csv"
    fieldnames = list(manifest_rows[0]) if manifest_rows else []
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary.replace(manifest)
    metadata = {
        "model_label": args.model_label,
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "git_commit": commit,
        "targets": targets,
        "num_designs_per_target": args.num_designs_per_target,
        "fixed_length": args.fixed_length,
        "length_range": [args.length_min, args.length_max],
        "seed": args.seed,
        "n_step": args.n_step,
        "sampler_mode": args.sampler_mode,
        "aa_readouts": readouts if args.mode == "proteoaa" else [],
        "manifest": str(manifest),
        "n_manifest_rows": len(manifest_rows),
    }
    (model_root / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
