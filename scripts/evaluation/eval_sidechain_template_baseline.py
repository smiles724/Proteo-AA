#!/usr/bin/env python3
"""Evaluate the side-chain initialization template on a Protenix monomer set."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    parser.add_argument("--filtered-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=491)
    parser.add_argument("--min-n-token", type=int, default=16)
    parser.add_argument("--max-n-token", type=int, default=640)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--max-crop-retries", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--template-provider", default="dunbrack_mode")
    parser.add_argument("--sigma-t", type=float, default=0.3)
    parser.add_argument("--protenix-code-dir", default="")
    parser.add_argument("--pxdesign-code-dir", default="")
    return parser.parse_args()


def _default_training_args(training_module) -> argparse.Namespace:
    saved_argv = sys.argv
    try:
        sys.argv = ["train_protenix_monomer.py"]
        return training_module.parse_args()
    finally:
        sys.argv = saved_argv


def main() -> None:
    args = parse_args()
    training_dir = Path(__file__).resolve().parent.parent / "training"
    sys.path.insert(0, str(training_dir))

    import train_protenix_monomer as training

    train_args = _default_training_args(training)
    train_args.training_stage = "sidechain_warmup"
    train_args.data_root = args.data_root
    train_args.output_dir = args.output_dir
    train_args.eval_interval = 1
    train_args.eval_samples = int(args.num_samples)
    train_args.eval_filtered_index = args.filtered_index
    train_args.min_n_token = int(args.min_n_token)
    train_args.max_n_token = int(args.max_n_token)
    train_args.crop_size = int(args.crop_size)
    train_args.max_crop_retries = int(args.max_crop_retries)
    train_args.eval_num_workers = int(args.num_workers)
    train_args.protenix_code_dir = args.protenix_code_dir
    train_args.pxdesign_code_dir = args.pxdesign_code_dir
    training.apply_training_stage_args(train_args)
    training._bootstrap_paths(train_args)

    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    import torch

    from pxdesign_train.sidechain.frames import phi_psi_from_ncac
    from pxdesign_train.sidechain.templates import PROVIDERS

    if args.template_provider not in PROVIDERS:
        raise ValueError(
            f"Unknown template provider {args.template_provider!r}; "
            f"choose from {sorted(PROVIDERS)}"
        )
    provider = PROVIDERS[args.template_provider]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_loader, n_eval, eval_index = training.build_eval_dataloader(
        train_args, output_dir
    )
    if eval_loader is None:
        raise RuntimeError("Validation loader was not constructed")

    rows: list[dict[str, float | int]] = []
    total_squared_error = 0.0
    total_atoms = 0
    for dataset_index, batch in enumerate(eval_loader):
        feat = batch["input_feature_dict"]
        type_idx = feat["aa_clean"].detach().cpu().long()
        atom_mask = feat["sc_atom_mask"].detach().cpu().bool()
        gt_local = feat["sc_gt_local"].detach().cpu().float()
        bb = feat["sc_bb_coords"].detach().cpu().float()
        bb_idx = feat["sc_bb_atom_idx"].detach().cpu().long()
        residue_index = feat.get("residue_index")
        asym_id = feat.get("asym_id")

        phi = psi = None
        if residue_index is not None and asym_id is not None:
            have = (bb_idx[..., :3] >= 0).all(dim=-1)
            phi, psi = phi_psi_from_ncac(
                bb[..., 0, :],
                bb[..., 1, :],
                bb[..., 2, :],
                residue_index.detach().cpu(),
                asym_id.detach().cpu(),
                have=have,
            )

        safe_type = type_idx.clamp(0, 19)
        template, template_mask = provider(safe_type, phi=phi, psi=psi)
        template = template.detach().cpu().float()
        valid = atom_mask & template_mask.detach().cpu().bool()
        n_atoms = int(valid.sum())
        if n_atoms == 0:
            continue

        squared_error = ((template - gt_local) ** 2).sum(dim=-1)
        sample_sum = float((squared_error * valid).sum())
        sample_mse = sample_sum / n_atoms
        rows.append(
            {
                "dataset_index": dataset_index,
                "n_sidechain_atoms": n_atoms,
                "template_mse": sample_mse,
                "template_rmsd": math.sqrt(sample_mse),
            }
        )
        total_squared_error += sample_sum
        total_atoms += n_atoms
        if (dataset_index + 1) % 50 == 0:
            print(f"processed={dataset_index + 1}/{n_eval}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No validation sample had resolved side-chain atoms")

    sample_mean_mse = float(frame["template_mse"].mean())
    atom_weighted_mse = total_squared_error / max(1, total_atoms)
    expected_noisy_mse = sample_mean_mse + 3.0 * float(args.sigma_t) ** 2
    result = {
        "template_provider": args.template_provider,
        "filtered_index": str(Path(eval_index).resolve()),
        "requested_samples": int(args.num_samples),
        "evaluated_samples": int(len(frame)),
        "sidechain_atoms": int(total_atoms),
        "sample_mean_mse": sample_mean_mse,
        "sample_mean_rmse": math.sqrt(sample_mean_mse),
        "sample_median_mse": float(frame["template_mse"].median()),
        "sample_p90_mse": float(frame["template_mse"].quantile(0.90)),
        "sample_p95_mse": float(frame["template_mse"].quantile(0.95)),
        "sample_max_mse": float(frame["template_mse"].max()),
        "atom_weighted_mse": atom_weighted_mse,
        "atom_weighted_rmse": math.sqrt(atom_weighted_mse),
        "sigma_t": float(args.sigma_t),
        "expected_template_plus_noise_mse": expected_noisy_mse,
    }
    metrics_path = output_dir / "template_baseline.json"
    samples_path = output_dir / "template_baseline_samples.csv"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    frame.to_csv(samples_path, index=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote_metrics={metrics_path}")
    print(f"wrote_samples={samples_path}")


if __name__ == "__main__":
    main()
