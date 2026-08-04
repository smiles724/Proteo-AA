#!/usr/bin/env python3
"""Evaluate the AA head during true free-backbone inference.

Unlike ``eval_aa_head_protenix_monomer.py``, this program never passes native
coordinates to the model. For each strict validation monomer it:

1. masks every design-residue identity;
2. starts backbone coordinates from Gaussian noise;
3. runs the full EDM reverse-diffusion trajectory with ``cogenerate``;
4. reads AA probabilities from the final generated-backbone state; and
5. uses the native sequence only after inference to calculate recovery metrics.

The resulting metrics therefore test the AA head under its intended inference
distribution rather than on a one-step noisy-GT denoising state.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

N_AA = 20


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--source-index", default="")
    p.add_argument("--filtered-index", default="")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Number of monomers to generate; use 0 for the complete validation set.",
    )
    p.add_argument("--n-step", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")

    # Arguments consumed by the shared monomer/config builders.
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--train-samples-per-epoch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0)
    p.add_argument("--eval-num-workers", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--iters-to-accumulate", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--dataset-limit", type=int, default=-1)
    p.add_argument("--aa-mask-mode", default="all")
    p.add_argument("--aa-mask-prob", type=float, default=1.0)
    p.add_argument("--aa-mask-min-prob", type=float, default=1.0)
    p.add_argument("--aa-mask-max-prob", type=float, default=1.0)
    p.add_argument("--aa-input-source", default="diffusion_internal")
    p.add_argument("--trunk-grad-scale", type=float, default=0.0)
    p.add_argument("--disable-aa-loss", action="store_true", default=False)
    p.add_argument("--disable-sidechain", action="store_true", default=True)
    p.add_argument("--enable-coevolution", action="store_true", default=False)
    p.add_argument("--predicted-frame", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--per-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--disable-template-init", action="store_true")
    p.add_argument("--sc-trunk-grad-scale", type=float, default=0.0)
    p.add_argument("--sc-ablation-arm", default="default")
    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--use-template", action="store_true")
    p.add_argument(
        "--ref-pos-augment", action=argparse.BooleanOptionalAction, default=False
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n_step) < 2:
        raise ValueError("--n-step must be at least 2")

    evaluation_dir = Path(__file__).resolve().parent
    training_dir = evaluation_dir.parent / "training"
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(training_dir))

    from eval_aa_head_protenix_monomer import _classification_metrics
    from export_protenix_backbones_for_mpnn import (
        AA1,
        AA3,
        _source_item_with_crop,
        _to_device,
    )
    from train_protenix_monomer import (
        _bootstrap_paths,
        _recent_index_path,
        apply_training_stage_args,
        build_components,
        build_configs,
        build_monomer_index,
    )

    args.training_stage = "aa_head_warmup"
    args.data_mode = "monomer"
    apply_training_stage_args(args)
    repo_root = _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    source_index = (
        Path(args.source_index).resolve()
        if args.source_index
        else _recent_index_path(data_root)
    )
    filtered_index = (
        Path(args.filtered_index).resolve()
        if args.filtered_index
        else output_dir / "cache" / "recentPDB_monomer_inference.csv.gz"
    )
    build_monomer_index(
        source_index=source_index,
        output_index=filtered_index,
        min_n_token=int(args.min_n_token),
        max_n_token=int(args.max_n_token),
        limit=int(args.limit_index),
        rebuild=bool(args.rebuild_index),
    )
    components, n_items = build_components(args, filtered_index)
    source_dataset = components.train_dataset.datasets[0]

    start = int(args.start_index)
    requested = int(args.max_samples)
    stop = n_items if requested <= 0 else min(n_items, start + requested)
    if start < 0 or start >= n_items or stop <= start:
        raise ValueError(f"Invalid sample interval [{start}, {stop}) for {n_items} rows")

    print(f"repo_root={repo_root}")
    print(f"checkpoint={checkpoint}")
    print(f"validation_index={filtered_index}")
    print(f"validation_rows={n_items}")
    print(f"inference_interval=[{start},{stop})")
    print(f"n_step={args.n_step}")
    print("coordinate_source=gaussian_noise")
    print("gt_coordinates_passed_to_model=false")

    if args.dry_run:
        batch, _ = _source_item_with_crop(source_dataset, start)
        feat = batch["input_feature_dict"]
        print(f"n_tokens={feat['aa_clean'].shape[-1]}")
        print(f"design_tokens={int(feat['design_token_mask'].sum())}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)

    from pxdesign_train.cogenerate import cogenerate
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents

    configs = build_configs(args, device)
    eval_components = TrainerComponents(
        train_dataset=components.train_dataset,
        schedule=components.schedule,
        train_samples_per_epoch=1,
        eval_dataloader=None,
    )
    trainer = PXDesignTrainer(
        configs=configs,
        components=eval_components,
        device=device,
        checkpoint_dir=None,
        load_checkpoint_path=None,
        checkpoint_params_only=True,
    )
    trainer.load_checkpoint(str(checkpoint), params_only=True)
    trainer.model.eval()
    precision = trainer._train_precision()

    per_sample_rows: list[dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fasta_path = output_dir / "inference_sequences.fasta"
    trajectory_path = output_dir / "inference_trajectories.jsonl"

    with fasta_path.open("w") as fasta, trajectory_path.open("w") as trajectory_fh:
        for idx in range(start, stop):
            sample_seed = int(args.seed) + idx
            _seed_everything(sample_seed)
            try:
                batch, _ = _source_item_with_crop(source_dataset, idx)
                # Leakage boundary: label_dict remains on CPU and is never passed to
                # cogenerate. Native labels are read only after generation finishes.
                input_features = _to_device(batch["input_feature_dict"], device)
                ctx = (
                    torch.autocast("cuda", dtype=precision, cache_enabled=False)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.no_grad(), ctx:
                    generated = cogenerate(
                        trainer.model,
                        input_feature_dict=input_features,
                        N_step=int(args.n_step),
                        sidechain_cycle=False,
                        seq_mode="complete_unmask",
                    )

                pred = generated["sequence"].detach().cpu().long()
                probs_t = generated.get("aa_probs")
                if probs_t is None:
                    raise RuntimeError("cogenerate returned no final AA probabilities")
                probs = probs_t.detach().float().cpu().numpy()
                labels = batch["input_feature_dict"]["aa_clean"].detach().cpu().long()
                design = (
                    batch["input_feature_dict"]["design_token_mask"]
                    .detach()
                    .cpu()
                    .bool()
                )
                valid = design & (labels >= 0) & (labels < N_AA) & (pred >= 0) & (pred < N_AA)
                if not bool(valid.any()):
                    raise ValueError("No valid design-token AA predictions")

                valid_np = valid.numpy()
                sample_probs = probs[valid_np]
                sample_labels = labels.numpy()[valid_np]
                sample_pred = pred.numpy()[valid_np]
                recovery = float((sample_pred == sample_labels).mean())
                confidence = float(sample_probs.max(axis=-1).mean())
                all_probs.append(sample_probs)
                all_labels.append(sample_labels)

                sample_id = f"val_{idx:05d}"
                native_sequence = "".join(AA1[int(x)] for x in sample_labels)
                predicted_sequence = "".join(AA1[int(x)] for x in sample_pred)
                fasta.write(f">{sample_id}|native\n{native_sequence}\n")
                fasta.write(f">{sample_id}|predicted\n{predicted_sequence}\n")
                trajectory = generated["trajectory"]
                trajectory_fh.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "dataset_index": idx,
                            "seed": sample_seed,
                            "trajectory": trajectory,
                        }
                    )
                    + "\n"
                )
                per_sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_index": idx,
                        "n_tokens": int(valid.sum()),
                        "recovery": recovery,
                        "mean_confidence": confidence,
                        "final_sigma": trajectory[-1]["sigma"] if trajectory else float("nan"),
                        "native_sequence": native_sequence,
                        "predicted_sequence": predicted_sequence,
                    }
                )
                print(
                    f"[{len(per_sample_rows)}/{stop-start}] {sample_id} "
                    f"tokens={int(valid.sum())} recovery={recovery:.4f} "
                    f"confidence={confidence:.4f}",
                    flush=True,
                )
            except Exception as exc:
                logging.exception("Inference failed for dataset index %d: %s", idx, exc)

    if not per_sample_rows:
        raise SystemExit("ERROR: no inference samples completed; inspect the log")

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics, per_class = _classification_metrics(probs_np, labels_np)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "coordinate_source": "gaussian_noise",
            "gt_coordinates_passed_to_model": False,
            "n_step": int(args.n_step),
            "seed": int(args.seed),
            "n_requested": stop - start,
            "n_completed": len(per_sample_rows),
            "mean_per_protein_recovery": float(
                np.mean([float(row["recovery"]) for row in per_sample_rows])
            ),
        }
    )
    for row in per_class:
        row["class_name"] = AA3[int(row["class_idx"])]

    summary_path = output_dir / "inference_summary.json"
    per_sample_path = output_dir / "inference_per_sample.csv"
    per_class_path = output_dir / "inference_per_class.csv"
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_csv(per_sample_path, per_sample_rows)
    _write_csv(per_class_path, per_class)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_per_sample={per_sample_path}")
    print(f"wrote_per_class={per_class_path}")
    print(f"wrote_fasta={fasta_path}")
    print(f"wrote_trajectories={trajectory_path}")


if __name__ == "__main__":
    main()
