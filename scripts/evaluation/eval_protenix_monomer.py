#!/usr/bin/env python3
"""Evaluate a Proteo-AA monomer checkpoint on Protenix monomer rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"runs/eval_protenix_monomer_{stamp}"


def _recent_index_path(data_root: Path) -> Path:
    return data_root / "indices" / "recentPDB_low_homology_maxtoken1536.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--training-stage",
        default="backbone_only",
        choices=["backbone_only", "sidechain_warmup", "joint"],
    )
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--source-index", default="")
    p.add_argument("--output-dir", default=_default_output_dir())
    p.add_argument("--filtered-index", default="")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--dataset-limit", type=int, default=-1)
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)

    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument(
        "--aa-mask-mode",
        default="all",
        choices=["all", "partial", "none", "time_dependent"],
    )
    p.add_argument("--aa-mask-prob", type=float, default=1.0)
    p.add_argument("--aa-mask-min-prob", type=float, default=0.0)
    p.add_argument("--aa-mask-max-prob", type=float, default=1.0)
    p.add_argument(
        "--aa-input-source",
        default="diffusion_internal",
        choices=["s_inputs", "diffusion_internal"],
    )
    p.add_argument("--trunk-grad-scale", type=float, default=1.0)
    p.add_argument("--disable-aa-loss", action="store_true")

    p.add_argument("--disable-sidechain", action="store_true")
    p.add_argument("--enable-coevolution", action="store_true")
    p.add_argument(
        "--predicted-frame", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--per-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--disable-template-init", action="store_true")
    p.add_argument("--sc-trunk-grad-scale", type=float, default=1.0)
    p.add_argument(
        "--sc-ablation-arm",
        default="default",
        choices=["default", "no", "a-indirect", "a-direct", "bbctx", "q", "a-direct+q"],
    )

    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--use-template", action="store_true")
    p.add_argument(
        "--ref-pos-augment", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = Path(__file__).resolve().parent.parent / "training"
    sys.path.insert(0, str(training_dir))

    from train_protenix_monomer import (
        _bootstrap_paths,
        apply_training_stage_args,
        build_components,
        build_configs,
        build_monomer_index,
    )

    apply_training_stage_args(args)
    repo_root = _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch
    from torch.utils.data import DataLoader, Subset

    from pxdesign_train.runner.trainer import (
        PXDesignTrainer,
        TrainerComponents,
        _identity_collate,
    )

    data_root = Path(args.data_root).resolve()
    source_index = (
        Path(args.source_index).resolve()
        if args.source_index
        else _recent_index_path(data_root)
    )
    output_dir = Path(args.output_dir).resolve()
    filtered_index = (
        Path(args.filtered_index).resolve()
        if args.filtered_index
        else output_dir / "cache" / "protenix_eval_monomer_index.csv.gz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.max_n_token) > int(args.crop_size):
        raise ValueError("--max-n-token must be <= --crop-size")

    build_monomer_index(
        source_index=source_index,
        output_index=filtered_index,
        min_n_token=int(args.min_n_token),
        max_n_token=int(args.max_n_token),
        limit=int(args.limit_index),
        rebuild=bool(args.rebuild_index),
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)

    # Reuse the training data builder, but use it as a fixed eval dataset here.
    args.train_samples_per_epoch = 1
    args.max_steps = 1
    args.lr = 5e-4
    args.warmup_steps = 0
    args.log_interval = 0
    args.eval_interval = 0
    args.checkpoint_interval = 0
    args.ema_decay = 0.0
    args.iters_to_accumulate = 1
    args.grad_clip_norm = 0.0

    components, n_items = build_components(args, filtered_index)
    eval_dataset = components.train_dataset
    if int(args.num_samples) > 0:
        eval_dataset = Subset(
            eval_dataset, range(min(int(args.num_samples), len(eval_dataset)))
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_identity_collate,
    )
    eval_components = TrainerComponents(
        train_dataset=components.train_dataset,
        schedule=components.schedule,
        train_samples_per_epoch=1,
        eval_dataloader=eval_loader,
    )

    configs = build_configs(args, device)
    trainer = PXDesignTrainer(
        configs=configs,
        components=eval_components,
        device=device,
        checkpoint_dir=None,
        load_checkpoint_path=str(Path(args.checkpoint).resolve()),
        checkpoint_params_only=True,
    )
    metrics = trainer.evaluate()

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_index": str(source_index),
        "filtered_index": str(filtered_index),
        "dataset_rows": n_items,
        "evaluated_samples": len(eval_dataset),
        # Set-wide means, plus the per-protein rows they were computed from.
        "metrics": metrics,
        "per_protein": trainer.last_eval_per_protein,
    }
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote_metrics={out_path}")
    logging.info("Repo root: %s", repo_root)


if __name__ == "__main__":
    main()
