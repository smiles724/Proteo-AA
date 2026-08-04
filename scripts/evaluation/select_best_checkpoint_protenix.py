#!/usr/bin/env python3
"""Select the best Proteo-AA checkpoint on all recent-PDB monomer validation rows.

The script evaluates every requested checkpoint on the strict monomer subset of
recentPDB_low_homology_maxtoken1536.csv, computes backbone structural metrics,
and plots C-alpha lDDT, TM-score, and C-alpha RMSD over checkpoint step.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

STEP_RE = re.compile(r"step(\d+)(?:[_-].*)?\.pt$")


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"runs/select_best_checkpoint_protenix_{stamp}"


def _checkpoint_step(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def _discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p).expanduser().resolve() for p in args.checkpoints]
    else:
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        paths = sorted(ckpt_dir.glob(args.checkpoint_pattern), key=_checkpoint_step)
    if args.min_step > 0:
        paths = [p for p in paths if _checkpoint_step(p) >= args.min_step]
    if args.max_step > 0:
        paths = [p for p in paths if _checkpoint_step(p) <= args.max_step]
    if args.checkpoint_stride > 1:
        paths = paths[:: int(args.checkpoint_stride)]
    if args.max_checkpoints > 0:
        paths = paths[: int(args.max_checkpoints)]
    if not paths:
        raise ValueError("No checkpoints selected")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _item_atom_mask(
    mask: torch.Tensor,
    *,
    item_shape: torch.Size,
    n_sample: int,
    n_atom: int,
    device: torch.device,
) -> torch.Tensor:
    mask = mask.to(device=device).bool()
    item_count = int(math.prod(item_shape)) if item_shape else 1
    if mask.dim() == 1:
        item_mask = mask.reshape(1, n_atom).expand(item_count, n_atom)
    else:
        item_mask = mask.expand(*item_shape, n_atom).reshape(item_count, n_atom)
    return (
        item_mask[:, None, :].expand(item_count, n_sample, n_atom).reshape(-1, n_atom)
    )


@torch.no_grad()
def ca_lddt_score(
    pred_coordinate: torch.Tensor,
    gt_coordinate_aug: torch.Tensor,
    coordinate_mask: torch.Tensor,
    ca_atom_mask: torch.Tensor,
    *,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """C-alpha lDDT score. Higher is better; no superposition is applied."""
    n_atom = pred_coordinate.shape[-2]
    n_sample = pred_coordinate.shape[-3]
    item_shape = pred_coordinate.shape[:-3]
    item_count = int(math.prod(item_shape)) if item_shape else 1
    device = pred_coordinate.device

    pred = (
        pred_coordinate.reshape(item_count, n_sample, n_atom, 3)
        .reshape(-1, n_atom, 3)
        .float()
    )
    gt = (
        gt_coordinate_aug.reshape(item_count, n_sample, n_atom, 3)
        .reshape(-1, n_atom, 3)
        .float()
    )
    mask = _item_atom_mask(
        coordinate_mask,
        item_shape=item_shape,
        n_sample=n_sample,
        n_atom=n_atom,
        device=device,
    ) & _item_atom_mask(
        ca_atom_mask,
        item_shape=item_shape,
        n_sample=n_sample,
        n_atom=n_atom,
        device=device,
    )

    scores = []
    thresholds = pred.new_tensor([0.5, 1.0, 2.0, 4.0])
    for p, g, m in zip(pred, gt, mask):
        if int(m.sum()) < 2:
            continue
        p_ca = p[m]
        g_ca = g[m]
        d_true = torch.cdist(g_ca, g_ca)
        d_pred = torch.cdist(p_ca, p_ca)
        pair_mask = (d_true > 0.0) & (d_true < float(cutoff))
        if not bool(pair_mask.any()):
            continue
        delta = (d_pred - d_true).abs()
        per_pair = (delta[..., None] < thresholds).float().mean(dim=-1)
        scores.append(per_pair[pair_mask].mean())
    if not scores:
        return pred_coordinate.new_tensor(float("nan"))
    return torch.stack(scores).mean()


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


@torch.no_grad()
def _forward_metrics(trainer, batch: dict[str, Any]) -> dict[str, float]:
    batch = trainer._to_device(batch)
    dtype = trainer._train_precision()
    ctx = (
        torch.autocast("cuda", dtype=dtype, cache_enabled=False)
        if trainer.device.type == "cuda"
        else torch.no_grad()
    )
    with ctx:
        out = trainer.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            mode="train",
        )
        rep_atom_mask = batch["input_feature_dict"]["distogram_rep_atom_mask"]
        loss_out = trainer.loss_fn(
            pred_coordinate=out["x_denoised"],
            gt_coordinate_aug=out["x_gt_aug"],
            sigma=out["sigma"],
            coordinate_mask=batch["label_dict"]["coordinate_mask"],
            rep_atom_mask=rep_atom_mask,
            distogram_logits=out.get("distogram_logits"),
            aa_logits=out.get("aa_logits"),
            aa_clean=batch["input_feature_dict"].get("aa_clean"),
            aa_loss_mask=batch["input_feature_dict"].get("aa_loss_mask"),
            aa_t=batch["input_feature_dict"].get("aa_t"),
            sc_pred_local=out.get("sc_pred_local"),
            sc_pred_global=out.get("sc_pred_global"),
            sc_frame_R=out.get("sc_frame_R"),
            sc_frame_t=out.get("sc_frame_t"),
            sc_gt_local=batch["input_feature_dict"].get("sc_gt_local"),
            sc_atom_mask=out.get("sc_atom_mask"),
            sc_type_match=out.get("sc_type_match"),
            sc_phys=out.get("sc_phys_val"),
            sc_global=out.get("sc_global_aux"),
            post_pred_coordinate=out.get("post_pred_coordinate"),
            post_gt_coordinate_aug=out.get("post_gt_coordinate_aug"),
            post_aa_logits=out.get("post_aa_logits"),
            eval_ca_atom_mask=batch["input_feature_dict"].get("eval_ca_atom_mask"),
            eval_backbone_atom_mask=batch["input_feature_dict"].get(
                "eval_backbone_atom_mask"
            ),
            weight_bb_post=getattr(trainer, "_weight_bb_post", 1.0),
            weight_aa_post=getattr(trainer, "_weight_aa_post", 1.0),
            backbone_atom_mask=batch["input_feature_dict"].get("backbone_loss_mask"),
        )
        ca_lddt = ca_lddt_score(
            out["x_denoised"],
            out["x_gt_aug"],
            batch["label_dict"]["coordinate_mask"],
            batch["input_feature_dict"]["eval_ca_atom_mask"],
        )

    metrics = {k: _to_float(v) for k, v in loss_out.items()}
    metrics["ca_lddt"] = _to_float(ca_lddt)
    metrics["lddt_loss"] = metrics.pop("lddt")
    return metrics


def evaluate_checkpoint(
    trainer, loader: DataLoader, checkpoint: Path, args: argparse.Namespace
) -> dict[str, Any]:
    _seed_everything(int(args.seed))
    trainer.load_checkpoint(str(checkpoint), params_only=True)
    trainer.model.eval()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_eval = 0
    for i, batch in enumerate(loader):
        if int(args.max_samples) > 0 and i >= int(args.max_samples):
            break
        metrics = _forward_metrics(trainer, batch)
        for key, value in metrics.items():
            if math.isfinite(value):
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        n_eval += 1

    row: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": _checkpoint_step(checkpoint),
        "n_eval": n_eval,
    }
    row.update({key: sums[key] / max(1, counts[key]) for key in sorted(sums)})
    return row


def _best_row(rows: list[dict[str, Any]], select_by: str) -> dict[str, Any]:
    if select_by == "ca_rmsd":
        return min(rows, key=lambda r: r.get("ca_rmsd", float("inf")))
    if select_by == "loss":
        return min(rows, key=lambda r: r.get("loss", float("inf")))
    return max(rows, key=lambda r: r.get(select_by, -float("inf")))


def plot_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("step")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    panels = [
        ("ca_lddt", "C-alpha lDDT", "higher is better"),
        ("tm_score", "TM-score", "higher is better"),
        ("ca_rmsd", "C-alpha RMSD (Angstrom)", "lower is better"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        ax.plot(df["step"], df[metric], marker="o", linewidth=1.6, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Proteo-AA full recent-PDB monomer validation", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-dir",
        default="/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_backbone/bb_100k_val/checkpoints",
    )
    p.add_argument("--checkpoint-pattern", default="step*.pt")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--checkpoint-stride", type=int, default=1)
    p.add_argument("--min-step", type=int, default=0)
    p.add_argument("--max-step", type=int, default=0)
    p.add_argument("--max-checkpoints", type=int, default=0)
    p.add_argument(
        "--select-by",
        choices=["ca_lddt", "tm_score", "ca_rmsd", "loss"],
        default="ca_lddt",
    )

    p.add_argument(
        "--training-stage",
        choices=["backbone_only", "sidechain_warmup", "joint"],
        default="backbone_only",
    )
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--source-index", default="")
    p.add_argument("--filtered-index", default="")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--output-dir", default=_default_output_dir())
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)
    p.add_argument("--dataset-limit", type=int, default=-1)
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--train-samples-per-epoch", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--iters-to-accumulate", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)

    p.add_argument(
        "--aa-mask-mode",
        default="none",
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
        _recent_index_path,
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
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    checkpoints = _discover_checkpoints(args)
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
        else output_dir / "cache" / "recentPDB_monomer_validation_all.csv.gz"
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

    print(f"repo_root={repo_root}")
    print(f"validation_index={filtered_index}")
    print(f"validation_rows={n_items}")
    print(f"checkpoints={len(checkpoints)}")
    if args.dry_run:
        for path in checkpoints[:10]:
            print(f"checkpoint step={_checkpoint_step(path)} path={path}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)

    from pxdesign_train.runner.trainer import (
        PXDesignTrainer,
        TrainerComponents,
        _identity_collate,
    )

    loader = DataLoader(
        components.train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_identity_collate,
    )
    eval_components = TrainerComponents(
        train_dataset=components.train_dataset,
        schedule=components.schedule,
        train_samples_per_epoch=1,
        eval_dataloader=loader,
    )
    configs = build_configs(args, device)
    trainer = PXDesignTrainer(
        configs=configs,
        components=eval_components,
        device=device,
        checkpoint_dir=None,
        load_checkpoint_path=None,
        checkpoint_params_only=True,
    )

    rows = []
    for i, checkpoint in enumerate(checkpoints, start=1):
        row = evaluate_checkpoint(trainer, loader, checkpoint, args)
        rows.append(row)
        print(
            f"[{i}/{len(checkpoints)}] step={row['step']} "
            f"ca_lddt={row.get('ca_lddt', float('nan')):.4f} "
            f"tm_score={row.get('tm_score', float('nan')):.4f} "
            f"ca_rmsd={row.get('ca_rmsd', float('nan')):.4f} "
            f"loss={row.get('loss', float('nan')):.4f}",
            flush=True,
        )

    import pandas as pd

    metrics_csv = output_dir / "checkpoint_metrics.csv"
    ranking_csv = output_dir / "checkpoint_ranking.csv"
    plot_path = output_dir / "checkpoint_structure_metrics.png"
    best_json = output_dir / "best_checkpoint.json"

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(metrics_csv, index=False)
    if args.select_by == "ca_rmsd" or args.select_by == "loss":
        ranked = df.sort_values(args.select_by, ascending=True)
    else:
        ranked = df.sort_values(args.select_by, ascending=False)
    ranked.to_csv(ranking_csv, index=False)
    best = _best_row(rows, args.select_by)
    best_json.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    plot_metrics(rows, plot_path)

    print(f"best_select_by={args.select_by}")
    print(f"best_step={best['step']}")
    print(f"best_checkpoint={best['checkpoint']}")
    print(f"wrote_metrics={metrics_csv}")
    print(f"wrote_ranking={ranking_csv}")
    print(f"wrote_best={best_json}")
    print(f"wrote_plot={plot_path}")


if __name__ == "__main__":
    main()
