#!/usr/bin/env python3
"""Evaluate the Proteo-AA residue-type prediction head on monomer checkpoints.

By default this evaluates strict monomers from the recent-PDB low-homology set.
Each residue in the monomer binder is masked (`aa_mask_mode=all`) and scored
against `aa_clean`. The script reports multiclass accuracy/F1 plus one-vs-rest
AUROC/AUPRC for the 20 amino-acid classes.
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
N_AA = 20
AA_IGNORE_INDEX = -100


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"runs/eval_aa_head_protenix_monomer_{stamp}"


def _checkpoint_step(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def _discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p).expanduser().resolve() for p in args.checkpoints]
    else:
        if not args.checkpoint_dir:
            raise ValueError("Specify --checkpoint-dir or --checkpoints")
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        paths = sorted(ckpt_dir.glob(args.checkpoint_pattern), key=_checkpoint_step)
    if args.min_step > 0:
        paths = [p for p in paths if _checkpoint_step(p) >= int(args.min_step)]
    if args.max_step > 0:
        paths = [p for p in paths if _checkpoint_step(p) <= int(args.max_step)]
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


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - np.nanmax(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-30, None)


def _select_probs(
    out: dict[str, torch.Tensor],
    aa_clean: torch.Tensor,
    *,
    logits_source: str,
) -> np.ndarray:
    if logits_source == "reduced":
        logits_t = out.get("aa_logits_reduced")
        if logits_t is None:
            logits_t = out.get("aa_logits")
    else:
        logits_t = out.get("aa_logits")
        if logits_t is None:
            logits_t = out.get("aa_logits_reduced")
    if logits_t is None:
        raise RuntimeError(
            "Model output did not contain aa_logits or aa_logits_reduced"
        )

    logits = _to_numpy(logits_t)
    label_ndim = aa_clean.dim()
    has_sample_axis = logits_t.dim() == label_ndim + 2

    if logits_source == "all_samples" and has_sample_axis:
        return _softmax_np(logits)
    if logits_source == "mean_sample" and has_sample_axis:
        return _softmax_np(logits).mean(axis=-3)
    if logits_source == "mean_sample":
        return _softmax_np(logits)
    if logits_source == "reduced":
        return _softmax_np(logits)
    return _softmax_np(logits)


def _flatten_eval_tokens(
    probs: np.ndarray,
    aa_clean: torch.Tensor,
    design_mask: torch.Tensor,
    aa_loss_mask: torch.Tensor | None,
    *,
    mask_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = aa_clean.detach().cpu().long().numpy()
    design = design_mask.detach().cpu().bool().numpy()
    if aa_loss_mask is None:
        aa_mask = design
    else:
        aa_mask = aa_loss_mask.detach().cpu().bool().numpy()
    if mask_source == "design":
        valid = design
    elif mask_source == "aa_loss":
        valid = aa_mask
    else:
        valid = design & aa_mask
    valid = valid & (labels >= 0) & (labels < N_AA)

    if probs.ndim == labels.ndim + 2:
        n_sample = probs.shape[-3]
        labels = np.expand_dims(labels, axis=-2)
        valid = np.expand_dims(valid, axis=-2)
        labels = np.broadcast_to(labels, probs.shape[:-1])
        valid = np.broadcast_to(valid, probs.shape[:-1])
        if n_sample <= 0:
            raise ValueError("Invalid sample axis in aa_logits")
    elif probs.shape[:-1] != labels.shape:
        raise ValueError(
            f"AA logits/labels shape mismatch: probs={probs.shape}, labels={labels.shape}"
        )

    return probs[valid].reshape(-1, N_AA), labels[valid].reshape(-1)


def _binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)

    sorted_score = score[order]
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_score[end] == sorted_score[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end

    sum_pos_ranks = ranks[y_true].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / (np.arange(y.size, dtype=np.float64) + 1.0)
    return float((precision * y).sum() / n_pos)


def _classification_metrics(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if labels.size == 0:
        raise ValueError("No valid AA labels found for evaluation")

    pred = probs.argmax(axis=-1)
    true_counts = np.bincount(labels, minlength=N_AA).astype(np.float64)
    pred_counts = np.bincount(pred, minlength=N_AA).astype(np.float64)
    tp = np.bincount(labels[pred == labels], minlength=N_AA).astype(np.float64)
    fp = pred_counts - tp
    fn = true_counts - tp

    precision = np.divide(tp, tp + fp, out=np.full(N_AA, np.nan), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full(N_AA, np.nan), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full(N_AA, np.nan),
        where=(precision + recall) > 0,
    )
    class_acc = np.divide(
        tp, true_counts, out=np.full(N_AA, np.nan), where=true_counts > 0
    )

    support_mask = true_counts > 0
    weighted = true_counts / max(1.0, true_counts.sum())
    y_score_true = probs[np.arange(labels.size), labels]
    ce = float(-np.log(np.clip(y_score_true, 1e-30, 1.0)).mean())
    top5 = np.argpartition(probs, kth=-5, axis=-1)[:, -5:]

    one_hot = labels[:, None] == np.arange(N_AA)[None, :]
    auroc = np.array([_binary_auroc(one_hot[:, c], probs[:, c]) for c in range(N_AA)])
    auprc = np.array([_binary_auprc(one_hot[:, c], probs[:, c]) for c in range(N_AA)])

    macro_f1 = float(np.nanmean(f1[support_mask]))
    weighted_f1 = float(np.nansum(f1 * weighted))
    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_precision = micro_tp / max(1e-30, micro_tp + micro_fp)
    micro_recall = micro_tp / max(1e-30, micro_tp + micro_fn)
    micro_f1 = (
        2.0
        * micro_precision
        * micro_recall
        / max(1e-30, micro_precision + micro_recall)
    )

    metrics = {
        "n_tokens": int(labels.size),
        "aa_ce": ce,
        "acc": float((pred == labels).mean()),
        "top5_acc": float((top5 == labels[:, None]).any(axis=-1).mean()),
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "f1_micro": float(micro_f1),
        "auroc_macro": float(np.nanmean(auroc[support_mask])),
        "auroc_weighted": float(np.nansum(auroc * weighted)),
        "auroc_micro": _binary_auroc(one_hot.reshape(-1), probs.reshape(-1)),
        "auprc_macro": float(np.nanmean(auprc[support_mask])),
        "auprc_weighted": float(np.nansum(auprc * weighted)),
        "auprc_micro": _binary_auprc(one_hot.reshape(-1), probs.reshape(-1)),
    }

    per_class = []
    for c in range(N_AA):
        per_class.append(
            {
                "class_idx": c,
                "support": int(true_counts[c]),
                "precision": (
                    float(precision[c])
                    if math.isfinite(float(precision[c]))
                    else float("nan")
                ),
                "recall": (
                    float(recall[c])
                    if math.isfinite(float(recall[c]))
                    else float("nan")
                ),
                "f1": float(f1[c]) if math.isfinite(float(f1[c])) else float("nan"),
                "class_acc": (
                    float(class_acc[c])
                    if math.isfinite(float(class_acc[c]))
                    else float("nan")
                ),
                "auroc": (
                    float(auroc[c]) if math.isfinite(float(auroc[c])) else float("nan")
                ),
                "auprc": (
                    float(auprc[c]) if math.isfinite(float(auprc[c])) else float("nan")
                ),
            }
        )
    return metrics, per_class


def _best_row(rows: list[dict[str, Any]], select_by: str) -> dict[str, Any]:
    if select_by == "aa_ce":
        return min(rows, key=lambda r: r.get(select_by, float("inf")))
    return max(rows, key=lambda r: r.get(select_by, -float("inf")))


@torch.no_grad()
def evaluate_checkpoint(
    trainer, loader: DataLoader, checkpoint: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(int(args.seed))
    trainer.load_checkpoint(str(checkpoint), params_only=True)
    trainer.model.eval()

    all_probs = []
    all_labels = []
    n_eval = 0
    dtype = trainer._train_precision()
    ctx = (
        torch.autocast("cuda", dtype=dtype, cache_enabled=False)
        if trainer.device.type == "cuda"
        else torch.no_grad()
    )
    for i, batch in enumerate(loader):
        if int(args.max_samples) > 0 and i >= int(args.max_samples):
            break
        batch = trainer._to_device(batch)
        with ctx:
            out = trainer.model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                mode="train",
            )
        feat = batch["input_feature_dict"]
        probs = _select_probs(out, feat["aa_clean"], logits_source=args.logits_source)
        p, y = _flatten_eval_tokens(
            probs,
            feat["aa_clean"],
            feat["design_token_mask"],
            feat.get("aa_loss_mask"),
            mask_source=args.mask_source,
        )
        if y.size:
            all_probs.append(p)
            all_labels.append(y)
        n_eval += 1

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics, per_class = _classification_metrics(probs_np, labels_np)
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": _checkpoint_step(checkpoint),
        "n_eval": n_eval,
        **metrics,
    }
    for class_row in per_class:
        class_row["checkpoint"] = str(checkpoint)
        class_row["step"] = row["step"]
    return row, per_class


def _apply_aa_eval_args(args: argparse.Namespace) -> None:
    args.training_stage = args.model_stage
    args.data_mode = "monomer"
    args.disable_sidechain = True
    # An AA-head eval is exactly the measurement the leaky input destroys
    # (~96% vs ~6% accuracy), so it is pinned off here rather than exposed.
    args.allow_binder_sidechain_leakage = False
    args.enable_coevolution = False
    args.disable_aa_loss = False
    args.aa_mask_mode = "all"
    args.aa_mask_prob = 1.0
    args.aa_mask_min_prob = 1.0
    args.aa_mask_max_prob = 1.0
    args.predicted_frame = True
    args.per_sigma = True
    args.template_provider = "dunbrack_mode"
    args.disable_template_init = False
    args.sc_trunk_grad_scale = 0.0
    args.sc_ablation_arm = "no"
    args.trunk_grad_scale = 0.0


def _aa_names(repo_root: Path, pxdesign_code_dir: str) -> list[str]:
    try:
        if pxdesign_code_dir:
            sys.path.insert(0, str(Path(pxdesign_code_dir).resolve()))
        from pxdesign.data.constants import PRO_STD_RESIDUES_NATURAL

        names = [f"AA{i}" for i in range(N_AA)]
        for name, idx in PRO_STD_RESIDUES_NATURAL.items():
            if 0 <= int(idx) < N_AA:
                names[int(idx)] = str(name)
        return names
    except Exception:
        return [f"AA{i}" for i in range(N_AA)]


def plot_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("step")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("acc", "Accuracy", "higher is better"),
        ("f1_macro", "Macro F1", "higher is better"),
        ("auroc_macro", "Macro AUROC", "higher is better"),
        ("auprc_macro", "Macro AUPRC", "higher is better"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels):
        ax.plot(df["step"], df[metric], marker="o", linewidth=1.6, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Proteo-AA monomer AA-head validation", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--checkpoint-pattern", default="step*.pt")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--checkpoint-stride", type=int, default=1)
    p.add_argument("--min-step", type=int, default=0)
    p.add_argument("--max-step", type=int, default=0)
    p.add_argument("--max-checkpoints", type=int, default=0)
    p.add_argument(
        "--select-by",
        choices=[
            "acc",
            "f1_macro",
            "f1_weighted",
            "auroc_macro",
            "auprc_macro",
            "aa_ce",
        ],
        default="f1_macro",
    )
    p.add_argument(
        "--logits-source",
        choices=["mean_sample", "reduced", "all_samples"],
        default="mean_sample",
    )
    p.add_argument(
        "--mask-source", choices=["design", "aa_loss", "intersection"], default="design"
    )
    p.add_argument(
        "--model-stage",
        choices=["backbone_only", "joint"],
        default="backbone_only",
        help="Model config to instantiate for checkpoint loading. Sidechain is disabled either way.",
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
    p.add_argument("--seed", type=int, default=20260725)
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
        "--aa-input-source",
        default="diffusion_internal",
        choices=["s_inputs", "diffusion_internal"],
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
        build_components,
        build_configs,
        build_monomer_index,
        fill_missing_args,
    )

    # Flags `build_configs`/`build_components` read but this parser omits; filled
    # from the training parser's defaults (see `fill_missing_args`). Runs first so
    # `_apply_aa_eval_args`'s deliberate pins always win.
    fill_missing_args(args)
    _apply_aa_eval_args(args)
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
    print(f"logits_source={args.logits_source}")
    print(f"mask_source={args.mask_source}")
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
    if hasattr(configs.training, "checkpoint_include_prefixes"):
        configs.training.checkpoint_include_prefixes = []
    trainer = PXDesignTrainer(
        configs=configs,
        components=eval_components,
        device=device,
        checkpoint_dir=None,
        load_checkpoint_path=None,
        checkpoint_params_only=True,
    )

    aa_names = _aa_names(Path(repo_root), args.pxdesign_code_dir)
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for i, checkpoint in enumerate(checkpoints, start=1):
        row, per_class = evaluate_checkpoint(trainer, loader, checkpoint, args)
        for c in per_class:
            c["class_name"] = aa_names[int(c["class_idx"])]
        rows.append(row)
        class_rows.extend(per_class)
        print(
            f"[{i}/{len(checkpoints)}] step={row['step']} "
            f"acc={row['acc']:.4f} f1_macro={row['f1_macro']:.4f} "
            f"auroc_macro={row['auroc_macro']:.4f} auprc_macro={row['auprc_macro']:.4f} "
            f"aa_ce={row['aa_ce']:.4f} n_tokens={row['n_tokens']}",
            flush=True,
        )

    import pandas as pd

    metrics_csv = output_dir / "aa_head_checkpoint_metrics.csv"
    ranking_csv = output_dir / "aa_head_checkpoint_ranking.csv"
    per_class_csv = output_dir / "aa_head_per_class_metrics.csv"
    plot_path = output_dir / "aa_head_metrics.png"
    best_json = output_dir / "best_aa_head_checkpoint.json"

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(metrics_csv, index=False)
    ranked = df.sort_values(args.select_by, ascending=(args.select_by == "aa_ce"))
    ranked.to_csv(ranking_csv, index=False)
    pd.DataFrame(class_rows).sort_values(["step", "class_idx"]).to_csv(
        per_class_csv, index=False
    )
    best = _best_row(rows, args.select_by)
    best_json.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    plot_metrics(rows, plot_path)

    print(f"best_select_by={args.select_by}")
    print(f"best_step={best['step']}")
    print(f"best_checkpoint={best['checkpoint']}")
    print(f"wrote_metrics={metrics_csv}")
    print(f"wrote_ranking={ranking_csv}")
    print(f"wrote_per_class={per_class_csv}")
    print(f"wrote_best={best_json}")
    print(f"wrote_plot={plot_path}")


if __name__ == "__main__":
    main()
