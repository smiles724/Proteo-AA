#!/usr/bin/env python3
"""Compare inference-style and native-AA backbone validation on PINDER binders.

Only PINDER's converted binder chain B is designed and scored.  The receptor
chain A remains conditioning context.  The paired conditions differ only in
whether binder residue identities are fully masked (inference_style) or left
visible (native_aa); binder side-chain coordinates are scrubbed in both.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


CONDITIONS = {
    "inference_style": "all",
    "native_aa": "none",
}
AA3 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def _build_validation_manifest(args: argparse.Namespace) -> tuple[Path, int]:
    """Stream the large PINDER index into a small, crop-valid held-out index."""
    source = Path(args.pinder_index_csv).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() / "cache" / (
        f"pinder_{args.split}_binder_crop{args.crop_size}.csv.gz"
    )
    if output.is_file() and not args.rebuild_manifest:
        with gzip.open(output, "rt", newline="") as handle:
            return output, sum(1 for _ in csv.DictReader(handle))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    kept = 0
    required = {
        "pinder_id",
        "pdb_path",
        "converted_binder_chain",
        "source_split",
        "num_tokens",
        "binder_tokens",
    }
    with gzip.open(source, "rt", newline="") as src, gzip.open(
        temporary, "wt", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"PINDER index is missing columns: {sorted(missing)}")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["source_split"] != args.split:
                continue
            try:
                n_token = int(float(row["num_tokens"]))
                n_binder = int(float(row["binder_tokens"]))
            except (TypeError, ValueError):
                continue
            if not (args.min_n_token <= n_token <= args.max_n_token):
                continue
            if n_binder <= 0 or n_binder > int(
                args.crop_size * args.max_binder_fraction
            ):
                continue
            if row["converted_binder_chain"] != "B":
                continue
            writer.writerow(row)
            kept += 1
    os.replace(temporary, output)
    return output, kept


def _make_dataset(args: argparse.Namespace, manifest: Path, condition: str):
    from pxdesign_train.runner.data import DesignSourceDataset
    from pxdesign_train.runner.pinder_provider import PinderPdbProvider

    limit = int(args.max_samples) if int(args.max_samples) > 0 else -1
    provider = PinderPdbProvider(
        manifest_path=manifest,
        pinder_root=args.pinder_root,
        cif_cache_dir=args.pinder_cif_cache,
        pdb_cache_dir=args.pinder_pdb_cache or None,
        archive_path=args.pinder_archive,
        split=args.split,
        limit=limit,
    )
    dataset = DesignSourceDataset(
        provider=provider,
        source_name=f"pinder_{args.split}_binder",
        crop_size=int(args.crop_size),
        max_binder_fraction=float(args.max_binder_fraction),
        hotspot_force_zero_prob=1.0,
        aa_mask_mode=CONDITIONS[condition],
        aa_mask_prob=1.0,
        aa_mask_min_prob=1.0 if condition == "inference_style" else 0.0,
        aa_mask_max_prob=1.0 if condition == "inference_style" else 0.0,
        compute_sidechain=False,
        # This scrubs binder side-chain coordinates in both arms.  Therefore
        # the paired comparison isolates visible vs masked binder AA identity.
        backbone_only_binder=True,
        max_crop_retries=int(args.max_crop_retries),
        seed=int(args.seed),
    )
    return provider, dataset


def _binder_atom_masks(feat: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    design_token = feat["design_token_mask"].bool()
    atom_to_token = feat["atom_to_token_idx"].long()
    binder_atom = design_token[atom_to_token]
    binder_ca = feat["eval_ca_atom_mask"].bool() & binder_atom
    binder_bb = feat["eval_backbone_atom_mask"].bool() & binder_atom
    return binder_ca, binder_bb


@torch.no_grad()
def _forward_metrics(trainer, batch: dict[str, Any], ca_lddt_score) -> dict[str, float]:
    feat_cpu = batch["input_feature_dict"]
    binder_token_cpu = batch["binder_token_mask"].bool()
    design_token_cpu = feat_cpu["design_token_mask"].bool()
    if not torch.equal(binder_token_cpu, design_token_cpu):
        raise AssertionError("design_token_mask is not exactly the PINDER binder mask")
    if int(design_token_cpu.sum()) <= 0:
        raise AssertionError("PINDER example has no designed binder tokens")
    if bool((feat_cpu["aa_loss_mask"].bool() & ~design_token_cpu).any()):
        raise AssertionError("AA mask extends outside the binder design region")

    batch = trainer._to_device(batch)
    feat = batch["input_feature_dict"]
    binder_ca, binder_bb = _binder_atom_masks(feat)
    coordinate_mask = batch["label_dict"]["coordinate_mask"]
    dtype = trainer._train_precision()
    ctx = (
        torch.autocast("cuda", dtype=dtype, cache_enabled=False)
        if trainer.device.type == "cuda"
        else nullcontext()
    )
    with ctx:
        out = trainer.model(
            input_feature_dict=feat,
            label_dict=batch["label_dict"],
            mode="train",
        )
        binder_ca_lddt = ca_lddt_score(
            out["x_denoised"],
            out["x_gt_aug"],
            coordinate_mask,
            binder_ca,
        )
        binder_ca_rmsd, binder_tm = trainer.loss_fn._aligned_rmsd_and_tm(
            out["x_denoised"],
            out["x_gt_aug"],
            coordinate_mask,
            binder_ca,
            compute_tm=True,
        )
        binder_bb_rmsd, _ = trainer.loss_fn._aligned_rmsd_and_tm(
            out["x_denoised"],
            out["x_gt_aug"],
            coordinate_mask,
            binder_bb,
            compute_tm=False,
        )

        # The main Stage-III AA head predicts from B_pre's diffusion-internal
        # token representation, before the side-chain/refinement pass.  Score it
        # only where the dataset masked a valid native binder identity.  The
        # receptor is explicitly excluded by the assertions above.
        aa_logits = out.get("aa_logits")
        aa_clean = feat["aa_clean"].long()
        aa_valid = feat["aa_loss_mask"].bool() & design_token_cpu.to(feat["aa_loss_mask"].device)
        aa_valid = aa_valid & aa_clean.ne(-100)
        if aa_logits is None:
            raise AssertionError("checkpoint/model produced no aa_logits")
        if aa_logits.dim() == aa_clean.dim() + 2:
            n_sample = aa_logits.shape[-3]
            pre, n_token = aa_clean.shape[:-1], aa_clean.shape[-1]
            aa_clean_eval = aa_clean.reshape(*pre, 1, n_token).expand(
                *pre, n_sample, n_token
            )
            aa_valid_eval = aa_valid.reshape(*pre, 1, n_token).expand(
                *pre, n_sample, n_token
            )
        else:
            aa_clean_eval = aa_clean
            aa_valid_eval = aa_valid
        if not aa_valid_eval.any():
            raise AssertionError("PINDER example has no valid masked binder AA labels")
        logp = torch.log_softmax(aa_logits.float(), dim=-1)
        target = aa_clean_eval.clamp_min(0)
        ce_token = -logp.gather(-1, target[..., None]).squeeze(-1)
        aa_n = int(aa_valid_eval.sum().item())
        aa_ce_sum = float(ce_token[aa_valid_eval].sum().detach().cpu())
        aa_correct = int(
            ((aa_logits.argmax(dim=-1) == aa_clean_eval) & aa_valid_eval).sum().item()
        )
        aa_pred = aa_logits.argmax(dim=-1)[aa_valid_eval].long()
        aa_target = aa_clean_eval[aa_valid_eval].long()
        aa_confusion = torch.bincount(
            aa_target * len(AA3) + aa_pred,
            minlength=len(AA3) * len(AA3),
        ).reshape(len(AA3), len(AA3)).cpu().numpy()

    aa_mask = feat["aa_loss_mask"].bool()
    design_token = feat["design_token_mask"].bool()
    return {
        "binder_ca_lddt": _to_float(binder_ca_lddt),
        "binder_ca_rmsd": _to_float(binder_ca_rmsd),
        "binder_bb_rmsd": _to_float(binder_bb_rmsd),
        "binder_tm_score": _to_float(binder_tm),
        "n_binder_tokens": float(design_token.sum().item()),
        "n_target_tokens": float((~design_token).sum().item()),
        "binder_aa_mask_fraction": _to_float(
            (aa_mask & design_token).sum().float() / design_token.sum().clamp_min(1)
        ),
        "binder_aa_ce": aa_ce_sum / aa_n,
        "binder_aa_acc": aa_correct / aa_n,
        "binder_aa_n_predictions": float(aa_n),
        "_aa_confusion": aa_confusion,
    }


def _summarize_aa_confusion(
    confusion: np.ndarray, condition: str
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Return collapse-sensitive aggregate and per-class sequence metrics."""
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    total = int(confusion.sum())
    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for idx, name in enumerate(AA3):
        precision = (
            float(true_positive[idx] / predicted[idx]) if predicted[idx] else 0.0
        )
        recall = float(true_positive[idx] / support[idx]) if support[idx] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        rows.append(
            {
                "condition": condition,
                "class_idx": idx,
                "class_name": name,
                "support": int(support[idx]),
                "predicted": int(predicted[idx]),
                "true_positive": int(true_positive[idx]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_fraction": float(support[idx] / total) if total else 0.0,
                "prediction_fraction": float(predicted[idx] / total) if total else 0.0,
            }
        )
    summary: dict[str, float | int] = {
        "binder_aa_macro_f1": float(np.mean(f1s)),
        "binder_aa_balanced_acc": float(np.mean(recalls)),
        "binder_aa_num_predicted_classes": int((predicted > 0).sum()),
    }
    return summary, rows


def _evaluate_condition(
    trainer,
    args: argparse.Namespace,
    manifest: Path,
    condition: str,
    ca_lddt_score,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _seed_everything(int(args.seed))
    provider, dataset = _make_dataset(args, manifest, condition)
    from pxdesign_train.runner.trainer import _identity_collate

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_identity_collate,
    )
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    aa_ce_weighted_sum = 0.0
    aa_correct_sum = 0.0
    aa_prediction_count = 0.0
    aa_confusion = np.zeros((len(AA3), len(AA3)), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for i, batch in enumerate(loader):
        metrics = _forward_metrics(trainer, batch, ca_lddt_score)
        aa_confusion += metrics.pop("_aa_confusion")
        row = {
            "condition": condition,
            "aa_mask_mode": CONDITIONS[condition],
            "sample_index": i,
            "pinder_id": provider._pinder_ids[i],
            **metrics,
        }
        rows.append(row)
        for key, value in metrics.items():
            if key in {"binder_aa_ce", "binder_aa_acc", "binder_aa_n_predictions"}:
                continue
            if math.isfinite(value):
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        n_aa = metrics["binder_aa_n_predictions"]
        aa_ce_weighted_sum += metrics["binder_aa_ce"] * n_aa
        aa_correct_sum += metrics["binder_aa_acc"] * n_aa
        aa_prediction_count += n_aa
        if (i + 1) % int(args.log_interval) == 0 or i == 0:
            print(
                f"condition={condition} completed={i + 1}/{len(dataset)} "
                f"binder_ca_lddt={metrics['binder_ca_lddt']:.4f} "
                f"binder_aa_ce={metrics['binder_aa_ce']:.4f} "
                f"binder_aa_acc={metrics['binder_aa_acc']:.4f}",
                flush=True,
            )

    aggregate: dict[str, Any] = {
        "condition": condition,
        "aa_mask_mode": CONDITIONS[condition],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "source": "PINDER binder-design only (converted chain B)",
        "n_eval": len(rows),
        "seed": int(args.seed),
        "crop_size": int(args.crop_size),
        "binder_sidechain_coordinates_scrubbed": True,
    }
    aggregate.update(
        {key: sums[key] / max(1, counts[key]) for key in sorted(sums)}
    )
    aggregate.update(
        {
            "binder_aa_ce": aa_ce_weighted_sum / max(1.0, aa_prediction_count),
            "binder_aa_acc": aa_correct_sum / max(1.0, aa_prediction_count),
            "binder_aa_n_predictions": int(aa_prediction_count),
        }
    )
    class_summary, class_rows = _summarize_aa_confusion(aa_confusion, condition)
    aggregate.update(class_summary)
    return aggregate, rows, class_rows


def _build_configs(args: argparse.Namespace, device: torch.device):
    training_dir = Path(__file__).resolve().parent.parent / "training"
    sys.path.insert(0, str(training_dir))
    from train_protenix_monomer import build_configs

    config_args = argparse.Namespace(
        seed=args.seed,
        dtype=args.dtype,
        crop_size=args.crop_size,
        max_steps=1,
        lr=2e-4,
        warmup_steps=0,
        log_interval=0,
        eval_interval=0,
        checkpoint_interval=0,
        ema_decay=0.0,
        num_workers=0,
        iters_to_accumulate=1,
        grad_clip_norm=0.0,
        aa_mask_mode="all",
        aa_mask_prob=1.0,
        aa_mask_min_prob=0.0,
        aa_mask_max_prob=1.0,
        aa_input_source="diffusion_internal",
        trunk_grad_scale=1.0,
        disable_aa_loss=True,
        disable_sidechain=True,
        enable_coevolution=False,
        predicted_frame=True,
        per_sigma=True,
        template_provider="dunbrack_mode",
        disable_template_init=True,
        sc_trunk_grad_scale=1.0,
        sc_ablation_arm="default",
        sc_frame_aware_head=None,
        sc_centre_coord_input=None,
        sc_template_residual=None,
        training_stage="backbone_only",
    )
    return build_configs(config_args, device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default=(
            "/hai/scratch/yfsun/proteo_aa_runs/stage2_complex_backbone/"
            "from_monomer_step96000_protenix_pinder/checkpoints/step50000.pt"
        ),
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=["inference_style", "native_aa"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--pinder-root", default="/hai/scratch/yfsun/pinder/2024-02")
    p.add_argument(
        "--pinder-index-csv",
        default="/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.csv.gz",
    )
    p.add_argument(
        "--pinder-cif-cache", default="/hai/scratch/yfsun/pinder/2024-02/cif_cache"
    )
    p.add_argument("--pinder-pdb-cache", default="")
    p.add_argument(
        "--pinder-archive", default="/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip"
    )
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=1536)
    p.add_argument("--max-binder-fraction", type=float, default=0.75)
    p.add_argument("--max-crop-retries", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rebuild-manifest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    training_dir = Path(__file__).resolve().parent.parent / "training"
    sys.path.insert(0, str(training_dir))
    from train_protenix_monomer import _bootstrap_paths

    _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, n_manifest = _build_validation_manifest(args)
    print(f"checkpoint={checkpoint}")
    print(f"validation_manifest={manifest}")
    print(f"validation_split={args.split}")
    print(f"eligible_binder_design_complexes={n_manifest}")
    print("binder_chain=B target_chain=A")
    if args.dry_run:
        for condition in args.conditions:
            provider, dataset = _make_dataset(args, manifest, condition)
            item = dataset[0]
            design = item["input_feature_dict"]["design_token_mask"].bool()
            aa_mask = item["input_feature_dict"]["aa_loss_mask"].bool()
            print(
                f"dry_run condition={condition} pinder_id={provider._pinder_ids[0]} "
                f"binder_tokens={int(design.sum())} total_tokens={design.numel()} "
                f"masked_binder_tokens={int((aa_mask & design).sum())}"
            )
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no GPU is available")
    device = torch.device(args.device)
    from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents
    from select_best_checkpoint_protenix import ca_lddt_score

    # Trainer initialization needs a dataset/schedule, but validation below uses
    # condition-specific loaders directly.
    _, first_dataset = _make_dataset(args, manifest, args.conditions[0])
    multi = CurriculumMultiDataset(
        datasets=[first_dataset],
        source_names=["pinder_binder_val"],
        per_item_weights=[[1.0] * len(first_dataset)],
    )
    schedule = CurriculumSchedule(
        stage1={"pinder_binder_val": 1.0},
        stage2={"pinder_binder_val": 1.0},
        stage1_end_step=0,
        stage2_start_step=0,
    )
    trainer = PXDesignTrainer(
        configs=_build_configs(args, device),
        components=TrainerComponents(
            train_dataset=multi,
            schedule=schedule,
            train_samples_per_epoch=1,
        ),
        device=device,
        checkpoint_dir=None,
    )
    trainer.load_checkpoint(str(checkpoint), params_only=True)
    trainer.model.eval()

    aggregates: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        aggregate, rows, per_class = _evaluate_condition(
            trainer, args, manifest, condition, ca_lddt_score
        )
        aggregates.append(aggregate)
        sample_rows.extend(rows)
        class_rows.extend(per_class)
        print(json.dumps(aggregate, sort_keys=True), flush=True)

    aggregate_csv = output_dir / "binder_input_comparison.csv"
    sample_csv = output_dir / "binder_input_comparison_per_complex.csv.gz"
    class_csv = output_dir / "binder_aa_per_class.csv"
    summary_json = output_dir / "binder_input_comparison.json"
    with aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with gzip.open(sample_csv, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    with class_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    payload: dict[str, Any] = {"conditions": aggregates}
    by_name = {row["condition"]: row for row in aggregates}
    if {"inference_style", "native_aa"}.issubset(by_name):
        inf = by_name["inference_style"]
        native = by_name["native_aa"]
        payload["paired_difference_inference_minus_native"] = {
            key: inf[key] - native[key]
            for key in (
                "binder_ca_lddt",
                "binder_ca_rmsd",
                "binder_bb_rmsd",
                "binder_tm_score",
            )
        }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote_aggregate={aggregate_csv}")
    print(f"wrote_per_complex={sample_csv}")
    print(f"wrote_per_class={class_csv}")
    print(f"wrote_summary={summary_json}")


if __name__ == "__main__":
    main()
