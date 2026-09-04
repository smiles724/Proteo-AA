#!/usr/bin/env python3
"""Evaluate AA readouts and final structure during true PINDER cogeneration.

Native coordinates and labels are retained on CPU until generation completes.
The model starts from Gaussian coordinate noise with every binder identity masked;
one trajectory supplies final, target-sigma, and per-token confidence readouts.
The final generated coordinates are scored on the binder chain with the same
per-complex, binder-aligned geometry metrics as the PINDER backbone evaluator.
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


@torch.no_grad()
def _free_generation_structure_metrics(
    *,
    generated_coordinate: torch.Tensor,
    native_coordinate: torch.Tensor,
    coordinate_mask: torch.Tensor,
    binder_ca_mask: torch.Tensor,
    binder_backbone_mask: torch.Tensor,
    loss_fn: Any,
    ca_lddt_score: Any,
) -> dict[str, float | int]:
    """Score one final generated binder against its native structure.

    C-alpha/backbone RMSD and TM-score use a Kabsch alignment over the binder
    atoms selected by the corresponding metric mask.  C-alpha lDDT is based on
    internal distances and needs no superposition.  These metrics measure the
    generated binder fold; they do not measure its receptor-relative pose.
    """
    pred = generated_coordinate.detach().float()
    native = native_coordinate.detach().to(device=pred.device, dtype=torch.float32)
    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(
            "generated coordinate must have shape [N_atom, 3], "
            f"got {tuple(pred.shape)}"
        )
    if native.shape != pred.shape:
        raise ValueError(
            "generated/native coordinate shape mismatch: "
            f"{tuple(pred.shape)} versus {tuple(native.shape)}"
        )

    resolved = coordinate_mask.detach().to(device=pred.device).bool()
    ca_mask = binder_ca_mask.detach().to(device=pred.device).bool()
    bb_mask = binder_backbone_mask.detach().to(device=pred.device).bool()
    if resolved.shape != pred.shape[:-1]:
        raise ValueError(
            "coordinate mask shape mismatch: "
            f"{tuple(resolved.shape)} versus {tuple(pred.shape[:-1])}"
        )
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(native).all(dim=-1)
    resolved = resolved & finite
    n_ca = int((resolved & ca_mask).sum().item())
    n_bb = int((resolved & bb_mask).sum().item())
    if n_ca < 3 or n_bb < 3:
        raise ValueError(
            f"Need at least 3 resolved binder atoms (C-alpha={n_ca}, backbone={n_bb})"
        )

    # The shared metric implementations expect an explicit diffusion-sample
    # dimension: [N_sample, N_atom, 3].  Free generation returns one final sample.
    pred_eval = pred.unsqueeze(0)
    native_eval = native.unsqueeze(0)
    binder_ca_lddt = ca_lddt_score(
        pred_eval,
        native_eval,
        resolved,
        ca_mask,
    )
    binder_ca_rmsd, binder_tm = loss_fn._aligned_rmsd_and_tm(
        pred_eval,
        native_eval,
        resolved,
        ca_mask,
        compute_tm=True,
    )
    binder_bb_rmsd, _ = loss_fn._aligned_rmsd_and_tm(
        pred_eval,
        native_eval,
        resolved,
        bb_mask,
        compute_tm=False,
    )
    return {
        "binder_ca_lddt": _to_float(binder_ca_lddt),
        "binder_ca_rmsd": _to_float(binder_ca_rmsd),
        "binder_bb_rmsd": _to_float(binder_bb_rmsd),
        "binder_tm_score": _to_float(binder_tm),
        "n_binder_ca": n_ca,
        "n_binder_backbone_atoms": n_bb,
    }


def _summarize_structure_metrics(
    rows: list[dict[str, Any]], checkpoint: str
) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {
        "checkpoint": checkpoint,
        "coordinate_source": "gaussian_noise_final",
        "metric_scope": "binder",
        "alignment": "per_complex_binder_kabsch",
        "pose_metrics_included": False,
        "n_completed": len(rows),
        "n_binder_ca_total": int(sum(int(row["n_binder_ca"]) for row in rows)),
        "n_binder_backbone_atoms_total": int(
            sum(int(row["n_binder_backbone_atoms"]) for row in rows)
        ),
    }
    for key in (
        "binder_ca_lddt",
        "binder_ca_rmsd",
        "binder_bb_rmsd",
        "binder_tm_score",
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
        summary[f"{key}_median"] = (
            float(np.median(finite)) if finite.size else float("nan")
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--pinder-root", default="/hai/scratch/yfsun/pinder/2024-02")
    p.add_argument(
        "--pinder-index-csv",
        default="/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.csv.gz",
    )
    p.add_argument("--pinder-cif-cache", default="/hai/scratch/yfsun/pinder/2024-02/cif_cache")
    p.add_argument("--pinder-pdb-cache", default="")
    p.add_argument("--pinder-archive", default="/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip")
    p.add_argument("--crop-size", type=int, default=448)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=1536)
    p.add_argument("--max-binder-fraction", type=float, default=0.75)
    p.add_argument("--max-crop-retries", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--n-step", type=int, default=20)
    p.add_argument("--aa-readout-sigma", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    p.add_argument("--rebuild-manifest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_step < 2:
        raise ValueError("--n-step must be >= 2")
    evaluation_dir = Path(__file__).resolve().parent
    training_dir = evaluation_dir.parent / "training"
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(training_dir))

    from eval_aa_head_protenix_monomer import _classification_metrics
    from eval_pinder_binder_backbone_inputs import (
        _binder_atom_masks,
        _build_configs,
        _build_validation_manifest,
        _make_dataset,
    )
    from select_best_checkpoint_protenix import ca_lddt_score
    from export_protenix_backbones_for_mpnn import _to_device
    from train_protenix_monomer import _bootstrap_paths, fill_missing_args

    fill_missing_args(args)
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
    manifest, n_manifest = _build_validation_manifest(args)
    provider, dataset = _make_dataset(args, manifest, "inference_style")
    print(f"repo_root={repo_root}")
    print(f"checkpoint={checkpoint}")
    print(f"eligible_validation_rows={n_manifest}")
    print(f"evaluation_rows={len(dataset)}")
    print(f"n_step={args.n_step}")
    print(f"aa_readout_sigma={args.aa_readout_sigma}")
    print("coordinate_source=gaussian_noise")
    print("gt_coordinates_passed_to_model=false")
    print("structure_metric_scope=binder")
    print("structure_alignment=per_complex_binder_kabsch")
    print("receptor_relative_pose_metrics=false")
    if args.dry_run:
        item = dataset[0]
        feat = item["input_feature_dict"]
        print(f"sample_id={provider._pinder_ids[0]}")
        print(f"tokens={feat['design_token_mask'].numel()}")
        print(f"binder_tokens={int(feat['design_token_mask'].sum())}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no GPU is available")
    device = torch.device(args.device)
    from pxdesign_train.cogenerate import cogenerate
    from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents

    multi = CurriculumMultiDataset(
        datasets=[dataset],
        source_names=["pinder_binder_val"],
        per_item_weights=[[1.0] * len(dataset)],
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
            train_dataset=multi, schedule=schedule, train_samples_per_epoch=1
        ),
        device=device,
        checkpoint_dir=None,
    )
    trainer.load_checkpoint(str(checkpoint), params_only=True)
    trainer.model.eval()
    precision = trainer._train_precision()

    modes = ("final", "target_sigma", "confidence_best")
    all_probs: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    all_labels: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    per_sample: list[dict[str, Any]] = []
    structure_per_sample: list[dict[str, Any]] = []
    trajectories = output_dir / "inference_trajectories.jsonl"
    with trajectories.open("w") as trajectory_handle:
        for idx in range(len(dataset)):
            _seed_everything(int(args.seed) + idx)
            try:
                batch = dataset[idx]
                feat_cpu = batch["input_feature_dict"]
                labels = feat_cpu["aa_clean"].detach().cpu().long()
                design = feat_cpu["design_token_mask"].detach().cpu().bool()
                native_coordinate = batch["label_dict"]["coordinate"].detach().cpu()
                coordinate_mask = batch["label_dict"]["coordinate_mask"].detach().cpu()
                binder_ca_mask, binder_backbone_mask = _binder_atom_masks(feat_cpu)
                input_features = _to_device(feat_cpu, device)
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
                        aa_readout_mode="final",
                        aa_readout_sigma=float(args.aa_readout_sigma),
                    )
                structure_metrics = _free_generation_structure_metrics(
                    generated_coordinate=generated["coordinate"],
                    native_coordinate=native_coordinate,
                    coordinate_mask=coordinate_mask,
                    binder_ca_mask=binder_ca_mask,
                    binder_backbone_mask=binder_backbone_mask,
                    loss_fn=trainer.loss_fn,
                    ca_lddt_score=ca_lddt_score,
                )
                valid_base = design & (labels >= 0) & (labels < 20)
                if not bool(valid_base.any()):
                    raise ValueError("No valid native binder AA labels")
                # DesignSourceDataset may retry a different provider row when the
                # requested complex cannot be cropped safely.  Prefer the identity
                # attached to the item that was actually returned.
                sample_id = str(
                    batch.get("sample_id", provider._pinder_ids[idx])
                )
                structure_per_sample.append(
                    {
                        "sample_id": sample_id,
                        "dataset_index": idx,
                        **structure_metrics,
                    }
                )
                for mode in modes:
                    readout = generated["aa_readouts"][mode]
                    pred = readout["sequence"].detach().cpu().long()
                    probs = readout["probs"].detach().float().cpu()
                    valid = valid_base & (pred >= 0) & (pred < 20)
                    sample_probs = probs[valid].numpy()
                    sample_labels = labels[valid].numpy()
                    sample_pred = pred[valid].numpy()
                    all_probs[mode].append(sample_probs)
                    all_labels[mode].append(sample_labels)
                    per_sample.append(
                        {
                            "sample_id": sample_id,
                            "dataset_index": idx,
                            "readout": mode,
                            "readout_sigma": readout["sigma"],
                            "n_tokens": int(valid.sum()),
                            "recovery": float((sample_pred == sample_labels).mean()),
                            "mean_confidence": float(sample_probs.max(axis=-1).mean()),
                        }
                    )
                trajectory_handle.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "dataset_index": idx,
                            "structure": structure_metrics,
                            "trajectory": generated["trajectory"],
                        }
                    )
                    + "\n"
                )
                print(f"[{idx + 1}/{len(dataset)}] {sample_id}", flush=True)
            except Exception as exc:
                logging.exception("Inference failed for index %d: %s", idx, exc)

    summaries = []
    per_class_rows = []
    for mode in modes:
        if not all_probs[mode]:
            continue
        probs = np.concatenate(all_probs[mode], axis=0)
        labels = np.concatenate(all_labels[mode], axis=0)
        metrics, class_rows = _classification_metrics(probs, labels)
        rows = [row for row in per_sample if row["readout"] == mode]
        metrics.update(
            {
                "readout": mode,
                "target_sigma": float(args.aa_readout_sigma),
                "n_completed": len(rows),
                "mean_per_complex_recovery": float(
                    np.mean([float(row["recovery"]) for row in rows])
                ),
                "checkpoint": str(checkpoint),
            }
        )
        summaries.append(metrics)
        for row in class_rows:
            row["readout"] = mode
            per_class_rows.append(row)

    if not summaries:
        raise SystemExit("ERROR: no PINDER inference samples completed")
    structure_summary = _summarize_structure_metrics(
        structure_per_sample, str(checkpoint)
    )
    structure_summary.update(
        {
            "n_step": int(args.n_step),
            "seed": int(args.seed),
            "n_requested": len(dataset),
            "n_failed": len(dataset) - len(structure_per_sample),
        }
    )
    payload = {"readouts": summaries, "structure": structure_summary}
    (output_dir / "inference_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "inference_summary.csv", summaries)
    _write_csv(output_dir / "inference_per_sample.csv", per_sample)
    _write_csv(output_dir / "inference_per_class.csv", per_class_rows)
    _write_csv(output_dir / "inference_structure_summary.csv", [structure_summary])
    _write_csv(output_dir / "inference_structure_per_sample.csv", structure_per_sample)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote_summary={output_dir / 'inference_summary.json'}")


if __name__ == "__main__":
    main()
