#!/usr/bin/env python3
"""Leakage-safe AA-head evaluation on native validation backbones.

This is deliberately different from free monomer generation.  The native
sequence is kept outside the model input, while the native N/CA/C/O geometry is
used so sequence recovery remains a meaningful metric.  For the strict input,
side-chain atoms are removed from the AtomArray *before* Protenix featurization
and every residue is featurized as the same GLY backbone template before its
sequence token is changed to PXDesign's masked ``xpb`` token.

The default ablations are:

* ``full_topology_scrub``: current featurization, with side-chain coordinates
  collapsed to CA but native atom counts/names/reference metadata retained;
* ``strict_native``: re-featurized uniform N/CA/C/O topology with native BB;
* ``strict_random``: the same strict topology with geometry destroyed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

AA1 = list("ARNDCQEGHILKMFPSTWYV")
AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
BACKBONE = ("N", "CA", "C", "O")
N_AA = 20

# These are supervision/evaluation products, not inference inputs.
SUPERVISION_KEYS = {
    "aa_clean",
    "aa_loss_mask",
    "eval_ca_atom_mask",
    "eval_backbone_atom_mask",
    "backbone_loss_mask",
    "design_sidechain_atom_mask",
    "sc_gt_local",
    "sc_atom_mask",
    "sc_atom_name_ids",
    "sc_frame_R",
    "sc_frame_t",
    "sc_bb_coords",
    "sc_bb_atom_idx",
    "sc_token_center_idx",
}


class FixedNoiseSampler:
    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, size: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.full(size, self.sigma, device=device, dtype=torch.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _crop_source_item(src, idx: int):
    atom_array, token_array, feat, label, binder_selector = src.provider[idx]
    selection = binder_selector(atom_array)
    if isinstance(selection, str):
        crop = src._cropper.crop(atom_array, token_array, binder_chain_id=selection)
    else:
        crop = src._cropper.crop(atom_array, token_array, binder_atom_mask=selection)
    return atom_array, token_array, feat, label, crop


def _native_labels(crop) -> torch.Tensor:
    from pxdesign.data.constants import PRO_STD_RESIDUES_NATURAL

    rep = crop.atom_array.distogram_rep_atom_mask.astype(bool)
    labels = []
    for name in np.asarray(crop.atom_array.res_name)[rep]:
        value = int(PRO_STD_RESIDUES_NATURAL.get(str(name), -100))
        labels.append(value if 0 <= value < N_AA else -100)
    if len(labels) != len(crop.token_array):
        raise ValueError(
            f"representative atoms ({len(labels)}) != tokens ({len(crop.token_array)})"
        )
    return torch.tensor(labels, dtype=torch.long)


def _design_selection(src, binder_atom_mask: np.ndarray, idx: int):
    from pxdesign_train.data.featurizer import DesignSelection

    return DesignSelection(
        binder_atom_mask=binder_atom_mask,
        hotspot_radius=src.hotspot_radius,
        hotspot_max_frac=src.hotspot_max_frac,
        hotspot_force_zero_prob=1.0,
        aa_mask_mode="all",
        aa_mask_prob=1.0,
        aa_mask_min_prob=1.0,
        aa_mask_max_prob=1.0,
        compute_sidechain=False,
        backbone_only_binder=True,
        rng=np.random.default_rng((src.seed + idx) % (2**32)),
    )


def _full_topology_input(src, original, idx: int) -> tuple[dict, dict]:
    """Ablation B: preserve native atom topology but scrub SC coordinates."""
    from pxdesign_train.data.featurizer import DesignFeaturizer
    from pxdesign_train.runner.data import _slice_feature_dict, _slice_label_dict

    atom_array, token_array, feat, label, crop = original
    feat = _slice_feature_dict(feat, atom_array, token_array, crop)
    label = _slice_label_dict(label, atom_array, token_array, crop)
    selection = _design_selection(src, crop.binder_atom_mask, idx)
    model_feat, model_label, _ = DesignFeaturizer(selection).transform(
        crop.atom_array.copy(), feat, label
    )
    return model_feat, model_label


def _strict_backbone_input(
    src, crop, native_labels: torch.Tensor, idx: int
) -> tuple[dict, dict, torch.Tensor]:
    """Ablation C: remove SC atoms, then rebuild every Protenix atom feature."""
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.tokenizer import AtomArrayTokenizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from pxdesign_train.data.featurizer import DesignFeaturizer
    from pxdesign_train.runner.data import canonicalise_backbone_reference_metadata

    aa = crop.atom_array
    keep = np.zeros(len(aa), dtype=bool)
    selected_tokens: list[int] = []
    for token_idx, token in enumerate(crop.token_array):
        if not (0 <= int(native_labels[token_idx]) < N_AA):
            continue
        by_name = {str(name): int(atom_idx) for name, atom_idx in zip(token.atom_names, token.atom_indices)}
        if not all(name in by_name for name in BACKBONE):
            continue
        selected_tokens.append(token_idx)
        for name in BACKBONE:
            keep[by_name[name]] = True
    bb = aa[keep].copy()
    if len(bb) == 0:
        raise ValueError("strict backbone AtomArray is empty")

    # Uniform residue identity before featurization is load-bearing: ref_pos,
    # ref_charge, ref_element, ref atom names and atom counts must all come from
    # one shared template rather than being inherited from the native residue.
    #
    # SHARED with the training path on purpose. This harness and
    # DesignSourceDataset differ in residue-SELECTION policy (this one drops an
    # unusable residue and reports which tokens survived; training raises), but
    # if they ever disagreed about WHICH per-atom channels get erased, the number
    # this script reports would not describe the pipeline that actually trains.
    canonicalise_backbone_reference_metadata(bb, np.ones(len(bb), dtype=bool))
    bb.mol_atom_index[:] = np.arange(len(bb), dtype=bb.mol_atom_index.dtype)
    synthetic_residue_index = np.repeat(np.arange(len(selected_tokens)), len(BACKBONE))
    if synthetic_residue_index.size != len(bb):
        raise ValueError("backbone atom ordering is not four contiguous atoms per selected token")
    bb.res_id[:] = synthetic_residue_index + 1
    if "label_seq_id" in bb.get_annotation_categories():
        bb.label_seq_id[:] = synthetic_residue_index + 1
    bb.ref_space_uid[:] = synthetic_residue_index.astype(bb.ref_space_uid.dtype)
    bb.mol_id[:] = 0
    bb.entity_mol_id[:] = 0
    # Nonstandard native residues can be atom-tokenized and therefore carry
    # several centre flags.  The synthetic protein representation must have
    # exactly one representative (CA) per residue.
    bb.centre_atom_mask[:] = 0
    bb.centre_atom_mask[np.asarray(bb.atom_name) == "CA"] = 1
    bb.distogram_rep_atom_mask[:] = 0
    bb.distogram_rep_atom_mask[np.asarray(bb.atom_name) == "CA"] = 1
    tokens = AtomArrayTokenizer(bb).get_token_array()
    if len(tokens) != len(selected_tokens):
        raise ValueError(f"strict tokens={len(tokens)} selected native tokens={len(selected_tokens)}")

    base = Featurizer(
        cropped_token_array=tokens,
        cropped_atom_array=bb,
        ref_pos_augment=False,
        lig_atom_rename=False,
    )
    feat = base.get_all_input_features()
    label = base.get_labels()
    feat = make_dummy_feature(features_dict=feat, dummy_feats=["msa", "template"])
    feat = data_type_transform(feat_or_label_dict=feat)
    label = data_type_transform(feat_or_label_dict=label)
    feat["is_distillation"] = torch.tensor([False])

    binder = np.ones(len(bb), dtype=bool)
    selection = _design_selection(src, binder, idx)
    model_feat, model_label, _ = DesignFeaturizer(selection).transform(bb, feat, label)

    counts = torch.bincount(
        model_feat["atom_to_token_idx"].long(), minlength=len(tokens)
    )
    if not torch.all(counts == 4):
        raise ValueError(f"strict input must have 4 atoms/token, got {counts.tolist()}")
    return model_feat, model_label, torch.tensor(selected_tokens, dtype=torch.long)


def _sanitize_model_features(feat: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in feat.items() if k not in SUPERVISION_KEYS}
    n_token = int(out["token_index"].shape[0])
    out["design_token_mask"] = torch.ones(n_token, dtype=torch.long)
    out["condition_token_mask"] = torch.zeros(n_token, dtype=torch.long)
    out["aa_t"] = torch.tensor(1.0, dtype=torch.float32)
    out["aa_mask_prob"] = torch.tensor(1.0, dtype=torch.float32)
    out["aa_corruption_mask"] = torch.ones(n_token, dtype=torch.long)
    for key in ("conditional_templ", "conditional_templ_mask", "hotspot", "plddt"):
        if key in out:
            out[key] = torch.zeros_like(out[key])
    for key in ("msa", "has_deletion", "deletion_value", "profile", "deletion_mean"):
        if key in out:
            out[key] = torch.zeros_like(out[key])
    # These are derived from atom/token topology inside model._train_forward.
    for key in ("relp", "d_lm", "v_lm", "pad_info"):
        out.pop(key, None)
    if "aa_clean" in out:
        raise AssertionError("native AA labels crossed the model-input boundary")
    return out


def _decode_atom_names(encoded: torch.Tensor) -> list[str]:
    codes = encoded.argmax(dim=-1).detach().cpu().numpy() + 32
    return ["".join(chr(int(value)) for value in row).strip() for row in codes]


def _complete_backbone_mask(feat: dict, label: dict) -> torch.Tensor:
    """Return tokens with resolved N, CA, C and O for either atom topology."""
    n_token = int(feat["token_index"].shape[0])
    atom_to_token = feat["atom_to_token_idx"].detach().cpu().long()
    names = _decode_atom_names(feat["ref_atom_name_chars"])
    resolved = label["coordinate_mask"].detach().cpu().bool()
    counts = torch.zeros(n_token, dtype=torch.long)
    for atom_idx, name in enumerate(names):
        if name in BACKBONE and bool(resolved[atom_idx]):
            counts[int(atom_to_token[atom_idx])] += 1
    return counts == len(BACKBONE)


def _classification_metrics(probs: np.ndarray, labels: np.ndarray):
    from eval_aa_head_protenix_monomer import _classification_metrics as impl

    return impl(probs, labels)


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    conditions = sorted({str(row["condition"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for condition in conditions:
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: float(row["sigma"]),
        )
        x = [float(row["sigma"]) for row in selected]
        axes[0].plot(x, [float(row["acc"]) for row in selected], marker="o", label=condition)
        axes[1].plot(x, [float(row["f1_macro"]) for row in selected], marker="o", label=condition)
    for ax, title, ylabel in zip(axes, ("AA recovery", "Macro F1"), ("accuracy", "macro F1")):
        ax.set_xscale("log")
        ax.set_xlabel("fixed diffusion noise sigma")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("AA head: topology-leakage validation ablations")
    fig.savefig(output, dpi=200)
    plt.close(fig)


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
    p.add_argument("--max-crop-retries", type=int, default=64)
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full validation set")
    p.add_argument("--sigmas", default="0.04,0.4,4.0")
    p.add_argument(
        "--conditions",
        default="full_topology_scrub,strict_native,strict_random",
        help="Comma-separated subset of full_topology_scrub,strict_native,strict_random",
    )
    p.add_argument("--random-coordinate-scale", type=float, default=10.0)
    p.add_argument("--diffusion-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    p.add_argument("--model-stage", choices=["aa_head_warmup", "joint"], default="joint")

    # Shared config/data builder arguments.
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
    p.add_argument("--sc-ablation-arm", default="no")
    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--use-template", action="store_true")
    p.add_argument("--ref-pos-augment", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sigmas = [float(value) for value in args.sigmas.split(",") if value.strip()]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    allowed = {"full_topology_scrub", "strict_native", "strict_random"}
    if not sigmas or any(sigma <= 0 for sigma in sigmas):
        raise ValueError("--sigmas must contain positive values")
    if not conditions or not set(conditions) <= allowed:
        raise ValueError(f"--conditions must be a subset of {sorted(allowed)}")

    evaluation_dir = Path(__file__).resolve().parent
    training_dir = evaluation_dir.parent / "training"
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(training_dir))
    from train_protenix_monomer import (
        _bootstrap_paths,
        _recent_index_path,
        apply_training_stage_args,
        build_components,
        build_configs,
        build_monomer_index,
    )

    args.training_stage = args.model_stage
    args.data_mode = "monomer"
    apply_training_stage_args(args)
    repo_root = _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    source_index = Path(args.source_index).resolve() if args.source_index else _recent_index_path(data_root)
    filtered_index = (
        Path(args.filtered_index).resolve()
        if args.filtered_index
        else output_dir / "cache" / "recentPDB_strict_monomer_validation.csv.gz"
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
    stop = n_items if int(args.max_samples) <= 0 else min(n_items, start + int(args.max_samples))
    if start < 0 or stop <= start or start >= n_items:
        raise ValueError(f"invalid interval [{start},{stop}) for {n_items} validation rows")

    print(f"repo_root={repo_root}")
    print(f"checkpoint={checkpoint}")
    print(f"validation_index={filtered_index}")
    print(f"validation_rows={n_items}")
    print(f"evaluation_interval=[{start},{stop})")
    print(f"conditions={conditions}")
    print(f"sigmas={sigmas}")
    print("native_labels_passed_to_model=false")
    print("strict_sidechains_removed_before_featurization=true")

    # A CPU-only dry run exercises the critical topology construction.
    if args.dry_run:
        original = _crop_source_item(source_dataset, start)
        labels = _native_labels(original[-1])
        strict_feat, strict_label, strict_tokens = _strict_backbone_input(
            source_dataset, original[-1], labels, start
        )
        clean = _sanitize_model_features(strict_feat)
        print(f"native_tokens={labels.numel()}")
        print(f"strict_tokens={strict_tokens.numel()}")
        print(f"n_atoms={strict_label['coordinate'].shape[-2]}")
        print(f"atoms_per_token={torch.bincount(clean['atom_to_token_idx']).unique().tolist()}")
        print(f"model_has_aa_clean={'aa_clean' in clean}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents

    configs = build_configs(args, device)
    configs.training.diffusion_batch_size = int(args.diffusion_samples)
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

    aggregate: dict[tuple[str, float], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"probs": [], "labels": []}
    )
    per_sample: list[dict[str, Any]] = []
    all_native_labels: list[np.ndarray] = []

    for idx in range(start, stop):
        _seed_everything(int(args.seed) + idx)
        try:
            original = _crop_source_item(source_dataset, idx)
            crop = original[-1]
            native_labels = _native_labels(crop)
            valid_label = (native_labels >= 0) & (native_labels < N_AA)
            all_native_labels.append(native_labels[valid_label].numpy())

            prepared: dict[str, tuple[dict, dict, torch.Tensor]] = {}
            if "full_topology_scrub" in conditions:
                full_feat, full_label = _full_topology_input(source_dataset, original, idx)
                prepared["full_topology_scrub"] = (full_feat, full_label, native_labels)
            if {"strict_native", "strict_random"} & set(conditions):
                strict_feat, strict_label, strict_tokens = _strict_backbone_input(
                    source_dataset, crop, native_labels, idx
                )
                strict_pair = (
                    strict_feat,
                    strict_label,
                    native_labels[strict_tokens],
                )
                if "strict_native" in conditions:
                    prepared["strict_native"] = strict_pair
                if "strict_random" in conditions:
                    prepared["strict_random"] = strict_pair

            for condition in conditions:
                raw_feat, raw_label, condition_labels = prepared[condition]
                model_feat = _sanitize_model_features(raw_feat)
                label = {
                    "coordinate": raw_label["coordinate"].clone(),
                    "coordinate_mask": raw_label["coordinate_mask"].clone(),
                }
                if condition == "strict_random":
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(int(args.seed) + 1000003 * idx)
                    label["coordinate"] = torch.randn(
                        label["coordinate"].shape,
                        generator=generator,
                        dtype=label["coordinate"].dtype,
                    ) * float(args.random_coordinate_scale)

                n_model_token = int(model_feat["token_index"].shape[0])
                if n_model_token != condition_labels.numel():
                    raise ValueError(f"model tokens={n_model_token}, labels={condition_labels.numel()}")
                condition_valid_label = (condition_labels >= 0) & (condition_labels < N_AA)
                valid = condition_valid_label & _complete_backbone_mask(raw_feat, raw_label)
                if not valid.any():
                    raise ValueError("no residues with a complete resolved N/CA/C/O backbone")

                model_feat = _to_device(model_feat, device)
                label_device = _to_device(label, device)
                for sigma in sigmas:
                    _seed_everything(int(args.seed) + idx + int(round(sigma * 10000)))
                    trainer.model.training_noise_sampler = FixedNoiseSampler(sigma)
                    ctx = (
                        torch.autocast("cuda", dtype=precision, cache_enabled=False)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with torch.no_grad(), ctx:
                        out = trainer.model(
                            input_feature_dict=model_feat,
                            label_dict=label_device,
                            mode="train",
                        )
                    logits = out["aa_logits"].detach().float().cpu()
                    if logits.dim() == native_labels.dim() + 2:
                        probs = logits.softmax(dim=-1).mean(dim=-3)
                    else:
                        probs = logits.softmax(dim=-1)
                    p = probs[valid].numpy()
                    y = condition_labels[valid].numpy()
                    pred = p.argmax(axis=-1)
                    recovery = float((pred == y).mean())
                    aggregate[(condition, sigma)]["probs"].append(p)
                    aggregate[(condition, sigma)]["labels"].append(y)
                    per_sample.append(
                        {
                            "dataset_index": idx,
                            "condition": condition,
                            "sigma": sigma,
                            "n_tokens": int(valid.sum()),
                            "recovery": recovery,
                            "mean_confidence": float(p.max(axis=-1).mean()),
                            "native_sequence": "".join(AA1[int(v)] for v in y),
                            "predicted_sequence": "".join(AA1[int(v)] for v in pred),
                        }
                    )
            print(f"[{idx-start+1}/{stop-start}] val_{idx:05d} complete", flush=True)
        except Exception as exc:
            logging.exception("validation item %d failed: %s", idx, exc)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for (condition, sigma), values in sorted(aggregate.items()):
        probs = np.concatenate(values["probs"], axis=0)
        labels = np.concatenate(values["labels"], axis=0)
        metrics, per_class = _classification_metrics(probs, labels)
        row = {
            "checkpoint": str(checkpoint),
            "condition": condition,
            "sigma": sigma,
            "n_completed": len({r["dataset_index"] for r in per_sample if r["condition"] == condition and r["sigma"] == sigma}),
            **metrics,
        }
        summary_rows.append(row)
        for class_row in per_class:
            class_row.update(
                {"condition": condition, "sigma": sigma, "class_name": AA3[int(class_row["class_idx"])]}
            )
            per_class_rows.append(class_row)

    if not summary_rows:
        raise SystemExit("ERROR: no validation ablation completed")
    native = np.concatenate(all_native_labels)
    frequencies = np.bincount(native, minlength=N_AA)
    prior_acc = float(frequencies.max() / frequencies.sum())
    metadata = {
        "checkpoint": str(checkpoint),
        "validation_index": str(filtered_index),
        "n_validation_rows": n_items,
        "n_requested": stop - start,
        "conditions": conditions,
        "sigmas": sigmas,
        "empirical_majority_class": AA3[int(frequencies.argmax())],
        "empirical_majority_accuracy": prior_acc,
        "native_labels_passed_to_model": False,
        "strict_sidechains_removed_before_featurization": True,
        "strict_atoms_per_token": 4,
    }
    _write_csv(output_dir / "strict_aa_summary.csv", summary_rows)
    _write_csv(output_dir / "strict_aa_per_sample.csv", per_sample)
    _write_csv(output_dir / "strict_aa_per_class.csv", per_class_rows)
    (output_dir / "strict_aa_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot(summary_rows, output_dir / "strict_aa_ablation.png")
    print(json.dumps(metadata, indent=2))
    for row in summary_rows:
        print(
            f"condition={row['condition']} sigma={row['sigma']} "
            f"acc={row['acc']:.4f} f1_macro={row['f1_macro']:.4f} n={row['n_tokens']}"
        )
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
