#!/usr/bin/env python3
"""Export Proteo-AA validation backbones for ProteinMPNN inverse folding.

This script exports strict monomer validation rows as backbone-only PDBs for
ProteinMPNN. By default it runs the Proteo-AA backbone model and selects one
denoised coordinate sample per item. With --coord-source gt it exports the
ground-truth backbone coordinates directly. The PDB residue names are dummy ALA
by default to avoid leaking the native sequence to ProteinMPNN. The GT sequence
is stored only in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

AA3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA1 = list("ARNDCQEGHILKMFPSTWYV")
BACKBONE = ("N", "CA", "C", "O")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {k: move(x) for k, x in v.items()}
        return v

    return move(batch)


def _tensor_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def _choose_coordinate_sample(
    x_denoised: torch.Tensor,
    sigma: torch.Tensor,
    *,
    mode: str,
    sample_index: int,
) -> tuple[torch.Tensor, float, int]:
    x = _tensor_cpu(x_denoised)
    s = _tensor_cpu(sigma)
    if x.dim() == 4 and x.shape[0] == 1:
        x = x[0]
    if s.dim() == 2 and s.shape[0] == 1:
        s = s[0]
    if x.dim() == 2:
        return x, float("nan"), 0
    if x.dim() != 3:
        raise ValueError(
            f"Expected x_denoised [N_sample,N_atom,3], got {tuple(x.shape)}"
        )
    if mode == "lowest_sigma":
        idx = int(torch.argmin(s.reshape(-1)).item())
    elif mode == "first":
        idx = int(sample_index)
    else:
        raise ValueError(f"Unsupported sample_select={mode}")
    if idx < 0 or idx >= x.shape[0]:
        raise IndexError(f"sample_index {idx} outside N_sample={x.shape[0]}")
    sigma_value = float(s.reshape(-1)[idx].item()) if s.numel() else float("nan")
    return x[idx], sigma_value, idx


def _source_item_with_crop(src, idx: int):
    """Reproduce DesignSourceDataset._get_one(), but also return the crop."""
    from pxdesign_train.data.featurizer import DesignFeaturizer, DesignSelection
    from pxdesign_train.runner.data import _slice_feature_dict, _slice_label_dict

    atom_array, token_array, feat, label, binder_selector = src.provider[idx]
    sel = binder_selector(atom_array)
    if isinstance(sel, str):
        crop = src._cropper.crop(atom_array, token_array, binder_chain_id=sel)
    else:
        crop = src._cropper.crop(atom_array, token_array, binder_atom_mask=sel)
    feat = _slice_feature_dict(feat, atom_array, token_array, crop)
    label = _slice_label_dict(label, atom_array, token_array, crop)
    rng = np.random.default_rng((src.seed + idx) % (2**32))
    selection = DesignSelection(
        binder_atom_mask=crop.binder_atom_mask,
        hotspot_radius=src.hotspot_radius,
        hotspot_max_frac=src.hotspot_max_frac,
        hotspot_force_zero_prob=src.hotspot_force_zero_prob,
        aa_mask_mode=src.aa_mask_mode,
        aa_mask_prob=src.aa_mask_prob,
        aa_mask_min_prob=src.aa_mask_min_prob,
        aa_mask_max_prob=src.aa_mask_max_prob,
        compute_sidechain=src.compute_sidechain,
        backbone_only_binder=src.backbone_only_binder,
        rng=rng,
    )
    new_feat, new_label, _ = DesignFeaturizer(selection).transform(
        crop.atom_array,
        feat,
        label,
    )
    batch = {
        "input_feature_dict": new_feat,
        "label_dict": new_label,
        "binder_token_mask": torch.from_numpy(crop.binder_token_mask),
        "source_name": src.source_name,
    }
    return batch, crop


def _write_pdb(
    path: Path,
    *,
    coord: torch.Tensor,
    coordinate_mask: torch.Tensor,
    crop,
    aa_clean: torch.Tensor,
    design_token_mask: torch.Tensor,
    dummy_resname: str,
    min_backbone_atoms: int,
) -> tuple[str, int, int]:
    coord_np = coord.detach().cpu().numpy()
    cmask = coordinate_mask.detach().cpu().bool().numpy()
    aa = aa_clean.detach().cpu().long().numpy()
    design = design_token_mask.detach().cpu().bool().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    gt_seq: list[str] = []
    kept_res = 0
    skipped_res = 0
    atom_serial = 1
    with path.open("w") as fh:
        fh.write(
            "REMARK Proteo-AA predicted backbone for ProteinMPNN inverse folding\n"
        )
        for tok_i, token in enumerate(crop.token_array):
            if tok_i >= len(aa) or tok_i >= len(design):
                continue
            aa_idx = int(aa[tok_i])
            if not design[tok_i] or aa_idx < 0 or aa_idx >= 20:
                continue
            atom_by_name = {
                str(name).strip(): int(atom_idx)
                for name, atom_idx in zip(token.atom_names, token.atom_indices)
            }
            present = [
                name
                for name in BACKBONE
                if name in atom_by_name
                and atom_by_name[name] < len(cmask)
                and bool(cmask[atom_by_name[name]])
                and np.isfinite(coord_np[atom_by_name[name]]).all()
            ]
            if len(present) < int(min_backbone_atoms):
                skipped_res += 1
                continue
            kept_res += 1
            gt_seq.append(AA1[aa_idx])
            resname = dummy_resname if dummy_resname else AA3[aa_idx]
            chain_id = "A"
            for atom_name in BACKBONE:
                if atom_name not in present:
                    continue
                atom_idx = atom_by_name[atom_name]
                x, y, z = coord_np[atom_idx]
                element = atom_name[0]
                fh.write(
                    f"ATOM  {atom_serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
                    f"{kept_res:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {element:>2s}\n"
                )
                atom_serial += 1
        fh.write("TER\nEND\n")
    return "".join(gt_seq), kept_res, skipped_res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--coord-source",
        choices=["generated", "gt"],
        default="generated",
        help="Use Proteo-AA generated coordinates or ground-truth label coordinates.",
    )
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--source-index", default="")
    p.add_argument("--filtered-index", default="")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--sample-select", choices=["lowest_sigma", "first"], default="lowest_sigma"
    )
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument(
        "--dummy-resname",
        default="ALA",
        help="Use empty string to write GT residue names.",
    )
    p.add_argument("--min-backbone-atoms", type=int, default=4)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    p.add_argument(
        "--training-stage", choices=["backbone_only", "joint"], default="backbone_only"
    )

    # Args consumed by train_protenix_monomer.build_configs/build_components.
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
    p.add_argument("--dataset-limit", type=int, default=-1)
    p.add_argument("--aa-mask-mode", default="none")
    p.add_argument("--aa-mask-prob", type=float, default=0.0)
    p.add_argument("--aa-mask-min-prob", type=float, default=0.0)
    p.add_argument("--aa-mask-max-prob", type=float, default=0.0)
    p.add_argument("--aa-input-source", default="diffusion_internal")
    p.add_argument("--trunk-grad-scale", type=float, default=1.0)
    p.add_argument("--disable-aa-loss", action="store_true", default=True)
    p.add_argument("--disable-sidechain", action="store_true", default=True)
    p.add_argument("--enable-coevolution", action="store_true", default=False)
    p.add_argument(
        "--predicted-frame", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--per-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--disable-template-init", action="store_true")
    p.add_argument("--sc-trunk-grad-scale", type=float, default=1.0)
    p.add_argument("--sc-ablation-arm", default="default")
    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--use-template", action="store_true")
    p.add_argument(
        "--ref-pos-augment", action=argparse.BooleanOptionalAction, default=False
    )
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
    from pxdesign_train.runner.trainer import TrainerComponents

    _seed_everything(int(args.seed))
    apply_training_stage_args(args)
    repo_root = _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.coord_source == "generated" and not args.checkpoint:
        raise SystemExit("ERROR: --checkpoint is required when --coord-source generated.")
    if args.device == "cuda" and args.coord_source == "generated" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)

    out_dir = Path(args.output_dir).expanduser().resolve()
    pdb_dir = out_dir / "pdbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    source_index = (
        Path(args.source_index).resolve()
        if args.source_index
        else _recent_index_path(data_root)
    )
    filtered_index = (
        Path(args.filtered_index).resolve()
        if args.filtered_index
        else out_dir / "cache" / "recentPDB_monomer_validation_for_mpnn.csv.gz"
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
    src = components.train_dataset.datasets[0]

    trainer = None
    dtype = None
    if args.coord_source == "generated":
        from pxdesign_train.runner.trainer import PXDesignTrainer

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
        trainer.load_checkpoint(str(Path(args.checkpoint).resolve()), params_only=True)
        trainer.model.eval()
        dtype = trainer._train_precision()

    manifest_path = out_dir / "backbone_manifest.csv"
    fasta_path = out_dir / "gt_sequences.fasta"
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else n_items
    start = int(args.start_index)
    stop = min(n_items, start + max_samples)

    rows = []
    with torch.no_grad(), fasta_path.open("w") as fasta:
        for idx in range(start, stop):
            try:
                batch, crop = _source_item_with_crop(src, idx)
                if args.coord_source == "gt":
                    coord = batch["label_dict"]["coordinate"]
                    sigma_value = float("nan")
                    selected_sample = -1
                else:
                    assert trainer is not None and dtype is not None
                    batch_dev = _to_device(batch, device)
                    ctx = (
                        torch.autocast("cuda", dtype=dtype, cache_enabled=False)
                        if device.type == "cuda"
                        else torch.no_grad()
                    )
                    with ctx:
                        out = trainer.model(
                            input_feature_dict=batch_dev["input_feature_dict"],
                            label_dict=batch_dev["label_dict"],
                            mode="train",
                        )
                    coord, sigma_value, selected_sample = _choose_coordinate_sample(
                        out["x_denoised"],
                        out["sigma"],
                        mode=args.sample_select,
                        sample_index=int(args.sample_index),
                    )
                sample_id = f"val_{idx:05d}"
                pdb_path = pdb_dir / f"{sample_id}.pdb"
                gt_seq, n_res, n_skipped = _write_pdb(
                    pdb_path,
                    coord=coord,
                    coordinate_mask=batch["label_dict"]["coordinate_mask"],
                    crop=crop,
                    aa_clean=batch["input_feature_dict"]["aa_clean"],
                    design_token_mask=batch["input_feature_dict"]["design_token_mask"],
                    dummy_resname=str(args.dummy_resname),
                    min_backbone_atoms=int(args.min_backbone_atoms),
                )
                if n_res <= 0:
                    raise ValueError("No residues with complete backbone were exported")
                fasta.write(f">{sample_id}\n{gt_seq}\n")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_index": idx,
                        "pdb_path": str(pdb_path),
                        "gt_sequence": gt_seq,
                        "n_res": n_res,
                        "n_skipped_res": n_skipped,
                        "sigma": sigma_value,
                        "selected_sample": selected_sample,
                        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
                        "coord_source": args.coord_source,
                    }
                )
                if len(rows) % 25 == 0:
                    print(f"exported={len(rows)} last={sample_id}", flush=True)
            except Exception as exc:
                logging.exception("Failed sample idx=%d: %s", idx, exc)

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "dataset_index",
                "pdb_path",
                "gt_sequence",
                "n_res",
                "n_skipped_res",
                "sigma",
                "selected_sample",
                "checkpoint",
                "coord_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"repo_root={repo_root}")
    print(f"coord_source={args.coord_source}")
    print(f"validation_rows={n_items}")
    print(f"exported_rows={len(rows)}")
    print(f"manifest={manifest_path}")
    print(f"gt_fasta={fasta_path}")
    print(f"pdb_dir={pdb_dir}")
    if not rows:
        raise SystemExit(
            "ERROR: exported zero backbones; see log for failed sample diagnostics."
        )


if __name__ == "__main__":
    main()
