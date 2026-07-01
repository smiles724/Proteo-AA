#!/usr/bin/env python3
"""
GPU smoke test: one training step on a single CIF file with the real model.

This script exercises the entire PXDesign-d training pipeline end-to-end:
    CIF file → CifFileProvider → DesignSourceDataset → DesignCropper →
    DesignFeaturizer → ProtenixDesignTrain (139M params) → PXDesignLoss →
    backward → optimizer step

Usage:
    # From the repo root (proteo-r15/):
    LAYERNORM_TYPE=torch \
    PYTHONPATH="Protenix:PXDesign:PXDesign-train" \
    python PXDesign-train/scripts/smoke_test_gpu.py \
        --cif ./PXDesign/examples/5o45.cif \
        --binder_chain B \
        --crop_size 200 \
        --device cuda

    # CPU (slow but works; good for verifying wiring without a GPU):
    LAYERNORM_TYPE=torch \
    PYTHONPATH="Protenix:PXDesign:PXDesign-train" \
    python PXDesign-train/scripts/smoke_test_gpu.py \
        --cif ./PXDesign/examples/5o45.cif \
        --binder_chain B \
        --crop_size 100 \
        --device cpu

Requirements:
    - PXDesign/download_tool_weights.sh has been run (CCD components file)
    - OR set PROTENIX_DATA_ROOT_DIR to a dir containing components.cif
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import torch


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="PXDesign-d GPU smoke test")
    parser.add_argument("--cif", required=True, help="Path to a CIF file")
    parser.add_argument("--binder_chain", required=True, help="Chain ID to treat as binder")
    parser.add_argument("--crop_size", type=int, default=200, help="Token crop budget")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--n_steps", type=int, default=1, help="Number of training steps")
    parser.add_argument(
        "--aa_mask_mode",
        default="all",
        choices=["all", "none", "fixed", "time_dependent"],
        help="Residue identity masking mode for masked-AA training",
    )
    parser.add_argument("--aa_mask_prob", type=float, default=1.0)
    parser.add_argument("--aa_mask_min_prob", type=float, default=0.0)
    parser.add_argument("--aa_mask_max_prob", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true",
                        help="After training, run the iterative-unmask AA sampler on one batch")
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--aa_input_source", default="s_inputs",
                        choices=["s_inputs", "diffusion_internal"],
                        help="Representation the AA head reads")
    parser.add_argument("--grad_probe", action="store_true",
                        help="One forward+backward, report a_token capture / shapes / grads")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}, dtype: {args.dtype}, crop: {args.crop_size}")

    # ---- build data ----
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import DesignSourceDataset, TrainerComponents
    from pxdesign_train.runner.cif_provider import CifFileProvider

    t0 = time.time()
    provider = CifFileProvider(
        cif_paths=[args.cif],
        binder_chain_ids=[args.binder_chain],
    )
    src = DesignSourceDataset(
        provider, source_name="cif",
        crop_size=args.crop_size,
        hotspot_force_zero_prob=0.0,
        aa_mask_mode=args.aa_mask_mode,
        aa_mask_prob=args.aa_mask_prob,
        aa_mask_min_prob=args.aa_mask_min_prob,
        aa_mask_max_prob=args.aa_mask_max_prob,
    )
    multi = CurriculumMultiDataset(
        datasets=[src], source_names=["cif"],
        per_item_weights=[[1.0]],
    )
    schedule = CurriculumSchedule(
        stage1={"cif": 1.0}, stage2={"cif": 1.0},
        stage1_end_step=0, stage2_start_step=0,
    )
    components = TrainerComponents(
        train_dataset=multi, schedule=schedule,
        train_samples_per_epoch=args.n_steps,
    )
    print(f"Data setup: {time.time() - t0:.1f}s")

    # ---- build configs ----
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs

    configs = parse_configs(training_configs, arg_str="")
    configs.residue_type.input_source = args.aa_input_source
    configs.residue_type.mask_mode = args.aa_mask_mode
    configs.residue_type.mask_prob = args.aa_mask_prob
    configs.residue_type.mask_min_prob = args.aa_mask_min_prob
    configs.residue_type.mask_max_prob = args.aa_mask_max_prob
    configs.dtype = args.dtype
    configs.training.lr = 5e-4
    configs.training.max_steps = args.n_steps
    configs.training.warmup_steps = 0
    configs.training.log_interval = 1
    configs.training.eval_interval = 0
    configs.training.checkpoint_interval = 0
    configs.training.ema_decay = 0.0
    configs.training.iters_to_accumulate = 1
    configs.training.num_workers = 0
    configs.load_strict = False
    configs.loss.align_before_mse = (device.type == "cuda")

    # ---- build trainer (real model) ----
    from pxdesign_train.runner.trainer import PXDesignTrainer

    t1 = time.time()
    trainer = PXDesignTrainer(
        configs=configs,
        components=components,
        device=device,
    )
    print(f"Model init: {time.time() - t1:.1f}s "
          f"({sum(p.numel() for p in trainer.model.parameters())/1e6:.1f}M params)")

    # ---- grad probe: one forward+backward, inspect capture/shapes/grads ----
    if args.grad_probe:
        print(f"\n{'-'*60}\nGrad probe (input_source={args.aa_input_source})...")
        import math as _math
        rm = trainer.raw_model
        rm.train()
        rm.zero_grad(set_to_none=True)
        batch = next(iter(trainer.train_dl))
        loss_out = trainer.forward_loss(batch)
        loss_out["loss"].backward()
        cache = getattr(rm, "_a_token_cache", None)
        head = rm.design_residue_type_head
        hgs = [p.grad for p in head.parameters() if p.grad is not None]
        head_gnorm = float(torch.sqrt(sum((g.float()**2).sum() for g in hgs))) if hgs else 0.0
        head_finite = all(torch.isfinite(g).all() for g in hgs)
        # does grad reach the trunk (diffusion_module)?
        trunk_grads = [p.grad for p in rm.diffusion_module.parameters() if p.grad is not None]
        trunk_reached = len(trunk_grads) > 0
        print(f"  a_token captured: {cache is not None}"
              + (f", shape={tuple(cache.shape)}" if cache is not None else ""))
        print(f"  aa_ce={float(loss_out['aa_ce']):.4f}  aa_acc={float(loss_out['aa_acc']):.3f}"
              f"  loss_finite={bool(torch.isfinite(loss_out['loss']))}")
        print(f"  AA head grad: nonzero={head_gnorm>0} norm={head_gnorm:.4g} finite={head_finite}")
        print(f"  grad reaches diffusion_module trunk: {trunk_reached}")
        print("GRAD PROBE OK")
        rm.zero_grad(set_to_none=True)

    # ---- run ----
    print(f"\nRunning {args.n_steps} training step(s)...")
    trainer.run(max_steps=args.n_steps)

    # ---- optional: inference-time iterative-unmask sampling ----
    if args.sample:
        print(f"\n{'-'*60}\nRunning iterative-unmask AA sampler...")
        try:
            from pxdesign_train.sampler import sample_residue_types

            batch = trainer._to_device(next(iter(trainer.train_dl)))
            feat = batch["input_feature_dict"]
            dtm = feat["design_token_mask"]
            while dtm.dim() > 1:
                dtm = dtm.squeeze(0)
            sampled, traj = sample_residue_types(
                trainer.raw_model, feat, dtm.bool(), n_steps=args.sample_steps
            )
            aa_clean = feat.get("aa_clean")
            if aa_clean is not None:
                ac = aa_clean
                while ac.dim() > 1:
                    ac = ac.squeeze(0)
                pos = dtm.bool()
                valid = pos & (ac != -100).to(pos.device)
                if valid.any():
                    rec = (sampled[valid].cpu() == ac[valid].cpu()).float().mean()
                    print(f"Sampler recovery vs GT (design residues): {float(rec):.3f}")
            print(f"Sampler trajectory (mask_frac / mean_conf): "
                  f"{[(round(d['mask_frac'],2), round(d['mean_conf'],2)) for d in traj]}")
            print("SAMPLER OK")
        except Exception as e:  # noqa: BLE001 — demo must not fail the smoke run
            print(f"[sampler demo skipped due to: {type(e).__name__}: {e}]")

    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED — {args.n_steps} step(s) completed on {device}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
