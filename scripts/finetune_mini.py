#!/usr/bin/env python3
"""Mini fine-tune experiment from the official PXDesign-d checkpoint.

Loads pxdesign_v0.1.0.pt (load_strict=False; AA head is new), evaluates on a
fixed batch BEFORE and AFTER a short s_inputs fine-tune, and reports:
  aa_ce, aa_acc, aa_mask_frac, coord mse/lddt/distogram,
  sampler mask_frac trajectory + mean_conf/entropy, held(=train)-structure AA
  recovery, and sequence diversity (unique fraction / mean per-position entropy).

NOTE: single shipped structure (5o45) -> this is a train-structure pre/post
(memorization-flavored). Multi-structure held-out is a follow-up. s_inputs only;
diffusion_internal is NOT used here.
"""
from __future__ import annotations
import argparse, json, time
import torch


def build(args, device):
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import DesignSourceDataset, TrainerComponents
    from pxdesign_train.runner.cif_provider import CifFileProvider
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs
    from pxdesign_train.runner.trainer import PXDesignTrainer

    provider = CifFileProvider(cif_paths=[args.cif], binder_chain_ids=[args.binder_chain])
    src = DesignSourceDataset(
        provider, source_name="cif", crop_size=args.crop_size,
        hotspot_force_zero_prob=0.0,
        aa_mask_mode="time_dependent", aa_mask_min_prob=0.15, aa_mask_max_prob=1.0,
        seed=0,
    )
    multi = CurriculumMultiDataset(datasets=[src], source_names=["cif"], per_item_weights=[[1.0]])
    schedule = CurriculumSchedule(stage1={"cif": 1.0}, stage2={"cif": 1.0},
                                  stage1_end_step=0, stage2_start_step=0)
    components = TrainerComponents(train_dataset=multi, schedule=schedule,
                                   train_samples_per_epoch=args.max_steps)

    configs = parse_configs(training_configs, arg_str="")
    configs.residue_type.input_source = args.aa_input_source
    configs.residue_type.mask_mode = "time_dependent"
    configs.residue_type.mask_min_prob = 0.15
    configs.dtype = args.dtype
    configs.training.lr = args.lr
    configs.training.max_steps = args.max_steps
    configs.training.warmup_steps = min(50, args.max_steps // 5)
    configs.training.log_interval = max(1, args.max_steps // 20)
    configs.training.eval_interval = 0
    configs.training.checkpoint_interval = 0
    configs.training.ema_decay = 0.0
    configs.training.iters_to_accumulate = 1
    configs.training.num_workers = 0
    configs.load_strict = False
    configs.loss.align_before_mse = (device.type == "cuda")
    trainer = PXDesignTrainer(configs=configs, components=components, device=device)
    return trainer


@torch.no_grad()
def evaluate(trainer, batch, k_coord=4, sample_steps=8, n_div=6):
    from pxdesign_train.sampler import sample_residue_types
    rm = trainer.raw_model
    rm.eval()
    feat = batch["input_feature_dict"]

    # --- AA metrics + coord, averaged over k forwards.
    # (s_inputs AA is deterministic; diffusion_internal AA is sigma/sample-noisy,
    #  so averaging matters there.)
    acc = {"aa_ce": [], "aa_acc": [], "aa_mask_frac": [], "mse": [], "lddt": [], "distogram": []}
    for _ in range(k_coord):
        lo = trainer.forward_loss(batch)
        for kk in acc:
            acc[kk].append(float(lo[kk]))
    out = {kk: sum(v) / len(v) for kk, v in acc.items()}

    # --- sampler (s_inputs cheap path only; internal needs a coord forward) ---
    sampler_out = {"sampler_recovery": None, "sampler_mask_frac_traj": None,
                   "sampler_mean_conf": None, "sampler_mean_entropy": None,
                   "seq_unique_frac": None, "seq_mean_pos_entropy": None}
    try:
        dtm = feat["design_token_mask"]
        while dtm.dim() > 1:
            dtm = dtm.squeeze(0)
        dtm = dtm.bool()
        sampled, traj = sample_residue_types(rm, feat, dtm, n_steps=sample_steps, temperature=0.0)
        ac = feat.get("aa_clean")
        while ac is not None and ac.dim() > 1:
            ac = ac.squeeze(0)
        if ac is not None:
            valid = dtm & (ac != -100).to(dtm.device)
            if valid.any():
                sampler_out["sampler_recovery"] = float(
                    (sampled[valid].cpu() == ac[valid].cpu()).float().mean())
        seqs = []
        for _ in range(n_div):
            s, _ = sample_residue_types(rm, feat, dtm, n_steps=sample_steps, temperature=1.0)
            seqs.append(tuple(s[dtm].cpu().tolist()))
        import math
        from collections import Counter
        uniq_frac = len(set(seqs)) / len(seqs) if seqs else 0.0
        ent = 0.0
        if dtm.sum().item() and seqs:
            cols = list(zip(*seqs))
            for col in cols:
                c = Counter(col); tot = len(col)
                ent += -sum((n / tot) * math.log(n / tot) for n in c.values())
            ent /= len(cols)
        sampler_out.update(
            sampler_recovery=sampler_out["sampler_recovery"],
            sampler_mask_frac_traj=[round(d["mask_frac"], 3) for d in traj],
            sampler_mean_conf=round(sum(d["mean_conf"] for d in traj) / len(traj), 3) if traj else None,
            sampler_mean_entropy=round(sum(d["mean_entropy"] for d in traj) / len(traj), 3) if traj else None,
            seq_unique_frac=round(uniq_frac, 3),
            seq_mean_pos_entropy=round(ent, 3),
        )
    except NotImplementedError:
        sampler_out["sampler_note"] = "skipped (input_source=diffusion_internal needs a coord forward)"

    return {**out, **sampler_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--binder_chain", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aa_input_source", default="s_inputs",
                    choices=["s_inputs", "diffusion_internal"])
    ap.add_argument("--out", default="mini_experiment.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    t0 = time.time()
    trainer = build(args, device)
    print(f"trainer built {time.time()-t0:.0f}s")
    trainer.load_checkpoint(args.ckpt, params_only=True)

    # Fixed eval batch (identical masking pre/post), moved to the model device.
    batch = trainer._to_device(next(iter(trainer.train_dl)))
    print("\n=== PRE (loaded ckpt, fresh AA head) ===")
    pre = evaluate(trainer, batch)
    print(json.dumps(pre, indent=2))

    print(f"\n=== fine-tune {args.max_steps} steps (lr={args.lr}) ===")
    trainer.run(max_steps=args.max_steps)

    print("\n=== POST ===")
    post = evaluate(trainer, batch)
    print(json.dumps(post, indent=2))

    result = {"args": vars(args), "pre": pre, "post": post}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print("\n=== PRE/POST summary ===")
    for k in ["aa_ce", "aa_acc", "sampler_recovery", "sampler_mean_conf",
              "seq_unique_frac", "seq_mean_pos_entropy", "mse", "lddt"]:
        print(f"  {k:22s} pre={pre.get(k)}  post={post.get(k)}")
    print("MINI EXPERIMENT DONE")


if __name__ == "__main__":
    main()
