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

    cifs = args.train_cifs.split(",") if args.train_cifs else [args.cif]
    chains = args.train_chains.split(",") if args.train_chains else [args.binder_chain]
    provider = CifFileProvider(cif_paths=cifs, binder_chain_ids=chains)
    src = DesignSourceDataset(
        provider, source_name="cif", crop_size=args.crop_size,
        hotspot_force_zero_prob=0.0,
        aa_mask_mode="time_dependent", aa_mask_min_prob=0.15, aa_mask_max_prob=1.0,
        seed=0,
    )
    multi = CurriculumMultiDataset(datasets=[src], source_names=["cif"],
                                   per_item_weights=[[1.0] * len(cifs)])
    schedule = CurriculumSchedule(stage1={"cif": 1.0}, stage2={"cif": 1.0},
                                  stage1_end_step=0, stage2_start_step=0)
    components = TrainerComponents(train_dataset=multi, schedule=schedule,
                                   train_samples_per_epoch=max(1, args.max_steps))

    configs = parse_configs(training_configs, arg_str="")
    configs.residue_type.input_source = args.aa_input_source
    configs.residue_type.trunk_grad_scale = args.trunk_grad_scale
    configs.residue_type.internal_reduce = args.internal_reduce
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


def make_batch_from_cif(trainer, cif, chain, crop_size):
    """Featurize one CIF into a device batch using the trainer's own collate."""
    from pxdesign_train.runner import DesignSourceDataset
    from pxdesign_train.runner.cif_provider import CifFileProvider
    from torch.utils.data import DataLoader
    prov = CifFileProvider(cif_paths=[cif], binder_chain_ids=[chain])
    ds = DesignSourceDataset(
        prov, source_name="heldout", crop_size=crop_size, hotspot_force_zero_prob=0.0,
        aa_mask_mode="time_dependent", aa_mask_min_prob=0.15, aa_mask_max_prob=1.0, seed=0)
    dl = DataLoader(ds, batch_size=1, collate_fn=trainer.train_dl.collate_fn)
    return trainer._to_device(next(iter(dl)))


@torch.no_grad()
def eval_aa(trainer, batch, k=10):
    """Average AA cross-entropy + accuracy on a batch's masked design residues."""
    ce, acc, frac = [], [], []
    for _ in range(k):
        lo = trainer.forward_loss(batch)
        ce.append(float(lo["aa_ce"])); acc.append(float(lo["aa_acc"]))
        frac.append(float(lo["aa_mask_frac"]))
    n = len(ce)
    return {"aa_ce": sum(ce) / n, "aa_acc": sum(acc) / n, "aa_mask_frac": sum(frac) / n}


@torch.no_grad()
def eval_coord_fixed_sigma(trainer, batch, sigmas=(1.0, 4.0, 16.0), n_seed=16):
    """Clean coordinate quality: denoise the GT structure at FIXED sigma over
    n_seed noise draws and report masked MSE per sigma. Removes the random-sigma
    noise that made earlier coord comparisons meaningless, so B (gradient into
    the trunk) vs s_inputs can be compared for structure degradation."""
    from protenix.model.protenix import update_input_feature_dict
    from protenix.model.utils import centre_random_augmentation
    rm = trainer.raw_model
    rm.eval()
    feat = dict(batch["input_feature_dict"])
    label = batch["label_dict"]
    feat = rm.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
    feat = update_input_feature_dict(feat)
    s_inputs, s, z = rm.get_condition_embedding(feat)
    gt = label["coordinate"]
    cmask = label["coordinate_mask"]
    batch_shape = gt.shape[:-2]
    device, dtype = gt.device, s_inputs.dtype
    out = {}
    for sig in sigmas:
        x_aug = centre_random_augmentation(gt, N_sample=n_seed, mask=cmask).to(dtype)
        sigma = torch.full((*batch_shape, n_seed), float(sig), device=device, dtype=dtype)
        x_noisy = x_aug + torch.randn_like(x_aug) * sigma[..., None, None]
        x_den = rm.diffusion_module(
            x_noisy=x_noisy, t_hat_noise_level=sigma, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_z=None, p_lm=None, c_l=None)
        se = ((x_den - x_aug) ** 2).sum(-1)          # [.., n_seed, N_atom]
        m = cmask.unsqueeze(-2)                        # [.., 1, N_atom]
        out[float(sig)] = float((se * m).sum() / m.expand_as(se).sum().clamp_min(1))
    return out


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
    ap.add_argument("--cif", default="")           # single-structure default; or use --train_cifs
    ap.add_argument("--binder_chain", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aa_input_source", default="s_inputs",
                    choices=["s_inputs", "diffusion_internal"])
    ap.add_argument("--trunk_grad_scale", type=float, default=1.0)
    ap.add_argument("--internal_reduce", default="mean", choices=["mean", "low_sigma"])
    ap.add_argument("--cogenerate", action="store_true",
                    help="Run #3 joint sequence-structure co-generation from noise (no fine-tune)")
    ap.add_argument("--cogen_steps", type=int, default=20)
    ap.add_argument("--coord_eval", action="store_true",
                    help="#4: fine-tune then report fixed-sigma coord MSE (structure-degradation check)")
    ap.add_argument("--train_cifs", default="", help="comma-separated CIFs for multi-structure training")
    ap.add_argument("--train_chains", default="", help="comma-separated binder chains (one per train CIF)")
    ap.add_argument("--heldout_cif", default="")
    ap.add_argument("--heldout_chain", default="")
    ap.add_argument("--heldout_eval", action="store_true",
                    help="#4 Part2: fine-tune on train_cifs, report AA recovery on a train vs held-out structure")
    ap.add_argument("--out", default="mini_experiment.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    t0 = time.time()
    trainer = build(args, device)
    print(f"trainer built {time.time()-t0:.0f}s")
    trainer.load_checkpoint(args.ckpt, params_only=True)

    # ---- #4 Part2: multi-structure train, AA recovery on train vs held-out ----
    if args.heldout_eval:
        import json as _json
        train_cif0 = (args.train_cifs.split(",")[0] if args.train_cifs else args.cif)
        train_chain0 = (args.train_chains.split(",")[0] if args.train_chains else args.binder_chain)
        train_batch = make_batch_from_cif(trainer, train_cif0, train_chain0, args.crop_size)
        held_batch = make_batch_from_cif(trainer, args.heldout_cif, args.heldout_chain, args.crop_size)
        pre_tr, pre_ho = eval_aa(trainer, train_batch), eval_aa(trainer, held_batch)
        if args.max_steps > 0:
            print(f"\n=== fine-tune {args.max_steps} steps on {len(args.train_cifs.split(',')) if args.train_cifs else 1} "
                  f"structure(s) (input_source={args.aa_input_source}) ===")
            trainer.run(max_steps=args.max_steps)
        post_tr, post_ho = eval_aa(trainer, train_batch), eval_aa(trainer, held_batch)
        print("\n=== AA recovery (aa_acc / aa_ce), pre -> post fine-tune ===")
        print(f"  TRAIN structure   : acc {pre_tr['aa_acc']:.3f}->{post_tr['aa_acc']:.3f} | "
              f"ce {pre_tr['aa_ce']:.3f}->{post_tr['aa_ce']:.3f}")
        print(f"  HELD-OUT structure: acc {pre_ho['aa_acc']:.3f}->{post_ho['aa_acc']:.3f} | "
              f"ce {pre_ho['aa_ce']:.3f}->{post_ho['aa_ce']:.3f}")
        with open(args.out, "w") as f:
            _json.dump({"args": vars(args), "train": {"pre": pre_tr, "post": post_tr},
                        "heldout": {"pre": pre_ho, "post": post_ho}}, f, indent=2)
        print(f"saved -> {args.out}\nHELDOUT EVAL OK")
        return

    # ---- #4: fixed-sigma coord eval (structure-degradation check), then exit ----
    if args.coord_eval:
        import json as _json
        batch = trainer._to_device(next(iter(trainer.train_dl)))
        pre = eval_coord_fixed_sigma(trainer, batch)
        if args.max_steps > 0:
            print(f"\n=== fine-tune {args.max_steps} steps (input_source={args.aa_input_source}, "
                  f"grad_scale={args.trunk_grad_scale}) ===")
            trainer.run(max_steps=args.max_steps)
        post = eval_coord_fixed_sigma(trainer, batch)
        print("\n=== fixed-sigma coord MSE (pre -> post fine-tune) ===")
        for sig in sorted(pre):
            print(f"  sigma={sig:>5}: {pre[sig]:.4f} -> {post[sig]:.4f}")
        with open(args.out, "w") as f:
            _json.dump({"args": vars(args), "pre": pre, "post": post}, f, indent=2)
        print(f"saved -> {args.out}\nCOORD EVAL OK")
        return

    # ---- #3: joint co-generation from noise (structure + sequence), then exit ----
    if args.cogenerate:
        from pxdesign_train.cogenerate import cogenerate
        if args.max_steps > 0:
            print(f"\n=== fine-tune {args.max_steps} steps (lr={args.lr}) before co-generation ===")
            trainer.run(max_steps=args.max_steps)
        batch = trainer._to_device(next(iter(trainer.train_dl)))
        feat = batch["input_feature_dict"]
        print(f"\n=== co-generate ({args.cogen_steps} steps, input_source={args.aa_input_source}) ===")
        res = cogenerate(trainer.raw_model, feat, N_step=args.cogen_steps)
        seq = res["sequence"]
        dtm = feat["design_token_mask"].bool()
        while dtm.dim() > 1:
            dtm = dtm.squeeze(0)
        gen = seq[dtm].cpu().tolist()
        print(f"coord shape: {tuple(res['coordinate'].shape)}")
        print(f"generated design seq (aa20 idx, n={len(gen)}): {gen}")
        print(f"distinct residues used: {len(set(gen))}")
        ac = feat.get("aa_clean")
        while ac is not None and ac.dim() > 1:
            ac = ac.squeeze(0)
        if ac is not None:
            valid = dtm & (ac != -100).to(dtm.device)
            if valid.any():
                rec = (seq[valid].cpu() == ac[valid].cpu()).float().mean()
                print(f"recovery vs GT: {float(rec):.3f}")
        print(f"trajectory (sigma/mask_frac/mean_conf): "
              f"{[(round(t['sigma'],1), round(t['mask_frac'],2), round(t['mean_conf'],2)) for t in res['trajectory']]}")
        print("COGENERATE OK")
        return

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
