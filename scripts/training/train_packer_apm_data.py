#!/usr/bin/env python3
"""Train the torsion packer on APM's data with APM's objective and schedule.

Everything that can be APM's own code is APM's own code: the featurisation
(`apm.data.datasets._process_csv_row_FAESM`), the loss (openfold's
`supervised_chi_loss` plus a transcription of `cal_sidechain_fape_loss` that
calls the same openfold functions), the length-bucketed cluster sampler
(`LengthBatcher_nonRep`, reimplemented for one replica), the filters, the
optimiser and the clip. What is ours is the network -- `sidechain/packer.py`,
which has been shown tensor-for-tensor and output-for-output equal to APM's
`SideChainModel` -- plus the sequence-conditioning arm being studied.

Deviations from APM's setup, all deliberate and all logged at startup:

  * one GPU with `--accum 8` instead of 8-way DDP, so the effective batch
    matches but the gradient is accumulated rather than all-reduced;
  * PDB monomers only. APM's pdb_dataset also sets `use_multimer: True` and
    `use_AFDB: True`. AFDB side chains are AlphaFold's predictions, so training
    a packer on them teaches it to reproduce AF2's packing rather than
    crystallography; multimer chains are excluded to keep train and the
    monomeric post-2021 validation set on the same footing.

Both deviations are identical across arms, so the arm comparison is unaffected.

    python scripts/training/train_packer_apm_data.py --arm none --out RUNDIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_ROOT = "/hai/scratch/yfsun/apm/extracted/data_APM"
APM_TEST_DIR = "/hai/scratch/shenjm/apm_weights/pdb_test"
APM_TEST_IDS = "/hai/scratch/shenjm/apm_weights/metadata_all/test_set_pdb_ids.csv"
ESM2_650M = "/hai/scratch/shenjm/plm_weights/esm2_t33_650M_UR50D.pt"


# --------------------------------------------------------------------------
# sampler: apm/data/protein_dataloader.py :: LengthBatcher_nonRep, one replica
# --------------------------------------------------------------------------
class LengthBatcher:
    """One sample per sequence cluster per epoch, batched by exact length.

    Exact-length grouping is why nothing in this script pads: every chain in a
    batch has the same number of residues, which is APM's arrangement too.
    """

    def __init__(self, csv, max_batch_size=64, max_num_res_squared=400_000,
                 seed=123, shuffle=True):
        self.csv = csv.reset_index(drop=True)
        self.csv["index"] = np.arange(len(self.csv))
        self.max_batch_size = int(max_batch_size)
        self.max_num_res_squared = int(max_num_res_squared)
        self.seed, self.shuffle, self.epoch = int(seed), bool(shuffle), 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        sub = self.csv.groupby("cluster").sample(1, random_state=self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(len(sub), generator=rng).tolist()
            sub = sub.iloc[order]
        batches = []
        for seq_len, g in sub.groupby("modeled_seq_len"):
            cropped = min(int(seq_len), 384)
            bs = min(self.max_batch_size, self.max_num_res_squared // cropped**2 + 1)
            for i in range(math.ceil(len(g) / bs)):
                idx = g.iloc[i * bs:(i + 1) * bs]["index"].tolist()
                # APM repeats a short batch up to the target size rather than
                # letting batch size wobble with cluster counts.
                batches.append(idx * max(1, bs // len(idx)))
        if self.shuffle:
            order = torch.randperm(len(batches),
                                   generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self):
        return len(list(iter(self)))


def collate(items):
    out = {}
    for k in items[0]:
        if k == "name":
            out[k] = [it[k] for it in items]
        else:
            out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


# --------------------------------------------------------------------------
def build_model(arm, plm_checkpoint, device):
    from pxdesign_train.sidechain.packer import TorsionPacker

    packer = TorsionPacker(
        c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
        ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=4, transformer_dropout=0.2, num_torsion_blocks=4,
        seq_cond=arm, embed_aatype=True, embed_rotvecs=True, use_mlp=True,
        embed_chain=True, random_torsion_input=True,
        plm_checkpoint=(plm_checkpoint if arm in ("plm", "both") else ""),
    ).to(device)
    return packer


def packer_inputs(batch, device):
    """APM's feature dict -> the packer's argument list."""
    from pxdesign_train.sidechain.instantiate import (STD_AA_3,
                                                      instantiate_from_type_indices)

    aat = batch["aatypes_1"].to(device).long()
    B, L = aat.shape
    n_aa = len(STD_AA_3)
    canonical = aat < n_aa
    # 21 wide, not 20: argmax inside the packer has to be able to return the UNK
    # index, and a 20-wide one-hot would silently turn every UNK into ALA.
    logits = torch.nn.functional.one_hot(aat.clamp(0, n_aa), n_aa + 1).float() * 40.0 - 20.0
    ids, chem = instantiate_from_type_indices(aat.clamp(0, n_aa - 1).cpu())
    ids, chem = ids.to(device), chem.to(device) & canonical[..., None]
    return dict(
        h_res=torch.zeros(B, L, 768, device=device),
        restype_logits=logits,
        atom_name_ids=ids,
        atom_mask=chem,
        noisy_coords=torch.zeros(B, L, chem.shape[-1], 3, device=device),
        t=torch.zeros(B, device=device),
        frame_R=batch["rotmats_1"].to(device).float(),
        frame_t=batch["trans_1"].to(device).float(),
        res_mask=batch["res_mask"].to(device).bool(),
        # APM's packing convention: every modelled residue is being generated,
        # GLY included. Passed explicitly; the packer's own default is the
        # side-chain-ownership mask our earlier runs used, which is not this.
        diffuse_mask=batch["diffuse_mask"].to(device).float(),
        residue_index=batch["res_idx"].to(device).long(),
        asym_id=batch["chain_idx"].to(device).long(),
    )


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------
@torch.no_grad()
def validate(packer, files, device, tables, limit=0):
    """chi1 accuracy, rotamer recovery and symmetry-aware atom14 side-chain RMSD.

    This is a *training-curve* metric computed in atom14 space off openfold's
    renamed ground truth. It is not the same estimator as
    `scripts/evaluation/eval_on_apm_testset.py`, which is the one every
    cross-model number in the write-up comes from. Numbers from here are only
    ever compared against other numbers from here.
    """
    from openfold.np.residue_constants import chi_pi_periodic
    from pxdesign_train.sidechain.apm_dataset import featurise
    from pxdesign_train.sidechain.apm_loss import predicted_atom14
    from openfold.utils.loss import compute_renamed_ground_truth

    packer.eval()
    periodic = torch.tensor(chi_pi_periodic, device=device).float()  # [21, 4]
    sq_err = n_atom = 0.0
    chi1_ok = chi1_n = rot_ok = rot_n = 0
    files = files[:limit] if limit else files
    for f in files:
        b = collate([featurise(f)])
        b = to_device(b, device)
        packer(**packer_inputs(b, device))
        unit = packer.last_torsions["unit"]

        _frames, pos = predicted_atom14(unit, b, tables)
        gt_keys = ("atom14_gt_positions", "atom14_alt_gt_positions", "atom14_gt_exists",
                   "atom14_atom_is_ambiguous", "atom14_alt_gt_exists")
        renamed = compute_renamed_ground_truth({k: b[k] for k in gt_keys}, pos)
        gt, exists = renamed["renamed_atom14_gt_positions"], renamed["renamed_atom14_gt_exists"]
        exists = exists.clone()
        exists[..., :4] = 0          # N, CA, C, O are backbone in every restype
        sq_err += ((pos - gt).square().sum(-1) * exists).sum().item()
        n_atom += exists.sum().item()

        pred_chi = torch.atan2(unit[..., 0], unit[..., 1])
        gt_chi, chi_mask = b["torsions_1"], b["torsions_mask"].bool()
        per = periodic[b["aatypes_1"].clamp(0, 20)].bool()
        d = pred_chi - gt_chi
        d = torch.atan2(d.sin(), d.cos()).abs()
        # A pi-periodic chi is indistinguishable from chi + pi, so fold it in.
        d = torch.where(per, torch.minimum(d, (math.pi - d).abs()), d)
        good = (d < math.radians(40.0)) | ~chi_mask
        chi1_ok += (good[..., 0] & chi_mask[..., 0]).sum().item()
        chi1_n += chi_mask[..., 0].sum().item()
        has_chi = chi_mask.any(-1)
        rot_ok += (good.all(-1) & has_chi).sum().item()
        rot_n += has_chi.sum().item()
    packer.train()
    return dict(atom14_symrmsd=math.sqrt(sq_err / max(n_atom, 1)),
                chi1_acc_40=chi1_ok / max(chi1_n, 1),
                rotamer_rec=rot_ok / max(rot_n, 1),
                n_chains=len(files))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["none", "a_token", "plm", "both"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--plm-checkpoint", default=ESM2_650M)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--accum", type=int, default=8, help="stands in for APM's 8 GPUs")
    # APM's sampler settings for 80GB GPUs (pretrain_sidechain.yaml). Exposed so
    # a smoke run can shrink them without editing the sampler.
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument("--max-num-res-squared", type=int, default=400_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=10, help="epochs")
    ap.add_argument("--val-n", type=int, default=100, help="0 = all 449")
    ap.add_argument("--time-limit-h", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt if it exists")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.arm in ("a_token", "both"):
        raise SystemExit(
            "a_token needs the frozen PXDesign trunk, which consumes a Protenix "
            "feature dict that APM's pickles do not carry. The feature bridge is "
            "a separate piece of work; this script trains none/plm only."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from pxdesign_train.sidechain.apm_dataset import (APMPackingDataset, PDB_FILTER,
                                                      post2021_val_files)
    from apm.data.datasets import _length_filter, _max_coil_filter, _rog_filter
    from pxdesign_train.sidechain.apm_loss import RigidGroupTables, packing_loss

    meta = pd.read_csv(Path(args.data_root) / "meta_data.csv", low_memory=False)
    meta = meta[meta["processed_path"].astype(str).str.contains("train_set")]
    n_raw = len(meta)
    meta = _length_filter(meta, PDB_FILTER["min_num_res"], PDB_FILTER["max_num_res"])
    meta = _max_coil_filter(meta, PDB_FILTER["max_coil_percent"])
    meta = _rog_filter(meta, PDB_FILTER["rog_quantile"])
    root = Path(args.data_root) / "pdb_monomer"
    meta = meta[[(root / f"{n}.pkl").is_file() for n in meta["pdb_name"].astype(str)]]
    meta = meta.reset_index(drop=True)
    files = [root / f"{n}.pkl" for n in meta["pdb_name"].astype(str)]

    train = APMPackingDataset(files, crop_size=None, seed=args.seed)
    sampler = LengthBatcher(meta, seed=args.seed,
                            max_batch_size=args.max_batch_size,
                            max_num_res_squared=args.max_num_res_squared)
    val = post2021_val_files(APM_TEST_DIR, APM_TEST_IDS)

    packer = build_model(args.arm, args.plm_checkpoint, device)
    trainable = [p for p in packer.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.95, 0.999))
    tables = RigidGroupTables(torch.float32, device)

    header = dict(arm=args.arm, params_total=n_par, train_chains=len(files),
                  filtered_out=n_raw - len(files), clusters=int(meta.cluster.nunique()),
                  batches_per_epoch=len(list(iter(sampler))), accum=args.accum,
                  val_chains=len(val), lr=args.lr, clip=args.clip,
                  max_batch_size=args.max_batch_size,
                  max_num_res_squared=args.max_num_res_squared,
                  max_epochs=args.max_epochs, seed=args.seed,
                  loss="supervised_chi_loss(1.0, norm 0.02) + sidechain_fape (APM, weight 1.0)",
                  diffuse_mask="APM packing convention: 1 for every modelled residue",
                  data="PDB monomers only (no AFDB, no multimer) -- see module docstring")
    print("CONFIG " + json.dumps(header), flush=True)
    (out / "config.json").write_text(json.dumps(header, indent=2))

    start_epoch, step = 0, 0
    if args.resume and (out / "last.pt").is_file():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if ck.get("config", {}).get("arm") not in (None, args.arm):
            raise SystemExit(f"{out/'last.pt'} is arm {ck['config']['arm']!r}, "
                             f"not {args.arm!r}")
        packer.load_state_dict(ck["state_dict"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        else:
            # Without the moments, resuming restarts Adam's bias correction and
            # the first steps take an effectively different step size. Say so
            # rather than letting the loss curve kink for no visible reason.
            print("RESUME warning: checkpoint has no optimizer state; "
                  "Adam moments restart from zero", flush=True)
        start_epoch, step = int(ck["epoch"]) + 1, int(ck["step"])
        print(f"RESUME from epoch {start_epoch}, step {step}", flush=True)

    log = (out / "train_log.jsonl").open("a")
    loader = torch.utils.data.DataLoader(
        train, batch_sampler=sampler, num_workers=args.workers, collate_fn=collate,
        pin_memory=True, persistent_workers=args.workers > 0)

    t0 = time.time()
    micro = 0
    epoch = start_epoch - 1          # defined even if the loop body never runs
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.max_epochs):
        sampler.set_epoch(epoch)
        run = {"chi": 0.0, "fape": 0.0, "total": 0.0, "n": 0}
        for batch in loader:
            b = to_device(batch, device)
            packer(**packer_inputs(b, device))
            tor = packer.last_torsions
            loss, parts = packing_loss(tor["unit"], tor["raw"], b, tables)
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} step {step}: {parts}")
            (loss / args.accum).backward()
            for k in ("chi", "fape", "total"):
                run[k] += float(parts[k].mean())
            run["n"] += 1
            micro += 1
            if micro % args.accum == 0:
                if step == 0:
                    # All four arms construct the same parameter set on purpose
                    # -- the ablation is meant to change information, not
                    # capacity -- so `params_total` is the same number for every
                    # arm and does not say how much of the model this arm
                    # actually trains. Report that separately, measured.
                    live = sum(p.numel() for p in trainable if p.grad is not None)
                    dead = sorted({n.split(".")[0] for n, p in packer.named_parameters()
                                   if p.requires_grad and p.grad is None})
                    print("PARAMS " + json.dumps(dict(
                        total=n_par, receiving_gradient=live, inert_modules=dead)),
                        flush=True)
                    header["params_receiving_gradient"] = live
                    header["params_inert_modules"] = dead
                    (out / "config.json").write_text(json.dumps(header, indent=2))
                gn = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    n = max(run["n"], 1)
                    rec = dict(epoch=epoch, step=step, chi=run["chi"] / n,
                               fape=run["fape"] / n, total=run["total"] / n,
                               grad_norm=float(gn), hours=(time.time() - t0) / 3600)
                    print("TRAIN " + json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    run = {"chi": 0.0, "fape": 0.0, "total": 0.0, "n": 0}

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.max_epochs:
            m = validate(packer, val, device, tables, limit=args.val_n)
            rec = dict(epoch=epoch, step=step, hours=(time.time() - t0) / 3600, **m)
            print("VAL " + json.dumps(rec), flush=True)
            log.write(json.dumps({"val": rec}) + "\n"); log.flush()
            # Optimizer state included so --resume continues the same run
            # rather than a differently-conditioned one.
            torch.save({"state_dict": packer.state_dict(),
                        "optimizer": opt.state_dict(), "epoch": epoch,
                        "step": step, "val": m, "config": header},
                       out / "last.pt")

        if args.time_limit_h and (time.time() - t0) / 3600 > args.time_limit_h:
            print(f"STOP time limit after epoch {epoch}", flush=True)
            break

    torch.save({"state_dict": packer.state_dict(), "epoch": epoch, "step": step,
                "config": header}, out / "final.pt")
    print("DONE " + json.dumps(dict(epochs=epoch + 1, steps=step,
                                    hours=(time.time() - t0) / 3600)), flush=True)


if __name__ == "__main__":
    main()
