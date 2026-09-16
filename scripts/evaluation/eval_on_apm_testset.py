#!/usr/bin/env python3
"""Score side-chain packing on APM's own post-2021 held-out monomers.

WHAT THIS IS FOR. Two numbers have to end up in the same column of the same
table: APM's released `sidechain_model.ckpt`, and the Dunbrack-mode template.
Running each under its own harness would repeat the mistake that cost a round
earlier in this project -- three "template baselines" in the repo that differ by
a factor of two purely because they are different estimators. So both are
scored here, on the same structures, through the same mask, with the same
`sidechain/metrics.packing_metrics` the training runs are scored by.

The APM checkpoint is run through OUR packer, which is legitimate only because
`check_packer_matches_apm.py` shows the two forwards agree to 1e-5 rad on the
same weights and input. That equivalence is the licence for this script.

DATA. APM's test set: `metadata_all/test_set_pdb_ids.csv` names 449 PDB ids, all
monomers, and every one of them is in `pdb_test/`. Each pickle is raw atom37:
aatype, atom_positions [L, 37, 3], atom_mask [L, 37], residue_index,
chain_index, modeled_idx.

    PYTHONPATH=$REPO:$REPO/PXDesign:$REPO/Protenix LAYERNORM_TYPE=torch \
    python scripts/evaluation/eval_on_apm_testset.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

# openfold atom37 ordering (openfold/np/residue_constants.py: atom_types)
ATOM37 = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
          'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2',
          'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH',
          'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-test-dir", default="/hai/scratch/shenjm/apm_weights/pdb_test")
    p.add_argument("--test-ids", default="/hai/scratch/shenjm/apm_weights/metadata_all/test_set_pdb_ids.csv")
    p.add_argument("--apm-weights", default="/hai/scratch/shenjm/apm_weights/sidechain_state_dict.pt")
    p.add_argument("--output", default="/hai/scratch/shenjm/proteo_aa_runs/apm_testset_eval/result.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=0, help="0 = no cap; else skip longer chains")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N chains")
    p.add_argument("--methods", default="apm_ckpt,dunbrack_template")
    p.add_argument("--our-run", default="/hai/scratch/shenjm/proteo_aa_runs/sc_torsion_packer/117035",
                   help="root holding <arm>/checkpoints/<step>.pt for our four arms")
    p.add_argument("--our-step", default="step50000.pt")
    p.add_argument("--our-weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--apm-data-run",
                   default="/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data",
                   help="root of the runs trained on APM data (method apmdata_<arm>)")
    p.add_argument("--apm-data-ckpt", default="final.pt")
    p.add_argument("--plm-checkpoint",
                   default="/hai/scratch/shenjm/plm_weights/esm2_t33_650M_UR50D.pt")
    return p.parse_args()


def slot_table(device):
    """[20, MAX_SC] atom37 index for each of our side-chain slots, and its mask."""
    from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3, sidechain_atoms
    idx = torch.zeros(len(STD_AA_3), MAX_SC, dtype=torch.long)
    ok = torch.zeros(len(STD_AA_3), MAX_SC, dtype=torch.bool)
    for t, name in enumerate(STD_AA_3):
        for j, atom in enumerate(sidechain_atoms(name)[:MAX_SC]):
            idx[t, j] = ATOM37.index(atom)
            ok[t, j] = True
    return idx.to(device), ok.to(device)


def load_chain(path, slot_idx, slot_ok, device):
    """APM pickle -> the tensors both methods and the metric need."""
    from pxdesign_train.sidechain.frames import build_frame, to_local, valid_ncac

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    keep = torch.as_tensor(d["modeled_idx"], dtype=torch.long)
    types = torch.as_tensor(d["aatype"], dtype=torch.long)[keep].to(device)
    pos = torch.as_tensor(d["atom_positions"], dtype=torch.float32)[keep].to(device)
    amask = torch.as_tensor(d["atom_mask"], dtype=torch.float32)[keep].to(device) > 0.5
    res_idx = torch.as_tensor(d["residue_index"], dtype=torch.long)[keep].to(device)
    chain_idx = torch.as_tensor(d["chain_index"], dtype=torch.long)[keep].to(device)

    canonical = (types >= 0) & (types < slot_idx.shape[0])
    tix = types.clamp(0, slot_idx.shape[0] - 1)
    n, ca, c = pos[:, 0], pos[:, 1], pos[:, 2]
    frame_ok = valid_ncac(n, ca, c, amask[:, :3]) & canonical
    R, t = build_frame(torch.nan_to_num(n), torch.nan_to_num(ca), torch.nan_to_num(c))
    eye = torch.eye(3, device=device)
    R = torch.where(frame_ok[:, None, None], R, eye)
    t = torch.where(frame_ok[:, None], t, torch.zeros_like(t))

    sel = slot_idx[tix]                                     # [L, MAX_SC]
    chem = slot_ok[tix] & canonical[:, None]                # chemistry: slot exists
    sc_global = torch.gather(pos, 1, sel[..., None].expand(-1, -1, 3))
    observed = torch.gather(amask, 1, sel) & chem & frame_ok[:, None]
    gt_local = to_local(sc_global, R, t)
    bb_local = to_local(pos[:, :3], R, t)
    return dict(types=types, tix=tix, R=R, t=t, chem=chem, observed=observed,
                gt_local=gt_local, bb_local=bb_local, res_idx=res_idx,
                chain_idx=chain_idx, frame_ok=frame_ok, canonical=canonical)


# Our four arms were trained BEFORE the packer was restructured to APM's module
# names, so their checkpoints carry the old flat layout. The rename is mechanical
# and total -- no tensor changes shape -- so remapping is exact rather than
# approximate. Listed explicitly so a future rename fails loudly here instead of
# silently leaving a submodule at its random initialization.
_OLD_TO_NEW = {
    "node_linear.": "node_feature_net.linear.",
    "aatype_embedding.": "node_feature_net.aatype_embedding.",
    "linear_s_p.": "edge_feature_net.linear_s_p.",
    "linear_relpos.": "edge_feature_net.linear_relpos.",
    "edge_embedder.": "edge_feature_net.edge_embedder.",
}


def remap_our_checkpoint(sd):
    out = {}
    for k, v in sd.items():
        if k == "torsion_embedding.freq_bands":
            # One buffer in the old layout, one per feature net in the new one.
            out["node_feature_net.torsion_embedding.freq_bands"] = v
            out["edge_feature_net.torsion_embedding.freq_bands"] = v
            continue
        for old, new in _OLD_TO_NEW.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_our_arm(root, arm, step, weights, plm_ckpt, device):
    """One of our trained arms, in the architecture it was TRAINED with."""
    from pxdesign_train.sidechain.packer import TorsionPacker
    path = Path(root) / arm / "checkpoints" / step
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = dict(ck["model"])
    if weights == "ema" and ck.get("ema"):
        sd.update(ck["ema"]["shadow"])
    sd = {k[len("sidechain_module."):]: v for k, v in sd.items()
          if k.startswith("sidechain_module.")}
    sd = remap_our_checkpoint(sd)
    packer = TorsionPacker(
        c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
        ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=4, num_torsion_blocks=4, seq_cond=arm,
        embed_aatype=True, embed_rotvecs=True,
        # As trained, NOT as APM: these two are exactly the gaps the alignment
        # closed afterwards, and loading these weights under the new defaults
        # would be loading them into a different model.
        use_mlp=False, embed_chain=False,
        random_torsion_input=True,
        plm_checkpoint=(plm_ckpt if arm in ("plm", "both") else ""),
    ).eval().to(device)
    missing, unexpected = packer.load_state_dict(sd, strict=False)
    # The PLM conditioner used to be constructed on every arm and is now built
    # only where it is read, so a none/a_token checkpoint from before that
    # change carries conditioner tensors this module has no home for. They
    # never received a gradient, so dropping them changes no output -- but say
    # so out loud rather than letting an unexpected-key list go unexamined.
    dead = [k for k in unexpected if k.startswith("plm_conditioner.")]
    if dead and arm not in ("plm", "both"):
        print(f"  {arm}: dropped {len(dead)} unused plm_conditioner tensors "
              f"(never trained on this arm)")
        unexpected = [k for k in unexpected if not k.startswith("plm_conditioner.")]
    if unexpected or missing:
        raise SystemExit(f"{arm}: {len(missing)} missing, {len(unexpected)} unexpected\n"
                         f"  missing={sorted(missing)[:6]}\n  unexpected={sorted(unexpected)[:6]}")
    return packer


def load_apm_data_arm(path, arm, plm_ckpt, device):
    """An arm retrained on APM data by `scripts/training/train_packer_apm_data.py`.

    Distinct from `load_our_arm` in exactly one way that matters: these were
    trained in APM's architecture (use_mlp / embed_chain both on), so they must
    be rebuilt that way. The earlier arms predate that alignment.
    """
    from pxdesign_train.sidechain.packer import TorsionPacker
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    cfg = ck.get("config", {})
    if cfg.get("arm") and cfg["arm"] != arm:
        raise SystemExit(f"{path} records arm={cfg['arm']!r}, not {arm!r}")
    packer = TorsionPacker(
        c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
        ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=4, num_torsion_blocks=4, seq_cond=arm,
        embed_aatype=True, embed_rotvecs=True, use_mlp=True, embed_chain=True,
        random_torsion_input=True,
        plm_checkpoint=(plm_ckpt if arm in ("plm", "both") else ""),
    ).eval().to(device)
    missing, unexpected = packer.load_state_dict(sd, strict=False)
    if unexpected or missing:
        raise SystemExit(f"{arm} (apm-data): {len(missing)} missing, "
                         f"{len(unexpected)} unexpected\n"
                         f"  missing={sorted(missing)[:6]}\n"
                         f"  unexpected={sorted(unexpected)[:6]}")
    print(f"  loaded apm-data arm {arm} from {path} "
          f"(epoch {ck.get('epoch')}, step {ck.get('step')})")
    return packer


def predict_apm(packer, ch, device, diffuse_mask=None):
    """Run a packer on one chain and return the side chain in the residue frame."""
    from pxdesign_train.sidechain.frames import to_local
    L = ch["types"].shape[0]
    logits = torch.nn.functional.one_hot(ch["tix"], 20).float() * 40.0 - 20.0
    ids = torch.zeros(1, L, ch["chem"].shape[-1], dtype=torch.long, device=device)
    with torch.no_grad():
        xyz, _, _ = packer(
            torch.zeros(1, L, 768, device=device), logits[None], ids,
            ch["chem"][None], torch.zeros(1, L, ch["chem"].shape[-1], 3, device=device),
            torch.zeros(1, device=device),
            frame_R=ch["R"][None], frame_t=ch["t"][None],
            res_mask=ch["frame_ok"][None], residue_index=ch["res_idx"][None],
            asym_id=ch["chain_idx"][None],
            diffuse_mask=(None if diffuse_mask is None else diffuse_mask[None]),
        )
    return to_local(xyz[0].float(), ch["R"], ch["t"])


def main() -> None:
    args = parse_args()
    import pandas as pd

    from pxdesign_train.sidechain.frames import build_frame, phi_psi_from_ncac, to_local
    from pxdesign_train.sidechain.metrics import packing_metrics, summarize_metrics
    from pxdesign_train.sidechain.packer import TorsionPacker
    from pxdesign_train.sidechain.templates import PROVIDERS

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    slot_idx, slot_ok = slot_table(device)
    ids = [str(x) for x in pd.read_csv(args.test_ids)["pdb_name"]]
    files = []
    for pdb in ids:
        f = Path(args.pdb_test_dir) / f"{pdb}.pkl"
        if f.is_file():
            files.append((pdb, f))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} of {len(ids)} test ids found in {args.pdb_test_dir}")

    methods = args.methods.split(",")
    packer = None
    if "apm_ckpt" in methods:
        packer = TorsionPacker(
            c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
            ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
            seq_tfmr_num_layers=4, num_torsion_blocks=4, seq_cond="none",
            embed_aatype=True, embed_rotvecs=True, use_mlp=True, embed_chain=True,
            random_torsion_input=True,
        ).eval().to(device)
        sd = torch.load(args.apm_weights, map_location="cpu", weights_only=True)
        miss, unexpected = packer.load_state_dict(sd, strict=False)
        if unexpected:
            raise SystemExit(f"unexpected tensors in APM weights: {unexpected[:5]}")
        print(f"APM weights loaded ({len(sd)} tensors; {len(miss)} of ours left at init)")
    ours = {}
    for m in methods:
        # `apmdata_<arm>` scores a checkpoint retrained on APM's data with APM's
        # objective; `ours_<arm>` scores the earlier PXDesign-data arms. They are
        # separate method names on purpose -- they are different models and would
        # be indistinguishable in a results table under one label.
        if m.startswith("apmdata_"):
            arm = m[len("apmdata_"):]
            path = Path(args.apm_data_run) / arm / args.apm_data_ckpt
            ours[m] = load_apm_data_arm(path, arm, args.plm_checkpoint, device)
        elif m.startswith("ours_"):
            arm = m[len("ours_"):]
            ours[m] = load_our_arm(args.our_run, arm, args.our_step, args.our_weights,
                                   args.plm_checkpoint, device)
            print(f"loaded our arm {arm} ({args.our_weights} weights)")

    totals = {m: {} for m in methods}
    n_used = skipped = 0
    for i, (pdb, path) in enumerate(files):
        try:
            ch = load_chain(path, slot_idx, slot_ok, device)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  skip {pdb}: {type(exc).__name__} {exc}")
            skipped += 1
            continue
        L = ch["types"].shape[0]
        if args.max_len and L > args.max_len:
            skipped += 1
            continue

        preds = {}
        if "apm_ckpt" in methods:
            # APM's packing task marks EVERY modelled residue as being
            # generated, GLY included; our own runs derive it from side-chain
            # ownership. Each set of weights gets the convention it was trained
            # under -- this is a network input, not a bookkeeping mask.
            preds["apm_ckpt"] = predict_apm(packer, ch, device,
                                            diffuse_mask=ch["frame_ok"].float())
        for m, mod in ours.items():
            # Same rule as for the released weights: every model gets the
            # diffuse_mask convention it was TRAINED under, because this is a
            # network input. The apm-data arms were trained on APM's (every
            # modelled residue is 1); the earlier arms on side-chain ownership,
            # which is the packer's own default.
            preds[m] = predict_apm(
                mod, ch, device,
                diffuse_mask=ch["frame_ok"].float() if m.startswith("apmdata_") else None)
        if "dunbrack_template" in methods:
            # phi/psi from the GLOBAL backbone: a dihedral spans three residues,
            # so it cannot be taken in any single residue's local frame.
            gN = torch.einsum("lij,lj->li", ch["R"], ch["bb_local"][:, 0]) + ch["t"]
            gCA = torch.einsum("lij,lj->li", ch["R"], ch["bb_local"][:, 1]) + ch["t"]
            gC = torch.einsum("lij,lj->li", ch["R"], ch["bb_local"][:, 2]) + ch["t"]
            phi, psi = phi_psi_from_ncac(gN, gCA, gC, ch["res_idx"], ch["chain_idx"],
                                         have=ch["frame_ok"])
            tmpl, _ = PROVIDERS["dunbrack_mode"](ch["tix"].cpu(), phi=phi.cpu(),
                                                 psi=psi.cpu())
            preds["dunbrack_template"] = tmpl.to(device).float()

        design = ch["canonical"] & ch["frame_ok"]
        for m, pred in preds.items():
            counts = packing_metrics(
                ch["types"], pred, ch["bb_local"], ch["chem"], design,
                target=ch["gt_local"], observed=ch["observed"],
                target_bb=ch["bb_local"])
            for k, v in counts.items():
                totals[m][k] = totals[m].get(k, 0.0) + v.detach().cpu()
        n_used += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} ...")

    record = dict(chains=n_used, skipped=skipped, test_ids=len(ids),
                  device=str(device), max_len=args.max_len)
    for m in methods:
        s = summarize_metrics({k: torch.as_tensor(v) for k, v in totals[m].items()})
        record[m] = {k: float(v) for k, v in s.items()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    keys = ["symmetry_rmsd", "chi_recovery_40deg", "chi_recovery_20deg",
            "chi1_accuracy_40deg", "chi1_chi2_accuracy_40deg", "rotamer_recovery",
            "bond_mae", "bad_bond_fraction", "completeness"]
    print(f"\n{'metric':28s}" + "".join(f"{m:>22s}" for m in methods))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{record[m].get(k, float('nan')):22.4f}" for m in methods))
    print(f"\nchains scored: {n_used}   skipped: {skipped}   -> {out}")


if __name__ == "__main__":
    main()
