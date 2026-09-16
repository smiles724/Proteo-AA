#!/usr/bin/env python3
"""Do the two side-chain estimators rank the arms the same way?

They disagreed. On 100 validation chains the in-training metric put a_token
slightly ahead of none (1.572 vs 1.577); on all 449, `eval_on_apm_testset.py`
put it clearly behind (1.686 vs 1.610). Two things differ at once -- the chain
subset and the estimator -- and until they are separated neither ordering can
be quoted.

This runs the IN-TRAINING estimator over ALL 449 chains for every finished arm.
Comparing its output here against the canonical table isolates the estimator,
because the chain set is then identical.

    python scripts/evaluation/compare_estimators_apmdata.py --arms none,plm,a_token
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RUNROOT = Path("/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data")
CACHE = Path("/hai/scratch/shenjm/proteo_aa_runs/a_token_cache")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="none,plm,a_token")
    ap.add_argument("--run-root", default=str(RUNROOT))
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 449")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "tr", str(REPO / "scripts" / "training" / "train_packer_apm_data.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    from pxdesign_train.sidechain.apm_dataset import post2021_val_files
    from pxdesign_train.sidechain.apm_loss import RigidGroupTables

    device = torch.device(args.device)
    tables = RigidGroupTables(torch.float32, device)
    val = post2021_val_files(tr.APM_TEST_DIR, tr.APM_TEST_IDS)
    print(f"{len(val)} validation chains", flush=True)

    rows = {}
    for arm in args.arms.split(","):
        path = Path(args.run_root) / arm / args.ckpt
        if not path.is_file():
            print(f"  {arm}: no {args.ckpt}, skipped", flush=True)
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        packer = tr.build_model(arm, tr.ESM2_650M, device)
        missing, unexpected = packer.load_state_dict(ck["state_dict"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"{arm}: {len(missing)} missing, {len(unexpected)} unexpected")
        a_dir = (CACHE / "val") if arm in ("a_token", "both") else None
        m = tr.validate(packer, val, device, tables,
                        limit=args.limit, a_token_dir=a_dir)
        m["epoch"] = int(ck.get("epoch", -1))
        rows[arm] = m
        print(f"  {arm:8s} epoch {m['epoch']:4d}  "
              f"atom14_symrmsd={m['atom14_symrmsd']:.4f}  "
              f"chi1_acc_40={m['chi1_acc_40']:.4f}  "
              f"rotamer_rec={m['rotamer_rec']:.4f}  n={m['n_chains']}", flush=True)
        del packer
        torch.cuda.empty_cache()

    print("\nin-training estimator, same 449 chains as the canonical table:")
    print(f"  {'arm':8s} {'atom14_symrmsd':>15s} {'chi1_acc_40':>12s} {'rotamer_rec':>12s}")
    for arm, m in rows.items():
        print(f"  {arm:8s} {m['atom14_symrmsd']:15.4f} {m['chi1_acc_40']:12.4f} "
              f"{m['rotamer_rec']:12.4f}")
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1))
        print("wrote", args.output)


if __name__ == "__main__":
    main()
