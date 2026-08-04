#!/usr/bin/env python3
"""Summarize ProteinMPNN recovery across Proteo-AA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_recovery(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "PENDING", ""
    data = json.loads(path.read_text())
    token = data.get("token_weighted_recovery")
    mean = data.get("mean_recovery")
    if token is None:
        return "NA", ""
    mean_s = "" if mean is None else f"{100.0 * float(mean):.2f}%"
    return f"{100.0 * float(token):.2f}%", mean_s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        default="/hai/scratch/yfsun/proteo_aa_runs/mpnn_recovery_monomer",
    )
    p.add_argument("--gt-run", default="gt_backbone_full491_h200")
    p.add_argument(
        "--steps",
        default="10000,20000,40000,60000,80000,100000",
        help="Comma-separated checkpoint steps.",
    )
    p.add_argument("--suffix", default="_full491_h200")
    args = p.parse_args()

    root = Path(args.run_root)
    gt_summary = root / args.gt_run / "proteinmpnn" / "proteinmpnn_recovery_summary.json"
    gt_token, gt_mean = _read_recovery(gt_summary)

    print("checkpoint,gt_backbone_token_weighted,gt_backbone_mean,pred_backbone_token_weighted,pred_backbone_mean,pred_summary")
    for step_s in args.steps.split(","):
        step_s = step_s.strip()
        if not step_s:
            continue
        pred_summary = (
            root
            / f"step{step_s}{args.suffix}"
            / "proteinmpnn"
            / "proteinmpnn_recovery_summary.json"
        )
        pred_token, pred_mean = _read_recovery(pred_summary)
        print(
            f"{int(step_s) // 1000}k,"
            f"{gt_token},{gt_mean},"
            f"{pred_token},{pred_mean},"
            f"{pred_summary}"
        )


if __name__ == "__main__":
    main()
