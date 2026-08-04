#!/usr/bin/env python3
"""Collect BB, AA-head, and ProteinMPNN metrics for one joint checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text())
    data["status"] = "ok"
    return data


def _load_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"status": "empty", "path": str(path)}
    out = dict(rows[0])
    out["status"] = "ok"
    return out


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key)
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-csv", default="")
    args = p.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    bb_path = root / "bb_eval" / "metrics.json"
    aa_path = root / "aa_eval" / "aa_head_checkpoint_metrics.csv"
    mpnn_path = root / "mpnn_recovery" / "proteinmpnn" / "proteinmpnn_recovery_summary.json"

    bb = _load_json(bb_path)
    aa = _load_first_csv_row(aa_path)
    mpnn = _load_json(mpnn_path)

    bb_metrics = bb.get("metrics", {}) if isinstance(bb.get("metrics"), dict) else {}
    summary = {
        "checkpoint": args.checkpoint or bb.get("checkpoint") or aa.get("checkpoint"),
        "run_root": str(root),
        "bb_eval_path": str(bb_path),
        "aa_eval_path": str(aa_path),
        "mpnn_eval_path": str(mpnn_path),
        "bb_status": bb.get("status"),
        "aa_status": aa.get("status"),
        "mpnn_status": mpnn.get("status"),
        "bb_evaluated_samples": bb.get("evaluated_samples"),
        "bb_loss": bb_metrics.get("loss"),
        "bb_mse": bb_metrics.get("mse"),
        "bb_ca_rmsd": bb_metrics.get("ca_rmsd"),
        "bb_bb_rmsd": bb_metrics.get("bb_rmsd"),
        "bb_tm_score": bb_metrics.get("tm_score"),
        "bb_lddt_loss": bb_metrics.get("lddt"),
        "bb_distogram": bb_metrics.get("distogram"),
        "bb_aa_acc_loss_path": bb_metrics.get("aa_acc"),
        "bb_aa_ce_loss_path": bb_metrics.get("aa_ce"),
        "aa_eval_samples": _float(aa, "n_eval"),
        "aa_tokens": _float(aa, "n_tokens"),
        "aa_acc": _float(aa, "acc"),
        "aa_top5_acc": _float(aa, "top5_acc"),
        "aa_f1_macro": _float(aa, "f1_macro"),
        "aa_f1_weighted": _float(aa, "f1_weighted"),
        "aa_auroc_macro": _float(aa, "auroc_macro"),
        "aa_auprc_macro": _float(aa, "auprc_macro"),
        "aa_ce": _float(aa, "aa_ce"),
        "mpnn_samples": mpnn.get("n_samples"),
        "mpnn_length_match_fraction": mpnn.get("length_match_fraction"),
        "mpnn_mean_recovery": mpnn.get("mean_recovery"),
        "mpnn_median_recovery": mpnn.get("median_recovery"),
        "mpnn_token_weighted_recovery": mpnn.get("token_weighted_recovery"),
        "mpnn_total_aligned_tokens": mpnn.get("total_aligned_tokens"),
    }

    out_json = Path(args.output_json).expanduser().resolve() if args.output_json else root / "combined_eval_summary.json"
    out_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else root / "combined_eval_summary.csv"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary_json={out_json}")
    print(f"wrote_summary_csv={out_csv}")


if __name__ == "__main__":
    main()
