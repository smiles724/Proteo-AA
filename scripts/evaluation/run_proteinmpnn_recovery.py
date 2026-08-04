#!/usr/bin/env python3
"""Run ProteinMPNN on exported Proteo-AA backbones and score sequence recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

AA = set("ACDEFGHIKLMNPQRSTVWY")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header = None
    seq_parts: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.replace("/", ""))
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def _find_mpnn_fasta(mpnn_out: Path, sample_id: str) -> Path:
    candidates = [
        mpnn_out / "seqs" / f"{sample_id}.fa",
        mpnn_out / "seqs" / f"{sample_id}.fasta",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((mpnn_out / "seqs").glob(f"{sample_id}*.fa*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No ProteinMPNN FASTA found for {sample_id} in {mpnn_out / 'seqs'}"
    )


def _pick_designed_sequence(records: list[tuple[str, str]]) -> tuple[str, str]:
    if len(records) < 2:
        raise ValueError("ProteinMPNN FASTA does not contain a designed sequence")
    for header, seq in records[1:]:
        clean = "".join(ch for ch in seq.upper() if ch in AA)
        if clean:
            return header, clean
    header, seq = records[-1]
    return header, "".join(ch for ch in seq.upper() if ch in AA)


def _parse_score(header: str, key: str) -> float:
    m = re.search(
        rf"\b{re.escape(key)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)", header
    )
    return float(m.group(1)) if m else float("nan")


def _recovery(gt: str, pred: str) -> dict[str, Any]:
    n = min(len(gt), len(pred))
    if n == 0:
        return {
            "n_aligned": 0,
            "n_gt": len(gt),
            "n_pred": len(pred),
            "recovery": float("nan"),
            "matches": 0,
            "length_match": len(gt) == len(pred),
        }
    gt_trim = gt[:n]
    pred_trim = pred[:n]
    matches = sum(a == b for a, b in zip(gt_trim, pred_trim))
    return {
        "n_aligned": n,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "recovery": matches / n,
        "matches": matches,
        "length_match": len(gt) == len(pred),
    }


def _run_mpnn_one(
    *,
    python_bin: str,
    mpnn_script: Path,
    pdb_path: Path,
    out_dir: Path,
    num_seq_per_target: int,
    sampling_temp: str,
    seed: int,
    batch_size: int,
    device: str,
    extra_args: list[str],
) -> None:
    cmd = [
        python_bin,
        str(mpnn_script),
        "--pdb_path",
        str(pdb_path),
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        str(num_seq_per_target),
        "--sampling_temp",
        str(sampling_temp),
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
    ]
    if device:
        cmd.extend(["--device", str(device)])
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--proteinmpnn-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--proteinmpnn-python", default=sys.executable)
    p.add_argument("--proteinmpnn-script", default="")
    p.add_argument("--num-seq-per-target", type=int, default=1)
    p.add_argument("--sampling-temp", default="0.1")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--device",
        default="",
        help="Optional: pass --device to ProteinMPNN only if that checkout supports it.",
    )
    p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra raw arg passed to protein_mpnn_run.py; repeatable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    mpnn_dir = Path(args.proteinmpnn_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mpnn_script = (
        Path(args.proteinmpnn_script).expanduser().resolve()
        if args.proteinmpnn_script
        else mpnn_dir / "protein_mpnn_run.py"
    )
    if not mpnn_script.exists():
        raise FileNotFoundError(f"ProteinMPNN script not found: {mpnn_script}")

    rows = _read_manifest(manifest)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    result_rows: list[dict[str, Any]] = []
    extra_args = list(args.extra_arg or [])

    for i, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        pdb_path = Path(row["pdb_path"]).expanduser().resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)
        fasta_path = out_dir / "seqs" / f"{sample_id}.fa"
        if not (bool(args.skip_existing) and fasta_path.exists()):
            _run_mpnn_one(
                python_bin=str(args.proteinmpnn_python),
                mpnn_script=mpnn_script,
                pdb_path=pdb_path,
                out_dir=out_dir,
                num_seq_per_target=int(args.num_seq_per_target),
                sampling_temp=str(args.sampling_temp),
                seed=int(args.seed) + i,
                batch_size=int(args.batch_size),
                device=str(args.device),
                extra_args=extra_args,
            )
        out_fasta = _find_mpnn_fasta(out_dir, sample_id)
        records = _read_fasta(out_fasta)
        pred_header, pred_seq = _pick_designed_sequence(records)
        gt_seq = "".join(ch for ch in row["gt_sequence"].upper() if ch in AA)
        rec = _recovery(gt_seq, pred_seq)
        result_rows.append(
            {
                **row,
                "pred_sequence": pred_seq,
                "mpnn_header": pred_header,
                "mpnn_score": _parse_score(pred_header, "score"),
                "mpnn_global_score": _parse_score(pred_header, "global_score"),
                **rec,
            }
        )
        if i % 25 == 0:
            print(
                f"scored={i} last={sample_id} recovery={rec['recovery']:.4f}",
                flush=True,
            )

    metrics_csv = out_dir / "proteinmpnn_recovery_metrics.csv"
    with metrics_csv.open("w", newline="") as fh:
        fieldnames = list(result_rows[0].keys()) if result_rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    recoveries = np.array([float(r["recovery"]) for r in result_rows], dtype=float)
    n_aligned = np.array([int(r["n_aligned"]) for r in result_rows], dtype=float)
    matches = np.array([int(r["matches"]) for r in result_rows], dtype=float)
    summary = {
        "n_samples": int(len(result_rows)),
        "mean_recovery": (
            float(np.nanmean(recoveries)) if recoveries.size else float("nan")
        ),
        "median_recovery": (
            float(np.nanmedian(recoveries)) if recoveries.size else float("nan")
        ),
        "token_weighted_recovery": (
            float(matches.sum() / max(1.0, n_aligned.sum()))
            if recoveries.size
            else float("nan")
        ),
        "total_aligned_tokens": int(n_aligned.sum()) if recoveries.size else 0,
        "length_match_fraction": (
            float(np.mean([bool(r["length_match"]) for r in result_rows]))
            if result_rows
            else float("nan")
        ),
        "metrics_csv": str(metrics_csv),
    }
    summary_json = out_dir / "proteinmpnn_recovery_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_metrics={metrics_csv}")
    print(f"wrote_summary={summary_json}")


if __name__ == "__main__":
    main()
