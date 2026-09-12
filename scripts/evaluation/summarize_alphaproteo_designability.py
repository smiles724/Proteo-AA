#!/usr/bin/env python3
"""Summarize strict AF2-IG designability with missing evaluations as failures."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_METRICS = (
    "unscaled_i_pAE", "pLDDT", "af2_binder_pred_design_rmsd",
    "i_pTM", "bound_unbound_RMSD",
)


def _parsed(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _truth(value: Any) -> bool:
    value = _parsed(value)
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_truth(item) for item in value)
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _float(value: Any) -> float | None:
    value = _parsed(value)
    if isinstance(value, (tuple, list)):
        values = [_float(item) for item in value]
        if not values or any(item is None for item in values):
            return None
        return sum(values) / len(values)
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def valid_score(row):
    return all(_float(row.get(metric)) is not None for metric in RAW_METRICS[:3])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-file", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.task_file).open() as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    detail_rows = []
    for task in tasks:
        score_csv = Path(task["output_dir"]) / "sample_level_output.csv"
        scores = []
        if score_csv.is_file():
            with score_csv.open() as handle:
                scores = list(csv.DictReader(handle))
        expected = int(task["expected_sequences"])
        valid = [row for row in scores if valid_score(row)]
        successes = sum(_truth(row.get("af2_opt_success")) for row in valid)
        row: dict[str, Any] = {
            "model_label": task["model_label"],
            "target": task["target"],
            "sequence_arm": task["sequence_arm"],
            "n_backbones": int(task["n_backbones"]),
            "expected_sequences": expected,
            "scored_sequences": len(scores),
            "valid_scored_sequences": len(valid),
            "invalid_scores": len(scores) - len(valid),
            "missing_scores": max(0, expected - len(scores)),
            "scored_failures": len(valid) - successes,
            "coverage": len(scores) / expected if expected else 0.0,
            "strict_af2ig_successes": successes,
            "designability": successes / expected if expected else 0.0,
            "score_csv": str(score_csv),
        }
        for metric in RAW_METRICS:
            values = [_float(item.get(metric)) for item in scores]
            finite = [value for value in values if value is not None]
            row[f"mean_{metric}"] = sum(finite) / len(finite) if finite else None
            row[f"n_{metric}"] = len(finite)
        detail_rows.append(row)

    aggregates = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["model_label"], row["sequence_arm"])].append(row)
    for (model, arm), rows in sorted(grouped.items()):
        expected = sum(row["expected_sequences"] for row in rows)
        scored = sum(row["scored_sequences"] for row in rows)
        successes = sum(row["strict_af2ig_successes"] for row in rows)
        aggregates.append(
            {
                "model_label": model,
                "sequence_arm": arm,
                "n_targets": len(rows),
                "expected_sequences": expected,
                "scored_sequences": scored,
                "coverage": scored / expected if expected else 0.0,
                "strict_af2ig_successes": successes,
                "pooled_designability": successes / expected if expected else 0.0,
                "macro_target_designability": sum(r["designability"] for r in rows) / len(rows),
            }
        )

        aggregate = aggregates[-1]
        for key in ("valid_scored_sequences", "invalid_scores", "missing_scores", "scored_failures"):
            aggregate[key] = sum(row[key] for row in rows)
        for metric in RAW_METRICS:
            count = sum(row[f"n_{metric}"] for row in rows)
            aggregate[f"mean_{metric}"] = sum((row[f"mean_{metric}"] or 0.) * row[f"n_{metric}"] for row in rows) / count if count else None

    if not detail_rows:
        raise ValueError("Task file contains no evaluation tasks")
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / "designability_by_target.csv"
    aggregate_path = out / "designability_summary.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader(); writer.writerows(detail_rows)
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader(); writer.writerows(aggregates)
    summary = {
        "strict_filter": {
            "unscaled_i_pAE": "< 7.0",
            "pLDDT": "> 0.9",
            "af2_binder_pred_design_rmsd": "< 1.5 A",
        },
        "missing_scores_count_as_failures": True,
        "by_target_csv": str(detail_path),
        "summary_csv": str(aggregate_path),
        "aggregates": aggregates,
    }
    (out / "designability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
