#!/usr/bin/env python3
"""Apply the AF2-IG filter to folded benchmark designs and report Designability.

Input is a CSV of AF2 initial-guess metrics, one row per design. Required columns:

    sample_id, target, binder_length, ipae, iptm, plddt, binder_bound_unbound_rmsd

Optional: `variant` (`co_design` / `pmpnn`), which is scored separately so the two
arms A-CODE reports in Table 4 stay apart.

The filter is the four-way conjunction quoted in A-CODE Appendix C.2 — ipAE <
10.85, ipTM > 0.5, pLDDT > 0.8, binder bound/unbound RMSD < 3.5 A — and lives in
`pxdesign_train.benchmarks.AF2IGFilter` so the thresholds are stated once. Success
counts are pooled over the length grid per target, which is what the paper does
("Across different binder lengths for the same target, the success counts are
summed"); pooling is not the same as averaging the per-length rates whenever the
per-length sample counts differ, so this matters.

Rows with a missing metric are an error, not a zero: a fold that failed and a
design that failed the filter are different facts, and silently merging them
would bias designability downward by however many jobs crashed. Drop them
upstream, or pass --allow-missing to count them as non-designable explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `conditional_binder` is imported as a top-level module rather than as
# `pxdesign_train.benchmarks.conditional_binder`, because the package __init__
# imports Protenix and this step must not need it. Applying a published
# threshold to a CSV is arithmetic; requiring the whole training stack to redo
# it -- after a threshold changes, say, or on a laptop -- is the kind of
# coupling that stops people from re-deriving a published number. The module
# itself is stdlib-only.
BENCHMARKS_DIR = REPO_ROOT / "pxdesign_train" / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from conditional_binder import (  # noqa: E402
    TABLE4_ORDER,
    ConditionalBinderDesignBenchmark,
)

REQUIRED = ("sample_id", "target", "binder_length", "ipae", "iptm", "plddt",
            "binder_bound_unbound_rmsd")


def _load_rows(path: Path, allow_missing: bool) -> tuple[list[dict], int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing_columns:
            raise SystemExit(
                f"{path}: missing required column(s) {missing_columns}; "
                f"found {reader.fieldnames}"
            )
        rows, dropped = [], 0
        for row in reader:
            blank = [
                c
                for c in ("ipae", "iptm", "plddt", "binder_bound_unbound_rmsd")
                if row.get(c) in (None, "", "nan", "NaN")
            ]
            if blank:
                if not allow_missing:
                    raise SystemExit(
                        f"{path}: {row.get('sample_id')} has no value for {blank}. "
                        "A failed fold is not a failed design — drop it upstream or "
                        "pass --allow-missing to count it as non-designable."
                    )
                dropped += 1
                # Counted as a sample that did not pass, which is what
                # --allow-missing opts into.
                row.update({c: "inf" if "rmsd" in c or c == "ipae" else "-inf" for c in blank})
            for key in ("ipae", "iptm", "plddt", "binder_bound_unbound_rmsd"):
                row[key] = float(row[key])
            row["binder_length"] = int(float(row["binder_length"]))
            rows.append(row)
    return rows, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-csv", required=True, help="AF2-IG metrics, one row per design")
    parser.add_argument("--output-dir", default=None, help="defaults to the metrics CSV's directory")
    parser.add_argument("--manifest", default=None, help="override the test-set manifest JSON")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="score only these variants (default: every variant present, plus 'all')",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else metrics_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = (
        ConditionalBinderDesignBenchmark.load(args.manifest)
        if args.manifest
        else ConditionalBinderDesignBenchmark.load()
    )
    rows, dropped = _load_rows(metrics_csv, args.allow_missing)
    if not rows:
        raise SystemExit(f"{metrics_csv}: no rows")

    present = sorted({row.get("variant", "co_design") or "co_design" for row in rows})
    variants = args.variants if args.variants else present
    unknown_targets = {row["target"] for row in rows} - {t.name for t in benchmark.targets}
    if unknown_targets:
        raise SystemExit(
            f"{metrics_csv}: target name(s) not in the benchmark: {sorted(unknown_targets)}"
        )

    report: dict[str, dict] = {}
    for variant in variants:
        subset = [r for r in rows if (r.get("variant") or "co_design") == variant]
        report[variant] = benchmark.designability(subset, variant=None)
    # "all" pools the arms; useful only as a sanity total, never as a reported number.
    if len(variants) > 1:
        report["all"] = benchmark.designability(rows, variant=None)

    per_target_csv = output_dir / "designability_per_target.csv"
    with per_target_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "target", "binder_length", "n_samples",
                         "n_designable", "designability_pct"])
        for variant, scores in report.items():
            for name in TABLE4_ORDER:
                bucket = scores.get(name)
                if bucket is None:
                    continue
                writer.writerow([variant, name, "all", bucket["n_samples"],
                                 bucket["n_designable"], f"{bucket['designability']:.2f}"])
                for length in sorted(bucket["per_length"]):
                    cell = bucket["per_length"][length]
                    writer.writerow([variant, name, length, cell["n_samples"],
                                     cell["n_designable"], f"{cell['designability']:.2f}"])

    summary = {
        "benchmark": "ConditionalBinderDesignBenchmark",
        "manifest_version": benchmark.version,
        "filter": benchmark.af2ig.describe(),
        "metrics_csv": str(metrics_csv),
        "n_rows": len(rows),
        "n_rows_with_missing_metrics_counted_as_failures": dropped,
        "targets_scored": sorted({r["target"] for r in rows}),
        "targets_missing_from_this_run": [
            name
            for name in TABLE4_ORDER
            if name not in {r["target"] for r in rows}
        ],
        "designability": {
            variant: {
                name: round(bucket["designability"], 2)
                for name, bucket in scores.items()
            }
            for variant, scores in report.items()
        },
    }
    (output_dir / "designability.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    for variant in variants:
        print(f"\n=== variant: {variant} ===")
        print(benchmark.summary_table(report[variant]))
    if summary["targets_missing_from_this_run"]:
        print(
            "\nnot scored (no rows): "
            + ", ".join(summary["targets_missing_from_this_run"])
        )
    print(f"\nwrote {per_target_csv}")
    print(f"wrote {output_dir / 'designability.json'}")


if __name__ == "__main__":
    main()
