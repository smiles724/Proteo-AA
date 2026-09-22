#!/usr/bin/env python3
"""Regenerate the combined AlphaProteo Table 4 from the scored metrics.

    python scripts/report_table4.py                    # markdown to stdout
    python scripts/report_table4.py --csv out.csv      # also a flat CSV
    python scripts/report_table4.py --data-root <dir>

Reads `designs_*/**/designs.csv` for the arm and target of each design and
`metrics*/**.csv` for its AF2-IG numbers, joins on `sample_id`, and applies
the four-way conjunction. Nothing is transcribed: the published rows below
are the only hand-entered numbers in this file, and this work's rows are
always recomputed.

Designability here is A-CODE Table 4's definition -- ipAE < 10.85 A,
ipTM > 0.5, pLDDT > 80%, binder bound/unbound RMSD < 3.5 A, all four
required, successes SUMMED across the length grid per target. Not a mean of
per-length rates: those are different numbers when the grid is unbalanced.
"""
from __future__ import annotations

import argparse
import csv
import collections
import glob
import math
from pathlib import Path

TARGETS = ["BHRF1", "H1", "IL17A", "IL7RA", "IR",
           "PDL1", "SC2RBD", "TNFa", "TrkA", "VEGFA"]

#: A-CODE Table 4 as published. The ONLY hand-entered numbers here.
PUBLISHED = [
    ("Two-Stage", "BoltzGen",
     [14.56, 19.21, 0.31, 12.41, 22.34, 14.26, 0.69, 0.88, 28.65, 7.69], 12.1),
    ("Two-Stage", "ODesign",
     [42.23, 7.29, 0.10, 7.61, 16.82, 10.26, 3.13, 0.00, 9.99, 3.71], 10.1),
    ("Two-Stage", "RFDiffusion-3",
     [33.38, 0.89, 0.82, 6.47, 18.17, 16.81, 4.34, 0.00, 14.44, 2.95], 9.8),
    ("Two-Stage", "PXDesign",
     [43.90, 12.08, 0.82, 29.80, 25.04, 45.33, 11.20, 3.43, 23.55, 16.72], 21.2),
    ("Two-Stage", "A-CODE (PMPNN)",
     [25.00, 65.59, 1.79, 4.93, 30.09, 39.96, 28.05, 6.16, 6.87, 1.37], 21.0),
    ("One-Stage", "Protpardelle-1c",
     [3.73, 0.27, 0.00, 0.09, 0.19, 7.04, 0.99, 0.00, 3.52, 0.17], 1.6),
    ("One-Stage", "A-CODE (Co-Design)",
     [22.87, 55.71, 1.24, 4.05, 41.67, 28.70, 37.50, 7.57, 3.70, 0.96], 20.4),
]

#: (arm key in designs.csv, row label). Order is the display order.
ARMS = [
    ("J03",    "PXD bb + FaMPNN 0.3 + A_BS joint (**J03** s0)"),
    ("J03_s1", "PXD bb + FaMPNN 0.3 + A_BS joint (**J03** s1)"),
    ("U03",    "PXD bb + FaMPNN 0.3, unadapted (**U03**)"),
    ("S03_s1", "PXD bb + FaMPNN 0.3 + A_BS sc-only (**S03** s1)"),
    ("S03_s0", "PXD bb + FaMPNN 0.3 + A_BS sc-only (**S03** s0)"),
    ("R0",     "PXD bb + ProteinMPNN (**R0**)"),
]


def designable(row: dict) -> bool | None:
    """The four-way conjunction. None when a metric is missing or unparseable."""
    try:
        return (float(row["ipae"]) < 10.85
                and float(row["iptm"]) > 0.5
                and float(row["plddt"]) > 0.80
                and float(row["binder_bound_unbound_rmsd"]) < 3.5)
    except (KeyError, TypeError, ValueError):
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent. Not normal-approximation: at n=48
    with rates near 0 the normal interval goes negative."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def collect(root: Path) -> dict:
    metrics: dict[str, dict] = {}
    for pattern in ("metrics/*.csv", "metrics_v2/*.csv", "metrics_all/*.csv"):
        for path in glob.glob(str(root / pattern)):
            with open(path) as handle:
                for row in csv.DictReader(handle):
                    if row.get("sample_id"):
                        metrics.setdefault(row["sample_id"], row)
    if not metrics:
        raise SystemExit(f"no metrics under {root}; nothing to report")

    tally = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    unscored = 0
    for pattern in ("designs_v1/*/designs.csv", "designs_r0/designs.csv",
                    "designs_v2/*/designs.csv"):
        for path in glob.glob(str(root / pattern)):
            with open(path) as handle:
                for design in csv.DictReader(handle):
                    metric = metrics.get(design["sample_id"])
                    verdict = designable(metric) if metric else None
                    if verdict is None:
                        unscored += 1
                        continue
                    # designs_r0 carries no `arm` column; it is the R0 row.
                    cell = tally[design.get("arm") or "R0"][design["target"]]
                    cell[1] += 1
                    cell[0] += int(verdict)
    return {"tally": tally, "unscored": unscored, "n_metrics": len(metrics)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=(
        "/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench"))
    parser.add_argument("--csv", default=None,
                        help="also write a flat arm,target,k,n,pct CSV")
    args = parser.parse_args()

    found = collect(Path(args.data_root))
    tally = found["tally"]

    mine = []
    for arm, label in ARMS:
        if arm not in tally:
            print(f"<!-- WARNING: no designs for arm {arm}; row omitted -->")
            continue
        rates, hits, total = [], 0, 0
        for target in TARGETS:
            k, n = tally[arm][target]
            hits += k
            total += n
            rates.append(100 * k / n if n else float("nan"))
        mine.append((label, rates, 100 * hits / total, hits, total, arm))

    best = [max(max(p[2][i] for p in PUBLISHED), max(m[1][i] for m in mine))
            for i in range(len(TARGETS))]
    best_mean = max(max(p[3] for p in PUBLISHED), max(m[2] for m in mine))

    def cell(value: float, target: float, places: int) -> str:
        text = f"{value:.{places}f}"
        return f"**{text}**" if abs(value - target) < 0.5 * 10 ** -places else text

    print("| Type | Method | " + " | ".join(TARGETS) + " | Mean |")
    print("|" + "---|" * (len(TARGETS) + 3))
    for kind, name, rates, mean in PUBLISHED:
        cells = " | ".join(cell(rates[i], best[i], 2) for i in range(len(TARGETS)))
        print(f"| {kind} | {name} | {cells} | {cell(mean, best_mean, 1)} |")
    for label, rates, mean, _hits, _total, _arm in mine:
        cells = " | ".join(cell(rates[i], best[i], 1) for i in range(len(TARGETS)))
        print(f"| This work | {label} | {cells} | {cell(mean, best_mean, 1)} |")

    print("\nBold = best on that target across ALL rows. Ties bolded jointly.\n")

    print("\n## 95% Wilson intervals\n")
    print("| target | " + " | ".join(a for a, _ in ARMS if a in tally) + " |")
    print("|" + "---|" * (1 + len(mine)))
    for target in TARGETS:
        cells = []
        for arm, _ in ARMS:
            if arm not in tally:
                continue
            k, n = tally[arm][target]
            lo, hi = wilson(k, n)
            cells.append(f"{100 * k / n:.1f} [{lo:.1f}, {hi:.1f}]" if n else "-")
        print(f"| {target} | " + " | ".join(cells) + " |")
    cells = []
    for _label, _rates, mean, hits, total, _arm in mine:
        lo, hi = wilson(hits, total)
        cells.append(f"**{mean:.1f} [{lo:.1f}, {hi:.1f}]**")
    print("| **ALL** | " + " | ".join(cells) + " |")

    per_target = {m[5]: m[4] // len(TARGETS) for m in mine}
    print(f"\nn per target: {sorted(set(per_target.values()))}; "
          f"finest non-zero rate expressible: "
          f"{100 / max(per_target.values()):.2f}%")
    print(f"metrics pool {found['n_metrics']} sample_id, "
          f"{found['unscored']} design(s) unscored")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["arm", "target", "designable", "n", "percent"])
            for arm, _ in ARMS:
                for target in TARGETS:
                    k, n = tally[arm][target]
                    writer.writerow([arm, target, k, n,
                                     f"{100 * k / n:.4f}" if n else ""])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
