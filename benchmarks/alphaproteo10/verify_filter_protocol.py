#!/usr/bin/env python3
"""Check which ProtDBench filter reproduces A-CODE Table 4.

A-CODE Appendix C.2 and ProtDBench's `af2_easy` list the same thresholds, but
that is an argument from two documents agreeing. This settles it from data:
ProtDBench publishes the per-design scores behind its own table, including a
`PXDesign` row that A-CODE also reports, so the two can be compared directly.

    git clone --depth 1 https://github.com/congliuUvA/ProtDBench
    python benchmarks/alphaproteo10/verify_filter_protocol.py --protdbench ProtDBench

Writes `filter_protocol_check.csv` next to this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A-CODE Table 4, PXDesign (two-stage) row, transcribed from the paper.
ACODE_TABLE4_PXDESIGN = {
    "BHRF1": 43.90, "H1": 12.08, "IL17A": 0.82, "IL7RA": 29.80, "IR": 25.04,
    "PDL1": 45.33, "SC2RBD": 11.20, "TNFa": 3.43, "TrkA": 23.55, "VEGFA": 16.72,
}
ORDER = list(ACODE_TABLE4_PXDESIGN)

# The two filters ProtDBench ships (protdbench/protd_configs/eval.py). They are
# not nested by construction: af2_easy also requires ipTM and measures the
# binder predicted alone against the binder chain of the complex prediction,
# while af2_opt drops ipTM and measures the binder predicted alone against the
# original design instead.
FILTERS = {
    "af2_easy": "pLDDT>0.8, i_pTM>0.5, i_pAE<0.35 (=10.85 A), bound_unbound_RMSD<3.5",
    "af2_opt": "pLDDT>0.9, unscaled_i_pAE<7.0, af2_binder_pred_design_rmsd<1.5",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protdbench", type=Path, required=True,
                    help="path to a cloned congliuUvA/ProtDBench")
    ap.add_argument("--method", default="PXDesign",
                    help="which released method table to check (default: PXDesign)")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("needs pandas", file=sys.stderr)
        return 1

    scores = (args.protdbench / "data" / "generative_benchmark" / "10_targets"
              / f"{args.method}_full_samples.csv.gz")
    if not scores.is_file():
        print(f"not found: {scores}\nclone ProtDBench first (its data/ is in the repo)",
              file=sys.stderr)
        return 1

    d = pd.read_csv(scores)
    print(f"{len(d)} designs, {d.Target.nunique()} targets\n")

    rows = {"A-CODE Table 4": pd.Series(ACODE_TABLE4_PXDESIGN)}
    for f in FILTERS:
        col = f"{f}_success"
        # The success columns are precomputed by ProtDBench; recomputing them
        # from raw scores would need the same per-model reduction it applies.
        rows[f] = d.groupby("Target")[col].mean().mul(100).round(2)
    t = pd.DataFrame(rows).reindex(ORDER)

    print(t.to_string(), "\n")
    ref = t["A-CODE Table 4"]
    for f in FILTERS:
        dev = (t[f] - ref).abs().mean()
        r = t[f].corr(ref)
        print(f"{f:9s} mean |deviation| {dev:6.2f}   r {r:.3f}   mean rate {t[f].mean():6.2f}%"
              f"   <- {FILTERS[f]}")
    print(f"{'':9s} {'':18s}        {'':5s}   mean rate {ref.mean():6.2f}%   <- A-CODE Table 4")

    out = Path(__file__).with_name("filter_protocol_check.csv")
    t.to_csv(out, index_label="target")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
