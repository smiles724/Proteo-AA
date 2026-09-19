#!/usr/bin/env python3
"""Collect the SC->BB feedback arms' offline results into one table.

The training logs cannot answer whether the feedback helps: the arms have
different trainable sets, so `like_s3` lowers loss_bb and loss_sc by changing
the backbone and the packer while `fb_only` cannot. A lower training loss there
is expected and carries no information. These two measures can answer it, and
this script refuses to print a number whose provenance does not match.

    python scripts/evaluation/summarize_sc_env_feedback.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GEOM_ROOT = "/hai/scratch/shenjm/proteo_aa_runs/sc_env_feedback_eval"
ARMS = ("fb_only", "like_s3", "official")

# From docs/ -- quoted only to be compared against a number measured under the
# SAME probe and step count, never against one from the other probe.
REFERENCE = {
    "geometry (this probe, 400 steps)": {
        "official PXDesign v0.1.0": 8.04,
        "Stage II step52500": 60.08,
        "Stage III 111408/step6000": 60.53,
    },
    "designability (AlphaProteo-10)": {
        "official PXDesign v0.1.0": 8.26,
        "Proteo-AA 111408/step6000": 0.0,
    },
}


def load_geometry(root: Path, arm: str):
    f = root / arm / "monomer_geometry_summary.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    return dict(
        bad_bond_pct=100.0 * d["bad_bond_fraction_mean"],
        ca_ca_median=d["ca_ca_median"],
        ca_ca_min=d["ca_ca_min"],
        rg_median=d.get("radius_of_gyration_median"),
        n_step=d.get("n_step"),
        n_samples=d.get("n_samples"),
        checkpoint=d.get("checkpoint"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry-root", default=GEOM_ROOT)
    ap.add_argument("--designability", default="",
                    help="designability_summary.csv from the AlphaProteo batch")
    args = ap.parse_args()

    root = Path(args.geometry_root)
    rows = {a: load_geometry(root, a) for a in ARMS}
    have = {a: r for a, r in rows.items() if r}

    print("=== backbone geometry, one probe, one step count ===")
    if not have:
        print(f"  nothing under {root} yet")
    else:
        steps = {r["n_step"] for r in have.values()}
        ns = {r["n_samples"] for r in have.values()}
        if len(steps) > 1 or len(ns) > 1:
            print(f"  !! NOT COMPARABLE: n_step={steps}, n_samples={ns}. Mixing "
                  f"step counts is how the official checkpoint reads 8.04% at "
                  f"400 and 88% at 20.")
        print(f"  {'arm':10s} {'bad bond %':>11s} {'CA-CA med':>10s} "
              f"{'CA-CA min':>10s} {'Rg med':>8s}  steps  n")
        for a in ARMS:
            r = rows[a]
            if not r:
                print(f"  {a:10s} {'(not run)':>11s}")
                continue
            print(f"  {a:10s} {r['bad_bond_pct']:11.2f} {r['ca_ca_median']:10.3f} "
                  f"{r['ca_ca_min']:10.3f} {r['rg_median'] or float('nan'):8.2f}"
                  f"  {r['n_step']:5d} {r['n_samples']:2d}")
        if "official" not in have:
            print("  !! the official reference was NOT re-measured in this batch; "
                  "do not compare against a number from another run")

    print("\n=== designability (AlphaProteo-10) ===")
    if not args.designability:
        print("  not supplied. Run submit_sc_env_feedback_alphaproteo.sh, then"
              "\n  pass --designability <designability_summary.csv>.")
    else:
        import pandas as pd
        df = pd.read_csv(args.designability)
        cov = [c for c in df.columns if "coverage" in c.lower()]
        if cov and (df[cov[0]] < 1.0).any():
            print(f"  !! coverage < 1.0 on some rows. A previous round read 0.0% "
                  f"for Proteo-AA when its scoring tasks had never run and the "
                  f"summariser counted missing scores as failures. Fix coverage "
                  f"before reading designability.")
        print(df.to_string(index=False))

    print("\n=== reference values (for context, not for mixing) ===")
    for measure, vals in REFERENCE.items():
        print(f"  {measure}")
        for k, v in vals.items():
            print(f"    {k:32s} {v:6.2f}%")
    print("\n  The 0.00% figure sometimes quoted for the official backbone comes "
          "\n  from a DIFFERENT probe (chain-gap aware, needs FaMPNN) and is not "
          "\n  comparable with anything above.")


if __name__ == "__main__":
    main()
