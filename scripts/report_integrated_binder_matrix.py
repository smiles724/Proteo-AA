#!/usr/bin/env python3
"""Report the integrated matrix: chemistry first, designability only if scored.

    python scripts/report_integrated_binder_matrix.py \
        --designs runs/integrated_feedback_v1/generation_smoke/designs.csv \
        --metrics-dir runs/integrated_feedback_v1/generation_smoke/metrics \
        --out runs/integrated_feedback_v1/generation_smoke/report

Every arm in this collection shares a bit-identical prefix per
``shared_prefix_id``, so comparisons are paired on that key rather than on a
seed that merely matches. Arms with no shared prefix in common are not
compared.

### Order of claims, deliberately

**Plumbing and chemistry first.** Injection counts, event-to-final
displacement, interface clashes. A pilot exists to find out whether the thing
runs and produces plausible complexes; that is the result it is entitled to
report.

**Designability only if AF2-IG metrics are present**, and then labelled a
pilot. A handful of targets at a handful of seeds is not a success-rate study,
and the four-way conjunction over a few dozen designs has a resolution of one
design.

**Failures stay in the denominator.** A design that errored and a design that
failed the filter are different facts, and merging them biases the rate
upward by however many jobs crashed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROTEO_AA = Path("/users/yfsun/Proteo-AA")


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", required=True)
    parser.add_argument("--metrics-dir", default=None,
                        help="AF2-IG metrics; omit for a chemistry-only report")
    parser.add_argument("--proteo-aa", default=str(PROTEO_AA))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(args.designs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = sorted({(r["arm"], r.get("bs_seed") or "") for r in rows})
    arms = [f"{a}@bs{s}" if s != "" else a for a, s in arms]

    # ---- plumbing ---------------------------------------------------------
    def _rows_for(label):
        arm, _, seed = label.partition("@bs")
        return [r for r in rows if r["arm"] == arm
                and (str(r.get("bs_seed") or "") == seed or seed == "")]

    plumbing = {}
    for arm in arms:
        subset = _rows_for(arm)
        injections = [int(r["conditioning_injections"] or 0) for r in subset]
        hooks = [int(r["decode_hook_calls"] or 0) for r in subset]
        plumbing[arm] = {
            "n": len(subset),
            "conditioning_injections": sorted(set(injections)),
            "decode_hook_calls": sorted(set(hooks)),
            "feedback_norm_mean": _mean(
                [number(r["feedback_norm"]) for r in subset]
            ),
            "delta_h_norm_mean": _mean(
                [number(r["delta_h_norm"]) for r in subset]
            ),
        }

    # ---- chemistry --------------------------------------------------------
    chemistry = {}
    for arm in arms:
        subset = _rows_for(arm)
        clashes = [int(r["interface_clashes"] or 0) for r in subset]
        mins = [number(r["min_bb_bb_distance"]) for r in subset]
        # Backbone-only is the discriminating number; all-atom under a 2.6 A
        # threshold flags 94 of 96 designs that went on to be AF2-IG scored.
        bb_mins = [number(r.get("min_bb_only_distance")) for r in subset]
        bb_clashes = [int(r.get("interface_clashes_bb_only") or 0)
                      for r in subset]
        transfer = [number(r["event_to_final_aligned_rmsd"]) for r in subset]
        chemistry[arm] = {
            "n": len(subset),
            "clash_free": sum(1 for c in clashes if c == 0),
            "clash_free_fraction": (
                sum(1 for c in clashes if c == 0) / len(clashes)
                if clashes else None
            ),
            "min_bb_bb_median": _median(mins),
            "min_bb_only_median": _median([m for m in bb_mins if m is not None]),
            "min_bb_only_worst": min([m for m in bb_mins if m is not None],
                                     default=None),
            "clash_free_bb_only": sum(1 for c in bb_clashes if c == 0),
            "min_bb_bb_worst": min([m for m in mins if m is not None],
                                   default=None),
            "event_to_final_aligned_rmsd_median": _median(transfer),
            "event_to_final_aligned_rmsd_range": [
                min([t for t in transfer if t is not None], default=None),
                max([t for t in transfer if t is not None], default=None),
            ],
        }

    # ---- paired comparisons on the shared prefix --------------------------
    # Keyed on (shared prefix, arm, A_BS seed). Keying on (prefix, arm) alone
    # collapsed both A_BS seeds' "J03" rows onto one entry, overwriting one
    # baseline during pairing.
    by_prefix = defaultdict(dict)
    for row in rows:
        by_prefix[row["shared_prefix_id"]][
            (row["arm"], row.get("bs_seed") or "")
        ] = row
    comparisons = []
    keys = sorted({k for d in by_prefix.values() for k in d})
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = [p for p, d in by_prefix.items() if a in d and b in d]
            if not shared:
                continue
            diffs = []
            for prefix in shared:
                x = number(by_prefix[prefix][a]["min_bb_bb_distance"])
                y = number(by_prefix[prefix][b]["min_bb_bb_distance"])
                if x is not None and y is not None:
                    diffs.append(x - y)
            comparisons.append({
                "comparison": f"{a[0]}@bs{a[1]} - {b[0]}@bs{b[1]}",
                "shared_prefixes": len(shared),
                "median_min_bb_bb_difference": _median(diffs),
                "identical_sequences": sum(
                    1 for p in shared
                    if by_prefix[p][a]["sequence"] == by_prefix[p][b]["sequence"]
                ),
                "note": "paired on shared_prefix_id, i.e. a bit-identical "
                        "prefix, not merely an equal seed",
            })

    report = {
        "n_designs": len(rows), "arms": arms,
        "plumbing": plumbing, "chemistry": chemistry,
        "comparisons": comparisons,
        "designability": None,
        "scope": "PILOT. Plumbing and chemistry only unless AF2-IG metrics "
                 "were supplied. Not a success-rate study and not a "
                 "reproduction of the benchmark.",
    }

    # ---- designability, only if metrics exist -----------------------------
    if args.metrics_dir and glob.glob(str(Path(args.metrics_dir) / "*.csv")):
        report["designability"] = _designability(
            args.metrics_dir, Path(args.proteo_aa), rows
        )

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"{len(rows)} design(s) over arm(s) {arms}\n")
    print("plumbing")
    for arm, facts in plumbing.items():
        print(f"  {arm:24s} n={facts['n']:3d} injections="
              f"{facts['conditioning_injections']} "
              f"decode_hooks={facts['decode_hook_calls']} "
              f"|feedback|={facts['feedback_norm_mean']}")
    print("\nchemistry")
    for arm, facts in chemistry.items():
        print(f"  {arm:24s} clash-free {facts['clash_free']}/{facts['n']}  "
              f"[bb-only] clash-free {facts['clash_free_bb_only']}/{facts['n']} "
              f"median {facts['min_bb_only_median']} worst "
              f"{facts['min_bb_only_worst']}  |  [all-atom] "
              f"min BB-BB median {facts['min_bb_bb_median']}  "
              f"event->final {facts['event_to_final_aligned_rmsd_median']} A")
    if comparisons:
        print("\npaired on the shared prefix")
        for c in comparisons:
            print(f"  {c['comparison']:34s} n={c['shared_prefixes']:3d}  "
                  f"identical sequences {c['identical_sequences']}  "
                  f"median min-BB-BB diff {c['median_min_bb_bb_difference']}")
    if report["designability"]:
        print("\ndesignability (PILOT)")
        for arm, cell in report["designability"]["per_arm"].items():
            print(f"  {arm:24s} {cell['successes']}/{cell['attempts']} "
                  f"{cell['rate']:.1%}  (failures kept in the denominator: "
                  f"{cell['errors']})")
    print(f"\nwrote {out / 'report.json'}")


def _designability(metrics_dir, proteo_aa, design_rows):
    import sys

    benchmarks = proteo_aa / "pxdesign_train" / "benchmarks"
    if not benchmarks.is_dir():
        return {"error": f"{benchmarks} not found; cannot import the filter"}
    sys.path.insert(0, str(benchmarks))
    from conditional_binder import AF2IGFilter

    af2ig = AF2IGFilter()
    rows = []
    for path in sorted(glob.glob(str(Path(metrics_dir) / "*.csv"))):
        rows.extend(load_rows(path))
    arm_of = {r["sample_id"]: r["arm"] for r in design_rows}

    per_arm = defaultdict(lambda: {"successes": 0, "attempts": 0, "errors": 0})
    for row in rows:
        arm = arm_of.get(row["sample_id"])
        if arm is None:
            continue
        cell = per_arm[arm]
        cell["attempts"] += 1
        try:
            if af2ig.is_designable({
                k: row.get(k) for k in
                ("ipae", "iptm", "plddt", "binder_bound_unbound_rmsd")
            }):
                cell["successes"] += 1
        except (KeyError, ValueError):
            cell["errors"] += 1
    # Designs that never produced a metric row are still attempts.
    for sample_id, arm in arm_of.items():
        if not any(r["sample_id"] == sample_id for r in rows):
            per_arm[arm]["attempts"] += 1
            per_arm[arm]["errors"] += 1
    return {
        "filter": {"ipae_max": af2ig.ipae_max, "iptm_min": af2ig.iptm_min,
                   "plddt_min": af2ig.plddt_min,
                   "binder_bound_unbound_rmsd_max":
                       af2ig.binder_bound_unbound_rmsd_max},
        "per_arm": {
            arm: {**cell,
                  "rate": cell["successes"] / cell["attempts"]
                  if cell["attempts"] else None}
            for arm, cell in sorted(per_arm.items())
        },
        "scope": "PILOT. Failures and unscored designs are kept in the "
                 "denominator; a crashed job and a filtered design are "
                 "different facts and merging them biases the rate upward.",
    }


def _mean(values):
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _median(values):
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 4) if clean else None


if __name__ == "__main__":
    main()
