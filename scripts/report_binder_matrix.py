#!/usr/bin/env python3
"""Table 4 designability per arm, and the paired differences between arms.

    python scripts/report_binder_matrix.py \
        --metrics-dir runs/binder_bench/metrics \
        --out runs/binder_bench/report

The four-way conjunction is imported from `pxdesign_train.benchmarks.
conditional_binder.AF2IGFilter` in the Proteo-AA checkout rather than restated
here -- ipAE < 10.85, ipTM > 0.5, pLDDT > 80%, bound/unbound RMSD < 3.5 A --
because a threshold copied into a second file is a threshold that will
eventually disagree with itself.

**Pooled, not averaged.** A-CODE sums success counts across the length grid
for a target. Averaging the per-length rates is a different number whenever
the per-length counts differ, so the per-target rate here is
``sum(successes) / sum(attempts)``.

### Why the comparisons are paired

Every arm designed on the SAME backbone, so a design_id is a matched triple
and the unit of comparison is the design, not the arm's marginal rate. For a
binary outcome that makes McNemar's test the right one: it conditions on the
discordant pairs and ignores the designs both arms got right or both got
wrong, which are uninformative about a difference. A two-sample test on the
marginal rates would throw the pairing away and be needlessly conservative.

### What each comparison licenses

``J03 - U03`` is the adapter effect: same donor, same context, same backbone,
residual on vs off. That one is clean.

``J03 - R0`` and ``U03 - R0`` change the designer AND the context level at
once, and R0's AF2 initial guess is backbone-only where the FaMPNN arms get
their packed side chains (`colabdesign/af/design.py:166` feeds
``all_atom_positions`` into ``prev_pos``). Both are reported as designer
comparisons and neither is an adapter gain.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROTEO_AA = Path("/users/yfsun/Proteo-AA")
ARM_ORDER = ("J03", "U03", "R0")


def load_filter(proteo_aa: Path):
    """A-CODE's own filter object, so the thresholds are defined in one place."""
    benchmarks = proteo_aa / "pxdesign_train" / "benchmarks"
    if not benchmarks.is_dir():
        raise SystemExit(
            f"{benchmarks} not found; pass --proteo-aa at the checkout that "
            "holds pxdesign_train/benchmarks/conditional_binder.py"
        )
    sys.path.insert(0, str(benchmarks))
    from conditional_binder import AF2IGFilter  # noqa: E402

    return AF2IGFilter()


def read_metrics(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(metrics_dir / "*.csv"))):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no metric rows under {metrics_dir}")
    return rows


def arm_of(sample_id: str) -> str:
    """`<design_id>__<arm>` -- the arm is encoded in the id the scorer echoes."""
    if "__" not in sample_id:
        raise SystemExit(f"{sample_id}: no '__<arm>' suffix; cannot tell the arm")
    return sample_id.rsplit("__", 1)[1]


def design_of(sample_id: str) -> str:
    return sample_id.rsplit("__", 1)[0]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval: behaves at 0 and at n, which Wald does not.

    Designability at 48 backbones per target lands on 0/48 often enough that
    a normal approximation would produce a zero-width interval there and
    invite the reading that the rate is known exactly.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts.

    ``b`` = first arm succeeded and second failed, ``c`` = the reverse. Exact
    binomial rather than the chi-square approximation, because the discordant
    counts here are small enough that the approximation is not trustworthy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--proteo-aa", default=str(PROTEO_AA))
    parser.add_argument("--allow-missing", action="store_true",
                        help="count a row with a missing metric as non-designable "
                             "instead of refusing")
    args = parser.parse_args()

    af2ig = load_filter(Path(args.proteo_aa))
    rows = read_metrics(Path(args.metrics_dir))

    # design_id -> arm -> bool
    outcome: dict[str, dict[str, bool]] = defaultdict(dict)
    target_of: dict[str, str] = {}
    length_of: dict[str, int] = {}
    missing = 0
    for row in rows:
        sid = row["sample_id"]
        arm, did = arm_of(sid), design_of(sid)
        # is_designable takes a dict, not keywords. TypeError is deliberately
        # NOT caught below: calling it wrong is my bug, and an earlier version
        # swallowed exactly that into the "missing metric" branch and reported
        # a complete row as a failed fold.
        try:
            ok = af2ig.is_designable({
                k: row.get(k) for k in
                ("ipae", "iptm", "plddt", "binder_bound_unbound_rmsd")
            })
        except (KeyError, ValueError):
            missing += 1
            if not args.allow_missing:
                raise SystemExit(
                    f"{sid}: a metric is missing or unparseable. A fold that "
                    "failed and a design that failed the filter are different "
                    "facts; pass --allow-missing to count it as non-designable."
                )
            ok = False
        outcome[did][arm] = ok
        target_of[did] = row["target"]
        length_of[did] = int(float(row["binder_length"]))

    arms = sorted({a for v in outcome.values() for a in v},
                  key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))
    complete = [d for d, v in outcome.items() if len(v) == len(arms)]
    incomplete = len(outcome) - len(complete)

    # ---- per-target, pooled over lengths ---------------------------------
    per_target: dict[str, dict[str, dict]] = defaultdict(dict)
    for arm in arms:
        by_target: dict[str, list[bool]] = defaultdict(list)
        for did in complete:
            by_target[target_of[did]].append(outcome[did][arm])
        for target, oks in sorted(by_target.items()):
            n, s = len(oks), sum(oks)
            lo, hi = wilson(s, n)
            per_target[target][arm] = {
                "successes": s, "attempts": n, "rate": s / n,
                "ci95": [lo, hi],
            }

    overall = {}
    for arm in arms:
        oks = [outcome[d][arm] for d in complete]
        n, s = len(oks), sum(oks)
        lo, hi = wilson(s, n)
        overall[arm] = {"successes": s, "attempts": n, "rate": s / n,
                        "ci95": [lo, hi]}

    # ---- paired comparisons ----------------------------------------------
    comparisons = []
    pairs = [("J03", "U03"), ("J03", "R0"), ("U03", "R0")]
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        only_a = sum(1 for d in complete if outcome[d][a] and not outcome[d][b])
        only_b = sum(1 for d in complete if outcome[d][b] and not outcome[d][a])
        both = sum(1 for d in complete if outcome[d][a] and outcome[d][b])
        neither = len(complete) - only_a - only_b - both
        comparisons.append({
            "comparison": f"{a} - {b}",
            "rate_difference": overall[a]["rate"] - overall[b]["rate"],
            f"only_{a}": only_a, f"only_{b}": only_b,
            "both": both, "neither": neither,
            "discordant": only_a + only_b,
            "mcnemar_p": mcnemar(only_a, only_b),
            "paired_n": len(complete),
            "kind": ("adapter effect: same donor, same context, residual on/off"
                     if {a, b} == {"J03", "U03"} else
                     "designer comparison: designer AND context differ, and R0's "
                     "initial guess is backbone-only"),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "filter": {
            "ipae_max": af2ig.ipae_max, "iptm_min": af2ig.iptm_min,
            "plddt_min": af2ig.plddt_min,
            "binder_bound_unbound_rmsd_max": af2ig.binder_bound_unbound_rmsd_max,
        },
        "pooling": "successes summed across the length grid, per A-CODE",
        "n_designs_scored": len(outcome),
        "n_complete_across_arms": len(complete),
        "n_incomplete_dropped": incomplete,
        "n_missing_metrics": missing,
        "arms": arms,
        "overall": overall,
        "per_target": per_target,
        "comparisons": comparisons,
    }
    (out / "designability.json").write_text(json.dumps(report, indent=2))

    with (out / "designability_per_target.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "arm", "successes", "attempts", "rate",
                         "ci95_lo", "ci95_hi"])
        for target in sorted(per_target):
            for arm in arms:
                cell = per_target[target].get(arm)
                if cell:
                    writer.writerow([target, arm, cell["successes"],
                                     cell["attempts"], f"{cell['rate']:.4f}",
                                     f"{cell['ci95'][0]:.4f}",
                                     f"{cell['ci95'][1]:.4f}"])

    # ---- human-readable ---------------------------------------------------
    print(f"filter: ipAE<{af2ig.ipae_max} ipTM>{af2ig.iptm_min} "
          f"pLDDT>{af2ig.plddt_min} RMSD<{af2ig.binder_bound_unbound_rmsd_max}")
    print(f"{len(complete)} design(s) complete across {len(arms)} arm(s)"
          + (f"; {incomplete} incomplete dropped" if incomplete else ""))
    print("\nDesignability, pooled across lengths (successes/attempts):")
    header = f"{'target':10s}" + "".join(f"{a:>18s}" for a in arms)
    print(header)
    for target in sorted(per_target):
        line = f"{target:10s}"
        for arm in arms:
            c = per_target[target].get(arm)
            line += f"{c['successes']:>8d}/{c['attempts']:<4d}{c['rate']:>5.1%}" if c else f"{'-':>18s}"
        print(line)
    line = f"{'ALL':10s}"
    for arm in arms:
        c = overall[arm]
        line += f"{c['successes']:>8d}/{c['attempts']:<4d}{c['rate']:>5.1%}"
    print(line)
    print("\nPaired comparisons (McNemar, exact, on the matched triples):")
    for c in comparisons:
        a, b = c["comparison"].split(" - ")
        print(f"  {c['comparison']:12s} rate diff {c['rate_difference']:+.3%}  "
              f"only_{a}={c[f'only_{a}']} only_{b}={c[f'only_{b}']} "
              f"(discordant {c['discordant']})  p={c['mcnemar_p']:.3g}")
        print(f"               {c['kind']}")
    print(f"\nwrote {out / 'designability.json'}")


if __name__ == "__main__":
    main()
