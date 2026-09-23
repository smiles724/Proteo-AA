#!/usr/bin/env python3
"""Score the integrated cells as a table, with effective n, not design count.

    python scripts/report_integrated_cells.py <cell-dir> [<cell-dir> ...]

Each cell is one (target, length, generation seed) with seven arms. Within
an A_BS seed group the arms can emit the SAME sequence -- guaranteed under
`event_fixed`, where one event decode is shared -- so counting seven designs
as seven observations inflates n threefold. Every rate here is over DISTINCT
sequences, and both counts are printed.

Designability is the A-CODE four-way conjunction: ipAE < 10.85 A,
ipTM > 0.5, pLDDT > 80%, binder bound/unbound RMSD < 3.5 A.

This is NOT a Table 4 row and must not be formatted as one: these cells are
one target at one length, far below the n=48 per target the published rows
use. See docs/handoff/hai_table_prompt.md.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

CRITERIA = (
    ("ipAE<10.85", lambda m: float(m["ipae"]) < 10.85),
    ("ipTM>0.5", lambda m: float(m["iptm"]) > 0.5),
    ("pLDDT>80%", lambda m: float(m["plddt"]) > 0.80),
    ("buRMSD<3.5", lambda m: float(m["binder_bound_unbound_rmsd"]) < 3.5),
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def load(cell: Path):
    designs = {r["sample_id"]: r for r in
               csv.DictReader(open(cell / "designs.csv"))}
    metrics_path = cell / "af2ig_metrics.csv"
    metrics = ({r["sample_id"]: r for r in csv.DictReader(open(metrics_path))}
               if metrics_path.is_file() else {})
    return designs, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+")
    args = parser.parse_args()

    for path in args.cells:
        cell = Path(path)
        designs, metrics = load(cell)
        policy = next(iter(designs.values())).get("sequence_policy") or "event_fixed"
        n_events = next(iter(designs.values())).get("n_events") or "1"
        print(f"\n=== {cell.name}   policy={policy}  events={n_events} ===")
        if not metrics:
            print("  NOT SCORED -- no af2ig_metrics.csv")
            continue

        # The sequence actually folded, deduped.
        seq_of = {s: (d.get("output_sequence") or d["sequence"])
                  for s, d in designs.items()}
        by_seq: dict[str, list[str]] = {}
        for sid, seq in seq_of.items():
            by_seq.setdefault(seq, []).append(sid)

        print(f"  {len(designs)} designs, {len(by_seq)} distinct sequences "
              f"-> effective n = {len(by_seq)}")

        print(f"\n  {'arm':<22} {'bs':<3} {'ipAE':>6} {'ipTM':>6} "
              f"{'pLDDT':>6} {'buRMSD':>7}  {'abcd':<5} verdict")
        passes = {name: 0 for name, _ in CRITERIA}
        designable_seqs = set()
        for sid in sorted(designs, key=lambda s: (designs[s].get("bs_seed") or "",
                                                  designs[s]["arm"])):
            m = metrics.get(sid)
            if m is None:
                print(f"  {designs[sid]['arm']:<22} UNSCORED")
                continue
            flags = [test(m) for _name, test in CRITERIA]
            for (name, _t), ok in zip(CRITERIA, flags):
                passes[name] += int(ok)
            if all(flags):
                designable_seqs.add(seq_of[sid])
            d = designs[sid]
            print(f"  {d['arm']:<22} {d.get('bs_seed') or '-':<3} "
                  f"{float(m['ipae']):6.2f} {float(m['iptm']):6.3f} "
                  f"{float(m['plddt']):6.3f} "
                  f"{float(m['binder_bound_unbound_rmsd']):7.2f}  "
                  f"{''.join('Y' if f else '.' for f in flags):<5} "
                  f"{'PASS' if all(flags) else 'fail'}")

        n_eff = len(by_seq)
        k = len(designable_seqs)
        lo, hi = wilson(k, n_eff)
        print(f"\n  component pass rates (over {len(metrics)} designs):")
        for name, _t in CRITERIA:
            print(f"    {name:<12} {passes[name]}/{len(metrics)}")
        print(f"\n  DESIGNABLE {k}/{n_eff} distinct sequences = "
              f"{100 * k / n_eff:.1f}%  95% Wilson [{lo:.1f}, {hi:.1f}]")
        print(f"  finest expressible non-zero rate at n={n_eff}: "
              f"{100 / n_eff:.1f}%")

    print("\nNOT a Table 4 row: one target at one length, against n=48 per "
          "target in the published rows. Report as a pilot.")


if __name__ == "__main__":
    main()
