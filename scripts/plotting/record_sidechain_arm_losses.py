#!/usr/bin/env python3
"""Record side-chain warm-up losses per ablation arm, for A/B comparison.

Why not `plot_train_val_metrics.py`: that script is a plotter, and it collapses
every logged step to ONE row (`drop_duplicates(..., keep="last")`). A step emits
one line per gradient-accumulation micro-batch (8 by default), so it discards
7/8 of the training signal and the survivor is an arbitrary micro-batch rather
than a step aggregate. For an arm-vs-arm comparison the per-step spread is the
point -- two arms can share a median and differ entirely in their tail.

Emits, per arm:
  <out_dir>/sidechain_arm_<label>_train.csv   per-step aggregate over micro-batches
  <out_dir>/sidechain_arm_<label>_val.csv     the 491-protein validation points
  <out_dir>/sidechain_arm_<label>_summary.json

`sc_local` is a masked mean of squared 3-D displacement (see
pxdesign_train/sidechain/losses.py), i.e. A^2 per atom -- the summaries carry a
`_rmsd_A` twin so the numbers can be read as distances without re-deriving it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

STEP_RE = re.compile(r"\bstep=(\d+)\b")
KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)")

# Only these are aggregated; everything else in the line is per-micro-batch noise
# that means nothing averaged across structures of different size.
TRAIN_METRICS = ("loss", "sc_local", "sc_phys", "sc_global")
VAL_METRICS = ("loss", "sc_local", "sc_phys", "sc_global")


def _parse_one(path: Path) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    train: dict[int, list[dict]] = {}
    val: dict[int, dict] = {}
    with path.open("r", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            pairs = {k: float(v) for k, v in KV_RE.findall(line) if k != "step"}
            if not pairs:
                continue
            if any(k.startswith("val_") for k in pairs):
                # Per-protein lines carry val_index; only the roll-up has val_n.
                if "val_n" not in pairs:
                    continue
                val[step] = {
                    k.removeprefix("val_"): v
                    for k, v in pairs.items()
                    if k.startswith("val_")
                }
            else:
                train.setdefault(step, []).append(pairs)
    return train, val


def parse(paths: list[Path]) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Return (train rows keyed by step, aggregate val rows keyed by step).

    A validation block logs one line per protein plus one `val_n=` aggregate; we
    keep only the aggregate, which is the number the run is judged on.

    The trainer emits every metric line TWICE -- once via `print` (-> .out) and
    once via `logging` (-> .err) -- with identical payloads. Callers pass both
    files because either can be truncated, so we merge per step by keeping the
    file that saw the most micro-batches rather than unioning them; a union
    would double `n` (harmless for median/p90, wrong for anything counting).
    """
    train: dict[int, list[dict]] = {}
    val: dict[int, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        t, v = _parse_one(path)
        for step, rows in t.items():
            if len(rows) > len(train.get(step, ())):
                train[step] = rows
        val.update(v)
    return train, val


def _stats(values: list[float]) -> dict[str, float]:
    vs = sorted(values)
    n = len(vs)

    def q(p: float) -> float:
        if n == 1:
            return vs[0]
        i = p * (n - 1)
        lo = math.floor(i)
        hi = math.ceil(i)
        return vs[lo] + (vs[hi] - vs[lo]) * (i - lo)

    return {
        "n": n,
        "mean": sum(vs) / n,
        "median": q(0.5),
        "p90": q(0.9),
        "max": vs[-1],
    }


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join("" if v is None else str(v) for v in row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", nargs="+", type=Path, required=True)
    p.add_argument("--label", required=True, help="arm name, e.g. local_head / global_head")
    p.add_argument("--out-dir", type=Path, default=Path("runs/metrics"))
    p.add_argument("--bucket", type=int, default=10000, help="step bucket for the summary")
    args = p.parse_args()

    train, val = parse(args.logs)
    if not train:
        raise SystemExit(f"no training rows found in {[str(x) for x in args.logs]}")

    # ---- per-step training aggregate -------------------------------------
    header = ["step", "n_micro"]
    for metric in TRAIN_METRICS:
        header += [f"{metric}_{s}" for s in ("mean", "median", "p90", "max")]
    rows = []
    for step in sorted(train):
        batch = train[step]
        row = [step, len(batch)]
        for metric in TRAIN_METRICS:
            vals = [b[metric] for b in batch if metric in b]
            if vals:
                st = _stats(vals)
                row += [round(st["mean"], 5), round(st["median"], 5),
                        round(st["p90"], 5), round(st["max"], 5)]
            else:
                row += [None] * 4
        rows.append(row)
    train_csv = args.out_dir / f"sidechain_arm_{args.label}_train.csv"
    write_csv(train_csv, header, rows)

    # ---- validation ------------------------------------------------------
    val_csv = args.out_dir / f"sidechain_arm_{args.label}_val.csv"
    if val:
        vheader = ["step", "n"] + list(VAL_METRICS)
        vrows = [
            [s, int(val[s].get("n", 0))] + [val[s].get(m) for m in VAL_METRICS]
            for s in sorted(val)
        ]
        write_csv(val_csv, vheader, vrows)

    # ---- summary ---------------------------------------------------------
    buckets = {}
    for step in sorted(train):
        lo = (step - 1) // args.bucket * args.bucket
        buckets.setdefault(lo, []).extend(
            b["sc_local"] for b in train[step] if "sc_local" in b
        )
    summary = {
        "label": args.label,
        "logs": [str(x) for x in args.logs],
        "unit": "A^2 (masked mean squared displacement per side-chain atom)",
        "steps_logged": len(train),
        "micro_batch_rows": sum(len(v) for v in train.values()),
        "step_max": max(train),
        "train_sc_local_by_bucket": {
            f"{lo + 1}-{lo + args.bucket}": {
                **{k: round(v, 5) for k, v in _stats(vs).items()},
                "median_rmsd_A": round(math.sqrt(_stats(vs)["median"]), 4),
            }
            for lo, vs in sorted(buckets.items())
            if vs
        },
    }
    if val:
        vsteps = sorted(val)
        finite = [(s, val[s]["sc_local"]) for s in vsteps if "sc_local" in val[s]]
        if finite:
            best_step, best = min(finite, key=lambda kv: kv[1])
            summary["val"] = {
                "n_points": len(finite),
                "n_proteins": int(val[vsteps[-1]].get("n", 0)),
                "first": {"step": finite[0][0], "sc_local": finite[0][1],
                          "rmsd_A": round(math.sqrt(finite[0][1]), 4)},
                "best": {"step": best_step, "sc_local": best,
                         "rmsd_A": round(math.sqrt(best), 4)},
                "final": {"step": finite[-1][0], "sc_local": finite[-1][1],
                          "rmsd_A": round(math.sqrt(finite[-1][1]), 4)},
            }
    summary_path = args.out_dir / f"sidechain_arm_{args.label}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote_train={train_csv}")
    if val:
        print(f"wrote_val={val_csv}")
    print(f"wrote_summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
