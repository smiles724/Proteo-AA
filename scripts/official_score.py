"""Score the official-runtime pilot: self-consistency first, guardrails beside it.

Primary comparison is quality -- self-consistency Ca RMSD and TM-score of
`full` against `bb_only`, since those two differ only in what the readout is
allowed to see. Each feedback arm against `baseline` is secondary. Interface
clashes, contacts and backbone chemistry are guardrails: they can veto a
result but they are not the result.

**Uncertainty is bootstrapped over targets, not over runs.** Two seeds of the
same target are not two independent observations of the effect -- they share
the target, its interface and its designed length. Resampling the eight
(target, seed) rows independently would treat four clusters as eight and
report intervals roughly sqrt(2) too narrow. Here whole targets are resampled
with replacement, carrying both their seeds and all their arms. With four
clusters the interval is wide and coarse by construction, which is why every
paired result is printed rather than summarised away.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pxf.eval.backbone_metrics import superposed_rmsd, tm_score

ARMS = ("baseline", "bb_only", "full")


def cluster_bootstrap(by_target, n_resamples=10000, alpha=0.05, seed=0):
    """Mean and percentile CI, resampling whole targets with replacement.

    ``by_target`` maps a target to the list of its per-run values. The
    statistic is the mean over every value in the resampled clusters, so a
    target contributes both its seeds or neither.
    """
    targets = sorted(by_target)
    if not targets:
        return None, None, None
    pooled = [v for t in targets for v in by_target[t]]
    point = float(np.mean(pooled))
    if len(targets) < 2:
        return point, None, None
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(targets), len(targets))
        vals = [v for j in pick for v in by_target[targets[j]]]
        draws[i] = np.mean(vals)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def load(base):
    rows, problems = [], []
    for d in sorted(p.parent for p in base.glob("*/single_event.json")):
        blob = json.loads((d / "single_event.json").read_text())
        meta, arms = blob.get("meta", {}), blob.get("arms", {})
        target = meta.get("target", d.name.rsplit("_s", 1)[0])
        seed = meta.get("seed")
        if not (d / "arms.npz").is_file():
            problems.append(dict(dir=d.name, problem="no arms.npz")); continue
        if not (d / "refold.npz").is_file():
            problems.append(dict(dir=d.name, problem="no refold.npz")); continue
        npz = np.load(d / "arms.npz", allow_pickle=True)
        ref = np.load(d / "refold.npz", allow_pickle=True)
        ca_ref = torch.as_tensor(ref["ca"], dtype=torch.float64)
        plddt = float(ref["mean_plddt"])
        for arm in ARMS:
            key = f"bb_{arm}"
            if key not in npz:
                problems.append(dict(dir=d.name, arm=arm, problem="arm missing"))
                continue
            bb = torch.as_tensor(npz[key], dtype=torch.float64)
            present = torch.as_tensor(npz[f"present_{arm}"])
            ca = bb[:, 1]
            ok = present[:, 1]
            n = min(len(ca), len(ca_ref))
            if n == 0 or int(ok[:n].sum()) < 3:
                problems.append(dict(dir=d.name, arm=arm, problem="too few CA"))
                continue
            if len(ca) != len(ca_ref):
                problems.append(dict(
                    dir=d.name, arm=arm,
                    problem=f"length {len(ca)} vs refold {len(ca_ref)}; truncated"))
            m = ok[:n]
            rows.append(dict(
                target=target, seed=seed, arm=arm,
                sc_rmsd=float(superposed_rmsd(ca[:n][m], ca_ref[:n][m])),
                tm=float(tm_score(ca[:n][m], ca_ref[:n][m])),
                mean_plddt=plddt,
                **{k: arms.get(arm, {}).get(k) for k in (
                    "min_dist", "clashes", "contacts", "overlap_depth_sum",
                    "overlap_depth_max", "ca_ca_mean", "ca_ca_max_dev",
                    "ca_ca_breaks", "ca_ca_compressions", "injections",
                    "residual_l2_mean", "seconds")},
            ))
    return rows, problems


def paired(rows, a, b, field):
    by = {(r["target"], r["seed"], r["arm"]): r for r in rows}
    keys = sorted({(r["target"], r["seed"]) for r in rows})
    out = {}
    for t, s in keys:
        ra, rb = by.get((t, s, a)), by.get((t, s, b))
        if ra is None or rb is None:
            continue
        if ra.get(field) is None or rb.get(field) is None:
            continue
        out.setdefault(t, []).append((s, ra[field] - rb[field]))
    return out


def show(rows, a, b, field, label):
    per_target = paired(rows, a, b, field)
    flat = {t: [v for _s, v in vs] for t, vs in per_target.items()}
    point, lo, hi = cluster_bootstrap(flat)
    if point is None:
        print(f"  {label}: no paired data")
        return
    every = "  ".join(
        f"{t}/{s}:{v:+.3f}" for t, vs in sorted(per_target.items()) for s, v in vs
    )
    ci = "n/a" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]"
    star = "" if (lo is None or lo <= 0 <= hi) else "   <-- excludes 0"
    print(f"  {label:26s} {point:+.4f}  {ci}  "
          f"({len(flat)} target clusters){star}")
    print(f"      every pair: {every}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.pilot_dir)
    rows, problems = load(base)
    if not rows:
        print("no scored rows", file=sys.stderr)
        return 1

    targets = sorted({r["target"] for r in rows})
    print(f"{len(rows)} arm-rows, {len(targets)} targets, "
          f"{len({(r['target'], r['seed']) for r in rows})} triplets\n")

    print("=== every run ===")
    hdr = (f"{'target':7s} {'seed':>4s} {'arm':9s} {'scRMSD':>7s} {'TM':>6s} "
           f"{'pLDDT':>6s} {'minD':>6s} {'clash':>5s} {'depth':>6s} "
           f"{'cont':>5s} {'CaCa':>6s} {'brk':>4s} {'s':>5s}")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["target"], r["seed"], r["arm"])):
        g = lambda k, d=float("nan"): r.get(k) if r.get(k) is not None else d
        print(f"{r['target']:7s} {r['seed']:4d} {r['arm']:9s} "
              f"{r['sc_rmsd']:7.3f} {r['tm']:6.3f} {g('mean_plddt'):6.2f} "
              f"{g('min_dist'):6.3f} {g('clashes',0):5.0f} "
              f"{g('overlap_depth_sum',0):6.3f} {g('contacts',0):5.0f} "
              f"{g('ca_ca_mean'):6.3f} {g('ca_ca_breaks',0):4.0f} "
              f"{g('seconds'):5.1f}")

    print("\n=== PRIMARY: self-consistency, full vs bb_only ===")
    show(rows, "full", "bb_only", "sc_rmsd", "sc_rmsd (A)")
    show(rows, "full", "bb_only", "tm", "TM-score")

    print("\n=== SECONDARY: each feedback arm vs baseline ===")
    for arm in ("bb_only", "full"):
        show(rows, arm, "baseline", "sc_rmsd", f"sc_rmsd  {arm}-baseline")
        show(rows, arm, "baseline", "tm", f"TM-score {arm}-baseline")

    print("\n=== GUARDRAILS (vetoes, not results) ===")
    for field in ("overlap_depth_sum", "min_dist", "contacts", "ca_ca_max_dev"):
        show(rows, "full", "bb_only", field, f"{field} full-bb_only")

    print("\n=== CONTEXT ===")
    pl = {t: sorted({r["mean_plddt"] for r in rows if r["target"] == t})
          for t in targets}
    print(f"  refold pLDDT by target (shared within a triplet): "
          f"{ {t: [round(x,1) for x in v] for t, v in pl.items()} }")
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    if secs:
        print(f"  per-arm sampling: mean {np.mean(secs):.1f}s, total {np.sum(secs):.0f}s")
    print(f"  problems: {problems if problems else 'none'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(rows=rows, problems=problems), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
