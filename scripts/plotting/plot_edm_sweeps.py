#!/usr/bin/env python3
"""Every arm on one axis: the EDM variants and the one-step arms together.

EVERY CURVE IS THE SAME NUMBER -- unweighted masked MSE over the supervised
side-chain atoms -- but it reaches the plot by two routes, and the difference is
worth stating because getting it wrong is the mistake this series of figures
keeps having to avoid:

  one-step arms : their TRAINING-time validation, which already is that metric
                  (fixed sigma_T, no weighting). Dense, every 2000 steps.
  EDM arms      : the SWEEP, i.e. `eval_sidechain_arms.py` re-scoring saved
                  checkpoints. Their training loss is lambda(sigma)-weighted and
                  lives on a different scale, so it can never share this axis.
                  Sparse, at the swept checkpoints.

The two routes were checked against each other on the same checkpoints: for
fixed_global they agree to 0.1-0.5% (1.854 vs 1.8561 at 48k, 1.863 vs 1.8530 at
50k). That agreement is what licenses putting them on one axis at all.

The ground-truth-start diagnostic is drawn as an open marker and never joined to
a line. It begins the reverse loop from the noised TARGET, so it is not reachable
at inference; it is here because the distance between it and the deployable point
IS the finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Documented categorical palette, light mode, unmodified. Slots 3 and 4 are
# adjacent in the fixed order, which is the pairlist a line chart uses; aqua
# stays "EDM" as in the other figure in this series.
GLOBAL_HUE = "#2a78d6"  # slot 1, blue   -- same meaning as in plot_val_comparison
LOCAL_HUE = "#eb6834"   # slot 2, orange
V1_HUE = "#1baf7a"      # slot 3, aqua   -- "EDM" in this series
V2_HUE = "#eda100"      # slot 4, yellow
DIAG_HUE = "#e34948"  # slot 8, red -- reserved-feeling on purpose: not a result
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def read_sweep(path: Path):
    if not path.exists():
        return [], []
    s, v = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            s.append(int(row["step"]))
            v.append(float(row["atom_weighted_mse"]))
    order = sorted(range(len(s)), key=lambda i: s[i])
    return [s[i] for i in order], [v[i] for i in order]


ONE_STEP = [
    ("fixed_global", "one-step global", GLOBAL_HUE, False),
    ("fixed_local", "one-step local", LOCAL_HUE, False),
    ("global_head", "one-step global, pre-fix", GLOBAL_HUE, True),
    ("local_head", "one-step local, pre-fix", LOCAL_HUE, True),
]


def read_val(metrics: Path, label: str):
    path = metrics / f"sidechain_arm_{label}_val.csv"
    if not path.exists():
        return [], []
    s, v = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("sc_local"):
                s.append(int(row["step"]))
                v.append(float(row["sc_local"]))
    return s, v


def read_point(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())["atom_weighted_mse"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path,
                   default=Path("/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup"))
    p.add_argument("--metrics-dir", type=Path, default=Path("runs/metrics"))
    p.add_argument("--template-baseline", type=float, default=4.664)
    p.add_argument("--no-prefix-arms", action="store_true",
                   help="drop the two pre-fix curves; they answer a different "
                        "question (the frame refactor) and are documented in "
                        "docs/prefix_vs_fixed_global.md")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    v1s, v1v = read_sweep(args.runs / "arm_sweep_edm_global" / "sweep_edm_global.csv")
    v2s, v2v = read_sweep(args.runs / "arm_sweep_edm_v2" / "sweep_edm_v2.csv")
    diag = read_point(args.runs / "arm_sweep_edm_global"
                      / "arm_eval_edm_global_step42000_gtstart.json")
    series = [(lbl, name, hue, dash) for lbl, name, hue, dash in ONE_STEP
              if not (dash and args.no_prefix_arms)]
    one_step = {lbl: read_val(args.metrics_dir, lbl) for lbl, *_ in series}
    one_step = {k: v for k, v in one_step.items() if v[0]}
    if not v1s and not v2s and not one_step:
        raise SystemExit("nothing to plot; run the sweeps / recorder first")

    all_steps = v1s + v2s + [s for sv, _ in one_step.values() for s in sv] + [50000]
    all_vals = v1v + v2v + [x for _, vv in one_step.values() for x in vv]
    xmax = max(all_steps)
    fig, ax = plt.subplots(figsize=(11.8, 5.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # --- references, drawn first so the data sits on top ---
    ax.axhline(args.template_baseline, color=MUTED, linewidth=1.2,
               linestyle=(0, (5, 4)), zorder=1)
    ax.text(xmax * 0.015, args.template_baseline + 0.09,
            f"ideal-rotamer template  {args.template_baseline:.2f} Å²  "
            f"({math.sqrt(args.template_baseline):.2f} Å)",
            color=MUTED, fontsize=8.5, va="bottom")
    # --- one-step arms: dense training-time validation, same metric ---
    ends = []
    for lbl, name, hue, dash in series:
        if lbl not in one_step:
            continue
        sv, vv = one_step[lbl]
        ax.plot(sv, vv, color=hue, linewidth=2.0, zorder=3,
                linestyle=(0, (4, 3)) if dash else "solid",
                alpha=0.45 if dash else 1.0)
        ends.append([vv[-1], sv[-1], f"{name}   {min(vv):.3f} Å²  "
                                     f"({math.sqrt(min(vv)):.3f} Å)"])

    # --- the two EDM arms ---
    # Markers at every point, because these come from a sparse re-scoring of saved
    # checkpoints rather than from a dense training log -- the reader should see
    # where the measurements actually are.
    for st, v, hue, name in ((v1s, v1v, V1_HUE, "EDM v1  σ_max=4"),
                             (v2s, v2v, V2_HUE, "EDM v2  σ_max=40")):
        if not st:
            continue
        ax.plot(st, v, color=hue, linewidth=2.0, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4)
        ends.append([v[-1], st[-1], f"{name}   {v[-1]:.3f} Å²  "
                                    f"({math.sqrt(v[-1]):.3f} Å)"])

    # Labels pushed apart from the bottom up: the two one-step arms end 0.09 A^2
    # apart, invisible at this scale.
    ends.sort(key=lambda e: e[0])
    top = max(8.0, max(all_vals) * 1.06)
    gap = top * 0.052
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for y_lab, x_end, text in ends:
        ax.annotate(text, xy=(x_end, y_lab), xytext=(9, 0),
                    textcoords="offset points", color=INK_2, fontsize=8.5,
                    va="center", ha="left", annotation_clip=False)

    # --- the diagnostic, deliberately not joined to anything ---
    if diag is not None and v1s:
        ax.plot([42000], [diag], marker="o", markersize=9, markerfacecolor="none",
                markeredgecolor=DIAG_HUE, markeredgewidth=2.0, zorder=4)
        ax.annotate(
            f"v1 @42k started from the noised TARGET: {diag:.3f} Å²\n"
            "DIAGNOSTIC — uses the answer, not reachable at inference",
            xy=(42000, diag), xytext=(-14, -34), textcoords="offset points",
            color=DIAG_HUE, fontsize=8.5, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", color=DIAG_HUE, linewidth=0.9,
                            shrinkA=0, shrinkB=6),
        )

    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(0, top)
    ax.set_xlabel("training step", color=INK_2, fontsize=9.5)
    ax.set_ylabel("side-chain MSE, own inference protocol  (Å²)",
                  color=INK_2, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(
        lambda x, _: "0" if not x else (f"{x/1000:g}k" if (x / 1000) == int(x / 1000)
                                        else f"{x/1000:.1f}k")
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)

    fig.subplots_adjust(left=0.068, right=0.735, top=0.700, bottom=0.105)
    fig.text(0.010, 0.960,
             "Both EDM variants sit far above the one-step arms",
             color=INK, fontsize=13.5, ha="left", va="top")
    fig.text(0.010, 0.893,
             "491 recent-PDB monomers · one metric throughout: unweighted side-chain "
             "MSE, each arm under its own inference protocol\n"
             "one-step curves are training-time validation (dense lines); EDM curves "
             "are saved checkpoints re-scored (markers), because an\n"
             "EDM training loss is λ(σ)-weighted and cannot share this axis",
             color=INK_2, fontsize=9.5, ha="left", va="top", linespacing=1.5)

    handles = [
        plt.Line2D([], [], color=GLOBAL_HUE, linewidth=2.0, label="one-step global head"),
        plt.Line2D([], [], color=LOCAL_HUE, linewidth=2.0, label="one-step local head"),
        plt.Line2D([], [], color=V1_HUE, linewidth=2.0, marker="o", markersize=4,
                   label="EDM v1  (σ_max=4, 8 steps)"),
        plt.Line2D([], [], color=V2_HUE, linewidth=2.0, marker="o", markersize=4,
                   label="EDM v2  (σ_max=40, 16 steps)"),
        plt.Line2D([], [], color=MUTED, linewidth=2.0, linestyle=(0, (4, 3)),
                   alpha=0.6, label="pre-fix (frame refactor)"),
        plt.Line2D([], [], color=DIAG_HUE, linewidth=0, marker="o", markersize=8,
                   markerfacecolor="none", markeredgewidth=2.0,
                   label="diagnostic — uses the answer"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.007, 0.790),
                     frameon=False, fontsize=9, ncol=3, handlelength=2.2,
                     columnspacing=1.8, borderpad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote_figure={args.out}")


if __name__ == "__main__":
    main()
