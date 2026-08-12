#!/usr/bin/env python3
"""Overlay the arms' validation curves — with the incomparable one held apart.

TWO PANELS, ON PURPOSE. The four one-step arms share a metric: unweighted masked
MSE over the supervised side-chain atoms, at the fixed sigma_T the template init
uses. They belong on one axis.

The EDM arm does not. Its training-time validation draws a random sigma per item
and weights by lambda(sigma) = 1/c_out^2, which grows without bound as sigma
falls. At low sigma the input is the target plus a whisker of noise, so the task
is nearly trivial AND carries enormous weight, and the weighted mean collapses
towards zero. Its ~0.33 is not three times better than the others' ~1.9; it is a
different quantity. Putting the two on one scale would be the same failure as
reporting the pre-fix 125 A^2 as model quality when it was an unresolved-atom
floor.

So: same figure, separate panels, separate axes, and the right-hand panel says
what it is. `eval_sidechain_arms.py` is what produces numbers that CAN be put
side by side — each arm under its own inference protocol, all scored unweighted.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Documented categorical palette, light mode, unmodified. Colour encodes the ARM
# (global vs local); dash encodes whether the frame fixes are in. Two hues rather
# than four keeps the pairing visible instead of making the reader decode a
# four-entry legend.
GLOBAL_HUE = "#2a78d6"   # slot 1, blue
LOCAL_HUE = "#eb6834"    # slot 2, orange
EDM_HUE = "#1baf7a"      # slot 3, aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

TEMPLATE_BASELINE = 4.664   # A^2, ideal-rotamer template over the same 491

# label -> (display name, hue, dashed, panel)
SERIES = [
    ("global_head", "global, pre-fix", GLOBAL_HUE, True),
    ("local_head", "local, pre-fix", LOCAL_HUE, True),
    ("fixed_global", "global, fixed", GLOBAL_HUE, False),
    ("fixed_local", "local, fixed", LOCAL_HUE, False),
]


def read_val(metrics: Path, label: str):
    path = metrics / f"sidechain_arm_{label}_val.csv"
    if not path.exists():
        return [], []
    steps, vals = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("sc_local"):
                steps.append(int(row["step"]))
                vals.append(float(row["sc_local"]))
    return steps, vals


def style_axes(ax, xmax, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax * 1.01)
    ax.set_xlabel("training step", color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", type=Path, default=Path("runs/metrics"))
    p.add_argument("--edm-label", default="edm_global")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = {lbl: read_val(args.metrics_dir, lbl) for lbl, *_ in SERIES}
    data = {k: v for k, v in data.items() if v[0]}
    if not data:
        raise SystemExit(f"no val curves found under {args.metrics_dir}")
    edm_steps, edm_vals = read_val(args.metrics_dir, args.edm_label)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15.2, 5.8), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.9, 1.0]},
    )

    # ---------------- left: the four comparable arms ----------------
    xmax = max(max(s) for s, _ in data.values())
    style_axes(axL, xmax, "validation  sc_local  (Å²)")
    axL.axhline(TEMPLATE_BASELINE, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    axL.text(xmax * 0.02, TEMPLATE_BASELINE + 0.1,
             f"ideal-rotamer template  {TEMPLATE_BASELINE:.2f} Å²  "
             f"({math.sqrt(TEMPLATE_BASELINE):.2f} Å)",
             color=MUTED, fontsize=8.5, va="bottom")

    ytop = max(6.2, TEMPLATE_BASELINE + 1.4)
    axL.set_ylim(0, ytop)
    ends = []
    for lbl, name, hue, dashed in SERIES:
        if lbl not in data:
            continue
        s, v = data[lbl]
        axL.plot(s, v, color=hue, linewidth=2.0, zorder=3,
                 linestyle=(0, (4, 3)) if dashed else "solid",
                 alpha=0.55 if dashed else 1.0,
                 marker="o", markersize=3.2,
                 markeredgecolor=SURFACE, markeredgewidth=0.7)
        axL.plot([s[-1]], [v[-1]], marker="o", markersize=6.0, color=hue,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                 alpha=0.55 if dashed else 1.0)
        best = min(v)
        ends.append([v[-1], s[-1], hue,
                     f"{name}   {best:.3f} Å²  ({math.sqrt(best):.3f} Å)"])
    # The two fixed arms end 0.06 A^2 apart, which is invisible at this scale, so
    # their labels would sit on top of each other. Push them apart from the bottom
    # up and draw a leader to the curve each one belongs to.
    ends.sort(key=lambda e: e[0])
    min_gap = ytop * 0.058
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < min_gap:
            ends[i][0] = ends[i - 1][0] + min_gap
    for y_lab, x_end, hue, text in ends:
        axL.annotate(
            text, xy=(x_end, y_lab), xytext=(8, 0), textcoords="offset points",
            color=INK_2, fontsize=8.5, va="center", ha="left",
            annotation_clip=False,
        )
    axL.set_title("same metric, one axis", color=INK, fontsize=10.5, loc="left", pad=8)

    # ---------------- right: EDM, held apart ----------------
    if edm_steps:
        style_axes(axR, max(edm_steps), "λ(σ)-weighted  sc_local")
        axR.plot(edm_steps, edm_vals, color=EDM_HUE, linewidth=2.0, marker="o",
                 markersize=3.2, markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
        axR.set_ylim(0, max(edm_vals) * 1.15)
        # Only one series here and the title already names it, so the label goes
        # to empty space rather than onto a descending line.
        axR.text(0.97, 0.96,
                 f"EDM, fixed\nbest {min(edm_vals):.3f}  ·  last {edm_vals[-1]:.3f}",
                 transform=axR.transAxes, color=INK_2, fontsize=8.5,
                 ha="right", va="top", linespacing=1.5)
        axR.set_title("DIFFERENT METRIC — not comparable to the left",
                      color=INK, fontsize=10.5, loc="left", pad=8)
        axR.text(
            0.07, 0.36,
            "random σ per item, weighted by λ(σ)=1/c_out².\n"
            "λ→∞ as σ→0, where the task is near-trivial,\n"
            "so the weighted mean collapses toward zero.\n"
            "NOT an accuracy in Å².",
            transform=axR.transAxes, color=MUTED, fontsize=8.5,
            ha="left", va="top", linespacing=1.6,
        )
    else:
        axR.axis("off")

    fig.subplots_adjust(left=0.056, right=0.975, top=0.715, bottom=0.108, wspace=0.50)
    fig.text(0.010, 0.965,
             "Frame fixes moved the needle far more than the local-vs-global choice",
             color=INK, fontsize=13.5, ha="left", va="top")
    fig.text(0.010, 0.900,
             "Side-chain warm-up on 491 recent-PDB monomers · dashed = before the "
             "single-frame refactor, solid = after · physical loss off in every arm",
             color=INK_2, fontsize=9.5, ha="left", va="top")

    handles = [
        plt.Line2D([], [], color=GLOBAL_HUE, linewidth=2.0, label="global head"),
        plt.Line2D([], [], color=LOCAL_HUE, linewidth=2.0, label="local head (frame-aware + template residual)"),
        plt.Line2D([], [], color=MUTED, linewidth=2.0, linestyle=(0, (4, 3)), label="pre-fix"),
        plt.Line2D([], [], color=MUTED, linewidth=2.0, label="fixed"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.007, 0.828),
                     frameon=False, fontsize=9, ncol=4, handlelength=2.4,
                     columnspacing=2.0, borderpad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote_figure={args.out}")


if __name__ == "__main__":
    main()
