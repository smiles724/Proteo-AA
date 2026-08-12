#!/usr/bin/env python3
"""Plot side-chain warm-up loss curves, one panel per ablation arm.

Reads what `record_sidechain_arm_losses.py` wrote, so the figure and the numbers
in the report cannot drift apart. Small multiples rather than more hues: adding
an arm adds a panel, never a series, so `train` and `validation` keep the same
two colours in every panel and across every future arm.

`sc_local` is A^2 (masked mean squared displacement per side-chain atom); the
right-hand annotations carry the sqrt so the axis can be read as a distance
without a second scale.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Slots 1-2 of the documented categorical palette, light mode, unmodified.
TRAIN = "#2a78d6"
VAL = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# eval_sidechain_template_baseline.py, 491 proteins, unresolved atoms excluded.
TEMPLATE_BASELINE = 4.664


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def rolling_median(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = sorted(values[max(0, i - window + 1): i + 1])
        n = len(chunk)
        out.append(chunk[n // 2] if n % 2 else 0.5 * (chunk[n // 2 - 1] + chunk[n // 2]))
    return out


def panel(ax, label: str, metrics: Path, window: int, show_ylabel: bool,
          show_panel_title: bool, max_step: int | None = None) -> None:
    train = read_csv(metrics / f"sidechain_arm_{label}_train.csv")
    val = read_csv(metrics / f"sidechain_arm_{label}_val.csv")
    if max_step is not None:
        train = [r for r in train if int(r["step"]) <= max_step]
        val = [r for r in val if int(r["step"]) <= max_step]
        if not train:
            raise SystemExit(f"{label}: no rows at or below step {max_step}")

    ts = [int(r["step"]) for r in train]
    tm = [float(r["sc_local_median"]) for r in train]
    vs = [int(r["step"]) for r in val]
    vm = [float(r["sc_local"]) for r in val]

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.axhline(TEMPLATE_BASELINE, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    ax.text(
        max(ts) * 0.075, TEMPLATE_BASELINE + 0.12,
        f"ideal-rotamer template  {TEMPLATE_BASELINE:.2f} Å²  ({math.sqrt(TEMPLATE_BASELINE):.2f} Å)",
        color=MUTED, fontsize=8.5, ha="left", va="bottom", zorder=1,
    )

    # Per-step spread stays visible behind the trend; the trend is the 2px mark.
    ax.scatter(ts, tm, s=3.5, color=TRAIN, alpha=0.16, linewidths=0, zorder=2)
    ax.plot(ts, rolling_median(tm, window), color=TRAIN, linewidth=2.0, zorder=3,
            solid_capstyle="round")
    ax.plot(vs, vm, color=VAL, linewidth=2.0, marker="o", markersize=4.0,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4)

    best_i = min(range(len(vm)), key=lambda i: vm[i])
    ax.plot([vs[best_i]], [vm[best_i]], marker="o", markersize=7.5, color=VAL,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
    # Flip the callout to the left once the best point is near the right edge --
    # which is exactly what happens on a window that ends at the best step.
    near_right = vs[best_i] > max(ts) * 0.72
    ax.annotate(
        f"best {vm[best_i]:.3f} Å²  ({math.sqrt(vm[best_i]):.3f} Å)  ·  step {vs[best_i]:,}",
        xy=(vs[best_i], vm[best_i]),
        xytext=(vs[best_i] + max(ts) * (-0.03 if near_right else 0.03), vm[best_i] + 0.62),
        color=INK_2, fontsize=8.5, va="bottom", ha="right" if near_right else "left",
        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9, shrinkA=2, shrinkB=5),
    )

    # Direct labels: identity is never colour-alone.
    ax.text(max(ts) * 1.012, vm[-1], "validation\n(491 proteins)", color=INK_2,
            fontsize=9, va="center", ha="left")
    ax.text(max(ts) * 1.012, rolling_median(tm, window)[-1], "train\n(per-step median)",
            color=INK_2, fontsize=9, va="center", ha="left")

    ax.set_xlim(0, max(ts) * 1.01)
    top = max(6.2, TEMPLATE_BASELINE + 1.4)
    ax.set_ylim(0, top)
    # Never crop silently: an arm with no template anchor starts far above the
    # band that makes the converged region readable, and a reader cannot tell a
    # clipped point from a missing one.
    clipped = sum(1 for v in tm if v > top)
    if clipped:
        ax.text(
            max(ts) * 0.985, top * 0.975,
            f"↑ {clipped} of {len(tm)} per-step medians above axis (max {max(tm):.1f} Å²)",
            color=MUTED, fontsize=8, ha="right", va="top",
        )
    ax.set_xlabel("training step", color=INK_2, fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("side-chain loss  sc_local  (Å²)", color=INK_2, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    # Keep the "k" honest on a short window, where the locator picks 2500-step
    # ticks and an int cast would silently print 2500 as "2k".
    def _kfmt(x, _):
        if not x:
            return "0"
        k = x / 1000
        return f"{k:g}k" if k == int(k) else f"{k:.1f}k"

    ax.xaxis.set_major_formatter(_kfmt)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    # A single panel is already named by the subtitle; a per-panel title only
    # earns its place once there is more than one arm to tell apart.
    if show_panel_title:
        ax.set_title(label.replace("_", " "), color=INK, fontsize=10.5, loc="left", pad=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--metrics-dir", type=Path, default=Path("runs/metrics"))
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument("--title", default="Side-chain warm-up: training keeps improving, validation does not")
    p.add_argument("--subtitle", default="")
    p.add_argument("--max-step", type=int, default=None,
                   help="truncate every arm to this step (both train and val)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    n = len(args.arms)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n + 2.4, 5.4), squeeze=False,
                             facecolor=SURFACE, sharey=True)
    for i, label in enumerate(args.arms):
        panel(axes[0][i], label, args.metrics_dir, args.rolling_window,
              show_ylabel=(i == 0), show_panel_title=(n > 1),
              max_step=args.max_step)

    # Explicit margins, not tight_layout: the direct labels live outside the axes
    # and tight_layout cannot see them, so it crops them.
    right = 0.80 if n == 1 else 0.90
    fig.subplots_adjust(left=0.075, right=right, top=0.745, bottom=0.115, wspace=0.30)

    fig.text(0.012, 0.955, args.title, color=INK, fontsize=13.5, ha="left", va="top")
    if args.subtitle:
        fig.text(0.012, 0.885, args.subtitle, color=INK_2, fontsize=9.5,
                 ha="left", va="top")

    handles = [
        plt.Line2D([], [], color=TRAIN, linewidth=2.0, label="train (per-step median)"),
        plt.Line2D([], [], color=VAL, linewidth=2.0, marker="o", markersize=4.0,
                   label="validation (491 proteins)"),
        plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)),
                   label="ideal-rotamer template baseline"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.845),
                     frameon=False, fontsize=9, ncol=3, handlelength=2.4,
                     columnspacing=2.0, borderpad=0)
    for text in leg.get_texts():
        text.set_color(INK_2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote_figure={args.out}")


if __name__ == "__main__":
    main()
