#!/usr/bin/env python3
"""Stage III: how much the side-chain feedback improves the backbone.

The figure has a MAIN panel and a SIDE panel, and the split is the argument:

  left  — the result. How much does the S->B feedback improve the backbone?
          Our method against its own control (the same run with the two feedback
          channels off), which is what licenses calling the improvement causal.
  right — the ablation, subordinate. Does the improvement survive when S_phi is
          handed the GT backbone instead of the predicted one? Only here does
          leakage have a path, so agreement between the two is the check.

ONE QUANTITY, NORMALISED, SHARED Y — on purpose. The instinct is to plot
`val_bb_post` in A^2 with `val_mse` beside it as the baseline. That would mislead:
`val_mse` is the same first-pass quantity in every arm, but it swings +-10% between
evals purely from which 308 proteins the eval draw lands on, and the arms draw
independently — so an absolute overlay puts ~2 A^2 of eval noise on the same axis
as a ~3 A^2 effect. Each arm is therefore scored against ITS OWN un-refined pass:

    gain(step) = (val_mse - val_bb_post) / val_mse

Both terms come from one eval on one draw, so the ratio is immune to that noise.
The baseline is then not a curve but the zero line: "the refinement pass changed
nothing". Absolute A^2 is kept as endpoint labels so the scale is not lost, and
both panels share the y scale so the side panel is readable against the main one.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Documented categorical palette, light mode, unmodified: slots 1 and 2. The
# reference palette records this pair as clearing every adjacent-pair gate in
# both modes (worst adjacent CVD dE 9.1 light), and slots 1-3 clear all-pairs.
OURS_HUE = "#2a78d6"   # slot 1, blue   — our method (frames from x_denoised)
GT_HUE = "#eb6834"     # slot 2, orange — the GT-backbone ablation
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

LOG_DIR = Path("logs/training/stage3_coevolution")

OURS = "103961"          # predicted frames, feedback ON  — the method
CONTROL = "103966"       # predicted frames, feedback OFF — its control
GT_ON = "104217"         # GT frames, feedback ON         — the ablation
GT_OFF = "104218"        # GT frames, feedback OFF        — its control
JOBS = (OURS, CONTROL, GT_ON, GT_OFF)


def read_val(log_dir: Path, job: str):
    """Return [(step, val_mse, val_bb_post, gain_pct)] from a Stage III log."""
    path = log_dir / f"proteo-aa-stage3-coev-{job}.err"
    if not path.exists():
        return []
    rows = []
    for line in path.open(errors="ignore"):
        if "val_n=" not in line:
            continue
        got = {}
        for key in ("step", "val_mse", "val_bb_post"):
            m = re.search(rf"\b{key}=([-\d.eE+]+)", line)
            if m:
                got[key] = float(m.group(1))
        if len(got) == 3 and got["val_mse"]:
            mse, post = got["val_mse"], got["val_bb_post"]
            rows.append((int(got["step"]), mse, post, (mse - post) / mse * 100.0))
    return rows


def style_axes(ax, xmax, xpad, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax * xpad)
    ax.set_xlabel("training step", color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(
        lambda x, _: "0" if not x else (f"{x/1000:g}k" if (x / 1000) == int(x / 1000)
                                        else f"{x/1000:.1f}k")
    )
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:+.0f}%" if y else "0")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)


def baseline(ax, xmax, label_x=0.99):
    """The zero line IS the no-refinement baseline."""
    ax.axhline(0, color=AXIS, linewidth=1.6, zorder=2)
    ax.annotate(
        "no refinement", xy=(xmax * label_x, 0), xytext=(0, 6),
        textcoords="offset points", color=MUTED, fontsize=8.5,
        va="bottom", ha="right",
    )


def curve(ax, rows, hue, label, *, on, zorder=5):
    ax.plot(
        [r[0] for r in rows], [r[3] for r in rows],
        color=hue, linewidth=2.2 if on else 1.6,
        linestyle="-" if on else (0, (4, 3)),
        marker="o" if on else None, markersize=5.4,
        markeredgecolor=SURFACE, markeredgewidth=1.4,
        alpha=1.0 if on else 0.8,
        label=label, zorder=zorder,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = {job: read_val(args.log_dir, job) for job in JOBS}
    missing = [j for j, rows in d.items() if not rows]
    if missing:
        raise SystemExit(f"no validation rows for job(s) {', '.join(missing)}")

    xmax = max(r[0] for rows in d.values() for r in rows)
    top = max(r[3] for rows in d.values() for r in rows) * 1.20

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.8, 6.5), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.17},
    )

    # ---------------------------------------------------------------- main panel
    style_axes(axL, xmax, 1.20,
               "backbone MSE reduction from the refinement pass  (%)")
    axL.set_ylim(top=top)
    baseline(axL, xmax, 1.19)

    curve(axL, d[OURS], OURS_HUE, "with side-chain feedback  (our method)", on=True)
    curve(axL, d[CONTROL], MUTED, "feedback channels off  (control)", on=False,
          zorder=3)

    step, mse, post, gain = d[OURS][-1]
    axL.annotate(
        f"{gain:+.1f}%\n{mse:.1f} → {post:.1f} Å$^2$",
        xy=(step, gain), xytext=(14, -6), textcoords="offset points",
        color=OURS_HUE, fontsize=11.5, fontweight="bold",
        va="center", ha="left", linespacing=1.3,
    )
    c_last = d[CONTROL][-1]
    axL.annotate(
        f"{c_last[3]:+.2f}%",
        xy=(c_last[0], c_last[3]), xytext=(14, 0), textcoords="offset points",
        color=INK_2, fontsize=10, va="center", ha="left",
    )

    axL.set_title(
        "Side-chain feedback cuts backbone error by 16%",
        color=INK, fontsize=14.5, fontweight="bold", loc="left", pad=32,
    )
    axL.text(
        0.0, 1.012,
        "Stage III, 308-protein monomer validation set. Identical runs; the "
        "control differs only in the two S→B channels.",
        transform=axL.transAxes, color=INK_2, fontsize=9.5, va="bottom",
    )
    axL.legend(loc="upper left", frameon=False, fontsize=10,
               labelcolor=INK_2, handlelength=2.4, borderaxespad=0.9)

    # ---------------------------------------------------------------- side panel
    style_axes(axR, xmax, 1.22)
    axR.set_ylim(top=top)
    axR.tick_params(labelleft=False)
    baseline(axR, xmax, 1.21)

    curve(axR, d[OURS], OURS_HUE, "predicted backbone  (our method)", on=True)
    curve(axR, d[GT_ON], GT_HUE, "ground-truth backbone", on=True)
    flat = [r[3] for j in (CONTROL, GT_OFF) for r in d[j]]
    axR.plot(
        [r[0] for r in d[GT_OFF]], [r[3] for r in d[GT_OFF]],
        color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)), alpha=0.8, zorder=3,
    )

    g_step, _, _, g_gain = d[GT_ON][-1]
    axR.annotate(
        f"{g_gain:+.1f}%", xy=(g_step, g_gain), xytext=(12, 9),
        textcoords="offset points", color=GT_HUE, fontsize=10.5,
        fontweight="bold", va="center", ha="left",
    )
    axR.annotate(
        f"{gain:+.1f}%", xy=(step, gain), xytext=(12, -11),
        textcoords="offset points", color=OURS_HUE, fontsize=10.5,
        fontweight="bold", va="center", ha="left",
    )
    axR.set_title(
        "Ablation: is it ground-truth leakage?",
        color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=32,
    )
    # The explanation lives INSIDE the panel, in the empty wedge under the
    # curves — above the axes it collides with the title.
    axR.text(
        0.985, 0.20,
        f"Same to {abs(g_gain - gain):.2f} pt.\n\n"
        "Feeding S_φ the true\n"
        "backbone opens a path\n"
        "from the GT coordinates\n"
        "into the loss. It buys\n"
        "nothing — so the gain is\n"
        "side-chain content, not\n"
        "GT geometry.",
        transform=axR.transAxes, color=INK_2, fontsize=9, va="bottom",
        ha="right", linespacing=1.5,
    )
    axR.legend(loc="upper left", frameon=False, fontsize=9,
               labelcolor=INK_2, handlelength=2.2, borderaxespad=0.9)

    last = {j: d[j][-1][0] for j in JOBS}
    fig.text(
        0.006, 0.042,
        "Gain = (val_mse − val_bb_post) / val_mse. The first backbone pass has not "
        "seen the side chain; the refinement pass has.",
        color=MUTED, fontsize=8,
    )
    fig.text(
        0.006, 0.023,
        "Feedback = a_direct_pre + q_direct. Control = ablation arm a-bs, identical "
        "in every other respect; dashed grey is feedback off for both frame "
        f"sources (median {statistics.median(flat):+.2f}%).",
        color=MUTED, fontsize=8,
    )
    fig.text(
        0.006, 0.004,
        "Warm start Stage II step52500 + AA head step9000, lr 1e-5, crop 384, "
        f"refinement_sigma 2.0. Last eval {last[OURS]}/{last[CONTROL]} predicted, "
        f"{last[GT_ON]}/{last[GT_OFF]} GT, of 8000; predicted-frame runs stopped at "
        "the 24 h wall clock, GT-frame runs still going.",
        color=MUTED, fontsize=8,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.062, right=0.995, top=0.845, bottom=0.155)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for job in JOBS:
        rows = d[job]
        print(f"  {job}  steps {rows[0][0]}-{rows[-1][0]}  "
              f"final gain {rows[-1][3]:+.2f}%")


if __name__ == "__main__":
    main()
