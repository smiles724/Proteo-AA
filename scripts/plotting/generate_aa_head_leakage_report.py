#!/usr/bin/env python3
"""Generate a concise one-page PDF for the AA-head leakage evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


COLORS = {
    "navy": "#18324A",
    "blue": "#2674A6",
    "orange": "#E17C24",
    "green": "#2A8C55",
    "red": "#B53A3A",
    "light_blue": "#EAF3F8",
    "light_red": "#FBECEC",
    "light_gray": "#F4F6F7",
    "gray": "#5C6770",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _condition_rows(rows: list[dict[str, str]], condition: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["condition"] == condition),
        key=lambda row: float(row["sigma"]),
    )


def _box(fig, xywh, facecolor, edgecolor="none", radius=0.012):
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=-1,
    )
    fig.patches.append(patch)


def generate(summary_csv: Path, metadata_json: Path, output_pdf: Path) -> None:
    rows = _read_rows(summary_csv)
    metadata = json.loads(metadata_json.read_text())
    sigmas = [float(value) for value in metadata["sigmas"]]

    full = _condition_rows(rows, "full_topology_scrub")
    strict = _condition_rows(rows, "strict_native")
    random = _condition_rows(rows, "strict_random")
    prior = 100.0 * float(metadata["empirical_majority_accuracy"])
    n_tokens = int(float(strict[0]["n_tokens"]))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.patches.append(
        Rectangle((0, 0.91), 1, 0.09, transform=fig.transFigure, color=COLORS["navy"])
    )
    fig.text(
        0.04,
        0.957,
        "AA-head inference-style validation",
        color="white",
        fontsize=20,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.96,
        0.957,
        "491 monomers  |  141,255 residues",
        color="#DDE8EF",
        fontsize=9,
        ha="right",
        va="center",
    )

    _box(fig, (0.04, 0.815, 0.92, 0.065), COLORS["light_red"], COLORS["red"])
    fig.text(
        0.06,
        0.852,
        "Finding",
        color=COLORS["red"],
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.135,
        0.852,
        "The original 98.5–100.0% accuracy is driven by native atom topology/reference metadata.\n"
        "With inference-style inputs, accuracy is 6.9% and native geometry is no better than random geometry.",
        color="#3B2525",
        fontsize=9.2,
        va="center",
    )

    fig.text(0.04, 0.775, "1. Input differences", fontsize=13, fontweight="bold", color=COLORS["navy"])
    ax_table = fig.add_axes([0.04, 0.405, 0.53, 0.35])
    ax_table.axis("off")
    input_rows = [
        ["Sequence token", "All design residues masked as xpb", "All design residues masked as xpb"],
        ["Coordinates", "Noised native structure", "Noised native N/Cα/C/O backbone*"],
        ["Atom rows", "Native full-atom rows retained", "Exactly N/Cα/C/O; side chains removed"],
        ["Reference\nmetadata", "Native AA-specific topology, names,\nelements, charges and ref positions", "Uniform canonical backbone metadata"],
        ["Sequence-derived\nfeatures", "MSA/profile masked, but native atom\nmetadata remains", "MSA/profile/template/hotspot zeroed"],
        ["Native labels", "aa_clean present for the training loss", "Stored outside model input; metrics only"],
    ]
    table = ax_table.table(
        cellText=input_rows,
        colLabels=["Feature", "Original training / validation", "Inference-style validation"],
        colWidths=[0.20, 0.40, 0.40],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D6DCE0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col == 0:
            cell.set_facecolor(COLORS["light_gray"])
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#FAFBFC")
        cell.PAD = 0.045
    fig.text(
        0.04,
        0.387,
        "*Native backbone coordinates are used only so predictions can be compared with a known sequence; "
        "free monomer inference starts from Gaussian coordinates.",
        fontsize=7.2,
        color=COLORS["gray"],
    )

    fig.text(0.62, 0.775, "2. Validation result", fontsize=13, fontweight="bold", color=COLORS["navy"])
    ax = fig.add_axes([0.64, 0.46, 0.32, 0.285])
    series = [
        (full, "Original topology + scrubbed SC coords", COLORS["blue"], "o"),
        (strict, "Inference-style + native backbone", COLORS["orange"], "s"),
        (random, "Inference-style + random geometry", COLORS["green"], "^")
    ]
    for values, label, color, marker in series:
        ax.plot(
            sigmas,
            [100.0 * float(row["acc"]) for row in values],
            marker=marker,
            linewidth=2.0,
            markersize=5,
            color=color,
            label=label,
        )
    ax.axhline(prior, color=COLORS["gray"], linestyle="--", linewidth=1.2, label=f"Majority baseline ({prior:.2f}%)")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Fixed diffusion noise σ")
    ax.set_ylabel("AA recovery (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center", fontsize=7.2, frameon=False)

    fig.text(0.04, 0.335, "Accuracy by noise level", fontsize=11, fontweight="bold", color=COLORS["navy"])
    ax_metrics = fig.add_axes([0.04, 0.19, 0.92, 0.125])
    ax_metrics.axis("off")
    metric_rows = [
        ["Original topology + scrubbed SC coords"] + [f"{100*float(row['acc']):.2f}%" for row in full],
        ["Inference-style + native backbone"] + [f"{100*float(row['acc']):.2f}%" for row in strict],
        ["Inference-style + random geometry"] + [f"{100*float(row['acc']):.2f}%" for row in random],
    ]
    metrics_table = ax_metrics.table(
        cellText=metric_rows,
        colLabels=["Condition"] + [f"σ = {sigma:g}" for sigma in sigmas],
        colWidths=[0.49, 0.17, 0.17, 0.17],
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(8.5)
    for (row, col), cell in metrics_table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EFF5F8" if row == 1 else "#FAF3EA" if row == 2 else "#EDF6F0")
            if col == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_fontweight("bold")
        cell.PAD = 0.06

    _box(fig, (0.04, 0.065, 0.92, 0.085), COLORS["light_blue"], COLORS["blue"])
    fig.text(0.06, 0.125, "Interpretation", fontsize=10.5, fontweight="bold", color=COLORS["navy"])
    fig.text(
        0.06,
        0.107,
        "Keeping native atom metadata preserves near-perfect prediction even after side-chain coordinates are scrubbed.\n"
        "Removing that metadata while retaining the native backbone collapses performance to the random-geometry control and below the 9.15% residue-prior baseline.\n"
        "The reported 90%+ result therefore does not measure backbone-to-sequence prediction.",
        fontsize=7.8,
        color="#263A49",
        va="top",
    )
    fig.text(
        0.04,
        0.025,
        f"Checkpoint: AA-head-init joint step 50,000  |  Validation: {metadata['n_validation_rows']} monomers, {n_tokens:,} residues  |  Generated 2026-08-03",
        fontsize=7,
        color=COLORS["gray"],
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_pdf.with_suffix(".png"), format="png", dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(Path(args.summary_csv), Path(args.metadata_json), Path(args.output))
