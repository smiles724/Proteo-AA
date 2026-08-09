#!/usr/bin/env python3
"""Plot Proteo-AA training metrics parsed from one or more Slurm logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP_RE = re.compile(r"\bstep=(\d+)\b")
KV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
)


def parse_metrics(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    job = path.stem.split("-")[-1]
    with path.open("r", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            match = STEP_RE.search(line)
            if not match:
                continue
            row: dict[str, float | int | str] = {
                "job": job,
                "file": str(path),
                "line": line_no,
                "step": int(match.group(1)),
            }
            for key, value in KV_RE.findall(line):
                if key == "step":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            if "loss" in row:
                rows.append(row)
    return pd.DataFrame(rows)


def build_frame(logs: list[Path], first_job_max_step: int) -> pd.DataFrame:
    frames = [parse_metrics(p) for p in logs]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise ValueError("No step=... metric rows found")

    # The first job wrote a corrupt step5600 checkpoint. Keep only the committed
    # portion through the last valid checkpoint, then use the resumed job.
    first_job = sorted(df["job"].unique())[0]
    keep = (df["job"] != first_job) | (df["step"] <= first_job_max_step)
    df = df.loc[keep].copy()

    # stdout/stderr can contain the same metric rows. Prefer the last observed
    # row per (job, step), after the committed-step filter above.
    df = df.sort_values(["job", "step", "line"]).drop_duplicates(
        subset=["job", "step"], keep="last"
    )
    return df.sort_values("step").reset_index(drop=True)


def add_rolling(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_roll"] = out[col].rolling(window, min_periods=1).median()
    return out


def plot_panel(
    ax, df: pd.DataFrame, cols: list[str], title: str, *, log_y: bool = False
):
    for col in cols:
        if col not in df.columns:
            continue
        roll = f"{col}_roll"
        ax.plot(df["step"], df[col], alpha=0.18, linewidth=0.8)
        if roll in df.columns:
            ax.plot(df["step"], df[roll], linewidth=1.8, label=col)
        else:
            ax.plot(df["step"], df[col], linewidth=1.5, label=col)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    if log_y:
        ax.set_yscale("log")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=[
            Path("logs/training/monomer_pretrain/proteo-aa-mono-90845.out"),
            Path("logs/training/monomer_pretrain/proteo-aa-mono-91031.out"),
        ],
    )
    p.add_argument("--first-job-max-step", type=int, default=5200)
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/figures/protenix_monomer_training_metrics.png"),
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("runs/figures/protenix_monomer_training_metrics.csv"),
    )
    args = p.parse_args()

    df = build_frame(args.logs, args.first_job_max_step)
    columns = [
        "loss",
        "mse",
        "bb_post",
        "lddt",
        "distogram",
        "aa_ce",
        "aa_acc",
        "sc_local",
        "sc_global",
        "sc_phys",
        "sigma_low_frac",
        "aa_mask_frac",
    ]
    df = add_rolling(df, columns, args.rolling_window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        f"Proteo-AA Protenix Monomer Pretraining ({int(df.step.min())}-{int(df.step.max())} steps)",
        fontsize=14,
    )

    plot_panel(axes[0, 0], df, ["loss"], "Total loss", log_y=True)
    plot_panel(
        axes[0, 1], df, ["mse", "bb_post"], "Backbone coordinate losses", log_y=True
    )
    plot_panel(axes[1, 0], df, ["aa_ce"], "Amino-acid cross entropy", log_y=True)
    plot_panel(axes[1, 1], df, ["aa_acc"], "Amino-acid recovery")
    axes[1, 1].set_ylim(max(0.0, df["aa_acc"].min() - 0.05), 1.02)
    plot_panel(
        axes[2, 0],
        df,
        ["sc_local", "sc_global"],
        "Side-chain coordinate losses",
        log_y=True,
    )
    plot_panel(
        axes[2, 1],
        df,
        ["lddt", "distogram", "sigma_low_frac"],
        "Auxiliary structure metrics",
    )

    for ax in axes.ravel():
        ax.axvline(
            args.first_job_max_step,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
        )
        ymin, ymax = ax.get_ylim()
        if math.isfinite(ymin) and math.isfinite(ymax):
            ax.text(
                args.first_job_max_step,
                ymax,
                "resume",
                ha="right",
                va="top",
                fontsize=8,
                alpha=0.7,
            )

    fig.savefig(args.out, dpi=200)
    print(f"rows={len(df)}")
    print(f"step_min={int(df.step.min())}")
    print(f"step_max={int(df.step.max())}")
    print(f"wrote_csv={args.csv}")
    print(f"wrote_figure={args.out}")


if __name__ == "__main__":
    main()
