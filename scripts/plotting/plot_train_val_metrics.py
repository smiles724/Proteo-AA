#!/usr/bin/env python3
"""Plot Proteo-AA training and validation metrics from Slurm logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP_RE = re.compile(r"\bstep=(\d+)\b")
KV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)=" r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
)


PANEL_COLUMNS = {
    "backbone_only": [
        ("loss", "Total loss", True),
        ("mse", "Backbone MSE", True),
        ("bb_post", "Backbone post-refine", True),
        ("lddt", "LDDT loss", False),
        ("distogram", "Distogram loss", False),
        ("sigma_low_frac", "Low-sigma fraction", False),
    ],
    "sidechain_warmup": [
        ("loss", "Total side-chain loss", True),
        ("sc_local", "Local coordinate loss", True),
        ("sc_phys", "Physical violation loss", True),
        ("sc_global", "Global auxiliary loss", True),
    ],
}


def parse_metrics(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    job = path.stem.split("-")[-1]
    with path.open("r", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            pairs = {
                key: float(value) for key, value in KV_RE.findall(line) if key != "step"
            }
            if not pairs:
                continue
            split = "val" if any(k.startswith("val_") for k in pairs) else "train"
            row: dict[str, float | int | str] = {
                "job": job,
                "file": str(path),
                "line": line_no,
                "split": split,
                "step": int(step_match.group(1)),
            }
            for key, value in pairs.items():
                if split == "val":
                    if key.startswith("val_"):
                        row[key.removeprefix("val_")] = value
                else:
                    row[key] = value
            if "loss" in row:
                rows.append(row)
    return pd.DataFrame(rows)


def build_frame(logs: list[Path]) -> pd.DataFrame:
    frames = [parse_metrics(path) for path in logs if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("No train/validation metric rows found in the provided logs")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["job", "split", "step", "line"]).drop_duplicates(
        subset=["job", "split", "step"],
        keep="last",
    )
    return df.sort_values(["step", "split"]).reset_index(drop=True)


def add_rolling(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    train_mask = out["split"].eq("train")
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_roll"] = math.nan
        out.loc[train_mask, f"{col}_roll"] = (
            out.loc[train_mask, col].rolling(window, min_periods=1).median()
        )
    return out


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str, *, log_y: bool) -> None:
    train = df[df["split"].eq("train") & df[metric].notna()]
    val = df[df["split"].eq("val") & df[metric].notna()]
    if not train.empty:
        ax.plot(
            train["step"], train[metric], color="#7fb3d5", alpha=0.22, linewidth=0.8
        )
        roll_col = f"{metric}_roll"
        y = train[roll_col] if roll_col in train.columns else train[metric]
        ax.plot(train["step"], y, color="#1f77b4", linewidth=1.8, label="train")
    if not val.empty:
        ax.plot(
            val["step"],
            val[metric],
            color="#d62728",
            marker="o",
            markersize=3.0,
            linewidth=1.4,
            label="validation",
        )
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(True, alpha=0.25)
    if log_y:
        values = pd.concat([train[metric], val[metric]], ignore_index=True)
        if not values.empty and (values > 0).any():
            ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)


def infer_stage(df: pd.DataFrame) -> str:
    sc_cols = [c for c in ("sc_local", "sc_phys", "sc_global") if c in df.columns]
    if sc_cols:
        values = df[sc_cols].fillna(0.0).abs().to_numpy()
        if values.size and values.max() > 0:
            return "sidechain_warmup"
    return "backbone_only"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", nargs="+", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=["auto", "backbone_only", "sidechain_warmup"],
        default="auto",
    )
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()

    df = build_frame(args.logs)
    stage = infer_stage(df) if args.stage == "auto" else args.stage
    panels = []
    for metric, title, log_y in PANEL_COLUMNS[stage]:
        if metric not in df.columns:
            continue
        values = df[metric].dropna()
        if values.empty:
            continue
        if values.abs().max() == 0:
            continue
        panels.append((metric, title, log_y))
    if not panels:
        raise ValueError(f"No plottable metrics found for stage={stage}")
    df = add_rolling(df, [metric for metric, _, _ in panels], args.rolling_window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    n_panels = len(panels)
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 4.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    step_min = int(df["step"].min())
    step_max = int(df["step"].max())
    title_stage = stage.replace("_", " ")
    fig.suptitle(
        f"Proteo-AA {title_stage}: train and recent-PDB validation ({step_min}-{step_max} steps)",
        fontsize=14,
    )
    for ax, (metric, title, log_y) in zip(axes.ravel(), panels):
        plot_metric(ax, df, metric, title, log_y=log_y)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    fig.savefig(args.out, dpi=200)
    n_train = int(df["split"].eq("train").sum())
    n_val = int(df["split"].eq("val").sum())
    print(f"stage={stage}")
    print(f"train_rows={n_train}")
    print(f"validation_rows={n_val}")
    print(f"step_min={step_min}")
    print(f"step_max={step_max}")
    print(f"wrote_csv={args.csv}")
    print(f"wrote_figure={args.out}")


if __name__ == "__main__":
    main()
