#!/usr/bin/env python3
"""Plot and summarize a side-chain-only warm-up Slurm log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP_RE = re.compile(r"\bstep=(\d+)\b")
KV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
)


def parse_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict[str, float]] = []
    val_rows: list[dict[str, float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if not match:
                continue
            # Per-protein validation rows (`val_protein=<id>`) are a breakdown OF
            # the aggregate line at the same step, not extra samples. Folding them
            # in would count each eval ~492 times and make `val_repeats`
            # meaningless; the aggregate line stays the single source of truth for
            # the curve. Use `--per-protein-csv` to export the breakdown instead.
            if "val_protein=" in line:
                continue
            values = {key: float(value) for key, value in KV_RE.findall(line)}
            step = int(match.group(1))
            if "val_sc_local" in values:
                val_rows.append(
                    {
                        "step": step,
                        "val_sc_loss": values["val_sc_local"],
                        "val_total_loss": values.get("val_loss", np.nan),
                        "val_sc_phys": values.get("val_sc_phys", np.nan),
                    }
                )
            elif "sc_local" in values:
                train_rows.append(
                    {
                        "step": step,
                        "train_sc_loss": values["sc_local"],
                        "train_total_loss": values.get("loss", np.nan),
                        "train_sc_phys": values.get("sc_phys", np.nan),
                    }
                )
    train = pd.DataFrame(train_rows).sort_values("step")
    val_raw = pd.DataFrame(val_rows).sort_values("step")
    val = (
        val_raw.groupby("step", as_index=False)
        .agg(
            val_sc_loss=("val_sc_loss", "mean"),
            val_sc_loss_std=("val_sc_loss", "std"),
            val_total_loss=("val_total_loss", "mean"),
            val_sc_phys=("val_sc_phys", "mean"),
            val_repeats=("val_sc_loss", "size"),
        )
        .sort_values("step")
    )
    if train.empty or val.empty:
        raise ValueError(f"Missing side-chain train/validation rows in {path}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


PER_PROTEIN_RE = re.compile(r"\bval_protein=(\S+)")


def parse_per_protein(path: Path) -> pd.DataFrame:
    """Per-protein validation rows over the whole training pathway.

    One row per (step, protein): the breakdown behind each aggregate validation
    point, so a checkpoint that improves on average can be checked for proteins
    that got worse. Empty for logs written before per-protein logging existed.
    """
    rows: list[dict[str, float | str | int]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            step_match = STEP_RE.search(line)
            name_match = PER_PROTEIN_RE.search(line)
            if not (step_match and name_match):
                continue
            values = {key: float(value) for key, value in KV_RE.findall(line)}
            row: dict[str, float | str | int] = {
                "step": int(step_match.group(1)),
                "sample_id": name_match.group(1),
                "index": int(values.get("val_index", -1)),
            }
            for key, value in values.items():
                if key.startswith("val_") and key not in ("val_index", "val_protein"):
                    row[key] = value
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["step", "sample_id", "index"])
    return pd.DataFrame(rows).sort_values(["step", "index"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--ranking-csv", required=True, type=Path)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--template-baseline", type=float, default=None)
    parser.add_argument(
        "--per-protein-csv",
        type=Path,
        default=None,
        help="Write the per-(step, protein) validation breakdown here, if the log has one.",
    )
    args = parser.parse_args()

    train, val = parse_log(args.log)
    if args.per_protein_csv is not None:
        per_protein = parse_per_protein(args.log)
        args.per_protein_csv.parent.mkdir(parents=True, exist_ok=True)
        per_protein.to_csv(args.per_protein_csv, index=False)
        print(f"per_protein_rows={len(per_protein)}")
        if not per_protein.empty:
            print(f"per_protein_steps={per_protein['step'].nunique()}")
            print(f"per_protein_distinct_proteins={per_protein['sample_id'].nunique()}")
        print(f"wrote_per_protein={args.per_protein_csv}")
    train["rolling_median"] = train["train_sc_loss"].rolling(
        args.rolling_window, min_periods=1
    ).median()
    train["step_bin"] = ((train["step"] - 1) // 1000 + 1) * 1000
    binned = (
        train.groupby("step_bin")["train_sc_loss"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
            maximum="max",
        )
        .reset_index()
    )

    ranking = val.sort_values(["val_sc_loss", "step"]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    best = ranking.iloc[0]
    first_val = float(val.iloc[0]["val_sc_loss"])
    final_val = float(val.iloc[-1]["val_sc_loss"])
    improvement = 100.0 * (first_val - float(best["val_sc_loss"])) / first_val

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            train.assign(split="train").rename(columns={"train_sc_loss": "sc_loss"}),
            val.assign(split="validation").rename(columns={"val_sc_loss": "sc_loss"}),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(args.metrics_csv, index=False)
    ranking.to_csv(args.ranking_csv, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.suptitle(
        "Proteo-AA monomer side-chain-only warm-up (GT backbone and AA type)",
        fontsize=15,
    )

    ax = axes[0]
    ax.plot(train["step"], train["train_sc_loss"], color="#7fb3d5", alpha=0.25, lw=0.8)
    ax.plot(train["step"], train["rolling_median"], color="#1769aa", lw=2.0, label="train median (20 logs)")
    ax.plot(val["step"], val["val_sc_loss"], color="#c62828", marker="o", ms=3.5, lw=1.5, label="validation")
    if args.template_baseline is not None:
        ax.axhline(
            args.template_baseline,
            color="#6a1b9a",
            ls="--",
            lw=1.7,
            label="Dunbrack template baseline",
        )
    ax.set_yscale("log")
    ax.set_title("Side-chain coordinate loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(val["step"], val["val_sc_loss"], color="#c62828", marker="o", ms=4, lw=1.7)
    if args.template_baseline is not None:
        ax.axhline(
            args.template_baseline,
            color="#6a1b9a",
            ls="--",
            lw=1.7,
            label=f"template: {args.template_baseline:.3g}",
        )
        ax.set_yscale("log")
    ax.scatter([best["step"]], [best["val_sc_loss"]], color="#14866d", s=55, zorder=3, label=f"best: step {int(best['step'])}")
    ax.axhline(float(best["val_sc_loss"]), color="#14866d", ls="--", lw=1, alpha=0.6)
    ax.set_title(f"Validation loss ({improvement:.1f}% best improvement)")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("validation side-chain loss")
    if args.template_baseline is None:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    x = binned["step_bin"].to_numpy(dtype=float)
    q25 = binned["q25"].to_numpy(dtype=float)
    q75 = binned["q75"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color="#80cbc4", alpha=0.35, label="IQR")
    ax.plot(x, binned["median"], color="#00796b", marker="o", ms=3.5, lw=1.8, label="median")
    ax.plot(x, binned["p90"], color="#ef6c00", lw=1.3, label="90th percentile")
    ax.set_yscale("log")
    ax.set_title("Training stability by 1k-step bin")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(args.out, dpi=200)

    early = train.loc[train["step"] <= 2000, "train_sc_loss"]
    late = train.loc[train["step"] > train["step"].max() - 2000, "train_sc_loss"]
    print(f"completed_step={int(train['step'].max())}")
    print(f"train_rows={len(train)}")
    print(f"validation_rows={len(val)}")
    print(f"first_validation_loss={first_val:.6g}")
    print(f"best_validation_step={int(best['step'])}")
    print(f"best_validation_loss={float(best['val_sc_loss']):.6g}")
    print(f"final_validation_loss={final_val:.6g}")
    print(f"best_validation_improvement_pct={improvement:.3f}")
    print(f"early_train_median={early.median():.6g}")
    print(f"late_train_median={late.median():.6g}")
    print(f"train_max={train['train_sc_loss'].max():.6g}")
    print(f"train_gt_100_count={int((train['train_sc_loss'] > 100).sum())}")
    if args.template_baseline is not None:
        print(f"template_baseline={args.template_baseline:.6g}")
    print(f"wrote_figure={args.out}")
    print(f"wrote_metrics={args.metrics_csv}")
    print(f"wrote_ranking={args.ranking_csv}")


if __name__ == "__main__":
    main()
