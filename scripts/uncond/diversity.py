#!/usr/bin/env python3
"""Diversity only: foldseek cluster counts per length and pooled.

    python scripts/uncond/diversity.py \
        --samples runs/uncond/baseline runs/uncond/baseline_L400 \
        --out runs/uncond/report_diversity

Cluster count is the score, so it is bounded above by the number of samples
and is only comparable between groups drawn at the SAME n. The per-length
table therefore also carries n and the count as a fraction of n; read the
fraction when n differs, not the raw count.

Pooled clustering over several lengths is reported too, but it answers a
different question -- structures of different length rarely cluster together,
so the pooled count is close to the sum of the per-length counts and mostly
reflects how many lengths were sampled. The per-length rows are the ones that
say something about the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

MODES = ("str", "seq", "str+seq")
ALIGNMENT = {"str": "1", "seq": "2", "str+seq": "0"}


def cluster_count(pdb_dir: Path, work: Path, binary: str, mode: str) -> int:
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / f"clu_{mode.replace('+', '')}"
    cmd = [binary, "easy-cluster", str(pdb_dir), str(prefix), str(work / "tmp"),
           "--alignment-type", ALIGNMENT[mode]]
    result = subprocess.run(cmd, capture_output=True, text=True)
    report = Path(f"{prefix}_cluster.tsv")
    if result.returncode != 0 or not report.is_file():
        print(f"  foldseek {mode} FAILED: {result.stderr.strip()[-300:]}")
        return -1
    with report.open() as handle:
        return len({line.split("\t")[0] for line in handle if line.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True,
                        help="one or more sample directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--foldseek",
                        default="/hai/scratch/yfsun/tools/foldseek/bin/foldseek")
    parser.add_argument("--pooled", action="store_true",
                        help="also cluster all lengths together")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_length: dict[int, list[Path]] = defaultdict(list)
    for directory in args.samples:
        d = Path(directory)
        for path in sorted(list(d.glob("L*.cif")) + list(d.glob("L*.pdb"))):
            match = re.match(r"L(\d+)_s\d+$", path.stem)
            if match:
                by_length[int(match.group(1))].append(path)
    if not by_length:
        raise SystemExit(f"no L<len>_s<n> structures under {args.samples}")

    stage_root = out / "_stage"
    rows, per_length = [], {}
    for length in sorted(by_length):
        paths = by_length[length]
        stage = stage_root / f"L{length}"
        stage.mkdir(parents=True, exist_ok=True)
        for old in stage.iterdir():
            old.unlink()
        for path in paths:
            (stage / path.name).symlink_to(path.resolve())
        counts = {m: cluster_count(stage, out / "_work" / f"L{length}",
                                   args.foldseek, m) for m in MODES}
        n = len(paths)
        per_length[str(length)] = {
            "n": n,
            **{m: counts[m] for m in MODES},
            **{f"{m}_frac": (round(counts[m] / n, 4) if counts[m] >= 0 else None)
               for m in MODES},
        }
        rows.append({"length": length, "n": n,
                     **{m.replace("+", "_"): counts[m] for m in MODES}})
        print(f"L{length}: n={n} " +
              " ".join(f"{m}={counts[m]}" for m in MODES))

    summary = {"per_length": per_length}

    if args.pooled:
        stage = stage_root / "pooled"
        stage.mkdir(parents=True, exist_ok=True)
        for old in stage.iterdir():
            old.unlink()
        total = 0
        for length, paths in by_length.items():
            for path in paths:
                (stage / path.name).symlink_to(path.resolve())
                total += 1
        counts = {m: cluster_count(stage, out / "_work" / "pooled",
                                   args.foldseek, m) for m in MODES}
        summary["pooled"] = {"n": total, **counts}
        print("pooled: n=%d " % total +
              " ".join(f"{m}={counts[m]}" for m in MODES))

    with (out / "diversity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["length", "n", "str", "seq", "str_seq"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary_diversity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
