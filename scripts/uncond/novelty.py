#!/usr/bin/env python3
"""Novelty: TM-score of each generated backbone to its closest PDB neighbour.

    python scripts/uncond/novelty.py \
        --samples-dir runs/uncond/baseline \
        --db /hai/scratch/yfsun/foldseek_db/pdb \
        --out runs/uncond/report_designability/novelty

Novelty is 1 - max TM-score over the reference set, so a sample that closely
reproduces a known fold scores near 0 and one with no close neighbour scores
near 1. The literature almost always reports the raw max TM-score instead,
and the conventional cut is < 0.5 -- below that the fold is usually called
novel -- so both columns are emitted and the summary carries both.

Two choices here are load-bearing and easy to get wrong:

  TMalign, not 3Di.  `--alignment-type 1` makes foldseek report a real
  TM-score. The default 3Di+AA mode is far faster but its alntmscore is an
  approximation, and novelty numbers from the two modes are not comparable
  to published ones.

  Designable samples only, when --per-sample is given.  Novelty over all
  samples rewards a model for emitting garbage: an unfoldable backbone has
  no close PDB neighbour and so scores as maximally novel. The convention is
  to measure novelty only on samples that already passed designability, so
  the two metrics cannot be traded against each other. Pass the scorer's
  per_sample.csv and the filter is applied; omit it and every sample counts,
  which is reported as `filtered: false`.

Normalisation: TM-score is asymmetric, and we want "how much of MY structure
is explained by a known one", so the query-normalised value is the one to
read. foldseek's alntmscore is query-normalised.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import statistics
from pathlib import Path

THRESHOLD_DESIGNABLE = 2.0
THRESHOLD_NOVEL = 0.5


def designable_ids(per_sample: Path, threshold: float) -> set[str]:
    keep = set()
    with per_sample.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("pmpnn8_scrmsd") or ""
            if value and float(value) < threshold:
                keep.add(row["sample_id"])
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--db", required=True, help="foldseek database prefix")
    parser.add_argument("--out", required=True)
    parser.add_argument("--foldseek",
                        default="/hai/scratch/yfsun/tools/foldseek/bin/foldseek")
    parser.add_argument("--per-sample", default=None,
                        help="score_uncond per_sample.csv; restricts to designable")
    parser.add_argument("--designable-threshold", type=float,
                        default=THRESHOLD_DESIGNABLE)
    parser.add_argument("--novel-threshold", type=float, default=THRESHOLD_NOVEL)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--exhaustive", action="store_true",
                        help="disable prefilter; slower, no missed neighbours")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)

    keep = None
    if args.per_sample:
        keep = designable_ids(Path(args.per_sample), args.designable_threshold)
        print(f"{len(keep)} designable sample(s) from {args.per_sample}")

    # foldseek searches a directory, so when filtering we stage links rather
    # than copies -- the sample set is a few hundred files but they are CIFs.
    query_dir = samples_dir
    if keep is not None:
        query_dir = out / "query"
        query_dir.mkdir(exist_ok=True)
        for old in query_dir.iterdir():
            old.unlink()
        staged = 0
        for path in sorted(samples_dir.glob("L*.cif")) + \
                    sorted(samples_dir.glob("L*.pdb")):
            if path.stem in keep:
                (query_dir / path.name).symlink_to(path.resolve())
                staged += 1
        if not staged:
            raise SystemExit("no designable samples staged; check --per-sample")
        print(f"staged {staged} query structure(s)")

    hits = out / "hits.tsv"
    cmd = [args.foldseek, "easy-search", str(query_dir), args.db, str(hits),
           str(out / "tmp"),
           "--alignment-type", "1",            # TMalign, not 3Di
           "--format-output", "query,target,alntmscore,qtmscore,ttmscore,lddt",
           "--threads", str(args.threads)]
    if args.exhaustive:
        cmd += ["--exhaustive-search", "1"]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"foldseek failed:\n{result.stderr[-2000:]}")

    best: dict[str, tuple[float, str]] = {}
    with hits.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            query, target = parts[0], parts[1]
            sample_id = Path(query).stem
            try:
                tm = float(parts[3])          # qtmscore: query-normalised
            except ValueError:
                continue
            if sample_id not in best or tm > best[sample_id][0]:
                best[sample_id] = (tm, target)

    queries = sorted({p.stem for p in query_dir.iterdir()
                      if p.suffix in (".cif", ".pdb")})
    rows = []
    for sample_id in queries:
        tm, target = best.get(sample_id, (0.0, ""))
        rows.append({"sample_id": sample_id,
                     "length": int(sample_id.split("_")[0].lstrip("L")),
                     "max_tm": round(tm, 4),
                     "novelty": round(1.0 - tm, 4),
                     "nearest": target,
                     "had_hit": bool(target)})

    with (out / "per_sample_novelty.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "length", "max_tm", "novelty",
                        "nearest", "had_hit"])
        writer.writeheader()
        writer.writerows(rows)

    def block(subset):
        tms = [r["max_tm"] for r in subset]
        if not tms:
            return None
        return {
            "n": len(tms),
            "mean_max_tm": round(statistics.mean(tms), 4),
            "median_max_tm": round(statistics.median(tms), 4),
            "mean_novelty": round(1.0 - statistics.mean(tms), 4),
            "pct_novel": round(
                100.0 * sum(t < args.novel_threshold for t in tms) / len(tms), 2),
            "no_hit": sum(1 for r in subset if not r["had_hit"]),
        }

    by_length = {}
    for length in sorted({r["length"] for r in rows}):
        by_length[str(length)] = block([r for r in rows if r["length"] == length])

    summary = {
        "n_queries": len(rows),
        "filtered_to_designable": keep is not None,
        "novel_threshold_max_tm": args.novel_threshold,
        "alignment": "tmalign",
        "pooled": block(rows),
        "by_length": by_length,
    }
    (out / "summary_novelty.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
