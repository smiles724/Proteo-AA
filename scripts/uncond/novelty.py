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
read. That is qtmscore. foldseek normalises alntmscore by ALIGNMENT length,
qtmscore by query length and ttmscore by target length (see
structureconvertalis.cpp), so alntmscore is NOT query-normalised and must
not be substituted here. Confirm the comparison benchmark normalises the
same way before putting these numbers beside its own.

Search protocol: without --exhaustive, foldseek prefilters, so the result is
the maximum over RETURNED CANDIDATES, not over the database. That maximum is
a lower bound on the true nearest neighbour, which makes novelty an UPPER
bound. Measured on the 20 least-similar samples of one run, exhaustive
search raised max TM for 15 of 20, by up to +0.2028. Treat filtered-search
novelty as an upper bound and say so, or pass --exhaustive.
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
    parser.add_argument("--reuse-hits", action="store_true",
                        help="re-aggregate an existing hits.tsv instead of "
                             "searching again; postprocessing fixes can be "
                             "applied without spending the search")
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
    if args.reuse_hits and hits.is_file():
        print(f"reusing {hits} ({hits.stat().st_size} bytes); not searching")
    else:
        print(" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"foldseek failed:\n{result.stderr[-2000:]}")

    queries = sorted({p.stem for p in query_dir.iterdir()
                      if p.suffix in (".cif", ".pdb")})
    known = set(queries)

    # foldseek may report a query as the bare stem or with a chain/model
    # suffix appended. Map explicitly and reject anything unrecognised: a
    # silent miss here turns a real hit into "no neighbour found", which the
    # old code then scored as maximum novelty.
    best: dict[str, tuple[float, str]] = {}
    unmapped: set[str] = set()
    with hits.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            query, target = parts[0], parts[1]
            stem = Path(query).stem
            if stem in known:
                sample_id = stem
            else:
                candidates = [q for q in known if stem.startswith(q)]
                if len(candidates) != 1:
                    unmapped.add(query)
                    continue
                sample_id = candidates[0]
            try:
                tm = float(parts[3])          # qtmscore: query-normalised
            except ValueError:
                continue
            if sample_id not in best or tm > best[sample_id][0]:
                best[sample_id] = (tm, target)
    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} foldseek query id(s) did not map to a staged "
            f"sample, e.g. {sorted(unmapped)[:5]}. Refusing to score: an "
            f"unmapped hit is indistinguishable from no neighbour, and would "
            f"be counted as maximally novel.")

    # An unresolved query is NOT a novel one. No returned hit can mean a
    # missed candidate, an unparsed structure or an id mismatch; it does not
    # establish zero similarity. Carry it as missing and exclude it from the
    # aggregate rather than scoring it 0.0 / novelty 1.0.
    rows = []
    for sample_id in queries:
        hit = best.get(sample_id)
        tm = hit[0] if hit else None
        rows.append({"sample_id": sample_id,
                     "length": int(sample_id.split("_")[0].lstrip("L")),
                     "max_tm": None if tm is None else round(tm, 4),
                     "novelty": None if tm is None else round(1.0 - tm, 4),
                     "nearest": hit[1] if hit else "",
                     "resolved": hit is not None,
                     "_tm": tm})              # full precision, for stats

    with (out / "per_sample_novelty.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "length", "max_tm", "novelty",
                        "nearest", "resolved"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def block(subset):
        # Full-precision values: classifying on the rounded column would put
        # a true 0.49996 on the wrong side of a 0.5 threshold.
        tms = [r["_tm"] for r in subset if r["_tm"] is not None]
        unresolved = sum(1 for r in subset if r["_tm"] is None)
        if not tms:
            return {"n": 0, "unresolved": unresolved}
        return {
            "n": len(tms),
            "unresolved": unresolved,
            "mean_max_tm": round(statistics.mean(tms), 4),
            "median_max_tm": round(statistics.median(tms), 4),
            "min_max_tm": round(min(tms), 4),
            "mean_novelty": round(1.0 - statistics.mean(tms), 4),
            "pct_novel": round(
                100.0 * sum(t < args.novel_threshold for t in tms) / len(tms), 2),
        }

    by_length = {}
    for length in sorted({r["length"] for r in rows}):
        by_length[str(length)] = block([r for r in rows if r["length"] == length])

    unresolved = [r["sample_id"] for r in rows if r["_tm"] is None]
    summary = {
        "n_queries": len(rows),
        "filtered_to_designable": keep is not None,
        "novel_threshold_max_tm": args.novel_threshold,
        "alignment": "tmalign",
        "tm_column": "qtmscore (query-normalised)",
        "exhaustive_search": bool(args.exhaustive),
        "search_is_database_wide_maximum": bool(args.exhaustive),
        "database": args.db,
        "foldseek_version": subprocess.run(
            [args.foldseek, "version"], capture_output=True,
            text=True).stdout.strip(),
        "command": " ".join(cmd),
        "unresolved_queries": unresolved,
        "pooled": block(rows),
        "by_length": by_length,
    }
    if unresolved:
        print(f"WARNING: {len(unresolved)} quer(ies) returned no hit and are "
              f"EXCLUDED from the aggregate, not scored as novel: "
              f"{unresolved[:5]}. Resolve these before reporting.")
    if not args.exhaustive:
        print("NOTE: filtered search -- max TM is over returned candidates, "
              "so it is a lower bound and novelty is an UPPER bound.")
    (out / "summary_novelty.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
