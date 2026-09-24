#!/usr/bin/env python3
"""Sequence diversity of DESIGNED sequences, by mmseqs2 clustering.

    python scripts/uncond/seq_diversity.py \
        --fasta-dir runs/uncond/codesign_baseline/samples \
        --out runs/uncond/report_seq_diversity

    python scripts/uncond/seq_diversity.py \
        --mpnn-csv runs/uncond/mpnn.csv --per-backbone 1 \
        --out runs/uncond/report_seq_diversity_mpnn

This is the metric diversity.py deliberately does NOT provide. foldseek has
no sequence-only mode -- its --alignment-type 2 is 3Di+AA, structure AND
amino acid -- and the generated backbones are all-glycine, so no foldseek
mode run on them can say anything about sequence. Sequence diversity has to
come from the designed sequences and a sequence tool.

Two input forms, because the two sources differ in shape:

  --fasta-dir     one sequence per backbone (the co-design arms). Clean:
                  every sequence comes from a different structure.
  --mpnn-csv      eight sequences per backbone (the designability set).
                  --per-backbone N keeps the N best-scoring per backbone;
                  at N=1 the set is comparable to --fasta-dir, and at N=8
                  the count mixes within-backbone variation with
                  between-backbone variation and is not comparable to it.

Interpretation caveat worth carrying: these sequences were designed ON a set
of backbones that is itself only tens of structural clusters. Sequence
diversity here is therefore partly inherited from structural diversity, and
isolating the sequence head's own contribution needs a within-backbone
comparison, not this number.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def read_fasta_dir(directory: Path) -> dict[str, str]:
    out = {}
    for path in sorted(directory.glob("*.fasta")):
        lines = [l.strip() for l in path.read_text().splitlines()]
        seq = "".join(l for l in lines if l and not l.startswith(">"))
        if seq:
            out[path.stem] = seq
    return out


def read_mpnn(csv_path: Path, per_backbone: int) -> dict[str, str]:
    by_sample = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_sample[row["sample_id"]].append(row)
    out = {}
    for sample_id, rows in by_sample.items():
        rows.sort(key=lambda r: float(r["score"]))
        for rank, row in enumerate(rows[:per_backbone]):
            key = sample_id if per_backbone == 1 else f"{sample_id}__{rank}"
            out[key] = row["sequence"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta-dir")
    source.add_argument("--mpnn-csv")
    parser.add_argument("--per-backbone", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mmseqs",
                        default="/hai/scratch/yfsun/tools/mmseqs/bin/mmseqs")
    parser.add_argument("--min-seq-id", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7],
                        help="cluster at each identity threshold")
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "_work").mkdir(parents=True, exist_ok=True)

    if args.fasta_dir:
        seqs = read_fasta_dir(Path(args.fasta_dir))
        origin = args.fasta_dir
    else:
        seqs = read_mpnn(Path(args.mpnn_csv), args.per_backbone)
        origin = f"{args.mpnn_csv} (top {args.per_backbone}/backbone)"
    if not seqs:
        raise SystemExit(f"no sequences read from {origin}")

    by_length = defaultdict(dict)
    for name, seq in seqs.items():
        by_length[len(seq)][name] = seq

    combined = out / "all.fasta"
    with combined.open("w") as handle:
        for name, seq in sorted(seqs.items()):
            handle.write(f">{name}\n{seq}\n")

    def cluster(fasta: Path, work: Path, identity: float) -> dict:
        work.mkdir(parents=True, exist_ok=True)
        prefix = work / f"clu_{int(identity * 100)}"
        cmd = [args.mmseqs, "easy-cluster", str(fasta), str(prefix),
               str(work / "tmp"), "--min-seq-id", str(identity),
               "-c", str(args.coverage), "--cov-mode", str(args.cov_mode)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        report = Path(f"{prefix}_cluster.tsv")
        if result.returncode != 0 or not report.is_file():
            raise SystemExit(f"mmseqs failed:\n{result.stderr[-2000:]}")
        reps, members = Counter(), set()
        with report.open() as handle:
            for line in handle:
                if line.strip():
                    rep, member = line.rstrip("\n").split("\t")[:2]
                    reps[rep] += 1
                    members.add(member)
        sizes = sorted(reps.values(), reverse=True)
        return {"clusters": len(reps), "clustered_members": len(members),
                "largest_cluster": sizes[0] if sizes else 0,
                "top3": sizes[:3], "command": " ".join(cmd)}

    per_length, rows = {}, []
    for length in sorted(by_length):
        group = by_length[length]
        fasta = out / f"L{length}.fasta"
        with fasta.open("w") as handle:
            for name, seq in sorted(group.items()):
                handle.write(f">{name}\n{seq}\n")
        entry = {"n": len(group)}
        for identity in args.min_seq_id:
            stats = cluster(fasta, out / "_work" / f"L{length}", identity)
            if stats["clustered_members"] != len(group):
                raise SystemExit(
                    f"L{length}@{identity}: clustered "
                    f"{stats['clustered_members']} of {len(group)}")
            stats["largest_cluster_frac"] = round(
                stats["largest_cluster"] / len(group), 4)
            entry[str(identity)] = stats
            print(f"L{length} id>={identity}: n={len(group)} "
                  f"clusters={stats['clusters']} "
                  f"largest={stats['largest_cluster']} "
                  f"({stats['largest_cluster_frac']:.1%})")
        per_length[str(length)] = entry
        rows.append({"length": length, "n": len(group),
                     **{f"id{int(i * 100)}_clusters": entry[str(i)]["clusters"]
                        for i in args.min_seq_id}})

    summary = {
        "source": origin,
        "n_sequences": len(seqs),
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "thresholds": args.min_seq_id,
        "per_length": per_length,
    }
    with (out / "seq_diversity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary_seq_diversity.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
