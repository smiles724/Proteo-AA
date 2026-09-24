#!/usr/bin/env python3
"""Structural diversity: foldseek cluster counts per length.

    python scripts/uncond/diversity.py \
        --samples runs/uncond/baseline runs/uncond/baseline_L400 \
        --out runs/uncond/report_diversity

WHAT THE MODES ARE. foldseek --alignment-type is, per `easy-cluster -h`:

    0: 3di alignment      structure, via the 3Di alphabet
    1: TM alignment       structure, via TMalign
    2: 3Di+AA             structure AND amino acid identity
    3: LoL alignment

NONE of these is sequence-only clustering, so this script does not claim to
measure sequence diversity. An earlier version labelled mode 2 "seq" and
mode 0 "str+seq"; both labels were wrong, and the "sequence collapse" they
appeared to show was not evidence of anything. Sequence diversity has to be
clustered from the DESIGNED SEQUENCES (the co-design FASTAs or the MPNN
table), not from backbone files.

That matters doubly here because sample_uncond.py writes backbone-only
structures whose residues are all GLY. Any AA-aware mode run on them is
comparing a constant, so mode 2 degenerates to mode 0 plus noise. This
script therefore refuses AA-aware modes on all-glycine input unless
--allow-poly-gly is passed.

THRESHOLDS ARE NOT OPTIONAL. foldseek's defaults are -c 0.0,
--tmscore-threshold 0.0, -e 10.0: no coverage requirement, no TM cutoff, and
a permissive E-value. Clustering at those defaults merges almost everything
through partial or local similarity -- on this benchmark it reported 7
clusters at L100 and 3 at L300, where TM>=0.5 with -c 0.8 gives 18 and 23.
The default-threshold numbers also produced a spurious monotone decay with
length that reverses once thresholds are applied. Defaults here are
therefore TM >= 0.5 over 0.8 coverage, and every run records the exact
parameters and the foldseek commit in its summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# name -> (--alignment-type, uses amino-acid identity)
MODES = {"3di": ("0", False), "tm": ("1", False), "3di_aa": ("2", True)}


def foldseek_version(binary: str) -> str:
    try:
        return subprocess.run([binary, "version"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def is_poly_gly(path: Path) -> bool:
    """True if every residue is glycine -- i.e. a backbone-only structure."""
    names = set()
    text = path.read_text(errors="ignore")
    if path.suffix == ".cif":
        for line in text.splitlines():
            if line.startswith("ATOM"):
                parts = line.split()
                if len(parts) > 5:
                    names.add(parts[5])
    else:
        for line in text.splitlines():
            if line.startswith("ATOM"):
                names.add(line[17:20].strip())
    return bool(names) and names <= {"GLY"}


def cluster(pdb_dir: Path, work: Path, binary: str, mode: str,
            tmscore: float, coverage: float, cov_mode: int,
            evalue: float) -> tuple[int, dict, list[str]]:
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / f"clu_{mode}"
    cmd = [binary, "easy-cluster", str(pdb_dir), str(prefix), str(work / "tmp"),
           "--alignment-type", MODES[mode][0],
           "-c", str(coverage), "--cov-mode", str(cov_mode),
           "-e", str(evalue)]
    if MODES[mode][0] == "1":
        cmd += ["--tmscore-threshold", str(tmscore)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    report = Path(f"{prefix}_cluster.tsv")
    if result.returncode != 0 or not report.is_file():
        raise SystemExit(f"foldseek {mode} failed:\n{result.stderr[-2000:]}")

    reps, members = Counter(), set()
    with report.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rep, member = line.rstrip("\n").split("\t")[:2]
            reps[rep] += 1
            members.add(member)
    sizes = sorted(reps.values(), reverse=True)
    stats = {
        "clusters": len(reps),
        "clustered_members": len(members),
        "largest_cluster": sizes[0] if sizes else 0,
        "top3_cluster_sizes": sizes[:3],
    }
    return len(reps), stats, cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--foldseek",
                        default="/hai/scratch/yfsun/tools/foldseek/bin/foldseek")
    parser.add_argument("--modes", nargs="+", default=["tm"],
                        choices=sorted(MODES))
    parser.add_argument("--tmscore-threshold", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--evalue", type=float, default=0.001)
    parser.add_argument("--subsample", type=int, default=None,
                        help="cluster this many per length; cluster count is "
                             "not linear in n, so cross-length comparison "
                             "needs equal sizes, not a count/n ratio")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-poly-gly", action="store_true",
                        help="permit AA-aware modes on all-glycine input")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # One structure per sample id. Accepting both .cif and .pdb for the same
    # id would cluster it twice and inflate every count.
    by_length: dict[int, dict[str, Path]] = defaultdict(dict)
    duplicates = []
    for directory in args.samples:
        for path in sorted(Path(directory).iterdir()):
            if path.suffix not in (".cif", ".pdb"):
                continue
            match = re.match(r"L(\d+)_s\d+$", path.stem)
            if not match:
                continue
            bucket = by_length[int(match.group(1))]
            if path.stem in bucket:
                duplicates.append(f"{bucket[path.stem]} vs {path}")
                continue
            bucket[path.stem] = path
    if duplicates:
        raise SystemExit("duplicate representations of the same sample:\n  " +
                         "\n  ".join(duplicates[:10]))
    if not by_length:
        raise SystemExit(f"no L<len>_s<n> structures under {args.samples}")

    probe = next(iter(next(iter(by_length.values())).values()))
    poly_gly = is_poly_gly(probe)
    aa_modes = [m for m in args.modes if MODES[m][1]]
    if poly_gly and aa_modes and not args.allow_poly_gly:
        raise SystemExit(
            f"{probe.name} is all-glycine (backbone only), so AA-aware "
            f"mode(s) {aa_modes} would compare a constant. Cluster the "
            f"designed sequences instead, or pass --allow-poly-gly.")

    sizes = {L: len(v) for L, v in by_length.items()}
    if args.subsample is None and len(set(sizes.values())) > 1:
        print(f"WARNING: unequal sample counts {sizes}; cluster counts are "
              f"not comparable across lengths. Use --subsample.")

    rows, per_length = [], {}
    commands = {}
    for length in sorted(by_length):
        paths = list(by_length[length].values())
        if args.subsample:
            if len(paths) < args.subsample:
                raise SystemExit(
                    f"L{length} has {len(paths)} < --subsample {args.subsample}")
            paths = rng.sample(sorted(paths), args.subsample)
        stage = out / "_stage" / f"L{length}"
        stage.mkdir(parents=True, exist_ok=True)
        for old in stage.iterdir():
            old.unlink()
        for path in paths:
            (stage / path.name).symlink_to(path.resolve())

        entry = {"n_input": len(paths)}
        for mode in args.modes:
            _, stats, cmd = cluster(
                stage, out / "_work" / f"L{length}" / mode, args.foldseek, mode,
                args.tmscore_threshold, args.coverage, args.cov_mode,
                args.evalue)
            commands[mode] = cmd
            if stats["clustered_members"] != len(paths):
                raise SystemExit(
                    f"L{length}/{mode}: foldseek clustered "
                    f"{stats['clustered_members']} of {len(paths)} structures; "
                    f"counting clusters over a partial set would understate "
                    f"diversity")
            stats["largest_cluster_frac"] = round(
                stats["largest_cluster"] / len(paths), 4)
            entry[mode] = stats
            print(f"L{length} {mode}: n={len(paths)} "
                  f"clusters={stats['clusters']} "
                  f"largest={stats['largest_cluster']} "
                  f"({stats['largest_cluster_frac']:.1%}) "
                  f"top3={stats['top3_cluster_sizes']}")
        per_length[str(length)] = entry
        rows.append({"length": length, "n": len(paths),
                     **{f"{m}_clusters": entry[m]["clusters"] for m in args.modes},
                     **{f"{m}_largest_frac": entry[m]["largest_cluster_frac"]
                        for m in args.modes}})

    summary = {
        "parameters": {
            "alignment_modes": {m: MODES[m][0] for m in args.modes},
            "tmscore_threshold": args.tmscore_threshold,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "evalue": args.evalue,
            "subsample": args.subsample,
            "foldseek_version": foldseek_version(args.foldseek),
            "commands": {m: " ".join(c) for m, c in commands.items()},
        },
        "input_all_glycine": poly_gly,
        "per_length": per_length,
    }
    with (out / "diversity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary_diversity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["parameters"], indent=2))


if __name__ == "__main__":
    main()
