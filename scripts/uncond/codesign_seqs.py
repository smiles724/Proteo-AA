#!/usr/bin/env python3
"""Collect co-design sequences into the fold_id,sequence table ESMFold wants.

    python scripts/uncond/codesign_seqs.py \
        --codesign-dir runs/uncond/codesign_baseline \
        --out runs/uncond/codesign_baseline/seqs.csv

Reads ``<codesign-dir>/samples/<sample_id>.fasta`` and emits one row per
sample with ``fold_id`` = ``<sample_id>__codesign`` -- the name
``score_uncond.py`` looks for under ``--refolds-dir``. The suffix keeps these
folds from colliding with the designability folds, which are keyed by the
ProteinMPNN candidate id and share the same output directory.

Only samples that have BOTH a .fasta and a .pdb are emitted: the .pdb is the
generated structure codesignability measures against, so a sequence without
one would fold into a row the scorer silently drops.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_fasta(path: Path) -> str:
    lines = [l.strip() for l in path.read_text().splitlines()]
    return "".join(l for l in lines if l and not l.startswith(">"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codesign-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suffix", default="__codesign")
    args = parser.parse_args()

    samples = Path(args.codesign_dir) / "samples"
    if not samples.is_dir():
        raise SystemExit(f"no samples/ under {args.codesign_dir}")

    rows, missing_pdb, empty = [], [], []
    for fasta in sorted(samples.glob("*.fasta")):
        sample_id = fasta.stem
        if not (samples / f"{sample_id}.pdb").is_file():
            missing_pdb.append(sample_id)
            continue
        sequence = read_fasta(fasta)
        if not sequence:
            empty.append(sample_id)
            continue
        rows.append({"fold_id": f"{sample_id}{args.suffix}",
                     "sequence": sequence})

    if not rows:
        raise SystemExit(f"no usable sequences in {samples}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_id", "sequence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} sequence(s) -> {out}")
    if missing_pdb:
        print(f"  skipped {len(missing_pdb)} without a .pdb: "
              f"{', '.join(missing_pdb[:5])}"
              f"{' ...' if len(missing_pdb) > 5 else ''}")
    if empty:
        print(f"  skipped {len(empty)} empty: {', '.join(empty[:5])}")


if __name__ == "__main__":
    main()
