#!/usr/bin/env python3
"""ProteinMPNN sequences for each generated backbone. Runs in the af2ig venv.

    python scripts/uncond/mpnn_designs.py --samples-dir <dir of L*.cif> \
        --out designs.csv --num-seqs 8

Designability is defined on sequences ProteinMPNN invents from the backbone
alone -- it deliberately does NOT use the sequence the generative model
produced. That is what separates it from codesignability.

Emits one row per (sample, sequence index) with a ``fold_id`` the ESMFold
stage consumes: ``<sample_id>__mpnn<k>``.

Weights come from ColabDesign's bundled ProteinMPNN (v_48_020, the original),
so nothing is downloaded.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def to_pdb(path: Path, scratch: Path) -> Path:
    """ColabDesign's parser wants PDB; the sampler writes CIF."""
    if path.suffix == ".pdb":
        return path
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    atoms = get_structure(CIFFile.read(str(path)), model=1)
    out = scratch / f"{path.stem}.pdb"
    handle = PDBFile()
    handle.set_structure(atoms)
    handle.write(str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-seqs", type=int, default=8,
                        help="PMPNN@8 is the published setting; @1 is read "
                             "off the first of these, not a separate run")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="original")
    args = parser.parse_args()

    from colabdesign.mpnn import mk_mpnn_model

    samples = sorted(Path(args.samples_dir).glob("L*.cif"))
    if not samples:
        samples = sorted(Path(args.samples_dir).glob("L*.pdb"))
    if not samples:
        raise SystemExit(f"no L<len>_s<idx> samples under {args.samples_dir}")

    scratch = Path(args.out).parent / "_pdb"
    scratch.mkdir(parents=True, exist_ok=True)
    model = mk_mpnn_model(weights=args.weights)

    rows = []
    for index, path in enumerate(samples, 1):
        pdb = to_pdb(path, scratch)
        model.prep_inputs(pdb_filename=str(pdb))
        model.set_seed(args.seed)
        out = model.sample(num=args.num_seqs, batch=args.num_seqs,
                           temperature=args.temperature)
        # `sample` returns one entry per design; score is the negative
        # log-likelihood ColabDesign reports, kept for ordering PMPNN@1.
        for k, (seq, score) in enumerate(zip(out["seq"], out["score"])):
            rows.append({
                "sample_id": path.stem,
                "mpnn_index": k,
                "fold_id": f"{path.stem}__mpnn{k}",
                "score": float(score),
                "sequence": seq if isinstance(seq, str) else str(seq),
            })
        if index % 20 == 0 or index == len(samples):
            print(f"  {index}/{len(samples)} backbone(s)")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "mpnn_index", "fold_id", "score", "sequence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} sequence(s) for {len(samples)} backbone(s) -> {args.out}")


if __name__ == "__main__":
    main()
