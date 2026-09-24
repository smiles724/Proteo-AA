#!/usr/bin/env python3
"""Designability, codesignability and diversity for the unconditional benchmark.

    python scripts/uncond/score_uncond.py \
        --samples-dir runs/uncond/baseline \
        --codesign-dir runs/uncond/codesign_baseline \
        --refolds-dir runs/uncond/refolds \
        --mpnn-designs runs/uncond/mpnn.csv \
        --out runs/uncond/report_baseline

Definitions follow the benchmark spec, and the distinction between the two
likelihood metrics is the whole point:

  Designability    ProteinMPNN invents sequences from the GENERATED BACKBONE,
                   ESMFold folds them, and the sample counts if the SMALLEST
                   scRMSD over those candidates is < 2.0 A. It does not use
                   the model's own sequence, so it says nothing about the
                   sequence modality. PMPNN@1 is the best-scoring candidate,
                   PMPNN@8 the minimum over all eight.

  Codesignability  ESMFold folds the sequence the MODEL produced, and the
                   sample counts if scRMSD between the generated structure
                   and that fold is < 2.0 A. This one does depend on the
                   sequence and on its agreement with the structure.

  Diversity        foldseek clusters; the count of clusters is the score.
                   Str, Seq and Str+Seq are different --alignment-type /
                   input settings, not different thresholds.

scRMSD is reported CA-only and all-atom. All-atom is the stricter of the two
and is only defined where the residue identities agree, which for
codesignability they do by construction and for designability they do not --
so designability is CA-only, as published.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

THRESHOLD = 2.0
BACKBONE = ("N", "CA", "C", "O")


def load(path):
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    if str(path).endswith(".cif"):
        atoms = get_structure(CIFFile.read(str(path)), model=1)
    else:
        atoms = PDBFile.read(str(path)).get_structure(model=1)
    return atoms[atoms.element != "H"]


def kabsch_rmsd(p, q):
    """RMSD after optimal superposition. Both [N,3], already corresponded."""
    if len(p) != len(q) or len(p) < 3:
        return float("nan")
    p = p - p.mean(0)
    q = q - q.mean(0)
    v, _s, w = np.linalg.svd(p.T @ q)
    if (np.linalg.det(v) * np.linalg.det(w)) < 0:
        v[:, -1] = -v[:, -1]
    return float(np.sqrt((((p @ (v @ w)) - q) ** 2).sum() / len(p)))


def ca_coords(atoms):
    sel = atoms[atoms.atom_name == "CA"]
    return sel.coord.astype(np.float64)


def matched_all_atom(a, b):
    """Coordinates of atoms present in BOTH, keyed by (residue order, name).

    Residue ORDER, not res_id: ESMFold renumbers from 1 and the generated
    structures may not, so joining on res_id silently produces an empty or
    shifted correspondence that still returns a number.
    """
    def table(atoms):
        order, seen = {}, {}
        for rid in atoms.res_id:
            if rid not in seen:
                seen[rid] = len(seen)
        for i in range(len(atoms)):
            order[(seen[atoms.res_id[i]], str(atoms.atom_name[i]))] = atoms.coord[i]
        return order

    ta, tb = table(a), table(b)
    keys = sorted(set(ta) & set(tb))
    if not keys:
        return None, None
    return (np.array([ta[k] for k in keys], dtype=np.float64),
            np.array([tb[k] for k in keys], dtype=np.float64))


def scrmsd(generated, refold, *, all_atom=False):
    if all_atom:
        p, q = matched_all_atom(generated, refold)
        if p is None:
            return float("nan")
        return kabsch_rmsd(p, q)
    p, q = ca_coords(generated), ca_coords(refold)
    n = min(len(p), len(q))
    if n < 3:
        return float("nan")
    return kabsch_rmsd(p[:n], q[:n])


def foldseek_clusters(pdb_dir: Path, work: Path, binary: str, mode: str) -> int:
    """Cluster count from `foldseek easy-cluster`. mode: str | seq | str+seq."""
    alignment = {"str": "1", "seq": "2", "str+seq": "0"}[mode]
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / f"clu_{mode.replace('+', '')}"
    cmd = [binary, "easy-cluster", str(pdb_dir), str(prefix), str(work / "tmp"),
           "--alignment-type", alignment]
    result = subprocess.run(cmd, capture_output=True, text=True)
    report = Path(f"{prefix}_cluster.tsv")
    if result.returncode != 0 or not report.is_file():
        print(f"  foldseek {mode} FAILED: {result.stderr.strip()[:200]}")
        return -1
    with report.open() as handle:
        return len({line.split("\t")[0] for line in handle if line.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", required=True,
                        help="generated backbones, L<len>_s<idx>.cif")
    parser.add_argument("--codesign-dir", default=None,
                        help="co-design output; its samples/ holds the "
                             "full-atom structures and sequences")
    parser.add_argument("--refolds-dir", required=True,
                        help="ESMFold output with pdb/<fold_id>.pdb")
    parser.add_argument("--mpnn-designs", default=None, help="mpnn CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--foldseek", default="/users/yfsun/tools/foldseek/bin/foldseek")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refolds = Path(args.refolds_dir) / "pdb"
    samples = {p.stem: p for p in sorted(Path(args.samples_dir).glob("L*.cif"))}
    if not samples:
        samples = {p.stem: p for p in sorted(Path(args.samples_dir).glob("L*.pdb"))}
    print(f"{len(samples)} generated backbone(s)")

    # ---- designability: PMPNN sequences folded against the backbone -------
    per_sample = defaultdict(dict)
    if args.mpnn_designs:
        by_sample = defaultdict(list)
        with open(args.mpnn_designs, newline="") as handle:
            for row in csv.DictReader(handle):
                by_sample[row["sample_id"]].append(row)
        for sample_id, rows in by_sample.items():
            if sample_id not in samples:
                continue
            generated = load(samples[sample_id])
            rows.sort(key=lambda r: float(r["score"]))   # best first, for @1
            values = []
            for row in rows:
                pdb = refolds / f"{row['fold_id']}.pdb"
                if not pdb.is_file():
                    continue
                values.append(scrmsd(generated, load(pdb)))
            if values:
                per_sample[sample_id]["pmpnn1_scrmsd"] = values[0]
                per_sample[sample_id]["pmpnn8_scrmsd"] = float(np.nanmin(values))
                per_sample[sample_id]["pmpnn_n"] = len(values)

    # ---- codesignability: the model's OWN sequence ------------------------
    if args.codesign_dir:
        design_dir = Path(args.codesign_dir) / "samples"
        for sample_id in samples:
            pdb = refolds / f"{sample_id}__codesign.pdb"
            own = design_dir / f"{sample_id}.pdb"
            if not (pdb.is_file() and own.is_file()):
                continue
            generated, refold = load(own), load(pdb)
            per_sample[sample_id]["codesign_scrmsd_ca"] = scrmsd(generated, refold)
            per_sample[sample_id]["codesign_scrmsd_allatom"] = scrmsd(
                generated, refold, all_atom=True)

    # ---- write per-sample and aggregate -----------------------------------
    fields = ["sample_id", "length", "pmpnn1_scrmsd", "pmpnn8_scrmsd", "pmpnn_n",
              "codesign_scrmsd_ca", "codesign_scrmsd_allatom"]
    with (out / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in sorted(per_sample):
            row = {"sample_id": sample_id,
                   "length": int(sample_id.split("_")[0].lstrip("L"))}
            row.update(per_sample[sample_id])
            writer.writerow(row)

    def rate(key):
        vals = [v[key] for v in per_sample.values()
                if key in v and not np.isnan(v[key])]
        if not vals:
            return None
        return {"n": len(vals),
                "pct": round(100 * sum(1 for x in vals if x < args.threshold) / len(vals), 2),
                "median_scrmsd": round(float(np.median(vals)), 3)}

    summary = {
        "n_samples": len(samples),
        "threshold_angstrom": args.threshold,
        "designability_pmpnn1": rate("pmpnn1_scrmsd"),
        "designability_pmpnn8": rate("pmpnn8_scrmsd"),
        "codesignability_ca": rate("codesign_scrmsd_ca"),
        "codesignability_allatom": rate("codesign_scrmsd_allatom"),
    }

    # ---- diversity --------------------------------------------------------
    if Path(args.foldseek).is_file():
        source = (Path(args.codesign_dir) / "samples" if args.codesign_dir
                  else Path(args.samples_dir))
        summary["diversity"] = {
            mode: foldseek_clusters(source, out / "_foldseek", args.foldseek, mode)
            for mode in ("str", "seq", "str+seq")
        }
    else:
        print(f"  foldseek not at {args.foldseek}; diversity skipped")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
