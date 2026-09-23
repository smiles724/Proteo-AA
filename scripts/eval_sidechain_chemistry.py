#!/usr/bin/env python3
"""GT-free side-chain chemistry on designed binders.

    python scripts/eval_sidechain_chemistry.py <cell-dir> [<cell-dir> ...]

RMSD, chi-recovery and rotamer-recovery all compare a side chain against a
NATIVE one, so they need the native sequence. A design pipeline emits a
designed sequence, so there is no per-residue correspondence and those
metrics are undefined on its output. What survives without ground truth is
whether the packing is internally sensible:

  chi1 rotamer outliers  chi1 more than 40 deg from the nearest canonical
                         well (-60 g-, 180 t, +60 g+). Rotamers are strongly
                         clustered; sitting between wells is a packing
                         failure regardless of what the native residue was.
  side-chain clashes     heavy atoms in different residues closer than the
                         threshold, excluding backbone and excluding
                         sequence neighbours, which are covalently close.

Neither says the design is good. They say the packer did not produce
something physically impossible, which is the only side-chain claim
available without a native to compare to.

CALIBRATE BEFORE READING. Both numbers are meaningless without a native
baseline under the identical metric. Ten native target structures from the
binder benchmark give:

    chi1 outliers          3.6 %
    clashes per 100 SC atoms   2.19   (range 1.30 - 3.11)

So a LOW clash count is not automatically good -- real proteins pack
tightly, and a design well under 2.19 is under-packed rather than clean.
And a 0 % chi1 outlier rate is not better than native: it means the packer
never leaves a rotamer well, where real side chains do so 3.6 % of the
time. Reproduce the baseline with:

    python scripts/eval_sidechain_chemistry.py \
        runs/binder_bench/targets/structures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

#: chi1 is N-CA-CB-<first gamma>. Per residue, the gamma atom that defines it.
GAMMA = {
    "ARG": "CG", "ASN": "CG", "ASP": "CG", "CYS": "SG", "GLN": "CG",
    "GLU": "CG", "HIS": "CG", "ILE": "CG1", "LEU": "CG", "LYS": "CG",
    "MET": "CG", "PHE": "CG", "PRO": "CG", "SER": "OG", "THR": "OG1",
    "TRP": "CG", "TYR": "CG", "VAL": "CG1",
}
#: ALA has no chi1; GLY has no CB.
WELLS = (-60.0, 60.0, 180.0)
OUTLIER_DEG = 40.0
BACKBONE = ("N", "CA", "C", "O", "OXT")
CLASH = 3.0


def dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def well_distance(chi1: float) -> float:
    """Angular distance to the nearest canonical rotamer well, in degrees."""
    return min(abs((chi1 - w + 180.0) % 360.0 - 180.0) for w in WELLS)


def analyse(path: Path, chain: str | None):
    import warnings
    warnings.filterwarnings("ignore")
    from biotite.structure.io.pdb import PDBFile

    atoms = PDBFile.read(str(path)).get_structure(model=1)
    atoms = atoms[atoms.element != "H"]
    chains = sorted(set(atoms.chain_id))
    if chain is None:
        # the binder is the shorter chain
        lengths = {c: len(set(atoms[atoms.chain_id == c].res_id)) for c in chains}
        chain = min(lengths, key=lengths.get)
    sel = atoms[atoms.chain_id == chain]

    outliers = scored = 0
    worst = 0.0
    for rid in sorted(set(sel.res_id)):
        res = sel[sel.res_id == rid]
        name = str(res.res_name[0])
        gamma = GAMMA.get(name)
        if gamma is None:
            continue                      # ALA, GLY: no chi1
        need = {"N": None, "CA": None, "CB": None, gamma: None}
        for atom_name in need:
            hit = res[res.atom_name == atom_name]
            if len(hit) != 1:
                break
            need[atom_name] = hit.coord[0].astype(np.float64)
        if any(v is None for v in need.values()):
            continue                      # an atom the packer never built
        chi1 = dihedral(need["N"], need["CA"], need["CB"], need[gamma])
        distance = well_distance(chi1)
        scored += 1
        worst = max(worst, distance)
        outliers += int(distance > OUTLIER_DEG)

    # side-chain clashes: different residues, neither atom backbone, and not
    # sequence neighbours (i, i+1 side chains are legitimately close)
    side = sel[~np.isin(sel.atom_name, BACKBONE)]
    clashes = 0
    if len(side) > 1:
        xyz = side.coord.astype(np.float64)
        rid = side.res_id
        d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        near = np.abs(rid[:, None] - rid[None, :]) <= 1
        d[near] = np.inf
        clashes = int((d < CLASH).sum() // 2)

    return {
        "chain": chain, "chi1_scored": scored, "chi1_outliers": outliers,
        "chi1_outlier_fraction": outliers / scored if scored else float("nan"),
        "worst_well_distance_deg": worst,
        "sidechain_clashes": clashes,
        "sidechain_atoms": len(side),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+")
    parser.add_argument("--chain", default=None,
                        help="binder chain; default is the shorter one")
    args = parser.parse_args()

    for spec in args.cells:
        cell = Path(spec)
        if (cell / "designs").is_dir():
            pdbs = sorted((cell / "designs").glob("*.pdb"))
        elif cell.is_dir():
            pdbs = sorted(cell.glob("*.pdb"))     # a directory of natives
        else:
            pdbs = [cell]
        if not pdbs:
            print(f"\n=== {cell.name}: no PDBs ===")
            continue
        print(f"\n=== {cell.name} ===")
        print(f"  {'design':<52} {'chi1 out':>9} {'worst':>7} {'clash':>6}")
        totals = {"scored": 0, "outliers": 0, "clashes": 0, "atoms": 0}
        for pdb in pdbs:
            r = analyse(pdb, args.chain)
            totals["scored"] += r["chi1_scored"]
            totals["outliers"] += r["chi1_outliers"]
            totals["clashes"] += r["sidechain_clashes"]
            totals["atoms"] += r["sidechain_atoms"]
            print(f"  {pdb.stem:<52} "
                  f"{r['chi1_outliers']:>3}/{r['chi1_scored']:<5} "
                  f"{r['worst_well_distance_deg']:>6.1f} "
                  f"{r['sidechain_clashes']:>6}")
        frac = (totals["outliers"] / totals["scored"]) if totals["scored"] else float("nan")
        per100 = (100 * totals["clashes"] / totals["atoms"]) if totals["atoms"] else float("nan")
        print(f"  {'POOLED':<52} {totals['outliers']:>3}/{totals['scored']:<5} "
              f"{'':>6} {totals['clashes']:>6}")
        print(f"  chi1 outlier fraction {100 * frac:.1f}% (native 3.6%)   "
              f"clashes per 100 SC atoms {per100:.2f} (native 2.19)")


if __name__ == "__main__":
    main()
