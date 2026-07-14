#!/usr/bin/env python
"""Build the compact backbone-dependent rotamer library used by Step 2 of the
Overleaf 0714 appendix ("Residue-Specific Side-Chain Template Construction").

    (a_hat, phi_hat, psi_hat)  ->  { p_{i,r},  chibar_{i,r} }_{r=1..R_i}

Source
------
The 2010 Backbone-Dependent Rotamer Library (BBDEP2010) of the Dunbrack lab —
the appendix's stated primary choice. BBDEP2010 has been distributed under the
Open Data Commons Attribution License (ODC-By) since 2019-07-25, so it can be
redistributed (academic *and* commercial) provided the source is cited:

    Shapovalov, M.V., and Dunbrack, R.L., Jr. (2011). A smoothed backbone-dependent
    rotamer library for proteins derived from adaptive kernel density estimates and
    regressions. Structure 19, 844-858.

NOTE ON THE PAPER'S CITATION: the draft calls this "the Dunbrack 2010 library" and
cite.bib currently only has `dunbrack1997bayesian`, which is a *different* (and
backbone-independent-era) paper. The correct citation for BBDEP2010 is the 2011
Structure paper above.

Input format (one row per rotamer per backbone bin)::

    res,phi,psi,r1,r2,r3,r4,prob,chi1-val,...,chi4-val,chi1-sig,...,chi4-sig
    ARG,-180,-180,1,2,2,1,0.249730,62.5,176.9,176.6,85.7,6.9,11.1,10.5,9.9

Output
------
``pxdesign_train/sidechain/data/dunbrack2010_bbdep.npz`` — a ragged table:

    counts  [20, 36, 36] int32   rotamers in each (restype, phi-bin, psi-bin) cell
    offsets [20, 36, 36] int64   start index into the flat arrays
    probs   [Ntot]       float32 p_{i,r}, descending within a cell, sums to 1
    chis    [Ntot, 4]    int16   chibar_{i,r} in TENTHS OF A DEGREE (exact: the
                                 source has one decimal place)
    marg_*                       the same, marginalised uniformly over the phi/psi
                                 grid — used only where phi/psi are undefined
                                 (chain termini), never as the main path.

The grid is 10 degrees, phi/psi in [-180, 170] (36 x 36). The source's +180 slice
is the periodic image of -180 and is dropped after verifying it is identical.

ALA and GLY have no chi and therefore no rotamers (count 0 everywhere); PRO's
torsions are ring-closed (see chi_constants.CHI_ROTATABLE) so its rotamers are
stored but not applied by buildsc.

Usage
-----
    python scripts/build_rotamer_library.py --download
    python scripts/build_rotamer_library.py dunbrack-2010.lib.csv
    python scripts/build_rotamer_library.py dunbrack-2010.lib.csv --check
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "Protenix", _REPO / "PXDesign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pxdesign_train.sidechain.instantiate import STD_AA_3  # noqa: E402

# ODC-By redistribution of BBDEP2010, parsed to CSV.
SOURCE_URL = "https://raw.githubusercontent.com/TKanX/dunbrack/main/data/dunbrack-2010.lib.csv"

OUT = _REPO / "pxdesign_train" / "sidechain" / "data" / "dunbrack2010_bbdep.npz"

N_BIN = 36          # 10-degree grid over [-180, 170]
BIN_DEG = 10
MAX_CHI = 4

# BBDEP2010 splits some residues by protonation / ring state. We use the generic
# entry for each: CYS (not CYH/CYD, which are the reduced/disulfide-bonded splits)
# and PRO (not CPR/TPR, the cis/trans-proline splits).
DUNBRACK_TO_STD = {r: r for r in STD_AA_3 if r not in ("ALA", "GLY")}
IGNORED = {"CYH", "CYD", "CPR", "TPR"}


def bin_of(angle: int) -> int:
    """Grid index of a phi/psi value, periodic: +180 folds onto -180."""
    return int(round((angle + 180) / BIN_DEG)) % N_BIN


def parse(csv_path: Path):
    """Read the CSV into cell -> list of (prob, chi[4]) sorted by descending prob."""
    cells = defaultdict(list)          # (res_idx, phi_bin, psi_bin) -> [(p, (c1..c4))]
    dup_check = defaultdict(dict)      # to verify the +180 periodic image

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            res = row["res"]
            if res in IGNORED or res not in DUNBRACK_TO_STD:
                continue
            r = STD_AA_3.index(DUNBRACK_TO_STD[res])
            phi, psi = int(row["phi"]), int(row["psi"])
            prob = float(row["prob"])
            chi = tuple(float(row[f"chi{k}-val"]) for k in range(1, MAX_CHI + 1))

            key = (r, bin_of(phi), bin_of(psi))
            rot = (row["r1"], row["r2"], row["r3"], row["r4"])
            if phi == 180 or psi == 180:
                dup_check[key][rot] = (prob, chi)   # periodic image, verify not store
            else:
                cells[key].append((prob, chi))
                dup_check[key].setdefault(rot, None)

    # The +180 slice must be the exact periodic image of the -180 slice.
    bad = 0
    for key, rots in dup_check.items():
        for rot, image in rots.items():
            if image is None:
                continue
            stored = [c for c in cells.get(key, [])]
            if not any(abs(p - image[0]) < 1e-6 for p, _ in stored):
                bad += 1
    if bad:
        print(f"WARNING: {bad} rotamers in the +180 slice are not periodic images of -180")

    for key in cells:
        cells[key].sort(key=lambda t: -t[0])
    return cells


def to_ragged(cells):
    counts = np.zeros((len(STD_AA_3), N_BIN, N_BIN), dtype=np.int32)
    for (r, i, j), rots in cells.items():
        counts[r, i, j] = len(rots)

    offsets = np.zeros_like(counts, dtype=np.int64)
    offsets.reshape(-1)[:] = np.concatenate([[0], np.cumsum(counts.reshape(-1))[:-1]])
    total = int(counts.sum())

    probs = np.zeros(total, dtype=np.float32)
    chis = np.zeros((total, MAX_CHI), dtype=np.int16)

    for (r, i, j), rots in cells.items():
        o = int(offsets[r, i, j])
        p = np.array([t[0] for t in rots], dtype=np.float64)
        s = p.sum()
        if s > 0:
            p = p / s                       # renormalise: p_{i,r} is a distribution
        probs[o : o + len(rots)] = p.astype(np.float32)
        for k, (_, chi) in enumerate(rots):
            # tenths of a degree, wrapped to [-1800, 1800) -> exactly representable
            q = [int(round(((c + 180.0) % 360.0 - 180.0) * 10)) for c in chi]
            chis[o + k] = np.array(q, dtype=np.int16)

    return counts, offsets, probs, chis


def marginal(cells):
    """Uniform marginal over the phi/psi grid, per residue type.

    Used ONLY where phi or psi is undefined (the first / last residue of a chain,
    which have no preceding C / following N). Rotamers are merged by their chi
    vector; p is averaged uniformly over the grid and chibar is p-weighted.
    """
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, np.zeros(MAX_CHI)]))
    ncell = defaultdict(int)
    for (r, i, j), rots in cells.items():
        ncell[r] += 1
        for p, chi in rots:
            key = tuple(int(round(c / 30.0)) for c in chi)   # coarse chi identity
            e = agg[r][key]
            e[0] += p
            e[1] += p * np.array(chi)

    counts = np.zeros(len(STD_AA_3), dtype=np.int32)
    rows = {}
    for r, d in agg.items():
        items = []
        for _, (psum, cw) in d.items():
            items.append((psum / max(ncell[r], 1), cw / max(psum, 1e-12)))
        items.sort(key=lambda t: -t[0])
        rows[r] = items
        counts[r] = len(items)

    offsets = np.zeros(len(STD_AA_3), dtype=np.int64)
    offsets[1:] = np.cumsum(counts)[:-1]
    total = int(counts.sum())
    probs = np.zeros(total, dtype=np.float32)
    chis = np.zeros((total, MAX_CHI), dtype=np.int16)
    for r, items in rows.items():
        o = int(offsets[r])
        p = np.array([t[0] for t in items], dtype=np.float64)
        p = p / p.sum() if p.sum() > 0 else p
        probs[o : o + len(items)] = p.astype(np.float32)
        for k, (_, chi) in enumerate(items):
            q = [int(round(((c + 180.0) % 360.0 - 180.0) * 10)) for c in chi]
            chis[o + k] = np.array(q, dtype=np.int16)
    return counts, offsets, probs, chis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="?", default=None)
    ap.add_argument("--download", action="store_true", help=f"fetch {SOURCE_URL}")
    ap.add_argument("--check", action="store_true", help="verify the shipped npz matches")
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    args = ap.parse_args()

    src = args.csv
    if args.download or src is None:
        src = Path("dunbrack-2010.lib.csv")
        if not src.exists():
            print(f"downloading {SOURCE_URL} ...", file=sys.stderr)
            urllib.request.urlretrieve(SOURCE_URL, src)

    cells = parse(src)
    counts, offsets, probs, chis = to_ragged(cells)
    m_counts, m_offsets, m_probs, m_chis = marginal(cells)

    print(f"cells with rotamers : {int((counts > 0).sum())}")
    print(f"total rotamers      : {len(probs)}")
    print(f"max rotamers / cell : {int(counts.max())}")

    if args.check:
        z = np.load(args.out)
        ok = (
            np.array_equal(z["counts"], counts)
            and np.array_equal(z["chis"], chis)
            and np.allclose(z["probs"], probs, atol=1e-6)
        )
        print("OK: shipped npz reproduces from source" if ok else "FAIL: shipped npz differs")
        return 0 if ok else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        counts=counts, offsets=offsets, probs=probs, chis=chis,
        marg_counts=m_counts, marg_offsets=m_offsets, marg_probs=m_probs, marg_chis=m_chis,
        restypes=np.array(STD_AA_3),
        bin_deg=np.int32(BIN_DEG),
        citation=np.array(
            "Shapovalov & Dunbrack (2011) Structure 19:844-858 (BBDEP2010, ODC-By)"
        ),
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
