#!/usr/bin/env python
"""Regenerate ``pxdesign_train/sidechain/chi_constants.py``.

This is the ``G_ideal(a)`` half of Overleaf 0714 Appendix "Residue-Specific
Side-Chain Template Construction", Step 1: the residue-specific *ideal covalent
geometry* — atom connectivity, bond lengths, bond angles and rigid-group
definitions — plus the torsion (chi) definitions and the count ``K_i`` of valid
side-chain torsions.

Where each piece actually comes from
------------------------------------
The appendix says "a lookup in the AlphaFold/OpenFold residue constants". Protenix
ships the AF-derived *torsion* tables (``_CHI_ANGLES_ATOMS``, ``_CHI_ANGLES_MASK``)
but NOT ``rigid_group_atom_positions`` — i.e. it has no ideal Cartesian coordinates
at all. So:

  * chi definitions + K_i      <- protenix.data.constants (AF/OpenFold, as specified)
  * ideal bond lengths/angles  <- wwPDB CCD ideal conformer (already the provenance of
                                  templates.IDEAL_SC_LOCAL; a rigid rotation about a
                                  torsion axis preserves every bond length and bond
                                  angle, so the CCD conformer *is* G_ideal's metric part)
  * connectivity / rigid groups<- wwPDB CCD ``_chem_comp_bond`` (derived here, not
                                  hand-written: the atoms that move with chi_k are
                                  exactly the connected component of the chi_k axis's
                                  distal atom after cutting the axis bond)

Rotatability
------------
A torsion is ROTATABLE iff cutting its axis bond (a1-a2) actually disconnects the
distal side from the proximal side. This is a real test, not an assumption: it
correctly reports PRO's chi1/chi2 as NOT rotatable, because the pyrrolidine ring
closes CD back onto the backbone N, so no rigid rotation about CA-CB exists that
keeps the ring intact. PRO therefore keeps its CCD conformer (see buildsc.py).

Usage
-----
    python scripts/build_chi_constants.py PXDesign/release_data/ccd_cache/components.v20240608.cif
    python scripts/build_chi_constants.py <ccd.cif> --check
    python scripts/build_chi_constants.py <ccd.cif> -o pxdesign_train/sidechain/chi_constants.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "Protenix", _REPO / "PXDesign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from protenix.data.constants import _CHI_ANGLES_ATOMS  # noqa: E402
from pxdesign_train.sidechain.frames import build_frame, to_local  # noqa: E402
from pxdesign_train.sidechain.instantiate import (  # noqa: E402
    MAX_SC,
    STD_AA_3,
    sidechain_atoms,
)

# Reuse the CCD reader that already backs templates.py — one parser, one provenance.
from build_sidechain_templates import _split_cif_row, read_ccd_ideal_coords  # noqa: E402

BACKBONE = ("N", "CA", "C")
MAX_CHI = 4
DECIMALS = 4


def read_ccd_bonds(cif_path: Path, wanted: List[str]) -> Dict[str, Set[Tuple[str, str]]]:
    """Return {comp_id: {(atom1, atom2), ...}} heavy-atom bonds from ``_chem_comp_bond``."""
    todo = set(wanted)
    out: Dict[str, Set[Tuple[str, str]]] = {c: set() for c in wanted}

    comp = None
    cols: List[str] = []
    in_loop = False
    in_bond_loop = False

    with cif_path.open("r", errors="replace") as fh:
        for raw in fh:
            s = raw.strip()

            if s.startswith("data_"):
                comp = s[5:].strip()
                in_loop = in_bond_loop = False
                cols = []
                continue

            if comp not in todo:
                continue

            if s == "loop_":
                in_loop, in_bond_loop, cols = True, False, []
                continue

            if in_loop and s.startswith("_"):
                if s.startswith("_chem_comp_bond."):
                    cols.append(s.split(".", 1)[1].split()[0])
                else:
                    in_loop, cols = False, []
                continue

            if in_loop and cols and not s.startswith("_"):
                in_loop, in_bond_loop = False, True

            if in_bond_loop:
                if not s or s.startswith("#") or s.startswith("_") or s == "loop_":
                    in_bond_loop = False
                    if s == "loop_":
                        in_loop, cols = True, []
                    continue
                tok = _split_cif_row(s)
                if len(tok) != len(cols):
                    continue
                row = dict(zip(cols, tok))
                a = row["atom_id_1"].strip('"').strip("'")
                b = row["atom_id_2"].strip('"').strip("'")
                out[comp].add((a, b))

    return out


def _downstream(bonds: Set[Tuple[str, str]], a1: str, a2: str) -> Tuple[Set[str], bool]:
    """Atoms rigidly carried by a rotation about the a1->a2 bond axis.

    Cut the (a1, a2) edge, then flood-fill from a2. If a1 is reachable anyway the
    bond lies on a ring and no rigid torsion exists -> rotatable=False.
    """
    adj: Dict[str, Set[str]] = defaultdict(set)
    for x, y in bonds:
        if {x, y} == {a1, a2}:
            continue  # the cut edge
        adj[x].add(y)
        adj[y].add(x)

    seen = {a2}
    q = deque([a2])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)

    rotatable = a1 not in seen
    return seen, rotatable


def build_tables(
    coords: Dict[str, Dict[str, Tuple[float, float, float]]],
    bonds: Dict[str, Set[Tuple[str, str]]],
):
    n_res = len(STD_AA_3)
    # Combined per-residue index space used by the chi definitions:
    #   0=N, 1=CA, 2=C, then 3+j = side-chain column j (instantiate.sidechain_atoms order)
    chi_atom_idx = torch.zeros(n_res, MAX_CHI, 4, dtype=torch.long)
    chi_mask = torch.zeros(n_res, MAX_CHI, dtype=torch.bool)
    chi_rotatable = torch.zeros(n_res, MAX_CHI, dtype=torch.bool)
    chi_downstream = torch.zeros(n_res, MAX_CHI, MAX_SC, dtype=torch.bool)
    ideal_bb = torch.zeros(n_res, 3, 3, dtype=torch.float64)

    for r, restype in enumerate(STD_AA_3):
        atoms = coords[restype]
        sc_names = sidechain_atoms(restype)
        combined = list(BACKBONE) + sc_names
        pos = {name: i for i, name in enumerate(combined)}

        n, ca, c = (torch.tensor(atoms[b], dtype=torch.float64) for b in BACKBONE)
        R, t = build_frame(n, ca, c)
        bb_glob = torch.tensor([atoms[b] for b in BACKBONE], dtype=torch.float64)
        ideal_bb[r] = to_local(bb_glob, R, t)

        chis = _CHI_ANGLES_ATOMS[restype]
        # Only bonds among the atoms we actually model (heavy, backbone + side chain).
        keep = set(combined)
        rb = {(a, b) for (a, b) in bonds[restype] if a in keep and b in keep}

        for k, quad in enumerate(chis):
            if any(a not in pos for a in quad):
                raise SystemExit(f"{restype} chi{k+1}: atom(s) {quad} not in {combined}")
            chi_mask[r, k] = True
            chi_atom_idx[r, k] = torch.tensor([pos[a] for a in quad], dtype=torch.long)

            a1, a2 = quad[1], quad[2]          # the rotatable bond axis
            moved, rotatable = _downstream(rb, a1, a2)
            chi_rotatable[r, k] = rotatable
            if rotatable:
                for j, name in enumerate(sc_names):
                    if name in moved:
                        chi_downstream[r, k, j] = True
                # a3 must move, or the torsion definition is inconsistent
                if not chi_downstream[r, k, pos[quad[3]] - len(BACKBONE)]:
                    raise SystemExit(f"{restype} chi{k+1}: distal atom {quad[3]} does not move")

    return chi_atom_idx, chi_mask, chi_rotatable, chi_downstream, ideal_bb


def render(chi_atom_idx, chi_mask, chi_rotatable, chi_downstream, ideal_bb) -> str:
    L: List[str] = []
    A = L.append
    A('"""Ideal covalent geometry G_ideal(a) and torsion definitions — GENERATED.')
    A("")
    A("Regenerate with::")
    A("")
    A("    python scripts/build_chi_constants.py <components.cif> -o pxdesign_train/sidechain/chi_constants.py")
    A("    python scripts/build_chi_constants.py <components.cif> --check")
    A("")
    A("Overleaf 0714 Appendix, Step 1 (`a_hat -> (A_sc, K_i, G_ideal)`). Do not hand-edit.")
    A("")
    A("Index space for CHI_ATOM_IDX: 0=N, 1=CA, 2=C, then 3+j = side-chain column j")
    A("in ``instantiate.sidechain_atoms(restype)`` order — the same column order as")
    A("``templates.IDEAL_SC_LOCAL``.")
    A('"""')
    A("import torch")
    A("")
    A("MAX_CHI = 4")
    A("N_BB_FRAME = 3  # N, CA, C occupy combined indices 0, 1, 2")
    A("")

    A("# [20, 4, 4] long — the four atoms defining each chi (combined index space).")
    A("CHI_ATOM_IDX = torch.tensor([")
    for r, restype in enumerate(STD_AA_3):
        quads = ", ".join(
            "[" + ", ".join(f"{int(v):2d}" for v in chi_atom_idx[r, k]) + "]"
            for k in range(MAX_CHI)
        )
        A(f"    [{quads}],  # {restype}")
    A("], dtype=torch.long)")
    A("")

    A("# [20, 4] bool — K_i: which chi angles the residue actually has (AF chi mask).")
    A("CHI_MASK = torch.tensor([")
    for r, restype in enumerate(STD_AA_3):
        vals = ", ".join(f"{bool(v)!s:5s}" for v in chi_mask[r])
        A(f"    [{vals}],  # {restype}")
    A("], dtype=torch.bool)")
    A("")

    A("# [20, 4] bool — torsion is a genuine rigid rotation (False on ring-closed")
    A("# torsions: PRO chi1/chi2, where CD bonds back to N and no rigid rotation exists).")
    A("CHI_ROTATABLE = torch.tensor([")
    for r, restype in enumerate(STD_AA_3):
        vals = ", ".join(f"{bool(v)!s:5s}" for v in chi_rotatable[r])
        A(f"    [{vals}],  # {restype}")
    A("], dtype=torch.bool)")
    A("")

    A("# [20, 4, MAX_SC] bool — side-chain atoms rigidly carried by each chi rotation")
    A("# (connected component of the axis's distal atom after cutting the axis bond).")
    A("CHI_DOWNSTREAM = torch.tensor([")
    for r, restype in enumerate(STD_AA_3):
        names = sidechain_atoms(restype)
        A(f"    [  # {restype}: {', '.join(names) if names else 'none'}")
        for k in range(MAX_CHI):
            vals = ", ".join(f"{bool(v)!s:5s}" for v in chi_downstream[r, k])
            moved = [names[j] for j in range(len(names)) if bool(chi_downstream[r, k, j])]
            tag = f"chi{k+1}: {', '.join(moved)}" if moved else f"chi{k+1}: -"
            A(f"        [{vals}],  # {tag}")
        A("    ],")
    A("], dtype=torch.bool)")
    A("")

    A("# [20, 3, 3] float32 — ideal N, CA, C in the residue-LOCAL frame (frames.build_frame).")
    A("# CA is the frame origin, so its row is exactly zero. chi1's first atom is N, which")
    A("# is why the backbone has to be in the same local frame as the side-chain template.")
    A("IDEAL_BB_LOCAL = torch.tensor([")
    for r, restype in enumerate(STD_AA_3):
        row = ", ".join(
            "[" + ", ".join(f"{round(float(v), DECIMALS) + 0.0:8.4f}" for v in ideal_bb[r, i]) + "]"
            for i in range(3)
        )
        A(f"    [{row}],  # {restype}")
    A("], dtype=torch.float32)")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ccd", type=Path, help="path to wwPDB components.cif")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="compare against the shipped chi_constants")
    args = ap.parse_args()

    coords = read_ccd_ideal_coords(args.ccd, STD_AA_3)
    bonds = read_ccd_bonds(args.ccd, STD_AA_3)
    tables = build_tables(coords, bonds)

    if args.check:
        from pxdesign_train.sidechain import chi_constants as cc

        names = ["CHI_ATOM_IDX", "CHI_MASK", "CHI_ROTATABLE", "CHI_DOWNSTREAM", "IDEAL_BB_LOCAL"]
        bad = 0
        for name, got in zip(names, tables):
            ref = getattr(cc, name)
            if name == "IDEAL_BB_LOCAL":
                err = (got.float() - ref).abs().max().item()
                ok = err <= 1.5e-4
                print(f"{name}: max|regen - shipped| = {err:.2e} {'OK' if ok else 'FAIL'}")
            else:
                ok = bool(torch.equal(got, ref))
                print(f"{name}: {'OK' if ok else 'FAIL'} (exact)")
            bad += (not ok)
        if bad:
            print(f"FAIL: {bad} table(s) differ")
            return 1
        print("OK: shipped chi_constants regenerates from the CCD + AF chi definitions")
        return 0

    text = render(*tables)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
