#!/usr/bin/env python
"""Regenerate the ``mu_ideal`` literal in ``pxdesign_train/sidechain/templates.py``.

This is the extraction artifact behind ``templates.IDEAL_SC_LOCAL``. It is an
OFFLINE tool: nothing in the training package imports it. It exists so the
provenance of the table is reproducible and so the frame convention cannot
silently drift -- it does not re-implement Gram-Schmidt, it calls
``frames.build_frame`` / ``frames.to_local``, the exact functions the model uses.

Pipeline
--------
  wwPDB Chemical Component Dictionary (components.cif)
    -> per-residue ideal heavy-atom coords  (_chem_comp_atom.pdbx_model_Cartn_*_ideal)
    -> frames.build_frame(N, CA, C)         (that residue's OWN ideal backbone)
    -> frames.to_local(side-chain atoms)    (origin = CA)
    -> literal, rows = instantiate.STD_AA_3, cols = instantiate.sidechain_atoms(r)

Usage
-----
    python scripts/build_sidechain_templates.py PXDesign/release_data/ccd_cache/components.v20240608.cif
    python scripts/build_sidechain_templates.py <ccd.cif> --check     # diff against the shipped table
    python scripts/build_sidechain_templates.py <ccd.cif> -o out.py   # write the literal to a file

``--check`` is the useful mode in CI-ish settings: it re-derives the table from
the CCD and fails if the shipped literal disagrees by more than --atol.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "Protenix", _REPO / "PXDesign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pxdesign_train.sidechain.frames import build_frame, to_local  # noqa: E402
from pxdesign_train.sidechain.instantiate import (  # noqa: E402
    MAX_SC,
    STD_AA_3,
    sidechain_atoms,
)

BACKBONE = ("N", "CA", "C")
DECIMALS = 4


# ---------------------------------------------------------------------------
# minimal streaming mmCIF reader (components.cif is ~1 GB; never load it whole)
# ---------------------------------------------------------------------------
def read_ccd_ideal_coords(cif_path: Path, wanted: List[str]) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Return {comp_id: {atom_id: (x, y, z)}} of *ideal* heavy-atom coords.

    Only the ``_chem_comp_atom`` loop is parsed, and only for the requested
    components. Hydrogens and leaving atoms are dropped. Streaming, one pass.
    """
    todo = set(wanted)
    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    comp = None            # current data_ block id
    cols: List[str] = []   # column names of the loop currently being read
    in_loop = False        # inside a `loop_` header
    in_atom_loop = False   # inside the _chem_comp_atom loop's data rows

    with cif_path.open("r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()

            if s.startswith("data_"):
                if not todo:
                    break
                comp = s[5:].strip()
                in_loop = in_atom_loop = False
                cols = []
                continue

            if comp not in todo:
                continue

            if s == "loop_":
                in_loop, in_atom_loop, cols = True, False, []
                continue

            if in_loop and s.startswith("_"):
                if s.startswith("_chem_comp_atom."):
                    cols.append(s.split(".", 1)[1].split()[0])
                else:  # some other loop -> ignore its rows
                    in_loop, cols = False, []
                continue

            if in_loop and cols and not s.startswith("_"):
                in_loop, in_atom_loop = False, True  # first data row of the atom loop

            if in_atom_loop:
                if not s or s.startswith("#") or s.startswith("_") or s == "loop_":
                    in_atom_loop = False
                    if s == "loop_":
                        in_loop, cols = True, []
                    continue
                tok = _split_cif_row(s)
                if len(tok) != len(cols):
                    continue
                row = dict(zip(cols, tok))
                if row.get("type_symbol", "").upper() == "H":
                    continue
                name = row["atom_id"].strip('"').strip("'")
                try:
                    xyz = (
                        float(row["pdbx_model_Cartn_x_ideal"]),
                        float(row["pdbx_model_Cartn_y_ideal"]),
                        float(row["pdbx_model_Cartn_z_ideal"]),
                    )
                except (KeyError, ValueError):
                    continue  # '?' / '.' -> no ideal coords for this atom
                out.setdefault(comp, {})[name] = xyz

    missing = set(wanted) - set(out)
    if missing:
        raise SystemExit(f"components not found in {cif_path}: {sorted(missing)}")
    return out


def _split_cif_row(s: str) -> List[str]:
    """Whitespace split that respects single/double quotes."""
    tok, cur, q = [], "", ""
    for ch in s:
        if q:
            if ch == q:
                q = ""
            else:
                cur += ch
        elif ch in "\"'":
            q = ch
        elif ch.isspace():
            if cur:
                tok.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        tok.append(cur)
    return tok


# ---------------------------------------------------------------------------
# CCD ideal coords -> residue-local template, via frames.py (no re-implementation)
# ---------------------------------------------------------------------------
def build_table(ccd: Dict[str, Dict[str, Tuple[float, float, float]]]) -> torch.Tensor:
    table = torch.zeros(len(STD_AA_3), MAX_SC, 3, dtype=torch.float64)
    for i, restype in enumerate(STD_AA_3):
        atoms = ccd[restype]
        for bb in BACKBONE:
            if bb not in atoms:
                raise SystemExit(f"{restype}: CCD entry is missing backbone atom {bb}")
        n, ca, c = (torch.tensor(atoms[b], dtype=torch.float64) for b in BACKBONE)

        # THE convention: whatever frames.py does, we do. Never re-derived here.
        R, t = build_frame(n, ca, c)

        names = sidechain_atoms(restype)  # column order contract
        if not names:
            continue  # GLY
        missing = [a for a in names if a not in atoms]
        if missing:
            raise SystemExit(f"{restype}: CCD entry is missing side-chain atoms {missing}")
        glob = torch.tensor([atoms[a] for a in names], dtype=torch.float64)  # [k, 3]
        table[i, : len(names)] = to_local(glob, R, t)
    return table


def render_literal(table: torch.Tensor) -> str:
    lines = ["_IDEAL_SC_LOCAL_LIST = ["]
    for i, restype in enumerate(STD_AA_3):
        names = sidechain_atoms(restype)
        n = len(names)
        head = f"    # {restype} ({n} side-chain heavy atom{'s' if n != 1 else ''}): "
        lines.append(head + (", ".join(names) if names else "none"))
        lines.append("    [")
        for j in range(MAX_SC):
            x, y, z = (round(float(v), DECIMALS) + 0.0 for v in table[i, j])
            label = names[j] if j < n else "pad"
            lines.append(f"        [{x:8.4f}, {y:8.4f}, {z:8.4f}],  # {label}")
        lines.append("    ],")
    lines.append("]")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ccd", type=Path, help="path to wwPDB components.cif")
    ap.add_argument("-o", "--out", type=Path, default=None, help="write the literal here (default: stdout)")
    ap.add_argument("--check", action="store_true", help="compare against the shipped templates.IDEAL_SC_LOCAL")
    ap.add_argument("--atol", type=float, default=1.5e-4, help="tolerance for --check (default: rounding)")
    args = ap.parse_args()

    ccd = read_ccd_ideal_coords(args.ccd, STD_AA_3)
    table = build_table(ccd)

    if args.check:
        from pxdesign_train.sidechain.templates import IDEAL_SC_LOCAL

        ref = IDEAL_SC_LOCAL.double()
        got = torch.tensor(  # round-trip through the literal's precision
            [[[round(float(v), DECIMALS) for v in table[i, j]] for j in range(MAX_SC)] for i in range(len(STD_AA_3))],
            dtype=torch.float64,
        )
        err = (got - ref).abs()
        bad = (err > args.atol).nonzero()
        print(f"max |regenerated - shipped| = {err.max().item():.2e} over {ref.numel()} entries")
        for i, j, k in bad.tolist():
            print(f"  MISMATCH {STD_AA_3[i]} col {j} axis {k}: {got[i, j, k]:.4f} vs {ref[i, j, k]:.4f}")
        if len(bad):
            print(f"FAIL: {len(bad)} entries differ by more than {args.atol}")
            return 1
        print("OK: shipped templates.IDEAL_SC_LOCAL reproduces from the CCD under frames.py's convention")
        return 0

    text = render_literal(table)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
