#!/usr/bin/env python3
"""Verify every target config against the structure it names.

The configs are a transcription of a table in someone else's paper, and the
failure mode of a bad transcription is silence: a crop range that does not
exist selects nothing, a hotspot on the wrong chain is ignored, and the run
proceeds to design a binder against an epitope nobody chose. This script exists
because that is exactly what happened with H1 -- AlphaProteo publishes it in HA
numbering while 5vli deposits HA2 offset by +500, so a literal "B1-68" matched
no residue at all.

    python benchmarks/alphaproteo10/check_targets.py

Exits non-zero if anything does not line up. Run
`benchmarks/alphaproteo10/download_structures.sh` first.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

try:
    import biotite.structure.io.pdbx as pdbx
except ImportError:  # pragma: no cover - environment, not logic
    sys.exit("needs biotite: pip install biotite")

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_BLOCK = re.compile(
    r"^    ([A-Za-z0-9]+):\n"
    r"      crop: \[([^\]]+)\]"
    r"((?:\n      hotspots: \[[^\]]+\])?)",
    re.M,
)


def _polymer(path: str):
    """auth chain / auth residue for polymer atoms only.

    HETATM rows (glycans, buffer, water) carry their own numbering and would
    otherwise widen a chain's apparent residue range far past the protein --
    5vli's chain A looks like 5-719 with them and 5-325 without.
    """
    site = pdbx.CIFFile.read(path).block["atom_site"]
    keep = np.array(site["group_PDB"].as_array()) == "ATOM"
    return (
        np.array(site["auth_asym_id"].as_array())[keep],
        np.array(site["auth_seq_id"].as_array(int))[keep],
    )


def check(cfg: str) -> tuple[list[str], list[str]]:
    text = open(cfg).read()
    name = os.path.basename(cfg)[:-5]
    m = re.search(r"structures/(\w+)\.cif", text)
    if not m:
        return [f"{name}: no structure path"], []

    path = os.path.join(HERE, "structures", f"{m.group(1)}.cif")
    if not os.path.exists(path):
        return [f"{name}: {os.path.basename(path)} not downloaded"], []

    chains, resids = _polymer(path)
    problems: list[str] = []
    notes: list[str] = []
    blocks = CHAIN_BLOCK.findall(text[text.index("chains:"):])
    if not blocks:
        return [f"{name}: no chains parsed"], []

    for chain, crop, hotspots in blocks:
        here = resids[chains == chain]
        if here.size == 0:
            problems.append(
                f"{name}: chain {chain} absent (has {sorted(set(chains))})"
            )
            continue
        present = set(here.tolist())
        for span in re.findall(r'"(\d+)-(\d+)"', crop):
            lo, hi = int(span[0]), int(span[1])
            missing = [r for r in range(lo, hi + 1) if r not in present]
            if len(missing) == hi - lo + 1:
                problems.append(
                    f"{name}: chain {chain} crop {lo}-{hi} matches NOTHING "
                    f"(chain spans {here.min()}-{here.max()})"
                )
            elif missing:
                # Unresolved stretches are ordinary in a crystal structure --
                # 4hsa's chain A is missing 30-40, a disordered loop the other
                # monomer of the same homodimer resolves. Report, do not fail:
                # the error this script is for is a range that matches NOTHING,
                # which is what a renumbering mistake produces.
                runs, run = [], [missing[0]]
                for r in missing[1:]:
                    if r != run[-1] + 1:
                        runs.append(run)
                        run = []
                    run.append(r)
                runs.append(run)
                spans = ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0])
                                  for r in runs)
                notes.append(
                    f"{name}: chain {chain} crop {lo}-{hi} unresolved at {spans} "
                    f"({len(missing)} residues, {len(runs)} run(s))"
                )
        for h in re.findall(r"\d+", hotspots):
            if int(h) not in present:
                problems.append(
                    f"{name}: chain {chain} hotspot {h} not in "
                    f"{here.min()}-{here.max()}"
                )
    return problems, notes


def main() -> int:
    configs = sorted(glob.glob(os.path.join(HERE, "targets", "*.yaml")))
    if len(configs) != 10:
        print(f"expected 10 target configs, found {len(configs)}", file=sys.stderr)
    all_problems: list[str] = []
    all_notes: list[str] = []
    for cfg in configs:
        problems, notes = check(cfg)
        label = os.path.basename(cfg)[:-5]
        state = "PROBLEM" if problems else ("ok, see note" if notes else "ok")
        print(f"  {label:<8} {state}")
        all_problems += problems
        all_notes += notes
    if all_notes:
        print("\nunresolved regions (expected in crystal structures):")
        for n in all_notes:
            print(f"  {n}")
    if all_problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in all_problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"\nall {len(configs)} targets check out against their structures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
