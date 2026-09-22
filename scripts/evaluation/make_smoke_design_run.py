#!/usr/bin/env python3
"""Fabricate a generation run so the AF2-IG half can be tested without a checkpoint.

`fold_af2ig.py` consumes what `eval_conditional_binder_benchmark.py` produces:
`designs.csv`, `designs/<sample_id>.pdb`, and `inputs/<task_id>.prep.json`.
Producing those for real needs a trained Stage III checkpoint and a GPU-hour per
handful of designs, which means the scoring half would otherwise be unverifiable
until the generation half is finished — the wrong order, since scoring is the
part with the external dependency (AlphaFold parameters, a second environment, a
JAX build) and therefore the part most likely to be broken on arrival.

This writes a run directory of the right *shape* with designs that are
**deliberately meaningless**: the real cropped target from the deposited mmCIF,
plus the same inert ideal α-helix `target_prep` uses as a placeholder during
input preparation, carrying an arbitrary heptad-patterned sequence. Nothing
here is a design. Every metric it produces is noise.

What it does verify, which is the whole point:

  * the binder chain resolves through the `.prep.json` sidecar;
  * ColabDesign parses a multi-chain design PDB and splits it where this
    harness thinks it does;
  * both AlphaFold passes run and their logs contain the four metrics;
  * ProteinMPNN redesigns the binder and leaves the target alone;
  * the metrics CSV the folder writes is the CSV the scorer reads.

One caveat specific to the PMPNN arm. `make_placeholder_binder` builds its
helix from an ideal CA trace, and the N/C/O it hangs off that trace are not
peptide-bonded: measured on this fixture, C(i)-N(i+1) is **0.81 A** against a
real 1.33 A. That is harmless for its actual job -- the placeholder's
coordinates never reach the model during input preparation -- but ProteinMPNN
reads backbone geometry and nothing else, so on this fixture it returns
poly-serine junk. That is the fixture, not the harness: real designs come out
of the diffusion model with real peptide geometry. Read the PMPNN arm here as
"it ran and the target stayed fixed", never as "the sequences look reasonable".

    python scripts/evaluation/make_smoke_design_run.py \
        --out /tmp/cbdb_smoke --mmcif-dir <mmcif> --targets PDL1 --n-samples 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Top-level imports, for the reason given in fold_af2ig.py: the AF2-IG
# environment has no Protenix, so `pxdesign_train/__init__.py` cannot run.
# Both of these modules are numpy + biotite only at import time.
BENCHMARKS_DIR = REPO_ROOT / "pxdesign_train" / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from conditional_binder import ConditionalBinderDesignBenchmark  # noqa: E402
from target_prep import (  # noqa: E402
    BACKBONE_ATOMS,
    choose_binder_chain_id,
    load_cropped_target,
    make_placeholder_binder,
    resolve_hotspots,
)

# An arbitrary amphipathic heptad. Not a design and not claimed to be one; it
# exists so AF2 has a sequence that at least prefers a helix, which makes the
# smoke output legible instead of uniformly zero.
HEPTAD = "LEKKLAE"

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def _sequence(length: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    base = (HEPTAD * (length // len(HEPTAD) + 1))[:length]
    # Perturb a tenth of the positions so the samples of one cell are not
    # byte-identical -- the unbound fold is cached by sequence, and a run where
    # every sequence is the same would exercise the cache instead of the model.
    out = list(base)
    for index in rng.choice(length, size=max(1, length // 10), replace=False):
        out[int(index)] = str(rng.choice(list("AVLIFKED")))
    return "".join(out)


def _structure_path(mmcif_dir: Path, pdb_id: str) -> Path:
    for name in (f"{pdb_id.lower()}.cif", f"{pdb_id.upper()}.cif",
                 f"{pdb_id.lower()}.cif.gz"):
        candidate = mmcif_dir / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no structure for {pdb_id} under {mmcif_dir}")


def _write_design(path: Path, target, binder, sequence: str) -> None:
    """Target first, binder last -- the chain order the folder assumes."""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    array = target + binder
    n_target = target.array_length()
    for i in range(n_target, array.array_length()):
        residue_index = (i - n_target) // len(BACKBONE_ATOMS)
        array.res_name[i] = AA3[sequence[residue_index]]
    pdb = PDBFile()
    pdb.set_structure(array)
    pdb.write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--mmcif-dir", required=True)
    parser.add_argument("--targets", nargs="*", default=["PDL1"])
    parser.add_argument("--lengths", nargs="*", type=int, default=[80])
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    mmcif_dir = Path(args.mmcif_dir).expanduser().resolve()
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "designs").mkdir(parents=True, exist_ok=True)

    benchmark = ConditionalBinderDesignBenchmark.load()
    by_name = {t.name: t for t in benchmark.targets}
    unknown = [name for name in args.targets if name not in by_name]
    if unknown:
        raise SystemExit(f"not in the benchmark: {unknown}")

    rows = []
    for target_name in args.targets:
        target = by_name[target_name]
        target.require_runnable()
        cropped = load_cropped_target(
            _structure_path(mmcif_dir, target.pdb_id), target.crop_ranges()
        )
        hotspots = resolve_hotspots(cropped, target.hotspots or ())
        binder_chain = choose_binder_chain_id(set(cropped.chain_id.tolist()))
        anchor = None
        if hotspots:
            mask = np.zeros(cropped.array_length(), dtype=bool)
            for chain_id, res_id in hotspots:
                mask |= (cropped.chain_id == chain_id) & (cropped.res_id == res_id)
            anchor = cropped.coord[mask].mean(axis=0)

        for length in args.lengths:
            task_id = f"{target_name}_L{length}"
            (out / "inputs" / f"{task_id}.prep.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "target_name": target_name,
                        "binder_chain_id": binder_chain,
                        "binder_length": length,
                        "hotspots_author": [list(h) for h in hotspots],
                        "smoke": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            binder = make_placeholder_binder(cropped, length, binder_chain, anchor=anchor)
            for i in range(args.n_samples):
                sample_id = f"{task_id}_s{i:04d}"
                sequence = _sequence(length, args.seed + i)
                pdb_path = out / "designs" / f"{sample_id}.pdb"
                _write_design(pdb_path, cropped, binder, sequence)
                (out / "designs" / f"{sample_id}.fasta").write_text(
                    f">{sample_id} target={target_name} length={length}\n{sequence}\n"
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target": target_name,
                        "pdb_id": (target.pdb_id or "").upper(),
                        "binder_length": length,
                        "seed": args.seed + i,
                        "variant": "co_design",
                        "sequence": sequence,
                        "design_pdb": str(pdb_path),
                        "n_hotspots": len(hotspots),
                        "has_full_atom_sidechain": False,
                        "n_sidechain_atoms": 0,
                        "seconds": 0.0,
                    }
                )

    designs_csv = out / "designs.csv"
    with designs_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} SMOKE designs (not designs) -> {out}")
    print("these exercise the folding path; every metric they produce is noise")
    print(f"\n  python scripts/evaluation/fold_af2ig.py --run-dir {out} --data-dir <af2 params>")


if __name__ == "__main__":
    main()
