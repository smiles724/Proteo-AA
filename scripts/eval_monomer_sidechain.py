#!/usr/bin/env python3
"""Side-chain packing inference on monomers, scored with the canonical metrics.

THE TASK, as side-chain-packing papers define it: give the model the
experimentally determined backbone and the true amino-acid sequence, ask it to
reconstruct the side-chain atoms, and score them. That is exactly what this
pipeline's side-chain module does -- FaMPNN in packing mode, sequence supplied,
nothing designed -- so a monomer benchmark measures it directly.

    python scripts/eval_monomer_sidechain.py \
        --pdb-dir fampnn/data/casp14/pdbs --label casp14 --out runs/casp14

Metrics come from Proteo-AA's own `pxdesign_train.sidechain` implementations
(`packing_metrics`, `summarize_metrics`, `sidechain_lddt`) rather than being
reimplemented, so they are comparable to the earlier Proteo-AA runs. Dataset
figures sum counts across targets and divide once, which makes them atom-weighted
rather than an average of per-target averages.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.eval")

# Reported per target and in the dataset summary, grouped as requested.
REPORT = {
    "lddt": ("lddt_sc_sc", "lddt_sc_env"),
    "rmsd": ("symmetry_rmsd",),
    "angle correctness": (
        "chi_recovery_20deg",
        "chi_recovery_40deg",
        "chi1_accuracy_20deg",
        "chi1_chi2_accuracy_20deg",
        "rotamer_recovery",
    ),
    "covalent failures": (
        "bad_bond_fraction",
        "bond_mae",
        "rotamer_outlier_fraction_40deg",
        "completeness",
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdb-dir", required=True, help="directory of monomer PDBs")
    parser.add_argument(
        "--pdb-key-list", default=None, help="optional file of stems to restrict to"
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--label", default=None, help="dataset label for the report")
    parser.add_argument("--weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--num-steps", type=int, default=None, help="FaMPNN side-chain diffusion steps"
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="packings per target; packing is a sampler"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--metrics-root",
        default=None,
        help="Proteo-AA checkout supplying the metric implementations",
    )
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb
    from natsort import natsorted

    from pxf import atom37
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics
    from pxf.eval.sidechain_metrics import aggregate, score
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker

    pdb_dir = Path(args.pdb_dir).resolve()
    if args.pdb_key_list:
        stems = [
            s.strip() for s in Path(args.pdb_key_list).read_text().split() if s.strip()
        ]
        paths = [pdb_dir / (s if s.endswith(".pdb") else f"{s}.pdb") for s in stems]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"missing PDBs from the key list: {missing[:6]}")
    else:
        paths = [Path(p) for p in natsorted(str(p) for p in pdb_dir.glob("*.pdb"))]
    if args.max_targets:
        paths = paths[: args.max_targets]
    if not paths:
        raise SystemExit(f"no PDB files found in {pdb_dir}")
    label = args.label or pdb_dir.parent.name

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    canonical = load_metrics(args.metrics_root)
    logger.info(
        "metrics from %s @ %s (not reimplemented)", canonical.root, canonical.revision[:10]
    )

    packer = FaMPNNSideChainPacker(
        args.checkpoint,
        variant=args.weights,
        num_steps=args.num_steps,
        strict_sources=not args.allow_unpinned_sources,
    )
    packer = packer.to(select_device(args.device))
    logger.info(
        "FaMPNN %s on %s; %d monomer target(s) from %s",
        packer.variant,
        packer.device,
        len(paths),
        label,
    )

    backbone = list(atom37.BACKBONE_SLOTS)
    per_target_counts, rows, skipped = [], [], []
    started = time.time()
    for index, path in enumerate(paths):
        single = process_single_pdb(load_feats_from_pdb(str(path)))
        native, native_mask = single["x"], single["atom_mask"]
        aatype = single["aatype"].long()
        length = native.shape[0]
        # The metrics are defined over the canonical twenty; anything else
        # (X/UNK from an unresolved residue) is excluded rather than guessed.
        canonical_res = aatype < 20
        if not bool(canonical_res.any()):
            skipped.append(dict(target=path.stem, reason="no canonical residues"))
            continue

        # Backbone-only input with the native sequence: the packing task.
        given = torch.zeros_like(native_mask)
        given[:, backbone] = native_mask[:, backbone]
        coords = native * given[..., None]

        for sample in range(args.samples):
            seed = args.seed + index * 1000 + sample
            packed = packer(
                coords_af2=coords[None],
                aatype=aatype[None],
                atom_mask=given[None],
                residue_index=single["residue_index"][None],
                chain_index=single["chain_index"][None],
                seed=seed,
            )
            counts, summary = score(
                packed["coords_af2"][0].cpu(),
                packed["atom_mask_af2"][0].cpu(),
                native,
                native_mask,
                aatype,
                canonical=canonical,
                residue_mask=canonical_res,
            )
            per_target_counts.append(counts)
            rows.append(
                dict(
                    target=path.stem,
                    sample=sample,
                    length=length,
                    scored_residues=summary["scored_residues"],
                    backbone_shift_angstrom=packed["backbone_shift"],
                    **{k: summary[k] for group in REPORT.values() for k in group},
                )
            )
        last = rows[-1]
        logger.info(
            "%-12s L=%-4d rmsd=%.3f lddt_sc=%.3f lddt_env=%.3f chi20=%.3f bad_bond=%.4f",
            path.stem,
            length,
            last["symmetry_rmsd"],
            last["lddt_sc_sc"],
            last["lddt_sc_env"],
            last["chi_recovery_20deg"],
            last["bad_bond_fraction"],
        )

    if not per_target_counts:
        raise SystemExit("every target was skipped")
    summary = aggregate(per_target_counts, canonical=canonical)
    elapsed = time.time() - started

    record = dict(
        label=label,
        dataset=str(pdb_dir),
        n_targets=len(paths),
        n_scored=len(rows),
        samples_per_target=args.samples,
        seed=args.seed,
        skipped=skipped,
        elapsed_seconds=round(elapsed, 1),
        summary=summary,
        per_target=rows,
        provenance=dict(
            sidechain=packer.identity,
            metrics=canonical.record(),
            task="backbone + native sequence -> side chains, no design",
        ),
        arguments=vars(args),
    )
    (out / "sidechain_metrics.json").write_text(json.dumps(record, indent=2, default=str))

    with (out / "per_target.csv").open("w") as stream:
        columns = list(rows[0])
        stream.write(",".join(columns) + "\n")
        for row in rows:
            stream.write(",".join(str(row[c]) for c in columns) + "\n")

    print(
        f"\n=== side-chain metrics: {label} "
        f"({len(rows)} packings over {len(paths)} monomers, atom-weighted) ==="
    )
    for group, keys in REPORT.items():
        print(f"\n  {group}")
        for key in keys:
            print(f"    {key:34s} {summary[key]:.4f}")
    print(
        f"\n  pairs: sc-sc {summary['n_pairs_sc_sc']:.0f}, "
        f"sc-env {summary['n_pairs_sc_env']:.0f}; "
        f"chi {float(summary['chi_count']):.0f}; "
        f"scored side-chain atoms {float(summary['observed_atoms']):.0f}"
    )
    print(f"  wrote {out}/sidechain_metrics.json  ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
