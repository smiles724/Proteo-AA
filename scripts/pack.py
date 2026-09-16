#!/usr/bin/env python3
"""Pack side chains onto existing backbones using each structure's own sequence.

This runs the side-chain module alone: the sequence is read from the input
structure and handed to FaMPNN, which infers side-chain conformations only. It
never designs a sequence.

    python scripts/pack.py --pdb-dir <dir of PDBs> --out packed/

By default the input's side chains are discarded and every residue is repacked
from backbone only. Use --keep-sidechain-context to instead condition on the
input rotamers where they exist.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.pack")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb-dir", help="directory of .pdb files to pack")
    source.add_argument("--pdb", nargs="+", help="explicit PDB paths")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--weights",
        default="0.0",
        choices=("0.0", "0.3", "0.3-cath"),
        help="FaMPNN variant; 0.0 is what upstream uses for packing",
    )
    parser.add_argument("--checkpoint", default=None, help="explicit checkpoint path")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--step-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--keep-sidechain-context",
        action="store_true",
        help="condition on the input side chains instead of repacking them",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cuda:N | cpu; default probes the GPU and falls back to CPU",
    )
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf import atom37
    from pxf.device import select_device
    from pxf.pipeline import FullAtomPipeline
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker

    if args.pdb_dir:
        from natsort import natsorted

        paths = [
            Path(p) for p in natsorted(str(p) for p in Path(args.pdb_dir).glob("*.pdb"))
        ]
    else:
        paths = [Path(p) for p in args.pdb]
    if not paths:
        raise SystemExit(f"no PDB files found in {args.pdb_dir}")

    out = Path(args.out).resolve()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    packer = FaMPNNSideChainPacker(
        args.checkpoint,
        variant=args.weights,
        num_steps=args.num_steps,
        step_scale=args.step_scale,
        strict_sources=not args.allow_unpinned_sources,
    )
    packer = packer.to(select_device(args.device))
    logger.info("FaMPNN %s loaded strictly on %s", packer.variant, packer.device)
    pipeline = FullAtomPipeline(None, packer)

    backbone_slots = list(atom37.BACKBONE_SLOTS)
    manifest = []
    for path in paths:
        single = process_single_pdb(load_feats_from_pdb(str(path)))
        length = single["x"].shape[0]
        sequence = atom37.sequence_from_aatype(single["aatype"])
        atom_mask = single["atom_mask"]
        if not args.keep_sidechain_context:
            kept = torch.zeros_like(atom_mask)
            kept[..., backbone_slots] = atom_mask[..., backbone_slots]
            atom_mask = kept
        coords = single["x"] * atom_mask[..., None]

        packed = packer(
            coords_af2=coords[None],
            aatype=single["aatype"][None],
            atom_mask=atom_mask[None],
            residue_index=single["residue_index"][None],
            chain_index=single["chain_index"][None],
            scn_context_mask=(
                torch.ones(1, length)
                if args.keep_sidechain_context
                else torch.zeros(1, length)
            ),
            batch_size=args.batch_size,
        )
        from pxf.pipeline import PackedDesign

        result = PackedDesign(
            sample_name=path.stem,
            sample_index=0,
            sequence=sequence,
            coords_af2=packed["coords_af2"][0].cpu(),
            atom_mask_af2=packed["atom_mask_af2"][0].cpu(),
            design_mask=torch.zeros(length, dtype=torch.bool),
            residue_index=single["residue_index"].long(),
            chain_index=single["chain_index"].long(),
            psce=packed["psce"][0].cpu(),
            backbone_shift=packed["backbone_shift"],
            metrics=dict(mean_psce=float(packed["psce"][0].mean())),
        )
        written = pipeline.write_pdb(result, out / "samples" / f"{path.stem}.pdb")
        manifest.append(
            dict(
                name=path.stem,
                source=str(path),
                length=length,
                sequence=sequence,
                pdb=str(written.relative_to(out)),
                mean_psce=result.metrics["mean_psce"],
                backbone_shift_angstrom=result.backbone_shift,
            )
        )
        logger.info(
            "%s: L=%d mean psce %.3f backbone shift %.4f A -> %s",
            path.stem,
            length,
            result.metrics["mean_psce"],
            result.backbone_shift,
            written.name,
        )

    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                packed=manifest,
                provenance=dict(sidechain=packer.identity),
                arguments=vars(args),
            ),
            indent=2,
            default=str,
        )
    )
    logger.info("packed %d structure(s) into %s", len(manifest), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
