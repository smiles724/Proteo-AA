#!/usr/bin/env python3
"""PXDesign backbone -> FaMPNN designs BOTH sequence and side chains.

This is the sequential baseline for the coupled architecture: PXDesign generates
a backbone, then FaMPNN's seq_design path (``SeqDenoiser.sample``) iteratively
unmasks residue identities and packs their side chains. Unlike
``scripts/design.py``, no sequence is supplied for the designed region -- FaMPNN
invents it.

Positions that already have an identity (a binder-design target) are held fixed,
so only PXDesign's ``xpb`` design tokens receive new residues.

    python scripts/design_fullatom.py \
        --input-json PXDesign/examples/PDL1_quick_start.yaml \
        --pxdesign-checkpoint-dir <dir with pxdesign_v0.1.0.pt> \
        --out runs/seqdes --n-sample 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.design_fullatom")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-json", required=True, help="PXDesign inference input (JSON or YAML)")
    p.add_argument("--pxdesign-checkpoint-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="pxdesign_v0.1.0")
    p.add_argument("--n-sample", type=int, default=8, help="backbones per target")
    p.add_argument("--n-step", type=int, default=200, help="backbone diffusion steps")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32", "fp16"))
    p.add_argument("--fampnn-weights", default="0.3", choices=("0.0", "0.3", "0.3-cath"),
                   help="0.3 is what upstream recommends for sequence design")
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--seq-steps", type=int, default=100, help="iterative unmasking steps")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--psce-threshold", type=float, default=0.3,
                   help="only condition on side chains better than this; <0 to keep all")
    p.add_argument("--no-repack-last", action="store_true")
    p.add_argument("--redesign-target", action="store_true",
                   help="also redesign positions that already have an identity")
    p.add_argument("--batch-size", type=int, default=None, help="designs per forward")
    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download-cache", action="store_true")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fampnn.model.sd_model import SeqDenoiser
    from pxf import atom37
    from pxf.backbone.pxdesign import PXDesignBackbone
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    out = Path(args.out).resolve()
    (out / "structures").mkdir(parents=True, exist_ok=True)
    strict = not args.allow_unpinned_sources

    logger.info("loading FaMPNN designer (variant %s, %d unmasking steps, T=%.2f)",
                args.fampnn_weights, args.seq_steps, args.temperature)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_weights, seq_steps=args.seq_steps,
        temperature=args.temperature,
        psce_threshold=None if args.psce_threshold < 0 else args.psce_threshold,
        repack_last=not args.no_repack_last, strict_sources=strict)
    designer = designer.to(select_device(args.device))
    logger.info("designer on %s", designer.device)

    logger.info("loading PXDesign backbone module (%s)", args.model_name)
    backbone = PXDesignBackbone(
        input_json=args.input_json, checkpoint_dir=args.pxdesign_checkpoint_dir,
        dump_dir=out / "pxdesign", model_name=args.model_name,
        n_sample=args.n_sample, n_step=args.n_step, use_msa=args.use_msa,
        dtype=args.dtype, download_cache=not args.no_download_cache,
        strict_sources=strict)
    logger.info("backbone on %s; %d input target(s)", backbone.device, len(backbone))

    manifest = []
    for batch in backbone.generate(seed=args.seed):
        known = batch.sequence_known.bool()
        design_tokens = batch.design_mask.bool()
        fixed = torch.zeros_like(known) if args.redesign_target else (known & ~design_tokens)
        aatype = atom37.aatype_from_sequence(batch.native_sequence, allow_unknown=True)
        logger.info("%s: %d samples, L=%d, %d design token(s), %d position(s) held fixed",
                    batch.sample_name, batch.num_samples, batch.length,
                    int(design_tokens.sum()), int(fixed.sum()))

        result = designer.design(
            coords_af2=batch.coords_af2, atom_mask=batch.atom_mask_af2,
            aatype=aatype.unsqueeze(0).expand(batch.num_samples, -1),
            fixed_sequence_mask=fixed.unsqueeze(0).expand(batch.num_samples, -1),
            residue_index=batch.residue_index.unsqueeze(0).expand(batch.num_samples, -1),
            chain_index=batch.chain_index.unsqueeze(0).expand(batch.num_samples, -1),
            seed=args.seed, batch_size=args.batch_size)

        for index, design in enumerate(result["designs"]):
            path = out / "structures" / f"{batch.sample_name}_{index:04d}.pdb"
            samples = {
                "x_denoised": design.coords_af2.unsqueeze(0).cpu(),
                "seq_mask": torch.ones(1, batch.length),
                "missing_atom_mask": torch.zeros(1, batch.length, atom37.NUM_ATOM37),
                "residue_index": batch.residue_index.unsqueeze(0).long(),
                "chain_index": batch.chain_index.unsqueeze(0).long(),
                "pred_aatype": design.aatype.unsqueeze(0).cpu(),
                "psce": design.psce.unsqueeze(0).cpu(),
            }
            SeqDenoiser.save_samples_to_pdb(samples, [str(path)])
            entry = dict(sample_name=batch.sample_name, sample_index=index,
                         pdb=str(path.relative_to(out)), length=batch.length,
                         sequence=design.sequence,
                         designed_sequence=design.designed_sequence(),
                         n_designed=int((~design.fixed_mask).sum()),
                         mean_psce=float(design.psce.mean()),
                         backbone_shift_angstrom=result["backbone_shift"])
            manifest.append(entry)
            logger.info("  sample %d: designed %d residues, mean psce %.3f -> %s",
                        index, entry["n_designed"], entry["mean_psce"], path.name)

    record = dict(designs=manifest,
                  provenance=dict(pipeline="pxdesign->fampnn(seq_design)",
                                  designs_sequence=True,
                                  backbone=backbone.identity,
                                  sidechain=designer.identity),
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()})
    (out / "manifest.json").write_text(json.dumps(record, indent=2, default=str))
    logger.info("wrote %d design(s) to %s", len(manifest), out)
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
