#!/usr/bin/env python3
"""Full-atom structures: PXDesign backbone, FaMPNN side chains, sequence supplied.

    python scripts/design.py \
        --input-json <pxdesign_input.json> \
        --pxdesign-checkpoint-dir <dir with pxdesign_v0.1.0.pt> \
        --out designs/ --n-sample 8 --seed 0 \
        --sequence-fasta binder.fasta

The input JSON is PXDesign's own inference input, unchanged. FaMPNN packs side
chains onto a sequence that is *given*, so any residue PXDesign generates (its
`xpb` design tokens) needs an identity from --sequence or --sequence-fasta. If
the input has no design tokens the native sequence is used throughout and no
sequence argument is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

logger = logging.getLogger("pxf.design")


def read_fasta(path):
    """First record of a FASTA, whitespace stripped."""
    lines, sequence = Path(path).read_text().splitlines(), []
    for line in lines:
        if line.startswith(">"):
            if sequence:
                break
            continue
        sequence.append(line.strip())
    if not sequence:
        raise ValueError(f"No sequence found in {path}")
    return "".join(sequence).upper()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-json", required=True, help="PXDesign inference input")
    parser.add_argument(
        "--pxdesign-checkpoint-dir",
        required=True,
        help="directory containing <model-name>.pt",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="pxdesign_v0.1.0")
    sequence = parser.add_mutually_exclusive_group()
    sequence.add_argument("--sequence", help="full-length sequence to pack")
    sequence.add_argument("--sequence-fasta", help="FASTA holding that sequence")
    parser.add_argument("--n-sample", type=int, default=8, help="backbones per target")
    parser.add_argument("--n-step", type=int, default=200, help="diffusion steps")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32", "fp16"))
    parser.add_argument(
        "--weights",
        default="0.0",
        choices=("0.0", "0.3", "0.3-cath"),
        help="FaMPNN variant; 0.0 is what upstream uses for packing",
    )
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument(
        "--num-steps", type=int, default=None, help="FaMPNN diffusion steps"
    )
    parser.add_argument("--step-scale", type=float, default=None)
    parser.add_argument(
        "--keep-sidechain-context",
        action="store_true",
        help="keep the target's input rotamers; repack only design tokens",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="samples packed at once")
    parser.add_argument("--use-msa", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-download-cache", action="store_true")
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cuda:N | cpu; default probes the GPU and falls back to CPU",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from pxf.backbone.pxdesign import PXDesignBackbone
    from pxf.device import select_device
    from pxf.pipeline import FullAtomPipeline
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker

    strict = not args.allow_unpinned_sources
    supplied = args.sequence or (
        read_fasta(args.sequence_fasta) if args.sequence_fasta else None
    )

    logger.info("loading side-chain module (FaMPNN %s, packing mode)", args.weights)
    sidechain = FaMPNNSideChainPacker(
        args.fampnn_checkpoint,
        variant=args.weights,
        num_steps=args.num_steps,
        step_scale=args.step_scale,
        strict_sources=strict,
    )
    sidechain = sidechain.to(select_device(args.device))
    logger.info(
        "side-chain module on %s (designs_sequence=%s)",
        sidechain.device,
        sidechain.identity["designs_sequence"],
    )

    logger.info("loading backbone module (PXDesign %s)", args.model_name)
    backbone = PXDesignBackbone(
        input_json=args.input_json,
        checkpoint_dir=args.pxdesign_checkpoint_dir,
        dump_dir=out / "pxdesign",
        model_name=args.model_name,
        n_sample=args.n_sample,
        n_step=args.n_step,
        use_msa=args.use_msa,
        dtype=args.dtype,
        download_cache=not args.no_download_cache,
        strict_sources=strict,
    )
    logger.info("backbone module on %s; %d input target(s)", backbone.device, len(backbone))

    pipeline = FullAtomPipeline(
        backbone,
        sidechain,
        sequence=supplied,
        batch_size=args.batch_size,
        scn_context="keep_context" if args.keep_sidechain_context else "none",
    )

    structures, manifest, count = out / "structures", [], 0
    for batch in backbone.generate(seed=args.seed):
        logger.info(
            "%s: %d backbone sample(s), %d residues, %d design token(s)",
            batch.sample_name,
            batch.num_samples,
            batch.length,
            int(batch.design_mask.sum()),
        )
        for result in pipeline.run_batch(batch):
            path = pipeline.write_pdb(
                result, structures / f"{result.sample_name}_{result.sample_index:04d}.pdb"
            )
            manifest.append(
                dict(
                    sample_name=result.sample_name,
                    sample_index=result.sample_index,
                    pdb=str(path.relative_to(out)),
                    length=result.length,
                    sequence=result.sequence,
                    designed_sequence=result.designed_sequence(),
                    backbone_shift_angstrom=result.backbone_shift,
                    **result.metrics,
                )
            )
            count += 1
            logger.info(
                "  sample %d: mean psce %.3f -> %s",
                result.sample_index,
                result.metrics["mean_psce"],
                path.name,
            )

    (out / "manifest.json").write_text(
        json.dumps(
            dict(designs=manifest, provenance=pipeline.identity, arguments=vars(args)),
            indent=2,
            default=str,
        )
    )
    logger.info("wrote %d structure(s) to %s", count, out)
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
