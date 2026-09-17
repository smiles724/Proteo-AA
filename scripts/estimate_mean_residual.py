#!/usr/bin/env python
"""Estimate ``mu(sigma)``: the adapter's average residual, over training proteins.

The mean arm asks whether a *shared* residual reproduces what the matched one
does. Answering that needs a mean estimated somewhere the evaluation cannot see:

    mu(sigma) = E_{training proteins} [ mean over valid residues of
                                        A_BS(a_i, sigma) ]

Equal weight per protein, not per residue, so a 500-residue structure does not
outvote a 50-residue one in a quantity that is meant to describe the adapter
rather than the panel's length distribution.

**Estimated from the training manifest only.** Taking the mean from CASP or the
AFDB val split would let the control see the set it is scored on, which is
precisely the confound this arm exists to rule out.

    python scripts/estimate_mean_residual.py \\
        --checkpoint runs/phase1/checkpoints/final.pt \\
        --structures configs/phase1_structures_afdb.txt \\
        --pxdesign-donor .../pxdesign_v0.1.0.pt \\
        --out configs/mean_residual_phase1.json

FaMPNN is deliberately not loaded: ``a_token`` comes from PXDesign, and the
adapter's output width is recoverable from its own checkpoint, so the side-chain
model has no role in this estimate and loading it would only cost memory.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import schedule
from pxf.eval import couple as ev

logger = logging.getLogger("pxf.mean_residual")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="the coupling checkpoint")
    p.add_argument(
        "--structures",
        required=True,
        help="TRAINING manifest -- never an evaluation panel",
    )
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/couple_phase1.yaml")
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--n-sigma", type=int, default=5)
    p.add_argument(
        "--max-proteins",
        type=int,
        default=256,
        help="subsample of the training panel; a mean converges long before the "
        "full 2,000 and the count is recorded either way",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--ema", dest="ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import sys

    import yaml

    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple.adapters import CouplingAdapters
    from pxf.device import select_device
    from pxf.provenance import file_sha256

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sigmas = ev.sweep_sigmas(schedule.from_config(config.get("sigma")), args.n_sigma)

    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_proteins and len(structures) > args.max_proteins:
        # Strided rather than random: reproducible without carrying a seed, and
        # the manifest is already strided across the length distribution.
        step = len(structures) / args.max_proteins
        structures = [structures[int(i * step)] for i in range(args.max_proteins)]
    logger.info(
        "%d training proteins, sigmas %s", len(structures), [round(s, 3) for s in sigmas]
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{args.checkpoint} is not a coupling checkpoint")
    weights = state["adapters"]
    # Widths come from the checkpoint itself, so FaMPNN never has to be loaded.
    d_backbone = int(weights["bb_to_sc.norm.weight"].shape[0])
    d_fampnn = int(weights["bb_to_sc.project_out.weight"].shape[0])
    adapter_cfg = dict(config.get("adapters", {}))
    adapters = CouplingAdapters(d_backbone, d_fampnn, **adapter_cfg)
    adapters.load_state_dict(weights)
    if args.ema and state.get("ema"):
        from pxf.train.ema import EMA

        settings = state.get("settings") or {}
        ema = EMA(
            adapters,
            decay=settings.get("ema_decay"),
            relative_length=(
                None
                if settings.get("ema_decay") is not None
                else settings.get("ema_relative_length") or 0.25
            ),
        )
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
        logger.info("using the EMA weights")
    device = select_device(args.device)
    adapters.eval().requires_grad_(False)
    adapters.to(device)

    px_model, _cfg, _rec = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    featurized = featurize_structures(
        structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )

    # Per-protein means, accumulated in float64: summing a few hundred vectors of
    # a few hundred residues each in float32 loses digits that matter for a mean.
    totals = {i: torch.zeros(d_fampnn, dtype=torch.float64) for i in range(len(sigmas))}
    counted, skipped = 0, []
    started = time.time()

    for index, (sample_id, source) in enumerate(featurized):
        try:
            structure = to_featurized(sample_id, source[0]).to(device)
        except (ValueError, KeyError, IndexError) as error:
            skipped.append(dict(target=sample_id, reason=str(error)[:200]))
            continue
        target = structure.backbone_target.float()
        conditioning = px_driver.conditioning(structure.feature_dict)
        denoise = px_driver.bind(conditioning)

        for si, sigma_value in enumerate(sigmas):
            seed = ev.target_seed(args.seed, sample_id, sigma_value)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(target.shape, generator=generator).to(device)
            x_noisy = (target + noise * float(sigma_value))[None]
            sigma = torch.full((1,), float(sigma_value), device=device)
            with torch.no_grad():
                _x0, a_token = denoise(x_noisy, sigma)
                delta = adapters.delta_h(a_token, sigma)
            # Mean over residues first: one vector per protein, so every protein
            # weighs the same regardless of length.
            totals[si] += delta.reshape(-1, d_fampnn).mean(0).double().cpu()
        counted += 1
        if index % 25 == 0 or index == len(featurized) - 1:
            logger.info(
                "%d/%d proteins, %.1fs", index + 1, len(featurized), time.time() - started
            )

    if not counted:
        raise SystemExit(f"nothing was featurized; {len(skipped)} skipped")

    vectors = [(totals[i] / counted).tolist() for i in range(len(sigmas))]
    blob = {
        "sigmas": [float(s) for s in sigmas],
        "vectors": vectors,
        "provenance": {
            "n_proteins": counted,
            "n_requested": len(structures),
            "n_skipped": len(skipped),
            "weighting": "equal per protein, mean over residues within a protein",
            "structures": str(args.structures),
            "structures_fingerprint": ev.structures_fingerprint(structures),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "ema": bool(args.ema and state.get("ema")),
            "seed_scheme": ev.SEED_SCHEME,
            "seed_base": int(args.seed),
            "crop_size": int(args.crop_size),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, indent=2))
    logger.info("wrote %s", args.out)

    norms = [float(torch.tensor(v).norm()) for v in vectors]
    print(f"\nmu(sigma) over {counted} training proteins, {args.checkpoint}")
    print(f"  {'sigma':>8} {'||mu||':>10}")
    for sigma_value, norm in zip(sigmas, norms):
        print(f"  {sigma_value:8.3f} {norm:10.4f}")
    if len(skipped) > 0:
        print(f"  ({len(skipped)} skipped, e.g. {skipped[:2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
