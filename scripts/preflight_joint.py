#!/usr/bin/env python3
"""Preflight for side-chain-supervised backbone fine-tuning.

Two of the gates the plan puts before any training job, run together because
they need the same expensive setup:

**Memory and timing on the real donor**, at the short and long ends of the
pilot's length range. "It fits" is not a default to assume from a CPU test.

**The auxiliary loss coefficients.** ``L_BB``, ``L_local`` and ``L_place`` are
normalized differently and are not comparable by their values, so each
auxiliary term's coefficient is set from the size of its *gradient at the
backbone*: lambda such that the median gradient norm is ``--ratio`` times the
anchor's. Sampled across backbone noise bands, because a median over one band
can hide a term that dominates at high noise and vanishes at low.

This runs no optimizer and writes no checkpoint. It reports what a training run
would need to be configured with, and fails loudly on a dead or non-finite
gradient rather than producing a coefficient that hides one.

    python scripts/preflight_joint.py --out runs/preflight --n-structures 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

DONOR = (
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt"
)
STRUCTURES = "/hai/scratch/yfsun/afdb_laproteina/cif_phase1"
# The plan's initial refinement window, in Angstroms.
SIGMA_BANDS = ((0.1, 0.3), (0.3, 0.7), (0.7, 1.3), (1.3, 2.0))
TRAINABLE_BLOCKS = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--structures", default=STRUCTURES, help="directory of training CIFs")
    p.add_argument("--donor", default=DONOR)
    p.add_argument("--fampnn-weights", default="0.0")
    p.add_argument("--n-structures", type=int, default=4)
    p.add_argument("--min-length", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--multiplier", type=int, default=8, help="side-chain noise clones")
    p.add_argument("--ratio", type=float, default=0.1, help="target gradient ratio")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple.losses import backbone_denoising_loss
    from pxf.couple.schedule import CouplingNoiseSchedule
    from pxf.device import select_device
    from pxf.joint import data as joint_data
    from pxf.joint import losses as joint_losses
    from pxf.joint import model as joint_model
    from pxf.joint import randomness as joint_random
    from pxf.joint.trainer import select_trainable
    from pxf.provenance import fampnn_checkpoint
    from pxf.train import losses as loss_fns

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    torch.manual_seed(args.seed)

    backbone, _bundle, donor_record = load_backbone_model(args.donor, device=device)
    driver = PXDesignBackboneDriver(backbone)
    # The same allowlist the trainer resolves, so the coefficients are
    # calibrated against the parameter set that will actually be trained.
    trainable = select_trainable(backbone, n_blocks=TRAINABLE_BLOCKS)
    n_trainable = sum(p.numel() for p in trainable.values())

    weights = torch.load(
        fampnn_checkpoint(args.fampnn_weights), map_location="cpu", weights_only=False
    )
    fampnn = SeqDenoiser(weights["model_cfg"])
    fampnn.load_state_dict(weights["state_dict"], strict=True)
    fampnn.to(device).eval()
    fampnn.requires_grad_(False)
    interpolant = fampnn.denoiser.scn_diffusion_module.scn_interpolant
    augment_eps = float(fampnn.denoiser.seq_design_module.features.augment_eps)

    paths = sorted(Path(args.structures).glob("*.cif"))
    if not paths:
        raise SystemExit(f"no CIFs under {args.structures}")

    schedule = CouplingNoiseSchedule(
        mode="trajectory", sigma_min=0.1, sigma_max=2.0, sigma_data=driver.sigma_data
    )
    sidechain_slots = None

    records, skipped = [], []
    for path in paths:
        if len(records) >= args.n_structures:
            break
        try:
            sample_id, source = featurize_structures([str(path)], crop_size=1024)[0]
            structure = to_featurized(sample_id, source[0]).to(device)
            if not args.min_length <= structure.num_tokens <= args.max_length:
                skipped.append(dict(sample_id=sample_id, reason=f"length {structure.num_tokens}"))
                continue
            native = process_single_pdb(load_feats_from_pdb(str(path)))
            batch = joint_data.build_joint_batch(fampnn, structure, native)
        except Exception as error:  # a rejected example is data, not a crash
            skipped.append(dict(sample_id=path.stem, reason=f"{type(error).__name__}: {error}"))
            continue

        if sidechain_slots is None:
            from fampnn.data import residue_constants as rc

            sidechain_slots = list(rc.non_bb_idxs)

        conditioning = driver.conditioning(structure.feature_dict)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bands = []
        for index, (low, high) in enumerate(SIGMA_BANDS):
            sigma = float(
                schedule.sample(
                    1, generator=torch.Generator().manual_seed(args.seed + index)
                ).clamp(low, high)
            )
            backbone_noise = joint_random.draw_backbone_noise(
                batch.backbone_target.shape,
                joint_random.generator_for(args.seed, sample_id, "backbone_noise", occurrence=index),
            )
            sidechain_noise = joint_random.sidechain_noise_for(
                interpolant,
                (args.multiplier, batch.length, len(sidechain_slots), 3),
                base=args.seed,
                sample_id=sample_id,
                occurrence=index,
                device=device,
            )
            started = time.time()
            forward = joint_model.joint_forward(
                driver,
                conditioning,
                fampnn,
                batch,
                sigma_b=sigma,
                backbone_noise=backbone_noise,
                sidechain_noise=sidechain_noise,
                multiplier=args.multiplier,
                self_cond_p=0.0,
            )
            forward_seconds = time.time() - started

            anchor = backbone_denoising_loss(
                forward.bb_pred,
                forward.bb_target,
                sigma=forward.sigma_b,
                sigma_data=driver.sigma_data,
                atom_mask=forward.bb_mask,
            ).total
            local, local_stats = loss_fns.sidechain_diffusion_loss(
                forward.prediction.q_pred,
                forward.prediction.q_target,
                forward.prediction.weight,
                forward.prediction.loss_mask,
            )
            native_scn = forward.prediction.clone(
                batch.native_batch["x"][..., sidechain_slots, :]
            )
            physical = forward.prediction.clone(batch.physical_mask)
            placement, place_stats = joint_losses.placement_loss(
                forward.placed, native_scn, physical, aatype=forward.prediction.aatype
            )
            frame_only, _frame_stats = joint_losses.frame_only_placement(forward, batch)

            # The calibration quantity: each term's gradient AT THE BACKBONE.
            norms = {}
            for name, term in (
                ("L_BB", anchor),
                ("L_local", local),
                ("L_place", placement),
                ("L_frame", frame_only),
            ):
                started = time.time()
                norms[name] = joint_losses.gradient_norm(term, forward.bb_pred)
                norms[f"{name}_grad_seconds"] = time.time() - started

            bands.append(
                dict(
                    band=[low, high],
                    sigma_b=sigma,
                    forward_seconds=round(forward_seconds, 3),
                    loss_bb=float(anchor),
                    loss_local=float(local),
                    loss_place=float(placement),
                    loss_frame=float(frame_only),
                    sidechain_mse_local=float(local_stats["sidechain_mse_local"]),
                    placement_rmsd=float(place_stats["placement_rmsd"]),
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in norms.items()},
                )
            )

        records.append(
            dict(
                sample_id=sample_id,
                length=batch.length,
                alignment_rmsd=batch.alignment_rmsd,
                **batch.counts,
                bands=bands,
                peak_memory_gb=(
                    torch.cuda.max_memory_allocated(device) / 1e9
                    if device.type == "cuda"
                    else None
                ),
            )
        )
        print(
            f"{sample_id}: L={batch.length} "
            f"physical={batch.counts['physical_atoms']:.0f} "
            f"ghost={batch.counts['ghost_fraction']:.2f} "
            f"forward={statistics.mean(b['forward_seconds'] for b in bands):.2f}s",
            flush=True,
        )

    if not records:
        raise SystemExit(f"no usable structure in {args.structures}; skipped {skipped[:4]}")

    def ratios(term):
        return [
            band[term] / band["L_BB"]
            for record in records
            for band in record["bands"]
            if band["L_BB"] > 0
        ]

    coefficients = {}
    for term in ("L_local", "L_place", "L_frame"):
        values = [
            band[term] for record in records for band in record["bands"]
        ]
        anchors = [band["L_BB"] for record in records for band in record["bands"]]
        median_aux = statistics.median(values)
        median_anchor = statistics.median(anchors)
        coefficients[term] = dict(
            median_gradient_norm=median_aux,
            median_anchor_norm=median_anchor,
            # calibrate() refuses a dead or non-finite gradient rather than
            # returning an enormous coefficient for a term that trains nothing.
            coefficient=joint_losses.calibrate(median_anchor, median_aux, ratio=args.ratio),
            per_band_ratio_to_anchor=[round(r, 6) for r in ratios(term)],
        )

    report = dict(
        device=str(device),
        donor=donor_record,
        fampnn_weights=args.fampnn_weights,
        augment_eps=augment_eps,
        trainable_parameters=int(n_trainable),
        trainable_modules=sorted({n.rsplit(".", 1)[0] for n in trainable}),
        multiplier=args.multiplier,
        target_ratio=args.ratio,
        sigma_schedule=schedule.identity(),
        randomness=joint_random.identity(base=args.seed),
        structures=records,
        skipped=skipped[:32],
        coefficients=coefficients,
        peak_memory_gb=max(
            (r["peak_memory_gb"] for r in records if r["peak_memory_gb"]), default=None
        ),
    )
    (out / "preflight.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(coefficients, indent=2))
    print(f"peak memory: {report['peak_memory_gb']} GB   -> {out / 'preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
