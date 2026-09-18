#!/usr/bin/env python
"""One-event SC -> BB feedback on target-conditioned PXDesign generations.

A generated backbone has no native counterpart, so RMSD to the original partner
is **not** a correctness metric here and is not computed. What is measured is
self-consistency, chemistry and target compatibility.

Per (target, generation seed):

1. Generate the partner backbone with PXDesign conditioned on the fixed target,
   recording sampler states at chosen solver steps
   (:mod:`pxf.couple.replay`). The native partner's coordinates and sequence are
   never supplied.
2. At each recorded event, build **one** shared preparation: the no-feedback
   clean backbone estimate, the generated chain extracted from it, a FaMPNN
   sequence design on that chain alone, that sequence frozen, a packing under it,
   and a fresh full-atom encode. Cached by ``(target, seed, event)``.
3. Replay the event three ways -- baseline / bb_only / full -- from the *same*
   saved state, with the same conditioning and the same later stochastic draws.
   One injection each, then the trajectory completes with no further feedback.
4. Repack each arm's final backbone under the shared sequence and paired packing
   randomness, and score.

**The preparation must not consume a solver step.** Its clean-estimate forward
pass is a separate evaluation of the denoiser at the saved state; the solver is
advanced exactly once per step in every arm, and the injection counter asserts
it.

**The arms share the sequence, so they share its ESMFold prediction.** pLDDT is
therefore a property of the sequence and cannot be an arm-specific improvement
metric; refolding is emitted as one FASTA entry per unique sequence for the
separate refolding job, and the consistency metric is each arm's own backbone
against that shared prediction.
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.gen_stress")

MANIFEST = "gen_stress_manifest.csv"
ARMS = ("baseline", "bb_only", "full")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prepared", required=True, help="from prepare_dimer_targets.py")
    p.add_argument("--out", required=True)
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="full=... and bb_only=...; the baseline needs none",
    )
    p.add_argument("--config", default="configs/couple_phase2_pilot.yaml")
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--pack-steps", type=int, default=50)
    p.add_argument("--n-step", type=int, default=200, help="sampler steps")
    p.add_argument(
        "--events",
        type=float,
        nargs="+",
        default=[0.6, 0.85],
        help="event positions as fractions of the schedule; resolved to actual "
        "solver steps and reported as (step, substage) keys, never as sigma "
        "thresholds",
    )
    p.add_argument("--max-targets", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--pool", default=None, choices=("pdb", "ted"))
    p.add_argument("--seq-temperature", type=float, default=0.1)
    p.add_argument("--device", default=None)
    p.add_argument("--ema", dest="ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument(
        "--scrambled",
        action="store_true",
        help="add a fourth arm: the full adapter reading rotamer-perturbed side "
        "chains through the same interface",
    )
    return p.parse_args(argv)


def sequence_hash(aatype):
    from pxf import atom37

    text = atom37.sequence_from_aatype(aatype.reshape(-1).cpu())
    return text, hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd
    import yaml
    from fampnn.model.sd_model import SeqDenoiser

    from pxf import provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple import mapping, replay, torsions
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.schedule import karras_sigmas
    from pxf.device import select_device
    from pxf.eval import gen_metrics

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_sb_feedback import load_arm

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sb_cfg = dict(config.get("sb_feedback", {}))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "structures").mkdir(exist_ok=True)

    frame = pd.read_parquet(args.prepared)
    if args.pool:
        frame = frame[frame.pool == args.pool]
    frame = frame.head(args.max_targets)
    logger.info("%d target(s) x %d seed(s)", len(frame), len(args.seeds))

    device = select_device(args.device)
    checkpoint = (
        Path(args.fampnn_checkpoint)
        if args.fampnn_checkpoint
        else provenance.fampnn_checkpoint(args.fampnn_weights)
    )
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval().requires_grad_(False)
    fampnn.to(device)
    c_h_V = node_feature_dim(fampnn)

    px_model, _cfgs, px_record = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    trained = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        trained[label] = load_arm(
            path,
            c_h_V=c_h_V,
            c_token=px_driver.c_token,
            sb_cfg=sb_cfg,
            use_ema=args.ema,
            device=device,
        )
        logger.info(
            "arm %s: variant=%s step=%d",
            label,
            trained[label]["variant"],
            trained[label]["step"],
        )
    adapters = CouplingAdapters(px_driver.c_token, c_h_V).to(device)
    adapters.eval().requires_grad_(False)
    controller = CoupledDenoiser(
        backbone=None,
        fampnn=fampnn,
        adapters=adapters,
        phase="sc_to_bb",
        pack_steps=args.pack_steps,
    )
    schedule = karras_sigmas(args.n_step).to(torch.float32)
    arms = list(ARMS) + (["scrambled"] if args.scrambled else [])

    rows, sequences, skipped = [], {}, []
    started = time.time()

    for entry in frame.itertuples():
        try:
            sid, dataset = featurize_structures(
                [entry.cif_path],
                crop_size=args.crop_size,
                binder_chain_ids=[entry.converted_binder_chain],
                parser_dataset="Distillation",
                proteoaa_root=args.proteoaa_root,
            )[0]
            structure = to_featurized(sid, dataset[0]).to(device)
        except Exception as error:  # noqa: BLE001
            skipped.append(dict(target=entry.example_id, reason=str(error)[:200]))
            logger.warning("skipping %s: %s", entry.example_id, str(error)[:160])
            continue

        design = structure.design_mask.reshape(-1).bool()
        n_atom = len(structure.topology.atom_names)
        token_map = mapping.build_mapping(
            structure.feature_dict, design, n_tokens=structure.num_tokens
        )
        fixed_atoms = mapping.target_atom_mask(
            structure.feature_dict, design, n_atom=n_atom
        )
        logger.info(
            "%s: %s | fixed target atoms %d/%d",
            entry.example_id,
            json.dumps(token_map.identity()),
            int(fixed_atoms.sum()),
            n_atom,
        )
        # PXDesign's own fixed-atom channel, not label_dict["coordinate"]:
        # fixed_atom_xyz/fixed_atom_mask are what the featurizer emits for this
        # purpose and what pxdesign_train/stage4.py consumes. Cross-checked
        # against the mask derived from atom_to_token_idx, since two disagreeing
        # notions of "the target" would be worse than either.
        feature_fixed = structure.feature_dict.get("fixed_atom_mask")
        feature_xyz = structure.feature_dict.get("fixed_atom_xyz")
        if feature_fixed is None or feature_xyz is None:
            raise SystemExit(
                "the featurizer emitted no fixed_atom_mask/fixed_atom_xyz, so "
                "there is no target-conditioning channel to hold fixed"
            )
        feature_fixed = feature_fixed.reshape(-1)[:n_atom].bool().to(device)
        derived = fixed_atoms.to(device)
        disagreement = int((feature_fixed ^ derived).sum())
        if disagreement:
            logger.warning(
                "%s: fixed_atom_mask and the design-derived target mask differ "
                "on %d atom(s); using the featurizer's channel",
                entry.example_id,
                disagreement,
            )
        fixed_target = replay.FixedTarget(
            reference=feature_xyz.reshape(-1, 3)[:n_atom][None].float().to(device),
            atom_mask=feature_fixed[None],
        )
        conditioning = px_driver.conditioning(structure.feature_dict)
        bound = px_driver.bind(conditioning)

        def denoise_for(bound=bound):
            """Bind this target's driver into the callable.

            Default-argument binding rather than closure capture: the loop
            rebinds `bound` per target, and a late-binding closure would
            silently denoise a later target's conditioning.
            """

            def call(x, sigma, *, feedback=None):
                return bound(x, sigma, feedback=feedback)[0]

            return call

        denoise = denoise_for()
        identity = dict(
            target=entry.example_id,
            pool=entry.pool,
            binder_chain=entry.converted_binder_chain,
            n_step=args.n_step,
            checkpoint=px_record["weights"]["sha256"][:12],
        )

        for seed in args.seeds:
            # Separate streams: FaMPNN must not be able to shift the backbone.
            bb_stream = replay.RngStream(
                f"bb:{entry.example_id}:{seed}", seed, device=device
            )
            sc_stream = replay.RngStream(
                f"sc:{entry.example_id}:{seed}", seed + 9973, device=device
            )
            steps = sorted(
                {
                    min(args.n_step - 1, max(0, int(round(f * (args.n_step - 1)))))
                    for f in args.events
                }
            )
            clock = time.perf_counter()
            with torch.no_grad():
                x0, records, stats = replay.run_trajectory(
                    denoise=denoise,
                    schedule=schedule,
                    n_atom=n_atom,
                    device=device,
                    batch_shape=(),
                    n_sample=1,
                    stream=bb_stream,
                    record_steps=steps,
                    fixed_target=fixed_target,
                    identity=identity,
                )
            generation_seconds = time.perf_counter() - clock
            logger.info(
                "%s seed %d: generated in %.1fs (%d calls, target drift %.2e A)",
                entry.example_id,
                seed,
                generation_seconds,
                stats["calls"],
                stats.get("target_max_displacement", float("nan")),
            )

            for state in records:
                prep = gen_metrics.shared_preparation(
                    controller=controller,
                    bound=bound,
                    state=state.to(device),
                    structure=structure,
                    token_map=token_map,
                    sc_stream=sc_stream,
                    pack_steps=args.pack_steps,
                    temperature=args.seq_temperature,
                )
                text, digest = sequence_hash(prep["aatype"])
                sequences[digest] = text
                logger.info(
                    "  event (step %d) sigma %.4f: sequence %s... (%s)",
                    state.step,
                    float(state.sigma.reshape(-1)[0]),
                    text[:24],
                    digest,
                )

                for arm in arms:
                    feedback_fn, module = gen_metrics.arm_feedback(
                        arm,
                        trained=trained,
                        adapters=adapters,
                        controller=controller,
                        prep=prep,
                        token_map=token_map,
                        torsions=torsions,
                        seed=seed,
                    )
                    if feedback_fn is None and arm != "baseline":
                        skipped.append(
                            dict(target=entry.example_id, reason=f"no checkpoint for {arm}")
                        )
                        continue
                    clock = time.perf_counter()
                    with torch.no_grad():
                        arm_x0, _r, arm_stats = replay.run_trajectory(
                            denoise=denoise,
                            schedule=schedule,
                            n_atom=n_atom,
                            device=device,
                            batch_shape=(),
                            n_sample=1,
                            stream=replay.RngStream(
                                f"bb:{entry.example_id}:{seed}", seed, device=device
                            ),
                            resume=state.to(device),
                            event=state.key,
                            feedback=feedback_fn,
                            fixed_target=fixed_target,
                            identity=identity,
                        )
                    arm_seconds = time.perf_counter() - clock
                    expected = 0 if arm == "baseline" else 1
                    assert arm_stats["injections"] == expected, (
                        f"{arm}: {arm_stats['injections']} injections, expected {expected}"
                    )
                    row = gen_metrics.score_arm(
                        arm=arm,
                        x0=arm_x0,
                        baseline_x0=x0,
                        structure=structure,
                        token_map=token_map,
                        fixed_target=fixed_target,
                        prep=prep,
                        controller=controller,
                        sc_stream=sc_stream,
                        pack_steps=args.pack_steps,
                        out_dir=out / "structures",
                        target=entry.example_id,
                        seed=seed,
                        event=state.key,
                        sequence_hash=digest,
                        seconds=arm_seconds,
                        denoiser_calls=arm_stats["calls"],
                        injections=arm_stats["injections"],
                        target_max_displacement=arm_stats.get("target_max_displacement"),
                        px_driver=px_driver,
                    )
                    row.update(
                        pool=entry.pool,
                        generation_seconds=generation_seconds,
                        checkpoint_id=(
                            trained[arm]["path"] if arm in trained else "baseline"
                        ),
                        contiguous_tail=token_map.contiguous_tail,
                    )
                    rows.append(row)
                    logger.info(
                        "    %-10s calls=%d inj=%d drift=%.3f A target=%.2e s=%.1f",
                        arm,
                        row["denoiser_calls"],
                        row["injection_count"],
                        row["drift_from_baseline"],
                        row["target_max_displacement"],
                        row["seconds"],
                    )

    if not rows:
        raise SystemExit(f"nothing produced; {len(skipped)} skipped: {skipped[:3]}")

    with (out / MANIFEST).open("w") as stream:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # One FASTA entry per UNIQUE sequence: arms share theirs, so the refold is
    # shared too and its pLDDT cannot be an arm-specific metric.
    with (out / "unique_sequences.fasta").open("w") as stream:
        for digest, text in sorted(sequences.items()):
            stream.write(f">{digest}\n{text}\n")
    (out / "run.json").write_text(
        json.dumps(
            dict(
                prepared=str(args.prepared),
                arms=arms,
                events=args.events,
                resolved_steps=sorted({r["event_step"] for r in rows}),
                n_step=args.n_step,
                pack_steps=args.pack_steps,
                seeds=args.seeds,
                ema=bool(args.ema),
                unique_sequences=len(sequences),
                rows=len(rows),
                pxdesign=px_record,
                fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
                checkpoints={k: v["path"] for k, v in trained.items()},
                skipped=skipped,
                seconds=round(time.time() - started, 1),
                note=(
                    "A generated backbone has no native counterpart: RMSD to the "
                    "original partner is not a correctness metric and is not "
                    "computed. Refolding is self-consistency, shared per sequence."
                ),
            ),
            indent=2,
            default=str,
        )
    )
    logger.info(
        "wrote %s (%d row(s), %d unique sequence(s))",
        out / MANIFEST,
        len(rows),
        len(sequences),
    )
    gen_metrics.report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
