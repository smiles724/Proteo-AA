#!/usr/bin/env python
"""Can the SC -> BB path carry signal on the real data? The representation check.

The tests establish this on one CASP structure. This establishes it on the panel
the pilot actually trains and is scored on, and on the realization the feedback
actually sees -- a *predicted* packing on a *denoised* backbone, not native side
chains on a deposited one.

For each target and each sigma it reports:

``floor``      re-encoding identical input twice. Zero for a deterministic
               encoder, and the scale every response has to clear.
``ceiling``    masked side chains to visible. The most the side-chain input can
               do to the encoder, so responses are readable as a fraction of it.
``chi_*``      rotating about chi axes by 10 to 120 degrees, backbone and
               sequence fixed. The only perturbation a packer could actually
               produce, and therefore the one that matters.
``gaussian_*`` coordinate noise, kept as a wiring diagnostic. It breaks bond
               lengths and angles, so a response to it says the encoder reads
               the block, not that it is sensitive to packing.
``invariants`` nonexistent slots, padded-row contents, and a rigid motion of the
               whole structure. These must not move the representation; a
               response that came from one of them is plumbing, not signal.

A verdict of "insensitive" here would mean the pilot cannot work no matter how
A_SB is trained, and is worth knowing before spending three training runs on it.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import schedule

logger = logging.getLogger("pxf.probe_sb_feedback")

REPORT_FILE = "sb_sensitivity.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--structures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/couple_phase2_pilot.yaml")
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--pack-steps", type=int, default=50)
    p.add_argument("--n-sigma", type=int, default=3)
    p.add_argument("--max-targets", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--native-sidechains",
        action="store_true",
        help="probe native side chains instead of the predicted packing. The "
        "comparison of interest, but not what the feedback ever reads",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yaml
    from fampnn.model.sd_model import SeqDenoiser

    from pxf import provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple import fampnn_iface as iface
    from pxf.couple import probes
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.device import select_device
    from pxf.eval import couple as ev

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sigma_schedule = schedule.from_config(config.get("sigma"))
    sigmas = ev.sweep_sigmas(sigma_schedule, args.n_sigma)
    structures = resolve_structures(args.structures, suffix=".cif")[
        : args.max_targets or None
    ]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

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

    px_model, _configs, px_record = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)
    controller = CoupledDenoiser(
        backbone=None,
        fampnn=fampnn,
        adapters=CouplingAdapters(px_driver.c_token, iface.node_feature_dim(fampnn)).to(
            device
        ),
        phase="sc_to_bb",
        pack_steps=args.pack_steps,
    )

    rows, skipped = [], []
    started = time.time()
    for index, path in enumerate(structures):
        try:
            sample_id, source = featurize_structures(
                [path], crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
            )[0]
            structure = to_featurized(sample_id, source[0]).to(device)
        except Exception as error:  # noqa: BLE001 - upstream raises broadly
            skipped.append(dict(target=Path(path).stem, reason=str(error)[:200]))
            logger.warning("skipping %s: %s", Path(path).stem, str(error)[:160])
            continue
        aatype = structure.aatype.reshape(-1)
        if aatype.numel() == 0 or int(aatype.max()) >= 20:
            skipped.append(dict(target=sample_id, reason="non-canonical residue"))
            continue
        target = structure.backbone_target.float()
        controller.backbone = px_driver.bind(px_driver.conditioning(structure.feature_dict))
        for sigma_value in sigmas:
            seed = ev.target_seed(args.seed, sample_id, sigma_value)
            sigma = torch.full((1,), float(sigma_value), device=device)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(target.shape, generator=generator).to(device)
            torch.manual_seed(seed)
            with torch.no_grad():
                upstream = controller.frozen_half(
                    structure.topology,
                    (target + noise * float(sigma_value))[None],
                    sigma,
                    aatype,
                    bs_delta_h=None,
                )
                inputs = upstream.inputs
                report = probes.sidechain_sensitivity(
                    fampnn,
                    inputs.coords_af2,
                    inputs.aatype,
                    seq_mask=inputs.seq_mask,
                    missing_atom_mask=inputs.missing_atom_mask,
                    residue_index=inputs.residue_index,
                    chain_index=inputs.chain_index,
                    supplied_atom_mask=inputs.atom_mask,
                    sidechains=None if args.native_sidechains else upstream.sidechains,
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
            summary = report.summary()
            rows.append(
                dict(
                    target=sample_id,
                    sigma=float(sigma_value),
                    length=int(aatype.shape[0]),
                    **{k: v for k, v in summary.items() if not isinstance(v, dict)},
                    **{f"response_{k}": v for k, v in summary["responses"].items()},
                    **{f"invariant_{k}": v for k, v in summary["invariants"].items()},
                    failures=sorted(summary["invariance_failures"]),
                    sidechain_atoms=summary["stats"]["sidechain_atoms_available"],
                    chis_measurable=summary["stats"]["chis_measurable"],
                )
            )
        logger.info(
            "%d/%d targets, %.1fs elapsed",
            index + 1,
            len(structures),
            time.time() - started,
        )

    if not rows:
        raise SystemExit(f"nothing was probed; {len(skipped)} skipped: {skipped[:3]}")

    def mean(key):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else float("nan")

    keys = sorted({k for r in rows for k in r if k.startswith(("response_", "invariant_"))})
    pooled = {
        "floor": mean("floor"),
        "ceiling": mean("ceiling"),
        **{k: mean(k) for k in keys},
    }
    verdicts = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    failures = sorted({f for r in rows for f in r["failures"]})
    record = dict(
        label="SC->BB representation sensitivity on the real panel",
        structures=str(args.structures),
        n_targets=len({r["target"] for r in rows}),
        n_probes=len(rows),
        sigma_values=[float(s) for s in sigmas],
        pack_steps=args.pack_steps,
        sidechains="native" if args.native_sidechains else "predicted",
        pooled=pooled,
        verdicts=verdicts,
        invariance_failures=failures,
        invariant_tolerance=dict(probes.INVARIANT_TOLERANCE),
        pxdesign=px_record,
        fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
        per_probe=rows,
        skipped=skipped,
        seconds=round(time.time() - started, 1),
    )
    (out / REPORT_FILE).write_text(json.dumps(record, indent=2, default=str))

    print(f"\n=== {record['label']} ===")
    print(
        f"  {record['n_targets']} target(s), {record['n_probes']} probe(s), "
        f"{record['sidechains']} side chains, {args.pack_steps} pack steps\n"
    )
    print(f"  {'quantity':34s} {'mean relative change':>22s}")
    # On a GPU the floor is ~1.7e-7 rather than the exact 0.0 a CPU gives:
    # re-encoding identical input twice is bit-identical only if the reductions
    # are. Every response and invariant below is read against it.
    print(f"  {'floor (identical input)':34s} {pooled['floor']:22.3e}")
    print(f"  {'ceiling (masked -> visible)':34s} {pooled['ceiling']:22.3e}")
    for key in keys:
        if key.startswith("response_"):
            print(f"  {key[len('response_') :]:34s} {pooled[key]:22.3e}")
    print()
    for key in keys:
        if key.startswith("invariant_"):
            name = key[len("invariant_") :]
            # The same rule SensitivityReport.invariance_failures uses: an
            # absolute budget OR a multiple of the encoder's own floor,
            # whichever is larger. Comparing against the absolute budget alone
            # flags "exactly invariant" as a failure wherever the floor is not
            # exactly zero -- which it is not on a GPU, where non-deterministic
            # reduction order puts it at ~1.7e-7 rather than the 0.0 seen on a
            # CPU. Scrambling nonexistent atoms lands *at* that floor, which is
            # the correct result and was being reported as "OVER TOLERANCE".
            absolute = probes.INVARIANT_TOLERANCE.get(name, 0.0)
            limit = max(pooled["floor"] * probes.INVARIANCE_MARGIN, absolute, 1e-9)
            flag = "  OVER TOLERANCE" if pooled[key] > limit else ""
            print(
                f"  invariant {name:26s} {pooled[key]:22.3e}  "
                f"(<= {limit:.3e}, {'floor-scaled' if limit > absolute else 'absolute'})"
                f"{flag}"
            )
    print(f"\n  verdicts: {verdicts}")
    if failures:
        print(f"  INVARIANCE FAILURES on some probes: {failures}")
    print(
        "\n  'insensitive' would mean no A_SB can transmit anything about the\n"
        "  packing, whatever it is trained on. 'informative' means the path\n"
        "  carries signal; whether the correction uses it is the pilot's job.\n"
    )
    logger.info("wrote %s", out / REPORT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
