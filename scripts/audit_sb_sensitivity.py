#!/usr/bin/env python
"""Where does side-chain information stop reaching the backbone coordinates?

A four-stage cascade, measured per target at fixed sigma, with identical
backbone inputs, sequence, noise and seeds across arms. The point is not to get
a number but to localise the *first* stage where the signal disappears, because
that is the only stage worth fixing.

    1. h_res        does the encoder represent the changed side chains?
    2. A_SB output  does the adapter transmit that change?
    3. coordinates  does the injected change reach the coordinates?
    4. quality      is the influence beneficial?

Three arms from one frozen proposal:

``original``   the predicted packing, re-encoded, through A_SB.
``perturbed``  chi torsions rotated with the backbone and the sequence held
               fixed, then **re-encoded from scratch**, through A_SB.
``bypass``     A_SB disabled; the plain denoiser.

**The fresh re-encode is the whole point of the perturbed arm.** An earlier
version of this control in ``scripts/eval_sb_feedback.py`` substituted the
rotated coordinates into the packed state but kept the *unperturbed*
``h_packed``, so the 128-dim node group -- the largest, and the encoder's own
view of the side chains -- never saw the perturbation and only the 26
explicitly-geometric dims did. That measures a much weaker intervention than it
appears to, and it reported the perturbation as inert. Here the structure is
re-encoded, so every feature group sees it.

Stage 3 is measured **without superposition**: both predictions come from the
same ``x_noisy`` and the same conditioning, so they are already in one frame and
aligning them would remove exactly the difference being measured.
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.eval import backbone_metrics as bb_metrics
from pxf.eval import couple as ev

logger = logging.getLogger("pxf.audit_sb")

REPORT_FILE = "sb_sensitivity_audit.json"
# Stage 2 and 3 are read against these. A change below the encoder's own
# re-encode floor, or below GPU non-determinism on the coordinates, is not a
# transmitted signal.
FLOOR_KEYS = ("h_res_floor", "coord_floor")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--structures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="the full-variant A_SB whose transmission is being audited",
    )
    p.add_argument("--config", default="configs/couple_phase2_pilot.yaml")
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--pack-steps", type=int, default=50)
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.847, 1.939],
        help="noise levels to audit; these are trajectory points of the "
        "published 400-step schedule",
    )
    p.add_argument("--max-targets", type=int, default=16)
    p.add_argument("--perturb-degrees", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--ema", dest="ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def relative(a, b):
    """Mean per-residue relative L2 change, the same measure probes.py uses."""
    delta = (a - b).float()
    scale = torch.maximum(a.float().norm(dim=-1), b.float().norm(dim=-1))
    return float((delta.norm(dim=-1) / scale.clamp_min(1e-8)).mean())


def rms(a, b, mask=None):
    """RMS coordinate difference, no superposition. Same frame by construction."""
    d = a.reshape(-1, 3).float() - b.reshape(-1, 3).float()
    if mask is not None:
        d = d[mask.reshape(-1).bool()]
    return float(d.pow(2).sum(-1).mean().sqrt()) if d.numel() else float("nan")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import yaml
    from fampnn.model.sd_model import SeqDenoiser

    from fampnn.data import residue_constants as rc
    from pxf import atom37, provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple import pilot, torsions
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_sb_feedback import load_arm
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sb_cfg = dict(config.get("sb_feedback", {}))
    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_targets:
        structures = structures[: args.max_targets]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info("%d target(s) x %d sigma(s)", len(structures), len(args.sigmas))

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
    arm = load_arm(
        args.checkpoint,
        c_h_V=c_h_V,
        c_token=px_driver.c_token,
        sb_cfg=sb_cfg,
        use_ema=args.ema,
        device=device,
    )
    logger.info(
        "auditing A_SB: variant=%s step=%d ema=%s",
        arm["variant"],
        arm["step"],
        arm["is_ema"],
    )
    adapters = CouplingAdapters(px_driver.c_token, c_h_V, sc_to_bb=arm["module"]).to(device)
    adapters.eval().requires_grad_(False)
    controller = CoupledDenoiser(
        backbone=None,
        fampnn=fampnn,
        adapters=adapters,
        phase="sc_to_bb",
        pack_steps=args.pack_steps,
    )
    canonical = load_metrics()
    bs_delta_h = None if (arm["bs_policy"] or "bypass") == "bypass" else None

    featurized = featurize_structures(
        structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )
    rows, skipped = [], []
    started = time.time()

    for index, (sample_id, source) in enumerate(featurized):
        try:
            structure = to_featurized(sample_id, source[0]).to(device)
            native = _native_parse(structures, sample_id)
            ev.check_alignment(sample_id, native["aatype"], structure.aatype)
        except Exception as error:  # noqa: BLE001 - upstream raises broadly
            skipped.append(dict(target=sample_id, reason=str(error)[:200]))
            logger.warning("skipping %s: %s", sample_id, str(error)[:160])
            continue
        aatype = structure.aatype.reshape(-1)
        if aatype.numel() == 0 or int(aatype.max()) >= 20:
            skipped.append(dict(target=sample_id, reason="non-canonical residue"))
            continue
        native37, native_mask = ev.native_atom37(native, rc)
        native37, native_mask = native37.cpu(), native_mask.cpu()
        target = structure.backbone_target.float()
        supervised = pilot.backbone_supervision_mask(
            structure.topology.atom_names,
            coordinate_mask=structure.label_dict.get("coordinate_mask"),
            device=device,
        )
        controller.backbone = px_driver.bind(px_driver.conditioning(structure.feature_dict))
        backbone_slots = list(atom37.BACKBONE_SLOTS)

        for sigma_value in args.sigmas:
            seed = ev.target_seed(args.seed, sample_id, sigma_value)
            sigma = torch.full((1,), float(sigma_value), device=device)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(target.shape, generator=generator).to(device)
            x_noisy = (target + noise * float(sigma_value))[None]

            torch.manual_seed(seed)
            with torch.no_grad():
                up = controller.frozen_half(
                    structure.topology, x_noisy, sigma, aatype, bs_delta_h=bs_delta_h
                )
                reference = controller._per_residue(up.a_token, int(aatype.shape[0]))

                # --- stage 1: does the encoder represent the change? ---
                # Floor first: re-encode the identical packing.
                repeat = controller.encode_predicted_packing(
                    up.inputs,
                    up.sidechains,
                    h_base=up.packed.h_base,
                    psce=up.packed.psce,
                )
                h_floor = relative(up.packed.h_packed, repeat.h_packed)

                deltas = torsions.random_chi_deltas(
                    up.packed.aatype,
                    args.perturb_degrees * math.pi / 180.0,
                    generator=torch.Generator().manual_seed(seed + 1),
                )
                moved = torsions.perturb_chi(
                    up.packed.coords37,
                    up.packed.aatype,
                    deltas,
                    available=up.packed.available,
                )
                # Backbone must be untouched, or stage 3 is confounded.
                assert torch.allclose(
                    moved[..., backbone_slots, :],
                    up.packed.coords37[..., backbone_slots, :],
                    atol=1e-5,
                ), "the torsion perturbation moved the backbone"
                sidechain_slots = list(atom37.SIDECHAIN_SLOTS)
                # THE FRESH RE-ENCODE. Substituting coords37 alone leaves
                # h_packed unperturbed and understates the intervention.
                packed_perturbed = controller.encode_predicted_packing(
                    up.inputs,
                    moved[..., sidechain_slots, :],
                    h_base=up.packed.h_base,
                    psce=up.packed.psce,
                )
                h_change = relative(up.packed.h_packed, packed_perturbed.h_packed)
                sc_rms = rms(
                    up.packed.coords37[..., sidechain_slots, :],
                    moved[..., sidechain_slots, :],
                )

                # --- stage 2: does the adapter transmit it? ---
                d_orig, s_orig = adapters.delta_a(up.packed, sigma, reference=reference)
                d_pert, s_pert = adapters.delta_a(
                    packed_perturbed, sigma, reference=reference
                )
                delta_change = relative(d_orig, d_pert)
                delta_rel_orig = s_orig.get("relative_residual")
                delta_rel_pert = s_pert.get("relative_residual")

                # --- stage 3: does it reach the coordinates? ---
                bb_bypass, _a = controller.backbone(x_noisy, sigma, feedback=None)
                bb_orig, _a = controller.backbone(x_noisy, sigma, feedback=d_orig)
                bb_pert, _a = controller.backbone(x_noisy, sigma, feedback=d_pert)
                # The floor: two identical calls differ only by GPU reduction order.
                bb_repeat, _a = controller.backbone(x_noisy, sigma, feedback=None)
                keep = supervised.bool()
                coord_floor = rms(bb_bypass, bb_repeat, keep)
                coord_orig_vs_bypass = rms(bb_orig, bb_bypass, keep)
                coord_orig_vs_pert = rms(bb_orig, bb_pert, keep)

                # --- stage 4: is it beneficial? ---
                scored = dict(
                    controller=controller,
                    topology=structure.topology,
                    aatype=aatype,
                    native37=native37,
                    native_mask=native_mask,
                    canonical=canonical,
                )
                q_bypass = _quality(bb_bypass, **scored)
                q_orig = _quality(bb_orig, **scored)
                q_pert = _quality(bb_pert, **scored)

            rows.append(
                dict(
                    target=sample_id,
                    sigma=float(sigma_value),
                    length=int(aatype.shape[0]),
                    # stage 1
                    h_res_floor=h_floor,
                    h_res_change=h_change,
                    sidechain_rms_moved=sc_rms,
                    # stage 2
                    delta_a_change=delta_change,
                    delta_a_rel_residual=delta_rel_orig,
                    delta_a_rel_residual_perturbed=delta_rel_pert,
                    delta_a_norm=s_orig.get("delta_a_norm"),
                    # stage 3
                    coord_floor=coord_floor,
                    coord_rms_feedback_vs_bypass=coord_orig_vs_bypass,
                    coord_rms_original_vs_perturbed=coord_orig_vs_pert,
                    # stage 4
                    bb_rmsd_bypass=q_bypass["backbone_rmsd"],
                    bb_rmsd_original=q_orig["backbone_rmsd"],
                    bb_rmsd_perturbed=q_pert["backbone_rmsd"],
                    lddt_bypass=q_bypass["lddt_backbone"],
                    lddt_original=q_orig["lddt_backbone"],
                    lddt_perturbed=q_pert["lddt_backbone"],
                )
            )
        logger.info(
            "%d/%d targets, %.1fs", index + 1, len(featurized), time.time() - started
        )

    if not rows:
        raise SystemExit(f"nothing audited; {len(skipped)} skipped: {skipped[:3]}")

    record = dict(
        label="SC -> BB sensitivity cascade",
        structures=str(args.structures),
        n_targets=len({r["target"] for r in rows}),
        sigmas=[float(s) for s in args.sigmas],
        perturb_degrees=args.perturb_degrees,
        pack_steps=args.pack_steps,
        checkpoint=dict(
            path=arm["path"],
            variant=arm["variant"],
            step=arm["step"],
            is_ema=arm["is_ema"],
            bs_policy=arm["bs_policy"],
        ),
        pxdesign=px_record,
        fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
        per_sigma=_summarise(rows, args.sigmas),
        rows=rows,
        skipped=skipped,
        seconds=round(time.time() - started, 1),
    )
    (out / REPORT_FILE).write_text(json.dumps(record, indent=2, default=str))
    with (out / "per_target.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", out / REPORT_FILE)
    report(record)
    return 0


def _quality(flat, *, controller, topology, aatype, native37, native_mask, canonical):
    """Backbone quality of one prediction against the native. Stage 4."""
    from pxf import atom37

    backbone = list(atom37.BACKBONE_SLOTS)
    dense = controller.densify(flat, topology, aatype)
    pred = torch.zeros_like(dense[0]).cpu()
    pred[:, backbone, :] = dense[0][:, backbone, :].cpu()
    mask = torch.zeros_like(native_mask)
    mask[:, backbone] = native_mask[:, backbone]
    return bb_metrics.backbone_report(
        pred, native37, mask, canonical=canonical, with_tm=False
    )


def _mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(values) / len(values) if values else float("nan")


def _summarise(rows, sigmas):
    out = []
    for value in sigmas:
        at = [r for r in rows if abs(r["sigma"] - value) < 1e-9]
        if not at:
            continue
        out.append(
            dict(
                sigma=float(value),
                n=len(at),
                **{
                    k: _mean(at, k)
                    for k in rows[0]
                    if k not in ("target", "sigma", "length")
                },
            )
        )
    return out


def report(record):
    print(f"\n=== {record['label']} ===")
    print(
        f"  {record['n_targets']} target(s), chi perturbation "
        f"{record['perturb_degrees']:g} deg, A_SB = {record['checkpoint']['variant']} "
        f"step {record['checkpoint']['step']} (ema={record['checkpoint']['is_ema']})\n"
    )
    for entry in record["per_sigma"]:
        print(f"  [sigma_B = {entry['sigma']:.3f} A, n={entry['n']}]")
        print(
            f"    stage 1  h_res      floor {entry['h_res_floor']:.3e}   "
            f"perturbed {entry['h_res_change']:.3e}   "
            f"({entry['h_res_change'] / max(entry['h_res_floor'], 1e-12):.0f}x floor)"
            f"   [side chains moved {entry['sidechain_rms_moved']:.3f} A RMS]"
        )
        print(
            f"    stage 2  A_SB       change {entry['delta_a_change']:.3e}   "
            f"residual/BB feat {entry['delta_a_rel_residual']:.4f}"
        )
        print(
            f"    stage 3  coords     floor {entry['coord_floor']:.3e}   "
            f"feedback vs bypass {entry['coord_rms_feedback_vs_bypass']:.3e} A   "
            f"original vs perturbed {entry['coord_rms_original_vs_perturbed']:.3e} A"
        )
        print(
            f"    stage 4  BB RMSD    bypass {entry['bb_rmsd_bypass']:.4f}   "
            f"feedback {entry['bb_rmsd_original']:.4f}   "
            f"perturbed {entry['bb_rmsd_perturbed']:.4f}"
        )
        print(
            f"             lDDT       bypass {entry['lddt_bypass']:.4f}   "
            f"feedback {entry['lddt_original']:.4f}   "
            f"perturbed {entry['lddt_perturbed']:.4f}"
        )
        # Where does it stop? Stage 4 asks two questions, not one: whether
        # feedback beats bypass at all, and -- the one that matters -- whether
        # the gain depends on the side chains being RIGHT. An earlier version
        # checked only the first and reported "signal survives to a quality
        # gain" on data where scrambling the rotamers retained 92% of it.
        gain = entry["bb_rmsd_bypass"] - entry["bb_rmsd_original"]
        gain_perturbed = entry["bb_rmsd_bypass"] - entry["bb_rmsd_perturbed"]
        retained = gain_perturbed / gain if abs(gain) > 1e-9 else float("nan")
        if entry["h_res_change"] <= max(entry["h_res_floor"], 1e-12) * 10:
            where = "1 -- the encoder does not represent the change"
        elif entry["delta_a_change"] <= 1e-3:
            where = "2 -- the adapter does not transmit it"
        elif (
            entry["coord_rms_original_vs_perturbed"]
            <= max(entry["coord_floor"], 1e-12) * 10
        ):
            where = "3 -- it does not reach the coordinates"
        elif gain <= 0:
            where = "4 -- it reaches them and makes things worse"
        elif not (retained < 0.5):
            where = (
                f"4 -- it reaches them, but {retained:.0%} of the gain survives "
                "scrambling the rotamers, so the benefit is not side-chain-specific"
            )
        else:
            where = "none -- a side-chain-specific quality gain survives"
        print(
            f"    stage 4b gain {gain:+.5f} A, scrambled {gain_perturbed:+.5f} A, "
            f"retained {retained:.0%}"
        )
        print(f"    -> first stage where the signal disappears: {where}\n")
    print(
        "  Stage 3 is measured without superposition: both predictions come from\n"
        "  one x_noisy and one conditioning, so aligning would remove exactly the\n"
        "  difference being measured.\n"
    )
    return 0


def _native_parse(structures, sample_id):
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    match = next((p for p in structures if Path(p).stem == sample_id), None)
    if match is None:
        raise ValueError(f"no source file for {sample_id}")
    return process_single_pdb(load_feats_from_pdb(str(match)))


if __name__ == "__main__":
    raise SystemExit(main())
