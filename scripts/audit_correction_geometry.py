#!/usr/bin/env python
"""What does the correction actually DO to the coordinates?

A maximum displacement answers almost nothing. 1.9 A of worst-atom movement for
0.018 A of mean RMSD gain is equally consistent with a few outliers, with a
global pose shift, or with large movement in directions that barely touch the
error -- and calling that "self-cancelling" is an inference the maximum cannot
support. This measures the alternatives apart:

``rms_displacement``     the typical correction, not the worst one
``median`` / ``p95`` / ``max``   whether the movement is localized or diffuse
``rms_after_superposition``      what survives rigid alignment of bb1 onto bb0.
                         The gap between this and rms_displacement IS the
                         global pose component, which changes every coordinate
                         and can improve or harm the unsuperposed RMSD without
                         being a local correction at all.
``gain`` versus displacement     whether the big corrections are the ones that
                         help. A positive slope means the correction is doing
                         work; a flat or negative one means the magnitude is
                         going somewhere other than accuracy.
backbone bonds / angles / clashes on the ACTUAL output, because a correction
                         that buys RMSD by stretching peptide bonds has not
                         bought anything.

Reported per arm. **If the full and BB-only arms move the same way, the movement
is a property of the receiving interface rather than of the side-chain
information** -- which is the distinction the whole experiment exists to make,
and it is visible here in a way that a single RMSD column is not.

Ideal geometry is Engh & Huber; the values are written out below rather than
imported because the side-chain chemistry tables in the canonical package are
built around side-chain rigid groups and do not carry the inter-residue peptide
bond, which is exactly the term a backbone correction is most likely to break.
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

from pxf.couple import schedule
from pxf.eval import backbone_metrics as bb_metrics
from pxf.eval import couple as ev

logger = logging.getLogger("pxf.audit_correction_geometry")

# atom37 slots. Spelled out rather than N/CA/C/O, because a bare `O` is an
# ambiguous single-character name and `C` would shadow nothing useful either.
SLOT_N, SLOT_CA, SLOT_C, SLOT_O = 0, 1, 2, 4
BACKBONE_SLOTS = (SLOT_N, SLOT_CA, SLOT_C, SLOT_O)
# Engh & Huber ideal backbone geometry. Lengths in Angstroms, angles in degrees.
BONDS = (
    ("N-CA", SLOT_N, SLOT_CA, 1.458),
    ("CA-C", SLOT_CA, SLOT_C, 1.525),
    ("C-O", SLOT_C, SLOT_O, 1.231),
)
PEPTIDE_BOND = ("C-N", 1.329)
ANGLES = (
    ("N-CA-C", SLOT_N, SLOT_CA, SLOT_C, 111.0),
    ("CA-C-O", SLOT_CA, SLOT_C, SLOT_O, 120.8),
)
# Across the peptide bond, ideal degrees. Handled separately below because each
# spans two residues.
LINK_CA_C_N, LINK_C_N_CA = 116.2, 121.7
# Two backbone heavy atoms closer than this, in residues that are not bonded
# neighbours, are overlapping rather than packed.
CLASH_RADIUS = 2.8


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--structures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/couple_early_e1.yaml")
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--pack-steps", type=int, default=50)
    p.add_argument("--n-sigma", type=int, default=4)
    p.add_argument("--max-targets", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--ema", dest="ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    return p.parse_args(argv)


# --- geometry ---------------------------------------------------------------


def displacement_stats(before, after, keep):
    """How far the correction moved things, and how much of it is a rigid motion.

    ``before``/``after`` are flat ``[N_atom, 3]``; ``keep`` selects the
    supervised backbone atoms. Superposition uses the evaluator's own Kabsch so
    the alignment convention matches every other number in this pipeline.
    """
    a, b = before.reshape(-1, 3)[keep], after.reshape(-1, 3)[keep]
    offset = (b - a).norm(dim=-1)
    rms = float(offset.pow(2).mean().sqrt())
    weight = torch.ones(a.shape[0], device=a.device)
    aligned = bb_metrics._kabsch(b[None], a[None], weight[None])[0]
    residual = (aligned - a).norm(dim=-1)
    rms_super = float(residual.pow(2).mean().sqrt())
    quantiles = torch.tensor([0.5, 0.95], device=offset.device)
    median, p95 = (float(v) for v in torch.quantile(offset, quantiles))
    return dict(
        rms_displacement=rms,
        rms_after_superposition=rms_super,
        # What fraction of the squared movement a rigid motion accounts for.
        # 1.0 means the correction is purely a pose change; 0.0 means it is
        # entirely internal.
        rigid_fraction=(
            float(max(0.0, 1.0 - (rms_super**2) / rms**2)) if rms > 1e-12 else 0.0
        ),
        displacement_median=median,
        displacement_p95=p95,
        displacement_max=float(offset.max()),
    )


def backbone_chemistry(dense, residue_index, seq_mask):
    """Bonds, angles and clashes on the structure actually produced.

    ``dense`` is ``[L, 37, 3]``. Inter-residue terms are computed only across
    consecutive residue indices, so a chain break does not register as a 40 A
    peptide bond.
    """
    coords = dense.float()
    length = coords.shape[0]
    real = seq_mask.reshape(-1).bool()[:length]
    index = residue_index.reshape(-1)[:length].long()
    linked = real[:-1] & real[1:] & ((index[1:] - index[:-1]) == 1)

    deviations, angle_deviations = [], []
    for _name, i, j, ideal in BONDS:
        d = (coords[:, j] - coords[:, i]).norm(dim=-1)[real]
        deviations.append(d - ideal)
    if bool(linked.any()):
        d = (coords[1:, SLOT_N] - coords[:-1, SLOT_C]).norm(dim=-1)[linked]
        deviations.append(d - PEPTIDE_BOND[1])

    def angle(p, q, r):
        u, v = p - q, r - q
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-8)
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    for _name, i, j, k, ideal in ANGLES:
        angle_deviations.append(angle(coords[:, i], coords[:, j], coords[:, k])[real] - ideal)
    if bool(linked.any()):
        # CA(i)-C(i)-N(i+1) and C(i)-N(i+1)-CA(i+1)
        angle_deviations.append(
            angle(coords[:-1, SLOT_CA], coords[:-1, SLOT_C], coords[1:, SLOT_N])[linked]
            - LINK_CA_C_N
        )
        angle_deviations.append(
            angle(coords[:-1, SLOT_C], coords[1:, SLOT_N], coords[1:, SLOT_CA])[linked]
            - LINK_C_N_CA
        )

    bond = torch.cat(deviations) if deviations else torch.zeros(1)
    ang = torch.cat(angle_deviations) if angle_deviations else torch.zeros(1)

    # Clashes between backbone heavy atoms of residues more than one apart.
    slots = list(BACKBONE_SLOTS)
    points = coords[real][:, slots, :].reshape(-1, 3)
    owner = torch.arange(int(real.sum()), device=coords.device).repeat_interleave(len(slots))
    distance = torch.cdist(points, points)
    separated = (owner[:, None] - owner[None, :]).abs() > 1
    clashes = int(((distance < CLASH_RADIUS) & separated).sum() // 2)
    return dict(
        bond_rms_deviation=float(bond.pow(2).mean().sqrt()),
        bond_max_deviation=float(bond.abs().max()),
        angle_rms_deviation_deg=float(ang.pow(2).mean().sqrt()),
        angle_max_deviation_deg=float(ang.abs().max()),
        backbone_clashes=clashes,
        backbone_clashes_per_residue=clashes / max(1, int(real.sum())),
    )


def score_backbone(controller, flat, topology, aatype, native37, native_mask, canonical):
    """``(dense, report)`` for one flat coordinate tensor, backbone slots only."""
    dense = controller.densify(flat, topology, aatype)
    slots = list(BACKBONE_SLOTS)
    pred = torch.zeros_like(dense[0]).cpu()
    pred[:, slots, :] = dense[0][:, slots, :].cpu()
    mask = torch.zeros_like(native_mask)
    mask[:, slots] = native_mask[:, slots]
    return dense, bb_metrics.backbone_report(
        pred, native37, mask, canonical=canonical, with_tm=False
    )


def pearson(xs, ys):
    """Correlation, or nan when one side does not vary."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


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
    from pxf.couple import pilot
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_sb_feedback import load_arm
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sigma_schedule = schedule.from_config(config.get("sigma"))
    sigmas = ev.sweep_sigmas(sigma_schedule, args.n_sigma)
    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_targets:
        structures = structures[: args.max_targets]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    checkpoint = provenance.fampnn_checkpoint(args.fampnn_weights)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval().requires_grad_(False).to(device)
    c_h_V = node_feature_dim(fampnn)

    px_model, _cfg, _rec = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    trained = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        trained[label] = load_arm(
            path, c_h_V=c_h_V, c_token=px_driver.c_token, c_s=px_driver.c_s,
            c_z=px_driver.c_z, sb_cfg=config.get("sb_feedback", {}), use_ema=args.ema,
            device=device, expect=label,
        )
        logger.info("arm %s: arch=%s variant=%s", label, trained[label]["arch"],
                    trained[label]["variant"])

    adapters = CouplingAdapters(px_driver.c_token, c_h_V).to(device).eval()
    adapters.requires_grad_(False)
    controller = CoupledDenoiser(
        backbone=None, fampnn=fampnn, adapters=adapters, phase="sc_to_bb",
        pack_steps=args.pack_steps,
    )
    from fampnn.data import residue_constants as rc
    from pxf.eval.canonical import load as load_metrics

    canonical = load_metrics()
    rows, started = [], time.time()
    featurized = featurize_structures(
        structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )
    for position, (sample_id, source) in enumerate(featurized):
        try:
            structure = to_featurized(sample_id, source[0]).to(device)
            native = _native_parse(structures, sample_id)
            ev.check_alignment(sample_id, native["aatype"], structure.aatype)
        except Exception as error:  # noqa: BLE001
            logger.warning("skipping %s: %s", sample_id, str(error)[:120])
            continue
        aatype = structure.aatype.reshape(-1)
        if aatype.numel() == 0 or int(aatype.max()) >= 20:
            continue
        native37, native_mask = ev.native_atom37(native, rc)
        native37, native_mask = native37.cpu(), native_mask.cpu()
        target = structure.backbone_target.float()
        supervised = pilot.backbone_supervision_mask(
            structure.topology.atom_names,
            coordinate_mask=structure.label_dict.get("coordinate_mask"), device=device,
        ).bool()
        controller.backbone = px_driver.bind(px_driver.conditioning(structure.feature_dict))

        for sigma_value in sigmas:
            seed = ev.target_seed(args.seed, sample_id, sigma_value)
            sigma = torch.full((1,), float(sigma_value), device=device)
            noise = torch.randn(
                target.shape, generator=torch.Generator().manual_seed(seed)
            ).to(device)
            x_noisy = (target + noise * float(sigma_value))[None]
            torch.manual_seed(seed)
            with torch.no_grad():
                upstream = controller.frozen_half(
                    structure.topology, x_noisy, sigma, aatype, bs_delta_h=None
                )
            reference = controller._per_residue(upstream.a_token, int(aatype.shape[0]))

            def scored(flat, _s=structure, _a=aatype, _n=native37, _m=native_mask):
                return score_backbone(
                    controller, flat, _s.topology, _a, _n, _m, canonical
                )

            bb0_dense, bb0_report = scored(upstream.bb0_flat)
            base = dict(
                target=sample_id, sigma=float(sigma_value), length=int(aatype.shape[0])
            )
            rows.append(
                dict(
                    base, arm="bb0", backbone_rmsd=bb0_report["backbone_rmsd"], gain=0.0,
                    **{k: 0.0 for k in (
                        "rms_displacement", "rms_after_superposition", "rigid_fraction",
                        "displacement_median", "displacement_p95", "displacement_max")},
                    **backbone_chemistry(
                        bb0_dense[0], upstream.inputs.residue_index,
                        upstream.packed.seq_mask
                    ),
                )
            )
            for label, arm in trained.items():
                adapters.sc_to_bb = arm["module"]
                with torch.no_grad():
                    delta, _stats = adapters.delta_a(
                        upstream.packed, sigma, reference=reference
                    )
                    bb1_flat, _a = controller.backbone(x_noisy, sigma, feedback=delta)
                bb1_dense, bb1_report = scored(bb1_flat)
                rows.append(
                    dict(
                        base, arm=label, variant=arm["variant"], arch=arm["arch"],
                        backbone_rmsd=bb1_report["backbone_rmsd"],
                        gain=bb0_report["backbone_rmsd"] - bb1_report["backbone_rmsd"],
                        **displacement_stats(upstream.bb0_flat, bb1_flat, supervised),
                        **backbone_chemistry(
                            bb1_dense[0], upstream.inputs.residue_index,
                            upstream.packed.seq_mask
                        ),
                    )
                )
        if position % 8 == 0:
            logger.info("%d/%d targets, %.0fs", position + 1, len(featurized),
                        time.time() - started)

    # --- aggregate ---
    arms = sorted({r["arm"] for r in rows if r["arm"] != "bb0"})
    summary = {}
    keys = (
        "rms_displacement", "rms_after_superposition", "rigid_fraction",
        "displacement_median", "displacement_p95", "displacement_max",
        "gain", "backbone_rmsd", "bond_rms_deviation", "bond_max_deviation",
        "angle_rms_deviation_deg", "backbone_clashes_per_residue",
    )
    for arm in ["bb0", *arms]:
        at = [r for r in rows if r["arm"] == arm]
        if not at:
            continue
        entry = {k: sum(r[k] for r in at) / len(at) for k in keys if k in at[0]}
        entry["n"] = len(at)
        if arm != "bb0":
            entry["corr_gain_vs_displacement"] = pearson(
                [r["rms_displacement"] for r in at], [r["gain"] for r in at]
            )
        summary[arm] = entry

    record = dict(
        label="what the correction does to the coordinates",
        structures=str(args.structures), n_targets=len({r["target"] for r in rows}),
        sigma_values=[float(s) for s in sigmas], ema=bool(args.ema),
        ideal_geometry="Engh & Huber", clash_radius=CLASH_RADIUS, arms=summary,
    )
    (out / "correction_geometry.json").write_text(json.dumps(record, indent=2, default=str))
    with (out / "per_target.csv").open("w") as stream:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report(record)
    return 0


def report(record):
    print(f"\n=== {record['label']} ===")
    print(f"  {record['n_targets']} targets x {len(record['sigma_values'])} sigma, "
          f"ema={record['ema']}\n")
    head = ("arm", "BB RMSD", "gain", "rms disp", "rms sup", "rigid", "median", "p95", "max")
    print("  " + " ".join(f"{h:>12s}" for h in head))
    for arm, e in record["arms"].items():
        print("  " + " ".join(f"{v:>12}" for v in (
            f"{arm:<12s}", f"{e['backbone_rmsd']:.4f}", f"{e['gain']:+.4f}",
            f"{e.get('rms_displacement', 0):.4f}", f"{e.get('rms_after_superposition', 0):.4f}",
            f"{e.get('rigid_fraction', 0):.3f}", f"{e.get('displacement_median', 0):.4f}",
            f"{e.get('displacement_p95', 0):.4f}", f"{e.get('displacement_max', 0):.4f}")))
    print("\n  rigid = fraction of squared movement a rigid motion of bb1 onto bb0")
    print("  explains. High means the correction is mostly a pose change.\n")
    head2 = ("arm", "bond RMS", "bond max", "angle RMS", "clash/res", "corr(disp,gain)")
    print("  " + " ".join(f"{h:>16s}" for h in head2))
    for arm, e in record["arms"].items():
        print("  " + " ".join(f"{v:>16}" for v in (
            f"{arm:<16s}", f"{e['bond_rms_deviation']:.4f}",
            f"{e['bond_max_deviation']:.4f}", f"{e['angle_rms_deviation_deg']:.3f}",
            f"{e['backbone_clashes_per_residue']:.4f}",
            f"{e.get('corr_gain_vs_displacement', float('nan')):+.3f}")))
    print("\n  If full and bb_only move the same way, the movement is a property")
    print("  of the receiving interface, not of the side-chain information.\n")


def _native_parse(structures, sample_id):
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    match = next((p for p in structures if Path(p).stem == sample_id), None)
    if match is None:
        raise ValueError(f"no source file for {sample_id}")
    return process_single_pdb(load_feats_from_pdb(str(match)))


if __name__ == "__main__":
    raise SystemExit(main())
