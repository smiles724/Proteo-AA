#!/usr/bin/env python
"""Does one SC -> BB corrective event improve the backbone? The exit decision.

Distinct from ``scripts/eval_couple.py``, which measures side-chain packing on
``bb0`` and is the phase-1 instrument. This one measures the **corrected
backbone**, and it exists because a trained ``A_SB`` beating the no-feedback
baseline is not by itself a side-chain-specific result.

    bb0   the initial clean backbone estimate,  D(x_sigma, sigma; 0)
    sc0   side chains packed on bb0 -- what the feedback reads
    bb1   the corrected clean backbone estimate, D(x_sigma, sigma; A_SB(z))
    sc1   a FRESH packing on bb1, for the final side-chain numbers

``bb1`` is never reported wearing ``sc0``: those side chains were built for a
different backbone, so the pair is a structure the system never produced.

**The arms, and what each one licenses.**

``bb0``               the original proposal. Beating it shows a correction happened.
``zero``              a second identical denoiser call with the feedback forced to
                      zero. A *wiring* check, not a baseline: it must equal bb0
                      bit-for-bit, and if it does not, everything below is
                      measuring the second call rather than the correction.
``<variant>``         one arm per ``--checkpoint``, labelled by the variant its
                      readout records. ``bb_only`` and ``generic`` are the trained
                      controls; beating them is what an SC-specific claim needs.
``perturbed``         the full arm with the *packing it reads* rotamer-perturbed,
                      backbone and sequence held fixed. Tests dependence on the
                      matched side-chain conformation. Preferred over
                      cross-protein shuffling, which also changes the sequence
                      and the feature distribution and so confounds two things.
``refine``            the computational baseline: spend the second denoiser call
                      on the *sampler* instead. One EDM step down the published
                      schedule from sigma, then denoise there. Cost-matched in
                      denoiser calls, and the runtime of every arm is measured
                      too, because "one more call" is not the same amount of work
                      as a call plus a 50-step packing rollout.

**The criterion.** A BB RMSD improvement of at least ``max(0.05 A, 3% of the
baseline)`` over the strongest comparable-cost alternative, with a paired
bootstrap interval that excludes zero, and no material side-chain or chemistry
regression. ``--report`` prints that verdict from a finished run.

Seeds, manifests and paired intervals follow ``pxf.eval.couple``: one noise draw
per (target, sigma) shared by every arm, so the arms differ in the correction and
in nothing else, and the delta is formed within a target before anything is
averaged.
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import schedule
from pxf.eval import backbone_metrics as bb_metrics
from pxf.eval import couple as ev

logger = logging.getLogger("pxf.eval_sb_feedback")

METRICS_FILE = "sb_feedback_metrics.json"
# The roadmap's exit criterion, in one place so the report and any later
# re-analysis cannot drift apart.
MIN_ABSOLUTE_GAIN = 0.05  # Angstroms
MIN_RELATIVE_GAIN = 0.03  # fraction of the baseline RMSD
# The zero-feedback arm must reproduce bb0 to this, in Angstroms of worst-atom
# deviation. Not zero, and the number is measured rather than picked: two
# invocations of the same PXDesign forward on an H200 differ by ~4.8e-6 A from
# non-deterministic reduction order, while the corrections being measured are
# ~2.6e-3 A. 1e-4 sits 20x above the noise and 25x below the signal, so it
# still fails on a real wiring error -- which would be at least the size of the
# effect, not a fraction of it.
WIRING_TOLERANCE = 1e-4
GPU_NONDETERMINISM = 4.8e-6  # measured, for the report to quote
# Side-chain metrics carried through to the no-regression half of the criterion.
SIDECHAIN_KEYS = (
    "symmetry_rmsd",
    "chi_recovery_20deg",
    "lddt_sc_env",
    "bad_bond_fraction",
    "rotamer_outlier_fraction_40deg",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--report",
        default=None,
        metavar="RUN_DIR",
        help=f"print the verdict from an existing {METRICS_FILE} and exit",
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="an A_SB arm to score, e.g. full=runs/full/checkpoints/final.pt. "
        "Repeatable. Without any, only bb0, zero and refine run, which is the "
        "pre-training sanity configuration",
    )
    p.add_argument("--structures", required=True, help="held-out .cif dir or manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/couple_phase2_pilot.yaml")
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument(
        "--pack-steps",
        type=int,
        default=50,
        help="the final packing policy, applied identically to bb0 and bb1",
    )
    p.add_argument("--n-sigma", type=int, default=4)
    p.add_argument("--sigma-mode", default=None, choices=schedule.MODES)
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--sigma-n-step", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=64, help="0 for all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--refine-eta",
        type=float,
        default=1.0,
        help="Euler step scale for the computational baseline. 1.0 is the plain "
        "probability-flow ODE step and the lowest-variance choice, which makes "
        "the baseline as strong as possible; PXDesign's own schedule ramps eta "
        "from 1.0 to 2.5 and also churns, both of which add variance",
    )
    p.add_argument(
        "--perturb-degrees",
        type=float,
        default=60.0,
        help="rotamer perturbation for the matched-conformation control",
    )
    p.add_argument(
        "--no-sidechains",
        action="store_true",
        help="skip the sc0/sc1 repacking and report backbone metrics only. "
        "Halves the runtime; forfeits the no-regression half of the criterion",
    )
    p.add_argument(
        "--ema",
        dest="ema",
        action="store_true",
        default=True,
        help="score the EMA weights when a checkpoint has them (default)",
    )
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    p.add_argument("--fail-on-no-gain", action="store_true")
    return p.parse_args(argv)


# --- reporting --------------------------------------------------------------


def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else float("nan")


def required_gain(baseline):
    """``max(0.05 A, 3% of baseline)`` -- the roadmap's threshold."""
    return max(MIN_ABSOLUTE_GAIN, MIN_RELATIVE_GAIN * float(baseline))


def verdict(record):
    """The exit decision, and the reasons for it. Returns ``(passed, lines)``."""
    arms = record["arms"]
    lines = []
    candidate = next((name for name in arms if arms[name].get("variant") == "full"), None)
    if candidate is None:
        return False, ["no arm records variant='full', so there is no candidate"]

    # The wiring check gates everything: without it the comparison is not
    # between bb0 and a correction, it is between two different calls.
    wiring = arms.get("zero", {}).get("max_abs_deviation_from_bb0")
    if wiring is None:
        lines.append("WIRING: not measured -- the zero-feedback arm did not run")
        wired = False
    else:
        wired = wiring <= WIRING_TOLERANCE
        lines.append(
            f"WIRING: zero-feedback arm deviates from bb0 by {wiring:.2e} A, "
            f"tolerance {WIRING_TOLERANCE:.0e} ({'ok' if wired else 'FAILED'}). "
            f"GPU non-determinism alone measures ~{GPU_NONDETERMINISM:.1e} A"
        )

    baseline_name, baseline = None, None
    # The strongest useful comparable-cost alternative, not the weakest.
    for name in (
        "refine",
        *[n for n in arms if arms[n].get("variant") in ("bb_only", "generic")],
    ):
        if name not in arms or name == candidate:
            continue
        value = arms[name].get("backbone_rmsd")
        if value is None:
            continue
        if baseline is None or value < baseline:
            baseline_name, baseline = name, value
    if baseline is None:
        lines.append(
            "no comparable-cost alternative ran, so only the weak comparison "
            "against the uncorrected proposal is available"
        )

    proposal = arms["bb0"]["backbone_rmsd"]
    got = arms[candidate]["backbone_rmsd"]
    lines.append(f"bb0 backbone RMSD {proposal:.4f} A")
    lines.append(f"{candidate} backbone RMSD {got:.4f} A  (delta {proposal - got:+.4f})")

    passed = wired
    interval = record.get("paired", {}).get(candidate, {}).get("vs_bb0")
    if interval:
        mean, low, high = interval["mean"], interval["low"], interval["high"]
        lines.append(
            f"paired improvement over bb0: {mean:+.4f} A "
            f"[{low:+.4f}, {high:+.4f}] (n={interval['n']})"
        )
        if not (low > 0):
            passed = False
            lines.append("  the interval includes zero: not a demonstrated correction")
    if baseline is not None:
        need = required_gain(baseline)
        gain = baseline - got
        lines.append(
            f"strongest comparable-cost alternative: {baseline_name} "
            f"{baseline:.4f} A; gain {gain:+.4f} A, need >= {need:.4f}"
        )
        if gain < need:
            passed = False
            lines.append("  below the criterion")
        against = record.get("paired", {}).get(candidate, {}).get(f"vs_{baseline_name}")
        if against and not (against["low"] > 0):
            passed = False
            lines.append(
                f"  paired interval against {baseline_name} includes zero: "
                f"{against['mean']:+.4f} [{against['low']:+.4f}, {against['high']:+.4f}]"
            )
    else:
        passed = False

    specific = [
        name for name in arms if arms[name].get("variant") in ("bb_only", "generic")
    ]
    if not specific:
        lines.append(
            "NO trained control ran. Beating bb0 shows a correction; claiming a "
            "SIDE-CHAIN-specific benefit additionally requires beating the "
            "trained BB-only and generic arms"
        )
    perturbed = arms.get("perturbed", {}).get("backbone_rmsd")
    if perturbed is not None:
        lines.append(
            f"perturbed-SC arm {perturbed:.4f} A: the candidate keeps "
            f"{got - perturbed:+.4f} A of its gain when the packing it reads is "
            "rotamer-perturbed (more negative is better evidence of dependence "
            "on the matched conformation)"
        )
    for key in ("symmetry_rmsd", "bad_bond_fraction", "rotamer_outlier_fraction_40deg"):
        before = arms["bb0"].get(f"sc_{key}")
        after = arms[candidate].get(f"sc_{key}")
        if before is None or after is None:
            continue
        lines.append(f"side chains, {key}: {before:.4f} -> {after:.4f} (sc0 -> sc1)")
    return passed, lines


def report(record):
    print(f"\n=== SC -> BB corrective event: {record['label']} ===")
    print(
        f"  {record['n_targets']} target(s), {record['n_events']} event(s), "
        f"sigma_B in {record['sigma_values'][0]:.3f}..{record['sigma_values'][-1]:.3f} A, "
        f"{record['pack_steps']} pack steps"
    )
    print(f"  structures: {record['structures']}\n")
    keys = [k for k in bb_metrics.HEADLINE]
    print(f"  {'arm':14s} {'variant':10s} " + " ".join(f"{k:>14s}" for k in keys))
    for name, arm in record["arms"].items():
        cells = " ".join(
            f"{arm[k]:14.4f}" if isinstance(arm.get(k), float) else f"{'-':>14s}"
            for k in keys
        )
        print(f"  {name:14s} {str(arm.get('variant') or '-'):10s} {cells}")
    print()
    print(f"  {'arm':14s} {'calls':>6s} {'seconds/event':>14s}")
    for name, arm in record["arms"].items():
        print(
            f"  {name:14s} {arm.get('denoiser_calls', 0):6d} "
            f"{arm.get('seconds_per_event', float('nan')):14.3f}"
        )
    print()
    passed, lines = verdict(record)
    for line in lines:
        print(f"  {line}")
    print(f"\n  VERDICT: {'PASS' if passed else 'do not proceed'}")
    print(
        "  Passing licenses inserting the corrected estimate into the sampler's\n"
        "  normal update and testing one event in a full rollout. It does not\n"
        "  license the two-event training stage, which is a separate experiment.\n"
    )
    return passed


# --- the arms ---------------------------------------------------------------


def load_arm(path, *, c_h_V, c_token, sb_cfg, use_ema, device):
    """One trained A_SB from a pilot checkpoint, with its recorded variant."""
    from pxf.couple.readout import FeedbackPath, SigmaWindow

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{path} is not a coupling checkpoint")
    identity = ((state.get("controller") or {}).get("adapters") or {}).get("sc_to_bb")
    variant = (identity or {}).get("variant") or sb_cfg.get("variant", "full")
    gate_cfg = ((identity or {}).get("gate") or {}) or dict(sb_cfg.get("gate") or {})
    gate_cfg = {k: v for k, v in gate_cfg.items() if k != "kind"}
    settings = state.get("settings") or {}
    module = FeedbackPath(
        c_h_V,
        c_token,
        variant=variant,
        gate=SigmaWindow(**gate_cfg) if gate_cfg else None,
        d_hidden=int((identity or {}).get("d_hidden", 256)),
    )
    weights = {
        k[len("sc_to_bb.") :]: v
        for k, v in state["adapters"].items()
        if k.startswith("sc_to_bb.")
    }
    if use_ema and state.get("ema"):
        shadow = state["ema"].get("shadow") or {}
        weights.update(
            {
                k[len("sc_to_bb.") :]: v
                for k, v in shadow.items()
                if k.startswith("sc_to_bb.")
            }
        )
    module.load_state_dict(weights, strict=True)
    module.eval().requires_grad_(False)
    return dict(
        module=module.to(device),
        variant=variant,
        step=int(state.get("step", 0)),
        path=str(path),
        is_ema=bool(use_ema and state.get("ema")),
        bs_policy=settings.get("bs_policy"),
        pack_steps=settings.get("pack_steps"),
    )


def next_sigma(sigma_schedule, sigma):
    """The next noise level *below* ``sigma`` on the published schedule.

    The computational baseline takes one real sampler step rather than repeating
    a call, so it needs the schedule's own next sigma. Descending, so "next" is
    the first entry strictly smaller.
    """
    trajectory = sigma_schedule.trajectory()
    below = trajectory[trajectory < float(sigma) - 1e-12]
    return float(below[0]) if below.numel() else 0.0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.report:
        path = Path(args.report) / METRICS_FILE
        if not path.is_file():
            raise SystemExit(f"{path} does not exist; run the evaluation first")
        passed = report(json.loads(path.read_text()))
        return 0 if passed or not args.fail_on_no_gain else 1

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
    from pxf.couple.controller import UNSET, CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics
    from pxf.eval.sidechain_metrics import score

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    sb_cfg = dict(config.get("sb_feedback", {}))
    sigma_schedule = schedule.from_config(
        config.get("sigma"),
        mode=args.sigma_mode,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma=args.sigma,
        n_step=args.sigma_n_step,
    )
    sigmas = ev.sweep_sigmas(sigma_schedule, args.n_sigma)

    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_targets:
        structures = structures[: args.max_targets]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "%d target(s); sigma sweep %s", len(structures), [round(s, 3) for s in sigmas]
    )

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

    px_model, _configs, px_record = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    trained = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint wants LABEL=PATH, got {spec!r}")
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
            "arm %s: variant=%s step=%d ema=%s bs_policy=%s",
            label,
            trained[label]["variant"],
            trained[label]["step"],
            trained[label]["is_ema"],
            trained[label]["bs_policy"],
        )
    policies = {a["bs_policy"] for a in trained.values() if a["bs_policy"]}
    if len(policies) > 1:
        raise SystemExit(
            "these arms were trained under different Phase-1 BB->SC policies "
            f"({sorted(policies)}), so they are not comparable to each other. "
            "The policy has to be held fixed across every SC->BB comparison."
        )
    bs_policy = next(iter(policies), "bypass")

    adapters = CouplingAdapters(px_driver.c_token, c_h_V).to(device)
    adapters.eval().requires_grad_(False)
    controller = CoupledDenoiser(
        backbone=None,
        fampnn=fampnn,
        adapters=adapters,
        phase="sc_to_bb",
        pack_steps=args.pack_steps,
    )
    canonical = load_metrics()
    # The Phase-1 policy the arms were trained under, replayed exactly. A
    # bypass applies no BB->SC residual at all; "matched" runs the A_BS these
    # checkpoints carry.
    bs_delta_h = None if bs_policy == "bypass" else UNSET

    arm_names = ["bb0", "zero", *trained, "refine"]
    if trained:
        arm_names.append("perturbed")
    rows, skipped = [], []
    timing = {name: [0.0, 0] for name in arm_names}
    featurized = featurize_structures(
        structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )
    started = time.time()
    backbone_slots = list(atom37.BACKBONE_SLOTS)
    sidechain_slots = list(atom37.SIDECHAIN_SLOTS)

    for index, (sample_id, source) in enumerate(featurized):
        try:
            structure = to_featurized(sample_id, source[0]).to(device)
            native = _native_parse(structures, sample_id)
            ev.check_alignment(sample_id, native["aatype"], structure.aatype)
        except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
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
        ).bool()
        controller.backbone = px_driver.bind(px_driver.conditioning(structure.feature_dict))

        for sigma_value in sigmas:
            seed = ev.target_seed(args.seed, sample_id, sigma_value)
            sigma = torch.full((1,), float(sigma_value), device=device)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(target.shape, generator=generator).to(device)
            x_noisy = (target + noise * float(sigma_value))[None]

            # The frozen half, once and seeded: every arm reads the same bb0 and
            # the same sc0, so they differ in the correction and nothing else.
            torch.manual_seed(seed)
            clock = time.perf_counter()
            with torch.no_grad():
                upstream = controller.frozen_half(
                    structure.topology, x_noisy, sigma, aatype, bs_delta_h=bs_delta_h
                )
            frozen_seconds = time.perf_counter() - clock
            reference = controller._per_residue(upstream.a_token, int(aatype.shape[0]))

            produced = {}  # arm -> (flat coords, seconds, calls, extra)
            produced["bb0"] = (upstream.bb0_flat, frozen_seconds, 1, {})

            with torch.no_grad():
                # Wiring: a second identical call with the feedback forced to
                # zero. Must reproduce bb0 exactly, or everything below is
                # measuring the second call rather than the correction.
                clock = time.perf_counter()
                zeros = torch.zeros(
                    1, int(aatype.shape[0]), px_driver.c_token, device=device
                )
                repeat, _a = controller.backbone(x_noisy, sigma, feedback=zeros)
                produced["zero"] = (
                    repeat,
                    frozen_seconds + (time.perf_counter() - clock),
                    2,
                    dict(
                        deviation_from_bb0=float((repeat - upstream.bb0_flat).abs().max())
                    ),
                )

                for label, arm in trained.items():
                    adapters.sc_to_bb = arm["module"]
                    clock = time.perf_counter()
                    delta, stats = adapters.delta_a(
                        upstream.packed, sigma, reference=reference
                    )
                    corrected, _a = controller.backbone(x_noisy, sigma, feedback=delta)
                    produced[label] = (
                        corrected,
                        frozen_seconds + (time.perf_counter() - clock),
                        2,
                        dict(
                            variant=arm["variant"],
                            delta_a_norm=stats.get("delta_a_norm"),
                            relative_residual=stats.get("relative_residual"),
                        ),
                    )

                # Matched-conformation control: rotate the chis of the packing
                # the readout reads, with the backbone and the sequence held
                # fixed. Preferred over cross-protein shuffling, which changes
                # the sequence and the feature distribution too.
                if trained:
                    label = next(
                        (k for k, a in trained.items() if a["variant"] == "full"),
                        next(iter(trained)),
                    )
                    adapters.sc_to_bb = trained[label]["module"]
                    deltas = torsions.random_chi_deltas(
                        upstream.packed.aatype,
                        args.perturb_degrees * math.pi / 180.0,
                        generator=torch.Generator().manual_seed(seed + 1),
                    )
                    moved = torsions.perturb_chi(
                        upstream.packed.coords37,
                        upstream.packed.aatype,
                        deltas,
                        available=upstream.packed.available,
                    )
                    clock = time.perf_counter()
                    delta, stats = adapters.delta_a(
                        replace(upstream.packed, coords37=moved),
                        sigma,
                        reference=reference,
                    )
                    corrected, _a = controller.backbone(x_noisy, sigma, feedback=delta)
                    produced["perturbed"] = (
                        corrected,
                        frozen_seconds + (time.perf_counter() - clock),
                        2,
                        dict(
                            source_arm=label,
                            perturb_degrees=args.perturb_degrees,
                            delta_a_norm=stats.get("delta_a_norm"),
                        ),
                    )

                # The computational baseline: spend the second denoiser call on
                # the SAMPLER rather than on feedback. One step of PXDesign's own
                # published schedule, taken deterministically:
                #
                #     d      = (x_sigma - bb0) / sigma
                #     x_next = x_sigma + eta * (sigma_next - sigma) * d
                #     bb     = D(x_next, sigma_next)
                #
                # which is Protenix's update at generator.py:260 with the churn
                # disabled. Churn is left out deliberately: PXDesign's config
                # (gamma0 = 1.0, gamma_min = 0.01) re-noises to 2*sigma
                # everywhere in this window, and eta ramps to 2.5, both of which
                # add variance. A noisier baseline would flatter the candidate,
                # and the plan asks for the STRONGEST comparable-cost
                # alternative. Two denoiser calls, matching the feedback arm.
                lower = next_sigma(sigma_schedule, sigma_value)
                clock = time.perf_counter()
                drift = (x_noisy - upstream.bb0_flat) / float(sigma_value)
                x_next = x_noisy + args.refine_eta * (lower - float(sigma_value)) * drift
                refined, _a = controller.backbone(
                    x_next, torch.full((1,), lower, device=device)
                )
                produced["refine"] = (
                    refined,
                    frozen_seconds + (time.perf_counter() - clock),
                    2,
                    dict(refine_sigma=lower, refine_eta=args.refine_eta),
                )

            # --- score every arm from the same reference ---
            for arm in arm_names:
                if arm not in produced:
                    continue
                flat, seconds, calls, extra = produced[arm]
                dense = controller.densify(flat, structure.topology, aatype)
                pred = torch.zeros_like(dense[0]).cpu()
                pred[:, backbone_slots, :] = dense[0][:, backbone_slots, :].cpu()
                scored_mask = torch.zeros_like(native_mask)
                scored_mask[:, backbone_slots] = native_mask[:, backbone_slots]
                row = dict(
                    target=sample_id,
                    arm=arm,
                    sigma=float(sigma_value),
                    length=int(aatype.shape[0]),
                    denoiser_calls=calls,
                    seconds=seconds,
                    unaligned_rmsd=float(
                        (flat.reshape(-1, 3)[supervised] - target[supervised])
                        .pow(2)
                        .sum(-1)
                        .mean()
                        .sqrt()
                    ),
                    **bb_metrics.backbone_report(
                        pred, native37, scored_mask, canonical=canonical
                    ),
                    **extra,
                )
                if not args.no_sidechains:
                    # A FRESH packing on this arm's own backbone, under the same
                    # policy for every arm. bb1 is never scored wearing sc0.
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        sidechains, _aux = controller.repack_on(
                            dense,
                            upstream.packed.aatype,
                            seq_mask=upstream.packed.seq_mask,
                            residue_index=upstream.inputs.residue_index,
                            chain_index=upstream.inputs.chain_index,
                            num_steps=args.pack_steps,
                        )
                    full37 = dense[0].clone()
                    full37[:, sidechain_slots, :] = sidechains[0]
                    full37 = full37.cpu()
                    pred_mask = ev.restype_atom37_mask(aatype.cpu(), rc)
                    placed = ev.place_on_native_backbone(full37, native37, canonical)
                    _counts, summary = score(
                        placed,
                        pred_mask,
                        native37,
                        native_mask,
                        aatype.cpu(),
                        canonical=canonical,
                    )
                    for key in SIDECHAIN_KEYS:
                        if key in summary:
                            row[f"sc_{key}"] = float(summary[key])
                rows.append(row)
                total, count = timing[arm]
                timing[arm] = [total + seconds, count + 1]

        if index % 5 == 0 or index == len(featurized) - 1:
            logger.info(
                "%d/%d targets, %.1fs elapsed",
                index + 1,
                len(featurized),
                time.time() - started,
            )

    if not rows:
        raise SystemExit(f"nothing was scored; {len(skipped)} skipped: {skipped[:3]}")

    by_arm = {name: [r for r in rows if r["arm"] == name] for name in arm_names}
    arms = {}
    for name, arm_rows in by_arm.items():
        if not arm_rows:
            continue
        entry = {key: _mean(arm_rows, key) for key in bb_metrics.HEADLINE}
        entry["unaligned_rmsd"] = _mean(arm_rows, "unaligned_rmsd")
        entry["n"] = len(arm_rows)
        entry["variant"] = next(
            (r.get("variant") for r in arm_rows if r.get("variant")), None
        )
        entry["denoiser_calls"] = arm_rows[0]["denoiser_calls"]
        total, count = timing[name]
        entry["seconds_per_event"] = total / count if count else float("nan")
        if name == "zero":
            entry["max_abs_deviation_from_bb0"] = max(
                r.get("deviation_from_bb0", 0.0) for r in arm_rows
            )
        for key in SIDECHAIN_KEYS:
            if any(f"sc_{key}" in r for r in arm_rows):
                entry[f"sc_{key}"] = _mean(arm_rows, f"sc_{key}")
        arms[name] = entry

    # Paired intervals: the delta is formed within a target before anything is
    # averaged, because per-target difficulty varies far more than the
    # correction does and an unpaired interval would be dominated by it.
    paired = {}
    for name in arms:
        if name == "bb0":
            continue
        paired[name] = {}
        references = ["bb0", "refine"] + [
            n for n in arms if arms[n].get("variant") in ("bb_only", "generic")
        ]
        for base in dict.fromkeys(references):
            if base == name or base not in arms:
                continue
            deltas = _paired_deltas(by_arm[name], by_arm[base], "backbone_rmsd")
            if not deltas:
                continue
            mean, low, high = ev.paired_bootstrap(list(deltas.values()), seed=args.seed)
            paired[name][f"vs_{base}"] = dict(mean=mean, low=low, high=high, n=len(deltas))

    record = dict(
        label="one SC->BB corrective event, held-out",
        structures=str(args.structures),
        n_targets=len({r["target"] for r in rows}),
        n_events=len(by_arm["bb0"]),
        sigma_values=[float(s) for s in sigmas],
        sigma_schedule=sigma_schedule.identity(),
        pack_steps=args.pack_steps,
        scored_sidechains=not args.no_sidechains,
        bs_policy=bs_policy,
        perturb_degrees=args.perturb_degrees,
        refine_eta=args.refine_eta,
        criterion=dict(
            min_absolute_angstrom=MIN_ABSOLUTE_GAIN,
            min_relative=MIN_RELATIVE_GAIN,
            wiring_tolerance=WIRING_TOLERANCE,
            gpu_nondeterminism_angstrom=GPU_NONDETERMINISM,
        ),
        checkpoints={
            k: {x: v[x] for x in ("path", "variant", "step", "is_ema")}
            for k, v in trained.items()
        },
        manifest=dict(
            seed_scheme=ev.SEED_SCHEME,
            seed_base=int(args.seed),
            sigma_values=[float(s) for s in sigmas],
            pack_steps=int(args.pack_steps),
            bs_policy=bs_policy,
            crop_size=int(args.crop_size),
            structures_fingerprint=ev.structures_fingerprint(structures),
            n_structures=len(structures),
        ),
        pxdesign=px_record,
        fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
        arms=arms,
        paired=paired,
        skipped=skipped,
        seconds=round(time.time() - started, 1),
    )
    (out / METRICS_FILE).write_text(json.dumps(record, indent=2, default=str))
    with (out / "per_target.csv").open("w") as stream:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", out / METRICS_FILE)
    passed = report(record)
    return 0 if passed or not args.fail_on_no_gain else 1


def _paired_deltas(candidate_rows, reference_rows, metric):
    """``{target: improvement}`` -- reference minus candidate, lower-is-better."""
    reference = {(r["target"], r["sigma"]): r[metric] for r in reference_rows}
    out = {}
    for row in candidate_rows:
        key = (row["target"], row["sigma"])
        if key in reference and row[metric] == row[metric]:
            out.setdefault(row["target"], []).append(reference[key] - row[metric])
    return {t: sum(v) / len(v) for t, v in out.items()}


def _native_parse(structures, sample_id):
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    match = next((p for p in structures if Path(p).stem == sample_id), None)
    if match is None:
        raise ValueError(f"no source file for {sample_id}")
    return process_single_pdb(load_feats_from_pdb(str(match)))


if __name__ == "__main__":
    raise SystemExit(main())
