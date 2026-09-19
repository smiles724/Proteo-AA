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
# Which arms genuinely need the 50-step packing and the re-encode, i.e. what
# each would cost *deployed*:
#
#   full, perturbed  read h_packed, the re-encoding of bb0 + sc0. Need both.
#   bb_only          reads h_base, the side-chain-masked encoding, which
#                    `propose` already produces alongside bb0. Needs neither.
#   generic          reads nothing at all. Needs neither.
#   bb0, zero, refine  no readout.
#
# Note this is the deployed cost, not the wall-clock of the arm as run: all
# three variants share one code path -- deliberately, so their parameter counts
# match exactly -- so the bb_only and generic arms as executed do compute a
# packing and then zero the features derived from it. Charging them for that
# would price an artefact of the parameter-matching rather than the method.
NEEDS_PACKING = ("full", "perturbed")


def needs_packing(variant):
    """Does a readout of this variant need the packing rollout, deployed?

    Keyed on the *variant* the checkpoint records rather than on the arm's
    label, so an arm named anything at all is priced by what it reads.
    """
    return variant in NEEDS_PACKING


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
    candidates = [name for name in arms if arms[name].get("variant") == "full"]
    if not candidates:
        return False, ["no arm records variant='full', so there is no candidate"]
    candidate = candidates[0]
    if len(candidates) > 1:
        # Two architectures in one run have two different same-architecture
        # controls, so one verdict cannot speak for both.
        lines.append(
            f"{len(candidates)} arms record variant='full' ({', '.join(candidates)}); "
            f"the verdict below is for {candidate} only. Score each architecture "
            "in its own run, against its own BB-only control"
        )

    # The wiring check gates everything: without it the comparison is not
    # between bb0 and a correction, it is between two different calls.
    wiring = arms.get("zero", {}).get("max_abs_deviation_from_bb0")
    if wiring is None:
        lines.append("WIRING: not measured -- the zero-feedback arm did not run")
        wired = False
    else:
        wired = wiring <= WIRING_TOLERANCE
        site = arms.get("zero", {}).get("zero_site") or "decoder"
        lines.append(
            f"WIRING: zero feedback at the {site} site deviates from bb0 by "
            f"{wiring:.2e} A, tolerance {WIRING_TOLERANCE:.0e} "
            f"({'ok' if wired else 'FAILED'}). GPU non-determinism alone "
            f"measures ~{GPU_NONDETERMINISM:.1e} A"
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
    print(
        f"  {'arm':14s} {'arch':9s} {'variant':10s} "
        + " ".join(f"{k:>14s}" for k in keys)
    )
    for name, arm in record["arms"].items():
        cells = " ".join(
            f"{arm[k]:14.4f}" if isinstance(arm.get(k), float) else f"{'-':>14s}"
            for k in keys
        )
        print(
            f"  {name:14s} {str(arm.get('arch') or '-'):9s} "
            f"{str(arm.get('variant') or '-'):10s} {cells}"
        )
    print()
    print(
        f"  {'arm':14s} {'calls':>6s} {'packing':>8s} {'seconds/event':>14s} {'vs bb0':>9s}"
    )
    baseline_seconds = record["arms"]["bb0"].get("seconds_per_event") or float("nan")
    for name, arm in record["arms"].items():
        seconds = arm.get("seconds_per_event", float("nan"))
        print(
            f"  {name:14s} {arm.get('denoiser_calls', 0):6d} "
            f"{'yes' if arm.get('needs_packing') else 'no':>8s} {seconds:14.3f} "
            f"{seconds / baseline_seconds:8.2f}x"
        )
    print(
        "\n  Deployed cost. Only the arms reading h_packed need the packing\n"
        "  rollout and the re-encode; bb_only reads h_base, which comes free\n"
        "  with bb0, and generic reads nothing. Equal denoiser-call counts do\n"
        "  not mean equal cost. (All variants share one code path so their\n"
        "  parameter counts match, so the controls as RUN do compute a packing\n"
        "  and discard it; that is an artefact of the matching, not the method.)\n"
    )
    for entry in record.get("per_sigma") or []:
        base = (entry["arms"].get("bb0") or {}).get("backbone_rmsd")
        head = f"  [sigma_B = {entry['sigma']:.3f} A]"
        print(f"{head}   bb0 backbone RMSD {base:.4f} A" if base else head)
        print(
            f"    {'arm':12s} {'BB RMSD':>9s} {'vs bb0':>10s} {'95% CI':>20s} "
            f"{'|delta_a|':>10s}"
        )
        for name, arm in entry["arms"].items():
            against = (entry["paired"].get(name) or {}).get("vs_bb0")
            if against:
                flag = "*" if against["low"] > 0 else " "
                delta = f"{against['mean']:+.4f}{flag}"
                interval = f"[{against['low']:+.4f}, {against['high']:+.4f}]"
            else:
                delta, interval = "-", ""
            norm = arm.get("delta_a_norm")
            norm_cell = f"{norm:10.4f}" if norm is not None else f"{'-':>10s}"
            print(
                f"    {name:12s} {arm['backbone_rmsd']:9.4f} {delta:>10s} "
                f"{interval:>20s} {norm_cell}"
            )
        print()
    print("  * = the paired interval excludes zero. Read these rows, not just the")
    print("  pooled table: the proposal's own difficulty varies ~8x across the")
    print("  sweep, so pooling averages over regimes that behave differently.\n")
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


def load_arm(path, *, c_h_V, c_token, c_s, c_z, sb_cfg, use_ema, device, expect=None):
    """One trained SC->BB arm, rebuilt as the architecture it was trained as.

    The architecture comes from the checkpoint, never from this script's config:
    E1's ``full`` and ``bb_only`` are the same shapes with different groups
    zeroed, and the late and early architectures both hang off the same
    ``sc_to_bb.`` prefix, so guessing would produce a clean load of the wrong
    thing.

    Which is also why the metadata cannot be cross-checked against the weights
    here -- it is the only record of which arm produced them. What is checked is
    that the record names a real arm, and that it is the arm the caller's label
    claims. ``--checkpoint early_s_full=<the control's checkpoint>`` is a
    command-line slip that produces a fully self-consistent run with the wrong
    names on the table, and ``expect`` is what catches it.
    """
    from pxf.couple.conditioning import (
        AtomConditioner,
        EarlySingleConditioner,
        check_is_the_expected_arm,
    )
    from pxf.couple.readout import FeedbackPath, SigmaWindow

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{path} is not a coupling checkpoint")
    identity = ((state.get("controller") or {}).get("adapters") or {}).get("sc_to_bb")
    identity = identity if isinstance(identity, dict) else {}
    variant = identity.get("variant") or sb_cfg.get("variant", "full")
    arch = identity.get("arch", "late")
    gate_cfg = (identity.get("gate") or {}) or dict(sb_cfg.get("gate") or {})
    gate_cfg = {k: v for k, v in gate_cfg.items() if k != "kind"}
    gate = SigmaWindow(**gate_cfg) if gate_cfg else None
    settings = state.get("settings") or {}
    d_hidden = int(identity.get("d_hidden", 256))
    # Before building anything: the record has to name an arm, and the arm the
    # caller's label claims. A constructor would also reject an impossible
    # variant, but it would name whichever field it happened to read first
    # rather than the actual problem.
    arm_name = check_is_the_expected_arm(identity, expect, path=path) if identity else None
    if arch == "late":
        module = FeedbackPath(
            c_h_V, c_token, variant=variant, gate=gate, d_hidden=d_hidden
        )
    elif arch == "early_s":
        module = EarlySingleConditioner(
            c_h_V, c_s, variant=variant, gate=gate, d_hidden=d_hidden
        )
    elif arch == "atom":
        module = AtomConditioner(
            c_s,
            c_z,
            variant=variant,
            pair=bool(identity.get("pair", True)),
            gate=gate,
            d_hidden=d_hidden,
        )
    else:
        raise SystemExit(f"{path} records unknown SC->BB architecture {arch!r}")
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
        arm=arm_name,
        variant=variant,
        arch=arch,
        pair=bool(identity.get("pair", False)),
        step=int(state.get("step", 0)),
        path=str(path),
        is_ema=bool(use_ema and state.get("ema")),
        bs_policy=settings.get("bs_policy"),
        pack_steps=settings.get("pack_steps"),
    )


def zero_payload(trained, *, length, c_token, c_s, c_z, device):
    """The wiring arm's feedback: exactly zero, at the site the arms inject into.

    The check is only worth anything at the *same* site. A zero ``a_token``
    residual would prove the decoder hook is harmless while saying nothing about
    a conditioning hook that is the thing actually being used, so the payload
    follows the architecture under test.
    """
    from pxf.couple.pxdesign_iface import ConditioningFeedback

    early = [arm for arm in trained.values() if arm["arch"] != "late"]
    if not early:
        return torch.zeros(1, int(length), int(c_token), device=device), "decoder"
    pair = any(arm["pair"] for arm in early)
    return (
        ConditioningFeedback(
            delta_single=torch.zeros(1, int(length), int(c_s), device=device),
            delta_pair=(
                torch.zeros(int(length), int(length), int(c_z), device=device)
                if pair
                else None
            ),
        ),
        "conditioning",
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
            c_s=px_driver.c_s,
            c_z=px_driver.c_z,
            sb_cfg=sb_cfg,
            use_ema=args.ema,
            device=device,
            # The label is the caller's claim about the file. When it names a
            # known arm, hold the file to it.
            expect=label,
        )
        logger.info(
            "arm %s: arch=%s variant=%s pair=%s step=%d ema=%s bs_policy=%s",
            label,
            trained[label]["arch"],
            trained[label]["variant"],
            trained[label]["pair"],
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
            stages = upstream.timings
            # bb0 costs one denoise. The feedback arms additionally need the
            # packing rollout and the re-encode; the BB-only alternatives do not.
            denoise_seconds = stages.get("denoise", frozen_seconds)
            packing_seconds = stages.get("pack", 0.0) + stages.get("reencode", 0.0)
            reference = controller._per_residue(upstream.a_token, int(aatype.shape[0]))

            produced = {}  # arm -> (flat coords, seconds, calls, extra)
            produced["bb0"] = (upstream.bb0_flat, denoise_seconds, 1, {})

            with torch.no_grad():
                # Wiring: a second identical call with the feedback forced to
                # zero. Must reproduce bb0 exactly, or everything below is
                # measuring the second call rather than the correction.
                clock = time.perf_counter()
                zeros, zero_site = zero_payload(
                    trained,
                    length=int(aatype.shape[0]),
                    c_token=px_driver.c_token,
                    c_s=px_driver.c_s,
                    c_z=px_driver.c_z,
                    device=device,
                )
                repeat, _a = controller.backbone(x_noisy, sigma, feedback=zeros)
                produced["zero"] = (
                    repeat,
                    denoise_seconds + (time.perf_counter() - clock),
                    2,
                    dict(
                        deviation_from_bb0=float(
                            (repeat - upstream.bb0_flat).abs().max()
                        ),
                        zero_site=zero_site,
                    ),
                )

                for label, arm in trained.items():
                    adapters.sc_to_bb = arm["module"]
                    clock = time.perf_counter()
                    delta, stats = adapters.delta_a(
                        upstream.packed, sigma, reference=reference
                    )
                    corrected, _a = controller.backbone(x_noisy, sigma, feedback=delta)
                    # One decision drives both the charge and the flag.
                    needs = needs_packing(arm["variant"])
                    produced[label] = (
                        corrected,
                        denoise_seconds
                        + (packing_seconds if needs else 0.0)
                        + (time.perf_counter() - clock),
                        2,
                        dict(
                            variant=arm["variant"],
                            arch=arm["arch"],
                            needs_packing=needs,
                            delta_a_norm=stats.get("delta_a_norm"),
                            delta_s_norm=stats.get("delta_s_norm"),
                            delta_z_norm=stats.get("delta_z_norm"),
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
                        denoise_seconds + packing_seconds + (time.perf_counter() - clock),
                        2,
                        dict(
                            needs_packing=True,
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
                    denoise_seconds + (time.perf_counter() - clock),
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
        entry["arch"] = next((r.get("arch") for r in arm_rows if r.get("arch")), None)
        if name == "zero":
            entry["zero_site"] = next(
                (r.get("zero_site") for r in arm_rows if r.get("zero_site")), None
            )
        entry["denoiser_calls"] = arm_rows[0]["denoiser_calls"]
        # Read off the rows rather than recomputed from the arm name: the flag
        # and the seconds it explains have to come from one decision, or the
        # table can say "no packing" in one column while charging for it in the
        # next -- which is exactly what happened when they were separate.
        entry["needs_packing"] = bool(arm_rows[0].get("needs_packing", False))
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

    # Per sigma as well as pooled. Not optional reporting: the proposal's own
    # difficulty varies ~8x across this sweep, so a pooled number averages over
    # regimes that behave differently -- and an arm that helps at low noise and
    # not at high noise is a real outcome that pooling hides entirely.
    def at(name, value):
        return [r for r in by_arm.get(name, []) if abs(r["sigma"] - value) < 1e-9]

    per_sigma = []
    for value in sigmas:
        entry = dict(sigma=float(value), arms={}, paired={})
        for name in arm_names:
            rows_at = at(name, value)
            if not rows_at:
                continue
            summary = {key: _mean(rows_at, key) for key in bb_metrics.HEADLINE}
            summary["n"] = len(rows_at)
            for key in (
                "delta_a_norm",
                "delta_s_norm",
                "delta_z_norm",
                "relative_residual",
            ):
                if any(key in r for r in rows_at):
                    summary[key] = _mean(rows_at, key)
            entry["arms"][name] = summary
        for name in entry["arms"]:
            if name == "bb0":
                continue
            entry["paired"][name] = {}
            for base in dict.fromkeys(["bb0", "refine", "bb_only", "generic"]):
                if base == name or base not in entry["arms"]:
                    continue
                deltas = _paired_deltas(at(name, value), at(base, value), "backbone_rmsd")
                if not deltas:
                    continue
                mean, low, high = ev.paired_bootstrap(list(deltas.values()), seed=args.seed)
                entry["paired"][name][f"vs_{base}"] = dict(
                    mean=mean, low=low, high=high, n=len(deltas)
                )
        per_sigma.append(entry)

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
            k: {
                **{
                    x: v[x]
                    for x in ("path", "arm", "variant", "arch", "pair", "step", "is_ema")
                },
                "identity": v["module"].identity(),
            }
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
