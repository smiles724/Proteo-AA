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
    p.add_argument(
        "--candidate",
        default=None,
        help="which arm label the verdict is for. Required when more than one "
        "arm could be the candidate -- E2's atom_sz_full and atom_s_full are "
        "both variant='full', so picking one by position would make the verdict "
        "depend on the order the --checkpoint flags were typed",
    )
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


# The prespecified no-regression checks, as (metric, absolute tolerance). A
# correction that buys backbone accuracy by wrecking side-chain chemistry has
# not bought anything, so these are part of the criterion rather than context
# printed underneath it. Tolerances are absolute on the sc0 -> sc1 change and
# are deliberately loose: the question is "no MATERIAL regression", and a tight
# bound would fail on packing-sampler noise.
SC_REGRESSION_LIMITS = (
    ("symmetry_rmsd", 0.05),
    ("bad_bond_fraction", 0.005),
    ("rotamer_outlier_fraction_40deg", 0.02),
)

# Verdicts. `incomplete` is distinct from `fail` on purpose: a run that did not
# produce the evidence has not disproved anything, and collapsing the two into
# a boolean is how a missing control becomes a pass.
PASS, FAIL, INCOMPLETE = "pass", "do not proceed", "incomplete"


def same_architecture_controls(arms, candidate):
    """The controls that can speak for ``candidate``: its own architecture only.

    An E1 BB-only arm is not a control for an E2 candidate. It differs in
    representation, capacity and cost, so beating it would say nothing about
    whether *this* architecture's side-chain input earned anything. When both
    experiments are scored in one run, selecting the best-looking BB-only arm
    across architectures is exactly the comparison the criterion forbids.
    """
    arch = arms[candidate].get("arch")
    return [
        name
        for name, arm in arms.items()
        if name != candidate
        and arm.get("arch") == arch
        and arm.get("variant") in ("bb_only", "generic")
    ]


def candidate_relation(arms, a, b):
    """What comparing two candidates is a comparison OF.

    Labelled rather than left to the reader: "atom_sz_full vs atom_s_full" and
    "early_s_full vs late_full" are both full-versus-full, and they answer
    completely different questions -- whether the pair branch earns its cost,
    and whether the injection site matters.
    """
    if arms.get(a, {}).get("arch") != arms.get(b, {}).get("arch"):
        return "injection site"
    if arms.get(a, {}).get("pair") != arms.get(b, {}).get("pair"):
        return "pair branch"
    return "candidates"


def comparison_plan(arms):
    """``(candidates, {candidate: [reference, ...]})`` -- stated, not inferred.

    Built once and used for the pooled intervals, the per-sigma intervals and
    the verdict alike, so the three cannot disagree about which comparisons the
    experiment was supposed to make. Inferring them from variant names was how
    the per-sigma table came to omit every control whose arm was not literally
    named ``bb_only``.
    """
    candidates = [
        name
        for name, arm in arms.items()
        if arm.get("variant") == "full" and name not in ("bb0", "zero", "refine")
    ]
    plan = {}
    for name in candidates:
        references = ["bb0", "refine", *same_architecture_controls(arms, name)]
        # Every OTHER candidate, whatever its architecture. Two relationships
        # hide here and neither is reachable by variant name, because both arms
        # are "full":
        #   same arch, different pair -> the pair branch's own ablation
        #   different arch            -> the injection-site comparison
        # The second is the whole reason to score two architectures on one
        # panel: without it the run yields two independent tables to compare by
        # eye, when what is wanted is a per-target PAIRED interval between the
        # sites. Restricting this to the same architecture silently produced
        # exactly that, while the surrounding prose claimed otherwise.
        references += [other for other in candidates if other != name]
        # The matched-conformation control, when it was produced for this arm.
        if arms.get("perturbed", {}).get("source_arm") == name:
            references.append("perturbed")
        plan[name] = [r for r in dict.fromkeys(references) if r in arms]
    return candidates, plan


def pairings(arms):
    """``{arm: [reference, ...]}`` -- every paired interval the run should form.

    A superset of :func:`comparison_plan`: the candidates' planned references,
    plus every other arm against ``bb0`` so the table can still report what each
    one did on its own. One function, so the pooled intervals, the per-sigma
    intervals and the verdict cannot disagree about what was compared.
    """
    _candidates, plan = comparison_plan(arms)
    out = {}
    for name in arms:
        if name == "bb0":
            continue
        references = ["bb0", *plan.get(name, [])]
        if arms.get(name, {}).get("variant") in ("bb_only", "generic"):
            references.append("refine")
        if name == "perturbed" and arms[name].get("source_arm"):
            # The direction the mechanism question is asked in: how much of the
            # candidate's gain survives re-encoding a perturbed packing.
            references.append(arms[name]["source_arm"])
        out[name] = [r for r in dict.fromkeys(references) if r in arms and r != name]
    return out


def verdict(record, candidate=None):
    """The exit decision and its reasons. Returns ``(outcome, lines)``.

    Every condition below is binding. An earlier version printed the missing
    ones and returned True anyway: a synthetic record with no trained control at
    all and a bad-bond fraction going 0.002 -> 0.500 passed, because only the
    RMSD arithmetic touched the return value. Anything the criterion names and
    the run did not measure now produces ``incomplete``.
    """
    arms = record["arms"]
    lines = []
    paired = record.get("paired", {})
    candidates, plan = comparison_plan(arms)
    if not candidates:
        return INCOMPLETE, ["no arm records variant='full', so there is no candidate"]
    if candidate is None:
        if len(candidates) > 1:
            return INCOMPLETE, [
                f"{len(candidates)} arms could be the candidate "
                f"({', '.join(candidates)}) and nothing in the record says which "
                "was prespecified. Pass --candidate, or score one architecture "
                "per run: each needs its own same-architecture BB-only control"
            ]
        candidate = candidates[0]
    if candidate not in arms:
        return INCOMPLETE, [f"--candidate {candidate!r} did not run"]
    lines.append(
        f"candidate: {candidate} "
        f"(arch={arms[candidate].get('arch')}, pair={arms[candidate].get('pair')})"
    )

    failures, missing = [], []

    # The wiring check gates everything: without it the comparison is not
    # between bb0 and a correction, it is between two different calls.
    wiring = arms.get("zero", {}).get("max_abs_deviation_from_bb0")
    if wiring is None:
        missing.append("the zero-feedback wiring arm did not run")
    else:
        site = arms.get("zero", {}).get("zero_site") or "decoder"
        lines.append(
            f"WIRING: zero feedback at the {site} site deviates from bb0 by "
            f"{wiring:.2e} A, tolerance {WIRING_TOLERANCE:.0e} "
            f"({'ok' if wiring <= WIRING_TOLERANCE else 'FAILED'}). GPU "
            f"non-determinism alone measures ~{GPU_NONDETERMINISM:.1e} A"
        )
        if wiring > WIRING_TOLERANCE:
            failures.append("the zero-feedback arm does not reproduce bb0")

    proposal = arms["bb0"]["backbone_rmsd"]
    got = arms[candidate]["backbone_rmsd"]
    lines.append(f"bb0 backbone RMSD {proposal:.4f} A")
    lines.append(f"{candidate} backbone RMSD {got:.4f} A  (delta {proposal - got:+.4f})")

    # 1. A correction happened at all.
    interval = paired.get(candidate, {}).get("vs_bb0")
    if not interval:
        missing.append("no paired interval against bb0")
    else:
        lines.append(
            f"paired improvement over bb0: {interval['mean']:+.4f} A "
            f"[{interval['low']:+.4f}, {interval['high']:+.4f}] (n={interval['n']})"
        )
        if not interval["low"] > 0:
            failures.append("the interval against bb0 includes zero")

    # 2. A same-architecture BB-only control ran, and the candidate beats the
    #    strongest comparable-cost alternative by the threshold.
    controls = same_architecture_controls(arms, candidate)
    if not controls:
        missing.append(
            f"no trained BB-only or generic control of architecture "
            f"{arms[candidate].get('arch')!r} ran. Beating bb0 shows a correction "
            "happened; a SIDE-CHAIN-specific claim needs the matched control"
        )
    alternatives = [
        (name, arms[name]["backbone_rmsd"])
        for name in ["refine", *controls]
        if name in arms and arms[name].get("backbone_rmsd") is not None
    ]
    if not alternatives:
        missing.append("no comparable-cost alternative ran")
    else:
        baseline_name, baseline = min(alternatives, key=lambda row: row[1])
        need = required_gain(baseline)
        gain = baseline - got
        lines.append(
            f"strongest comparable-cost alternative: {baseline_name} "
            f"{baseline:.4f} A; gain {gain:+.4f} A, need >= {need:.4f}"
        )
        if gain < need:
            failures.append(f"gain over {baseline_name} is below the criterion")
        against = paired.get(candidate, {}).get(f"vs_{baseline_name}")
        if not against:
            missing.append(f"no paired interval against {baseline_name}")
        elif not against["low"] > 0:
            failures.append(
                f"the paired interval against {baseline_name} includes zero "
                f"({against['mean']:+.4f} [{against['low']:+.4f}, {against['high']:+.4f}])"
            )

    # 3. No material side-chain or chemistry regression. Part of the criterion,
    #    not a footnote: buying backbone accuracy with broken geometry is not a
    #    gain, and printing the numbers without testing them let a bad-bond
    #    fraction of 0.5 through.
    for key, tolerance in SC_REGRESSION_LIMITS:
        before = arms["bb0"].get(f"sc_{key}")
        after = arms[candidate].get(f"sc_{key}")
        if before is None or after is None:
            missing.append(
                f"side-chain metric {key} was not scored (--no-sidechains forfeits "
                "the no-regression half of the criterion)"
            )
            continue
        change = after - before
        lines.append(
            f"side chains, {key}: {before:.4f} -> {after:.4f} "
            f"({change:+.4f}, tolerance +{tolerance:g})"
        )
        if change > tolerance:
            failures.append(f"{key} regressed by {change:+.4f}")

    # Reported, never scored: dependence on the matched conformation is
    # diagnostic of the mechanism, and a threshold on it is not part of the
    # criterion.
    perturbed = arms.get("perturbed", {}).get("backbone_rmsd")
    if perturbed is not None:
        against = paired.get("perturbed", {}).get(f"vs_{candidate}")
        detail = ""
        if against:
            detail = (
                f", paired {against['mean']:+.4f} "
                f"[{against['low']:+.4f}, {against['high']:+.4f}]"
            )
        lines.append(
            f"perturbed-SC arm {perturbed:.4f} A: the candidate keeps "
            f"{got - perturbed:+.4f} A of its gain when the packing it reads is "
            f"re-encoded after a rotamer perturbation{detail} (diagnostic, not "
            "part of the criterion)"
        )
    for other in plan.get(candidate, []):
        if arms.get(other, {}).get("variant") != "full":
            continue
        against = paired.get(candidate, {}).get(f"vs_{other}")
        if against:
            lines.append(
                f"{candidate_relation(arms, candidate, other)}, {candidate} vs "
                f"{other}: {against['mean']:+.4f} A "
                f"[{against['low']:+.4f}, {against['high']:+.4f}]"
            )

    for line in missing:
        lines.append(f"  MISSING: {line}")
    for line in failures:
        lines.append(f"  FAILED: {line}")
    if missing:
        return INCOMPLETE, lines
    return (FAIL if failures else PASS), lines


def report(record, candidate=None):
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
    outcome, lines = verdict(record, candidate=candidate)
    for line in lines:
        print(f"  {line}")
    print(f"\n  VERDICT: {outcome}")
    if outcome == INCOMPLETE:
        print(
            "  'incomplete' is not 'do not proceed': the run did not produce\n"
            "  evidence the criterion requires, so it has neither shown nor\n"
            "  disproved anything. Supply the missing arms and re-score.\n"
        )
    print(
        "  Passing licenses inserting the corrected estimate into the sampler's\n"
        "  normal update and testing one event in a full rollout. It does not\n"
        "  license the two-event training stage, which is a separate experiment.\n"
    )
    return outcome


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
        check_feature_schema,
        check_is_the_expected_arm,
        overridden_settings,
        reconstruct_kwargs,
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
    # Before building anything: the record has to name an arm, and the arm the
    # caller's label claims. A constructor would also reject an impossible
    # variant, but it would name whichever field it happened to read first
    # rather than the actual problem.
    arm_name = check_is_the_expected_arm(identity, expect, path=path) if identity else None
    # Rebuild the function that was TRAINED, not the one today's constants
    # describe. Every setting the checkpoint records and the constructor accepts
    # is taken from the checkpoint; anything left over is compared below.
    built = reconstruct_kwargs(identity)
    for name, value, default in overridden_settings(identity):
        logger.warning(
            "%s was trained with %s=%r; this build defaults to %r. Using the "
            "checkpoint's value -- and note this arm is not comparable to one "
            "trained on the default",
            path,
            name,
            value,
            default,
        )
    if arch == "late":
        module = FeedbackPath(c_h_V, c_token, variant=variant, gate=gate, **built)
    elif arch == "early_s":
        module = EarlySingleConditioner(c_h_V, c_s, variant=variant, gate=gate, **built)
    elif arch == "atom":
        built.pop("sequence_width", None)  # E2 has its own embedding, not the readout's
        module = AtomConditioner(
            c_s, c_z, variant=variant, pair=bool(identity.get("pair", True)),
            gate=gate, **built,
        )
    else:
        raise SystemExit(f"{path} records unknown SC->BB architecture {arch!r}")
    if identity:
        # The one comparison with an independent second source of truth: what
        # the checkpoint says its features were, against what this code computes.
        check_feature_schema(identity, module.identity(), path=path)
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


def check_arms_comparable(trained):
    """Refuse a comparison between arms trained under different settings.

    Each arm can load correctly and the set still not be an experiment: one
    trained with 16 neighbours against one trained with 32 differ in receptive
    field and capacity, so the delta between them is not attributable to the
    information. Nothing in either checkpoint alone can see this.
    """
    from pxf.couple.conditioning import comparability

    identities = {
        label: arm["module"].identity()
        for label, arm in trained.items()
        if hasattr(arm["module"], "identity")
    }
    if len(identities) < 2:
        return
    differences = comparability(identities)
    if differences:
        raise SystemExit(
            "these arms were trained under different feature settings, so a "
            "difference between them is not attributable to the information "
            "they read: "
            + "; ".join(f"{key}={values}" for key, values in differences)
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
        outcome = report(json.loads(path.read_text()), candidate=args.candidate)
        return 0 if outcome == PASS or not args.fail_on_no_gain else 1

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
    check_arms_comparable(trained)
    from pxf.couple.readout import needs_sequence_controls

    needs_controls = any(
        needs_sequence_controls(arm["variant"]) for arm in trained.values()
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
    # Which arm the matched-conformation control perturbs. Stated once here
    # rather than re-derived per event as "the first arm whose variant is full",
    # which made it depend on the order the --checkpoint flags were typed as
    # soon as two arms shared that variant (atom_sz_full and atom_s_full do).
    perturb_source = None
    if trained:
        if args.candidate and args.candidate in trained:
            perturb_source = args.candidate
        else:
            full_arms = [k for k, a in trained.items() if a["variant"] == "full"]
            if len(full_arms) > 1:
                raise SystemExit(
                    f"{len(full_arms)} arms are variant='full' ({full_arms}); pass "
                    "--candidate to say which one the perturbed control should "
                    "perturb, or the answer depends on flag order"
                )
            perturb_source = full_arms[0] if full_arms else next(iter(trained))
        arm_names.append("perturbed")
        logger.info("matched-conformation control perturbs %s", perturb_source)
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
            if needs_controls:
                upstream = replace(
                    upstream,
                    packed=controller.encode_sequence_controls(
                        upstream.inputs, upstream.packed
                    ),
                )
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
                            pair=arm["pair"],
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
                    label = perturb_source
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
                    # RE-ENCODE, rather than substituting coords37 into the
                    # existing packed state. h_packed is FaMPNN's node readout
                    # for the structure it was computed on, and an arm that
                    # reads it -- every `late_*` and `early_s_*` variant -- would
                    # otherwise be handed the ORIGINAL node features alongside
                    # perturbed chi and environment features. That is a partial
                    # intervention reported as a full one, and it understates
                    # the response in the direction that flatters the arm.
                    # pxf/couple/probes.py and scripts/audit_sb_sensitivity.py
                    # already did it this way; this path did not.
                    perturbed_packed = controller.encode_predicted_packing(
                        upstream.inputs,
                        moved[..., sidechain_slots, :],
                        h_base=upstream.packed.h_base,
                        # Carried over unchanged: this is a GEOMETRY-only
                        # intervention. The packer is not re-run, so there is no
                        # new psCE to report, and inventing one would make the
                        # arm test two things at once.
                        psce=upstream.packed.psce,
                    )
                    delta, stats = adapters.delta_a(
                        perturbed_packed, sigma, reference=reference
                    )
                    corrected, _a = controller.backbone(x_noisy, sigma, feedback=delta)
                    produced["perturbed"] = (
                        corrected,
                        denoise_seconds + packing_seconds + (time.perf_counter() - clock),
                        2,
                        dict(
                            needs_packing=True,
                            source_arm=label,
                            arch=trained[label]["arch"],
                            perturbed_psce="carried over (geometry-only)",
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
        entry["pair"] = next(
            (r.get("pair") for r in arm_rows if r.get("pair") is not None), None
        )
        entry["source_arm"] = next(
            (r.get("source_arm") for r in arm_rows if r.get("source_arm")), None
        )
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
            # The fields comparison_plan identifies candidates and controls by.
            # Without them every per-sigma entry looks like an unlabelled arm,
            # comparison_plan finds no candidate, and the table collapses to
            # "vs bb0" -- which is exactly what it did: the per-sigma view lost
            # the full-versus-control comparison, the one the whole experiment
            # turns on, while the pooled table still had it.
            for field in ("variant", "arch", "pair", "source_arm"):
                summary[field] = arms.get(name, {}).get(field)
            for key in (
                "delta_a_norm",
                "delta_s_norm",
                "delta_z_norm",
                "relative_residual",
            ):
                if any(key in r for r in rows_at):
                    summary[key] = _mean(rows_at, key)
            entry["arms"][name] = summary
        # The same plan the pooled table and the verdict use. Hard-coding the
        # reference NAMES here silently dropped every control of this
        # experiment: an arm labelled `early_s_bb_only` is not called
        # `bb_only`, so `base not in entry["arms"]` skipped it and the per-sigma
        # table showed the candidate against bb0 and nothing else.
        for name, references in pairings(entry["arms"]).items():
            entry["paired"][name] = {}
            for base in references:
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
    for name, references in pairings(arms).items():
        paired[name] = {}
        for base in references:
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
        # Computed just above -- one bootstrap per (sigma, arm, reference) --
        # and, until now, dropped on the floor: never written here and so never
        # printed by report(), which has always had the code to display it. The
        # per-sigma breakdown is where the late pilot's two sharpest findings
        # came from (the gain lives where there is least to gain; SC-specificity
        # inverts at one noise level), and both had to be recovered from
        # per_target.csv by hand because the tool silently withheld them.
        per_sigma=per_sigma,
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
    outcome = report(record, candidate=args.candidate)
    return 0 if outcome == PASS or not args.fail_on_no_gain else 1


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
