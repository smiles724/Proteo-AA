"""The exit criterion, as a function.

``verdict`` is where the roadmap's threshold is enforced, and it is the one
place a plausible-looking pilot could be declared a success it did not earn. So
it is tested against synthetic records rather than only exercised on real ones:
the cases that must fail are cheap to construct and expensive to notice.

Expensive to notice was not hypothetical. An earlier version printed the
conditions it did not enforce and returned success anyway, and four defects
reached this file's own coverage:

1. A record with NO trained control and a bad-bond fraction going 0.002 -> 0.500
   passed. Only the RMSD arithmetic touched the return value; the controls and
   the chemistry were printed underneath it. ``test_missing_trained_controls``
   below used to *assert* that pass.
2. A missing paired interval passed, because the code that would have failed it
   sat inside ``if interval:``.
3. The per-sigma table compared against arms literally named ``bb_only`` and
   ``generic``, so every control of the early-conditioning experiments --
   ``early_s_bb_only``, ``atom_sz_bb_only`` -- was silently absent from it.
4. Two arms can both be ``variant="full"`` (``atom_sz_full`` and
   ``atom_s_full``), so which one the verdict judged depended on the order the
   ``--checkpoint`` flags were typed.

Hence three outcomes rather than a boolean. ``incomplete`` is not
``do not proceed``: a run that never produced the evidence has neither shown nor
disproved anything, and collapsing the two is how a missing control becomes a
pass.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_sb_feedback", REPO / "scripts" / "eval_sb_feedback.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_sb_feedback"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


# Side-chain levels that do not regress. Present on every arm of the base
# record, because "no material regression" is half the criterion and a record
# that omits it is incomplete rather than passing by default.
CLEAN_SC = dict(
    sc_symmetry_rmsd=1.44,
    sc_bad_bond_fraction=0.002,
    sc_rotamer_outlier_fraction_40deg=0.33,
)


def record(**overrides):
    """A passing record: real gain over bb0 and over the best alternative.

    Every arm carries an ``arch``. The controls must be the candidate's OWN
    architecture -- an E1 BB-only arm cannot speak for an E2 candidate, since it
    differs in representation, capacity and cost.
    """
    base = dict(
        label="synthetic",
        n_targets=40,
        n_events=160,
        sigma_values=[0.1, 2.0],
        pack_steps=50,
        arms={
            "bb0": dict(
                backbone_rmsd=1.000,
                ca_rmsd=0.9,
                variant=None,
                arch=None,
                denoiser_calls=1,
                seconds_per_event=1.0,
                **CLEAN_SC,
            ),
            "zero": dict(
                backbone_rmsd=1.000,
                variant=None,
                arch=None,
                denoiser_calls=2,
                seconds_per_event=2.0,
                max_abs_deviation_from_bb0=1e-8,
                zero_site="decoder",
            ),
            "full": dict(
                backbone_rmsd=0.800,
                variant="full",
                arch="late",
                denoiser_calls=2,
                seconds_per_event=3.0,
                **CLEAN_SC,
            ),
            "bb_only": dict(
                backbone_rmsd=0.900,
                variant="bb_only",
                arch="late",
                denoiser_calls=2,
                seconds_per_event=3.0,
            ),
            "generic": dict(
                backbone_rmsd=0.960,
                variant="generic",
                arch="late",
                denoiser_calls=2,
                seconds_per_event=3.0,
            ),
            "refine": dict(
                backbone_rmsd=0.950,
                variant=None,
                arch=None,
                denoiser_calls=2,
                seconds_per_event=1.6,
            ),
        },
        paired={
            "full": {
                "vs_bb0": dict(mean=0.200, low=0.150, high=0.250, n=40),
                "vs_bb_only": dict(mean=0.100, low=0.060, high=0.140, n=40),
                "vs_refine": dict(mean=0.150, low=0.110, high=0.190, n=40),
                "vs_generic": dict(mean=0.160, low=0.120, high=0.200, n=40),
            }
        },
    )
    for key, value in overrides.items():
        if key == "arms":
            for arm, fields in value.items():
                base["arms"].setdefault(arm, {}).update(fields)
        elif key == "paired":
            for arm, fields in value.items():
                base["paired"].setdefault(arm, {}).update(fields)
        elif value is None:
            base.pop(key, None)
        else:
            base[key] = value
    return base


# --- the threshold ----------------------------------------------------------


def test_the_threshold_is_the_larger_of_absolute_and_relative(mod):
    # 3% of 0.9 is 0.027, below the 0.05 A floor.
    assert mod.required_gain(0.9) == pytest.approx(0.05)
    # 3% of 5.0 is 0.15, above it.
    assert mod.required_gain(5.0) == pytest.approx(0.15)
    assert mod.MIN_ABSOLUTE_GAIN == 0.05 and mod.MIN_RELATIVE_GAIN == 0.03


def test_a_clean_result_passes(mod):
    outcome, lines = mod.verdict(record())
    assert outcome == mod.PASS, lines


def test_a_gain_below_the_criterion_fails(mod):
    """0.92 against a 0.90 bb_only baseline is 0.02 A: real, and not enough."""
    outcome, lines = mod.verdict(
        record(
            arms={"full": dict(backbone_rmsd=0.920)},
            paired={"full": {"vs_bb_only": dict(mean=0.02, low=0.01, high=0.03, n=40)}},
        )
    )
    assert outcome == mod.FAIL
    assert any("below the criterion" in line for line in lines)


def test_beating_bb0_but_not_the_trained_control_fails(mod):
    """The distinction the whole pilot is designed around."""
    outcome, lines = mod.verdict(
        record(
            arms={"bb_only": dict(backbone_rmsd=0.805)},
            paired={"full": {"vs_bb_only": dict(mean=0.005, low=0.001, high=0.009, n=40)}},
        )
    )
    assert outcome == mod.FAIL
    assert any("below the criterion" in line for line in lines)
    assert any("bb_only" in line for line in lines)


def test_the_strongest_alternative_is_chosen_not_the_weakest(mod):
    """refine at 0.85 must be the baseline, not generic at 0.96."""
    blob = record(
        arms={"refine": dict(backbone_rmsd=0.850)},
        paired={"full": {"vs_refine": dict(mean=0.05, low=0.03, high=0.07, n=40)}},
    )
    _outcome, lines = mod.verdict(blob)
    chosen = [line for line in lines if "strongest comparable-cost" in line]
    assert chosen and "refine" in chosen[0], lines
    assert "0.8500" in chosen[0]


# --- the wiring gate --------------------------------------------------------


def test_a_failed_wiring_check_fails_everything(mod):
    """If the zero arm does not reproduce bb0, nothing below it means anything."""
    outcome, lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=1e-2)})
    )
    assert outcome == mod.FAIL
    assert any("WIRING" in line and "FAILED" in line for line in lines)


def test_gpu_nondeterminism_alone_does_not_fail_the_wiring_check(mod):
    """The tolerance has to clear float noise and still catch a real fault.

    Two invocations of the same PXDesign forward on an H200 differ by ~4.8e-6 A,
    which an earlier 1e-6 tolerance reported as a wiring failure on a run where
    every backbone metric agreed to four decimals. The corrections being
    measured are ~2.6e-3 A, so the tolerance sits between the two.
    """
    outcome, lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=mod.GPU_NONDETERMINISM)})
    )
    assert outcome == mod.PASS, lines
    assert mod.GPU_NONDETERMINISM < mod.WIRING_TOLERANCE < 2.6e-3
    # A deviation the size of the effect is a real fault and must still fail.
    outcome, _lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=2.6e-3)})
    )
    assert outcome == mod.FAIL


def test_an_unmeasured_wiring_check_is_incomplete(mod):
    blob = record()
    del blob["arms"]["zero"]
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("wiring arm did not run" in line for line in lines)


def test_the_wiring_arm_names_the_site_it_tested(mod):
    """A zero a_token residual says nothing about a conditioning hook."""
    blob = record(arms={"zero": dict(zero_site="conditioning")})
    _outcome, lines = mod.verdict(blob)
    assert any("zero feedback at the conditioning site" in line for line in lines)


# --- finding 1: missing controls and chemistry regressions ------------------


def test_missing_trained_controls_are_incomplete_not_a_pass(mod):
    """THE defect. This test used to assert the opposite.

    Its old body deleted both controls, checked that a line said so, and then
    asserted ``passed``. The reasoning written beside it -- "it can still pass
    against refine; that is a correction, not an SC-specific claim, and the line
    above is what says so" -- is what a reader would have to notice and
    disbelieve. A verdict nobody reads the prose of is a verdict that passed.
    """
    blob = record()
    for arm in ("bb_only", "generic"):
        del blob["arms"][arm]
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("no trained BB-only or generic control" in line for line in lines)


def test_no_control_plus_destroyed_chemistry_is_not_a_pass(mod):
    """The reviewer's reproduction, verbatim: this combination returned True."""
    blob = record()
    for arm in ("bb_only", "generic"):
        del blob["arms"][arm]
    blob["arms"]["full"]["sc_bad_bond_fraction"] = 0.500
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE, lines


def test_a_chemistry_regression_fails_on_its_own(mod):
    outcome, lines = mod.verdict(
        record(arms={"full": dict(sc_bad_bond_fraction=0.500)})
    )
    assert outcome == mod.FAIL
    assert any("bad_bond_fraction regressed" in line for line in lines)


def test_a_symmetry_rmsd_regression_fails(mod):
    outcome, lines = mod.verdict(record(arms={"full": dict(sc_symmetry_rmsd=1.60)}))
    assert outcome == mod.FAIL, lines


def test_side_chain_noise_within_tolerance_still_passes(mod):
    """The bound is on MATERIAL regression; packing is a sampler."""
    outcome, lines = mod.verdict(record(arms={"full": dict(sc_symmetry_rmsd=1.47)}))
    assert outcome == mod.PASS, lines


def test_unscored_side_chains_are_incomplete(mod):
    """--no-sidechains forfeits half the criterion; it must not be silent."""
    blob = record()
    for key in CLEAN_SC:
        blob["arms"]["full"].pop(key)
        blob["arms"]["bb0"].pop(key)
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("was not scored" in line for line in lines)


def test_a_control_of_another_architecture_does_not_count(mod):
    """An E1 BB-only arm is not a control for an E2 candidate."""
    blob = record(arms={"full": dict(arch="atom", pair=True)})
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("architecture 'atom'" in line for line in lines)


# --- finding 2: missing intervals -------------------------------------------


def test_an_interval_spanning_zero_fails_even_with_a_big_mean(mod):
    outcome, lines = mod.verdict(
        record(paired={"full": {"vs_bb0": dict(mean=0.20, low=-0.05, high=0.45, n=40)}})
    )
    assert outcome == mod.FAIL
    assert any("includes zero" in line for line in lines)


def test_a_missing_interval_against_bb0_is_incomplete(mod):
    blob = record()
    blob["paired"]["full"].pop("vs_bb0")
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("no paired interval against bb0" in line for line in lines)


def test_a_missing_interval_against_the_baseline_is_incomplete(mod):
    blob = record()
    blob["paired"]["full"].pop("vs_bb_only")
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("no paired interval against bb_only" in line for line in lines)


def test_no_comparable_cost_alternative_is_incomplete(mod):
    blob = record()
    for arm in ("refine", "bb_only", "generic"):
        del blob["arms"][arm]
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("no comparable-cost alternative" in line for line in lines)


def test_no_candidate_arm_is_refused(mod):
    blob = record()
    blob["arms"]["full"]["variant"] = "bb_only"
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.INCOMPLETE
    assert any("no arm records variant='full'" in line for line in lines)


# --- finding 4: candidate selection -----------------------------------------


def two_full_arms():
    """E2's shape: atom_sz_full and atom_s_full are BOTH variant='full'."""
    blob = record()
    del blob["arms"]["full"], blob["arms"]["bb_only"], blob["arms"]["generic"]
    del blob["paired"]["full"]
    blob["arms"].update(
        atom_sz_full=dict(
            backbone_rmsd=0.800, variant="full", arch="atom", pair=True, **CLEAN_SC
        ),
        atom_s_full=dict(
            backbone_rmsd=0.850, variant="full", arch="atom", pair=False, **CLEAN_SC
        ),
        atom_sz_bb_only=dict(
            backbone_rmsd=0.900, variant="bb_only", arch="atom", pair=True
        ),
    )
    good = dict(mean=0.10, low=0.06, high=0.14, n=40)
    blob["paired"].update(
        atom_sz_full={
            "vs_bb0": good, "vs_atom_sz_bb_only": good,
            "vs_refine": good, "vs_atom_s_full": good,
        },
        atom_s_full={"vs_bb0": good, "vs_atom_sz_bb_only": good, "vs_refine": good},
    )
    return blob


def test_an_ambiguous_candidate_is_incomplete_rather_than_first_wins(mod):
    outcome, lines = mod.verdict(two_full_arms())
    assert outcome == mod.INCOMPLETE
    assert any("nothing in the record says which" in line for line in lines)


def test_naming_the_candidate_resolves_it(mod):
    outcome, lines = mod.verdict(two_full_arms(), candidate="atom_sz_full")
    assert outcome == mod.PASS, lines
    assert any(line.startswith("candidate: atom_sz_full") for line in lines)


def test_the_pair_ablation_is_reported_for_the_pair_arm(mod):
    _outcome, lines = mod.verdict(two_full_arms(), candidate="atom_sz_full")
    assert any("pair branch, atom_sz_full vs atom_s_full" in line for line in lines)


def test_a_candidate_that_did_not_run_is_refused(mod):
    outcome, lines = mod.verdict(record(), candidate="nonexistent")
    assert outcome == mod.INCOMPLETE
    assert any("did not run" in line for line in lines)


# --- finding 3: the comparison plan -----------------------------------------


def test_the_plan_finds_controls_whose_names_are_not_bb_only(mod):
    arms = two_full_arms()["arms"]
    candidates, plan = mod.comparison_plan(arms)
    assert set(candidates) == {"atom_sz_full", "atom_s_full"}
    assert "atom_sz_bb_only" in plan["atom_sz_full"]
    assert "atom_s_full" in plan["atom_sz_full"], "the pair ablation is not inferable"
    assert "refine" in plan["atom_sz_full"]


def test_the_cross_site_comparison_is_planned(mod):
    """The paired interval between injection sites, which is why both are scored.

    comparison_plan restricted candidate-versus-candidate to one architecture,
    so early_s_full vs late_full was never computed and a run containing both
    produced two independent tables -- exactly what scoring them together is
    meant to avoid.
    """
    arms = record()["arms"]
    arms["early_s_full"] = dict(
        backbone_rmsd=0.78, variant="full", arch="early_s", pair=False, **CLEAN_SC
    )
    arms["early_s_bb_only"] = dict(
        backbone_rmsd=0.88, variant="bb_only", arch="early_s", pair=False
    )
    _candidates, plan = mod.comparison_plan(arms)
    assert "late_full" not in plan  # the late candidate is labelled "full" here
    assert "full" in plan["early_s_full"], "no paired interval between the sites"
    assert "early_s_full" in plan["full"]
    # And it is labelled for what it is, not lumped in with the pair ablation.
    assert mod.candidate_relation(arms, "early_s_full", "full") == "injection site"
    assert mod.candidate_relation(arms, "full", "full") == "candidates"


def test_the_pair_ablation_and_the_site_are_different_relations(mod):
    arms = two_full_arms()["arms"]
    assert mod.candidate_relation(arms, "atom_sz_full", "atom_s_full") == "pair branch"


def test_a_cross_site_reference_is_not_a_control(mod):
    """Pairing against the other site must not make it a baseline for the bar."""
    arms = record()["arms"]
    arms["early_s_full"] = dict(
        backbone_rmsd=0.78, variant="full", arch="early_s", pair=False, **CLEAN_SC
    )
    assert mod.same_architecture_controls(arms, "early_s_full") == []
    assert "bb_only" not in mod.same_architecture_controls(arms, "early_s_full")


def test_the_plan_does_not_mix_architectures(mod):
    arms = record()["arms"]
    arms["other_bb"] = dict(backbone_rmsd=0.70, variant="bb_only", arch="atom", pair=True)
    _candidates, plan = mod.comparison_plan(arms)
    assert "other_bb" not in plan["full"]
    assert "bb_only" in plan["full"]


def test_the_perturbed_arm_is_paired_against_the_arm_it_perturbed(mod):
    arms = record()["arms"]
    arms["perturbed"] = dict(backbone_rmsd=0.99, source_arm="full", arch="late")
    assert "full" in mod.pairings(arms)["perturbed"]
    assert "perturbed" in mod.comparison_plan(arms)[1]["full"]


def test_per_sigma_entries_carry_what_the_plan_identifies_arms_by(mod):
    """A per-sigma summary is an arm too, and the plan reads labels off it.

    The per-sigma tables are built from trimmed copies of the arm entries. When
    those copies dropped variant/arch/pair, comparison_plan saw no candidates
    and every per-sigma row collapsed to "vs bb0" -- losing the
    full-versus-control comparison at each noise level, which is where the late
    pilot's sharpest findings came from, while the pooled table still had it.
    """
    full_entry = dict(
        backbone_rmsd=0.80, variant="full", arch="early_s", pair=False, n=64
    )
    trimmed = {k: v for k, v in full_entry.items() if k in ("backbone_rmsd", "n")}
    arms_labelled = {"bb0": dict(backbone_rmsd=0.9), "full": full_entry,
                     "bb_only": dict(backbone_rmsd=0.85, variant="bb_only",
                                     arch="early_s", pair=False)}
    arms_trimmed = {"bb0": dict(backbone_rmsd=0.9), "full": trimmed,
                    "bb_only": dict(backbone_rmsd=0.85)}
    assert "bb_only" in mod.pairings(arms_labelled)["full"]
    assert "bb_only" not in mod.pairings(arms_trimmed)["full"], (
        "unlabelled arms should not resolve controls -- this is the state the "
        "per-sigma loop must never be in"
    )


def test_every_arm_is_still_paired_against_bb0(mod):
    arms = two_full_arms()["arms"]
    plan = mod.pairings(arms)
    for name in arms:
        if name == "bb0":
            continue
        assert "bb0" in plan[name], name


# --- diagnostics that are reported, never scored ----------------------------


def test_the_perturbed_arm_is_reported_as_evidence_not_a_gate(mod):
    blob = record(
        arms={
            "perturbed": dict(
                backbone_rmsd=0.990,
                variant=None,
                arch="late",
                source_arm="full",
                denoiser_calls=2,
                seconds_per_event=3.0,
            )
        }
    )
    outcome, lines = mod.verdict(blob)
    assert outcome == mod.PASS, lines
    assert any("perturbed-SC arm" in line for line in lines)


def test_sidechain_levels_are_reported_with_their_tolerance(mod):
    _outcome, lines = mod.verdict(record())
    reported = [line for line in lines if "symmetry_rmsd" in line]
    assert reported and "tolerance" in reported[0], lines


# --- cost -------------------------------------------------------------------


def test_only_the_feedback_arms_are_charged_for_the_packing(mod):
    """Equal denoiser-call counts are not equal cost.

    A BB-only alternative needs bb0 and nothing else; a feedback arm also pays
    for the 50-step rollout and the re-encode. Charging every arm for the whole
    frozen half made them look cost-matched, in the direction that flatters
    feedback.
    """
    # Only the arms that read a packing. bb_only reads h_base, which `propose`
    # produces alongside bb0, and generic reads nothing -- so neither needs the
    # 50-step rollout when deployed, however the shared code path executes them.
    assert set(mod.NEEDS_PACKING) == {"full", "perturbed"}
    for arm in ("bb0", "zero", "refine", "bb_only", "generic"):
        assert arm not in mod.NEEDS_PACKING
    # Keyed on the recorded variant, so an arm labelled anything is priced by
    # what its readout reads.
    assert mod.needs_packing("full") and mod.needs_packing("perturbed")
    assert not mod.needs_packing("bb_only") and not mod.needs_packing("generic")
