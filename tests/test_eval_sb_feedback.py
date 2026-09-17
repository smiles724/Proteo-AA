"""The exit criterion, as a function.

``verdict`` is where the roadmap's threshold is enforced, and it is the one place
a plausible-looking pilot could be declared a success it did not earn. So it is
tested against synthetic records rather than only exercised on real ones: the
cases that must fail are cheap to construct and expensive to notice.
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


def record(**overrides):
    """A passing record: real gain over bb0 and over the best alternative."""
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
                denoiser_calls=1,
                seconds_per_event=1.0,
            ),
            "zero": dict(
                backbone_rmsd=1.000,
                variant=None,
                denoiser_calls=2,
                seconds_per_event=2.0,
                max_abs_deviation_from_bb0=1e-8,
            ),
            "full": dict(
                backbone_rmsd=0.800, variant="full", denoiser_calls=2, seconds_per_event=3.0
            ),
            "bb_only": dict(
                backbone_rmsd=0.900,
                variant="bb_only",
                denoiser_calls=2,
                seconds_per_event=3.0,
            ),
            "generic": dict(
                backbone_rmsd=0.960,
                variant="generic",
                denoiser_calls=2,
                seconds_per_event=3.0,
            ),
            "refine": dict(
                backbone_rmsd=0.950, variant=None, denoiser_calls=2, seconds_per_event=1.6
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


def test_the_threshold_is_the_larger_of_absolute_and_relative(mod):
    # 3% of 0.9 is 0.027, below the 0.05 A floor.
    assert mod.required_gain(0.9) == pytest.approx(0.05)
    # 3% of 5.0 is 0.15, above it.
    assert mod.required_gain(5.0) == pytest.approx(0.15)
    assert mod.MIN_ABSOLUTE_GAIN == 0.05 and mod.MIN_RELATIVE_GAIN == 0.03


def test_a_clean_result_passes(mod):
    passed, lines = mod.verdict(record())
    assert passed, lines


def test_a_failed_wiring_check_fails_everything(mod):
    """If the zero arm does not reproduce bb0, nothing below it means anything."""
    passed, lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=1e-2)})
    )
    assert not passed
    assert any("WIRING" in line and "FAILED" in line for line in lines)


def test_gpu_nondeterminism_alone_does_not_fail_the_wiring_check(mod):
    """The tolerance has to clear float noise and still catch a real fault.

    Two invocations of the same PXDesign forward on an H200 differ by ~4.8e-6 A,
    which an earlier 1e-6 tolerance reported as a wiring failure on a run where
    every backbone metric agreed to four decimals. The corrections being
    measured are ~2.6e-3 A, so the tolerance sits between the two.
    """
    passed, lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=mod.GPU_NONDETERMINISM)})
    )
    assert passed, lines
    assert mod.GPU_NONDETERMINISM < mod.WIRING_TOLERANCE < 2.6e-3
    # A deviation the size of the effect is a real fault and must still fail.
    passed, _lines = mod.verdict(
        record(arms={"zero": dict(max_abs_deviation_from_bb0=2.6e-3)})
    )
    assert not passed


def test_an_unmeasured_wiring_check_fails(mod):
    blob = record()
    del blob["arms"]["zero"]
    passed, lines = mod.verdict(blob)
    assert not passed
    assert any("not measured" in line for line in lines)


def test_a_gain_below_the_criterion_fails(mod):
    """0.92 against a 0.90 bb_only baseline is 0.02 A: real, and not enough."""
    passed, lines = mod.verdict(
        record(
            arms={"full": dict(backbone_rmsd=0.920)},
            paired={"full": {"vs_bb_only": dict(mean=0.02, low=0.01, high=0.03, n=40)}},
        )
    )
    assert not passed
    assert any("below the criterion" in line for line in lines)


def test_an_interval_spanning_zero_fails_even_with_a_big_mean(mod):
    passed, lines = mod.verdict(
        record(paired={"full": {"vs_bb0": dict(mean=0.20, low=-0.05, high=0.45, n=40)}})
    )
    assert not passed
    assert any("includes zero" in line for line in lines)


def test_beating_bb0_but_not_the_trained_control_fails(mod):
    """The distinction the whole pilot is designed around."""
    passed, lines = mod.verdict(
        record(
            arms={"bb_only": dict(backbone_rmsd=0.805)},
            paired={"full": {"vs_bb_only": dict(mean=0.005, low=0.001, high=0.009, n=40)}},
        )
    )
    assert not passed
    assert any("below the criterion" in line for line in lines)
    assert any("bb_only" in line for line in lines)


def test_the_strongest_alternative_is_chosen_not_the_weakest(mod):
    """refine at 0.85 must be the baseline, not generic at 0.96."""
    blob = record(
        arms={"refine": dict(backbone_rmsd=0.850)},
        paired={"full": {"vs_refine": dict(mean=0.05, low=0.03, high=0.07, n=40)}},
    )
    _passed, lines = mod.verdict(blob)
    chosen = [line for line in lines if "strongest comparable-cost" in line]
    assert chosen and "refine" in chosen[0], lines
    assert "0.8500" in chosen[0]


def test_missing_trained_controls_are_called_out(mod):
    blob = record()
    for arm in ("bb_only", "generic"):
        del blob["arms"][arm]
    passed, lines = mod.verdict(blob)
    assert any("NO trained control ran" in line for line in lines)
    # It can still pass against refine -- that is a correction, not an
    # SC-specific claim, and the line above is what says so.
    assert passed


def test_no_candidate_arm_is_refused(mod):
    blob = record()
    blob["arms"]["full"]["variant"] = "bb_only"
    passed, lines = mod.verdict(blob)
    assert not passed
    assert any("no arm records variant='full'" in line for line in lines)


def test_no_comparable_cost_alternative_cannot_pass(mod):
    blob = record()
    for arm in ("refine", "bb_only", "generic"):
        del blob["arms"][arm]
    passed, lines = mod.verdict(blob)
    assert not passed
    assert any("only the weak comparison" in line for line in lines)


def test_only_the_feedback_arms_are_charged_for_the_packing(mod):
    """Equal denoiser-call counts are not equal cost.

    A BB-only alternative needs bb0 and nothing else; a feedback arm also pays
    for the 50-step rollout and the re-encode. Charging every arm for the whole
    frozen half made them look cost-matched, in the direction that flatters
    feedback.
    """
    # Only the arms that read h_packed. bb_only reads h_base, which `propose`
    # produces alongside bb0, and generic reads nothing -- so neither needs the
    # 50-step rollout when deployed, however the shared code path executes them.
    assert set(mod.NEEDS_PACKING) == {"full", "perturbed"}
    for arm in ("bb0", "zero", "refine", "bb_only", "generic"):
        assert arm not in mod.NEEDS_PACKING


def test_the_perturbed_arm_is_reported_as_evidence_not_a_gate(mod):
    blob = record(
        arms={
            "perturbed": dict(
                backbone_rmsd=0.990, variant=None, denoiser_calls=2, seconds_per_event=3.0
            )
        }
    )
    passed, lines = mod.verdict(blob)
    assert passed, lines
    assert any("perturbed-SC arm" in line for line in lines)


def test_sidechain_levels_are_reported_when_present(mod):
    blob = record(
        arms={
            "bb0": dict(sc_symmetry_rmsd=1.10, sc_bad_bond_fraction=0.01),
            "full": dict(sc_symmetry_rmsd=1.08, sc_bad_bond_fraction=0.01),
        }
    )
    _passed, lines = mod.verdict(blob)
    assert any("symmetry_rmsd" in line and "sc0 -> sc1" in line for line in lines)
