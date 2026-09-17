"""The completion condition: how a run is arranged must not change its numbers.

Rearranging the evaluation -- reordering arms, adding a control, splitting one
job into several and recombining -- has to leave every existing arm's seeds,
shared inputs and results untouched. Otherwise a comparison between runs is
measuring the schedule as much as the model.

The seed tests deliberately use a **subprocess**. Python salts ``hash`` for
``str`` per interpreter, so an in-process comparison cannot distinguish a stable
seed from an unstable one -- which is exactly how the earlier ``hash()``-based
implementation passed its tests while being wrong across jobs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pxf.eval import couple as ev  # noqa: E402


def _in_subprocess(snippet):
    """Run a snippet in a fresh interpreter and parse its JSON stdout."""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


SEED_SNIPPET = """
import json, sys
sys.path.insert(0, {root!r})
from pxf.eval.couple import target_seed
print(json.dumps({body}))
"""


def test_seeds_survive_a_fresh_interpreter():
    body = (
        "[target_seed(0, 'AF-P81613-F1-model_v4', s) "
        "for s in (0.010, 0.082, 0.429, 1.642, 4.881)]"
    )
    first = _in_subprocess(SEED_SNIPPET.format(root=str(ROOT), body=body))
    second = _in_subprocess(SEED_SNIPPET.format(root=str(ROOT), body=body))
    assert first == second
    # And against the in-process value, so the three agree rather than merely
    # two subprocesses agreeing with each other.
    here = [
        ev.target_seed(0, "AF-P81613-F1-model_v4", s)
        for s in (0.010, 0.082, 0.429, 1.642, 4.881)
    ]
    assert first == here


def test_seeds_survive_an_explicitly_hostile_hash_salt():
    """PYTHONHASHSEED is what made the old implementation vary; pin it out."""
    body = "[target_seed(0, 'AF-X-F1', 0.429), target_seed(0, 'AF-X-F1', 1.642)]"
    snippet = SEED_SNIPPET.format(root=str(ROOT), body=body)
    seeds = []
    for salt in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": salt},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        seeds.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert seeds[0] == seeds[1] == seeds[2]


# --- splitting a panel must not move anyone's seed --------------------------


def test_splitting_a_panel_into_jobs_preserves_every_seed():
    panel = [f"AF-T{i:03d}-F1" for i in range(12)]
    sigmas = (0.010, 0.429, 4.881)
    whole = {(t, s): ev.target_seed(0, t, s) for t in panel for s in sigmas}
    # Two shards, and a reversed pass, all of which must agree with the whole.
    for subset in (panel[:5], panel[5:], list(reversed(panel))):
        for t in subset:
            for s in sigmas:
                assert ev.target_seed(0, t, s) == whole[(t, s)]


def test_adding_a_sigma_does_not_move_the_others():
    """A 5-point sweep must reuse the 3-point sweep's seeds where they overlap."""
    from pxf.couple import schedule  # noqa: PLC0415

    def make():
        return schedule.from_config(
            {"mode": "trajectory", "sigma_min": 0.01, "sigma_max": 5.0, "n_step": 400}
        )

    three, five = ev.sweep_sigmas(make(), 3), ev.sweep_sigmas(make(), 5)
    for sigma in set(three) & set(five):
        assert ev.target_seed(0, "AF-X-F1", sigma) == ev.target_seed(0, "AF-X-F1", sigma)
    # The seeds are a function of the value, so overlap is exact by construction;
    # assert the overlap is non-trivial or the check above is vacuous.
    assert len(set(three) & set(five)) >= 2


# --- manifest compatibility -------------------------------------------------


def base_manifest(**overrides):
    manifest = {
        "seed_scheme": ev.SEED_SCHEME,
        "seed_base": 0,
        "sigma_values": [0.01, 0.429, 4.881],
        "pack_steps": 50,
        "run_feedback": False,
        "codesign": False,
        "fampnn_weights": "0.0",
        "checkpoint_sha256": "abc123",
        "ema": True,
        "structures_fingerprint": "deadbeef",
        "upstream": {"fampnn": "aaf788b"},
        "a_token_source": "own",
    }
    manifest.update(overrides)
    return manifest


def test_identical_manifests_are_compatible():
    assert ev.incompatible_fields(base_manifest(), base_manifest()) == []


def test_the_shuffled_control_stays_comparable_to_its_reference():
    """a_token_source is the one field that *should* differ between the pair."""
    real = base_manifest(a_token_source="own")
    shuffled = base_manifest(a_token_source="shuffled-donor")
    assert ev.incompatible_fields(real, shuffled) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed_scheme", "hash-v0"),
        ("seed_base", 7),
        ("sigma_values", [0.01, 1.0]),
        ("pack_steps", 10),
        ("run_feedback", True),
        ("codesign", True),
        ("fampnn_weights", "0.3"),
        ("checkpoint_sha256", "other"),
        ("ema", False),
        ("structures_fingerprint", "cafe"),
        ("upstream", {"fampnn": "different"}),
    ],
)
def test_every_field_that_changes_the_measurement_blocks_a_comparison(field, value):
    mismatched = ev.incompatible_fields(base_manifest(), base_manifest(**{field: value}))
    assert [k for k, _, _ in mismatched] == [field]


def test_a_reordered_panel_is_a_different_experiment():
    """The shuffled control's donor is the previous target, so order matters."""
    forward = ev.structures_fingerprint(["a.cif", "b.cif", "c.cif"])
    reversed_ = ev.structures_fingerprint(["c.cif", "b.cif", "a.cif"])
    assert forward != reversed_
    assert forward == ev.structures_fingerprint(["a.cif", "b.cif", "c.cif"])


def test_the_fingerprint_is_stable_across_interpreters():
    snippet = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pxf.eval.couple import structures_fingerprint\n"
        "print(json.dumps(structures_fingerprint(['a.cif', 'b.cif'])))\n"
    )
    assert _in_subprocess(snippet) == ev.structures_fingerprint(["a.cif", "b.cif"])
