"""Resume must actually happen on requeue, and refusals must say why.

These jobs are `--requeue`, and a requeued job keeps its job ID, so the
OUTPUT_DIR that embeds it still holds the checkpoints written before the
preemption. The launcher used to hardcode a donor warm start, so a restart
silently began again at step 0 and discarded every hour already spent.
"""
import pathlib
import re

import pytest

LAUNCHER = (pathlib.Path(__file__).resolve().parents[1]
            / "scripts" / "training" / "slurm_stage4_ligandmpnn_binder_hai.sh")


@pytest.fixture(scope="module")
def launcher() -> str:
    return LAUNCHER.read_text()


def test_launcher_prefers_a_checkpoint_over_the_donor(launcher):
    assert "RESUME_FROM" in launcher
    assert 'START_OPTIONS=(--load-checkpoint "$RESUME_FROM")' in launcher
    # The donor path must be the FALLBACK, and must still be params-only:
    # a donor is a warm start, never a resume.
    donor = launcher.index('START_OPTIONS=(--load-checkpoint "$STAGE3_CHECKPOINT" --warm-start-params-only)')
    resume = launcher.index('START_OPTIONS=(--load-checkpoint "$RESUME_FROM")')
    assert resume < donor, "the donor branch must be the else"
    # And exactly one of them reaches the command line.
    assert launcher.count('"${START_OPTIONS[@]}"') == 1
    assert '--load-checkpoint "$STAGE3_CHECKPOINT" --warm-start-params-only \\' not in launcher


def test_resume_is_a_full_resume_not_params_only(launcher):
    """--warm-start-params-only would drop optimizer state and the step count."""
    block = launcher[launcher.index("RESUME_FROM"):launcher.index("# CHECKPOINT_INTERVAL")]
    resume_line = next(l for l in block.splitlines() if 'START_OPTIONS=(--load-checkpoint "$RESUME_FROM")' in l)
    assert "--warm-start-params-only" not in resume_line


def test_checkpoint_selection_is_numeric(launcher):
    """step150 must not beat step1000; a lexical sort would let it."""
    assert "sort -k1,1n" in launcher, "numeric sort on the step number"
    assert "ls -1t" not in launcher, "mtime is the wrong key; an interrupted save is newer"


def test_identity_refusal_names_the_field_that_moved():
    """A bare 'identity differs' cannot be acted on.

    `implementation_sha256` covers every pxdesign_train/**/*.py and
    `proteoaa_revision` is git HEAD, so the gate also trips on a commit that
    touched nothing this run reads. Whether to restart or to warm-start
    deliberately depends entirely on which field moved.
    """
    import inspect

    from pxdesign_train.runner.trainer import PXDesignTrainer

    source = inspect.getsource(PXDesignTrainer.load_checkpoint)
    assert "differing" in source and "detail" in source
    assert "--warm-start-params-only to accept losing optimizer state" in source
    # Still a refusal, not a warning.
    assert re.search(r"raise ValueError\(\s*f?\"Stage IV resume identity differs", source)
