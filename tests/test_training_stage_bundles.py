"""Each named training stage must be a coherent, complete objective bundle.

The gap this closes: the paper defines four stages (I backbone, II side-chain
warmup, III joint co-evolution, IV predicted-mask robustness), but the training
entry only had bundles for I and II. `joint` is BB + AA head with
`enable_sidechain=False`, so everything Stage III is *about* — S_phi, the
feedback channel, the refinement pass, L_bb^post / L_aa^post — was reachable
only from the small example script. A stage that exists in the paper and not in
the entry point quietly becomes a stage nobody runs.

These tests read the entry as source rather than importing it: the module pulls
in the full Protenix training stack at import time, and what is being asserted
is the configuration contract, not runtime behaviour.
"""
import ast
import pathlib

import pytest

ENTRY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "training" / "train_protenix_monomer.py"
)
PAPER_STAGES = {
    "backbone_only": "I",
    "sidechain_warmup": "II",
    "coevolution": "III",
    "predicted_mask": "IV",
}


@pytest.fixture(scope="module")
def source() -> str:
    return ENTRY.read_text()


def test_entry_parses(source):
    ast.parse(source)


def test_every_paper_stage_has_an_entry(source):
    """A stage in the paper with no bundle here is a stage that never runs."""
    for stage, roman in PAPER_STAGES.items():
        assert f'"{stage}"' in source, f"paper Stage {roman} has no --training-stage bundle"


def test_stage_names_are_selectable_on_the_cli(source):
    start = source.index('"--training-stage"')
    block = source[start:start + 600]
    for stage in PAPER_STAGES:
        assert f'"{stage}"' in block, f"{stage} is not in the --training-stage choices"


def test_coevolution_turns_on_the_machinery_it_is_named_for(source):
    """Stage III without the refinement pass is just Stage I with extra loss terms."""
    start = source.index('elif args.training_stage in ("coevolution", "predicted_mask")')
    block = source[start:source.index("    return configs", start)]
    assert "configs.enable_sidechain = True" in block
    assert "configs.enable_coevolution = True" in block
    assert "configs.sidechain.predicted_frame = True" in block
    assert "configs.sidechain.per_sigma = True" in block
    # teacher-forced GT logits belong to the Stage II warmup, not here
    assert "configs.sidechain.force_gt_type_logits = False" in block


def test_joint_stage_still_declares_itself_sidechain_free(source):
    """`joint` is BB + AA only. Keep that explicit so it is not mistaken for Stage III."""
    start = source.index('elif args.training_stage == "joint"')
    block = source[start:source.index("elif args.training_stage in (", start)]
    assert "configs.enable_sidechain = False" in block
    assert "configs.enable_coevolution = False" in block


def test_stage_iv_is_stage_iii_plus_predicted_atom_sets(source):
    """The only thing separating IV from III is where the atom set comes from."""
    start = source.index('if args.training_stage == "predicted_mask"')
    block = source[start:start + 900]
    assert "configs.sidechain.predicted_mask = True" in block
    assert "configs.sidechain.route_by_type = True" in block


def test_every_stage_sets_the_dataset_args_it_needs(source):
    """`apply_training_stage_args` and the config bundle must cover the same stages.

    A stage present in one and missing from the other silently inherits CLI
    defaults for half its configuration.
    """
    start = source.index("def apply_training_stage_args")
    block = source[start:source.index("def parse_args", start)]
    for stage in PAPER_STAGES:
        assert f'"{stage}"' in block, (
            f"{stage} has a config bundle but no dataset-arg bundle"
        )


# ------------------------------------------------------- guarded layout switches

# `PXDesignTrainer.SIDECHAIN_LAYOUT_KEYS` decides the SHAPE and MEANING of what
# S_phi reads and writes, so a checkpoint trained under one setting cannot be
# warm-started under another. The guard raises rather than degrading silently --
# which means a stage that forgets to declare one of these does not quietly
# underperform, it fails to start.
#
# That is exactly what happened here: commit 09e6b43 turned all three on for
# `sidechain_warmup` (measured 7x better on the single-structure smoke; see the
# frame_aware_head table in docs/sidechain_config_notes.md) but never touched the
# Stage III bundle, so Stage III silently fell back to the base-config defaults
# (all False) and would have refused the Stage II checkpoint -- via the guard in
# PXDesignTrainer._check_sidechain_arch -- the first time anyone ran it.
SIDECHAIN_LAYOUT_SWITCHES = (
    "frame_aware_head", "local_coord_input", "template_residual",
)


def _stage_block(source: str, start: str, end: str) -> str:
    i = source.index(start)
    return source[i:source.index(end, i)]


def _warmup_block(source: str) -> str:
    return _stage_block(
        source,
        'elif args.training_stage == "sidechain_warmup"',
        'elif args.training_stage == "joint"',
    )


def _coevolution_block(source: str) -> str:
    return _stage_block(
        source,
        'elif args.training_stage in ("coevolution", "predicted_mask")',
        "    return configs",
    )


def _assignment(block: str, key: str):
    """Whitespace-normalised right-hand side of `configs.sidechain.<key> = ...`.

    Returns None when the stage never assigns the key -- which is the failure
    mode this file is about: an unset switch is not "left alone", it silently
    takes the base-config default.
    """
    marker = f"configs.sidechain.{key} ="
    if marker not in block:
        return None
    rest = block[block.index(marker) + len(marker):].lstrip()
    if rest.startswith("("):
        depth, out = 0, []
        for ch in rest:
            out.append(ch)
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                break
        text = "".join(out)
    else:
        text = rest.split("\n", 1)[0]
    return " ".join(text.split())


def test_stage_iii_declares_every_guarded_layout_switch(source):
    """A layout switch Stage III does not set is a Stage III that cannot start."""
    block = _coevolution_block(source)
    for key in SIDECHAIN_LAYOUT_SWITCHES:
        assert _assignment(block, key) is not None, (
            f"Stage III never assigns sidechain.{key}, so it falls back to the "
            f"base-config default while the Stage II checkpoint was trained with "
            f"the opposite value. The LAYOUT guard then refuses the warm-start."
        )


def test_stage_iii_layout_matches_stage_ii(source):
    """The two stages must declare the switches identically, not merely both.

    Declaring them with different defaults would swap one failure (refuses to
    start) for a worse one (starts, and the head means something else).
    """
    ii, iii = _warmup_block(source), _coevolution_block(source)
    for key in SIDECHAIN_LAYOUT_SWITCHES:
        assert _assignment(ii, key) is not None, (
            f"Stage II stopped setting sidechain.{key} -- this test's premise is gone"
        )
        assert _assignment(iii, key) == _assignment(ii, key), (
            f"sidechain.{key} is declared differently in Stage II and Stage III:\n"
            f"  Stage II : {_assignment(ii, key)}\n"
            f"  Stage III: {_assignment(iii, key)}"
        )


def test_stage_ii_and_iii_derive_bb_context_from_the_ablation_arm(source):
    """`bb_context` is guarded, so both stages must pin it -- but not with a literal.

    Two failure modes meet here. Leaving it unset makes the stage depend on the
    base-config default (the bug this file already documents for the other three
    switches). Assigning it a literal is worse: `apply_sidechain_ablation_arm`
    runs BEFORE the stage bundles, so a literal silently un-does the arm, which is
    exactly how `no-bbctx` -- the one 10-slot arm -- became a no-op in Stage II.
    Deriving it from the arm satisfies both.
    """
    for name, block in (("II", _warmup_block(source)), ("III", _coevolution_block(source))):
        rhs = _assignment(block, "bb_context")
        assert rhs is not None, f"Stage {name} does not pin sidechain.bb_context"
        assert "arm_bb_context" in rhs, (
            f"Stage {name} assigns sidechain.bb_context = {rhs!r}. A literal "
            f"overrides the ablation arm, because the arm was applied earlier."
        )


def test_the_cli_keeps_no_second_copy_of_the_arm_names(source):
    """The hand-maintained choices list drifted: 21 arms exist, it offered 6.

    `no-bbctx` -- the one 10-slot arm, and the reason this matters -- could not be
    passed at all, while `bbctx`, a name renamed long ago, was still accepted by
    argparse and then raised inside `apply_sidechain_ablation_arm`.

    Deriving the list in place is not possible: parse_args() runs before
    _bootstrap_paths() puts Protenix on sys.path, so SC_ABLATION_ARMS cannot be
    imported at that point. So the duplicate is removed instead and
    `apply_sidechain_ablation_arm` stays the single validator -- it already raises
    with the full sorted list, at config-build time, before training starts.
    """
    start = source.index('"--sc-ablation-arm"')
    block = source[start:source.index("    )", start)]
    assert "choices=" not in block, (
        "--sc-ablation-arm carries a hand-maintained choices list again; it drifts "
        "from SC_ABLATION_ARMS every time an arm is added or renamed"
    )
