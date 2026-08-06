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
