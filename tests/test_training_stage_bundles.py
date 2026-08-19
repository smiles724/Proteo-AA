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


# ---------------------------------------------------------------------------
# Stage II -> Stage III layout handoff
# ---------------------------------------------------------------------------

def test_coevolution_adopts_the_checkpoint_layout(source):
    """Wiring guard: the Stage III/IV bundle must consult the checkpoint's layout.

    Stage III continues a Stage II side-chain module, so S_phi's input layout
    belongs to that checkpoint, not to this stage's defaults. Without this call the
    bundle fell back to the config defaults and disagreed with every real Stage II
    checkpoint (centre_coord_input False vs True -> `_check_sidechain_arch` aborts;
    q_bs True vs False -> an untrained fusion channel switched on silently).
    """
    start = source.index('elif args.training_stage in ("coevolution", "predicted_mask")')
    block = source[start:source.index("    return configs", start)]
    assert "adopt_sidechain_arch_from_checkpoint(configs, args)" in block, (
        "Stage III/IV must adopt the warm-start checkpoint's side-chain layout"
    )


def _load_entry():
    """Import the training entry, skipping where its heavy deps are absent."""
    pytest.importorskip("torch")
    pytest.importorskip("protenix")
    pytest.importorskip("pxdesign")
    import importlib.util

    spec = importlib.util.spec_from_file_location("tpm_for_test", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_args(ckpt, **over):
    import argparse as _a

    ns = _a.Namespace(
        load_checkpoint=str(ckpt) if ckpt else None,
        sc_centre_coord_input=None, sc_frame_aware_head=None,
        sc_template_residual=None, sc_edm=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _fake_configs(**over):
    import types

    # The defaults that disagreed with real Stage II checkpoints.
    sc = types.SimpleNamespace(
        bb_context=True, centre_coord_input=False, frame_aware_head=False,
        template_residual=False, type_logits_input=True, edm=False,
        a_bs_concat=True, q_bs=True,
    )
    for k, v in over.items():
        setattr(sc, k, v)
    return types.SimpleNamespace(sidechain=sc)


ARCH = {
    "bb_context": True, "centre_coord_input": True, "frame_aware_head": False,
    "template_residual": False, "type_logits_input": True, "edm": False,
    "a_bs_concat": True, "q_bs": False,
}


def test_adoption_matches_the_checkpoint_and_unblocks_warm_start(tmp_path):
    """The exact failure job 101060 hit: centre_coord_input True vs False."""
    mod = _load_entry()
    import torch

    ckpt = tmp_path / "stage2.pt"
    torch.save({"sidechain_arch": dict(ARCH)}, ckpt)

    cfg = _fake_configs()
    changed = mod.adopt_sidechain_arch_from_checkpoint(cfg, _fake_args(ckpt))

    assert changed == {"centre_coord_input": True, "q_bs": False}
    # Every key now agrees, which is what `_check_sidechain_arch` requires.
    for key, want in ARCH.items():
        assert bool(getattr(cfg.sidechain, key)) == want, key


def test_an_explicit_cli_flag_beats_the_checkpoint(tmp_path):
    """Adopting over an explicit flag would be worse than the mismatch it fixes."""
    mod = _load_entry()
    import torch

    ckpt = tmp_path / "stage2.pt"
    torch.save({"sidechain_arch": dict(ARCH)}, ckpt)

    cfg = _fake_configs()
    changed = mod.adopt_sidechain_arch_from_checkpoint(
        cfg, _fake_args(ckpt, sc_centre_coord_input=False)
    )
    assert "centre_coord_input" not in changed
    assert bool(cfg.sidechain.centre_coord_input) is False
    # Keys without an explicit flag are still adopted.
    assert bool(cfg.sidechain.q_bs) is False


def test_adoption_is_a_no_op_without_a_checkpoint_or_a_record(tmp_path):
    """Training Stage III from scratch, or from a checkpoint that predates the
    record, must not be blocked -- and must not silently invent a layout."""
    mod = _load_entry()
    import torch

    cfg = _fake_configs()
    assert mod.adopt_sidechain_arch_from_checkpoint(cfg, _fake_args(None)) == {}
    assert bool(cfg.sidechain.centre_coord_input) is False   # untouched

    old = tmp_path / "no_record.pt"
    torch.save({"model": {}}, old)
    assert mod.adopt_sidechain_arch_from_checkpoint(_fake_configs(), _fake_args(old)) == {}


def test_adoption_survives_an_unreadable_checkpoint(tmp_path):
    """A bad path is the loader's error to raise later, not a crash in config build."""
    mod = _load_entry()
    bad = tmp_path / "not_a_checkpoint.pt"
    bad.write_text("this is not a torch archive")
    assert mod.adopt_sidechain_arch_from_checkpoint(_fake_configs(), _fake_args(bad)) == {}
