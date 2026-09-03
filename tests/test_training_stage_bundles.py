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
import argparse
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
    assert 'args.training_stage == "predicted_mask"' in block
    assert "configs.sidechain.per_sigma = True" in block
    # Stage III teacher-forces geometry/atom composition, while the AA logits
    # remain a learned feature supplied by B_pre as Fang specified.
    assert "configs.sidechain.force_gt_type_logits = False" in block


def test_joint_stage_still_declares_itself_sidechain_free(source):
    """`joint` is BB + AA only. Keep that explicit so it is not mistaken for Stage III."""
    start = source.index('elif args.training_stage == "joint"')
    block = source[start:source.index("elif args.training_stage in (", start)]
    assert "configs.enable_sidechain = False" in block
    assert "configs.enable_coevolution = False" in block


def test_stage_iv_opens_predicted_geometry_and_atom_sets(source):
    """Stage III uses GT geometry/atom sets; Stage IV matches inference inputs."""
    bundle = source[
        source.index('elif args.training_stage in ("coevolution", "predicted_mask")'):
        source.index("    return configs")
    ]
    assert 'args.training_stage == "predicted_mask"' in bundle
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


def test_stage_iii_iv_frame_curriculum_is_explicit_in_both_bundles(source):
    """III teacher-forces geometry; IV alone opens predicted frames."""
    config_start = source.index(
        'elif args.training_stage in ("coevolution", "predicted_mask")'
    )
    config_block = source[config_start:source.index("    return configs", config_start)]
    assert (
        'configs.sidechain.predicted_frame = (\n'
        '            args.training_stage == "predicted_mask"\n'
        '        )'
    ) in config_block

    apply_start = source.index("def apply_training_stage_args")
    apply_block = source[apply_start:source.index("def parse_args", apply_start)]
    assert (
        'args.predicted_frame = args.training_stage == "predicted_mask"'
        in apply_block
    )


@pytest.mark.parametrize(
    ("training_stage", "want_predicted_frame"),
    [("coevolution", False), ("predicted_mask", True)],
)
def test_stage_iii_iv_frame_curriculum_at_runtime(
    training_stage: str, want_predicted_frame: bool
) -> None:
    mod = _load_entry()
    args = argparse.Namespace(training_stage=training_stage)

    mod.apply_training_stage_args(args)

    assert args.predicted_frame is want_predicted_frame


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


def test_aa_head_on_stage2_freezes_everything_but_the_head(source):
    """Train the AA head on a frozen Stage II model, in the Stage III configuration.

    An AA head is not portable across configurations. A head trained by
    `aa_head_warmup` (enable_sidechain and enable_coevolution both False) produced
    aa_ce 54-76 at chance accuracy when grafted into Stage III (both True) -- and
    that held even for a donor whose backbone was byte-identical (636/636) to the
    Stage II trunk. So this bundle fits the head to the feature path it will
    actually run in, while the backbone and S_phi stay fixed.
    """
    start = source.index('elif args.training_stage == "aa_head_on_stage2"')
    block = source[start:source.index("    elif args.training_stage in (", start)]

    # Stage III's feature path, so the head is fitted to what it will see.
    assert "configs.enable_sidechain = True" in block
    assert "configs.enable_coevolution = True" in block
    assert "configs.sidechain.predicted_frame = False" in block
    assert "configs.sidechain.per_sigma = True" in block

    # ...but only the head learns.
    assert 'configs.training.trainable_param_keywords = ["design_residue_type_head."]' in block
    assert "configs.residue_type.trunk_grad_scale = 0.0" in block

    # Only the AA objective is live: the modules the other losses would train are
    # frozen, so they contribute no gradient.
    assert "configs.loss.weight_aa = 1.0" in block
    for zeroed in ("weight_mse", "weight_lddt", "weight_disto", "weight_sc_local",
                   "weight_sc_phys", "weight_sc_global", "weight_bb_post", "weight_aa_post"):
        assert f"configs.loss.{zeroed} = 0.0" in block, zeroed

    # Loads the Stage II checkpoint whole (backbone + conditioning + S_phi) and
    # honours the layout it recorded.
    assert "configs.training.checkpoint_include_prefixes = []" in block
    assert "adopt_sidechain_arch_from_checkpoint(configs, args)" in block


def test_aa_head_on_stage2_is_a_selectable_stage(source):
    block = source[source.index("--training-stage"):]
    block = block[:block.index(")")]
    assert '"aa_head_on_stage2"' in block


# ---------------------------------------------------------------------------
# AA-head masking contract
# ---------------------------------------------------------------------------

def _stage_args_fn(source: str) -> str:
    """Just `apply_training_stage_args`'s body.

    Several stage names appear twice in this file -- once in `build_configs` and
    once here -- so a whole-file `index` silently reads the wrong branch.
    """
    start = source.index("def apply_training_stage_args")
    return source[start:source.index("\ndef ", start + 1)]


FEATURIZER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "pxdesign_train" / "data" / "featurizer.py"
)


def test_every_stage_that_trains_the_aa_head_uses_partial_masks(source):
    """The masking schedule is a property of the objective, not of one bundle.

    Every stage that trains `design_residue_type_head` used
    `aa_mask_mode="all"` -- every design position masked at once -- which pins
    `aa_t` at 1.0 and collapses the masked-diffusion objective: the MDLM `1/t`
    weight becomes 1/1, and the head's time embedding contributes a constant. The
    head then never sees a partially revealed sequence and cannot learn to read
    already-decided neighbours -- the signal ProteinMPNN's recovery rests on.

    Asserting the shared constant rather than the literal is the point: four
    separate string literals are four places for this to drift back.
    """
    fn = _stage_args_fn(source)
    for stage in ("aa_head_warmup", "joint", "aa_head_on_stage2",
                  'in ("coevolution", "predicted_mask")'):
        i = fn.index(f'args.training_stage == "{stage}"'
                     if not stage.startswith("in ") else f"args.training_stage {stage}")
        j = fn.find("elif args.training_stage", i + 10)
        block = fn[i:j if j != -1 else len(fn)]
        assert "args.disable_aa_loss = False" in block, f"{stage} should train the head"
        assert "args.aa_mask_mode = AA_MASK_MODE_TRAINING" in block, stage
    assert 'AA_MASK_MODE_TRAINING = "time_dependent"' in source


def test_stages_that_do_not_train_the_head_set_no_schedule(source):
    """`backbone_only` and `sidechain_warmup` disable the AA loss entirely.

    The masking schedule is not theirs to choose, and giving them a real one
    would spend featurizer work on a target nothing reads.
    """
    fn = _stage_args_fn(source)
    for stage in ("backbone_only", "sidechain_warmup"):
        i = fn.index(f'args.training_stage == "{stage}"')
        block = fn[i:fn.index("elif args.training_stage", i + 10)]
        assert "args.disable_aa_loss = True" in block, stage
        assert 'args.aa_mask_mode = "none"' in block, stage


def test_validation_masking_is_pinned_regardless_of_training(source):
    """`val_aa_acc` has to keep meaning the same thing across runs.

    `eval_args` is a whole-namespace copy of the training args, so a stage that
    switches to partial masking switches validation with it. Partial masking is a
    strictly easier task -- neighbouring identities are visible -- so val_aa_acc
    would rise for a reason unrelated to the model, and stop being comparable to
    the 0.1310 every stage has reported.
    """
    start = source.index("eval_args = argparse.Namespace(**vars(args))")
    block = source[start:source.index("build_components(eval_args", start)]
    assert 'eval_args.aa_mask_mode = "all"' in block
    assert "eval_args.aa_mask_prob = 1.0" in block
    assert "eval_args.aa_mask_min_prob = 0.0" in block
    assert "eval_args.aa_mask_max_prob = 1.0" in block


def test_cli_mask_modes_match_the_featurizer(source):
    """A mode argparse accepts but the featurizer rejects fails after scheduling.

    `--aa-mask-mode partial` passed the CLI and then raised inside
    `DesignSelection.__post_init__`, i.e. once the job was already queued and had
    loaded its dataset index. The two lists have to agree.
    """
    start = source.index('"--aa-mask-mode"')
    ch = source.index("choices=", start)
    cli = source[ch:source.index("]", ch) + 1]
    feat = FEATURIZER.read_text()
    valid = feat[feat.index("valid_modes = {"):]
    valid = valid[:valid.index("}") + 1]
    for mode in ("all", "none", "fixed", "time_dependent"):
        assert f'"{mode}"' in valid, f"featurizer no longer accepts {mode}"
        assert f'"{mode}"' in cli, f"--aa-mask-mode cannot select {mode}"
    assert '"partial"' not in cli, "'partial' is not a featurizer mode"
