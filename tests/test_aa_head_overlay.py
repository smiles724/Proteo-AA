"""Composing a Stage III start out of two checkpoints.

No single checkpoint in this project carries all three trained components:

  * a Stage II side-chain run has the backbone and S_phi, but its warm start
    filtered the AA head out and then froze it, so the saved AA head is a fresh
    random init (measured on a real Stage III smoke: aa_ce 2.991 against
    ln(20)=2.996, aa_acc 5-7% against chance 1/20);
  * a `joint` run has a trained backbone and AA head, but `enable_sidechain=False`
    means S_phi is never constructed and never saved (743 tensors, zero
    `sidechain_module.*`).

Stage III needs all three, so `overlay_module_from_checkpoint` copies just the AA
head over an already-loaded Stage II checkpoint.
"""
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


class _Model(torch.nn.Module):
    """Stand-in with the module names the overlay selects on."""

    def __init__(self):
        super().__init__()
        self.diffusion_module = torch.nn.Linear(4, 4)
        self.design_residue_type_head = torch.nn.Linear(4, 3)
        self.sidechain_module = torch.nn.Linear(4, 4)


def _trainer(model):
    """A bare object carrying only what the overlay method touches."""
    from pxdesign_train.runner.trainer import PXDesignTrainer

    t = types.SimpleNamespace(
        model=model, device=torch.device("cpu"), use_ddp=False, logs=[],
    )
    t._log = t.logs.append
    t.overlay_module_from_checkpoint = types.MethodType(
        PXDesignTrainer.overlay_module_from_checkpoint, t
    )
    t.AA_HEAD_PREFIXES = PXDesignTrainer.AA_HEAD_PREFIXES
    return t


def _donor(tmp_path, name="joint.pt", fill=7.0, include_sc=False):
    donor = _Model()
    with torch.no_grad():
        donor.design_residue_type_head.weight.fill_(fill)
        donor.design_residue_type_head.bias.fill_(fill)
        donor.diffusion_module.weight.fill_(-99.0)   # must NOT be copied
    state = {k: v for k, v in donor.state_dict().items()
             if include_sc or not k.startswith("sidechain_module.")}
    path = tmp_path / name
    torch.save({"model": state}, path)
    return path


def test_overlay_replaces_only_the_aa_head(tmp_path):
    model = _Model()
    before_diff = model.diffusion_module.weight.detach().clone()
    before_sc = model.sidechain_module.weight.detach().clone()
    t = _trainer(model)

    keys = t.overlay_module_from_checkpoint(
        str(_donor(tmp_path)), t.AA_HEAD_PREFIXES, label="AA head")

    assert keys == ["design_residue_type_head.bias", "design_residue_type_head.weight"]
    assert torch.allclose(model.design_residue_type_head.weight,
                          torch.full_like(model.design_residue_type_head.weight, 7.0))
    # The whole point: the backbone and S_phi that the primary load supplied stay.
    assert torch.equal(model.diffusion_module.weight, before_diff)
    assert torch.equal(model.sidechain_module.weight, before_sc)


def test_overlay_raises_when_the_donor_has_no_aa_head(tmp_path):
    """A silent no-op would leave a chance-level AA head in a run whose command
    line claims otherwise."""
    model = _Model()
    path = tmp_path / "sc_only.pt"
    torch.save({"model": {"sidechain_module.weight": torch.zeros(4, 4)}}, path)
    t = _trainer(model)
    with pytest.raises(ValueError, match="no usable AA head tensors"):
        t.overlay_module_from_checkpoint(str(path), t.AA_HEAD_PREFIXES, label="AA head")


def test_overlay_skips_shape_mismatches_rather_than_crashing(tmp_path):
    model = _Model()
    path = tmp_path / "wrong_shape.pt"
    torch.save({"model": {
        "design_residue_type_head.weight": torch.zeros(99, 4),   # wrong
        "design_residue_type_head.bias": torch.full((3,), 5.0),  # right
    }}, path)
    t = _trainer(model)
    keys = t.overlay_module_from_checkpoint(str(path), t.AA_HEAD_PREFIXES, label="AA head")
    assert keys == ["design_residue_type_head.bias"]
    assert torch.allclose(model.design_residue_type_head.bias,
                          torch.full_like(model.design_residue_type_head.bias, 5.0))
    assert any("incompatible shapes" in m for m in t.logs)


def test_overlay_tolerates_donor_keys_this_model_does_not_have(tmp_path):
    """An older aa_head_warmup lineage also saved `backbone_aa_encoder.*`, which the
    current model does not build."""
    model = _Model()
    path = tmp_path / "older.pt"
    torch.save({"model": {
        "design_residue_type_head.weight": torch.full((3, 4), 2.0),
        "design_residue_type_head.bias": torch.full((3,), 2.0),
        "design_residue_type_head.nonexistent.weight": torch.zeros(3, 3),
    }}, path)
    t = _trainer(model)
    keys = t.overlay_module_from_checkpoint(str(path), t.AA_HEAD_PREFIXES, label="AA head")
    assert "design_residue_type_head.weight" in keys
    assert any("not in this model" in m for m in t.logs)


def test_aa_head_prefixes_exclude_the_module_the_model_does_not_build():
    """`backbone_aa_encoder.` must stay out of the prefix set: overlaying it would
    only generate unexpected-key noise."""
    from pxdesign_train.runner.trainer import PXDesignTrainer

    assert PXDesignTrainer.AA_HEAD_PREFIXES == ("design_residue_type_head.",)


def test_overlay_onto_a_full_resume_is_refused():
    """Overlaying on a resume would roll the AA head back to the donor's weights
    while the optimizer state carried on -- silent, and destructive."""
    import inspect

    from pxdesign_train.runner.trainer import PXDesignTrainer

    src = inspect.getsource(PXDesignTrainer.__init__)
    assert "Refusing to overlay an AA head on top of a FULL resume" in src
    assert "checkpoint_params_only" in src


def test_cli_and_runner_expose_the_overlay():
    """The flag has to reach the trainer, or it is decoration."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    entry = (root / "scripts" / "training" / "train_protenix_monomer.py").read_text()
    runner = (root / "pxdesign_train" / "runner" / "train.py").read_text()
    assert '"--load-aa-head-from"' in entry
    assert "overlay_aa_head_path=args.load_aa_head_from or None" in entry
    assert "overlay_aa_head_path" in runner


def test_stage2_carries_the_aa_head_along_the_lineage():
    """Stage II must not drop the AA head from its warm start.

    It did, and that is why a Stage III built off any Stage II checkpoint began
    with a chance-level identity predictor: the head was excluded from the load,
    re-initialised, frozen (weight_aa=0, not in trainable_param_keywords), and then
    saved into the file Stage III inherits wholesale.

    Carrying it is free here and is the only way the three components stay mutually
    consistent -- grafting a head trained against a different backbone does not
    work (measured: aa_ce ~50 at chance accuracy).
    """
    import pathlib

    entry = (pathlib.Path(__file__).resolve().parents[1]
             / "scripts" / "training" / "train_protenix_monomer.py").read_text()
    start = entry.index('elif args.training_stage == "sidechain_warmup"')
    block = entry[start:entry.index('elif args.training_stage == "joint"', start)]
    assert "checkpoint_include_prefixes" in block
    filt = block[block.index("checkpoint_include_prefixes"):]
    filt = filt[:filt.index("]") + 1]
    for prefix in ('"diffusion_module."', '"design_condition_embedder."',
                   '"design_residue_type_head."'):
        assert prefix in filt, f"Stage II warm start must carry {prefix}"
