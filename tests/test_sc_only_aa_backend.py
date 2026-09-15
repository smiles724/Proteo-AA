"""`aa_backend="sc_only"`: the Stage IV pack/refine forward with NO AA head.

WHY THIS EXISTS. The supervised SC phases (sc_warmup / sc_geometry_repair /
sc_complex_adapt) build their residue-type logits from `aa_clean`, pin the AA
objective coefficients to zero, and never enter the decode cycle. FaMPNN is
loaded, frozen, put in eval mode -- and then never called. The only thing that
required it there was the requirement itself, and that requirement makes those
phases unrunnable for anyone without the released weights.

WHAT MUST NOT HAPPEN. "No AA head" must not become a quiet way to run a phase
that needs one. So: the backend is refused for every other phase, refused
together with a residue-type head, refused together with a FaMPNN checkpoint,
and the integrated checkpoint records `backend="absent"` rather than omitting
the field -- a reader can tell "no head" from "not written".
"""
import copy
import inspect
from pathlib import Path

import pytest
import torch

from pxdesign_train.stage4 import SUPERVISED_SC_PHASES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "training" / "train_protenix_monomer.py"


def _configs(backend="sc_only", phase="sc_warmup", **overrides):
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs
    t = copy.deepcopy(training_configs)
    t["enable_sidechain"] = True
    t["enable_residue_type_head"] = overrides.pop("enable_residue_type_head", False)
    t["residue_type"]["backend"] = backend
    t["stage4"]["phase"] = phase
    t["sidechain"].update(torsion_packer=True, packer_c_node=32, packer_c_pair=16,
                          packer_n_blocks=1, packer_n_heads=4, a_bs_concat=False, q_bs=False)
    t["sidechain"].update(overrides)
    cfg = parse_configs(t, arg_str="")
    cfg.load_strict = False
    return cfg


def test_sc_only_builds_no_aa_head_but_keeps_the_packing_forward():
    from pxdesign_train.model import ProtenixDesignTrain
    m = ProtenixDesignTrain(_configs())
    assert not hasattr(m, "aa_head")
    assert not hasattr(m, "design_residue_type_head")
    # The forward routing, the optimizer groups, the phase freeze and the
    # checkpoint identity all hang off this one predicate -- not off "is the AA
    # head FaMPNN", which is a different question.
    assert m.packing_stack is True
    assert m.aa_input_source == "diffusion_internal", "S_phi still reads a_token"


def test_sc_only_is_refused_for_any_phase_that_decodes():
    from pxdesign_train.model import ProtenixDesignTrain
    for phase in ("sc_adapt", "feedback_adapt", "joint_adapt", "IV-A"):
        assert phase not in SUPERVISED_SC_PHASES
        with pytest.raises(ValueError, match="sc_only"):
            ProtenixDesignTrain(_configs(phase=phase))


def test_sc_only_is_refused_together_with_a_residue_type_head():
    from pxdesign_train.model import ProtenixDesignTrain
    with pytest.raises(ValueError, match="contradictory"):
        ProtenixDesignTrain(_configs(enable_residue_type_head=True))


def test_absent_head_is_recorded_explicitly_not_omitted():
    from pxdesign_train.checkpoints import aa_head_identity

    class _NoHead:
        pass

    class _WithHead:
        class aa_head:
            identity = {"backend": "fampnn", "checkpoint_sha256": "abc"}

    absent = aa_head_identity(_NoHead())
    assert absent["backend"] == "absent" and "reason" in absent
    assert aa_head_identity(_WithHead())["backend"] == "fampnn"


def test_checkpoint_identity_survives_without_a_head():
    """`checkpoint_identity` is written into every saved checkpoint."""
    from pxdesign_train.model import ProtenixDesignTrain
    from pxdesign_train.stage4 import checkpoint_identity
    m = ProtenixDesignTrain(_configs())
    record = checkpoint_identity(m)
    assert record["backend"] == "absent"
    assert record["phase"] == "sc_warmup"


def test_cli_refuses_sc_only_with_a_decoding_phase_or_a_fampnn_checkpoint():
    src = SCRIPT.read_text()
    assert '"--aa-backend"' in src
    assert 'configs.residue_type.backend = "sc_only"' in src
    # Both refusals must be in the driver, not only in the model: a run that dies
    # after the data loader has warmed up has already burned the allocation.
    assert "decodes a sequence" in src
    assert "--aa-backend sc_only and --fampnn-checkpoint contradict each other" in src


def test_no_gate_still_asks_whether_the_aa_head_is_fampnn():
    """The trainer's branches mean "Stage IV forward", not "the head is FaMPNN".

    Leaving one of them testing the backend string would give sc_only runs a
    silently different optimizer, DDP setting or checkpoint record.
    """
    from pxdesign_train.runner import trainer
    src = inspect.getsource(trainer)
    assert 'aa_backend' not in src, "use model.packing_stack for Stage IV gates"
