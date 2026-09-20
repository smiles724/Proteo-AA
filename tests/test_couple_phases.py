"""The staged phases must each move exactly the adapter they own.

The regression these guard against is subtle: a phase can appear to train --
losses logged, optimizer stepping, gradients clipped -- while the adapter it is
supposed to fit never moves, because the loss has no gradient path to it. That
happened twice while building this: once from a stub that multiplied the
feedback by zero, and once from a configuration where the proposal already
equalled the target, putting the zero-initialized adapter at an exact stationary
point of L_BB.
"""

import pytest
import torch

from pxf import atom37, provenance
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import CoupledDenoiser, Topology
from pxf.couple.trainer import CoupledBatch, CoupledTrainer, CoupleSettings
from pxf.train.trainer import OptimSettings

C_TOKEN = 384


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser

    bundle = torch.load(
        provenance.fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False
    )
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def batch_parts():
    from fampnn.data import residue_constants as rc
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    dataset = StructureCropDataset(
        [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")],
        crop_size=32,
        noise=0.0,
        seed=0,
    )
    item = collate([dataset[0]])
    length = item["aatype"].shape[1]
    aatype = item["aatype"][0].long()
    slots = list(atom37.BACKBONE_SLOTS)
    clean = item["x"][0][:, slots, :].reshape(-1, 3)
    topology = Topology(
        atom_names=[atom37.ATOM37[i] for i in slots] * length,
        atom_to_token_idx=[r for r in range(length) for _ in slots],
        num_tokens=length,
        res_names=[rc.restype_1to3[atom37.AA_ORDER[int(a)]] for a in aatype for _ in slots],
    )
    return dict(item=item, length=length, aatype=aatype, clean=clean, topology=topology)


def stub_backbone(length, channels):
    """Proposal carries the noise, and the correction reaches the coordinates."""
    generator = torch.Generator().manual_seed(0)
    projection = torch.randn(9, channels, generator=generator) * (channels**-0.5)

    def backbone(x_noisy, sigma, *, feedback=None):
        flat = x_noisy.reshape(1, length, 4, 3)
        features = flat.reshape(1, length, 12)[..., :9] @ projection
        if feedback is not None:
            return x_noisy + feedback.mean(), features
        return x_noisy, features

    return backbone


def make_trainer(fampnn, parts, phase, out_dir, *, perfect_proposal=False):
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V
    )
    controller = CoupledDenoiser(
        stub_backbone(parts["length"], C_TOKEN), fampnn, adapters, phase=phase, pack_steps=3
    )
    trainer = CoupledTrainer(
        controller,
        out_dir=out_dir,
        optim=OptimSettings(optimizer="adamw", lr=1e-2, warmup_steps=1),
        settings=CoupleSettings(
            phase=phase, max_steps=4, log_every=2, checkpoint_every=0, pack_steps=3
        ),
        frozen_identity={"fampnn": "0.0"},
    )
    clean = parts["clean"]
    noisy = clean if perfect_proposal else clean + torch.randn_like(clean) * 0.5
    batch = CoupledBatch(
        topology=parts["topology"],
        x_noisy=noisy,
        sigma=torch.tensor([1.0]),
        aatype=parts["aatype"],
        sidechain_batch={k: v for k, v in parts["item"].items() if torch.is_tensor(v)},
        backbone_target=clean,
        name="T1031",
    )
    return trainer, adapters, batch


def moved(adapter):
    """How far an adapter's zero-initialized output layer has travelled."""
    return float(
        adapter.project_out.weight.abs().sum() + adapter.project_out.bias.abs().sum()
    )


def test_phase_one_moves_only_a_bs(fampnn, batch_parts, tmp_path):
    trainer, adapters, batch = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path)
    trainer.train((batch for _ in range(4)), progress=None)
    assert moved(adapters.bb_to_sc) > 0
    assert moved(adapters.sc_to_bb) == 0


def test_phase_two_moves_only_a_sb(fampnn, batch_parts, tmp_path):
    """The case that silently failed: L_BB must reach A_SB."""
    trainer, adapters, batch = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path)
    trainer.train((batch for _ in range(4)), progress=None)
    assert moved(adapters.sc_to_bb) > 0, "L_BB has no gradient path to A_SB"
    assert moved(adapters.bb_to_sc) == 0


def test_phase_three_moves_both(fampnn, batch_parts, tmp_path):
    trainer, adapters, batch = make_trainer(fampnn, batch_parts, "joint", tmp_path)
    trainer.train((batch for _ in range(4)), progress=None)
    assert moved(adapters.bb_to_sc) > 0 and moved(adapters.sc_to_bb) > 0


def test_a_perfect_proposal_is_a_stationary_point(fampnn, batch_parts, tmp_path):
    """Documents why the smoke configuration must noise its input.

    With the proposal already equal to the target, L_BB is zero at zero-init and
    its gradient vanishes -- so a phase-2 run would look healthy and learn
    nothing. Asserted rather than commented, so the trap stays visible.
    """
    trainer, adapters, batch = make_trainer(
        fampnn, batch_parts, "sc_to_bb", tmp_path, perfect_proposal=True
    )
    trainer.train((batch for _ in range(4)), progress=None)
    assert moved(adapters.sc_to_bb) == 0


def test_donors_must_be_frozen(fampnn, batch_parts, tmp_path):
    """A 'coupling' gain from an unfrozen donor is really fine-tuning."""
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V
    )
    controller = CoupledDenoiser(
        stub_backbone(batch_parts["length"], C_TOKEN),
        fampnn,
        adapters,
        phase="bb_to_sc",
        pack_steps=3,
    )
    fampnn.denoiser.scn_diffusion_module.scn_denoiser.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="frozen-component"):
            CoupledTrainer(
                controller, out_dir=tmp_path, settings=CoupleSettings(phase="bb_to_sc")
            )
    finally:
        fampnn.requires_grad_(False)


def test_a_phase_with_nothing_trainable_is_refused(fampnn, batch_parts, tmp_path):
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V
    )
    controller = CoupledDenoiser(
        stub_backbone(batch_parts["length"], C_TOKEN),
        fampnn,
        adapters,
        phase="frozen",
        pack_steps=3,
    )
    with pytest.raises(ValueError, match="nothing to optimize"):
        CoupledTrainer(
            controller, out_dir=tmp_path, settings=CoupleSettings(phase="frozen")
        )


def test_a_batch_missing_its_target_is_refused(fampnn, batch_parts, tmp_path):
    trainer, _, batch = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path)
    batch.backbone_target = None
    with pytest.raises(ValueError, match="no backbone_target"):
        trainer.loss_for(batch, "backbone")
    batch.sidechain_batch = None
    with pytest.raises(ValueError, match="no sidechain_batch"):
        trainer.loss_for(batch, "sidechain")


def test_checkpoints_resume_and_refuse_mismatched_donors(fampnn, batch_parts, tmp_path):
    trainer, _, batch = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path)
    result = trainer.train((batch for _ in range(4)), progress=None)
    twin, _, _ = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path)
    assert twin.resume(result["checkpoint"]) == trainer.step
    other, _, _ = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path)
    other.frozen_identity = {"fampnn": "different"}
    with pytest.raises(ValueError, match="different frozen components"):
        other.resume(result["checkpoint"])


# --- crossing from one phase to the next ------------------------------------
#
# The suite covered same-phase resume only, and the gap hid a live bug: the
# launcher documented `PHASE=2 RESUME=<phase1 final.pt>`, which performed zero
# updates and reported success. Both failure modes are pinned below, plus the
# path that is actually correct.


def test_a_run_that_would_do_nothing_is_refused(fampnn, batch_parts, tmp_path):
    """The root cause, independent of how the step counter got there.

    A loaded checkpoint at or past max_steps made `train` break on its first
    batch, write a 'final' checkpoint and return `steps: <target>`. Nothing
    raised; the only sign was an empty train_log.jsonl.
    """
    trainer, _adapters, batch = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path)
    trainer.step = trainer.settings.max_steps
    with pytest.raises(ValueError, match="zero optimizer updates"):
        trainer.train((batch for _ in range(4)), progress=None)


def test_resuming_across_phases_is_refused(fampnn, batch_parts, tmp_path):
    """--resume is for one experiment; it does not transfer across phases."""
    first, _a1, b1 = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path / "p1")
    result = first.train((b1 for _ in range(4)), progress=None)
    second, _a2, _b2 = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path / "p2")
    with pytest.raises(ValueError, match="--init-from"):
        second.resume(result["checkpoint"])


def test_initializing_across_phases_trains_from_step_zero(fampnn, batch_parts, tmp_path):
    """The correct chain: phase 1's A_BS inherited, A_SB fresh, step 0."""
    first, adapters1, b1 = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path / "p1")
    result = first.train((b1 for _ in range(4)), progress=None)
    assert moved(adapters1.bb_to_sc) > 0 and moved(adapters1.sc_to_bb) == 0

    second, adapters2, b2 = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path / "p2")
    record = second.initialize_from(result["checkpoint"])
    assert record["source_phase"] == "bb_to_sc"
    assert record["prefixes"] == ["bb_to_sc."]
    assert record["step"] == 0 and record["optimizer_reset"] is True
    # A_BS came across; A_SB is still the exact no-op it starts as.
    assert moved(adapters2.bb_to_sc) > 0
    assert moved(adapters2.sc_to_bb) == 0

    # And phase 2 then actually runs, which the --resume path did not.
    second.train((b2 for _ in range(4)), progress=None)
    assert second.step == second.settings.max_steps
    assert moved(adapters2.sc_to_bb) > 0, "phase 2 performed no updates"


def test_initialization_does_not_inherit_the_previous_optimizer(
    fampnn, batch_parts, tmp_path
):
    """The second failure: identically shaped adapters load by position.

    A_BS and A_SB are both six-parameter ResidualAdapters, so
    optimizer.load_state_dict matched them without complaint and phase 1's AdamW
    moments drove A_SB. initialize_from rebuilds the optimizer instead.
    """
    first, _a1, b1 = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path / "p1")
    result = first.train((b1 for _ in range(4)), progress=None)
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    phase1_moments = [
        float(v["exp_avg"].abs().sum()) for v in saved["optimizer"]["state"].values()
    ]
    assert any(m > 0 for m in phase1_moments), "phase 1 accumulated no moments"

    second, _a2, _b2 = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path / "p2")
    second.initialize_from(result["checkpoint"])
    assert second.optimizer.state_dict()["state"] == {}, (
        "the new phase started with optimizer moments it did not earn"
    )


def test_initialization_records_what_it_inherited(fampnn, batch_parts, tmp_path):
    """A checkpoint has to say which policy it was held fixed against."""
    first, _a1, b1 = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path / "p1")
    result = first.train((b1 for _ in range(4)), progress=None)
    second, _a2, b2 = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path / "p2")
    second.initialize_from(result["checkpoint"])
    written = second.train((b2 for _ in range(4)), progress=None)
    state = torch.load(written["checkpoint"], map_location="cpu", weights_only=False)
    record = state["initialized_from"]
    assert record["source_phase"] == "bb_to_sc"
    assert record["source"] == str(result["checkpoint"])
    assert record["source_step"] == first.settings.max_steps


def test_a_source_phase_that_trains_nothing_is_refused(fampnn, batch_parts, tmp_path):
    trainer, _adapters, batch = make_trainer(fampnn, batch_parts, "bb_to_sc", tmp_path)
    result = trainer.train((batch for _ in range(4)), progress=None)
    path = tmp_path / "frozen.pt"
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    state["settings"] = dict(state["settings"], phase="frozen")
    torch.save(state, path)
    other, _a, _b = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path / "p2")
    with pytest.raises(ValueError, match="trains no adapter direction"):
        other.initialize_from(path)
