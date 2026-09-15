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
from pxf.couple.trainer import CoupleSettings, CoupledBatch, CoupledTrainer
from pxf.train.trainer import OptimSettings

C_TOKEN = 384


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser
    bundle = torch.load(provenance.fampnn_checkpoint("0.0"), map_location="cpu",
                        weights_only=False)
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
        crop_size=32, noise=0.0, seed=0)
    item = collate([dataset[0]])
    length = item["aatype"].shape[1]
    aatype = item["aatype"][0].long()
    slots = list(atom37.BACKBONE_SLOTS)
    clean = item["x"][0][:, slots, :].reshape(-1, 3)
    topology = Topology(
        atom_names=[atom37.ATOM37[i] for i in slots] * length,
        atom_to_token_idx=[r for r in range(length) for _ in slots],
        num_tokens=length,
        res_names=[rc.restype_1to3[atom37.AA_ORDER[int(a)]] for a in aatype for _ in slots])
    return dict(item=item, length=length, aatype=aatype, clean=clean, topology=topology)


def stub_backbone(length, channels):
    """Proposal carries the noise, and the correction reaches the coordinates."""
    generator = torch.Generator().manual_seed(0)
    projection = torch.randn(9, channels, generator=generator) * (channels ** -0.5)

    def backbone(x_noisy, sigma, *, feedback=None):
        flat = x_noisy.reshape(1, length, 4, 3)
        features = flat.reshape(1, length, 12)[..., :9] @ projection
        if feedback is not None:
            return x_noisy + feedback.mean(), features
        return x_noisy, features
    return backbone


def make_trainer(fampnn, parts, phase, out_dir, *, perfect_proposal=False):
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)
    controller = CoupledDenoiser(stub_backbone(parts["length"], C_TOKEN), fampnn,
                                 adapters, phase=phase, pack_steps=3)
    trainer = CoupledTrainer(
        controller, out_dir=out_dir, optim=OptimSettings(lr=1e-2, warmup_steps=1),
        settings=CoupleSettings(phase=phase, max_steps=4, log_every=2,
                                checkpoint_every=0, pack_steps=3),
        frozen_identity={"fampnn": "0.0"})
    clean = parts["clean"]
    noisy = clean if perfect_proposal else clean + torch.randn_like(clean) * 0.5
    batch = CoupledBatch(
        topology=parts["topology"], x_noisy=noisy, sigma=torch.tensor([1.0]),
        aatype=parts["aatype"],
        sidechain_batch={k: v for k, v in parts["item"].items() if torch.is_tensor(v)},
        backbone_target=clean, name="T1031")
    return trainer, adapters, batch


def moved(adapter):
    """How far an adapter's zero-initialized output layer has travelled."""
    return float(adapter.project_out.weight.abs().sum()
                 + adapter.project_out.bias.abs().sum())


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
    trainer, adapters, batch = make_trainer(fampnn, batch_parts, "sc_to_bb", tmp_path,
                                            perfect_proposal=True)
    trainer.train((batch for _ in range(4)), progress=None)
    assert moved(adapters.sc_to_bb) == 0


def test_donors_must_be_frozen(fampnn, batch_parts, tmp_path):
    """A 'coupling' gain from an unfrozen donor is really fine-tuning."""
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)
    controller = CoupledDenoiser(stub_backbone(batch_parts["length"], C_TOKEN),
                                 fampnn, adapters, phase="bb_to_sc", pack_steps=3)
    fampnn.denoiser.scn_diffusion_module.scn_denoiser.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="frozen-component"):
            CoupledTrainer(controller, out_dir=tmp_path,
                           settings=CoupleSettings(phase="bb_to_sc"))
    finally:
        fampnn.requires_grad_(False)


def test_a_phase_with_nothing_trainable_is_refused(fampnn, batch_parts, tmp_path):
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)
    controller = CoupledDenoiser(stub_backbone(batch_parts["length"], C_TOKEN),
                                 fampnn, adapters, phase="frozen", pack_steps=3)
    with pytest.raises(ValueError, match="nothing to optimize"):
        CoupledTrainer(controller, out_dir=tmp_path,
                       settings=CoupleSettings(phase="frozen"))


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
