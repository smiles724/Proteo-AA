"""The coupled cycle, against the real FaMPNN and a stub backbone.

The backbone is stubbed on purpose: the controller's logic is independent of how
PXDesign is driven, and PXDesign's own inference runner cannot generate monomers
at all, so the driver is expected to change. What must hold regardless is
phase-0 equivalence and the gradient routing the staged plan depends on.
"""

import pytest
import torch

from pxf import atom37, provenance
from pxf.couple import fampnn_iface as iface
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import PHASES, CoupledDenoiser, GradientPolicy, Topology

C_TOKEN = 384
PACK_STEPS = 3


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
def case():
    """A short native crop plus the flat-atom topology a backbone would emit."""
    from fampnn.data import residue_constants as rc
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    dataset = StructureCropDataset(
        [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")],
        crop_size=32,
        noise=0.0,
        seed=0,
    )
    batch = collate([dataset[0]])
    length = batch["aatype"].shape[1]
    aatype = batch["aatype"][0].long()
    slots = list(atom37.BACKBONE_SLOTS)
    names = [atom37.ATOM37[i] for i in slots] * length
    tokens = [residue for residue in range(length) for _ in slots]
    flat = batch["x"][0][:, slots, :].reshape(-1, 3)
    topology = Topology(
        atom_names=names,
        atom_to_token_idx=tokens,
        num_tokens=length,
        res_names=[rc.restype_1to3[atom37.AA_ORDER[int(a)]] for a in aatype for _ in slots],
    )
    return dict(batch=batch, length=length, aatype=aatype, flat=flat, topology=topology)


def make_backbone(length, record):
    a_token = torch.zeros(1, length, C_TOKEN)

    def backbone(x_noisy, sigma, *, feedback=None):
        record["calls"] += 1
        if feedback is not None:
            record["feedback"] += 1
            return x_noisy + feedback.mean(), a_token
        return x_noisy, a_token

    return backbone, a_token


def build(fampnn, case, phase="joint", **kwargs):
    record = {"calls": 0, "feedback": 0}
    backbone, _ = make_backbone(case["length"], record)
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V, **kwargs
    )
    controller = CoupledDenoiser(
        backbone, fampnn, adapters, phase=phase, pack_steps=PACK_STEPS
    )
    return controller, adapters, record


def test_the_cycle_runs_both_directions(fampnn, case):
    controller, _, record = build(fampnn, case)
    torch.manual_seed(0)
    out = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert out.bb0_dense.shape == (1, case["length"], 37, 3)
    assert out.h_base.shape[:2] == (1, case["length"])
    assert out.sidechains.shape == (1, case["length"], 33, 3)
    assert out.h_packed is not None and out.ran_feedback
    # Two backbone evaluations, the second carrying the feedback -- the two
    # PXDesign passes the architecture specifies.
    assert record["calls"] == 2 and record["feedback"] == 1


def test_phase_zero_equivalence_is_exact(fampnn, case):
    """Zero-init adapters must reproduce the uncoupled packing bit for bit."""
    controller, adapters, _ = build(fampnn, case)
    assert adapters.is_identity()
    torch.manual_seed(0)
    coupled = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    inputs = coupled.aux["inputs"]
    _, _, features = iface.encode(
        fampnn,
        coupled.bb0_dense,
        inputs.aatype,
        seq_mask=inputs.seq_mask,
        missing_atom_mask=inputs.missing_atom_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
    )
    torch.manual_seed(0)
    reference, _ = iface.pack_from_features(
        fampnn,
        features,
        inputs.aatype,
        seq_mask=inputs.seq_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        num_steps=PACK_STEPS,
    )
    torch.manual_seed(0)
    again = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert torch.allclose(again.sidechains, reference, atol=1e-6)
    assert bool((again.delta_h == 0).all()) and bool((again.delta_a == 0).all())
    assert torch.equal(again.h_cond, again.h_base)


def test_bb_to_sc_gradients_reach_only_a_bs(fampnn, case):
    controller, adapters, _ = build(fampnn, case, phase="bb_to_sc")
    torch.manual_seed(0)
    out = controller.forward(
        case["topology"],
        case["flat"],
        torch.tensor([1.0]),
        case["aatype"],
        run_feedback=False,
    )
    out.sidechains.sum().backward()
    into_bs = sum(
        float(p.grad.abs().sum())
        for p in adapters.bb_to_sc.parameters()
        if p.grad is not None
    )
    into_sb = sum(
        float(p.grad.abs().sum())
        for p in adapters.sc_to_bb.parameters()
        if p.grad is not None
    )
    assert into_bs > 0 and into_sb == 0


def test_packing_is_detached_from_the_feedback_branch(fampnn, case):
    """Phase 2 must not backpropagate L_BB through the side-chain sampler."""
    controller, adapters, _ = build(fampnn, case, phase="sc_to_bb")
    assert controller.policy.detach_sidechains and controller.policy.detach_h_packed
    torch.manual_seed(0)
    out = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert not out.h_packed.requires_grad, "h_packed should enter A_SB detached"


def test_disabled_directions_short_circuit(fampnn, case):
    controller, _, record = build(fampnn, case, phase="joint", enable_sc_to_bb=False)
    torch.manual_seed(0)
    out = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert out.delta_a is None and out.bb1_flat is None and not out.ran_feedback
    assert record["calls"] == 1, "no second backbone pass without feedback"

    controller, _, _ = build(fampnn, case, phase="joint", enable_bb_to_sc=False)
    torch.manual_seed(0)
    out = controller.forward(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert out.delta_h is None and torch.equal(out.h_cond, out.h_base)


def test_a_backbone_without_token_features_is_refused(fampnn, case):
    adapters = CouplingAdapters(
        C_TOKEN, fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V
    )
    controller = CoupledDenoiser(
        lambda x, s, feedback=None: (x, None), fampnn, adapters, pack_steps=PACK_STEPS
    )
    with pytest.raises(ValueError, match="no token features"):
        controller.forward(
            case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
        )


def test_every_phase_has_a_gradient_policy():
    for phase in PHASES:
        assert isinstance(GradientPolicy.for_phase(phase), GradientPolicy)
    with pytest.raises(ValueError, match="Unknown phase"):
        GradientPolicy.for_phase("phase4")


def test_identity_is_serializable(fampnn, case):
    import json

    controller, _, _ = build(fampnn, case)
    json.loads(json.dumps(controller.identity()))


# --- the shuffled-a_token control -------------------------------------------


def test_an_a_token_override_changes_the_residual_but_not_the_record(fampnn, case):
    """The information-content control must alter one thing and log honestly.

    Substituting a donor protein's token features has to reach ``delta_h`` --
    otherwise the control is inert and would fake a null result -- while
    ``out.a_token`` keeps this structure's own features, so a shuffled run
    cannot be mistaken for a real one by reading the cycle output.
    """
    controller, adapters, _record = build(fampnn, case, phase="bb_to_sc")
    # A trained adapter: zero-initialized weights would make every residual zero
    # and the comparison vacuous.
    with torch.no_grad():
        for parameter in adapters.bb_to_sc.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    args = (case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"])
    torch.manual_seed(0)
    plain = controller.forward(*args, run_feedback=False)
    donor = torch.randn_like(plain.a_token)
    torch.manual_seed(0)
    swapped = controller.forward(*args, run_feedback=False, a_token_override=donor)

    assert not torch.allclose(plain.delta_h, swapped.delta_h), (
        "the override never reached A_BS, so the control cannot detect anything"
    )
    assert torch.allclose(plain.a_token, swapped.a_token), (
        "out.a_token was overwritten; a shuffled run would look like a real one"
    )
    # The backbone proposal is the recipient's own either way, so the arms stay
    # comparable -- only the residual moved.
    assert torch.allclose(plain.bb0_flat, swapped.bb0_flat)


def test_no_override_is_identical_to_passing_none(fampnn, case):
    controller, _adapters, _record = build(fampnn, case, phase="bb_to_sc")
    args = (case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"])
    torch.manual_seed(0)
    a = controller.forward(*args, run_feedback=False)
    torch.manual_seed(0)
    b = controller.forward(*args, run_feedback=False, a_token_override=None)
    assert torch.allclose(a.bb0_dense, b.bb0_dense)


# --- shared proposal, packed per arm ----------------------------------------


def _fingerprint(proposal):
    """Every shared tensor, cloned, so mutation during packing is detectable."""
    import torch as _t

    out = {
        "bb0_flat": proposal.bb0_flat.clone(),
        "a_token": proposal.a_token.clone(),
        "h_base": proposal.h_base.clone(),
        "coords_af2": proposal.inputs.coords_af2.clone(),
    }
    out.update(
        {
            f"features.{k}": v.clone()
            for k, v in proposal.features.items()
            if _t.is_tensor(v)
        }
    )
    return out


def test_packing_does_not_mutate_the_shared_proposal(fampnn, case):
    """Arms must be independent; a mutated proposal would couple them silently."""
    controller, adapters, _record = build(fampnn, case, phase="bb_to_sc")
    with torch.no_grad():
        for parameter in adapters.bb_to_sc.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    proposal = controller.propose(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    before = _fingerprint(proposal)
    torch.manual_seed(0)
    controller.pack_proposal(proposal, run_feedback=False)
    torch.manual_seed(0)
    controller.pack_proposal(
        proposal, a_token_override=torch.randn_like(proposal.a_token), run_feedback=False
    )
    after = _fingerprint(proposal)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"packing mutated the shared {key}"


def test_both_arms_see_a_bit_identical_backbone(fampnn, case):
    """The point of sharing: no reliance on the backbone being deterministic."""
    controller, adapters, _record = build(fampnn, case, phase="bb_to_sc")
    with torch.no_grad():
        for parameter in adapters.bb_to_sc.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    proposal = controller.propose(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    adapters.enable_bb_to_sc = True
    torch.manual_seed(0)
    coupled = controller.pack_proposal(proposal, run_feedback=False)
    adapters.enable_bb_to_sc = False
    torch.manual_seed(0)
    uncoupled = controller.pack_proposal(proposal, run_feedback=False)

    assert torch.equal(coupled.bb0_dense, uncoupled.bb0_dense)
    assert torch.equal(coupled.a_token, uncoupled.a_token)
    # ... and the residual is the only thing that actually differed.
    assert uncoupled.delta_h is None and coupled.delta_h is not None
    assert not torch.allclose(coupled.sidechains, uncoupled.sidechains)


def test_propose_then_pack_reproduces_forward(fampnn, case):
    """The split must be a refactor, not a behaviour change."""
    controller, _adapters, _record = build(fampnn, case, phase="bb_to_sc")
    args = (case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"])
    torch.manual_seed(3)
    whole = controller.forward(*args, run_feedback=False)
    proposal = controller.propose(*args)
    torch.manual_seed(3)
    split = controller.pack_proposal(proposal, run_feedback=False)
    assert torch.equal(whole.bb0_dense, split.bb0_dense)
    assert torch.equal(whole.sidechains, split.sidechains)


def test_one_proposal_serves_many_arms_without_recomputing_the_backbone(fampnn, case):
    """Backbone calls must not scale with the number of arms."""
    controller, _adapters, record = build(fampnn, case, phase="bb_to_sc")
    before = record["calls"]
    proposal = controller.propose(
        case["topology"], case["flat"], torch.tensor([1.0]), case["aatype"]
    )
    assert record["calls"] == before + 1
    for _ in range(3):
        controller.pack_proposal(proposal, run_feedback=False)
    assert record["calls"] == before + 1, "packing re-ran the backbone"
