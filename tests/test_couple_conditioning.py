"""The early conditioners: the contract at the site, and what each arm may read.

Three families.

**The site.** Zero feedback must reproduce the unmodified denoiser exactly, the
payload must be refused at the wrong width instead of broadcast, and the two
injection sites must be mutually exclusive -- an arm cannot quietly be applied
at both. These run against a stub diffusion module, because they are argument
plumbing and should not need a GPU or a 500 MB donor.

**What the arms read.** E1's controls must be blind to a rotamer change and its
candidate must not be; E2 must be invariant to a rigid motion of the whole
structure, must be unable to see atom slots that do not exist, and its BB-only
arm must be unchanged by a rotamer flip in both inputs and outputs. These use a
real packed structure through the real packer, so a feature that only looks
invariant on synthetic coordinates fails here.

**Gradients and checkpoints.** The output projections are zero, so the encoders
have no gradient until the projection has moved -- checked in that order, since
checking the encoder first would look like a broken graph. And a checkpoint
whose architecture metadata disagrees with the module built to load it must be
refused: E1's ``full`` and ``bb_only`` have identical parameter shapes by
design, so ``strict=True`` cannot tell them apart.
"""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from pxf import atom37, provenance
from pxf.couple import conditioning as cond
from pxf.couple import fampnn_iface as iface
from pxf.couple import frames, torsions
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import CoupledDenoiser, Topology
from pxf.couple.pxdesign_iface import (
    BackboneTap,
    ConditioningFeedback,
    conditioning_widths,
)
from pxf.couple.readout import SigmaWindow

C_TOKEN = 384
C_S = 48
C_Z = 16
LENGTH = 6


# --- the site ---------------------------------------------------------------


class FakeConditioning(nn.Module):
    """Returns ``(s_single, z_pair)`` in the runtime's own singleton shapes."""

    def __init__(self, c_s=C_S, c_z=C_Z, length=LENGTH):
        super().__init__()
        self.c_s, self.c_z = c_s, c_z
        self.single = torch.randn(1, length, c_s)
        self.pair = torch.randn(length, length, c_z)

    def forward(self):
        return self.single.clone(), self.pair.clone()


class FakeDecoder(nn.Module):
    def forward(self, atom_to_token_idx=None, a=None, **kwargs):
        return a


class FakeDiffusionModule(nn.Module):
    def __init__(self, width=C_TOKEN):
        super().__init__()
        self.c_token = width
        self.layernorm_a = nn.LayerNorm(width)
        self.atom_attention_decoder = FakeDecoder()
        self.diffusion_conditioning = FakeConditioning()


@pytest.fixture
def module():
    torch.manual_seed(0)
    return FakeDiffusionModule()


def test_the_widths_come_from_the_model_and_are_not_c_token(module):
    assert conditioning_widths(module) == (C_S, C_Z)
    assert module.c_token not in (C_S, C_Z)


def test_zero_feedback_reproduces_the_conditioning_exactly(module):
    """The wiring check, at the early site. Bit-for-bit, not within a tolerance."""
    with BackboneTap(module) as tap:
        before_s, before_z = module.diffusion_conditioning()
        tap.feedback = ConditioningFeedback(
            delta_single=torch.zeros(1, LENGTH, C_S),
            delta_pair=torch.zeros(LENGTH, LENGTH, C_Z),
        )
        after_s, after_z = module.diffusion_conditioning()
    assert torch.equal(before_s, after_s)
    assert torch.equal(before_z, after_z)
    assert tap.conditioning_injections == 1


def test_the_residuals_are_added_where_they_are_meant_to_be(module):
    delta_s = torch.randn(1, LENGTH, C_S)
    delta_z = torch.randn(LENGTH, LENGTH, C_Z)
    with BackboneTap(module) as tap:
        base_s, base_z = module.diffusion_conditioning()
        tap.feedback = ConditioningFeedback(delta_single=delta_s, delta_pair=delta_z)
        got_s, got_z = module.diffusion_conditioning()
    assert torch.allclose(got_s, base_s + delta_s)
    assert torch.allclose(got_z, base_z + delta_z)


def test_a_c_token_wide_payload_is_refused(module):
    """The mistake the width check exists for: c_s is not c_token."""
    with BackboneTap(module) as tap:
        tap.feedback = ConditioningFeedback(
            delta_single=torch.zeros(1, LENGTH, C_TOKEN)
        )
        with pytest.raises(ValueError, match="c_token is not either of them"):
            module.diffusion_conditioning()


def test_a_wrong_length_payload_is_refused(module):
    with BackboneTap(module) as tap:
        tap.feedback = ConditioningFeedback(
            delta_single=torch.zeros(1, LENGTH + 1, C_S)
        )
        with pytest.raises(ValueError, match="delta_single addresses"):
            module.diffusion_conditioning()


def test_several_diffusion_samples_are_refused_rather_than_broadcast():
    """One protein and one sample per call is the scope, and it is enforced."""
    torch.manual_seed(0)
    module = FakeDiffusionModule()
    module.diffusion_conditioning.single = torch.randn(2, LENGTH, C_S)
    with BackboneTap(module) as tap:
        tap.feedback = ConditioningFeedback(delta_single=torch.zeros(1, LENGTH, C_S))
        with pytest.raises(ValueError, match="non-singleton leading dimensions"):
            module.diffusion_conditioning()


def test_the_two_injection_sites_are_mutually_exclusive(module):
    """A ConditioningFeedback must never also perturb a_token."""
    with BackboneTap(module) as tap:
        tap.feedback = ConditioningFeedback(delta_single=torch.ones(1, LENGTH, C_S))
        assert (
            tap._inject(module.atom_attention_decoder, (), {"a": torch.zeros(1)}) is None
        )
        module.diffusion_conditioning()
    assert tap.injections == 0
    assert tap.conditioning_injections == 1


def test_the_hook_is_removed_with_the_context(module):
    tap = BackboneTap(module)
    with tap:
        pass
    tap.feedback = ConditioningFeedback(delta_single=torch.ones(1, LENGTH, C_S))
    before, _z = module.diffusion_conditioning()
    assert torch.equal(before, module.diffusion_conditioning()[0])


def test_the_payload_reaches_the_conditioner_through_the_hook(module):
    """The gradient path the corrective loss depends on, at the early site."""
    head = nn.Linear(C_S, C_S)
    with BackboneTap(module) as tap:
        tap.feedback = ConditioningFeedback(
            delta_single=head(torch.ones(1, LENGTH, C_S))
        )
        single, _pair = module.diffusion_conditioning()
        single.sum().backward()
    assert head.weight.grad is not None
    assert float(head.weight.grad.abs().sum()) > 0


# --- the arms ---------------------------------------------------------------


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser

    bundle = torch.load(
        provenance.fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False
    )
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def packed(fampnn):
    """A real packed structure: real converter, real packer, real re-encode."""
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
    topology = Topology(
        atom_names=[atom37.ATOM37[i] for i in slots] * length,
        atom_to_token_idx=[r for r in range(length) for _ in slots],
        num_tokens=length,
        res_names=[rc.restype_1to3[atom37.AA_ORDER[int(a)]] for a in aatype for _ in slots],
    )

    def backbone(x_noisy, sigma, *, feedback=None):
        return x_noisy, torch.zeros(1, length, C_TOKEN)

    controller = CoupledDenoiser(
        backbone,
        fampnn,
        CouplingAdapters(C_TOKEN, iface.node_feature_dim(fampnn)),
        phase="sc_to_bb",
        pack_steps=3,
    )
    flat = item["x"][0][:, slots, :].reshape(-1, 3)
    proposal = controller.propose(topology, flat, torch.tensor([1.0]), aatype)
    torch.manual_seed(0)
    sidechains, aux = iface.pack_from_features(
        fampnn,
        proposal.features,
        proposal.inputs.aatype,
        seq_mask=proposal.inputs.seq_mask,
        residue_index=proposal.inputs.residue_index,
        chain_index=proposal.inputs.chain_index,
        num_steps=3,
    )
    packed = controller.encode_predicted_packing(
        proposal.inputs, sidechains, h_base=proposal.h_base, psce=aux.get("psce")
    )
    # Fill the sequence-attribution encodings through the real helper, so the
    # arms that read them are exercised against what the pipeline actually
    # produces rather than against a stand-in.
    return controller.encode_sequence_controls(proposal.inputs, packed).detach()


@pytest.fixture(scope="module")
def c_h_V(fampnn):
    return iface.node_feature_dim(fampnn)


SIGMA = torch.tensor([0.5])
GATE = SigmaWindow(0.1, 2.0, taper=2.0)


def e1(c_h_V, variant):
    torch.manual_seed(0)
    return cond.EarlySingleConditioner(c_h_V, C_S, variant=variant, gate=GATE).eval()


def e2(variant="full", pair=True):
    torch.manual_seed(0)
    return cond.AtomConditioner(C_S, C_Z, variant=variant, pair=pair, gate=GATE).eval()


def rotamer_flip(packed, degrees=90.0, seed=0):
    deltas = torsions.random_chi_deltas(
        packed.aatype,
        math.radians(degrees),
        generator=torch.Generator().manual_seed(seed),
    )
    moved = torsions.perturb_chi(
        packed.coords37, packed.aatype, deltas, available=packed.available
    )
    return replace(packed, coords37=moved)


def test_every_arm_starts_as_an_exact_no_op(packed, c_h_V):
    """Step 0 must reproduce PXDesign, or the pilot has no baseline."""
    for arm in cond.EARLY_ARMS:
        module = cond.build_conditioner(
            arm, c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
        ).eval()
        assert module.is_identity(), arm
        with torch.no_grad():
            feedback, _stats = module(packed, SIGMA)
        assert float(feedback.delta_single.abs().max()) == 0.0, arm
        if feedback.delta_pair is not None:
            assert float(feedback.delta_pair.abs().max()) == 0.0, arm


def test_the_payload_has_the_conditioning_widths_not_c_token(packed, c_h_V):
    length = packed.coords37.shape[1]
    single = e1(c_h_V, "full")(packed, SIGMA)[0]
    assert single.delta_single.shape == (1, length, C_S)
    assert single.delta_pair is None
    both = e2()(packed, SIGMA)[0]
    assert both.delta_single.shape == (1, length, C_S)
    assert both.delta_pair.shape == (length, length, C_Z)
    assert e2(pair=False)(packed, SIGMA)[0].delta_pair is None


def _response(module, packed, moved):
    """Relative change of ``delta_single`` under an intervention, unmasked."""
    with torch.no_grad():
        # The output projection is zero at initialization, so a response would be
        # zero for every arm and the test would pass vacuously. Give it weights.
        before = module(packed, SIGMA)[0].delta_single
        after = module(moved, SIGMA)[0].delta_single
    return float((after - before).norm() / before.norm().clamp_min(1e-12))


def _wake(module):
    """Move the zero-initialized output projections off zero."""
    with torch.no_grad():
        for head in (getattr(module, "single_head", None), getattr(module, "pair_head", None)):
            if head is not None:
                head[-1].weight.normal_(0.0, 0.05)
                head[-1].bias.normal_(0.0, 0.05)
    return module


def test_e1_full_responds_to_a_rotamer_change(packed, c_h_V):
    module = _wake(e1(c_h_V, "full"))
    assert _response(module, packed, rotamer_flip(packed)) > 1e-3


@pytest.mark.parametrize("variant", ("bb_only", "generic"))
def test_e1_controls_are_blind_to_a_rotamer_change(packed, c_h_V, variant):
    module = _wake(e1(c_h_V, variant))
    assert _response(module, packed, rotamer_flip(packed)) == pytest.approx(0.0, abs=1e-12)


def test_e2_full_responds_to_a_rotamer_change(packed):
    module = _wake(e2())
    assert _response(module, packed, rotamer_flip(packed)) > 1e-3


def test_e2_bb_only_is_unchanged_by_a_rotamer_change(packed):
    """Inputs and outputs both: the control must not see the side chains at all."""
    module = _wake(e2(variant="bb_only"))
    moved = rotamer_flip(packed)
    with torch.no_grad():
        before, _s = module(packed, SIGMA)
        after, _s = module(moved, SIGMA)
    assert torch.equal(before.delta_single, after.delta_single)
    assert torch.equal(before.delta_pair, after.delta_pair)


def test_e2_is_invariant_to_a_rigid_motion(packed):
    """Every E2 feature is expressed in residue i's own frame, so this must hold."""
    module = _wake(e2())
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = replace(
        packed,
        coords37=packed.coords37 @ rotation.T + torch.tensor([3.0, -7.0, 11.0]),
    )
    with torch.no_grad():
        before, _s = module(packed, SIGMA)
        after, _s = module(moved, SIGMA)
    assert torch.allclose(before.delta_single, after.delta_single, atol=1e-4)
    assert torch.allclose(before.delta_pair, after.delta_pair, atol=1e-4)


def test_e2_cannot_see_atom_slots_that_do_not_exist(packed):
    """Absent slots are excluded from the pools, not fed in as zeros or origins."""
    module = _wake(e2())
    absent = packed.available <= 0
    garbage = packed.coords37.clone()
    garbage[absent] = torch.randn(int(absent.sum()), 3) * 40.0
    with torch.no_grad():
        before, _s = module(packed, SIGMA)
        after, _s = module(replace(packed, coords37=garbage), SIGMA)
    assert torch.equal(before.delta_single, after.delta_single)
    assert torch.equal(before.delta_pair, after.delta_pair)


def test_e2_pair_output_is_confined_to_the_neighbour_graph(packed):
    module = _wake(e2())
    with torch.no_grad():
        feedback, stats = module(packed, SIGMA)
    written = (feedback.delta_pair.abs().sum(-1) > 0).sum()
    length = packed.coords37.shape[1]
    assert stats["edges"] > 0
    assert int(written) <= stats["edges"]
    assert float(feedback.delta_pair.diagonal(dim1=0, dim2=1).abs().max()) == 0.0
    assert feedback.delta_pair.shape == (length, length, C_Z)


def test_e2_pair_output_is_directed(packed):
    """V_ij uses residue i's frame, so V_ji is a different quantity."""
    module = _wake(e2())
    with torch.no_grad():
        pair = module(packed, SIGMA)[0].delta_pair
    both = (pair.abs().sum(-1) > 0) & (pair.abs().sum(-1) > 0).T
    assert bool(both.any()), "no reciprocal edge to compare"
    assert not torch.allclose(pair[both], pair.transpose(0, 1)[both])


def test_invalid_frames_produce_masked_zeros_and_not_nans(packed):
    """A predicted backbone can collapse a residue's N/CA/C onto one point."""
    module = _wake(e2())
    broken = packed.coords37.clone()
    n, ca, c = cond.frames_slots()
    broken[0, 0, [n, ca, c], :] = broken[0, 0, ca, :].clone()
    visibility = replace(
        packed.visibility,
        frame_valid=packed.visibility.frame_valid.clone(),
    )
    visibility.frame_valid[0, 0] = False
    with torch.no_grad():
        feedback, _stats = module(
            replace(packed, coords37=broken, visibility=visibility), SIGMA
        )
    assert torch.isfinite(feedback.delta_single).all()
    assert torch.isfinite(feedback.delta_pair).all()
    assert float(feedback.delta_single[0, 0].abs().max()) == 0.0
    assert float(feedback.delta_pair[0].abs().max()) == 0.0
    assert float(feedback.delta_pair[:, 0].abs().max()) == 0.0


def test_the_gate_is_one_inside_the_window_and_tapers_outside(packed, c_h_V):
    module = _wake(e1(c_h_V, "full"))
    with torch.no_grad():
        inside = module(packed, torch.tensor([0.5]))[0].delta_single.norm()
        outside = module(packed, torch.tensor([20.0]))[0].delta_single.norm()
    assert float(inside) > 0
    assert float(outside) == 0.0


# --- gradients --------------------------------------------------------------


@pytest.mark.parametrize("arm", cond.EARLY_ARMS)
def test_the_output_projection_moves_before_the_encoders_do(packed, c_h_V, arm):
    """Zero output projections mean the encoders have no gradient on step one.

    Checked in this order on purpose. An encoder with no gradient at
    initialization is the *expected* state, not a broken graph, and testing it
    first would invite 'fixing' the zero initialization that makes step 0 an
    exact no-op.
    """
    module = cond.build_conditioner(
        arm, c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
    )
    feedback, _stats = module(packed, SIGMA)
    total = feedback.delta_single.sum()
    if feedback.delta_pair is not None:
        total = total + feedback.delta_pair.sum()
    total.backward()
    final = module.single_head[-1]
    assert float(final.weight.grad.abs().sum()) > 0, "the head itself got no gradient"

    module.zero_grad(set_to_none=True)
    _wake(module)
    feedback, _stats = module(packed, SIGMA)
    total = feedback.delta_single.sum()
    if feedback.delta_pair is not None:
        total = total + feedback.delta_pair.sum()
    total.backward()
    trained = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
    ]
    encoders = [name for name in trained if "head" not in name]
    if arm == "early_s_generic":
        # By construction: every readout group is replaced by `zeros_like`, which
        # is not connected to the graph, so the LayerNorm and the sequence
        # embedding cannot receive a gradient. That is what "sigma only" means
        # here, and an encoder gradient would mean the control is reading
        # something.
        assert not encoders, f"the sigma-only control trained {encoders}"
    else:
        assert encoders, (
            f"{arm}: only the heads got a gradient once the projection had "
            f"moved: {trained}"
        )


def test_the_frozen_upstream_is_not_mutated_by_any_arm(packed, c_h_V):
    """The cached frozen half is shared across arms; an in-place write would leak."""
    before = {
        name: getattr(packed, name).clone()
        for name in ("coords37", "h_packed", "aatype", "seq_mask")
    }
    for arm in cond.EARLY_ARMS:
        module = _wake(
            cond.build_conditioner(
                arm, c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
            ).eval()
        )
        with torch.no_grad():
            module(packed, SIGMA)
        for name, value in before.items():
            assert torch.equal(getattr(packed, name), value), f"{arm} mutated {name}"


# --- checkpoint metadata ----------------------------------------------------


def test_a_checkpoint_from_another_architecture_is_refused(c_h_V):
    early = e1(c_h_V, "full")
    atoms = e2()
    with pytest.raises(ValueError, match="same parameter shapes"):
        cond.check_compatible(atoms.identity(), early.identity())


def test_a_checkpoint_from_the_control_is_refused_by_its_candidate(c_h_V):
    """The load that strict=True cannot catch: identical shapes, different arm."""
    full, control = e1(c_h_V, "full"), e1(c_h_V, "bb_only")
    assert [p.shape for p in full.parameters()] == [p.shape for p in control.parameters()]
    control.load_state_dict(full.state_dict(), strict=True)  # succeeds, and is wrong
    with pytest.raises(ValueError, match="variant"):
        cond.check_compatible(full.identity(), control.identity())


def test_the_single_only_ablation_is_not_the_pair_arm():
    with pytest.raises(ValueError, match="pair"):
        cond.check_compatible(e2(pair=True).identity(), e2(pair=False).identity())


def test_an_identity_maps_back_to_the_arm_that_produced_it(c_h_V):
    for arm in cond.ARMS:
        module = cond.build_conditioner(
            arm, c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
        )
        assert cond.arm_for(module.identity()) == arm


def test_a_record_that_names_no_arm_is_refused(c_h_V):
    identity = dict(e1(c_h_V, "full").identity(), variant="not_an_arm")
    assert cond.arm_for(identity) is None
    with pytest.raises(ValueError, match="not a row of ARMS"):
        cond.check_is_a_known_arm(identity)


def test_a_checkpoint_is_held_to_the_label_it_was_given(c_h_V):
    """The mistake the load-time check actually catches, and the one it cannot.

    It cannot cross-examine the metadata against the weights: the metadata is
    the *only* record of which arm produced them, since the controls are the
    same shapes on purpose. What it can catch is the caller's claim --
    ``--checkpoint early_s_full=<the control's file>`` is a command-line slip
    that otherwise yields a completely self-consistent run with the wrong names
    on the results table.
    """
    control = e1(c_h_V, "bb_only").identity()
    assert cond.check_is_the_expected_arm(control, "early_s_bb_only") == "early_s_bb_only"
    with pytest.raises(ValueError, match="labelled 'early_s_full'"):
        cond.check_is_the_expected_arm(control, "early_s_full")
    # A label that is not an arm name is a free-form column heading, not a claim.
    assert cond.check_is_the_expected_arm(control, "control") == "early_s_bb_only"


def test_every_named_arm_records_its_architecture(c_h_V):
    for arm, spec in cond.ARMS.items():
        module = cond.build_conditioner(
            arm, c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
        )
        identity = module.identity()
        assert identity["variant"] == spec["variant"], arm
        if spec["arch"] != "late":
            assert identity["arch"] == spec["arch"], arm
            assert identity["pair"] == spec["pair"], arm
            assert identity["version"] == cond.CONDITIONER_VERSION


# --- the feature definition a checkpoint was trained against ----------------


def test_configurable_settings_are_reconstructed_from_the_checkpoint():
    """A loaded arm must be the function that was TRAINED, not today's defaults."""
    trained = cond.AtomConditioner(
        C_S, C_Z, variant="full", pair=True, gate=GATE,
        max_neighbours=8, neighbour_radius=15.0, max_atom_pairs=12,
        atom_pair_radius=9.0,
    )
    kwargs = cond.reconstruct_kwargs(trained.identity())
    assert kwargs["max_neighbours"] == 8
    assert kwargs["neighbour_radius"] == 15.0
    assert kwargs["max_atom_pairs"] == 12
    assert kwargs["atom_pair_radius"] == 9.0
    rebuilt = cond.AtomConditioner(C_S, C_Z, variant="full", pair=True, gate=GATE, **kwargs)
    assert rebuilt.identity()["neighbourhood"] == trained.identity()["neighbourhood"]
    # And a rebuild that ignored them would silently be a different function.
    assert cond.AtomConditioner(C_S, C_Z, gate=GATE).max_neighbours != 8


def test_every_reconstructed_setting_can_report_an_override():
    """The promise that overrides are logged has to be able to fire.

    It could not: the loader looked up `conditioning.max_neighbours`, and the
    constant is `MAX_NEIGHBOURS`. Every miss returned None and was read as
    "matches the default", so an arm reconstructed onto non-default settings
    said nothing at all.
    """
    trained = cond.AtomConditioner(
        C_S, C_Z, variant="full", pair=True, gate=GATE, max_neighbours=8,
        atom_pair_radius=9.0, d_hidden=128,
    )
    overridden = {name: (was, now) for name, was, now in
                  cond.overridden_settings(trained.identity())}
    assert overridden["max_neighbours"] == (8, cond.MAX_NEIGHBOURS)
    assert overridden["atom_pair_radius"] == (9.0, cond.ATOM_PAIR_RADIUS)
    assert overridden["d_hidden"] == (128, cond.D_HIDDEN)
    # An arm on the defaults reports nothing.
    assert not cond.overridden_settings(e2().identity())
    # And every reconstructed name must have a default to compare against, or
    # the report silently skips it. E1 contributes sequence_width.
    assert not cond.overridden_settings(e1(c_h_V=128, variant="full").identity())


def test_a_changed_hard_coded_feature_constant_is_refused(monkeypatch):
    """The reviewer's reproduction: changing the RBF bandwidth used to pass.

    RBF width is not a scale the first linear layer can absorb -- it changes the
    nonlinear basis, so the same weights mean something else. It is not a
    constructor argument, so a checkpoint recording a different one was produced
    by code this build does not implement.
    """
    trained = e2().identity()
    monkeypatch.setattr(cond, "RBF_SIGMA", 1.6)
    rebuilt = e2().identity()
    assert cond.check_compatible(trained, rebuilt), "the arm check cannot see this"
    with pytest.raises(ValueError, match="rbf_sigma"):
        cond.check_feature_schema(trained, rebuilt)


def test_a_changed_coordinate_scale_is_refused(monkeypatch):
    trained = e2().identity()
    monkeypatch.setattr(cond, "COORDINATE_SCALE", 1.0)
    with pytest.raises(ValueError, match="coordinate_scale"):
        cond.check_feature_schema(trained, e2().identity())


def test_a_changed_width_is_refused(monkeypatch):
    trained = e2().identity()
    monkeypatch.setattr(cond, "D_ATOM_SLOT", 8)
    with pytest.raises(ValueError, match="widths"):
        cond.check_feature_schema(trained, e2().identity())


def test_runtime_properties_are_not_part_of_the_schema(c_h_V):
    """A trained checkpoint necessarily disagrees with a fresh module on these."""
    fresh = e2()
    trained = e2()
    _wake(trained)
    assert fresh.identity()["zero_initialized"] != trained.identity()["zero_initialized"]
    assert cond.check_feature_schema(trained.identity(), fresh.identity())


def test_arms_trained_under_different_settings_are_not_comparable():
    """Each loads correctly; the SET is still not an experiment."""
    sixteen = cond.AtomConditioner(C_S, C_Z, gate=GATE, max_neighbours=16).identity()
    thirtytwo = cond.AtomConditioner(C_S, C_Z, gate=GATE, max_neighbours=32).identity()
    assert not cond.comparability({"a": sixteen, "b": dict(sixteen)})
    differences = dict(cond.comparability({"a": sixteen, "b": thirtytwo}))
    assert "max_neighbours" in differences


def test_comparability_is_within_an_architecture_not_across(c_h_V):
    """Arms of different architectures are SUPPOSED to differ.

    Refusing on that blocked the one run worth doing: a late arm beside an early
    one on a single panel, which is the only way to get a paired interval
    between the two injection sites. An older late checkpoint also records no
    `version` at all, so the cross-architecture comparison refused on a field
    that simply did not exist yet.
    """
    late = cond.build_conditioner(
        "late_full", c_h_V=c_h_V, c_token=C_TOKEN, c_s=C_S, c_z=C_Z, gate=GATE
    ).identity()
    late.pop("version", None)  # as an older checkpoint records it
    early = e1(c_h_V, "full").identity()
    assert not cond.comparability({"late_full": late, "early_s_full": early})
    # Within an architecture the check still bites.
    drifted = cond.AtomConditioner(C_S, C_Z, gate=GATE, max_neighbours=32).identity()
    differences = dict(cond.comparability({"a": e2().identity(), "b": drifted}))
    assert "max_neighbours" in differences
    # And a drifted pair does not become comparable by adding a third arch.
    differences = dict(
        cond.comparability({"a": e2().identity(), "b": drifted, "c": late})
    )
    assert "max_neighbours" in differences


def test_comparability_ignores_settings_only_one_arm_has(c_h_V):
    """A pair-less ablation records no pair settings; that is design, not drift."""
    both = e2(pair=True).identity()
    single = e2(pair=False).identity()
    assert not cond.comparability({"sz": both, "s": single})


# --- the frame convention ---------------------------------------------------


def test_the_frame_matches_the_projects_canonical_one():
    """One convention, pinned. Two orthonormal frames both look fine alone."""
    canonical = pytest.importorskip("pxf.eval.canonical")
    try:
        upstream = canonical.load().frames
    except Exception as error:  # noqa: BLE001 - the checkout may not be present
        pytest.skip(f"canonical metrics unavailable: {str(error)[:80]}")
    torch.manual_seed(0)
    n, ca, c = (torch.randn(5, 3) for _ in range(3))
    mine = frames.build_frame(n, ca, c)
    theirs = upstream.build_frame(n, ca, c)
    assert torch.allclose(mine[0], theirs[0], atol=1e-6)
    assert torch.allclose(mine[1], theirs[1], atol=1e-6)


def test_the_frame_is_right_handed_and_orthonormal():
    torch.manual_seed(0)
    n, ca, c = (torch.randn(5, 3) for _ in range(3))
    rotation, _t = frames.build_frame(n, ca, c)
    eye = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(5), atol=1e-5)


def test_a_degenerate_frame_is_reported_invalid():
    ca = torch.zeros(1, 3)
    assert not bool(frames.frame_is_valid(ca, ca, ca)[0])
