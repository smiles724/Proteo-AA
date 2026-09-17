"""Predicted side chains must actually be visible to the re-encode.

The regression these guard is silent by construction. ``build_atom_mask``
multiplies by ``1 - missing_atom_mask``, and the converter marks every
side-chain slot missing for a backbone-only proposal, so feeding that mask back
into the second encode leaves ``h_packed == h_base``: the shapes are right, no
error is raised, and the SC -> BB adapter trains against a feature that contains
no information about the packing it is supposed to report on.

So these tests go through the **real** path -- converter, packer, re-encode --
rather than calling ``encode`` with default masks, because the defaults are
exactly what hid the bug. ``encode(model, coords, aatype, sidechains=...)`` with
no ``missing_atom_mask`` works fine; it is the converter's mask that breaks it,
and only the full path shows that.
"""

import pytest
import torch

from pxf import atom37, provenance
from pxf.couple import fampnn_iface as iface
from pxf.couple import visibility as vis
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import CoupledDenoiser, Topology

C_TOKEN = 384


@pytest.fixture(scope="module")
def rc():
    from fampnn.data import residue_constants as rc

    return rc


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
def parts():
    """One real structure, reduced to the backbone PXDesign would emit."""
    from fampnn.data import residue_constants as rc
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    dataset = StructureCropDataset(
        [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")],
        crop_size=24,
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
    return dict(
        item=item,
        length=length,
        aatype=aatype,
        flat=item["x"][0][:, slots, :].reshape(-1, 3),
        topology=topology,
    )


def stub_backbone(length, channels):
    generator = torch.Generator().manual_seed(0)
    projection = torch.randn(9, channels, generator=generator) * (channels**-0.5)

    def backbone(x_noisy, sigma, *, feedback=None):
        flat = x_noisy.reshape(1, length, 4, 3)
        features = flat.reshape(1, length, 12)[..., :9] @ projection
        if feedback is not None:
            return x_noisy + feedback.mean(), features
        return x_noisy, features

    return backbone


def controller_for(fampnn, parts, **kwargs):
    adapters = CouplingAdapters(C_TOKEN, iface.node_feature_dim(fampnn))
    return CoupledDenoiser(
        stub_backbone(parts["length"], C_TOKEN),
        fampnn,
        adapters,
        phase="sc_to_bb",
        pack_steps=3,
        **kwargs,
    )


def run_cycle(fampnn, parts, **kwargs):
    controller = controller_for(fampnn, parts, **kwargs)
    torch.manual_seed(0)
    return controller.forward(
        parts["topology"],
        parts["flat"],
        torch.tensor([1.0]),
        parts["aatype"],
        run_feedback=True,
    )


# --- the mask itself --------------------------------------------------------


def test_the_converter_marks_every_sidechain_slot_missing(fampnn, parts, rc):
    """The premise. If this ever stops holding, the rest is moot."""
    controller = controller_for(fampnn, parts)
    proposal = controller.propose(
        parts["topology"], parts["flat"], torch.tensor([1.0]), parts["aatype"]
    )
    missing = proposal.inputs.missing_atom_mask
    exists = vis.atom_exists(proposal.inputs.aatype, rc=rc)
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    expected = exists[..., sidechain].sum()
    assert float(missing[..., sidechain].sum()) == pytest.approx(float(expected))


def test_availability_reveals_generated_sidechains(fampnn, parts, rc):
    cycle = run_cycle(fampnn, parts)
    visibility = cycle.visibility
    exists = visibility.exists
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    generated = float(visibility.available[..., sidechain].sum())
    assert generated > 0, "no generated side-chain atom was made available"
    # Every side-chain slot the fixed sequence has, and nothing else.
    assert generated == pytest.approx(float(exists[..., sidechain].sum()))


def test_availability_preserves_actual_backbone_availability(fampnn, parts, rc):
    """A dropped backbone atom must not be conjured into existence."""
    controller = controller_for(fampnn, parts)
    proposal = controller.propose(
        parts["topology"], parts["flat"], torch.tensor([1.0]), parts["aatype"]
    )
    inputs = proposal.inputs
    supplied = inputs.atom_mask.clone()
    supplied[0, 3, atom37.ATOM37.index("O")] = 0.0  # this proposal has no O here
    packed = torch.zeros(1, parts["length"], 33, 3)
    visibility = vis.predicted_availability(
        inputs.aatype, inputs.seq_mask, supplied, inputs.coords_af2, sidechains=packed
    )
    assert float(visibility.available[0, 3, atom37.ATOM37.index("O")]) == 0.0
    assert float(visibility.available[0, 3, atom37.ATOM37.index("CA")]) == 1.0


def test_nonexistent_and_padded_atoms_are_never_available(fampnn, parts, rc):
    length = 4
    aatype = torch.tensor([[7, 0, 18, 0]])  # GLY, ALA, TRP, ALA
    seq_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # last residue is padding
    supplied = torch.ones(1, length, 37)
    coords = torch.randn(1, length, 37, 3)
    packed = torch.randn(1, length, 33, 3)
    visibility = vis.predicted_availability(
        aatype, seq_mask, supplied, coords, sidechains=packed
    )
    assert float(visibility.available[0, 3].sum()) == 0.0, "padding became available"
    exists = vis.atom_exists(aatype, rc=rc)
    assert torch.all(visibility.available <= exists + 1e-6)
    # GLY has no side chain past CB, TRP has the most.
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    assert float(visibility.available[0, 0, sidechain].sum()) < float(
        visibility.available[0, 2, sidechain].sum()
    )


def test_an_invalid_frame_hides_its_sidechain(rc):
    """A side chain hanging off an unusable N/CA/C is not evidence."""
    aatype = torch.tensor([[18, 18]])
    seq_mask = torch.ones(1, 2)
    supplied = torch.ones(1, 2, 37)
    coords = torch.randn(1, 2, 37, 3)
    coords[0, 1, 1] = float("nan")  # CA of residue 1
    packed = torch.randn(1, 2, 33, 3)
    visibility = vis.predicted_availability(
        aatype, seq_mask, supplied, coords, sidechains=packed
    )
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    assert bool(visibility.frame_valid[0, 0])
    assert not bool(visibility.frame_valid[0, 1])
    assert float(visibility.available[0, 1, sidechain].sum()) == 0.0
    assert float(visibility.available[0, 0, sidechain].sum()) > 0.0


def test_a_nonfinite_generated_atom_is_not_available(rc):
    aatype = torch.tensor([[18]])
    packed = torch.zeros(1, 1, 33, 3)
    packed[0, 0, 5] = float("inf")
    visibility = vis.predicted_availability(
        aatype,
        torch.ones(1, 1),
        torch.ones(1, 1, 37),
        torch.zeros(1, 1, 37, 3),
        sidechains=packed,
    )
    assert float(visibility.available[0, 0, atom37.SIDECHAIN_SLOTS[5]]) == 0.0


# --- the effect on the re-encode -------------------------------------------


def test_reencoding_through_the_real_path_sees_the_packing(fampnn, parts):
    """The load-bearing test: h_packed must differ from h_base.

    Run through converter -> pack -> re-encode exactly as the cycle does. With
    the input's missing mask forwarded, these two tensors are equal.
    """
    cycle = run_cycle(fampnn, parts)
    assert cycle.h_base is not None and cycle.h_packed is not None
    relative = float(
        (cycle.h_packed - cycle.h_base).norm() / cycle.h_base.norm().clamp_min(1e-8)
    )
    assert relative > 1e-3, (
        "h_packed is indistinguishable from the side-chain-masked encoding, so "
        f"the feedback path carries nothing (relative change {relative:.2e})"
    )


def test_the_old_mask_is_what_made_them_equal(fampnn, parts):
    """Pin the mechanism, so the fix cannot be reverted without a failure."""
    controller = controller_for(fampnn, parts)
    proposal = controller.propose(
        parts["topology"], parts["flat"], torch.tensor([1.0]), parts["aatype"]
    )
    inputs = proposal.inputs
    torch.manual_seed(0)
    sidechains, _aux = iface.pack_from_features(
        fampnn,
        proposal.features,
        inputs.aatype,
        seq_mask=inputs.seq_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        num_steps=3,
    )
    # The buggy call: the input's observation mask, with side chains "visible".
    _, h_wrong, _ = iface.encode(
        fampnn,
        inputs.coords_af2,
        inputs.aatype,
        sidechains=sidechains,
        seq_mask=inputs.seq_mask,
        missing_atom_mask=inputs.missing_atom_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
    )
    assert torch.allclose(h_wrong, proposal.h_base, atol=1e-5), (
        "the premise no longer holds: forwarding the converter's missing mask "
        "should reproduce the backbone-only encoding exactly"
    )
    packed = controller.encode_predicted_packing(inputs, sidechains)
    assert not torch.allclose(packed.h_packed, proposal.h_base, atol=1e-5)


def test_availability_and_sidechain_visible_cannot_both_be_given(fampnn, parts):
    controller = controller_for(fampnn, parts)
    proposal = controller.propose(
        parts["topology"], parts["flat"], torch.tensor([1.0]), parts["aatype"]
    )
    inputs = proposal.inputs
    with pytest.raises(ValueError, match="two ways to say the same thing"):
        iface.encode(
            fampnn,
            inputs.coords_af2,
            inputs.aatype,
            sidechains=torch.zeros(1, parts["length"], 33, 3),
            atom_availability=torch.ones(1, parts["length"], 37),
            sidechain_visible=torch.ones(1, parts["length"]),
        )


def test_native_observation_masks_stay_out_of_the_encoder_input(fampnn, parts):
    """Availability must not depend on which native atoms were resolved."""
    controller = controller_for(fampnn, parts)
    proposal = controller.propose(
        parts["topology"], parts["flat"], torch.tensor([1.0]), parts["aatype"]
    )
    inputs = proposal.inputs
    packed = torch.zeros(1, parts["length"], 33, 3)
    baseline = vis.predicted_availability(
        inputs.aatype,
        inputs.seq_mask,
        inputs.atom_mask,
        inputs.coords_af2,
        sidechains=packed,
    )
    # A native structure missing half its side chains changes the supervision
    # mask and must change nothing here.
    native_missing = torch.ones_like(inputs.missing_atom_mask)
    observation = vis.native_observation_mask(
        inputs.aatype, inputs.seq_mask, native_missing
    )
    assert float(observation.sum()) == 0.0
    again = vis.predicted_availability(
        inputs.aatype,
        inputs.seq_mask,
        inputs.atom_mask,
        inputs.coords_af2,
        sidechains=packed,
    )
    assert torch.equal(baseline.available, again.available)
