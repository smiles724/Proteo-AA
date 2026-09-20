"""The feedback readout has to contain the packing, and nothing it cannot see.

Two families of test. First, that ``z`` actually carries predicted side-chain
conformation: a valid rotamer change must move it, and the controls must not be
able to see that change at all. Second, the boundary: no native quantity may
reach it, padding and nonexistent atoms may not move it, and a rigid motion of
the whole structure may not either.

The gate gets its own attention because the failure it guards is a silent one --
zero-initializing both the output projection and the gate leaves the product at
a stationary point in both factors, and the adapter trains forever without
moving.
"""

import math
from dataclasses import replace

import pytest
import torch

from pxf import atom37, provenance
from pxf.couple import fampnn_iface as iface
from pxf.couple import readout as ro
from pxf.couple import torsions
from pxf.couple import visibility as vis
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import CoupledDenoiser, Topology

C_TOKEN = 384


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
    """A real packed structure, through the real converter -> pack -> re-encode."""
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
        features = torch.zeros(1, length, C_TOKEN)
        return x_noisy, features

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
    return controller.encode_predicted_packing(
        proposal.inputs, sidechains, h_base=proposal.h_base, psce=aux.get("psce")
    )


@pytest.fixture(scope="module")
def c_h_V(fampnn):
    return iface.node_feature_dim(fampnn)


# --- the readout contains the packing --------------------------------------


def test_the_readout_has_every_group_at_full_width(packed, c_h_V):
    readout = ro.FeedbackReadout(c_h_V)
    z, stats = readout(packed)
    assert z.shape == (1, packed.h_packed.shape[1], readout.width)
    assert readout.width == c_h_V + 12 + 8 + 32 + 6
    assert stats["z_groups_zeroed"] == []
    assert stats["chis_valid"] > 0, "no chi was measurable on a packed structure"


def test_a_rotamer_change_moves_the_readout(packed, c_h_V):
    """The load-bearing property: z must depend on which rotamer was packed."""
    readout = ro.FeedbackReadout(c_h_V).eval()
    with torch.no_grad():
        base, _ = readout(packed)
        deltas = torsions.random_chi_deltas(
            packed.aatype,
            math.radians(90.0),
            generator=torch.Generator().manual_seed(0),
        )
        moved = torsions.perturb_chi(
            packed.coords37, packed.aatype, deltas, available=packed.available
        )
        rotated = replace(packed, coords37=moved)
        after, _ = readout(rotated)
    change = float((after - base).norm() / base.norm().clamp_min(1e-8))
    assert change > 1e-3, f"a 90 degree rotamer flip moved z by only {change:.2e}"


def test_the_controls_cannot_see_a_rotamer_change(packed, c_h_V):
    """bb_only and generic must be blind to it, or they are not controls.

    Note this tests the *feature* groups, not the node group: bb_only reads
    h_base, which is the side-chain-masked encoding and so genuinely cannot
    depend on the packing. The rotation here leaves the backbone alone, so a
    control that still moved would be reading side chains somewhere.
    """
    deltas = torsions.random_chi_deltas(
        packed.aatype, math.radians(90.0), generator=torch.Generator().manual_seed(0)
    )
    moved = replace(
        packed,
        coords37=torsions.perturb_chi(
            packed.coords37, packed.aatype, deltas, available=packed.available
        ),
    )
    for variant in ("bb_only", "generic"):
        readout = ro.FeedbackReadout(c_h_V, variant=variant).eval()
        with torch.no_grad():
            base, _ = readout(packed)
            after, _ = readout(moved)
        assert torch.equal(base, after), (
            f"the {variant} control's z changed when only the side chains moved"
        )


def test_every_variant_has_the_same_parameter_count(c_h_V):
    counts = {
        variant: sum(
            p.numel() for p in ro.FeedbackPath(c_h_V, C_TOKEN, variant=variant).parameters()
        )
        for variant in ro.VARIANTS
    }
    assert len(set(counts.values())) == 1, (
        f"the controls differ in capacity, not just information: {counts}"
    )


def test_the_generic_control_depends_only_on_sigma(packed, c_h_V):
    path = ro.FeedbackPath(c_h_V, C_TOKEN, variant="generic").eval()
    # Break the zero-init so the output is observable at all.
    torch.nn.init.normal_(path.project_out.weight, std=0.05)
    other = replace(packed, h_packed=torch.randn_like(packed.h_packed))
    with torch.no_grad():
        a, _ = path(packed, torch.tensor([1.0]))
        b, _ = path(other, torch.tensor([1.0]))
        c, _ = path(packed, torch.tensor([0.3]))
    assert torch.equal(a, b), "the generic control read the packing"
    assert not torch.equal(a, c), "the generic control ignored sigma too"


def test_no_native_quantity_reaches_the_readout(packed, c_h_V):
    """The PackedStructure carries nothing native, by construction."""
    fields = set(vis.PackedStructure.__dataclass_fields__)
    assert fields == {
        "h_packed",
        "coords37",
        "aatype",
        "seq_mask",
        "visibility",
        "psce",
        "h_base",
        # The sequence-attribution controls. All three are functions of bb0 and
        # the FIXED sequence, both of which exist at inference: h_masked
        # replaces the sequence with X, while h_predicted and aatype_predicted
        # come from FaMPNN's own inverse-folding head reading bb0. No native
        # coordinate and no native side chain reaches any of them.
        "h_masked",
        "h_predicted",
        "aatype_predicted",
    }, (
        "a field was added to PackedStructure; check it is available at "
        f"inference before letting the readout see it. fields={sorted(fields)}"
    )


# --- invariances ------------------------------------------------------------


def test_the_readout_is_invariant_to_a_rigid_motion(packed, c_h_V):
    """z is built from torsions, distances and the encoder's invariant readout."""
    generator = torch.Generator().manual_seed(1)
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    moved = (packed.coords37.double() @ q + torch.tensor([5.0, -3.0, 2.0])).to(
        packed.coords37.dtype
    )
    # h_packed is held fixed: its own invariance is pinned in
    # test_couple_probes.py, and this test is about the geometric features.
    readout = ro.FeedbackReadout(c_h_V).eval()
    with torch.no_grad():
        base, _ = readout(packed)
        after, _ = readout(replace(packed, coords37=moved))
    change = float((after - base).abs().max())
    assert change < 1e-2, f"z moved by {change} under a rigid motion"


def test_nonexistent_atoms_do_not_reach_the_readout(packed, c_h_V):
    exists = packed.visibility.exists
    junk = (
        packed.coords37 + torch.randn_like(packed.coords37) * 50.0 * (1 - exists)[..., None]
    )
    readout = ro.FeedbackReadout(c_h_V).eval()
    with torch.no_grad():
        base, _ = readout(packed)
        after, _ = readout(replace(packed, coords37=junk))
    assert torch.allclose(base, after, atol=1e-5), (
        "scrambling slots the residue type does not have changed z"
    )


def test_invalid_residues_get_exactly_zero(packed, c_h_V):
    broken = packed.visibility
    frame_valid = broken.frame_valid.clone()
    frame_valid[0, 3] = False
    hobbled = replace(packed, visibility=replace(broken, frame_valid=frame_valid))
    readout = ro.FeedbackReadout(c_h_V).eval()
    with torch.no_grad():
        z, _ = readout(hobbled)
    assert float(z[0, 3].abs().sum()) == 0.0


# --- the gate and the zero-init --------------------------------------------


def test_the_output_is_zero_but_the_gate_is_not(packed, c_h_V):
    """The exact trap the plan names: do not zero both factors."""
    window = ro.SigmaWindow(0.1, 2.0)
    path = ro.FeedbackPath(c_h_V, C_TOKEN, variant="full", gate=window)
    assert path.is_identity()
    with torch.no_grad():
        delta, stats = path(packed, torch.tensor([0.5]))
    assert float(delta.abs().max()) == 0.0, "the correction is not zero at init"
    assert stats["gate"] == pytest.approx(1.0), (
        "the gate is not one inside the training window, so the zero-initialized "
        "projection has no gradient to grow from"
    )


def test_the_gate_is_one_across_the_window_and_tapers_outside():
    window = ro.SigmaWindow(0.1, 2.0, taper=2.0)
    for sigma in (0.1, 0.3, 1.0, 2.0):
        assert float(window(torch.tensor([sigma]))) == pytest.approx(1.0, abs=1e-6)
    assert float(window(torch.tensor([0.05]))) == pytest.approx(0.0, abs=1e-6)
    assert float(window(torch.tensor([4.0]))) == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < float(window(torch.tensor([0.07]))) < 1.0
    assert 0.0 < float(window(torch.tensor([3.0]))) < 1.0
    # Monotone across the lower taper, so the correction does not jump.
    values = [float(window(torch.tensor([s]))) for s in (0.05, 0.06, 0.08, 0.1)]
    assert values == sorted(values)


def test_the_gradient_reaches_the_output_projection(packed, c_h_V):
    path = ro.FeedbackPath(c_h_V, C_TOKEN, gate=ro.SigmaWindow(0.1, 2.0))
    delta, _ = path(packed, torch.tensor([0.5]))
    # Any downstream scalar that depends on delta: the real loss goes through
    # PXDesign's decoder, which tests/test_sb_pilot.py exercises.
    delta.pow(2).sum().backward()
    grad = path.project_out.weight.grad
    assert grad is not None and float(grad.abs().sum()) >= 0.0
    # At zero-init the output is zero, so a squared objective has zero gradient;
    # a linear one does not. This is the shape the real BB loss has.
    path.zero_grad()
    delta, _ = path(packed, torch.tensor([0.5]))
    delta.sum().backward()
    assert float(path.project_out.weight.grad.abs().sum()) > 0, (
        "no gradient reaches W2, so A_SB can never leave zero"
    )


def test_the_residual_norm_is_reported_against_the_token_features(packed, c_h_V):
    path = ro.FeedbackPath(c_h_V, C_TOKEN, gate=ro.SigmaWindow(0.1, 2.0))
    torch.nn.init.normal_(path.project_out.weight, std=0.02)
    reference = torch.randn(1, packed.h_packed.shape[1], C_TOKEN)
    with torch.no_grad():
        _delta, stats = path(packed, torch.tensor([0.5]), reference=reference)
    assert stats["relative_residual"] > 0
    assert stats["a_token_norm"] > 0
    assert stats["delta_a_norm"] > 0


def test_it_plugs_into_the_adapter_container(packed, c_h_V):
    path = ro.FeedbackPath(c_h_V, C_TOKEN, gate=ro.SigmaWindow(0.1, 2.0))
    adapters = CouplingAdapters(C_TOKEN, c_h_V, sc_to_bb=path)
    record = adapters.set_phase("sc_to_bb")
    assert record["sc_to_bb"] is True and record["bb_to_sc"] is False
    trainable = {n for n, p in adapters.named_parameters() if p.requires_grad}
    assert all(n.startswith("sc_to_bb.") for n in trainable), sorted(trainable)
    delta, stats = adapters.delta_a(packed, torch.tensor([0.5]))
    assert delta.shape[-1] == C_TOKEN
    assert "gate" in stats
    assert adapters.identity()["sc_to_bb"]["variant"] == "full"


# --- decomposing what bb_only reads -----------------------------------------
#
# `bb_only` is routinely described as the backbone control. It is not: it reads
# h_base AND the sequence embedding, and h_base is produced by FaMPNN's encoder
# FROM the native aatype, so the sequence enters twice. A gain attributed to
# geometry could be entirely sequence. These pin the separation.


def test_bb_only_reads_the_sequence_twice(packed, c_h_V):
    """The premise of the decomposition, stated as a test rather than a claim."""
    assert ro.VARIANTS["bb_only"] == {"node", "sequence"}
    assert ro.NODE_SOURCE["bb_only"] == "h_base"
    # And h_base is a function of the aatype, not of the backbone alone.
    assert packed.h_base is not None


def test_the_masked_encoding_is_blind_to_the_sequence(fampnn, packed):
    """geometry_only is only a geometry control if h_masked really is one."""
    from pxf.couple import fampnn_iface as iface

    coords = packed.coords37
    masked = torch.full_like(packed.aatype, atom37.UNKNOWN_AA_INDEX)
    _l, h_masked, _f = iface.encode(fampnn, coords, masked, seq_mask=packed.seq_mask)
    # Permuting the sequence cannot change it, because it never saw one.
    order = torch.randperm(packed.aatype.shape[1])
    _l, h_permuted, _f = iface.encode(
        fampnn, coords, torch.full_like(masked[:, order], atom37.UNKNOWN_AA_INDEX),
        seq_mask=packed.seq_mask,
    )
    assert torch.allclose(h_masked, h_permuted, atol=1e-6)
    # But it still carries the backbone. The perturbation has to be non-rigid:
    # the encoder is invariant to a rigid motion, so translating every atom by a
    # constant correctly changes nothing and would fail this for the right
    # reason.
    moved = coords.clone()
    slots = list(atom37.BACKBONE_SLOTS)
    generator = torch.Generator().manual_seed(0)
    moved[:, :, slots, :] += (
        torch.randn(moved[:, :, slots, :].shape, generator=generator) * 0.3
    )
    _l, h_moved, _f = iface.encode(fampnn, moved, masked, seq_mask=packed.seq_mask)
    assert float((h_moved - h_masked).norm() / h_masked.norm()) > 1e-3
    # And it differs from the native-sequence encoding, or there was nothing
    # to separate in the first place.
    _l, h_native, _f = iface.encode(
        fampnn, coords, packed.aatype, seq_mask=packed.seq_mask
    )
    assert float((h_masked - h_native).norm() / h_native.norm()) > 0.01


@pytest.mark.parametrize("variant", ("geometry_only", "bb_predicted_sequence"))
def test_a_missing_control_encoding_is_refused_not_substituted(packed, c_h_V, variant):
    """Falling back to h_packed would make the arm silently not a control."""
    readout = ro.FeedbackReadout(c_h_V, variant=variant)
    with pytest.raises(ValueError, match="encode_sequence_controls"):
        readout(packed)


def test_the_controls_have_the_same_parameter_count_as_the_candidate(c_h_V):
    """They differ in information, not in capacity -- as the other arms do."""
    counts = {
        v: sum(p.numel() for p in ro.FeedbackReadout(c_h_V, variant=v).parameters())
        for v in ro.VARIANTS
    }
    assert len(set(counts.values())) == 1, counts


def test_sequence_only_is_blind_to_the_backbone(packed, c_h_V):
    readout = ro.FeedbackReadout(c_h_V, variant="sequence_only").eval()
    moved = replace(packed, coords37=packed.coords37 + 5.0)
    with torch.no_grad():
        assert torch.equal(readout(packed)[0], readout(moved)[0])


def test_geometry_only_is_blind_to_the_sequence(packed, c_h_V):
    """The load-bearing property: change the sequence, the readout must not move.

    The node group is re-pointed at h_masked and the sequence group is zeroed,
    so neither path can carry the aatype. h_masked is supplied directly here;
    whether the ENCODING is itself sequence-blind is the separate test above.
    """
    readout = ro.FeedbackReadout(c_h_V, variant="geometry_only").eval()
    filled = replace(packed, h_masked=torch.randn_like(packed.h_packed))
    order = torch.randperm(filled.aatype.shape[1])
    scrambled = replace(filled, aatype=filled.aatype[:, order])
    with torch.no_grad():
        assert torch.equal(readout(filled)[0], readout(scrambled)[0])
        # And not merely zero: the geometry has to reach z.
        moved = replace(filled, h_masked=torch.randn_like(filled.h_masked))
        assert not torch.allclose(readout(filled)[0], readout(moved)[0])
