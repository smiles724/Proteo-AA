"""The data contract: pose recovery, residue correspondence, and the three masks.

The masks are the part most likely to be silently conflated, so most of this
file is about keeping them apart. The rest is about refusing an example rather
than fitting away a mismatch between the two parses of one file.
"""

import os
from pathlib import Path

import pytest
import torch

from pxf.joint import data as D

DONOR = os.environ.get(
    "PXDESIGN_DONOR",
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt",
)
CIF = os.environ.get("PXF_TEST_CIF", "/hai/scratch/yfsun/casp14/cif/T1031.cif")

needs_featurizer = pytest.mark.skipif(
    not Path(CIF).is_file(),
    reason="set PXF_TEST_CIF to run against a real featurized structure",
)


# ---- the rigid map ----------------------------------------------------------


def _random_rotation(seed=0):
    generator = torch.Generator().manual_seed(seed)
    q, _r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    return q * torch.sign(torch.det(q))


def test_rigid_transform_recovers_a_known_pose():
    points = torch.randn(40, 3, dtype=torch.float64)
    rotation = _random_rotation()
    translation = torch.tensor([3.0, -7.0, 11.0], dtype=torch.float64)
    moved = points @ rotation + translation
    recovered, offset, rmsd = D.rigid_transform(points, moved)
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(recovered, rotation, atol=1e-9)
    assert torch.allclose(offset, translation, atol=1e-9)


def test_rigid_transform_never_produces_a_reflection():
    """A mirrored point set is not a pose change, and must not be fitted as one."""
    points = torch.randn(40, 3, dtype=torch.float64)
    mirrored = points * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    rotation, _t, rmsd = D.rigid_transform(points, mirrored)
    assert float(torch.det(rotation)) == pytest.approx(1.0, abs=1e-9)
    assert rmsd > 0.1, "a reflection should not superpose to zero"


def test_rigid_transform_reports_a_geometry_change_as_residual():
    points = torch.randn(60, 3, dtype=torch.float64) * 10
    bent = points.clone()
    bent[:10] += 2.0  # move part of the structure, not all of it
    _r, _t, rmsd = D.rigid_transform(points, bent)
    assert rmsd > 0.1


# ---- synthetic two-parse fixtures ------------------------------------------


def _helix_backbone(length):
    """``[L, 4, 3]`` N/CA/C/O on a helix.

    Deliberately not a straight line or a lattice: matched points that span a
    degenerate volume leave the superposition's rotation unconstrained about the
    missing axis, so a fixture built that way tests the wrong thing -- the
    residual comes out at 1e-6 and the recovered transform is still wrong for
    every atom off the axis.
    """
    turn = torch.arange(length, dtype=torch.float32) * 1.75
    rise = torch.arange(length, dtype=torch.float32) * 1.5
    centre = torch.stack([2.3 * torch.cos(turn), 2.3 * torch.sin(turn), rise], dim=-1)
    offsets = torch.tensor(
        [[-1.2, 0.3, -0.4], [0.0, 0.0, 0.0], [1.1, 0.5, 0.3], [1.3, 1.6, 0.2]]
    )
    return centre[:, None, :] + offsets[None, :, :]


def _structure(length=6, *, coords=None, aatype=None, sample_id="synthetic"):
    """A minimal FeaturizedStructure with a backbone-only flat axis."""
    from fampnn.data import residue_constants as rc
    from pxf import atom37
    from pxf.backbone.driver import FeaturizedStructure
    from pxf.couple.controller import Topology

    names = list(atom37.BACKBONE_ATOMS)
    aatype = torch.zeros(length, dtype=torch.long) if aatype is None else aatype
    atom_names = [n for _ in range(length) for n in names]
    tokens = torch.tensor([r for r in range(length) for _ in names])
    if coords is None:
        coords = _helix_backbone(length).reshape(-1, 3)
    topology = Topology(
        atom_names=atom_names,
        atom_to_token_idx=tokens,
        num_tokens=length,
        res_names=[rc.restype_1to3[atom37.AA_ORDER[int(aatype[t])]] for t in tokens],
    )
    return FeaturizedStructure(
        sample_id=sample_id,
        feature_dict={},
        label_dict={"coordinate": coords, "coordinate_mask": torch.ones(len(atom_names))},
        topology=topology,
        aatype=aatype,
        design_mask=torch.zeros(length, dtype=torch.bool),
        backbone_target=coords,
        num_tokens=length,
    )


def _native(length=6, *, aatype=None, jitter=0.0, seed=0):
    """A native atom37 parse whose backbone matches ``_structure``'s flat axis."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc
    from pxf import atom37

    aatype = torch.zeros(length, dtype=torch.long) if aatype is None else aatype
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)
    x = torch.zeros(length, atom37.NUM_ATOM37, 3)
    flat = _helix_backbone(length)
    for position, name in enumerate(atom37.BACKBONE_ATOMS):
        x[:, atom37.ATOM37.index(name)] = flat[:, position]
    # Side chains sit somewhere plausible relative to CA.
    generator = torch.Generator().manual_seed(seed)
    side = torch.tensor(rc.non_bb_idxs)
    x[:, side] = x[:, 1:2] + torch.rand(
        length, len(side), 3, generator=generator
    )
    x = x * exists.unsqueeze(-1)
    if jitter:
        x = x + jitter
    return {
        "x": x,
        "aatype": aatype,
        "seq_mask": torch.ones(length),
        "missing_atom_mask": torch.zeros(length, atom37.NUM_ATOM37),
        "residue_index": torch.arange(1, length + 1),
        "chain_index": torch.zeros(length, dtype=torch.long),
    }


def test_a_pose_difference_is_recovered_and_applied():
    """The native parse in another pose is brought into the featurizer's frame."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    structure, native = _structure(), _native()
    rotation = _random_rotation(1).float()
    shift = torch.tensor([10.0, -4.0, 2.0])
    posed = dict(native)
    posed["x"] = native["x"] @ rotation + shift
    _r, _t, rmsd = D.recover_preprocessing_transform(structure, posed)
    assert rmsd == pytest.approx(0.0, abs=1e-3)

    present = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, native["aatype"])
    back = D.align_native_to_features(
        posed["x"],
        *D.recover_preprocessing_transform(structure, posed)[:2],
        mask=present,
    )
    # 1e-2 A, not 1e-6: these are float32 coordinates tens of Angstroms from the
    # origin passed through two rotations. The transform itself is solved in
    # float64 -- that is what keeps the *residual* meaningful as an accept/reject
    # signal -- but the round trip is limited by the coordinates' own precision.
    assert torch.allclose(back, native["x"], atol=1e-2)


def test_an_absent_slot_is_a_sentinel_not_a_point():
    """A rigid transform moves "absent" onto the translation unless it is masked.

    An unfilled atom37 slot holds an exact zero meaning "no atom here", not a
    coordinate at the origin. The native parse's zeros are in the file's frame,
    so any transform with a nonzero translation lands them on a plausible-looking
    position a few Angstroms from the structure -- which the physical mask would
    then have to be trusted to exclude. Zeroing them on the way through means
    the coordinates and the masks cannot disagree.
    """
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    native = _native()
    present = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, native["aatype"])
    ghost = present == 0
    assert bool(ghost.any())
    assert float(native["x"][ghost].abs().max()) == 0.0, "absent slots start at zero"

    rotation = torch.eye(3)
    translation = torch.tensor([10.0, -4.0, 2.0])

    unmasked = D.align_native_to_features(native["x"], rotation, translation)
    assert float(unmasked[ghost].abs().max()) > 1.0, "the sentinel moved, as it must"

    masked = D.align_native_to_features(
        native["x"], rotation, translation, mask=present
    )
    assert float(masked[ghost].abs().max()) == 0.0


def test_a_geometry_change_is_rejected_not_fitted():
    structure = _structure()
    native = _native()
    native["x"][0, 1] = native["x"][0, 1] + 5.0  # move one CA
    with pytest.raises(D.PreprocessingMismatch, match="changed the geometry"):
        D.recover_preprocessing_transform(structure, native)


def test_a_degenerate_match_set_is_refused_despite_a_tiny_residual():
    """Collinear backbones superpose to ~0 and still place side chains wrong.

    This is the failure the residual alone cannot catch: the rotation about the
    degenerate axis is free, so Kabsch reports a perfect fit and every atom off
    that axis -- which is every side chain -- is moved somewhere else.
    """
    line = torch.arange(6 * 4 * 3, dtype=torch.float32).reshape(-1, 3)
    structure = _structure(coords=line)
    native = _native()
    from pxf import atom37

    collinear = native["x"].clone()
    for position, name in enumerate(atom37.BACKBONE_ATOMS):
        collinear[:, atom37.ATOM37.index(name)] = line.reshape(6, 4, 3)[:, position]
    native["x"] = collinear

    source, target = D.matched_backbone_atoms(structure, native)
    _r, _t, rmsd = D.rigid_transform(source, target)
    assert rmsd < 1e-6, "the residual looks perfect, which is the trap"
    assert D.extent_ratio(source) < D.MIN_EXTENT_RATIO
    with pytest.raises(D.PreprocessingMismatch, match="degenerate volume"):
        D.recover_preprocessing_transform(structure, native)


def test_extent_ratio_separates_a_real_structure_from_a_line():
    line = torch.stack([torch.arange(30.0)] * 3, dim=-1)
    assert D.extent_ratio(line) < 1e-6
    assert D.extent_ratio(_helix_backbone(12).reshape(-1, 3)) > 0.05


def test_too_few_matched_atoms_is_refused():
    structure, native = _structure(length=2), _native(length=2)
    with pytest.raises(D.PreprocessingMismatch, match="too thin"):
        D.recover_preprocessing_transform(structure, native)


def test_a_sequence_disagreement_is_refused():
    aatype = torch.zeros(6, dtype=torch.long)
    structure = _structure(aatype=aatype)
    other = aatype.clone()
    other[2] = 5
    with pytest.raises(D.PreprocessingMismatch, match="disagree at 1"):
        D.assert_correspondence(structure, _native(aatype=other))


def test_a_length_disagreement_is_refused():
    with pytest.raises(D.PreprocessingMismatch, match="residues and the native parse"):
        D.assert_correspondence(_structure(length=6), _native(length=5))


def test_duplicate_residue_keys_are_refused():
    """Two copies of a chain share sequence; only the keys can tell them apart."""
    structure = _structure(length=6)
    native = _native(length=6)
    native["residue_index"] = torch.tensor([1, 2, 3, 1, 2, 3])
    with pytest.raises(D.PreprocessingMismatch, match="share a \\(chain, number\\) key"):
        D.assert_correspondence(structure, native)
    # Distinguishing the copies by chain makes the same residues acceptable.
    native["chain_index"] = torch.tensor([0, 0, 0, 1, 1, 1])
    assert len(D.assert_correspondence(structure, native)) == 6


# ---- the three masks --------------------------------------------------------


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def _batched(native):
    return {
        k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in native.items()
    }


def test_local_includes_ghosts_and_physical_does_not(fampnn):
    """The two side-chain masks are different sets, and the difference is ghosts."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    # Alanine: 5 real atoms, so 1 real side-chain slot (CB) out of 33.
    native = _batched(_native(length=6))
    masks = D.supervision_masks(fampnn, native)
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, native["aatype"])[
        ..., rc.non_bb_idxs
    ]
    ghost = (1 - exists) * native["seq_mask"].unsqueeze(-1)

    assert float((masks["physical_mask"] * ghost).sum()) == 0.0
    assert float((masks["local_mask"] * ghost).sum()) > 0.0
    # Physical is a strict subset of local.
    assert torch.all(masks["physical_mask"] <= masks["local_mask"] + 1e-6)
    assert float(masks["physical_mask"].sum()) < float(masks["local_mask"].sum())


def test_glycine_supervises_ghosts_but_places_no_atoms(fampnn):
    """Glycine has no side chain, so it is all ghost: local yes, physical no."""
    from fampnn.data import residue_constants as rc

    glycine = torch.full((4,), rc.restype_order_with_x["G"], dtype=torch.long)
    native = _batched(_native(length=4, aatype=glycine))
    masks = D.supervision_masks(fampnn, native)
    assert float(masks["physical_mask"].sum()) == 0.0
    assert float(masks["local_mask"].sum()) > 0.0
    counts = D.mask_counts(masks)
    assert counts["physical_atoms"] == 0.0
    assert counts["ghost_fraction"] == pytest.approx(1.0)


def test_a_quality_veto_drops_physical_atoms_but_keeps_ghost_supervision(fampnn):
    """The veto marks real atoms missing; the residue's ghost slots survive.

    That asymmetry is deliberate -- the local term reproduces the source
    objective, which vetoes through ``missing_atom_mask`` -- but it means the
    two counts diverge for a reason other than glycine, so it is pinned.
    """
    from pxf.train.protenix import veto_unsupervised_sidechains

    native = _batched(_native(length=6, aatype=torch.full((6,), 9, dtype=torch.long)))
    before = D.mask_counts(D.supervision_masks(fampnn, native))

    supervise = torch.ones(6, dtype=torch.bool)
    supervise[:3] = False
    vetoed = dict(native)
    vetoed["missing_atom_mask"] = veto_unsupervised_sidechains(
        native["missing_atom_mask"][0], supervise, aatype=native["aatype"][0]
    ).unsqueeze(0)
    after = D.mask_counts(D.supervision_masks(fampnn, vetoed))

    assert after["physical_atoms"] < before["physical_atoms"]
    assert after["ghost_atoms"] == pytest.approx(before["ghost_atoms"])


def test_the_encoder_mask_never_reveals_a_side_chain():
    """Even handed a backbone mask that claims side-chain atoms, none get through.

    The encoder is looking at a *predicted* backbone; a side-chain slot it
    appears to supply is an artefact, and letting one through would leak a
    target into the input.
    """
    from fampnn.data import residue_constants as rc
    from pxf import atom37

    aatype = torch.zeros(1, 5, dtype=torch.long)
    seq_mask = torch.ones(1, 5)
    claimed = torch.ones(1, 5, atom37.NUM_ATOM37)  # every slot, including side chains
    available = D.encoder_availability(aatype, seq_mask, claimed)
    assert float(available[..., rc.non_bb_idxs].sum()) == 0.0
    assert float(available[..., rc.bb_idxs].sum()) == 5 * len(rc.bb_idxs)


def test_the_encoder_mask_respects_padding_and_the_supplied_backbone():
    from pxf import atom37

    aatype = torch.zeros(1, 4, dtype=torch.long)
    seq_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    supplied = torch.zeros(1, 4, atom37.NUM_ATOM37)
    supplied[..., list(atom37.BACKBONE_SLOTS)] = 1.0
    supplied[0, 1, atom37.ATOM37.index("O")] = 0.0  # one backbone atom not produced
    available = D.encoder_availability(aatype, seq_mask, supplied)
    assert float(available[0, 2:].sum()) == 0.0
    assert float(available[0, 0].sum()) == 4
    assert float(available[0, 1].sum()) == 3


# ---- end to end on a real structure ----------------------------------------


@pytest.fixture(scope="module")
def real_pair():
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf.backbone.driver import featurize_structures, to_featurized

    sample_id, source = featurize_structures([CIF], crop_size=256)[0]
    structure = to_featurized(sample_id, source[0])
    native = process_single_pdb(load_feats_from_pdb(CIF))
    return structure, native


@needs_featurizer
def test_the_two_real_parses_already_share_a_frame(real_pair):
    """Measured, not assumed: on this path the recovered transform is the identity.

    The featurizer's label coordinates come back in the file's own frame for the
    monomer configuration, so the recovery is a *check* rather than a fix. It
    earns its place by being run per example -- a configuration that centers or
    reposes would otherwise put the side-chain targets several Angstroms from
    the backbone they belong to, and nothing downstream would say so.
    """
    structure, native = real_pair
    rotation, translation, rmsd = D.recover_preprocessing_transform(structure, native)
    assert rmsd == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(rotation, torch.eye(3, dtype=rotation.dtype), atol=1e-6)
    assert float(translation.abs().max()) < 1e-6


@needs_featurizer
def test_build_joint_batch_on_a_real_structure(fampnn, real_pair):
    structure, native = real_pair
    batch = D.build_joint_batch(fampnn, structure, native, split="dev")
    assert batch.length == structure.num_tokens == 95
    assert batch.backbone_target.shape == (380, 3)
    # T1031 featurizes to a backbone-only flat axis, so every atom is supervised.
    assert float(batch.backbone_mask.sum()) == 380
    assert batch.local_mask.shape == (1, 95, 33)
    assert torch.all(batch.physical_mask <= batch.local_mask + 1e-6)
    assert 0.0 < batch.counts["ghost_fraction"] < 1.0
    assert not batch.local_target.requires_grad
    identity = batch.identity()
    assert identity["sample_id"] == "T1031" and identity["alignment_rmsd"] < 1e-4


@needs_featurizer
def test_the_batch_moves_to_a_device_whole(fampnn, real_pair):
    """Every tensor, including the ones inside the two dicts and the topology."""
    structure, native = real_pair
    batch = D.build_joint_batch(fampnn, structure, native).to("cpu")
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            assert value.device.type == "cpu", name
    for value in batch.native_batch.values():
        assert not torch.is_tensor(value) or value.device.type == "cpu"
