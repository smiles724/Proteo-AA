"""The conversion layer owns every representation mismatch; each is checked."""

import pytest
import torch

from pxf import atom37
from pxf.couple.converter import PXFaRepresentationConverter


@pytest.fixture(scope="module")
def converter():
    return PXFaRepresentationConverter()


def _two_residues():
    names = ["N", "CA", "C", "O", "CB", "CG", "CD1"] + ["N", "CA", "C", "O", "OXT"]
    tokens = [0] * 7 + [1] * 5
    res_names = ["LEU"] * 7 + ["xpb"] * 5
    coords = torch.arange(len(names) * 3, dtype=torch.float32).reshape(-1, 3)
    return names, tokens, res_names, coords


def test_densification_places_every_atom(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    assert inputs.coords_af2.shape == (1, 2, 37, 3)
    assert inputs.atom_mask[0].sum(-1).tolist() == [7.0, 5.0]
    for atom, (name, token) in enumerate(zip(names, tokens)):
        slot = atom37.ATOM37.index(name)
        assert torch.equal(inputs.coords_af2[0, token, slot], coords[atom])


def test_design_tokens_become_unknown_identities(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    assert inputs.design_mask.tolist() == [False, True]
    assert inputs.sequence_known.tolist() == [True, False]
    assert inputs.aatype[0].tolist() == [
        atom37.AA_ORDER.index("L"),
        atom37.UNKNOWN_AA_INDEX,
    ]


def test_sample_axis_folds_into_the_batch(converter):
    names, tokens, res_names, coords = _two_residues()
    stacked = coords.expand(3, len(names), 3)
    inputs = converter.px_backbone_to_fampnn(stacked, names, tokens, 2, res_names=res_names)
    assert inputs.batch == 3 and inputs.length == 2
    assert inputs.aatype.shape == (3, 2) and inputs.seq_mask.shape == (3, 2)


def test_missing_atom_mask_follows_fampnns_convention(converter):
    """1 where an atom should exist for the residue type but was not supplied."""
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    # LEU has CB,CG,CD1,CD2; CD2 was not supplied, so exactly one slot is missing.
    assert float(inputs.missing_atom_mask[0, 0].sum()) == 1.0
    # The design token only owns backbone slots, all of which were supplied.
    assert float(inputs.missing_atom_mask[0, 1].sum()) == 0.0


def test_backbone_is_preserved_when_placing_side_chains(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    block = torch.full((2, 33, 3), 9.0)
    out = converter.fampnn_sidechains_to_px(block, inputs.coords_af2[0])
    assert torch.equal(
        out[:, converter.backbone_slots], inputs.coords_af2[0][:, converter.backbone_slots]
    )
    assert bool((out[:, converter.sidechain_slots] == 9.0).all())


def test_full_atom37_input_is_accepted_too(converter):
    block = torch.zeros(2, 37, 3)
    block[:, converter.sidechain_slots] = 4.0
    out = converter.fampnn_sidechains_to_px(block)
    assert bool((out[:, converter.sidechain_slots] == 4.0).all())
    with pytest.raises(ValueError, match="side-chain atoms"):
        converter.fampnn_sidechains_to_px(torch.zeros(2, 11, 3))


def test_flat_round_trip_is_exact(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    flat, valid = converter.scatter_to_px_atoms(inputs.coords_af2[0], names, tokens)
    assert torch.equal(flat[valid], coords[valid])


def test_unknown_atom_names_are_marked_invalid_not_placed(converter):
    coords = torch.zeros(1, 37, 3)
    flat, valid = converter.scatter_to_px_atoms(coords, ["CA", "H1"], [0, 0])
    assert valid.tolist() == [True, False]


def test_chain_ids_are_compacted_and_numbering_kept(converter):
    residue, chain = converter.map_chain_and_residue_indices(
        4, chain_index=[7, 7, 9, 9], residue_index=[10, 11, 3, 4]
    )
    assert chain.tolist() == [0, 0, 1, 1]
    assert residue.tolist() == [10, 11, 3, 4]


def test_defaults_are_a_single_chain_in_order(converter):
    residue, chain = converter.map_chain_and_residue_indices(3)
    assert residue.tolist() == [0, 1, 2] and chain.tolist() == [0, 0, 0]


def test_residue_mask_broadcast_is_checked(converter):
    assert converter.px_residue_mask_to_fampnn([1, 0, 1], 3, batch=2).shape == (2, 3)
    with pytest.raises(ValueError, match="entries for"):
        converter.px_residue_mask_to_fampnn([1, 0], 3)


def test_fampnn_kwargs_cover_the_module_signature(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    assert set(inputs.fampnn_kwargs()) == {
        "coords_af2",
        "atom_mask",
        "seq_mask",
        "residue_index",
        "chain_index",
    }


# ---- device consistency ----------------------------------------------------
#
# The coupling path crossed devices in three places before this was pinned, and
# each one surfaced somewhere unhelpful: the featurizer's CPU tensors blew up in
# the condition embedder's first F.linear, the topology's CPU index blew up in a
# scatter_reduce, and the per-token annotations blew up later still. They all
# have the same shape -- a tensor built from a Python list, on a path whose
# coordinates live on the accelerator.
#
# The invariant is: every tensor the converter returns is on the *coordinates'*
# device. On a CPU-only machine the assertion is trivially satisfied, so the
# CUDA-gated test below is the one that can actually fail; the ungated one exists
# to state the contract where it is read.

TENSOR_FIELDS = (
    "coords_af2",
    "atom_mask",
    "aatype",
    "seq_mask",
    "missing_atom_mask",
    "residue_index",
    "chain_index",
    "design_mask",
    "sequence_known",
)


def test_every_output_is_on_the_coordinates_device(converter):
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(coords, names, tokens, 2, res_names=res_names)
    for field in TENSOR_FIELDS:
        value = getattr(inputs, field)
        assert torch.is_tensor(value), field
        assert value.device == inputs.coords_af2.device, field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_a_cpu_topology_does_not_leak_into_cuda_outputs(converter):
    """The real case: coordinates on the GPU, topology and res_names on the CPU.

    ``pxdesign_train``'s featurizer always emits CPU tensors, so this is the
    normal state of affairs rather than an unusual one.
    """
    names, tokens, res_names, coords = _two_residues()
    inputs = converter.px_backbone_to_fampnn(
        coords.cuda(),  # coordinates on the accelerator
        names,  # atom names: a Python list
        torch.as_tensor(tokens),  # topology index: deliberately left on the CPU
        2,
        res_names=res_names,
        residue_index=torch.arange(2),
        chain_index=torch.zeros(2, dtype=torch.long),
    )
    assert inputs.coords_af2.is_cuda
    for field in TENSOR_FIELDS:
        assert getattr(inputs, field).is_cuda, field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_the_design_mask_reduction_survives_a_cpu_index():
    """``scatter_reduce_`` refuses a CPU index against a CUDA output."""
    from pxf import bridge

    _, tokens, res_names, _ = _two_residues()
    mask = bridge.design_mask_from_res_names(res_names, torch.as_tensor(tokens).cuda(), 2)
    assert mask.is_cuda and mask.tolist() == [False, True]
