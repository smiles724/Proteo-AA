"""The residual must land on the generated chain and nowhere else.

The assumption these guard against is that the design region is the final
contiguous slice of the token axis. It is not: the binder is whichever chain is
*smaller*, which for much of the prepared panel is chain A -- the front of the
axis. A tail-slice scatter would write the residual onto the fixed target for
those targets, silently.
"""

import pytest
import torch

from pxf.couple.mapping import (
    TokenMapping,
    build_mapping,
    scatter_design_residual,
    target_atom_mask,
)


def features(chains):
    """A feature dict from ``[(asym_id, n_residues)]``, tokens in that order."""
    asym, residue = [], []
    for chain_id, count in chains:
        asym += [chain_id] * count
        residue += list(range(count))
    return dict(
        asym_id=torch.tensor(asym),
        residue_index=torch.tensor(residue),
    )


def design_for(chains, generated_chain):
    mask = []
    for chain_id, count in chains:
        mask += [chain_id == generated_chain] * count
    return torch.tensor(mask, dtype=torch.bool)


# --- building the mapping ---------------------------------------------------


def test_the_generated_chain_can_be_the_front_of_the_axis():
    """The case a tail-slice scatter gets wrong."""
    chains = [(0, 4), (1, 6)]  # chain 0 is generated and comes FIRST
    mapping = build_mapping(features(chains), design_for(chains, 0))
    assert mapping.gen_to_px.tolist() == [0, 1, 2, 3]
    assert mapping.contiguous_tail is False
    assert mapping.identity()["first_token"] == 0


def test_the_generated_chain_can_be_the_tail():
    chains = [(0, 6), (1, 4)]
    mapping = build_mapping(features(chains), design_for(chains, 1))
    assert mapping.gen_to_px.tolist() == [6, 7, 8, 9]
    assert mapping.contiguous_tail is True


def test_reordered_chains_still_map_correctly():
    """The acceptance test: interleave the chains and the mapping follows.

    Built by identifier lookup, so a token order that does not group chains
    contiguously is handled rather than corrupted.
    """
    # Chains interleaved: generated residues sit at 0, 2, 4 and target at 1, 3.
    feature = dict(
        asym_id=torch.tensor([0, 1, 0, 1, 0]),
        residue_index=torch.tensor([0, 0, 1, 1, 2]),
    )
    design = torch.tensor([True, False, True, False, True])
    mapping = build_mapping(feature, design)
    assert mapping.gen_to_px.tolist() == [0, 2, 4]
    assert mapping.contiguous_tail is False
    # And with FaMPNN receiving them in a different order.
    shuffled = build_mapping(feature, design, generated_keys=[(0, 2), (0, 0), (0, 1)])
    assert shuffled.gen_to_px.tolist() == [4, 0, 2]


def test_a_mapping_onto_a_target_token_is_refused():
    chains = [(0, 3), (1, 3)]
    with pytest.raises(ValueError, match="not design tokens"):
        build_mapping(
            features(chains), design_for(chains, 0), generated_keys=[(0, 0), (1, 0)]
        )


def test_a_non_injective_mapping_is_refused():
    chains = [(0, 3), (1, 3)]
    with pytest.raises(ValueError, match="not injective"):
        build_mapping(
            features(chains),
            design_for(chains, 0),
            generated_keys=[(0, 0), (0, 0), (0, 1)],
        )


def test_a_partial_mapping_is_refused():
    """Fewer mapped residues than design tokens means something was dropped."""
    chains = [(0, 4), (1, 4)]
    with pytest.raises(ValueError, match="disagree"):
        build_mapping(
            features(chains), design_for(chains, 0), generated_keys=[(0, 0), (0, 1)]
        )


def test_ambiguous_identifiers_are_refused():
    feature = dict(asym_id=torch.tensor([0, 0]), residue_index=torch.tensor([5, 5]))
    with pytest.raises(ValueError, match="does not identify a token uniquely"):
        build_mapping(feature, torch.tensor([True, True]))


# --- the scatter ------------------------------------------------------------


def test_the_scatter_places_rows_and_zeroes_the_target():
    chains = [(0, 3), (1, 2)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    delta_gen = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) + 1.0
    out = scatter_design_residual(delta_gen, mapping)
    assert out.shape == (1, 5, 4)
    assert torch.equal(out[0, :3], delta_gen)
    assert float(out[0, 3:].abs().sum()) == 0.0


def test_a_bias_laden_residual_is_still_zero_on_the_target():
    """The reason the mask goes after the projections.

    A_SB's output projection has a bias, so even an all-zero readout emits a
    non-zero residual on every token. Masking the input would leave that bias
    to be written onto fixed coordinates.
    """
    chains = [(0, 2), (1, 3)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    # Every row non-zero, as a biased projection produces.
    delta_gen = torch.full((2, 6), 0.37)
    out = scatter_design_residual(delta_gen, mapping)
    target = out[0, ~mapping.design_mask]
    assert int(torch.count_nonzero(target)) == 0
    assert float(out[0, mapping.design_mask].abs().min()) == pytest.approx(0.37)


def test_a_batched_residual_is_accepted_and_a_batch_is_not():
    chains = [(0, 2), (1, 2)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    assert scatter_design_residual(torch.ones(1, 2, 3), mapping).shape == (1, 4, 3)
    with pytest.raises(ValueError, match="expected one example"):
        scatter_design_residual(torch.ones(2, 2, 3), mapping)


def test_a_width_mismatch_against_the_token_features_is_refused():
    chains = [(0, 2), (1, 2)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    with pytest.raises(ValueError, match="does not match the token features"):
        scatter_design_residual(torch.ones(2, 3), mapping, reference=torch.zeros(1, 4, 8))


def test_a_row_count_mismatch_is_refused():
    chains = [(0, 2), (1, 2)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    with pytest.raises(ValueError, match="rows for"):
        scatter_design_residual(torch.ones(3, 4), mapping)


# --- the atom-level target mask --------------------------------------------


def test_the_target_atom_mask_follows_atom_to_token():
    # Tokens: 0,1 generated; 2 target. Atoms: two per token, interleaved.
    feature = dict(atom_to_token_idx=torch.tensor([0, 2, 1, 2, 0, 1]))
    design = torch.tensor([True, True, False])
    mask = target_atom_mask(feature, design)
    assert mask.tolist() == [False, True, False, True, False, False]
    assert int(mask.sum()) == 2


def test_the_mapping_reports_what_it_assumed():
    chains = [(0, 4), (1, 6)]
    mapping = build_mapping(features(chains), design_for(chains, 0))
    record = mapping.identity()
    assert record["generated_residues"] == 4
    assert record["design_tokens"] == 4
    assert record["n_tokens"] == 10
    assert record["contiguous_tail"] is False


def test_a_hand_built_mapping_validates_on_construction():
    with pytest.raises(ValueError, match="not injective"):
        TokenMapping(
            gen_to_px=torch.tensor([0, 0]),
            design_mask=torch.tensor([True, True, False]),
            n_tokens=3,
            keys=[(0, 0), (0, 0)],
        )
