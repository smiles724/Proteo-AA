"""Cross-residue attention at ATOM granularity, which is what the appendix says.

The appendix specifies "cross-residue geometric attention between nearby ATOMS".
The original block pooled a residue's fourteen slots into one vector and attended
residue-to-residue, and its own docstring conceded the gap ("at residue-pooled
granularity"). That pooling is mean over slots, so it is direction-blind: a side
chain could learn that some residue is nearby, never which atom is in its way or
from which side -- and that is precisely the quantity side-chain packing turns on.

These tests pin the mechanism rather than any loss value:

  * the neighbourhood is chosen by distance, and invalid residues never enter it;
  * masked slots contribute nothing, so padding cannot leak into an attention row;
  * geometry actually matters -- moving a neighbour changes the output;
  * a context residue (one owning no S_phi atom) still participates, via a single
    virtual atom at its CA, so the atom-level rewrite did not cost the receptor
    awareness the pooled version had;
  * `a_bs_concat` still does something. Under residue granularity it fused into
    the pooled vector that got broadcast back; under atom granularity `pooled`
    only seeds context slots, so without an explicit atom-level path the channel
    would silently no-op on any structure with no context tokens.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)

from pxdesign_train.sidechain.module import (  # noqa: E402
    SideChainModule,
    _CrossAtomBlock,
)

B, L, A, C, H = 1, 5, 4, 16, 4


def _block(n_neighbors=3):
    torch.manual_seed(0)
    blk = _CrossAtomBlock(C, n_heads=H, n_neighbors=n_neighbors)
    torch.nn.init.normal_(blk.o.weight, std=0.5)
    return blk


def _inputs(seed=0):
    torch.manual_seed(seed)
    feats = torch.randn(B, L, A, C)
    coords = torch.randn(B, L, A, 3) * 5
    mask = torch.ones(B, L, A, dtype=torch.bool)
    ca = coords.mean(dim=2)
    res_mask = torch.ones(B, L, dtype=torch.bool)
    return feats, coords, mask, ca, res_mask


# ---------------------------------------------------------------- neighbourhood

def test_neighbours_are_the_nearest_residues():
    ca = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [9, 0, 0], [2, 0, 0]]])
    res_mask = torch.ones(1, 4, dtype=torch.bool)
    idx = _CrossAtomBlock.residue_neighbours(ca, res_mask, n_neighbors=2)
    # residue 0 (x=0): itself, then x=1
    assert idx[0, 0].tolist() == [0, 1]
    # residue 2 (x=9): itself, then x=2 (the next closest)
    assert idx[0, 2].tolist() == [2, 3]


def test_invalid_residues_are_not_selected_while_valid_ones_remain():
    ca = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    res_mask = torch.tensor([[True, False, True, True]])
    idx = _CrossAtomBlock.residue_neighbours(ca, res_mask, n_neighbors=2)
    assert 1 not in idx[0, 0].tolist(), "a masked residue was chosen over a valid one"


def test_neighbour_count_clamps_to_the_available_residues():
    ca = torch.randn(1, 3, 3)
    idx = _CrossAtomBlock.residue_neighbours(ca, torch.ones(1, 3, dtype=torch.bool), 64)
    assert idx.shape == (1, 3, 3)


# ---------------------------------------------------------------------- masking

def test_masked_key_slots_do_not_influence_the_output():
    """Changing a masked slot's feature must not move any output."""
    blk = _block()
    feats, coords, mask, ca, res_mask = _inputs()
    mask[0, 2, 1] = False

    out_a = blk(feats, coords, mask, ca, res_mask)
    perturbed = feats.clone()
    perturbed[0, 2, 1] = 1e3
    out_b = blk(perturbed, coords, mask, ca, res_mask)

    # every row except the masked slot's own (which is zeroed) must be unchanged
    keep = mask.clone()
    assert torch.allclose(out_a[keep], out_b[keep], atol=1e-5)


def test_a_fully_masked_row_stays_finite():
    """A residue with no valid slot must not produce NaN through an empty softmax."""
    blk = _block()
    feats, coords, mask, ca, res_mask = _inputs()
    mask[0, 3] = False
    out = blk(feats, coords, mask, ca, res_mask)
    assert torch.isfinite(out).all()


def test_padding_slots_are_zeroed_not_left_with_the_bias():
    blk = _block()
    feats, coords, mask, ca, res_mask = _inputs()
    mask[0, 1, 2] = False
    out = blk(feats, coords, mask, ca, res_mask)
    # residual base is kept; only the attention update is zeroed on padding
    assert torch.allclose(out[0, 1, 2], feats[0, 1, 2], atol=1e-6)


# --------------------------------------------------------------------- geometry

def test_output_depends_on_where_the_neighbour_atoms_are():
    """The whole point: geometry enters at atom resolution, not residue resolution."""
    blk = _block()
    feats, coords, mask, ca, res_mask = _inputs()
    out_a = blk(feats, coords, mask, ca, res_mask)

    moved = coords.clone()
    moved[0, 1, 0] += 4.0          # move ONE atom, keep its residue's CA where it was
    out_b = blk(feats, moved, mask, ca, res_mask)

    assert not torch.allclose(out_a, out_b, atol=1e-6), (
        "moving an individual neighbour atom changed nothing -- the block is still "
        "effectively residue-resolution"
    )


def test_distance_bias_is_a_penalty_by_construction():
    """softplus keeps the sign fixed: nearer can never be learned into farther."""
    blk = _block()
    with torch.no_grad():
        blk.dist_scale.fill_(-50.0)          # even a large negative raw value
    assert torch.all(torch.nn.functional.softplus(blk.dist_scale) >= 0)


# ------------------------------------------------------------ module-level wiring

def _sc_inputs(n_res=4, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(1, 10, (1, n_res, 10))
    mask = torch.ones(1, n_res, 10, dtype=torch.bool)
    return (
        torch.randn(1, n_res, 8),                 # h_res
        torch.randn(1, n_res, 20),                # logits
        ids, mask,
        torch.randn(1, n_res, 10, 3),             # noisy
        torch.randn(1, n_res, 3) * 4,             # ca
    )


def test_module_defaults_to_atom_granularity():
    m = SideChainModule(c_res=8, c_atom=16, n_type=20)
    assert m.cross_granularity == "atom"
    assert isinstance(m.cross_res_blocks[0], _CrossAtomBlock)


def test_granularity_is_validated():
    with pytest.raises(ValueError, match="cross_granularity"):
        SideChainModule(c_res=8, c_atom=16, n_type=20, cross_granularity="token")


def test_atom_and_residue_granularity_are_different_experiments():
    torch.manual_seed(0)
    atom = SideChainModule(c_res=8, c_atom=16, n_type=20, cross_granularity="atom").eval()
    torch.manual_seed(0)
    res = SideChainModule(c_res=8, c_atom=16, n_type=20, cross_granularity="residue").eval()
    h, l, ids, m_, noisy, ca = _sc_inputs()
    with torch.no_grad():
        ya, _ = atom(h, l, ids, m_, noisy, torch.ones(1), ca_coords=ca)
        yr, _ = res(h, l, ids, m_, noisy, torch.ones(1), ca_coords=ca)
    assert not torch.allclose(ya, yr, atol=1e-6)


def test_context_residue_still_participates_under_atom_granularity():
    """A receptor residue owns no S_phi slot; it must still be visible.

    Under residue granularity it was seeded into `pooled` and attended as a key.
    The atom-level rewrite has to keep that, or the side chain loses sight of the
    thing it is packing against.
    """
    m = SideChainModule(c_res=8, c_atom=16, n_type=20).eval()
    h, l, ids, mask, noisy, ca = _sc_inputs()
    mask[0, 3] = False                                  # residue 3 owns no atoms
    ctx = torch.zeros(1, 4, dtype=torch.bool)
    ctx[0, 3] = True

    with torch.no_grad():
        y_near, _ = m(h, l, ids, mask, noisy, torch.ones(1), ca_coords=ca, ctx_mask=ctx)
        far = ca.clone()
        far[0, 3] += 500.0                              # push the context residue away
        y_far, _ = m(h, l, ids, mask, noisy, torch.ones(1), ca_coords=far, ctx_mask=ctx)

    assert not torch.allclose(y_near, y_far, atol=1e-6), (
        "moving the context residue changed nothing -- it is not being attended to"
    )


def test_a_bs_concat_still_has_an_effect_under_atom_granularity():
    """Regression: `pooled` no longer carries the cross-residue mixing, so the
    B->S residue channel needs its own atom-level path or it silently no-ops."""
    torch.manual_seed(0)
    on = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=True).eval()
    torch.nn.init.normal_(on.a_bs_concat_fusion.mlp[-1].weight, std=0.5)
    torch.nn.init.normal_(on.a_bs_concat_fusion.mlp[-1].bias, std=0.5)
    h, l, ids, mask, noisy, ca = _sc_inputs()          # no context tokens at all
    with torch.no_grad():
        y_on, _ = on(h, l, ids, mask, noisy, torch.ones(1), ca_coords=ca)
        on.a_bs_concat = False
        y_off, _ = on(h, l, ids, mask, noisy, torch.ones(1), ca_coords=ca)
    assert not torch.allclose(y_on, y_off, atol=1e-6)


def test_gradients_reach_the_neighbour_atoms():
    m = SideChainModule(c_res=8, c_atom=16, n_type=20)
    h, l, ids, mask, noisy, ca = _sc_inputs()
    noisy = noisy.clone().requires_grad_(True)
    y, _ = m(h, l, ids, mask, noisy, torch.ones(1), ca_coords=ca)
    y.square().mean().backward()
    assert noisy.grad is not None and torch.isfinite(noisy.grad).all()
    assert float(noisy.grad.abs().max()) > 0.0
