"""DIRECT a-level side-chain -> backbone feedback (FangWu's slide) + the
batch>1 fix in the template-init type source.

    a'_bb = a_bb + MLP(LayerNorm(concat(a_bb, W a_sc)))

The existing feedback is INDIRECT: h_res' -> HResInjector -> s_trunk, and the
DiffusionModule then recomputes a_token from scratch, so the fused representation
never *is* the next round's token. `sidechain.a_direct` fuses at the a_token level
itself via a forward hook on `DiffusionModule.layernorm_a` (a hook returning a
non-None value REPLACES the module's output) and keeps the previous backbone token
as the residual base.

Invariants pinned here:
  * a_direct=False  -> hook only caches; a_token comes out unchanged.
  * a_direct=True   -> injection fires ONLY in the refinement pass (flag armed
    around that diffusion call) and only when a_sc from round 1 exists.
  * zero-init residual -> exact no-op at step 0 (cannot perturb a pretrained
    backbone), while the branch's own output layer still gets gradient.
  * idempotent under a repeated hook call (activation-checkpoint recompute).
  * gradients flow from the fused token back into S_phi's atom features.
  * template-init residue types are tiled PER ITEM (batch>1 regression).
"""
import inspect
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.model import ProtenixDesignTrain, _tile_per_sample
from pxdesign_train.sidechain.coevolution import ATokenFusion, pool_side_chain_atoms
from pxdesign_train.sidechain.feedback import HResFeedback
from pxdesign_train.sidechain.init import template_init_local
from pxdesign_train.sidechain.instantiate import MAX_SC, STD_AA_3, instantiate_from_type_indices
from pxdesign_train.sidechain.module import SideChainModule

C_TOKEN, C_ATOM, L, A = 24, 16, 5, MAX_SC


def _feats(B=2, N=3, seed=0):
    torch.manual_seed(seed)
    a_bb = torch.randn(B, N, L, C_TOKEN)
    a_sc = torch.randn(B, L, C_ATOM)
    return a_bb, a_sc


# ---------------------------------------------------------------- ATokenFusion


def test_zero_init_fusion_is_exact_noop():
    """Enabling a_direct cannot perturb a pretrained backbone at step 0."""
    fus = ATokenFusion(c_token=C_TOKEN, c_atom=C_ATOM)
    a_bb, a_sc = _feats(B=1)
    out = fus(a_bb[:, 0], a_sc)
    assert torch.equal(out, a_bb[:, 0])          # bit-exact identity, not "close"


def test_zero_init_still_leaves_the_zero_point():
    """Zero-init kills the gradient INTO a_sc at step 0 (by construction), but the
    residual branch's own output layer still gets gradient, so it starts learning."""
    fus = ATokenFusion(c_token=C_TOKEN, c_atom=C_ATOM)
    a_bb, a_sc = _feats(B=1)
    a_sc = a_sc.requires_grad_(True)
    fus(a_bb[:, 0], a_sc).pow(2).sum().backward()
    assert float(a_sc.grad.abs().max()) == 0.0              # documented step-0 behaviour
    assert float(fus.mlp[-1].weight.grad.abs().max()) > 0.0  # ... but it can move


def test_fusion_keeps_the_previous_token_as_residual_base():
    fus = ATokenFusion(c_token=C_TOKEN, c_atom=C_ATOM, zero_init=False)
    torch.nn.init.normal_(fus.mlp[-1].weight, std=0.5)
    a_bb, a_sc = _feats(B=2)
    out = fus(a_bb[:, 0], a_sc)
    assert out.shape == a_bb[:, 0].shape
    assert not torch.allclose(out, a_bb[:, 0])   # it does something ...
    delta = out - a_bb[:, 0]                     # ... and what it does is a residual
    expected = fus.mlp(fus.ln(torch.cat([a_bb[:, 0], fus.sc_proj(a_sc)], dim=-1)))
    assert torch.allclose(delta, expected, atol=1e-6)


def test_fusion_forward_is_pure_and_idempotent():
    """Re-running the fusion on the same inputs (checkpoint recompute) must not
    compound the residual — no += on a cached tensor."""
    fus = ATokenFusion(c_token=C_TOKEN, c_atom=C_ATOM, zero_init=False)
    a_bb, a_sc = _feats(B=1)
    base = a_bb[:, 0].clone()
    o1 = fus(a_bb[:, 0], a_sc)
    o2 = fus(a_bb[:, 0], a_sc)
    assert torch.equal(o1, o2)
    assert torch.equal(a_bb[:, 0], base)         # input untouched in place


def test_pool_matches_hresfeedback_pooling():
    """a_sc is the SAME per-token side-chain summary HResFeedback pools, not a
    second, inconsistent one."""
    torch.manual_seed(1)
    atom_feats = torch.randn(2, L, A, C_ATOM)
    mask = torch.zeros(2, L, A, dtype=torch.bool)
    mask[..., :3] = True
    mask[0, 1] = False                                   # a residue with no side chain
    fb = HResFeedback(c_atom=C_ATOM, c_res=C_TOKEN)
    m = mask[..., None].to(atom_feats.dtype)
    ref = (atom_feats * m).sum(dim=2) / (m.sum(dim=2) + fb.eps)   # feedback.py's pooling
    got = pool_side_chain_atoms(atom_feats, mask)
    assert torch.allclose(got, ref, atol=1e-7)
    assert float(got[0, 1].abs().max()) == 0.0           # empty residue -> 0, no NaN
    assert torch.isfinite(got).all()


# ------------------------------------------------- hook scoping (the real code)


class _StubModel:
    """Duck-typed stand-in carrying the REAL hook methods off ProtenixDesignTrain
    (building the full model needs a Protenix checkpoint-scale config)."""

    _a_token_forward_hook = ProtenixDesignTrain._a_token_forward_hook
    _align_a_sc = ProtenixDesignTrain._align_a_sc

    def __init__(self, a_direct=True, zero_init=False, aa_source="diffusion_internal"):
        self.enable_residue_type_head = True
        self.aa_input_source = aa_source
        self.sc_a_direct = a_direct
        self._a_token_cache = None
        self._a_direct_active = False
        self._a_sc_cache = None
        self.a_token_fusion = ATokenFusion(
            c_token=C_TOKEN, c_atom=C_ATOM, zero_init=zero_init
        )
        if not zero_init:
            torch.nn.init.normal_(self.a_token_fusion.mlp[-1].weight, std=0.5)


def _hook(stub, out):
    return stub._a_token_forward_hook(None, None, out)


def test_a_direct_false_reproduces_todays_behaviour():
    """No injection: the hook returns None (a_token unchanged) and still caches."""
    stub = _StubModel(a_direct=False)
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc          # even with features around ...
    stub._a_direct_active = False    # ... the flag is never armed when a_direct=False
    assert _hook(stub, a_bb) is None
    assert stub._a_token_cache is a_bb


def test_no_injection_in_the_first_pass():
    """First pass: S_phi has not run, so a_sc does not exist -> no injection even
    if the flag were (wrongly) armed."""
    stub = _StubModel()
    a_bb, _ = _feats()
    assert _hook(stub, a_bb) is None              # flag down, no cache
    stub._a_direct_active = True
    stub._a_sc_cache = None
    assert _hook(stub, a_bb) is None              # armed but nothing to inject
    assert stub._a_token_cache is a_bb            # cached the raw token


def test_injection_fires_in_the_refinement_pass():
    stub = _StubModel()
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True
    fused = _hook(stub, a_bb)
    assert fused is not None and fused.shape == a_bb.shape
    assert not torch.allclose(fused, a_bb)
    # a_sc is broadcast over the N_sample axis, so every sample row sees the same
    # per-token side-chain summary of its own item.
    a_sc_e = a_sc.unsqueeze(1).expand(a_bb.shape[0], a_bb.shape[1], L, C_ATOM)
    assert torch.allclose(fused, stub.a_token_fusion(a_bb, a_sc_e), atol=1e-6)
    # what the atom decoder consumes is what the AA head reads
    assert stub._a_token_cache is fused


def test_injection_is_idempotent_under_repeated_hook_call():
    """Activation-checkpoint recomputation fires the hook a second time on the same
    layernorm output; the residual must not compound."""
    stub = _StubModel()
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True
    f1 = _hook(stub, a_bb)
    f2 = _hook(stub, a_bb)                       # recompute
    assert torch.equal(f1, f2)
    assert torch.equal(stub._a_sc_cache, a_sc)   # cache never mutated
    # and the whole pass is repeatable: a third call still lands on the same value
    assert torch.equal(_hook(stub, a_bb), f1)


def test_hook_return_actually_replaces_the_module_output():
    """The mechanism itself: a forward hook returning non-None replaces the output
    of DiffusionModule.layernorm_a, so a'_bb IS the token the decoder consumes."""
    ln = torch.nn.LayerNorm(C_TOKEN)
    stub = _StubModel()
    a_bb, a_sc = _feats(B=1)
    stub._a_sc_cache = a_sc
    x = torch.randn(1, 3, L, C_TOKEN)

    ln.register_forward_hook(stub._a_token_forward_hook)
    stub._a_direct_active = False
    assert torch.equal(ln(x), torch.nn.functional.layer_norm(
        x, (C_TOKEN,), ln.weight, ln.bias))       # first pass: untouched
    stub._a_direct_active = True
    fused = ln(x)                                 # refinement pass: replaced
    assert not torch.allclose(fused, torch.nn.functional.layer_norm(
        x, (C_TOKEN,), ln.weight, ln.bias))


def test_align_a_sc_handles_batched_and_unbatched_a_token():
    stub = _StubModel()
    a4 = torch.randn(2, 3, L, C_TOKEN)                       # [B, N_sample, L, c]
    got = stub._align_a_sc(torch.randn(2, L, C_ATOM), a4)
    assert got.shape == (2, 3, L, C_ATOM)
    a3 = torch.randn(3, L, C_TOKEN)                          # [N_sample, L, c]
    got = stub._align_a_sc(torch.randn(L, C_ATOM), a3)
    assert got.shape == (3, L, C_ATOM)
    # unreconcilable shapes degrade to "no injection", never a wrong fusion
    assert stub._align_a_sc(torch.randn(2, L + 1, C_ATOM), a4) is None


def test_gradients_flow_from_fused_token_into_sphi_atom_features():
    """The point of the direct form: a loss on the refined token trains S_phi."""
    torch.manual_seed(0)
    sphi = SideChainModule(c_res=C_TOKEN, c_atom=C_ATOM, n_type=len(STD_AA_3))
    stub = _StubModel()                                   # non-zero residual branch
    B, N = 1, 2
    types = torch.randint(0, len(STD_AA_3), (L,))
    ids, mask = instantiate_from_type_indices(types)
    h_res = torch.randn(B, L, C_TOKEN)
    logits = torch.randn(B, L, len(STD_AA_3))
    noisy = torch.randn(B, L, A, 3)
    _y0, atom_feats = sphi(h_res, logits, ids[None], mask[None], noisy, torch.ones(B))
    assert atom_feats.requires_grad
    stub._a_sc_cache = stub.a_token_fusion.pool(atom_feats, mask[None])
    stub._a_direct_active = True
    fused = _hook(stub, torch.randn(B, N, L, C_TOKEN))
    fused.pow(2).sum().backward()
    g = sphi.atom_embed.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0.0
    gx = sphi.w_xyz.weight.grad
    assert gx is not None and float(gx.abs().max()) > 0.0


# ------------------------------------------------ PART 1: batch>1 template init


def test_type_and_mask_are_tiled_to_the_same_rows():
    """Regression: the type source must be tiled PER ITEM the way sc_mask is, not
    collapsed to item 0 and broadcast to every row."""
    B, N = 3, 4
    torch.manual_seed(7)
    types = torch.stack([torch.randint(0, len(STD_AA_3), (L,)) for _ in range(B)])
    masks = torch.stack([instantiate_from_type_indices(t)[1] for t in types])
    sample_shape, flat_B = torch.Size([B]), B * N
    t_tiled = _tile_per_sample(types, 1, sample_shape, N, flat_B)
    m_tiled = _tile_per_sample(masks, 2, sample_shape, N, flat_B)
    assert t_tiled.shape == (flat_B, L) and m_tiled.shape == (flat_B, L, A)
    for r in range(flat_B):
        b = r // N                                  # row-major: row b*N+s -> item b
        assert torch.equal(t_tiled[r], types[b])
        assert torch.equal(m_tiled[r], masks[b])
    # the buggy behaviour (item-0 types under per-item masks) is genuinely different
    old = types[0].reshape(1, -1).expand(flat_B, -1)
    assert not torch.equal(old, t_tiled)


def test_template_init_is_per_item_for_batch_gt_1():
    """End to end through the real initializer: item 1's atoms come from item 1's
    residue templates."""
    B, N = 2, 2
    types = torch.stack([
        torch.full((L,), STD_AA_3.index("TRP")),
        torch.full((L,), STD_AA_3.index("LEU")),
    ])
    masks = torch.stack([instantiate_from_type_indices(t)[1] for t in types])
    sample_shape, flat_B = torch.Size([B]), B * N
    t_tiled = _tile_per_sample(types, 1, sample_shape, N, flat_B)
    m_tiled = _tile_per_sample(masks, 2, sample_shape, N, flat_B)
    g = torch.Generator().manual_seed(0)
    y = template_init_local(t_tiled, m_tiled, sigma_T=0.0, generator=g)
    assert y.shape == (flat_B, L, A, 3)
    trp, leu = y[0], y[B * N - 1]                    # item 0 row vs item 1 row
    # TRP has more side-chain atoms than LEU: the init geometry differs per item
    assert int(m_tiled[0].sum()) != int(m_tiled[-1].sum())
    assert not torch.allclose(trp, leu)
    # every row of an item is that item's template (sigma_T=0 -> deterministic)
    assert torch.equal(y[0], y[1])
    assert torch.equal(y[2], y[3])


def test_single_item_tiling_unchanged():
    """B=1 (today's DataLoader) must be bit-identical to the old collapse."""
    N = 3
    types = torch.randint(0, len(STD_AA_3), (1, L))
    tiled = _tile_per_sample(types, 1, torch.Size([1]), N, N)
    old = types[0].reshape(1, -1).expand(N, -1)
    assert torch.equal(tiled, old)


def test_no_item0_collapse_left_in_the_type_source():
    """Source guard so the regression cannot silently return."""
    src = inspect.getsource(ProtenixDesignTrain._train_forward)
    assert "sc_type_idx = sc_type_idx[0]" not in src
    assert "_tile_per_sigma(sc_type_idx" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
