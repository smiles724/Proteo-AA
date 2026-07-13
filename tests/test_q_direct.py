"""DIRECT q-level (ATOM-level) side-chain -> backbone feedback (FangWu's slide,
"Interconnection between Backbone Module and Side-chain Module").

    q'_bb = q_bb + MLP(LayerNorm(concat(q_bb, W q_sc_bb)))

`a_direct` closes the backbone<->side-chain loop at the TOKEN level (one vector per
residue). `q_direct` closes it at the ATOM level: S_phi keeps all 14 ATOM14 slots —
(N, CA, C, O) + 10 side-chain slots — and "by changing the last 10 it adjusts the
first 4"; those 4 per-atom features are fused into the Backbone Module's own per-atom
features (AtomAttentionEncoder `q_skip`) for the SAME 4 atoms, by rewriting `q_skip`
in a forward-PRE-hook on `DiffusionModule.atom_attention_decoder`. No submodule edit.

Invariants pinned here:
  * q_direct=False        -> pre-hook is inert; the call is byte-identical to today.
  * the rewrite touches ONLY the 4 backbone-atom rows of binder residues; receptor
    atoms, binder side-chain atoms and unresolved atoms are passed through unchanged
    (element-wise).
  * injection fires ONLY in the refinement pass (q_sc_bb does not exist before it).
  * idempotent under a repeated pre-hook call (activation-checkpoint recompute).
  * zero-init residual -> exact no-op at step 0.
  * gradient flows from the backbone decoder's q back into S_phi's SIDE-CHAIN atom
    features (the last 10 slots) — the atom channel is real, not decorative.
  * both decoder call forms are intercepted: kwargs (normal) and POSITIONAL (the
    fine-grained-checkpoint branch calls the decoder through checkpoint_fn).
"""
import inspect
import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.configs.configs_train import training_configs
from pxdesign_train.model import ProtenixDesignTrain
from pxdesign_train.sidechain.coevolution import QAtomFusion
from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    N_BB,
    STD_AA_3,
    instantiate_from_type_indices,
)
from pxdesign_train.sidechain.module import SideChainModule

C_Q, C_ATOM, C_RES = 12, 8, 10      # backbone q dim / S_phi atom dim / h_res dim
L, N_ATOM = 4, 30                   # residues (tokens) / total atoms in the complex

# Per-token (N, CA, C, O) atom indices, exactly the layout featurizer.sc_bb_atom_idx
# emits. Residue 1's O is unresolved (-1 in the O column while 0:3 are valid) and
# residue 3 is a non-binder token (all -1). Atoms 11..29 are "everything else":
# receptor atoms and the binder's own side-chain atoms.
BB_IDX = torch.tensor([
    [0, 1, 2, 3],
    [4, 5, 6, -1],
    [7, 8, 9, 10],
    [-1, -1, -1, -1],
], dtype=torch.long)
TOUCHED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UNTOUCHED = [i for i in range(N_ATOM) if i not in TOUCHED]


def _fusion(zero_init=False, seed=0):
    torch.manual_seed(seed)
    fus = QAtomFusion(c_q=C_Q, c_atom=C_ATOM, zero_init=zero_init)
    if not zero_init:                       # make the residual branch actually do something
        nn.init.normal_(fus.mlp[-1].weight, std=0.5)
        nn.init.normal_(fus.mlp[-1].bias, std=0.5)
    return fus


# ------------------------------------------------------------------ QAtomFusion


def test_zero_init_fusion_is_exact_noop():
    """Turning q_direct on cannot perturb a pretrained backbone at step 0."""
    fus = QAtomFusion(c_q=C_Q, c_atom=C_ATOM)          # zero_init default
    torch.manual_seed(1)
    q_bb = torch.randn(2, L, N_BB, C_Q)
    q_sc = torch.randn(2, L, N_BB, C_ATOM)
    assert torch.equal(fus(q_bb, q_sc), q_bb)          # bit-exact identity, not "close"


def test_zero_init_still_leaves_the_zero_point():
    """Zero-init kills the gradient INTO q_sc_bb at step 0 (by construction), but the
    residual branch's own output layer still gets gradient, so it starts learning."""
    fus = QAtomFusion(c_q=C_Q, c_atom=C_ATOM)
    q_bb = torch.randn(1, L, N_BB, C_Q)
    q_sc = torch.randn(1, L, N_BB, C_ATOM, requires_grad=True)
    fus(q_bb, q_sc).pow(2).sum().backward()
    assert float(q_sc.grad.abs().max()) == 0.0                # documented step-0 behaviour
    assert float(fus.mlp[-1].weight.grad.abs().max()) > 0.0   # ... but it can move


def test_fusion_keeps_the_backbone_q_as_residual_base():
    fus = _fusion()
    q_bb = torch.randn(2, L, N_BB, C_Q)
    q_sc = torch.randn(2, L, N_BB, C_ATOM)
    out = fus(q_bb, q_sc)
    assert out.shape == q_bb.shape
    assert not torch.allclose(out, q_bb)
    delta = out - q_bb
    expected = fus.mlp(fus.ln(torch.cat([q_bb, fus.sc_proj(q_sc)], dim=-1)))
    assert torch.allclose(delta, expected, atol=1e-6)


def test_fusion_forward_is_pure_and_idempotent():
    fus = _fusion()
    q_bb = torch.randn(1, L, N_BB, C_Q)
    q_sc = torch.randn(1, L, N_BB, C_ATOM)
    base = q_bb.clone()
    assert torch.equal(fus(q_bb, q_sc), fus(q_bb, q_sc))
    assert torch.equal(q_bb, base)                     # never mutated in place


# ------------------------------------------------- hook scoping (the real code)


class _StubModel:
    """Duck-typed stand-in carrying the REAL hook methods off ProtenixDesignTrain
    (building the full model needs a Protenix checkpoint-scale config)."""

    _q_skip_encoder_hook = ProtenixDesignTrain._q_skip_encoder_hook
    _q_skip_decoder_pre_hook = ProtenixDesignTrain._q_skip_decoder_pre_hook
    _fuse_q_backbone_atoms = ProtenixDesignTrain._fuse_q_backbone_atoms

    def __init__(self, q_direct=True, zero_init=False):
        self.sc_q_direct = q_direct
        self._q_direct_active = False
        self._q_sc_cache = None
        self._q_bb_idx_cache = None
        self._q_skip_cache = None
        self.q_atom_fusion = _fusion(zero_init=zero_init)


def _inputs(B=2, N=3, seed=0):
    torch.manual_seed(seed)
    q_skip = torch.randn(B, N, N_ATOM, C_Q)            # [B, N_sample, N_atom, c_q]
    q_sc = torch.randn(B, L, N_BB, C_ATOM)             # [B, L, 4, c_atom] (sigma-reduced)
    return q_skip, q_sc


def _armed(stub, q_sc, bb_idx=BB_IDX):
    stub._q_sc_cache = q_sc
    stub._q_bb_idx_cache = bb_idx
    stub._q_direct_active = True


def _prehook(stub, q_skip, positional=False):
    """Call the real pre-hook the way DiffusionModule calls the atom decoder."""
    if positional:
        # fine-grained-checkpoint branch: checkpoint_fn(decoder, a2t, a, q_skip, c, p, ...)
        args = (torch.zeros(N_ATOM, dtype=torch.long), torch.zeros(1), q_skip,
                torch.zeros(1), torch.zeros(1), False, None)
        return stub._q_skip_decoder_pre_hook(None, args, {})
    args = ()
    kwargs = {"atom_to_token_idx": torch.zeros(N_ATOM, dtype=torch.long),
              "a": torch.zeros(1), "q_skip": q_skip,
              "c_skip": torch.zeros(1), "p_skip": torch.zeros(1)}
    return stub._q_skip_decoder_pre_hook(None, args, kwargs)


def _q_out(ret, positional=False):
    args, kwargs = ret
    return args[2] if positional else kwargs["q_skip"]


def test_encoder_hook_reads_q_and_leaves_the_output_alone():
    stub = _StubModel()
    q = torch.randn(2, 3, N_ATOM, C_Q)
    out = ("a_token", q, "c_skip", "p_skip")
    assert stub._q_skip_encoder_hook(None, None, out) is None   # read-only
    assert stub._q_skip_cache is q


def test_q_direct_false_is_byte_identical_to_today():
    """The flag is never armed when q_direct=False, so the pre-hook returns None and
    the atom decoder receives exactly the q_skip the encoder produced."""
    stub = _StubModel(q_direct=False)
    q_skip, q_sc = _inputs()
    stub._q_sc_cache = q_sc                # even with features lying around ...
    stub._q_bb_idx_cache = BB_IDX
    stub._q_direct_active = False          # ... the flag is down
    assert _prehook(stub, q_skip) is None
    assert _prehook(stub, q_skip, positional=True) is None


def test_no_injection_in_the_first_pass():
    """First pass: S_phi has not run, so q_sc_bb does not exist -> no injection even if
    the flag were (wrongly) armed."""
    stub = _StubModel()
    q_skip, q_sc = _inputs()
    assert _prehook(stub, q_skip) is None            # flag down, nothing cached
    stub._q_direct_active = True
    stub._q_sc_cache = None
    stub._q_bb_idx_cache = None
    assert _prehook(stub, q_skip) is None            # armed but nothing to inject
    stub._q_sc_cache = q_sc                          # half-populated is still a no-op
    assert _prehook(stub, q_skip) is None


def test_injection_fires_in_the_refinement_pass():
    stub = _StubModel()
    q_skip, q_sc = _inputs()
    _armed(stub, q_sc)
    q_new = _q_out(_prehook(stub, q_skip))
    assert q_new.shape == q_skip.shape
    assert not torch.allclose(q_new, q_skip)


def test_only_the_four_backbone_rows_of_binder_residues_change():
    """The heart of it: every other atom row — receptor atoms, the binder's own
    side-chain atoms, the unresolved O, the non-binder token — is passed through
    byte-for-byte, and the touched rows carry exactly q'_bb."""
    stub = _StubModel()
    q_skip, q_sc = _inputs()
    _armed(stub, q_sc)
    q_new = _q_out(_prehook(stub, q_skip))

    # (a) untouched rows are bit-identical (element-wise, not allclose)
    assert torch.equal(q_new[..., UNTOUCHED, :], q_skip[..., UNTOUCHED, :])
    # (b) touched rows all moved
    assert (q_new[..., TOUCHED, :] != q_skip[..., TOUCHED, :]).all()
    # (c) and each one is exactly the fusion of ITS OWN q_bb with ITS OWN q_sc_bb
    valid = BB_IDX >= 0
    q_bb = q_skip[..., BB_IDX.clamp_min(0), :] * valid[..., None]     # [B,N,L,4,c_q]
    q_sc_e = q_sc.unsqueeze(1).expand(q_bb.shape[0], q_bb.shape[1], L, N_BB, C_ATOM)
    expect = stub.q_atom_fusion(q_bb, q_sc_e)                         # [B,N,L,4,c_q]
    for r in range(L):
        for s in range(N_BB):
            a = int(BB_IDX[r, s])
            if a < 0:
                continue
            assert torch.allclose(q_new[..., a, :], expect[..., r, s, :], atol=1e-6)
    # (d) an absent atom (-1) must NOT be clamped onto atom 0: residue 0's N still
    #     carries residue 0's fusion, not residue 1's unresolved O.
    assert torch.allclose(q_new[..., 0, :], expect[..., 0, 0, :], atol=1e-6)


def test_injection_is_idempotent_under_repeated_prehook_call():
    """Activation-checkpoint recomputation fires the pre-hook a second time on the same
    q_skip; the residual must not compound."""
    stub = _StubModel()
    q_skip, q_sc = _inputs()
    _armed(stub, q_sc)
    base_q, base_sc = q_skip.clone(), q_sc.clone()
    q1 = _q_out(_prehook(stub, q_skip))
    q2 = _q_out(_prehook(stub, q_skip))               # recompute
    q3 = _q_out(_prehook(stub, q_skip))
    assert torch.equal(q1, q2) and torch.equal(q1, q3)
    assert torch.equal(q_skip, base_q)               # incoming tensor untouched in place
    assert torch.equal(stub._q_sc_cache, base_sc)    # cache never mutated
    # feeding the FUSED q back in would compound — proof the purity above is load-bearing
    assert not torch.equal(_q_out(_prehook(stub, q1)), q1)


def test_zero_init_injection_is_an_exact_noop_at_step_0():
    stub = _StubModel(zero_init=True)
    q_skip, q_sc = _inputs()
    _armed(stub, q_sc)
    assert torch.equal(_q_out(_prehook(stub, q_skip)), q_skip)


def test_both_call_forms_are_intercepted():
    """kwargs (the normal DiffusionModule call) and POSITIONAL (the fine-grained-
    checkpoint branch, which calls the decoder through checkpoint_fn) must produce the
    same rewritten q_skip, in the right slot."""
    stub = _StubModel()
    q_skip, q_sc = _inputs()
    _armed(stub, q_sc)

    args_k, kwargs_k = _prehook(stub, q_skip)
    assert args_k == () and set(kwargs_k) >= {"q_skip", "c_skip", "p_skip"}
    q_kw = kwargs_k["q_skip"]

    args_p, kwargs_p = _prehook(stub, q_skip, positional=True)
    assert kwargs_p == {} and len(args_p) == 7
    q_pos = args_p[2]
    assert torch.equal(q_kw, q_pos)
    assert not torch.equal(q_pos, q_skip)
    # the other positional args are passed through untouched
    assert args_p[5] is False and args_p[6] is None


def test_prehook_actually_replaces_the_decoder_input():
    """The mechanism itself, on a real nn.Module: a forward-pre-hook registered with
    with_kwargs=True and returning (args, kwargs) replaces what the module receives."""

    class _Decoder(nn.Module):
        def forward(self, atom_to_token_idx, a, q_skip, c_skip, p_skip,
                    inplace_safe=False, chunk_size=None):
            return q_skip

    dec = _Decoder()
    stub = _StubModel()
    q_skip, q_sc = _inputs(B=1, N=2)
    dec.register_forward_pre_hook(stub._q_skip_decoder_pre_hook, with_kwargs=True)

    z = torch.zeros(1)
    a2t = torch.zeros(N_ATOM, dtype=torch.long)
    stub._q_direct_active = False                         # first pass: untouched
    assert torch.equal(dec(a2t, z, q_skip, z, z), q_skip)

    _armed(stub, q_sc)                                    # refinement pass: replaced
    got_kw = dec(a2t, z, q_skip=q_skip, c_skip=z, p_skip=z)
    got_pos = dec(a2t, z, q_skip, z, z, False, None)
    assert not torch.equal(got_kw, q_skip)
    assert torch.equal(got_kw, got_pos)


def test_alignment_handles_batched_and_unbatched_q_skip():
    stub = _StubModel()
    q4 = torch.randn(2, 3, N_ATOM, C_Q)                   # [B, N_sample, N_atom, c_q]
    got = stub._fuse_q_backbone_atoms(q4, torch.randn(2, L, N_BB, C_ATOM), BB_IDX)
    assert got.shape == q4.shape
    q3 = torch.randn(3, N_ATOM, C_Q)                      # [N_sample, N_atom, c_q]
    got = stub._fuse_q_backbone_atoms(q3, torch.randn(L, N_BB, C_ATOM), BB_IDX)
    assert got.shape == q3.shape
    # unreconcilable shapes degrade to "no injection", never to a wrong fusion
    assert stub._fuse_q_backbone_atoms(
        q4, torch.randn(2, L, N_BB, C_ATOM), BB_IDX[: L - 1]
    ) is None


def test_per_item_rows_are_not_broadcast_from_item_0():
    """q_sc_bb is per ITEM: item 1's backbone rows must be fused with item 1's
    side-chain features, and every sigma row of an item sees that item's features."""
    stub = _StubModel()
    q_skip = torch.randn(2, 3, N_ATOM, C_Q)
    q_sc = torch.randn(2, L, N_BB, C_ATOM)
    q_new = stub._fuse_q_backbone_atoms(q_skip, q_sc, BB_IDX)
    for b in range(2):
        one = stub._fuse_q_backbone_atoms(q_skip[b], q_sc[b], BB_IDX)
        assert torch.allclose(q_new[b], one, atol=1e-6)
    # item 0's features under item 1's q would be a different answer
    swapped = stub._fuse_q_backbone_atoms(q_skip, q_sc[[0, 0]], BB_IDX)
    assert not torch.allclose(q_new[1], swapped[1])


# ---------------------------------------------- the atom channel is real (S_phi)


def _sphi_bb_feats(sphi, noisy, bb_local):
    B = noisy.shape[0]
    torch.manual_seed(3)
    types = torch.randint(0, len(STD_AA_3), (L,))
    ids, mask = instantiate_from_type_indices(types)
    h_res = torch.randn(B, L, C_RES)
    logits = torch.randn(B, L, len(STD_AA_3))
    return sphi(
        h_res, logits, ids[None].expand(B, -1, -1), mask[None].expand(B, -1, -1),
        noisy, torch.ones(B), bb_local=bb_local,
    )


def test_sphi_emits_backbone_slot_features_and_keeps_the_sidechain_axis():
    torch.manual_seed(0)
    sphi = SideChainModule(c_res=C_RES, c_atom=C_ATOM, n_type=len(STD_AA_3))
    noisy = torch.randn(1, L, MAX_SC, 3)
    bb_local = torch.randn(1, L, N_BB, 3)
    y0, atom_feats, bb_feats = _sphi_bb_feats(sphi, noisy, bb_local)
    assert y0.shape == (1, L, MAX_SC, 3)               # coords: side chains only
    assert atom_feats.shape == (1, L, MAX_SC, C_ATOM)  # pooling contract unchanged
    assert bb_feats.shape == (1, L, N_BB, C_ATOM)      # q_sc_bb: the 4 backbone slots


def test_gradient_reaches_sphi_sidechain_atoms_through_q():
    """The whole point of the atom channel: a loss on the BACKBONE decoder's q must
    train S_phi through its side-chain (last-10) slots. If S_phi's backbone slots did
    not attend to the side-chain slots, `noisy.grad` would be exactly zero and the
    channel would be decorative."""
    torch.manual_seed(0)
    sphi = SideChainModule(c_res=C_RES, c_atom=C_ATOM, n_type=len(STD_AA_3))
    noisy = torch.randn(1, L, MAX_SC, 3, requires_grad=True)   # the LAST 10 slots
    bb_local = torch.randn(1, L, N_BB, 3)
    _y0, _af, bb_feats = _sphi_bb_feats(sphi, noisy, bb_local)

    stub = _StubModel()                                        # non-zero residual branch
    q_skip = torch.randn(1, 2, N_ATOM, C_Q, requires_grad=True)
    _armed(stub, bb_feats)
    q_new = _q_out(_prehook(stub, q_skip))
    q_new.pow(2).sum().backward()                              # a loss on the backbone q

    assert noisy.grad is not None
    assert torch.isfinite(noisy.grad).all()
    assert float(noisy.grad.abs().max()) > 0.0                 # side-chain slots move
    for p in ("w_xyz", "w_res"):
        g = getattr(sphi, p).weight.grad
        assert g is not None and float(g.abs().max()) > 0.0
    g_embed = sphi.atom_embed.weight.grad
    assert g_embed is not None and float(g_embed.abs().max()) > 0.0
    # and the backbone's own encoder still gets gradient (q_bb is the residual base)
    assert q_skip.grad is not None and float(q_skip.grad.abs().max()) > 0.0


# ----------------------------------------------------------- wiring / ablation


def test_config_default_is_off_and_independent_of_a_direct():
    sc = training_configs["sidechain"]
    assert sc["q_direct"] is False and sc["a_direct"] is False
    assert sc["q_direct_zero_init"] is True
    # two independent switches -> the no / a-only / q-only / a+q ablation
    assert "a_direct" in sc and "q_direct" in sc


def test_q_direct_requires_coevolution_and_is_pass_scoped():
    """Source guards: q_direct is gated on enable_coevolution (there is no refinement
    pass without it), and the flag is armed around the refinement diffusion call only,
    with a finally-clause so a later first pass can never inherit a live flag."""
    init_src = inspect.getsource(ProtenixDesignTrain.__init__)
    assert "self.sc_q_direct and not self.enable_coevolution" in init_src
    assert "register_forward_pre_hook" in init_src

    fwd = inspect.getsource(ProtenixDesignTrain._train_forward)
    assert fwd.count("self._q_direct_active = False") >= 2      # reset + finally
    assert "self._q_direct_active = bool(" in fwd
    armed = fwd.index("self._q_direct_active = bool(")
    assert fwd.index("finally:", armed) > armed


def test_cogenerate_mirrors_the_injection_at_inference():
    """A trained QAtomFusion must not be dead weight at sampling."""
    from pxdesign_train import cogenerate as cg

    src = inspect.getsource(cg.cogenerate)
    assert "model._q_sc_cache = q_sc_inject" in src
    assert "model._q_bb_idx_cache = q_idx_inject" in src
    assert "model._q_direct_active = bool(" in src
    assert "model._q_direct_active = False" in src               # finally-disarm
    assert "bb_local" in src                                     # S_phi gets its 4 slots


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
