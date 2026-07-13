"""The q-injection must survive activation-checkpoint RECOMPUTATION.

This is the test the rest of test_q_direct.py structurally cannot be: every hook test
there calls `model._q_skip_decoder_pre_hook(...)` directly, so none of them ever runs
the decoder inside a real checkpoint region — and that is exactly where q_direct broke.

PXDesign's default training config (pxdesign/configs/configs_base.py:
`use_fine_grained_checkpoint=True`, `blocks_per_ckpt=1`) calls the atom decoder through
`torch.utils.checkpoint(..., use_reentrant=False)`, which RE-RUNS it during backward.
A pre-hook gated on a time-varying flag (armed before the refinement call, cleared in a
`finally`) therefore injects on the forward and NOT on the recompute:

    CheckpointError: A different number of tensors was saved during the original
                     forward and recomputation. forward: 9, recomputation: 4

...and QAtomFusion / S_phi get no gradient at all. The fix is to key the decision on the
CALL (the identity of the incoming q_skip tensor), never on a flag, so forward and
recompute make the same decision. These tests pin that.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pxdesign_train.model import ProtenixDesignTrain
from pxdesign_train.sidechain.coevolution import QAtomFusion


class _Decoder(nn.Module):
    """Stand-in for Protenix's AtomAttentionDecoder: consumes q_skip, is checkpointed."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.lin = nn.Linear(c, c)

    def forward(self, atom_to_token_idx=None, a=None, q_skip=None, c_skip=None, p_skip=None):
        return self.lin(q_skip).sum(dim=-1, keepdim=True)


class _Model:
    """Minimal carrier for the real pre-hook + real fusion (no Protenix weights needed)."""

    _q_skip_decoder_pre_hook = ProtenixDesignTrain._q_skip_decoder_pre_hook
    _fuse_q_backbone_atoms = ProtenixDesignTrain._fuse_q_backbone_atoms

    def __init__(self, c_q: int, c_sc: int) -> None:
        self.q_atom_fusion = QAtomFusion(c_q=c_q, c_atom=c_sc)
        self._q_direct_active = False
        self._q_sc_cache = None
        self._q_bb_idx_cache = None

    def _log(self, *_a, **_k):  # pragma: no cover
        pass


def _build(n_atom=12, c_q=8, c_sc=8, n_res=2):
    m = _Model(c_q, c_sc)
    # De-zero the fusion so injection is observable (it ships zero-init by design).
    with torch.no_grad():
        for p in m.q_atom_fusion.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    dec = _Decoder(c_q)
    dec.register_forward_pre_hook(m._q_skip_decoder_pre_hook, with_kwargs=True)
    # residue r owns backbone atoms 4r..4r+3
    # q_skip [N_sample=1, N_atom, c_q]; q_sc must match its rank; bb_idx one less.
    bb_idx = torch.arange(n_res * 4).reshape(n_res, 4)
    q_skip = torch.randn(1, n_atom, c_q)
    return m, dec, q_skip, bb_idx


def _run(m, dec, q_skip, q_sc, bb_idx, *, checkpointed: bool):
    """One refinement-pass decoder call, optionally inside a checkpoint region."""
    m._q_sc_cache = q_sc
    m._q_bb_idx_cache = bb_idx
    m._q_inject_calls = {}
    m._q_direct_active = True
    try:
        if checkpointed:
            out = checkpoint(
                dec, None, None, q_skip, None, None, use_reentrant=False
            )
        else:
            out = dec(atom_to_token_idx=None, a=None, q_skip=q_skip,
                      c_skip=None, p_skip=None)
    finally:
        m._q_direct_active = False          # exactly what the real model does
    return out


def test_backward_survives_checkpoint_recomputation():
    """The regression: forward injects, recompute must inject too, or backward explodes."""
    m, dec, q_skip, bb_idx = _build()
    q_sc = torch.randn(2, 4, 8, requires_grad=True)

    out = _run(m, dec, q_skip, q_sc, bb_idx, checkpointed=True)
    out.sum().backward()                    # <-- CheckpointError before the fix

    assert q_sc.grad is not None, "S_phi's backbone-slot features got NO gradient"
    assert q_sc.grad.abs().sum() > 0
    g = m.q_atom_fusion.mlp[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "QAtomFusion got NO gradient"


def test_checkpointed_and_uncheckpointed_agree():
    """The recompute must reproduce the forward's value, not a different branch."""
    m, dec, q_skip, bb_idx = _build()
    q_sc = torch.randn(2, 4, 8)

    plain = _run(m, dec, q_skip, q_sc, bb_idx, checkpointed=False)
    ckpt = _run(m, dec, q_skip, q_sc, bb_idx, checkpointed=True)
    assert torch.allclose(plain, ckpt, atol=1e-6)


def test_first_pass_never_injects_even_on_recompute():
    """The first pass's decoder also recomputes; a stale armed flag would inject there."""
    m, dec, q_skip, bb_idx = _build()
    q_sc = torch.randn(2, 4, 8)

    # Refinement pass: records its own q_skip as an injecting call.
    _run(m, dec, q_skip, q_sc, bb_idx, checkpointed=True)

    # A LATER first pass: cache is populated, flag is down -> must be a pass-through,
    # both on its forward and on its backward recomputation.
    q_first = torch.randn_like(q_skip)
    m._q_direct_active = False
    ref = dec.lin(q_first).sum(dim=-1, keepdim=True)
    got = checkpoint(dec, None, None, q_first, None, None, use_reentrant=False)
    assert torch.allclose(got, ref, atol=1e-6), "first pass was injected into"
    got.sum().backward()                    # must not raise


def test_flag_based_gating_would_have_failed():
    """Guard the guard: prove this harness really does exercise the recompute path.

    We simulate the OLD (flag-keyed) hook. If this does not blow up, the test above is
    vacuous and the checkpoint region is not actually recomputing.
    """
    m, dec_unused, q_skip, bb_idx = _build()
    q_sc = torch.randn(2, 4, 8, requires_grad=True)
    dec = _Decoder(8)

    def _flag_keyed(_module, args, kwargs):
        if not m._q_direct_active:                       # the old, broken gate
            return None
        pos = None
        if "q_skip" in kwargs:
            q_in = kwargs["q_skip"]
        elif len(args) > 2 and torch.is_tensor(args[2]):
            pos, q_in = 2, args[2]
        else:
            return None
        q_new = m._fuse_q_backbone_atoms(q_in, m._q_sc_cache, m._q_bb_idx_cache)
        if q_new is None:
            return None
        if pos is None:
            kwargs = dict(kwargs)
            kwargs["q_skip"] = q_new
        else:
            args = list(args)
            args[pos] = q_new
            args = tuple(args)
        return args, kwargs

    dec.register_forward_pre_hook(_flag_keyed, with_kwargs=True)
    m._q_sc_cache, m._q_bb_idx_cache = q_sc, bb_idx
    m._q_direct_active = True
    out = checkpoint(dec, None, None, q_skip, None, None, use_reentrant=False)
    m._q_direct_active = False               # the `finally` — flag down before backward

    with pytest.raises(Exception):           # CheckpointError (or a silent-grad mismatch)
        out.sum().backward()
