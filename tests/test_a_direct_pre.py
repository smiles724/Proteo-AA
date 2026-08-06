"""`a_direct_pre`: the SAME fusion as `a_direct`, injected one stage earlier.

Why the injection point is not a detail. Inside `DiffusionModule` the order is

    q (per atom) -> [sequence-local atom attention] -> a = mean(q)
      -> a + W.LN(s_single) -> [GLOBAL cross-residue attention]  <-- DiffusionTransformer
      -> layernorm_a -> [sequence-local atom attention] -> coordinates

The DiffusionTransformer is the ONLY place where two residues that are far apart
in sequence but adjacent in space exchange information; every other attention in
the module is windowed over the atom index. `a_direct` hooks `layernorm_a`, i.e.
*after* that step, so a side-chain summary injected there can never reach the
backbone of a spatial neighbour — which is exactly the interaction ("a bulky side
chain displaces the residue packed against it") that motivates the co-evolution
channel. `a_direct_pre` rewrites the DiffusionTransformer's `a` argument instead.

These tests pin the mechanism, not a loss value: that the pre-hook rewrites the
argument the global attention consumes, that it stays pass-scoped and idempotent,
and that it is genuinely a different injection point from `a_direct`.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault(
    "fast_layer_norm_cuda_v2", types.ModuleType("fast_layer_norm_cuda_v2")
)

from pxdesign_train.model import ProtenixDesignTrain  # noqa: E402
from pxdesign_train.sidechain.coevolution import ATokenFusion  # noqa: E402

B, N_SAMPLE, L, C_TOKEN, C_ATOM = 1, 2, 5, 16, 8


def _feats():
    torch.manual_seed(0)
    a_bb = torch.randn(B, N_SAMPLE, L, C_TOKEN)
    a_sc = torch.randn(B, L, C_ATOM)
    return a_bb, a_sc


class _StubModel:
    """Minimal object carrying just what the pre-hook touches."""

    _a_token_pre_attention_hook = ProtenixDesignTrain._a_token_pre_attention_hook
    _align_a_sc = ProtenixDesignTrain._align_a_sc
    _warn_once = ProtenixDesignTrain._warn_once

    def __init__(self, enabled=True, zero_init=False):
        self.sc_a_direct_pre = enabled
        self._a_direct_active = False
        self._a_sc_cache = None
        self.a_token_fusion_pre = ATokenFusion(
            c_token=C_TOKEN, c_atom=C_ATOM, zero_init=zero_init
        )
        if not zero_init:
            torch.nn.init.normal_(self.a_token_fusion_pre.mlp[-1].weight, std=0.5)


def _hook(stub, a_bb, **extra):
    kwargs = {"a": a_bb, "s": None, "z": None}
    kwargs.update(extra)
    return stub._a_token_pre_attention_hook(None, (), kwargs)


def test_no_injection_in_the_first_pass():
    """S_phi has not run yet, so there is nothing to inject."""
    stub = _StubModel()
    a_bb, _ = _feats()
    assert _hook(stub, a_bb) is None            # flag down
    stub._a_direct_active = True
    stub._a_sc_cache = None
    assert _hook(stub, a_bb) is None            # armed, but no side-chain summary


def test_injection_rewrites_the_attention_input():
    stub = _StubModel()
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True

    result = _hook(stub, a_bb)
    assert result is not None
    args, kwargs = result
    fused = kwargs["a"]
    assert fused.shape == a_bb.shape
    assert not torch.allclose(fused, a_bb)
    a_sc_e = a_sc.unsqueeze(1).expand(B, N_SAMPLE, L, C_ATOM)
    assert torch.allclose(fused, stub.a_token_fusion_pre(a_bb, a_sc_e), atol=1e-6)
    # every other argument is passed through untouched
    assert set(kwargs) == {"a", "s", "z"}
    assert args == ()


def test_injection_is_idempotent_under_repeated_hook_call():
    """Activation-checkpoint recomputation fires the hook again on the same input;
    the residual must not compound."""
    stub = _StubModel()
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True
    f1 = _hook(stub, a_bb)[1]["a"]
    f2 = _hook(stub, a_bb)[1]["a"]
    assert torch.equal(f1, f2)
    assert torch.equal(stub._a_sc_cache, a_sc)   # cache never mutated


def test_zero_init_is_an_exact_noop():
    """Turning the arm on must not perturb a trained model at step 0."""
    stub = _StubModel(zero_init=True)
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True
    assert torch.allclose(_hook(stub, a_bb)[1]["a"], a_bb, atol=1e-7)


def test_positional_call_is_a_noop_not_a_wrong_injection():
    """If DiffusionModule ever stops passing `a` as a keyword, skip rather than
    rewrite whatever happens to sit in kwargs."""
    stub = _StubModel()
    a_bb, a_sc = _feats()
    stub._a_sc_cache = a_sc
    stub._a_direct_active = True
    assert stub._a_token_pre_attention_hook(None, (a_bb,), {"s": None}) is None


def test_hook_target_is_before_the_global_attention():
    """The two arms must hook DIFFERENT stages, or they are the same experiment.

    `a_direct` -> `layernorm_a` (after the DiffusionTransformer);
    `a_direct_pre` -> `diffusion_transformer` itself (before it).
    """
    import inspect

    src = inspect.getsource(ProtenixDesignTrain.__init__)
    assert "layernorm_a.register_forward_hook" in src
    assert "diffusion_transformer.register_forward_pre_hook" in src

    from protenix.model.modules.diffusion import DiffusionModule

    body = inspect.getsource(DiffusionModule.f_forward)
    i_attn = body.index("self.diffusion_transformer(")
    i_ln = body.index("self.layernorm_a(")
    assert i_attn < i_ln, (
        "DiffusionModule no longer runs the global attention before layernorm_a; "
        "the premise separating a_direct_pre from a_direct is gone."
    )
