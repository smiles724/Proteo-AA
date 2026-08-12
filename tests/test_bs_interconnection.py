import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from pxdesign_train.sidechain.coevolution import AResBSConcat
from pxdesign_train.sidechain.module import SideChainModule


def test_aresbsconcat_zero_init_is_identity():
    m = AResBSConcat(c_atom=16)
    pooled = torch.randn(2, 5, 16)
    a_token = torch.randn(2, 5, 16)
    out = m(pooled, a_token)
    assert torch.allclose(out, pooled, atol=1e-6), "zero-init must be exact identity"


def test_aresbsconcat_a_token_reaches_output_when_armed():
    m = AResBSConcat(c_atom=16)
    torch.nn.init.normal_(m.mlp[-1].weight, std=0.1)  # arm the residual branch
    pooled = torch.randn(1, 3, 16)
    out_a = m(pooled, torch.zeros(1, 3, 16))
    out_b = m(pooled, torch.ones(1, 3, 16))
    assert not torch.allclose(out_a, out_b), "a_token must influence the output when armed"


def _sc_inputs():
    B, L, A = 1, 3, 10
    h_res = torch.randn(B, L, 8)
    logits = torch.randn(B, L, 20)
    ids = torch.randint(1, 5, (B, L, A))
    mask = torch.ones(B, L, A, dtype=torch.bool)
    noisy = torch.randn(B, L, A, 3)
    ca = torch.randn(B, L, 3)
    return h_res, logits, ids, mask, noisy, ca


def test_a_bs_concat_off_matches_baseline():
    torch.manual_seed(0)
    base = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=False).eval()
    on = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=True).eval()
    on.load_state_dict(base.state_dict(), strict=False)  # shared weights; fusion is zero-init identity
    h, l, ids, m, noisy, ca = _sc_inputs()
    with torch.no_grad():
        y0, _ = base(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
        y1, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
    assert torch.allclose(y0, y1, atol=1e-6), "zero-init a_bs_concat must match baseline"


def test_a_bs_concat_changes_output_when_armed():
    torch.manual_seed(0)
    on = SideChainModule(c_res=8, c_atom=16, n_type=20, a_bs_concat=True).eval()
    torch.nn.init.normal_(on.a_bs_concat_fusion.mlp[-1].weight, std=0.1)
    h, l, ids, m, noisy, ca = _sc_inputs()
    with torch.no_grad():
        y_on, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
        on.a_bs_concat = False
        y_off, _ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca)
    assert not torch.allclose(y_on, y_off, atol=1e-6), "armed a_bs_concat must change the output"


from pxdesign_train.sidechain.coevolution import QAtomBSFusion


def test_qatombsfusion_zero_init_identity_then_reaches_output():
    m = QAtomBSFusion(c_atom=16, c_q=8)
    bb_slot = torch.randn(1, 3, 4, 16)
    bb_q = torch.randn(1, 3, 4, 8)
    assert torch.allclose(m(bb_slot, bb_q), bb_slot, atol=1e-6), "zero-init identity"
    torch.nn.init.normal_(m.mlp[-1].weight, std=0.1)
    out_a = m(bb_slot, torch.zeros(1, 3, 4, 8))
    out_b = m(bb_slot, torch.ones(1, 3, 4, 8))
    assert not torch.allclose(out_a, out_b), "backbone q must influence the slots when armed"


def test_q_bs_backbone_q_reaches_slots_and_off_is_baseline():
    torch.manual_seed(0)
    B, L, A = 1, 3, 10
    h, l, ids, m, noisy, ca = _sc_inputs()
    bb_local = torch.randn(B, L, 4, 3)        # triggers the 14-slot path
    bb_q_near = torch.randn(B, L, 4, 8)
    bb_q_far = bb_q_near + 5.0

    on = SideChainModule(c_res=8, c_atom=16, n_type=20, q_bs=True, c_q=8).eval()
    torch.nn.init.normal_(on.q_bs_fusion.mlp[-1].weight, std=0.1)
    with torch.no_grad():
        y_near, *_ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca, bb_coords=bb_local, bb_q=bb_q_near)
        y_far, *_ = on(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca, bb_coords=bb_local, bb_q=bb_q_far)
    assert not torch.allclose(y_near, y_far, atol=1e-6), "backbone q must reach the side chain"

    off = SideChainModule(c_res=8, c_atom=16, n_type=20, q_bs=False, c_q=8).eval()
    with torch.no_grad():
        y_a, *_ = off(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca, bb_coords=bb_local, bb_q=bb_q_near)
        y_b, *_ = off(h, l, ids, m, noisy, torch.ones(1), ca_coords=ca, bb_coords=bb_local, bb_q=bb_q_far)
    assert torch.allclose(y_a, y_b, atol=1e-6), "q_bs off must ignore bb_q entirely"


def test_model_exposes_bs_flags_default_off():
    import copy
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs
    from pxdesign_train.model import ProtenixDesignTrain

    t = copy.deepcopy(training_configs)
    t["enable_sidechain"] = True
    t["sidechain"]["q_bs"] = True
    t["sidechain"]["a_bs_concat"] = True
    cfg = parse_configs(t, arg_str="")
    cfg.load_strict = False
    m = ProtenixDesignTrain(cfg)
    assert m.sc_a_bs_concat is True and m.sc_q_bs is True
    assert m.sc_bb_context is True, "q_bs must imply bb_context (14-slot)"


# ----------------------------------------------------------------------
# q_bs READ-path cache must survive activation-checkpoint recomputation.
#
# `_q_skip_encoder_hook` is a plain forward hook (not a pre-hook, not gated on
# a flag): it just caches `out[1]` unconditionally every time the atom
# attention encoder runs, and returns None so the encoder's output is never
# replaced. Unlike the q_direct WRITE path (test_q_checkpoint_recompute.py),
# this hook does not inject anything into the checkpointed call, so it cannot
# desync the checkpoint's saved-tensor count. What it must still get right:
# the backbone is wrapped in `torch.utils.checkpoint`, whose backward RE-RUNS
# the encoder — re-firing this hook — so `_q_skip_cache` gets overwritten a
# second time, AFTER the side-chain module has already consumed it during the
# forward. That overwrite must reproduce the same value the forward produced
# (deterministic recompute), not silently diverge, or leave the cache in an
# inconsistent state for whatever later reads it.
# ----------------------------------------------------------------------

from pxdesign_train.model import ProtenixDesignTrain


class _Encoder(nn.Module):
    """Stand-in for Protenix's AtomAttentionEncoder: returns
    (a_token, q_skip, c_skip, p_skip), the tuple `_q_skip_encoder_hook` reads.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        self.lin = nn.Linear(c, c)

    def forward(self, x):
        q_skip = self.lin(x)
        a_token = q_skip.sum(dim=-2)
        c_skip = torch.zeros_like(q_skip)
        p_skip = torch.zeros_like(q_skip)
        return a_token, q_skip, c_skip, p_skip


class _CacheModel:
    """Minimal carrier for the real encoder hook (no Protenix weights needed)."""

    _q_skip_encoder_hook = ProtenixDesignTrain._q_skip_encoder_hook

    def __init__(self) -> None:
        self._q_skip_cache = None


def test_q_skip_cache_populated_by_checkpointed_forward():
    """Under PXDesign's actual default checkpoint settings
    (`use_reentrant=False`, early-stop ON — see the next test for why that
    matters), the side-chain gather reads `_q_skip_cache` synchronously
    during the forward. Confirm it is populated with exactly that forward's
    q_skip, and that running backward afterwards (whatever it does to the
    cache) never leaves it diverged from what was already consumed."""
    m = _CacheModel()
    enc = _Encoder(6)
    enc.register_forward_hook(m._q_skip_encoder_hook)

    x = torch.randn(2, 4, 6, requires_grad=True)
    out = checkpoint(enc, x, use_reentrant=False)
    a_token, q_skip, _c_skip, _p_skip = out

    # This is the moment the side-chain gather (point-3 READ path) would
    # consume `model._q_skip_cache` — synchronously, during the forward.
    forward_cache = m._q_skip_cache
    assert forward_cache is not None, "encoder hook never populated the cache"
    assert torch.equal(forward_cache, q_skip), (
        "cache read during the forward must be exactly the forward's q_skip"
    )
    forward_snapshot = forward_cache.clone()

    (a_token.sum() + q_skip.sum()).backward()  # must not raise

    assert m._q_skip_cache is not None
    assert torch.allclose(m._q_skip_cache, forward_snapshot, atol=1e-6), (
        "backward left the cache diverged from the value the side-chain "
        "gather actually consumed during the forward"
    )


def test_q_skip_cache_matches_forward_even_when_recompute_genuinely_refires():
    """Guard the guard, and the real point of this file.

    `_q_skip_encoder_hook` is a plain forward POST-hook. Empirically (verified
    above by construction), PyTorch's non-reentrant checkpoint's default
    "early stop" optimization means backward often does NOT re-run the
    encoder all the way to its `return` — so `register_forward_hook`
    callbacks frequently do *not* fire a second time, unlike the q_direct
    WRITE path's forward PRE-hook (`test_q_checkpoint_recompute.py`), which
    always fires because pre-hooks run before any op (hence before early-stop
    can possibly trigger). That makes the previous test structurally unable
    to prove the cache survives a genuine second write.

    `torch.utils.checkpoint.set_checkpoint_early_stop(False)` forces the
    backward recompute to run the encoder to completion, guaranteeing the
    post-hook fires again. We use it here to prove: IF the hook ever does
    refire on recompute (this setting, a future encoder shape, or a future
    torch version), the value it writes is byte-identical to the forward's
    (deterministic recompute) — so `_q_skip_cache` can never end up holding
    something the side-chain gather never actually saw.
    """
    calls = []
    m = _CacheModel()
    enc = _Encoder(6)
    enc.register_forward_hook(lambda *_a, **_k: calls.append(1))
    enc.register_forward_hook(m._q_skip_encoder_hook)

    x = torch.randn(2, 4, 6, requires_grad=True)
    with set_checkpoint_early_stop(False):
        out = checkpoint(enc, x, use_reentrant=False)
    a_token, q_skip, _c_skip, _p_skip = out
    assert len(calls) == 1, "checkpoint must defer recomputation to backward"
    forward_snapshot = m._q_skip_cache.clone()

    (a_token.sum() + q_skip.sum()).backward()

    assert len(calls) == 2, (
        "recompute did not actually re-run the encoder -- this test is "
        "vacuous and not exercising what it claims to"
    )
    assert torch.allclose(m._q_skip_cache, forward_snapshot, atol=1e-6), (
        "recompute re-fired the hook with a q_skip that diverges from the "
        "forward pass -- the cache would silently corrupt for a later reader"
    )


def test_q_skip_cache_unaffected_when_encoder_output_not_tuple():
    """Defensive: the hook only touches the cache for the (a, q, c, p) shape it
    documents; anything else must leave a pre-existing cache alone (not crash,
    not clobber it with garbage)."""
    m = _CacheModel()
    m._q_skip_cache = torch.ones(1)
    result = m._q_skip_encoder_hook(None, None, torch.zeros(3))
    assert result is None
    assert torch.equal(m._q_skip_cache, torch.ones(1)), (
        "a non-tuple encoder output must not disturb the existing cache"
    )
