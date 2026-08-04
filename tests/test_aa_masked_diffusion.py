"""Residue-type masked-diffusion: time-conditioned head, MDLM-weighted loss,
iterative-unmask sampler, and the 20->design-restype vocab map.

All CPU-only; the sampler is tested with a stub logits_fn so it needs no model.
"""
import torch

from pxdesign_train.heads import DesignResidueTypeHead, sinusoidal_time_embedding
from pxdesign_train.loss import PXDesignLoss
from pxdesign_train.sampler import (
    build_aa20_to_restype36,
    iterative_unmask,
    _unmask_counts,
)


# ---------- time-conditioned head ----------

def test_time_embedding_shape_and_finite():
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_time_embedding(t, 128)
    assert emb.shape == (3, 128)
    assert torch.isfinite(emb).all()


def test_head_without_time_matches_v1_path():
    """aa_t=None must be identical to running the plain proj MLP."""
    head = DesignResidueTypeHead(c_s=449, no_bins=20, use_time=True).eval()
    x = torch.randn(2, 7, 449)
    with torch.no_grad():
        out_none = head(x, aa_t=None)
        out_proj = head.proj(x)
    assert out_none.shape == (2, 7, 20)
    assert torch.allclose(out_none, out_proj)


def test_head_time_conditioning_changes_output():
    head = DesignResidueTypeHead(c_s=449, no_bins=20, use_time=True).eval()
    x = torch.randn(1, 5, 449)
    with torch.no_grad():
        o0 = head(x, aa_t=torch.tensor(0.1))
        o1 = head(x, aa_t=torch.tensor(0.9))
    assert o0.shape == (1, 5, 20)
    assert not torch.allclose(o0, o1)  # time actually enters the head


# ---------- MDLM time-weighted loss ----------

def _loss(**kw):
    return PXDesignLoss(weight_mse=0.0, weight_lddt=0.0, weight_disto=0.0,
                        weight_aa=1.0, **kw)


def test_mdlm_weighting_differs_from_mean_and_flows_grad():
    torch.manual_seed(0)
    n = 6
    logits = torch.randn(1, n, 20, requires_grad=True)
    clean = torch.randint(0, 20, (1, n))
    lossmask = torch.ones(1, n, dtype=torch.long)

    plain = _loss(aa_time_weighting=False)
    weighted = _loss(aa_time_weighting=True)

    ce_plain, _, _ = plain._aa_term(logits, clean, lossmask, aa_t=torch.tensor(0.2))
    ce_w, _, _ = weighted._aa_term(logits, clean, lossmask, aa_t=torch.tensor(0.2))
    # 1/t weighting with a single t rescales but normalisation cancels -> equal
    # here; the real difference shows across a batch of mixed t.
    assert torch.isfinite(ce_w)

    t_mixed = torch.tensor([0.05, 0.9])
    logits2 = torch.randn(2, n, 20, requires_grad=True)
    clean2 = torch.randint(0, 20, (2, n))
    lm2 = torch.ones(2, n, dtype=torch.long)
    ce_mean, _, _ = plain._aa_term(logits2, clean2, lm2, aa_t=t_mixed)
    ce_mdlm, _, _ = weighted._aa_term(logits2, clean2, lm2, aa_t=t_mixed)
    assert not torch.isclose(ce_mean, ce_mdlm)  # weighting changes the scalar
    ce_mdlm.backward()
    assert logits2.grad is not None and torch.isfinite(logits2.grad).all()


def test_aa_term_empty_mask_zero_grad():
    logits = torch.randn(1, 4, 20, requires_grad=True)
    clean = torch.full((1, 4), -100, dtype=torch.long)
    lossmask = torch.zeros(1, 4, dtype=torch.long)
    ce, acc, frac = _loss(aa_time_weighting=True)._aa_term(
        logits, clean, lossmask, aa_t=torch.tensor(0.5)
    )
    ce.backward()
    assert torch.all(logits.grad == 0)
    assert float(frac) == 0.0


# ---------- vocab map ----------

def test_aa20_to_restype36_identity_overlap():
    mapping, xpb = build_aa20_to_restype36()
    assert mapping.shape == (20,)
    assert (mapping[:20] == torch.arange(20)).all()  # 20 AA embed as-is
    assert xpb == 32


# ---------- iterative unmask sampler (stub model) ----------

def test_unmask_counts_sum_to_n():
    assert sum(_unmask_counts(10, 4)) == 10
    assert sum(_unmask_counts(3, 8)) == 3
    assert _unmask_counts(0, 4) == []


# ---------- input_source switch (s_inputs vs diffusion_internal) ----------

def test_config_default_input_source_is_diffusion_internal():
    # The structure-aware a_token (diffusion_internal) is the default;
    # s_inputs is a baseline/ablation only.
    from pxdesign_train.configs.configs_train import training_configs
    assert training_configs["residue_type"]["input_source"] == "diffusion_internal"
    assert training_configs["residue_type"]["trunk_grad_scale"] == 1.0


def test_head_builds_and_runs_at_both_input_dims():
    # A path reads s_inputs (449); B path reads a_token (c_token=768).
    for c_in in (449, 768):
        head = DesignResidueTypeHead(c_s=c_in, no_bins=20, use_time=True).eval()
        x = torch.randn(1, 6, c_in)
        with torch.no_grad():
            out = head(x, aa_t=torch.tensor(0.4))
        assert out.shape == (1, 6, 20)
        assert torch.isfinite(out).all()


def test_a_token_nsample_mean_reduce_contract():
    # B collapses the diffusion N_sample axis by mean before the head:
    # a_token [.., N_sample, N_token, c_token] -> [.., N_token, c_token].
    a = torch.randn(8, 6, 768)  # (N_sample, N_token, c_token)
    reduced = a.mean(dim=-3)
    assert reduced.shape == (6, 768)
    assert torch.allclose(reduced, a.mean(0))


def test_trunk_grad_scale_detach_mix():
    # Forward value is identical for any scale; gradient into the trunk is g*upstream.
    x = torch.randn(3, 4)
    for g in (0.0, 0.25, 1.0):
        xg = x.clone().requires_grad_(True)
        y = g * xg + (1.0 - g) * xg.detach()
        assert torch.allclose(y, xg)                       # forward unchanged
        y.sum().backward()
        assert torch.allclose(xg.grad, torch.full_like(xg, g))  # grad scaled by g


def test_low_sigma_reduce_picks_min_sigma_sample():
    # a: [N_sample, N_token, c]; sigma: [N_sample] -> pick least-noisy sample.
    a = torch.randn(4, 5, 8)
    sigma = torch.tensor([0.9, 0.1, 0.5, 0.7])
    idx = sigma.argmin(dim=-1)
    idx_e = idx[..., None, None, None].expand(*idx.shape, 1, a.shape[-2], a.shape[-1])
    picked = a.gather(dim=-3, index=idx_e).squeeze(-3)
    assert picked.shape == (5, 8)
    assert torch.allclose(picked, a[1])                    # sigma min at index 1


def test_iterative_unmask_fills_all_positions():
    torch.manual_seed(0)
    n_token = 12
    positions = torch.tensor([2, 3, 5, 7, 8])
    # Stub: a fixed "true" identity per position; logits peak there, and grow
    # more confident as neighbours get revealed (not required, just realistic).
    true_aa = {int(p): (int(p) % 20) for p in positions}

    def logits_fn(sampled, mask):
        logits = torch.zeros(n_token, 20)
        for p, aa in true_aa.items():
            logits[p, aa] = 5.0
        return logits

    sampled, traj = iterative_unmask(logits_fn, positions, n_steps=3)
    # every design position filled with its argmax identity
    for p, aa in true_aa.items():
        assert int(sampled[p]) == aa
    # non-design positions stay masked (-1)
    assert int(sampled[0]) == -1
    # trajectory mask fraction is monotonically non-increasing, ends at 0
    fracs = [d["mask_frac"] for d in traj]
    assert fracs == sorted(fracs, reverse=True)
    assert fracs[-1] == 0.0
