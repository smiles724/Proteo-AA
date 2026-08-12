"""Tests for the EDM side-chain diffusion (pxdesign_train/sidechain/edm.py).

The properties pinned here are the ones whose violation is silent: a
preconditioning that does not reduce to the identity at sigma=0, a loss weight
that does not undo c_out (so high-sigma draws dominate the gradient), and a
"reverse loop" that does not actually carry state -- which is precisely the bug
in the current sampler that A2 exists to fix.
"""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.edm import (
    DEFAULT_SIGMA_DATA,
    SideChainEDM,
    SideChainNoiseSampler,
    edm_loss_weight,
    edm_scalings,
    noise_sidechains,
    sidechain_reverse_loop,
)
from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    sidechain_atom_name_ids,
    sidechain_mask,
)
from pxdesign_train.sidechain.module import SideChainModule

C_RES = 16


def _module(**kw):
    return SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                           n_heads=4, **kw)


def _batch():
    restypes = ["ALA", "PHE", "LYS"]
    atom_mask = sidechain_mask(restypes)[None]
    atom_ids = sidechain_atom_name_ids(restypes)[None]
    h_res = torch.randn(1, 3, C_RES)
    logits = torch.randn(1, 3, 20)
    ca = torch.randn(1, 3, 3) * 30.0        # a real assembly is far from the origin
    x = ca[:, :, None, :] + torch.randn(1, 3, MAX_SC, 3)
    return atom_mask, atom_ids, h_res, logits, ca, x


# ---------------------------------------------------------------- scalings ----

def test_scalings_reduce_to_the_identity_at_zero_noise():
    """sigma -> 0 must give D(x; sigma) -> x. If c_skip does not go to 1 the
    denoiser destroys a clean input, which shows up as a floor on the loss that
    no amount of training removes."""
    s = torch.tensor([1e-4])
    c_skip, c_in, c_out, _ = edm_scalings(s, DEFAULT_SIGMA_DATA)
    assert c_skip.item() == pytest.approx(1.0, abs=1e-6)
    assert c_out.item() == pytest.approx(0.0, abs=1e-3)
    assert c_in.item() == pytest.approx(1.0 / DEFAULT_SIGMA_DATA, rel=1e-4)


def test_scalings_hand_everything_to_the_network_at_high_noise():
    s = torch.tensor([1e4])
    c_skip, _, c_out, _ = edm_scalings(s, DEFAULT_SIGMA_DATA)
    assert c_skip.item() == pytest.approx(0.0, abs=1e-6)
    assert c_out.item() == pytest.approx(DEFAULT_SIGMA_DATA, rel=1e-3)


def test_loss_weight_is_exactly_one_over_c_out_squared():
    """lambda(sigma) = 1 / c_out(sigma)^2 is what makes the effective target
    uniform across sigma. Getting it wrong is the standard quiet EDM failure:
    training still converges, just weighted towards the noise levels that matter
    least."""
    s = torch.logspace(-2, 1, 25)
    _, _, c_out, _ = edm_scalings(s, DEFAULT_SIGMA_DATA)
    assert torch.allclose(edm_loss_weight(s, DEFAULT_SIGMA_DATA), 1.0 / c_out**2,
                          rtol=1e-5)


def test_sigma_data_is_a_side_chain_scale_not_the_backbone_one():
    """Reusing the backbone's 16.0 would put the whole side-chain sigma range in
    the corner where c_skip ~ 1 -- the network would barely be consulted."""
    s = torch.tensor([2.0])
    c_skip_sc, *_ = edm_scalings(s, DEFAULT_SIGMA_DATA)
    c_skip_bb, *_ = edm_scalings(s, 16.0)
    assert c_skip_sc.item() < 0.6, "side-chain scaling should consult the network"
    assert c_skip_bb.item() > 0.95, "the backbone scale would pass the input through"


# ------------------------------------------------------------------ sampler ---

def test_sampler_respects_its_clamp_and_covers_the_real_error_scale():
    smp = SideChainNoiseSampler()
    s = smp((20000,), generator=torch.Generator().manual_seed(0))
    assert (s >= smp.sigma_min).all() and (s <= smp.sigma_max).all()
    # The measured template-to-target error is 2.18 A; a sampler that never draws
    # near it trains the model on a regime it will not meet.
    assert (s > 2.0).float().mean() > 0.02, "sigma range misses the template's error"
    assert (s < 0.5).float().mean() > 0.05, "sigma range misses the fine-structure end"


def test_schedule_is_descending_and_lands_on_zero():
    sch = SideChainNoiseSampler().schedule(8)
    assert sch.shape == (9,)
    assert torch.all(sch[:-1] > sch[1:]), "schedule must decrease monotonically"
    assert sch[-1].item() == 0.0


def test_noise_has_the_requested_scale():
    x = torch.zeros(4, 50, MAX_SC, 3)
    sig = torch.tensor([0.1, 0.5, 1.0, 3.0])
    out = noise_sidechains(x, sig, generator=torch.Generator().manual_seed(0))
    for i, s in enumerate(sig.tolist()):
        assert out[i].std().item() == pytest.approx(s, rel=0.1)


# ---------------------------------------------------------------- denoiser ----

def test_denoise_preserves_the_module_s_extra_outputs():
    """The wrapper must stay transparent: callers still need atom_feats (h_res'
    pooling) and, on the 14-slot axis, bb_feats (the q_direct channel)."""
    torch.manual_seed(0)
    atom_mask, atom_ids, h_res, logits, ca, x = _batch()
    edm = SideChainEDM(_module(centre_coord_input=True).eval())
    with torch.no_grad():
        out = edm.denoise(x, torch.tensor([1.0]), ca,
                          h_res, logits, atom_ids, atom_mask)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1].shape == (1, 3, MAX_SC, 32)


def test_denoise_is_near_identity_at_tiny_sigma():
    torch.manual_seed(0)
    atom_mask, atom_ids, h_res, logits, ca, x = _batch()
    edm = SideChainEDM(_module(centre_coord_input=True).eval())
    with torch.no_grad():
        x0, _feats = edm.denoise(x, torch.tensor([1e-4]), ca,
                                 h_res, logits, atom_ids, atom_mask)
    # c_skip ~ 1 and c_out ~ 0, so whatever the network says is scaled away.
    assert torch.allclose(x0, x, atol=1e-2)


def test_denoise_moves_the_input_at_realistic_sigma():
    torch.manual_seed(0)
    atom_mask, atom_ids, h_res, logits, ca, x = _batch()
    edm = SideChainEDM(_module(centre_coord_input=True).eval())
    with torch.no_grad():
        x0, _feats = edm.denoise(x, torch.tensor([2.0]), ca,
                                 h_res, logits, atom_ids, atom_mask)
    assert not torch.allclose(x0, x, atol=1e-3)
    assert torch.isfinite(x0).all()


def test_edm_refuses_to_stack_on_template_residual():
    """Both add a skip from the input to the output. c_skip(sigma) already IS the
    template residual, with a sigma-dependent magnitude; keeping the fixed one as
    well double-counts the template."""
    with pytest.raises(ValueError, match="template_residual"):
        SideChainEDM(_module(template_residual=True))


# -------------------------------------------------------------- reverse loop --

class _ConstantDenoiser:
    """Always predicts the same x0, so the loop's fixed point is known exactly."""

    def __init__(self, target):
        self.target = target
        self.calls = 0

    def denoise(self, x, sigma, ca, *a, **k):
        self.calls += 1
        return self.target


def test_reverse_loop_carries_state_and_converges_to_the_fixed_point():
    """The property the current sampler does NOT have.

    With a denoiser whose answer is constant, Euler integration from sigma_max to
    0 must land on that answer. A loop that re-initialised from the template each
    step (what cogenerate does today) would instead return one step's worth of
    progress, however many steps it took.
    """
    target = torch.randn(1, 3, MAX_SC, 3)
    sch = SideChainNoiseSampler().schedule(24)
    x_init = target + sch[0] * torch.randn_like(target)
    d = _ConstantDenoiser(target)
    out = sidechain_reverse_loop(d, x_init, sch, torch.zeros(1, 3, 3))
    assert d.calls == 24
    assert torch.allclose(out, target, atol=1e-4), (
        "the loop did not carry its estimate forward"
    )


def test_more_steps_get_closer_than_one_step():
    """The point of A2: N calls should buy N calls' worth of refinement. Today N
    independent re-initialised calls buy one."""
    target = torch.randn(1, 3, MAX_SC, 3)
    smp = SideChainNoiseSampler()
    err = []
    for n in (1, 4, 16):
        sch = smp.schedule(n)
        x_init = target + sch[0] * torch.randn(target.shape,
                                               generator=torch.Generator().manual_seed(0))
        out = sidechain_reverse_loop(_ConstantDenoiser(target), x_init, sch,
                                     torch.zeros(1, 3, 3))
        err.append((out - target).abs().max().item())
    assert err[0] >= err[1] >= err[2] - 1e-9, f"error did not fall with steps: {err}"


def test_reverse_loop_keeps_padding_slots_zero():
    target = torch.randn(1, 3, MAX_SC, 3)
    mask = sidechain_mask(["ALA", "PHE", "LYS"])[None]
    sch = SideChainNoiseSampler().schedule(4)
    out = sidechain_reverse_loop(_ConstantDenoiser(target), target.clone(), sch,
                                 torch.zeros(1, 3, 3), atom_mask=mask)
    assert torch.count_nonzero(out[0, 0, 1:]) == 0      # ALA owns one slot
