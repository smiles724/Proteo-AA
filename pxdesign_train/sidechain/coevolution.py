"""Backbone<->side-chain cycle closure (Stage II-B co-evolution).

Overleaf (SideCraft) §Shared residue representation / §Iterative Coupling:
after the Side-Chain Module updates h_res -> h_res', the SAME Backbone Module is
reused to refine backbone geometry and residue identity:

    (x0_post, p_post(a)) = B_theta^refine(h_res', x0, p(a), e_t)

with post-refinement losses L_bb^post and L_aa^post. We do NOT add a separate
refinement head — we reuse B_theta.

On our PXDesign substrate the backbone's token representation (a_token) is
computed *inside* the DiffusionModule, so we feed h_res' back by injecting it
into the backbone's token trunk `s_trunk` (which PXDesign leaves as zeros). The
reused denoise pass then becomes side-chain-aware. `HResInjector` is that
projection; everything else in the cycle reuses existing modules (B_theta, the
AA head, S_phi, HResFeedback).
"""
import torch
import torch.nn as nn


class HResInjector(nn.Module):
    """Project the side-chain-updated h_res' into the backbone trunk space
    (c_s), to be added to s_trunk before the reused refinement denoise pass."""

    def __init__(self, c_hres: int, c_trunk: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_hres),
            nn.Linear(c_hres, c_trunk),
        )

    def forward(self, h_res_prime: torch.Tensor) -> torch.Tensor:
        """h_res_prime: [..., N_token, c_hres] -> [..., N_token, c_trunk]."""
        return self.proj(h_res_prime)
