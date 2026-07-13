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
from typing import Optional

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


def pool_side_chain_atoms(
    atom_feats: torch.Tensor,   # [..., L, A, c_atom]
    atom_mask: torch.Tensor,    # [..., L, A] bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked mean over a residue's side-chain atoms -> [..., L, c_atom].

    This is the SAME pooling operator `HResFeedback` applies to the same
    `atom_feats` / `atom_mask` (masked mean over the atom axis); `ATokenFusion`
    reuses it so the a-level side-chain summary and the h_res-level one are the
    same quantity, only projected into different spaces. Residues with no
    side-chain atoms pool to 0.
    """
    m = atom_mask[..., None].to(atom_feats.dtype)
    return (atom_feats * m).sum(dim=-2) / (m.sum(dim=-2) + eps)


class ATokenFusion(nn.Module):
    """DIRECT a-level side-chain -> backbone feedback (FangWu's slide):

        a'_bb = a_bb + MLP(LayerNorm(concat(a_bb, W a_sc)))

    The indirect path (`HResInjector`) pushes h_res' into `s_trunk` and lets the
    DiffusionModule recompute `a_token` from scratch, so the fused representation
    never *is* the next round's token. This module fuses at the token level
    itself and keeps the CURRENT pass's backbone token as the residual base (NOT round 1's: a_bb is
    the refinement pass's own freshly recomputed layernorm_a output; only a_sc is
    carried over from round 1), so the
    refinement pass literally consumes a'_bb.

    The residual branch's output layer is zero-initialised: at step 0 the fusion
    is an exact identity, so switching it on cannot perturb a pretrained
    backbone. The output layer's own gradient is non-zero, so the branch leaves
    the zero point after the first optimiser step (and from then on gradients
    reach `a_sc`, i.e. S_phi's atom features).

    `a_sc` is only available AFTER the first round (S_phi has to run first), so
    this fires only in the refinement pass of the co-evolution cycle.
    """

    def __init__(
        self,
        c_token: int,
        c_atom: int,
        c_hidden: Optional[int] = None,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        c_hidden = c_hidden or c_token
        self.eps = eps
        self.sc_proj = nn.Linear(c_atom, c_token)
        self.ln = nn.LayerNorm(2 * c_token)
        self.mlp = nn.Sequential(
            nn.Linear(2 * c_token, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_token),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def pool(self, atom_feats: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Side-chain atom features -> per-token side-chain summary a_sc."""
        return pool_side_chain_atoms(atom_feats, atom_mask, eps=self.eps)

    def forward(self, a_bb: torch.Tensor, a_sc: torch.Tensor) -> torch.Tensor:
        """a_bb: [..., N_token, c_token]; a_sc: [..., N_token, c_atom]
        (already broadcast to a_bb's leading dims) -> [..., N_token, c_token].

        Pure function of its inputs — no in-place update of any cached tensor —
        so re-running it (e.g. activation-checkpoint recomputation firing the
        forward hook a second time) reproduces the same value instead of
        compounding the residual.
        """
        s = self.sc_proj(a_sc.to(a_bb.dtype))
        delta = self.mlp(self.ln(torch.cat([a_bb, s], dim=-1)))
        return a_bb + delta


class QAtomFusion(nn.Module):
    """DIRECT q-level (ATOM-level) side-chain -> backbone feedback (FangWu's slide,
    "Interconnection between Backbone Module and Side-chain Module"):

        q'_bb = q_bb + MLP(LayerNorm(concat(q_bb, W q_sc_bb)))

    `ATokenFusion` closes the loop at the TOKEN level (one vector per residue).
    This closes it at the ATOM level: `q_bb` are the Backbone Module's per-atom
    features (`AtomAttentionEncoder`'s `q_skip`, c_atom=128) for a residue's four
    backbone atoms (N, CA, C, O), and `q_sc_bb` are the Side-Chain Module's
    features for the SAME four atoms — S_phi keeps all 14 ATOM14 slots in its
    representation, attends over them jointly, and "by changing the last 10 it
    adjusts the first 4". Those first 4 come back here and are written into the
    backbone atom decoder's skip connection, so the backbone's own atom rows for
    N/CA/C/O literally consume the side-chain-aware features.

    Structurally identical to `ATokenFusion` (same residual-MLP form, same
    zero-init discipline); only the axis it operates on differs — [..., L, 4, c]
    instead of [..., L, c]. Since both operands are per-atom, the two channels can
    be enabled independently (the no / a / q / a+q ablation).

    The residual branch's output layer is zero-initialised: at step 0 the fusion is
    a bit-exact identity, so switching `q_direct` on cannot perturb a pretrained
    backbone. The output layer's own gradient is non-zero, so it leaves the zero
    point after the first optimiser step, and from then on gradient reaches
    `q_sc_bb` — i.e. S_phi's atom features, including the 10 side-chain slots the
    backbone slots attended to.

    `q_sc_bb` only exists AFTER the first round (S_phi has to run first), so this
    fires only in the refinement pass of the co-evolution cycle.
    """

    def __init__(
        self,
        c_q: int,
        c_atom: int,
        c_hidden: Optional[int] = None,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        c_hidden = c_hidden or c_q
        self.c_q = c_q
        self.sc_proj = nn.Linear(c_atom, c_q)
        self.ln = nn.LayerNorm(2 * c_q)
        self.mlp = nn.Sequential(
            nn.Linear(2 * c_q, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_q),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q_bb: torch.Tensor, q_sc_bb: torch.Tensor) -> torch.Tensor:
        """q_bb: [..., L, 4, c_q] — Backbone Module per-atom features of N, CA, C, O.
        q_sc_bb: [..., L, 4, c_atom] — Side-Chain Module features of the SAME atoms.
        -> [..., L, 4, c_q].

        Pure function of its inputs — no in-place update of any cached tensor — so
        re-running it (activation-checkpoint recomputation firing the pre-hook a
        second time) reproduces the same value instead of compounding the residual.
        """
        s = self.sc_proj(q_sc_bb.to(q_bb.dtype))
        delta = self.mlp(self.ln(torch.cat([q_bb, s], dim=-1)))
        return q_bb + delta
