"""Stage III: side-chain-aware residue compatibility (L_SC-AA).

Overleaf (ICLR SideCraft) §Stage III, second objective: give residue-identity
prediction a *direct* side-chain-aware signal. For selected positions we build a
candidate AA set C_i = {GT} u TopK(p(a_i)); for each candidate we instantiate its
side chain, run S_phi, and score the geometry with differentiable compatibility
energies E_i(a) = clash + rot + pack + contact. A margin-ranking loss then pushes
the AA logits toward candidates whose side chains place with lower energy:

    L_SC-AA = sum_i sum_{a- in N_i} max(0, m - l_i(a+) + l_i(a-))

where a+ is the lowest-energy candidate (or GT) and a- are worse candidates.
No external physics engine (Rosetta) is needed — energies come from predicted
coordinates via the physical terms.

This module implements the ranking loss and a compatibility-energy helper. The
per-candidate S_phi orchestration lives with the model (it re-runs S_phi under
each candidate mask); this file is the self-contained, testable core.
"""
from typing import Optional

import torch

from pxdesign_train.sidechain.physical import clash_loss, contact_loss


def candidate_energy(
    sc_global: torch.Tensor,           # [K, A, 3] predicted global side chain per candidate
    valid_mask: torch.Tensor,          # [K, A] bool
    backbone_coords: Optional[torch.Tensor] = None,  # [M, 3] this residue's + neighbours' bb
) -> torch.Tensor:
    """Differentiable compatibility energy per candidate (lower = better).
    clash (+ contact when backbone given). Returns [K]."""
    K = sc_global.shape[0]
    energies = []
    for k in range(K):
        c = clash_loss(sc_global[k : k + 1], valid_mask=valid_mask[k : k + 1])
        if backbone_coords is not None:
            c = c + contact_loss(sc_global[k : k + 1], backbone_coords[None],
                                 valid_mask[k : k + 1])
        energies.append(c)
    return torch.stack(energies)  # [K]


def sc_aa_ranking_loss(
    aa_logits: torch.Tensor,          # [L, C] residue-type logits
    candidate_types: torch.Tensor,    # [L, K] long — candidate AA indices per position
    candidate_energies: torch.Tensor, # [L, K] — compatibility energy (lower better)
    position_mask: Optional[torch.Tensor] = None,  # [L] bool — which positions are scored
    margin: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Margin-ranking loss pushing AA logits toward the lowest-energy candidate."""
    L, K = candidate_types.shape
    logits_c = torch.gather(aa_logits, -1, candidate_types)          # [L, K]
    best = candidate_energies.argmin(dim=-1, keepdim=True)           # [L, 1]
    l_plus = logits_c.gather(-1, best)                               # [L, 1]
    e_best = candidate_energies.gather(-1, best)                     # [L, 1]
    worse = (candidate_energies > e_best).float()                   # [L, K]
    per = torch.relu(margin - l_plus + logits_c) * worse            # [L, K]
    if position_mask is not None:
        per = per * position_mask.float()[:, None]
    return per.sum() / (worse.sum() + eps)
