"""Atom -> residue feedback: pool side-chain atom features into h_res.

Bidirectional h_res, write direction (SideCraft spec §2.1 / Overleaf
"Atom-to-Residue Feedback"): after side-chain decoding, side-chain atom features
are pooled per residue and used to produce an updated persistent residue
representation h_res' that the Backbone Module consumes on the next step. This is
what turns S_phi from a post-hoc packer into an active co-evolution component.

`detach=True` cuts the side-chain -> next-backbone gradient specifically on the
pooled feature path (independent of S_phi's own h_res read-scale).
"""
import torch
import torch.nn as nn


class HResFeedback(nn.Module):
    def __init__(self, c_atom: int, c_res: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.pool_proj = nn.Linear(c_atom, c_res)
        self.update = nn.Sequential(
            nn.LayerNorm(2 * c_res),
            nn.Linear(2 * c_res, c_res),
            nn.ReLU(),
            nn.Linear(c_res, c_res),
        )

    def forward(
        self,
        atom_feats: torch.Tensor,   # [B, L, A, c_atom]
        atom_mask: torch.Tensor,    # [B, L, A] bool
        h_res: torch.Tensor,        # [B, L, c_res]
        detach: bool = False,
    ) -> torch.Tensor:
        if detach:
            atom_feats = atom_feats.detach()
        m = atom_mask[..., None].to(atom_feats.dtype)          # [B, L, A, 1]
        pooled = (atom_feats * m).sum(dim=2) / (m.sum(dim=2) + self.eps)  # [B, L, c_atom]
        g = self.pool_proj(pooled)                              # [B, L, c_res]
        delta = self.update(torch.cat([h_res, g], dim=-1))      # [B, L, c_res]
        return h_res + delta                                    # persistent h_res'
