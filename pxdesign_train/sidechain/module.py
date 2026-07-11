"""Side-Chain Module S_phi: one-step, global-coordinate atom denoiser.

Follows the SideCraft Overleaf appendix ("Side-Chain Module"): a light atom
transformer with local (intra-residue) attention. Atom features are initialised
as

    u_ij = Embed_atom(name_ij) + W_res h_res_i + W_aa softmax(p(a_i))
           + W_t e_t + W_xyz x_noisy_ij

and one-step-decoded to global-frame side-chain coordinates x0. There is no
side-chain reverse-diffusion loop here (decode-first, APM-borrowed one-step).

The `trunk_grad_scale` knob controls how much of the side-chain loss gradient
flows back into h_res (and thus the Backbone Module) — the same mechanism the
residue-type head uses. scale=1.0 = full co-evolution coupling; 0.0 = read-only.

Reusing Protenix's AF3 `AtomAttentionDecoder` is a possible future optimisation;
the Overleaf explicitly lists "an atom transformer with geometric bias" as a
valid S_phi backbone, which is what we implement (and can unit-test on CPU).
"""
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pxdesign_train.heads import sinusoidal_time_embedding
from pxdesign_train.sidechain.instantiate import ATOM_VOCAB_SIZE


class _CrossResBlock(nn.Module):
    """Cross-residue geometric attention: each residue's pooled side-chain
    feature attends to nearby residues, with attention biased by CA-CA distance
    (closer residues attend more). Captures the Overleaf's "cross-residue
    geometric attention between nearby atoms" — side-chain<->side-chain and
    side-chain<->backbone interactions, at residue-pooled granularity."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.o = nn.Linear(c, c)
        self.dist_scale = nn.Parameter(torch.tensor(0.1))
        self.c = c

    def forward(self, x, ca, res_mask):
        # x [B,L,c], ca [B,L,3], res_mask [B,L] bool
        h = self.ln(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.c)     # [B,L,L]
        d = torch.cdist(ca, ca)                                    # [B,L,L]
        scores = scores - F.softplus(self.dist_scale) * d          # closer -> higher
        scores = scores.masked_fill(~res_mask[:, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)                             # rows with no valid key
        out = attn @ v                                            # [B,L,c]
        return x + self.o(out)


class _AtomBlock(nn.Module):
    """Pre-norm masked self-attention + FFN over the atoms of one residue."""

    def __init__(self, c_atom: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(c_atom)
        self.attn = nn.MultiheadAttention(c_atom, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(c_atom)
        self.ff = nn.Sequential(
            nn.Linear(c_atom, 2 * c_atom), nn.ReLU(), nn.Linear(2 * c_atom, c_atom)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class SideChainModule(nn.Module):
    def __init__(
        self,
        c_res: int,
        c_atom: int = 128,
        n_type: int = 20,
        c_time: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        trunk_grad_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.c_atom = c_atom
        self.c_time = c_time
        self.trunk_grad_scale = float(trunk_grad_scale)

        self.atom_embed = nn.Embedding(ATOM_VOCAB_SIZE, c_atom, padding_idx=0)
        self.w_res = nn.Linear(c_res, c_atom)
        self.w_aa = nn.Linear(n_type, c_atom)
        self.w_t = nn.Sequential(nn.Linear(c_time, c_atom), nn.ReLU(), nn.Linear(c_atom, c_atom))
        self.w_xyz = nn.Linear(3, c_atom)

        self.blocks = nn.ModuleList([_AtomBlock(c_atom, n_heads) for _ in range(n_blocks)])
        self.cross_res = _CrossResBlock(c_atom)
        self.out_ln = nn.LayerNorm(c_atom)
        self.out = nn.Linear(c_atom, 3)

    def _scale_grad(self, h_res: torch.Tensor) -> torch.Tensor:
        s = self.trunk_grad_scale
        if s == 1.0:
            return h_res
        return s * h_res + (1.0 - s) * h_res.detach()

    def forward(
        self,
        h_res: torch.Tensor,          # [B, L, c_res]
        restype_logits: torch.Tensor, # [B, L, n_type]
        atom_name_ids: torch.Tensor,  # [B, L, A] long
        atom_mask: torch.Tensor,      # [B, L, A] bool
        noisy_local: torch.Tensor,    # [B, L, A, 3] global coords in current path
        t: torch.Tensor,              # [B] or scalar diffusion time
        ca_coords: Optional[torch.Tensor] = None,  # [B, L, 3] residue CA (frame origin)
    ):
        B, L, A = atom_name_ids.shape
        h_res = self._scale_grad(h_res)

        te = sinusoidal_time_embedding(torch.as_tensor(t, device=h_res.device).float(), self.c_time)
        te = self.w_t(te)                                  # [B, c_atom]
        if te.dim() == 1:
            te = te[None]
        res_feat = self.w_res(h_res)[:, :, None, :]        # [B, L, 1, c_atom]
        type_feat = self.w_aa(torch.softmax(restype_logits, dim=-1))[:, :, None, :]
        atom_feat = self.atom_embed(atom_name_ids)         # [B, L, A, c_atom]
        xyz_feat = self.w_xyz(noisy_local)                 # [B, L, A, c_atom]
        u = atom_feat + res_feat + type_feat + xyz_feat + te[:, None, None, :]

        # Intra-residue attention: flatten (B*L) as the batch, A as sequence.
        x = u.reshape(B * L, A, self.c_atom)
        kpm = ~atom_mask.reshape(B * L, A)                 # True = pad/ignore
        fully_pad = kpm.all(dim=1)                         # residues with no side chain
        kpm = kpm & ~fully_pad[:, None]                    # avoid all-masked NaN rows
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        atom_feats = x.reshape(B, L, A, self.c_atom)

        # Cross-residue geometric attention (side-chain<->neighbour context).
        if ca_coords is not None:
            am = atom_mask.to(atom_feats.dtype)[..., None]         # [B,L,A,1]
            res_feat = (atom_feats * am).sum(2) / (am.sum(2) + 1e-6)  # [B,L,c]
            res_mask = atom_mask.any(dim=-1)                        # [B,L] bool
            res_ctx = self.cross_res(res_feat, ca_coords, res_mask)  # [B,L,c]
            atom_feats = atom_feats + res_ctx[:, :, None, :]        # broadcast back

        # S_phi emits GLOBAL coordinates. The head predicts atom offsets anchored
        # at the current residue CA/global frame origin; if no CA is provided, the
        # offsets themselves are interpreted as global coordinates for backward
        # compatibility with small unit tests.
        x0_global = self.out(self.out_ln(atom_feats))      # [B, L, A, 3]
        if ca_coords is not None:
            x0_global = x0_global + ca_coords[:, :, None, :].to(x0_global.dtype)
        x0_global = x0_global * atom_mask[..., None].to(x0_global.dtype)
        return x0_global, atom_feats
