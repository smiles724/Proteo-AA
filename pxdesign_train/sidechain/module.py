"""Side-Chain Module S_phi: one-step, local-frame atom denoiser.

Follows the SideCraft Overleaf appendix ("Side-Chain Module"): a light atom
transformer with local (intra-residue) attention. Atom features are initialised
as

    u_ij = Embed_atom(name_ij) + W_res h_res_i + W_aa softmax(p(a_i))
           + W_t e_t + W_xyz y_noisy_ij

and one-step-decoded to local-frame side-chain coordinates y0. There is no
side-chain reverse-diffusion loop here (decode-first, APM-borrowed one-step).

The `trunk_grad_scale` knob controls how much of the side-chain loss gradient
flows back into h_res (and thus the Backbone Module) — the same mechanism the
residue-type head uses. scale=1.0 = full co-evolution coupling; 0.0 = read-only.

Reusing Protenix's AF3 `AtomAttentionDecoder` is a possible future optimisation;
the Overleaf explicitly lists "an atom transformer with geometric bias" as a
valid S_phi backbone, which is what we implement (and can unit-test on CPU).
"""
from typing import Optional

import torch
import torch.nn as nn

from pxdesign_train.heads import sinusoidal_time_embedding
from pxdesign_train.sidechain.instantiate import ATOM_VOCAB_SIZE


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
        noisy_local: torch.Tensor,    # [B, L, A, 3]
        t: torch.Tensor,              # [B] or scalar diffusion time
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

        y0_local = self.out(self.out_ln(atom_feats))       # [B, L, A, 3]
        y0_local = y0_local * atom_mask[..., None].to(y0_local.dtype)
        return y0_local, atom_feats
