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

from pxdesign_train.sidechain.frames import to_global
import torch.nn.functional as F

from pxdesign_train.heads import sinusoidal_time_embedding
from pxdesign_train.sidechain.instantiate import (
    ATOM_VOCAB_SIZE,
    BACKBONE_ATOM_NAME_IDS,
    N_BB,
)


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
        frame_R: Optional[torch.Tensor] = None,    # [B, L, 3, 3] local->global rotation
        frame_t: Optional[torch.Tensor] = None,    # [B, L, 3] local->global translation
        bb_local: Optional[torch.Tensor] = None,   # [B, L, 4, 3] N,CA,C,O in the LOCAL frame
        res_mask: Optional[torch.Tensor] = None,   # [B, L] bool — residue exists
        ctx_mask: Optional[torch.Tensor] = None,   # [B, L] bool — context (receptor/motif) token
    ):
        """One-step side-chain denoise.

        INTERNAL 14-ATOM AXIS. When ``bb_local`` is given, S_phi builds the ATOM14
        layout internally — slots 0..3 are the residue's backbone atoms (N, CA, C, O)
        and slots 4..13 are its ``MAX_SC=10`` side-chain slots — and attends over all
        14. Backbone slots are pure CONTEXT: their coordinates are KNOWN (they come
        from the Backbone Module prediction), they are never denoised, never
        template-initialised and never supervised. The side chain can nevertheless
        *move* them, because they sit in the same attention stream — which is the
        whole point: the updated backbone-slot features are handed back so the
        Backbone Module can be given an atom-level (q-level) side-chain signal.

        The EXTERNAL contract is unchanged: ``atom_name_ids`` / ``atom_mask`` /
        ``noisy_local`` stay 10-slot, and the coordinate output stays [B, L, 10, 3].

        Returns:
            ``(x0_global, atom_feats)`` when ``bb_local is None`` (bit-identical to
            the pre-14-slot module), otherwise ``(x0_global, atom_feats, bb_feats)``
            with ``bb_feats`` [B, L, 4, c_atom]. ``atom_feats`` is ALWAYS the 10
            side-chain slots only, so HResFeedback / ATokenFusion pooling against
            ``atom_mask`` is unaffected.
        """
        B, L, A_sc = atom_name_ids.shape
        h_res = self._scale_grad(h_res)

        # --- build the atom axis: 4 backbone context slots + the 10 side-chain slots ---
        if bb_local is None:
            n_bb = 0
            ids, mask, coords = atom_name_ids, atom_mask, noisy_local
        else:
            n_bb = N_BB
            bb_ids = BACKBONE_ATOM_NAME_IDS.to(atom_name_ids.device).view(1, 1, n_bb)
            bb_ids = bb_ids.expand(B, L, n_bb)
            if res_mask is None:
                # No explicit residue mask: every row of the batch is a real residue.
                bb_mask = torch.ones(B, L, n_bb, dtype=atom_mask.dtype, device=atom_mask.device)
            else:
                bb_mask = res_mask[..., None].expand(B, L, n_bb).to(atom_mask.dtype)
            ids = torch.cat([bb_ids, atom_name_ids], dim=2)              # [B, L, 14]
            mask = torch.cat([bb_mask, atom_mask], dim=2)                # [B, L, 14]
            coords = torch.cat(
                [bb_local.to(noisy_local.dtype), noisy_local], dim=2     # [B, L, 14, 3]
            )
        A = ids.shape[2]

        te = sinusoidal_time_embedding(torch.as_tensor(t, device=h_res.device).float(), self.c_time)
        te = self.w_t(te)                                  # [B, c_atom]
        if te.dim() == 1:
            te = te[None]
        h_proj = self.w_res(h_res)                         # [B, L, c_atom]
        res_feat = h_proj[:, :, None, :]                   # [B, L, 1, c_atom]
        type_feat = self.w_aa(torch.softmax(restype_logits, dim=-1))[:, :, None, :]
        atom_feat = self.atom_embed(ids)                   # [B, L, A, c_atom]
        xyz_feat = self.w_xyz(coords)                      # [B, L, A, c_atom]
        u = atom_feat + res_feat + type_feat + xyz_feat + te[:, None, None, :]

        # Intra-residue attention: flatten (B*L) as the batch, A as sequence.
        # With backbone slots present GLY (0 side-chain atoms) still has 4 valid
        # keys, so its row is not all-masked; the fully_pad guard below covers the
        # 10-slot path and any residue whose res_mask is False.
        x = u.reshape(B * L, A, self.c_atom)
        kpm = ~mask.reshape(B * L, A).bool()               # True = pad/ignore
        fully_pad = kpm.all(dim=1)                         # residues with no valid atom
        kpm = kpm & ~fully_pad[:, None]                    # avoid all-masked NaN rows
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        atom_feats = x.reshape(B, L, A, self.c_atom)

        # Cross-residue geometric attention (side-chain <-> neighbour context).
        #
        # CONTEXT KEYS. Appendix ("Side-Chain Module"): "inter-residue and context
        # attention capture side-chain--side-chain, side-chain--backbone, and
        # side-chain--RECEPTOR interactions", and the global-frame state exists
        # precisely so side chains "directly attend to neighboring residues,
        # receptor atoms, fixed motifs, ligands, and other spatial context".
        # Keying only on tokens that own S_phi atoms makes every receptor/motif/
        # ligand token an all-masked key, so the side chain could never see the
        # thing it is packing against. `ctx_mask` marks those tokens; they carry no
        # S_phi atoms (their pooled feature is 0), so we seed them from h_res — the
        # only representation that exists for EVERY token. They are keys only: their
        # own query output is discarded downstream (they own no side-chain slot).
        if ca_coords is not None:
            am = mask.to(atom_feats.dtype)[..., None]                 # [B,L,A,1]
            pooled = (atom_feats * am).sum(2) / (am.sum(2) + 1e-6)    # [B,L,c]
            has_sc = mask.bool().any(dim=-1)                          # [B,L] bool
            keys_mask = has_sc
            if ctx_mask is not None:
                ctx_mask = ctx_mask.to(has_sc.device).bool()
                pooled = torch.where(has_sc[..., None], pooled, h_proj.to(pooled.dtype))
                keys_mask = has_sc | ctx_mask
            res_ctx = self.cross_res(pooled, ca_coords, keys_mask)    # [B,L,c]
            atom_feats = atom_feats + res_ctx[:, :, None, :]          # broadcast back

        # Split the 14 slots back apart. Coordinates are decoded for the 10
        # SIDE-CHAIN slots only — backbone slots never produce coordinates and so
        # can never enter the coordinate loss.
        bb_feats = atom_feats[:, :, :n_bb, :] if n_bb else None    # [B,L,4,c_atom]
        atom_feats = atom_feats[:, :, n_bb:, :]                    # [B,L,10,c_atom]

        # S_phi emits GLOBAL coordinates (Overleaf par.204). HOW it gets there matters:
        #
        #   frame-aware head (frame_R/frame_t given): the head predicts RESIDUE-LOCAL
        #     offsets and the caller-supplied rigid frame maps them to global,
        #     x0_global = F_hat . out(atom_feats). The output space is still global, but
        #     the regression target the MLP sees is rotation-INVARIANT, so it does not
        #     have to learn to apply a rotation it inferred from its own input -- a
        #     bilinear operation a plain MLP approximates very poorly. Measured: the
        #     CA-anchored variant below plateaus ~7x worse on a single-structure
        #     memorization smoke (3.8 vs 0.51) even when the initialization carries the
        #     orientation, because the head, not the init, is the bottleneck.
        #
        #   CA-anchored head (frame_R/frame_t None): legacy behaviour, kept for A/B.
        y0 = self.out(self.out_ln(atom_feats))             # [B, L, A, 3]
        if frame_R is not None and frame_t is not None:
            x0_global = to_global(y0, frame_R, frame_t)
        else:
            x0_global = y0
            if ca_coords is not None:
                x0_global = x0_global + ca_coords[:, :, None, :].to(x0_global.dtype)
        x0_global = x0_global * atom_mask[..., None].to(x0_global.dtype)
        if bb_feats is None:
            # Legacy arity: existing callers (model.py, feedback, a_direct tests)
            # unpack exactly two values. The 3rd element appears only when the
            # caller opted into the 14-slot axis by passing bb_local.
            return x0_global, atom_feats
        return x0_global, atom_feats, bb_feats
