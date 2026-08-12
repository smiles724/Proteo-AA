"""Side-Chain Module S_phi: one-step, global-coordinate atom denoiser.

Follows the SideCraft Overleaf appendix ("Side-Chain Module"): a configurable
atom transformer with local (intra-residue) attention. The training config sizes
it from the backbone diffusion transformer by default; small CPU tests can still
instantiate the light setting directly. Atom features are initialised
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

from pxdesign_train.sidechain.frames import to_global, to_local
import torch.nn.functional as F

from pxdesign_train.heads import sinusoidal_time_embedding
from pxdesign_train.sidechain.coevolution import AResBSConcat, QAtomBSFusion
from pxdesign_train.sidechain.instantiate import (
    ATOM_VOCAB_SIZE,
    BACKBONE_ATOM_NAME_IDS,
    N_BB,
)


class _CrossAtomBlock(nn.Module):
    """Cross-residue attention BETWEEN ATOMS, over a residue-KNN neighbourhood.

    This is what the appendix actually specifies -- "cross-residue geometric
    attention between nearby ATOMS". `_CrossResBlock` below pools each residue's
    fourteen slots into one vector first and attends residue-to-residue, which
    throws away exactly the thing side-chain packing is about: a side-chain atom
    cannot tell WHICH neighbouring atom sits in its way, or in which direction,
    only that some residue is nearby. Mean-pooling is direction-blind.

    Neighbourhood selection is two-level, and deliberately so. A flat atom KNN
    needs an N_atom x N_atom distance matrix -- at the 640-token crop that is
    ~9k atoms and ~80M pairs, which dwarfs the attention it feeds. Selecting the
    M nearest RESIDUES by CA distance first costs L x L (~0.4M) and then attends
    over every atom of those residues, so the query sees real atom-atom geometry
    while the selection stays cheap. With M=16 and 14 slots a query attends to
    224 keys, versus L=640 keys for the residue-level block: comparable cost,
    atom-level resolution.

    Context tokens (receptor / motif / ligand) own no S_phi atom, so they enter as
    ONE virtual atom placed at their CA and carrying the residue feature the
    caller supplies. That preserves exactly what the pooled block could see --
    the side chain still knows a receptor residue is there -- without pretending
    we have its individual atoms in S_phi's slot space.
    """

    def __init__(self, c: int, n_heads: int, n_neighbors: int) -> None:
        super().__init__()
        if c % n_heads != 0:
            raise ValueError(f"c={c} must be divisible by n_heads={n_heads}")
        self.ln = nn.LayerNorm(c)
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.o = nn.Linear(c, c)
        # Per-head distance bias, softplus-gated to stay a *penalty*: the sign is
        # fixed so "closer attends more" cannot be learned away into its opposite.
        self.dist_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.c = c
        self.n_heads = n_heads
        self.head_dim = c // n_heads
        self.n_neighbors = int(n_neighbors)

    @staticmethod
    def residue_neighbours(ca, res_mask, n_neighbors):
        """Indices of the M nearest residues per residue, by CA distance.

        Invalid residues are pushed to +inf so they are only ever selected as
        padding once the real neighbours run out; the caller masks them anyway.
        Each residue's own index is included (it is its own nearest), so a side
        chain still sees its residue's other atoms here as well as in the
        intra-residue block -- harmless, and it keeps the gather uniform.
        """
        d = torch.cdist(ca, ca)                                    # [B, L, L]
        big = torch.finfo(d.dtype).max
        d = d.masked_fill(~res_mask[:, None, :], big)
        m = min(int(n_neighbors), d.shape[-1])
        return d.topk(m, dim=-1, largest=False).indices            # [B, L, M]

    def forward(self, atom_feats, atom_coords, atom_mask, ca, res_mask, nbr_idx=None):
        """atom_feats [B,L,A,c]; atom_coords [B,L,A,3]; atom_mask [B,L,A] bool;
        ca [B,L,3]; res_mask [B,L] bool. Returns atom_feats + update."""
        B, L, A, c = atom_feats.shape
        H, D = self.n_heads, self.head_dim
        if nbr_idx is None:
            nbr_idx = self.residue_neighbours(ca, res_mask, self.n_neighbors)
        M = nbr_idx.shape[-1]

        h = self.ln(atom_feats)
        q = self.q(h).reshape(B, L, A, H, D)

        def gather_residues(x):
            """[B,L,...] -> [B,L,M,...] indexed by nbr_idx."""
            trail = x.shape[2:]
            idx = nbr_idx.reshape(B, L * M)
            for _ in trail:
                idx = idx.unsqueeze(-1)
            idx = idx.expand(B, L * M, *trail)
            return x.gather(1, idx).reshape(B, L, M, *trail)

        k = gather_residues(self.k(h)).reshape(B, L, M * A, H, D)
        v = gather_residues(self.v(h)).reshape(B, L, M * A, H, D)
        k_xyz = gather_residues(atom_coords).reshape(B, L, M * A, 3)
        k_ok = gather_residues(atom_mask).reshape(B, L, M * A)

        # [B, L, A, M*A, H]
        scores = torch.einsum("blahd,blkhd->blakh", q, k) / math.sqrt(D)
        d_atom = torch.cdist(atom_coords.reshape(B * L, A, 3),
                             k_xyz.reshape(B * L, M * A, 3)).reshape(B, L, A, M * A)
        scores = scores - (
            F.softplus(self.dist_scale).view(1, 1, 1, 1, H) * d_atom.unsqueeze(-1)
        )
        scores = scores.masked_fill(~k_ok[:, :, None, :, None], float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-2))   # rows with no key
        out = torch.einsum("blakh,blkhd->blahd", attn, v).reshape(B, L, A, c)
        # Zero AFTER the output projection, not before: `self.o` carries a bias, so
        # projecting a zeroed row still yields that bias and a padding slot would
        # drift away from its residual base every block.
        upd = self.o(out) * atom_mask[..., None].to(out.dtype)
        return atom_feats + upd


class _AtomBlock(nn.Module):
    """Pre-norm masked self-attention + FFN over the atoms of one residue."""

    def __init__(self, c_atom: int, n_heads: int, ff_mult: int = 2) -> None:
        super().__init__()
        hidden = int(ff_mult) * c_atom
        self.ln1 = nn.LayerNorm(c_atom)
        self.attn = nn.MultiheadAttention(c_atom, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(c_atom)
        self.ff = nn.Sequential(
            nn.Linear(c_atom, hidden), nn.ReLU(), nn.Linear(hidden, c_atom)
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
        n_cross_blocks: int = 1,
        ff_mult: int = 2,
        trunk_grad_scale: float = 1.0,
        a_bs_concat: bool = False,
        q_bs: bool = False,
        c_q: int = 128,
        cross_neighbors: int = 16,
        template_residual: bool = False,
        centre_coord_input: bool = False,
    ) -> None:
        super().__init__()
        if c_atom % n_heads != 0:
            raise ValueError(f"c_atom={c_atom} must be divisible by n_heads={n_heads}")
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if n_cross_blocks < 1:
            raise ValueError("n_cross_blocks must be >= 1")
        self.c_atom = c_atom
        self.c_time = c_time
        self.trunk_grad_scale = float(trunk_grad_scale)
        self.template_residual = bool(template_residual)
        # ONE coordinate convention: every coordinate that enters this module is in
        # the GLOBAL frame. `centre_coord_input` only decides whether the per-atom
        # embedding sees them CA-centred; it never rotates anything, so no per-residue
        # frame can leak into a tensor that is later compared across residues.
        #
        # Why centring at all: `w_xyz` is a single Linear(3, c_atom) over the whole
        # atom axis, and a raw global coordinate carries the residue's absolute
        # position -- tens to hundreds of Angstrom -- on top of ~4 A of side-chain
        # geometry. Without centring the layer mostly encodes "where in the assembly
        # is this residue", which is not what the side-chain shape depends on.
        # Subtracting CA is translation-free and costs nothing; it is the part of the
        # old residue-local input that was actually doing the work.
        self.centre_coord_input = bool(centre_coord_input)

        self.atom_embed = nn.Embedding(ATOM_VOCAB_SIZE, c_atom, padding_idx=0)
        self.w_res = nn.Linear(c_res, c_atom)
        self.w_aa = nn.Linear(n_type, c_atom)
        self.w_t = nn.Sequential(nn.Linear(c_time, c_atom), nn.ReLU(), nn.Linear(c_atom, c_atom))
        self.w_xyz = nn.Linear(3, c_atom)

        self.blocks = nn.ModuleList([
            _AtomBlock(c_atom, n_heads, ff_mult=ff_mult) for _ in range(n_blocks)
        ])
        self.cross_neighbors = int(cross_neighbors)
        self.cross_res_blocks = nn.ModuleList([
            _CrossAtomBlock(c_atom, n_heads, cross_neighbors)
            for _ in range(n_cross_blocks)
        ])
        # Backwards-compatible debug/test handle for the first cross-residue block.
        self.cross_res = self.cross_res_blocks[0]
        self.a_bs_concat = bool(a_bs_concat)
        self.a_bs_concat_fusion = AResBSConcat(c_atom) if self.a_bs_concat else None
        self.q_bs = bool(q_bs)
        self.q_bs_fusion = QAtomBSFusion(c_atom, c_q) if self.q_bs else None
        self.out_ln = nn.LayerNorm(c_atom)
        self.out = nn.Linear(c_atom, 3)
        if self.template_residual:
            # Start from the high-quality template/noisy input and initially make
            # no learned correction. This also keeps the first optimizer update
            # well behaved when S_phi itself is randomly initialized.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

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
        noisy_coords: torch.Tensor,   # [B, L, A, 3] side-chain atoms, GLOBAL frame
        t: torch.Tensor,              # [B] or scalar diffusion time
        ca_coords: Optional[torch.Tensor] = None,  # [B, L, 3] residue CA, GLOBAL frame
        frame_R: Optional[torch.Tensor] = None,    # [B, L, 3, 3] local->global rotation
        frame_t: Optional[torch.Tensor] = None,    # [B, L, 3] local->global translation
        bb_coords: Optional[torch.Tensor] = None,  # [B, L, 4, 3] N,CA,C,O, GLOBAL frame
        res_mask: Optional[torch.Tensor] = None,   # [B, L] bool — residue exists
        ctx_mask: Optional[torch.Tensor] = None,   # [B, L] bool — context (receptor/motif) token
        bb_q: Optional[torch.Tensor] = None,       # [B, L, 4, c_q] backbone q from Backbone Module
    ):
        """One-step side-chain denoise.

        INTERNAL 14-ATOM AXIS. When ``bb_coords`` is given, S_phi builds the ATOM14
        layout internally — slots 0..3 are the residue's backbone atoms (N, CA, C, O)
        and slots 4..13 are its ``MAX_SC=10`` side-chain slots — and attends over all
        14. Backbone slots are pure CONTEXT: their coordinates are KNOWN (they come
        from the Backbone Module prediction), they are never denoised, never
        template-initialised and never supervised. The side chain can nevertheless
        *move* them, because they sit in the same attention stream — which is the
        whole point: the updated backbone-slot features are handed back so the
        Backbone Module can be given an atom-level (q-level) side-chain signal.

        The EXTERNAL contract is unchanged: ``atom_name_ids`` / ``atom_mask`` /
        ``noisy_coords`` stay 10-slot, and the coordinate output stays [B, L, 10, 3].

        Returns:
            ``(x0_global, atom_feats)`` when ``bb_coords is None`` (bit-identical to
            the pre-14-slot module), otherwise ``(x0_global, atom_feats, bb_feats)``
            with ``bb_feats`` [B, L, 4, c_atom]. ``atom_feats`` is ALWAYS the 10
            side-chain slots only, so HResFeedback / ATokenFusion pooling against
            ``atom_mask`` is unaffected.
        """
        B, L, A_sc = atom_name_ids.shape
        h_res = self._scale_grad(h_res)

        # ONE coordinate tensor, one frame. `coords` is global for every slot -- the
        # four backbone context atoms and the ten side-chain atoms alike -- because
        # `w_xyz` is a single Linear over that axis and the cross-residue distance
        # bias is compared against `ca_coords`, which is global. Mixing frames on
        # this axis (backbone rows residue-local, side-chain rows global) makes the
        # embedding read two coordinate systems out of one weight matrix and makes
        # every cross-residue distance meaningless; both were live bugs.
        if bb_coords is None:
            n_bb = 0
            ids, mask, coords = atom_name_ids, atom_mask, noisy_coords
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
                [bb_coords.to(noisy_coords.dtype), noisy_coords], dim=2  # [B, L, 14, 3]
            )
        A = ids.shape[2]
        # The embedding may see the SAME coordinates recentred on the residue's CA.
        # A pure translation, applied per residue, so it is only ever used here --
        # never for anything compared across residues.
        if self.centre_coord_input and ca_coords is not None:
            embed_coords = coords - ca_coords[:, :, None, :].to(coords.dtype)
        else:
            embed_coords = coords

        te = sinusoidal_time_embedding(torch.as_tensor(t, device=h_res.device).float(), self.c_time)
        te = self.w_t(te)                                  # [B, c_atom]
        if te.dim() == 1:
            te = te[None]
        h_proj = self.w_res(h_res)                         # [B, L, c_atom]
        res_feat = h_proj[:, :, None, :]                   # [B, L, 1, c_atom]
        type_feat = self.w_aa(torch.softmax(restype_logits, dim=-1))[:, :, None, :]
        atom_feat = self.atom_embed(ids)                   # [B, L, A, c_atom]
        xyz_feat = self.w_xyz(embed_coords)                # [B, L, A, c_atom]
        u = atom_feat + res_feat + type_feat + xyz_feat + te[:, None, None, :]

        if self.q_bs and self.q_bs_fusion is not None and bb_q is not None and n_bb > 0:
            # point 3: seed the backbone slots with the Backbone Module's q for those atoms.
            bb_slot = u[:, :, :n_bb, :]
            fused = self.q_bs_fusion(bb_slot, bb_q.to(u.dtype))
            u = torch.cat([fused, u[:, :, n_bb:, :]], dim=2)

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
            if self.a_bs_concat and self.a_bs_concat_fusion is not None:
                # point 2: fuse the backbone a_token (h_proj) into the side-chain's all-atom
                # residue representation (pooled), symmetric to a-direct on the S→B side.
                fused_pooled = self.a_bs_concat_fusion(pooled, h_proj.to(pooled.dtype))
                # The atoms carry themselves through the cross-residue stage and
                # `pooled` only seeds context slots, so the B->S contribution has to
                # reach the atoms directly. Fusing into `pooled` alone would make
                # this channel a silent no-op on any structure with no context
                # tokens -- which is every monomer, i.e. all of Stage II.
                atom_feats = atom_feats + (fused_pooled - pooled).unsqueeze(2)
                pooled = fused_pooled

            # Every slot attends to the individual atoms of the M
            # nearest residues, so a side chain can tell WHICH atom is in its
            # way and in which direction. A context residue owns no S_phi slot,
            # so it joins as ONE virtual atom at its CA carrying `pooled` --
            # the same feature the residue-level block would have shown it,
            # placed somewhere geometrically meaningful.
            virt_feat = pooled.unsqueeze(2)                        # [B,L,1,c]
            virt_xyz = ca_coords.unsqueeze(2).to(coords.dtype)     # [B,L,1,3]
            is_virtual = keys_mask & ~mask.bool().any(dim=-1)      # context only
            feats_ext = torch.cat([atom_feats, virt_feat.to(atom_feats.dtype)], dim=2)
            # GLOBAL frame here, not `coords` -- `virt_xyz` is `ca_coords`, which is
            # global, and the cross-residue distances have to be commensurate with it.
            xyz_ext = torch.cat([coords.to(atom_feats.dtype), virt_xyz.to(atom_feats.dtype)], dim=2)
            mask_ext = torch.cat([mask.bool(), is_virtual.unsqueeze(-1)], dim=2)
            nbr_idx = self.cross_res_blocks[0].residue_neighbours(
                ca_coords, keys_mask, self.cross_neighbors
            )
            for cross_block in self.cross_res_blocks:
                feats_ext = cross_block(
                    feats_ext, xyz_ext, mask_ext, ca_coords, keys_mask,
                    nbr_idx=nbr_idx,
                )
            atom_feats = feats_ext[:, :, :A, :]                    # drop the virtual slot

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
        if self.template_residual:
            if frame_R is None or frame_t is None:
                raise ValueError(
                    "template_residual requires a frame-aware head so the residual "
                    "base and learned correction are both residue-local"
                )
            # `y0` is a residue-LOCAL offset (the frame maps it out below), so the
            # residual base has to be local too. The input arrives global under the
            # single-frame contract, so map it in here rather than asking the caller
            # for a second tensor -- that second tensor is exactly what used to drift
            # out of sync with this one.
            y0 = to_local(noisy_coords.float(), frame_R.float(), frame_t.float()).to(y0.dtype) + y0
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
            # caller opted into the 14-slot axis by passing bb_coords.
            return x0_global, atom_feats
        return x0_global, atom_feats, bb_feats
