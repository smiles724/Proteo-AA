"""TorsionPacker -- APM's Sidechain Module, on this repo's I/O contract.

    p_SC( chi_{1:4} | X_BB, res_type, [a_token], [PLM] )  ->  BuildSC  ->  X_SC

PROVENANCE. The architecture is a port of APM (https://github.com/bytedance/apm,
Apache-2.0), `apm/models/side_chain_model.py` + `node_feature_net.py` +
`edge_feature_net.py`, with the sizes from `apm/configs/model.yaml`'s
`packing_model` block. Copyright (2025) Bytedance Ltd. and/or its affiliates.
The IPA and structure-module blocks live in `sidechain/ipa.py`; the feature
helpers below (`get_index_embedding`, `get_time_embedding`, `calc_distogram`,
`AngularEncoding`, `rotmat_to_rotvec`) are ported from `apm/models/utils.py` and
`apm/data/so3_utils.py`.

    trunk = 6 x [ IPA -> LayerNorm -> TransformerEncoder(4 layers, 4 heads)
                  -> post Linear -> StructureModuleTransition
                  -> EdgeTransition (all but the last block) ]
    head  = AngleResnet (AF2 Alg. 20 lines 11-14, 4 blocks)

WHERE THIS DELIBERATELY DIFFERS FROM APM, and why. Each of these is a config
switch, and each default is stated here rather than buried:

1. `seq_cond` (default "a_token"). APM's packer has NO a_token -- it is a
   standalone module whose sequence information arrives as a frozen ESM-2
   embedding. Ours exists to measure whether Stage 1's a_token carries anything,
   so the sequence channel is pluggable: "none" | "a_token" | "plm" | "both".
   The insertion point is APM's: both are projected to c_node and ADDED to the
   node embedding, exactly where `init_node_embed += plm_s` sits.
2. `embed_aatype` (default True; APM's packing config says False). With
   embed_aatype=False the "none" and "a_token" arms would have no residue-type
   channel at all, because APM leans on the PLM for it. Residue type is an
   explicit input of our Stage 2, so it stays.
3. Coordinates come back through `BuildSC` (this repo's ideal-geometry builder)
   rather than openfold's rigid groups. Same map, different table.
4. `random_torsion_input` (default True, matching APM). In APM's packing mode
   `torsions_t` is a UNIFORM RANDOM angle and `torsions_sc` is identically zero
   during training (`interpolant.py:589`); self-conditioning is filled in only
   inside the sampling loop. So both channels carry no information here, and the
   random one makes the packer's output stochastic. Set False to feed zeros and
   get a deterministic packer -- at the cost of no longer matching APM.
5. `embed_rotvecs` (default True, matching APM). The frame's global rotation
   vector is a node feature, so the node channel is NOT SE(3)-invariant even
   though IPA is. Set False for a strictly invariant model.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from pxdesign_train.sidechain.buildsc import build_sidechain_local
from pxdesign_train.sidechain.chi_constants import CHI_MASK, MAX_CHI
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.sidechain.instantiate import (
    ATOM_VOCAB_SIZE,
    BACKBONE_ATOM_NAME_IDS,
    N_BB,
    STD_AA_3,
)
from pxdesign_train.sidechain.ipa import (
    AngleResnet,
    EdgeTransition,
    InvariantPointAttention,
    Linear,
    StructureModuleTransition,
)

# APM scales translations to nanometres before the trunk (`rigids_ang_to_nm`).
# The IPA point term carries a hard-coded 9/2 variance constant that assumes
# that scale, so this is load-bearing, not cosmetic.
ANG_TO_NM_SCALE = 0.1


# --------------------------------------------------------------------------
# Feature helpers, ported from apm/models/utils.py and apm/data/so3_utils.py
# --------------------------------------------------------------------------

def get_index_embedding(indices, embed_size, max_len=2056):
    K = torch.arange(embed_size // 2, device=indices.device)
    denom = max_len ** (2 * K[None] / embed_size)
    pos = indices[..., None] / denom
    return torch.cat([torch.sin(pos * math.pi), torch.cos(pos * math.pi)], dim=-1)


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                                 device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1), mode="constant")
    return emb


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], dim=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    return ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)


class AngularEncoding(nn.Module):
    """APM's angular featurisation: [x, sin(f x), cos(f x)] over 6 frequencies."""

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer("freq_bands", torch.FloatTensor(
            [i + 1 for i in range(num_funcs)] + [1.0 / (i + 1) for i in range(num_funcs)]))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2 * 2 * self.num_funcs)

    def forward(self, x):
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)
        code = torch.cat(
            [x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)
        return code.reshape(shape)


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:
    return torch.broadcast_to(torch.eye(3, device=target.device, dtype=target.dtype),
                              target.shape)


def _skew_matrix_to_vector(skew: torch.Tensor) -> torch.Tensor:
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def _angle_from_rotmat(R: torch.Tensor):
    skew = R - R.transpose(-2, -1)
    angles_sin = torch.norm(_skew_matrix_to_vector(skew), dim=-1) / 2.0
    angles_cos = (torch.einsum("...ii", R) - 1.0) / 2.0
    return torch.atan2(angles_sin, angles_cos), angles_sin, angles_cos


def rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithmic map, with APM's handling of theta ~ 0 and theta ~ pi."""
    angles, angles_sin, _ = _angle_from_rotmat(R)
    vector = _skew_matrix_to_vector(R - R.transpose(-2, -1))

    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    mask_pi = torch.isclose(angles, torch.full_like(angles, math.pi),
                            atol=1e-2).to(angles.dtype)
    mask_else = (1 - mask_zero) * (1 - mask_pi)

    numerator = mask_zero / 2.0 + angles * mask_else
    denominator = ((1.0 - angles ** 2 / 6.0) * mask_zero
                   + 2.0 * angles_sin * mask_else
                   + mask_pi)
    vector = vector * (numerator / denominator)[..., None]

    id3 = _broadcast_identity(R)
    skew_outer = (id3 + R) / 2.0
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3
    vector_pi = torch.sqrt(torch.diagonal(torch.clamp(skew_outer, min=1e-8),
                                          dim1=-2, dim2=-1))
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    signs_line = torch.take_along_dim(
        skew_outer, dim=-2, indices=signs_line_idx[..., None, None]).squeeze(-2)
    vector_pi = vector_pi * angles[..., None] * torch.sign(signs_line)
    return vector + vector_pi * mask_pi[..., None]


def _cross_concat(feats_1d, num_batch, num_res):
    return torch.cat([
        torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
        torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
    ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])


# --------------------------------------------------------------------------


class PackingNodeFeatureNet(nn.Module):
    """APM's `PackingNodeFeatureNet`, name-for-name.

    The submodule name and the internal names (`aatype_embedding`,
    `torsion_embedding`, `linear`) are APM's, because that is what makes the
    released `sidechain_model.ckpt` loadable into this module. The feature
    ORDER is theirs too, and it is load-bearing for weight compatibility:
    pos_emb, diffuse_mask, aatype, rotvecs, time, torsions, torsions_sc.
    """

    def __init__(self, c_s, c_pos_emb, c_timestep_emb, embed_aatype, embed_rotvecs,
                 use_mlp, n_type=20):
        super().__init__()
        self.c_s = c_s
        self.c_pos_emb = c_pos_emb
        self.c_timestep_emb = c_timestep_emb
        self.embed_aatype = bool(embed_aatype)
        self.embed_rotvecs = bool(embed_rotvecs)
        embed_size = c_pos_emb + 1
        if self.embed_aatype:
            self.aatype_embedding = nn.Embedding(n_type + 1, c_s)
            embed_size += c_s
        self.torsion_embedding = AngularEncoding()
        embed_size += c_timestep_emb + self.torsion_embedding.get_out_dim(4) * 2
        if self.embed_rotvecs:
            embed_size += 3
        if use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, c_s), nn.ReLU(),
                nn.Linear(c_s, c_s), nn.ReLU(),
                nn.Linear(c_s, c_s), nn.LayerNorm(c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, c_s)

    def forward(self, *, tor_t, res_mask, diffuse_mask, pos, aatypes, rotvecs,
                torsions, torsions_sc):
        B, L = res_mask.shape
        pos_emb = get_index_embedding(pos, self.c_pos_emb) * res_mask[..., None]
        feats = [pos_emb, diffuse_mask[..., None]]
        if self.embed_aatype:
            feats.append(self.aatype_embedding(aatypes))
        if self.embed_rotvecs:
            feats.append(rotvecs)
        time_emb = get_time_embedding(tor_t[:, 0], self.c_timestep_emb,
                                      max_positions=2056)[:, None, :].repeat(1, L, 1)
        feats.append(time_emb * res_mask[..., None])
        feats.append(self.torsion_embedding(torsions))
        feats.append(self.torsion_embedding(torsions_sc))
        return self.linear(torch.cat(feats, dim=-1))


class FullEdgeFeatureNet(nn.Module):
    """APM's `FullEdgeFeatureNet`, name-for-name. See the note above.

    Feature order: cross-concat(node), relpos, CA distogram, cross-concat of the
    two torsion encodings, chain embedding, diffuse mask. The chain embedding
    sits BEFORE the diffuse mask -- swapping them silently shifts 128 input
    columns of `edge_embedder.0` and the released weights stop meaning anything.
    """

    def __init__(self, c_s, c_p, feat_dim, num_bins, embed_chain):
        super().__init__()
        self.c_p = c_p
        self.feat_dim = feat_dim
        self.num_bins = num_bins
        self.embed_chain = bool(embed_chain)
        self.linear_s_p = nn.Linear(c_s, feat_dim)
        self.linear_relpos = nn.Linear(feat_dim, feat_dim)
        self.torsion_embedding = AngularEncoding()
        total = (feat_dim * 3 + num_bins
                 + self.torsion_embedding.get_out_dim(4) * 2 * 2
                 + 2)                                  # embed_diffuse_mask=True
        if self.embed_chain:
            self.rel_chain_emb = nn.Embedding(2, c_p)
            total += c_p
        self.edge_embedder = nn.Sequential(
            nn.Linear(total, c_p), nn.ReLU(),
            nn.Linear(c_p, c_p), nn.ReLU(),
            nn.Linear(c_p, c_p), nn.LayerNorm(c_p),
        )

    def forward(self, s, trans, torsions, torsions_sc, edge_mask, diffuse_mask,
                residue_index, chain_index):
        B, L, _ = s.shape
        p_i = self.linear_s_p(s)
        feats = [_cross_concat(p_i, B, L)]
        d = residue_index[:, :, None] - residue_index[:, None, :]
        feats.append(self.linear_relpos(get_index_embedding(d, self.feat_dim)))
        feats.append(calc_distogram(trans, min_bin=1e-3, max_bin=20.0,
                                    num_bins=self.num_bins))
        feats.append(_cross_concat(self.torsion_embedding(torsions), B, L))
        feats.append(_cross_concat(self.torsion_embedding(torsions_sc), B, L))
        if self.embed_chain:
            same = (chain_index[:, :, None] == chain_index[:, None, :]).long()
            feats.append(self.rel_chain_emb(same))
        feats.append(_cross_concat(diffuse_mask[..., None], B, L))
        return self.edge_embedder(torch.cat(feats, dim=-1)) * edge_mask[..., None]


class TorsionPacker(nn.Module):
    """APM's SideChainModel, signature-compatible with `SideChainModule`.

    The forward signature matches `SideChainModule.forward` so that
    `model.pack_backbone_state` needs one construction-time branch and no second
    call path -- every mask, frame, tiling and feedback route stays the one that
    is already tested. Arguments this module does not use are accepted and
    IGNORED; `noisy_coords` is the important one, and a test pins that the
    output does not depend on it.
    """

    def __init__(
        self,
        c_res: int,
        c_node: int = 256,
        c_pair: int = 128,
        n_blocks: int = 6,
        ipa_c_hidden: int = 16,
        ipa_no_heads: int = 8,
        no_qk_points: int = 8,
        no_v_points: int = 12,
        seq_tfmr_num_heads: int = 4,
        seq_tfmr_num_layers: int = 4,
        transformer_dropout: float = 0.2,
        num_torsion_blocks: int = 4,
        c_pos_emb: int = 128,
        c_timestep_emb: int = 128,
        edge_feat_dim: int = 64,
        edge_num_bins: int = 22,
        n_type: int = 20,
        seq_cond: str = "a_token",
        embed_aatype: bool = True,
        embed_rotvecs: bool = True,
        use_mlp: bool = True,
        embed_chain: bool = True,
        random_torsion_input: bool = True,
        plm_checkpoint: str = "",
        plm_num_layers: int = 33,
        plm_embed_dim: int = 1280,
        trunk_grad_scale: float = 1.0,
        angle_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if seq_cond not in ("none", "a_token", "plm", "both"):
            raise ValueError(
                f"seq_cond must be one of none/a_token/plm/both, got {seq_cond!r}")
        if seq_cond in ("plm", "both") and not plm_checkpoint:
            raise ValueError(
                "seq_cond includes a PLM but no plm_checkpoint was given; the run "
                "would silently train an arm with one fewer input than its name says"
            )
        self.c_atom = int(c_node)      # width handed to the feedback contract
        self.c_node = int(c_node)
        self.c_pair = int(c_pair)
        self.n_blocks = int(n_blocks)
        self.seq_cond = str(seq_cond)
        self.embed_aatype = bool(embed_aatype)
        self.embed_rotvecs = bool(embed_rotvecs)
        self.random_torsion_input = bool(random_torsion_input)
        self.trunk_grad_scale = float(trunk_grad_scale)
        self.angle_eps = float(angle_eps)

        # ---- node features: APM's module, APM's names ----
        self.node_feature_net = PackingNodeFeatureNet(
            c_s=c_node, c_pos_emb=c_pos_emb, c_timestep_emb=c_timestep_emb,
            embed_aatype=self.embed_aatype, embed_rotvecs=self.embed_rotvecs,
            use_mlp=use_mlp, n_type=n_type)

        # ---- sequence conditioning, inserted where APM inserts the PLM ----
        # BOTH projections are always constructed, whichever arm is running, so
        # the four arms have bit-identical parameter sets, shapes and initial
        # weights and differ ONLY in information. (APM builds its PLM branch
        # conditionally; with seq_cond="plm" this module computes exactly what
        # APM computes, the extra projection simply sits unused.) The a_token
        # channel is switched off by multiplying by zero rather than by being
        # absent, so its gradient is defined and zero.
        from pxdesign_train.sidechain.plm import FrozenESM2, PLMConditioner

        self.a_proj = Linear(c_res, c_node, init="final")
        self.plm_conditioner = PLMConditioner(plm_num_layers, plm_embed_dim, c_node)
        # Plain object, never a submodule -- see plm.py's module docstring.
        self.plm_runner = FrozenESM2(plm_checkpoint) if self.seq_cond in (
            "plm", "both") else None

        # ---- edge features: APM's module, APM's names ----
        self.edge_feature_net = FullEdgeFeatureNet(
            c_s=c_node, c_p=c_pair, feat_dim=edge_feat_dim, num_bins=edge_num_bins,
            embed_chain=embed_chain)

        # ---- trunk ----
        self.trunk = nn.ModuleDict()
        for b in range(self.n_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(
                c_s=c_node, c_z=c_pair, c_hidden=ipa_c_hidden, no_heads=ipa_no_heads,
                no_qk_points=no_qk_points, no_v_points=no_v_points, dropout=0.0)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(c_node)
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_node, nhead=seq_tfmr_num_heads, dim_feedforward=c_node,
                batch_first=True, dropout=transformer_dropout, norm_first=False)
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                tfmr_layer, seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f"post_tfmr_{b}"] = Linear(c_node, c_node, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=c_node)
            if b < self.n_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=c_node, edge_embed_in=c_pair, edge_embed_out=c_pair)

        # ---- head ----
        self.torsion_pred_net = AngleResnet(
            c_in=c_node, c_hidden=c_node, no_blocks=num_torsion_blocks,
            no_angles=MAX_CHI, epsilon=angle_eps)

        # Per-atom features for the feedback contract (HResFeedback, a_direct,
        # q_direct). This module is residue-level by construction, so the atom
        # features are the residue feature plus the atom-name embedding.
        self.atom_embed = nn.Embedding(ATOM_VOCAB_SIZE, c_node, padding_idx=0)

        # Forward-scoped handoff, read by model.py right after the call (the same
        # pattern as `_a_token_cache`).
        self.last_torsions = None

    def _scale_grad(self, h_res: torch.Tensor) -> torch.Tensor:
        s = self.trunk_grad_scale
        if s == 1.0:
            return h_res
        return s * h_res + (1.0 - s) * h_res.detach()

    def _sequence_channel(self, h_res, type_idx, res_mask) -> torch.Tensor:
        """APM's `init_node_embed += plm_s`, generalised over the four arms."""
        out = self.a_proj(self._scale_grad(h_res).float())
        if self.seq_cond not in ("a_token", "both"):
            out = out * 0.0
        if self.plm_runner is not None:
            reps = self.plm_runner.representations(type_idx, res_mask)
            out = out + self.plm_conditioner(reps.float())
        return out

    def forward(
        self,
        h_res: torch.Tensor,            # [B, L, c_res]  a_token
        restype_logits: torch.Tensor,   # [B, L, n_type]
        atom_name_ids: torch.Tensor,    # [B, L, 10] long
        atom_mask: torch.Tensor,        # [B, L, 10] bool -- chemical/model slots
        noisy_coords: torch.Tensor,     # IGNORED (no x_t^SC in this module)
        t: torch.Tensor,                # IGNORED (no diffusion time)
        ca_coords: Optional[torch.Tensor] = None,
        frame_R: Optional[torch.Tensor] = None,
        frame_t: Optional[torch.Tensor] = None,
        bb_coords: Optional[torch.Tensor] = None,
        res_mask: Optional[torch.Tensor] = None,    # [B, L] bool
        ctx_mask: Optional[torch.Tensor] = None,    # [B, L] bool (context tokens)
        bb_q: Optional[torch.Tensor] = None,        # IGNORED
        coord_scale: Optional[torch.Tensor] = None,  # IGNORED (EDM c_in)
        bb_atom_mask: Optional[torch.Tensor] = None,  # [B, L, 4] bool
        residue_index: Optional[torch.Tensor] = None,  # [B, L] long
        asym_id: Optional[torch.Tensor] = None,        # [B, L] long, chain index
    ):
        if frame_R is None or frame_t is None:
            raise ValueError(
                "TorsionPacker needs the residue frames: IPA operates on them and "
                "BuildSC's output is mapped out with the same frame. model.py "
                "passes them unconditionally for this packer."
            )
        B, L, _ = atom_name_ids.shape
        device = h_res.device

        with torch.autocast(device_type=device.type, enabled=False):
            R = frame_R.float()
            trans = frame_t.float()
            type_idx = restype_logits.float().argmax(-1)
            node_mask = (torch.ones(B, L, dtype=torch.bool, device=device)
                         if res_mask is None else res_mask.bool())
            if ctx_mask is not None:
                node_mask = node_mask | ctx_mask.to(device).bool()
            node_mask_f = node_mask.float()
            edge_mask = node_mask_f[:, None, :] * node_mask_f[:, :, None]

            # Ownership of a side chain -- APM's `diffuse_mask`, i.e. "is this
            # position being generated".
            diffuse_mask = atom_mask.bool().any(-1).float()

            if residue_index is None:
                residue_index = torch.arange(L, device=device)[None].expand(B, L)
            residue_index = residue_index.to(device).float()

            # APM's packing mode: a uniform random torsion input, and a
            # self-conditioning channel that training always feeds zeros.
            if self.random_torsion_input:
                torsions_t = torch.rand(B, L, MAX_CHI, device=device) * 2 * math.pi
            else:
                torsions_t = torch.zeros(B, L, MAX_CHI, device=device)
            torsions_sc = torch.zeros(B, L, MAX_CHI, device=device)
            tor_t = torch.zeros(B, 1, device=device)

            # ---- node embedding (APM's PackingNodeFeatureNet) ----
            canonical = (type_idx >= 0) & (type_idx < len(STD_AA_3))
            aa_idx = torch.where(canonical, type_idx,
                                 torch.full_like(type_idx, len(STD_AA_3)))
            init_node_embed = self.node_feature_net(
                tor_t=tor_t, res_mask=node_mask_f, diffuse_mask=diffuse_mask,
                pos=residue_index, aatypes=aa_idx, rotvecs=rotmat_to_rotvec(R),
                torsions=torsions_t, torsions_sc=torsions_sc)

            init_node_embed = init_node_embed + self._sequence_channel(
                h_res, type_idx, node_mask)
            init_node_embed = init_node_embed * node_mask_f[..., None]

            # ---- edge embedding (APM's FullEdgeFeatureNet) ----
            chain_idx = (torch.zeros(B, L, dtype=torch.long, device=device)
                         if asym_id is None else asym_id.to(device).long())
            edge_embed = self.edge_feature_net(
                init_node_embed, trans, torsions_t, torsions_sc, edge_mask,
                diffuse_mask, residue_index, chain_idx)

            # ---- trunk ----
            node_embed = init_node_embed
            trans_nm = trans * ANG_TO_NM_SCALE
            for b in range(self.n_blocks):
                ipa_embed = self.trunk[f"ipa_{b}"](
                    node_embed, edge_embed, R, trans_nm, node_mask_f)
                ipa_embed = ipa_embed * node_mask_f[..., None]
                node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
                seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                    node_embed, src_key_padding_mask=~node_mask)
                node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
                node_embed = self.trunk[f"node_transition_{b}"](node_embed)
                node_embed = node_embed * node_mask_f[..., None]
                if b < self.n_blocks - 1:
                    edge_embed = self.trunk[f"edge_transition_{b}"](
                        node_embed, edge_embed) * edge_mask[..., None]

            # ---- head -> BuildSC ----
            unnorm_sincos, unit_sincos = self.torsion_pred_net(node_embed, init_node_embed)
            # APM's channel order is (sin, cos), and so is openfold's
            # `supervised_chi_loss`, which is what consumes the raw output.
            #
            # A MASKED ROW PRODUCES EXACTLY (0, 0). `node_embed` is multiplied by
            # the residue mask, AngleResnet's residual branch is zero-initialised
            # ("final") and every Linear's bias starts at 0, so the head's output
            # for a masked row is the zero vector -- not approximately, exactly.
            # atan2(0, 0) is 0 going forward and NaN going backward, and masking
            # afterwards does not help: NaN * 0 is NaN. Substitute (0, 1) BEFORE
            # the atan2 so its backward never sees the origin. Rows that are not
            # degenerate are untouched, and the substituted value is discarded by
            # the atom mask anyway.
            degenerate = unnorm_sincos.square().sum(-1) <= self.angle_eps
            sin_c = torch.where(degenerate, torch.zeros_like(unit_sincos[..., 0]),
                                unit_sincos[..., 0])
            cos_c = torch.where(degenerate, torch.ones_like(unit_sincos[..., 1]),
                                unit_sincos[..., 1])
            chi = torch.atan2(sin_c, cos_c)
            chi_mask = CHI_MASK.to(device)[type_idx.clamp(0, CHI_MASK.shape[0] - 1)]
            chi_built = torch.where(chi_mask, chi, torch.full_like(chi, float("nan")))
            local, _ = build_sidechain_local(type_idx, chi_built)
            x0 = to_global(local, R, trans)

        x0_global = torch.where(atom_mask.bool()[..., None], x0.to(h_res.dtype), 0.0)
        self.last_torsions = dict(raw=unnorm_sincos, unit=unit_sincos, chi=chi,
                                  chi_mask=chi_mask, type_idx=type_idx)

        h_out = node_embed.to(h_res.dtype)
        sc_feats = self.atom_embed(atom_name_ids) + h_out[:, :, None, :]
        sc_feats = torch.where(atom_mask.bool()[..., None], sc_feats, 0.0)
        bb_ids = BACKBONE_ATOM_NAME_IDS.to(atom_name_ids.device).view(1, 1, N_BB)
        bb_feats = self.atom_embed(bb_ids.expand(B, L, N_BB)) + h_out[:, :, None, :]
        if bb_atom_mask is not None:
            bb_feats = torch.where(bb_atom_mask.bool()[..., None], bb_feats, 0.0)
        return x0_global, sc_feats, bb_feats


__all__ = ["TorsionPacker", "AngularEncoding", "rotmat_to_rotvec", "ANG_TO_NM_SCALE"]
