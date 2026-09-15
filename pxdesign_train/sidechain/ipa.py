"""Invariant Point Attention and the structure-module blocks around it.

PROVENANCE. Ported from APM (https://github.com/bytedance/apm, Apache-2.0),
`apm/models/ipa_pytorch.py`, which is itself openfold's implementation of
AlphaFold2 Algorithm 22 as carried through FrameDiff / FrameFlow / MultiFlow.
Copyright (2025) Bytedance Ltd. and/or its affiliates; Copyright 2021 AlQuraishi
Laboratory. Reproduced here under Apache-2.0.

THE ONE DELIBERATE CHANGE. APM passes frames as an openfold `Rigid` object and
calls `r[..., None].apply(pts)` / `r[..., None, None].invert_apply(pts)`. This
port takes the rotation and translation as plain tensors `(R, t)`, which is the
representation `sidechain/frames.py` already produces everywhere else in this
repo, and does the same two operations explicitly. Adopting `Rigid` instead
would mean vendoring openfold's rigid_utils + geometry package for two method
calls. The maths is identical and `tests/test_sc_ipa.py` pins it: a global
rotation+translation of the inputs leaves the IPA output unchanged.

UNITS. APM scales translations to nanometres before the trunk
(`rigids_ang_to_nm`, 1/10). That is not cosmetic: the point-attention term
carries a hard-coded `9/2` variance constant that assumes nanometre-scale
distances. The caller must pass `t` in nanometres; `packer.py` does the scaling.
"""
import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """nn.Linear with AF2's named initializers (openfold 1.11.4 + source)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        elif init == "default":
            lecun_normal_init_(self.weight)
        elif init == "relu":
            he_normal_init_(self.weight)
        elif init == "glorot":
            glorot_uniform_init_(self.weight)
        elif init == "gating":
            gating_init_(self.weight)
            if bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        else:
            raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()
        self.c = c
        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return self.ln(s)


class EdgeTransition(nn.Module):
    def __init__(self, *, node_embed_size, edge_embed_in, edge_embed_out,
                 num_layers=2, node_dilation=2):
        super(EdgeTransition, self).__init__()
        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(batch_size * num_res ** 2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        return edge_embed.reshape(batch_size, num_res, num_res, -1)


class InvariantPointAttention(nn.Module):
    """AlphaFold2 Algorithm 22, on (R, t) instead of an openfold Rigid."""

    def __init__(self, c_s: int, c_z: int, c_hidden: int, no_heads: int,
                 no_qk_points: int, no_v_points: int, dropout: float = 0.0,
                 inf: float = 1e5, eps: float = 1e-8):
        super(InvariantPointAttention, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These differ from the supplement (no bias, Glorot) and follow the
        # official source (bias, LeCun), as openfold and APM both do.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        self.linear_q_points = Linear(self.c_s, self.no_heads * self.no_qk_points * 3)
        self.linear_kv_points = Linear(
            self.c_s, self.no_heads * (self.no_qk_points + self.no_v_points) * 3)
        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)

    # NOT `_apply`: nn.Module._apply is what `.to(device)` calls internally, and
    # shadowing it turns every device move into a TypeError. (It did -- job
    # 116989, which is why these are named after the frame instead.)
    @staticmethod
    def _frame_apply(R, t, pts):
        """Local -> global for [B, L, P, 3] points under per-residue frames."""
        return torch.einsum("blij,blpj->blpi", R, pts) + t[:, :, None, :]

    @staticmethod
    def _frame_invert_apply(R, t, pts):
        """Global -> local for [B, L, H, P, 3] points under per-residue frames."""
        return torch.einsum(
            "blij,blhpj->blhpi", R.transpose(-1, -2), pts - t[:, :, None, None, :])

    def forward(self, s: torch.Tensor, z: torch.Tensor, R: torch.Tensor,
                t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, L, c_s] single representation
            z: [B, L, L, c_z] pair representation
            R: [B, L, 3, 3] local->global rotation
            t: [B, L, 3] frame origin, IN NANOMETRES (see module docstring)
            mask: [B, L] float/bool
        Returns:
            [B, L, c_s] update to the single representation
        """
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = self._frame_apply(R, t, q_pts)
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = self._frame_apply(R, t, kv_pts)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        b = self.linear_b(z)

        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        a = a + (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        mask = mask.to(a.dtype)
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        if self.dropout_prob > 0.0:
            a = self.dropout(a)
        a = self.softmax(a)

        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (a[..., None, :, :, None]
             * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = self._frame_invert_apply(R, t, o_pt)

        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        pair_z = self.down_z(z)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        return self.linear_out(torch.cat(o_feats, dim=-1))


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        super(AngleResnetBlock, self).__init__()
        self.linear_1 = Linear(c_hidden, c_hidden, init="relu")
        self.linear_2 = Linear(c_hidden, c_hidden, init="final")
        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        return a + s_initial


class AngleResnet(nn.Module):
    """AlphaFold2 Algorithm 20, lines 11-14. Returns (unnormalised, unit)."""

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        super(AngleResnet, self).__init__()
        self.no_angles = no_angles
        self.eps = epsilon
        self.linear_in = Linear(c_in, c_hidden)
        self.linear_initial = Linear(c_in, c_hidden)
        self.layers = nn.ModuleList(
            [AngleResnetBlock(c_hidden=c_hidden) for _ in range(no_blocks)])
        self.linear_out = Linear(c_hidden, no_angles * 2)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor):
        # The ReLUs on the inputs are absent from the AF2 supplement pseudocode
        # but present in the source; openfold and APM both follow the source.
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for layer in self.layers:
            s = layer(s)

        s = self.relu(s)
        s = self.linear_out(s)
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=self.eps))
        return unnormalized_s, s / norm_denom


__all__ = [
    "Linear", "StructureModuleTransition", "EdgeTransition",
    "InvariantPointAttention", "AngleResnet", "AngleResnetBlock",
]
