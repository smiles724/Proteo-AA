"""Leakage-resistant backbone geometry representation for the AA head.

The encoder's contract is intentionally narrow: it receives predicted N/CA/C/O
coordinates, chain/residue ordering, and diffusion sigma.  It never receives
restype, reference atom metadata, side-chain rows, MSA, or the atom-aware
DiffusionModule ``a_token``.  Its output is rigid-transform invariant.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


BACKBONE_WIDTH = 4  # columns are N, CA, C, O


def _expand_before_trailing(
    value: torch.Tensor,
    target_lead: torch.Size,
    trailing_ndim: int,
) -> torch.Tensor:
    """Broadcast an item-level tensor over coordinate sample dimensions."""
    trail = value.shape[-trailing_ndim:]
    lead = value.shape[:-trailing_ndim]
    while len(lead) < len(target_lead):
        value = value.unsqueeze(-trailing_ndim - 1)
        lead = value.shape[:-trailing_ndim]
    return value.expand(*target_lead, *trail)


def _shift_tokens(value: torch.Tensor, offset: int) -> torch.Tensor:
    """At token i return value at i+offset, zero outside the chain array."""
    out = torch.zeros_like(value)
    if offset > 0:
        out[..., :-offset, :] = value[..., offset:, :]
    elif offset < 0:
        n = -offset
        out[..., n:, :] = value[..., :-n, :]
    else:
        out.copy_(value)
    return out


def _shift_mask(value: torch.Tensor, offset: int) -> torch.Tensor:
    out = torch.zeros_like(value)
    if offset > 0:
        out[..., :-offset] = value[..., offset:]
    elif offset < 0:
        n = -offset
        out[..., n:] = value[..., :-n]
    else:
        out.copy_(value)
    return out


class BackboneGeometryEncoder(nn.Module):
    """Encode only predicted backbone geometry into per-token AA features.

    Local residue frames remove global rotation/translation.  Each token sees
    its own N/CA/C/O geometry and the same-chain backbone atoms at sequence
    offsets -2, -1, +1 and +2.  A sigma embedding tells the encoder how noisy
    the denoised state is.  No atom/reference embeddings enter this module.
    """

    offsets = (-2, -1, 1, 2)

    def __init__(
        self,
        c_out: int = 384,
        c_hidden: int = 384,
        n_blocks: int = 3,
        sigma_dim: int = 16,
        spatial_neighbors: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if sigma_dim % 2:
            raise ValueError("sigma_dim must be even")
        self.c_out = int(c_out)
        self.sigma_dim = int(sigma_dim)
        self.spatial_neighbors = int(spatial_neighbors)
        self.eps = float(eps)
        # own N/C/O in the CA frame: 3*3; four neighbor backbones: 4*(4*3+1);
        # valid-token flag: 1; sigma embedding: sigma_dim.
        c_in = 9 + len(self.offsets) * 13 + 1 + self.sigma_dim
        self.input_proj = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            nn.SiLU(),
            nn.LayerNorm(c_hidden),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c_hidden, 2 * c_hidden),
                    nn.SiLU(),
                    nn.Linear(2 * c_hidden, c_hidden),
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(c_hidden) for _ in self.blocks])
        # Geometry-only graph messages add tertiary and inter-chain context.
        # Edge features are the neighbor CA vector in the query residue's local
        # frame plus distance; neighbor selection uses CA distance only.
        self.spatial_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c_hidden + 4, c_hidden),
                    nn.SiLU(),
                    nn.Linear(c_hidden, c_hidden),
                )
                for _ in self.blocks
            ]
        )
        self.spatial_norms = nn.ModuleList(
            [nn.LayerNorm(c_hidden) for _ in self.blocks]
        )
        self.output = nn.Sequential(nn.LayerNorm(c_hidden), nn.Linear(c_hidden, c_out))

    def _spatial_graph(
        self,
        ca: torch.Tensor,
        frame: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fixed kNN indices, invariant edge features, and edge mask."""
        lead = ca.shape[:-2]
        n_token = ca.shape[-2]
        flat_b = math.prod(lead) if lead else 1
        ca_f = ca.reshape(flat_b, n_token, 3)
        frame_f = frame.reshape(flat_b, n_token, 3, 3)
        valid_f = valid.reshape(flat_b, n_token)
        distance = torch.cdist(ca_f, ca_f)
        pair_valid = valid_f[:, :, None] & valid_f[:, None, :]
        pair_valid &= ~torch.eye(
            n_token, device=ca.device, dtype=torch.bool
        ).unsqueeze(0)
        distance = distance.masked_fill(~pair_valid, float("inf"))
        k = min(max(self.spatial_neighbors, 1), max(n_token - 1, 1))
        nearest_distance, nearest_idx = distance.topk(k, dim=-1, largest=False)
        batch = torch.arange(flat_b, device=ca.device)[:, None, None]
        neighbor_ca = ca_f[batch, nearest_idx]
        relative = neighbor_ca - ca_f[:, :, None, :]
        local = torch.einsum("blkc,blcd->blkd", relative, frame_f)
        edge_valid = torch.isfinite(nearest_distance)
        safe_distance = nearest_distance.masked_fill(~edge_valid, 0.0)
        edge = torch.cat(
            [local.clamp(-20.0, 20.0) / 10.0, safe_distance.log1p()[..., None]],
            dim=-1,
        )
        edge = edge * edge_valid[..., None].to(edge.dtype)
        return nearest_idx, edge, edge_valid

    def _sigma_embedding(
        self, sigma: Optional[torch.Tensor], lead: torch.Size, n_token: int, device
    ) -> torch.Tensor:
        if sigma is None:
            log_sigma = torch.zeros(*lead, device=device, dtype=torch.float32)
        else:
            log_sigma = torch.as_tensor(sigma, device=device).float().clamp_min(1e-8).log()
            while log_sigma.dim() < len(lead):
                log_sigma = log_sigma.unsqueeze(-1)
            if lead:
                log_sigma = log_sigma.expand(*lead)
            elif log_sigma.numel() == 1:
                log_sigma = log_sigma.reshape(())
            else:
                raise ValueError(
                    f"sigma shape {tuple(log_sigma.shape)} is incompatible with "
                    "coordinates without leading sample dimensions"
                )
        half = self.sigma_dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = log_sigma[..., None] * freq
        emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return emb.unsqueeze(-2).expand(*lead, n_token, self.sigma_dim)

    def forward(
        self,
        coordinates: torch.Tensor,
        backbone_atom_idx: torch.Tensor,
        *,
        asym_id: Optional[torch.Tensor] = None,
        residue_index: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[..., N_token, c_out]`` for ``[..., N_atom, 3]`` coords."""
        if coordinates.shape[-1] != 3:
            raise ValueError(f"coordinates must end in 3, got {coordinates.shape}")
        if backbone_atom_idx.shape[-1] != BACKBONE_WIDTH:
            raise ValueError(
                f"backbone_atom_idx must end in (L,4), got {backbone_atom_idx.shape}"
            )

        lead = coordinates.shape[:-2]
        n_atom = coordinates.shape[-2]
        idx = _expand_before_trailing(
            torch.as_tensor(backbone_atom_idx, device=coordinates.device).long(),
            lead,
            trailing_ndim=2,
        )
        n_token = idx.shape[-2]
        valid = (idx >= 0).all(dim=-1) & (idx < n_atom).all(dim=-1)

        flat = coordinates.reshape(-1, n_atom, 3)
        idx_flat = idx.clamp(0, max(n_atom - 1, 0)).reshape(-1, n_token * 4)
        batch = torch.arange(flat.shape[0], device=flat.device)[:, None]
        bb = flat[batch, idx_flat].reshape(*lead, n_token, 4, 3).float()
        bb = bb * valid[..., None, None].to(bb.dtype)

        n, ca, c, o = bb.unbind(dim=-2)
        ex = F.normalize(c - ca, dim=-1, eps=self.eps)
        n_vec = n - ca
        ez = F.normalize(torch.cross(ex, n_vec, dim=-1), dim=-1, eps=self.eps)
        ey = F.normalize(torch.cross(ez, ex, dim=-1), dim=-1, eps=self.eps)
        frame = torch.stack([ex, ey, ez], dim=-1)  # columns are local axes
        spatial_idx, spatial_edge, spatial_valid = self._spatial_graph(
            ca, frame, valid
        )

        def to_local(points: torch.Tensor) -> torch.Tensor:
            relative = points - ca[..., None, :]
            return torch.einsum("...lac,...lcd->...lad", relative, frame)

        own_local = to_local(torch.stack([n, c, o], dim=-2)).reshape(
            *lead, n_token, 9
        )
        pieces = [own_local]

        if asym_id is None:
            chain = torch.zeros(n_token, device=bb.device, dtype=torch.long)
        else:
            chain = torch.as_tensor(asym_id, device=bb.device).long()
        chain = _expand_before_trailing(chain, lead, trailing_ndim=1)
        # residue_index is deliberately not embedded.  It is used only to reject
        # nonconsecutive neighbors when native numbering contains gaps.
        if residue_index is None:
            resid = torch.arange(n_token, device=bb.device, dtype=torch.long)
        else:
            resid = torch.as_tensor(residue_index, device=bb.device).long()
        resid = _expand_before_trailing(resid, lead, trailing_ndim=1)

        for offset in self.offsets:
            neighbor = _shift_tokens(bb.reshape(*lead, n_token, 12), offset).reshape(
                *lead, n_token, 4, 3
            )
            neighbor_valid = _shift_mask(valid, offset)
            neighbor_chain = _shift_mask(chain, offset)
            neighbor_resid = _shift_mask(resid, offset)
            expected = resid + offset
            same_polymer = neighbor_valid & (neighbor_chain == chain) & (neighbor_resid == expected)
            local = to_local(neighbor).reshape(*lead, n_token, 12)
            local = local * same_polymer[..., None].to(local.dtype)
            pieces.extend([local, same_polymer[..., None].float()])

        pieces.extend(
            [
                valid[..., None].float(),
                self._sigma_embedding(sigma, lead, n_token, bb.device),
            ]
        )
        features = torch.cat(pieces, dim=-1)
        h = self.input_proj(features.to(self.input_proj[0].weight.dtype))
        flat_b = math.prod(lead) if lead else 1
        batch = torch.arange(flat_b, device=h.device)[:, None, None]
        for block, norm, spatial_block, spatial_norm in zip(
            self.blocks, self.norms, self.spatial_blocks, self.spatial_norms
        ):
            h_flat = h.reshape(flat_b, n_token, h.shape[-1])
            neighbor_h = h_flat[batch, spatial_idx]
            messages = spatial_block(
                torch.cat([neighbor_h, spatial_edge.to(neighbor_h.dtype)], dim=-1)
            )
            messages = messages * spatial_valid[..., None].to(messages.dtype)
            denom = spatial_valid.sum(dim=-1, keepdim=True).clamp_min(1)
            aggregate = (messages.sum(dim=-2) / denom).reshape_as(h)
            h = spatial_norm(h + aggregate)
            h = norm(h + block(h))
        out = self.output(h)
        return out * valid[..., None].to(out.dtype)
