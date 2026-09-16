"""Zero-initialized, noise-conditioned residual adapters between the modules.

Two separate adapters, one per direction:

    A_BS : R^{d_BB} -> R^{d_FA}     backbone token features -> FaMPNN node features
    A_SB : R^{d_FA} -> R^{d_BB}     FaMPNN node features -> backbone token features

Both compute

    A(z, sigma_B) = W_out * SiLU( W_in [ LN(z), e(log sigma_B) ] )

with ``W_out`` and ``b_out`` zero, so at initialization every adapter outputs
exactly zero and the coupled system reproduces the two pretrained models
bit-for-bit. That property is what makes the staged training plan safe to start:
phase 0 equivalence is not approximate, it is exact, and
:func:`ResidualAdapter.is_identity` asserts it.

Conditioning on the backbone noise level matters because the same FaMPNN features
mean different things at high and low sigma_B -- early in the trajectory the
backbone is barely determined, so side-chain evidence should count for less.
"""

import math

import torch
from torch import nn


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding of ``log sigma``, as used for diffusion timesteps."""

    def __init__(self, dim=64, max_period=10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"Noise embedding dim must be even, got {dim}")
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, sigma):
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        if sigma.dim() == 0:
            sigma = sigma.reshape(1)
        value = torch.log(sigma.clamp_min(1e-8))
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=value.device)
            / half
        )
        angles = value[..., None] * frequencies
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class ResidualAdapter(nn.Module):
    """One direction of coupling: a zero-initialized residual generator.

    ``forward`` returns a tensor shaped like the *target* latent, to be added to
    it. Input is ``[..., L, d_in]`` and ``sigma`` is per-example ``[B]`` (or a
    scalar), broadcast across residues.
    """

    def __init__(self, d_in, d_out, *, d_hidden=256, d_noise=64, dropout=0.0):
        super().__init__()
        self.d_in, self.d_out = int(d_in), int(d_out)
        self.norm = nn.LayerNorm(self.d_in)
        self.noise = NoiseEmbedding(d_noise)
        self.project_in = nn.Linear(self.d_in + d_noise, int(d_hidden))
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.project_out = nn.Linear(int(d_hidden), self.d_out)
        # Zero-init the output so the adapter starts as an exact no-op.
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, source, sigma):
        if source.shape[-1] != self.d_in:
            raise ValueError(
                f"Adapter expects last dim {self.d_in}, got {source.shape[-1]}"
            )
        normalized = self.norm(source)
        embedded = self.noise(sigma).to(normalized.dtype).to(normalized.device)
        if embedded.shape[0] == 1 and normalized.shape[0] > 1:
            embedded = embedded.expand(normalized.shape[0], -1)
        if embedded.shape[0] != normalized.shape[0]:
            raise ValueError(
                f"Got {embedded.shape[0]} sigma values for {normalized.shape[0]} examples"
            )
        # Broadcast the per-example noise embedding across residues.
        shape = (
            (normalized.shape[0],) + (1,) * (normalized.dim() - 2) + (embedded.shape[-1],)
        )
        embedded = embedded.reshape(shape).expand(
            *normalized.shape[:-1], embedded.shape[-1]
        )
        hidden = self.activation(self.project_in(torch.cat([normalized, embedded], -1)))
        return self.project_out(self.dropout(hidden))

    @torch.no_grad()
    def is_identity(self):
        """True while the adapter is still an exact no-op."""
        return bool(torch.all(self.project_out.weight == 0)) and bool(
            torch.all(self.project_out.bias == 0)
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class CouplingAdapters(nn.Module):
    """The pair of adapters, with the phase gating the training plan needs.

    ``enabled`` masks a direction at the call site rather than by zeroing weights,
    so the ablation table (none / BS / SB / both) is a runtime switch and the
    "second PXDesign pass with A_SB = 0" control costs nothing to run.
    """

    def __init__(
        self,
        d_backbone,
        d_fampnn,
        *,
        d_hidden=256,
        d_noise=64,
        enable_bb_to_sc=True,
        enable_sc_to_bb=True,
    ):
        super().__init__()
        self.d_backbone, self.d_fampnn = int(d_backbone), int(d_fampnn)
        self.bb_to_sc = ResidualAdapter(
            d_backbone, d_fampnn, d_hidden=d_hidden, d_noise=d_noise
        )
        self.sc_to_bb = ResidualAdapter(
            d_fampnn, d_backbone, d_hidden=d_hidden, d_noise=d_noise
        )
        self.enable_bb_to_sc = bool(enable_bb_to_sc)
        self.enable_sc_to_bb = bool(enable_sc_to_bb)

    def delta_h(self, a_token, sigma):
        """BB -> SC residual for FaMPNN's node features, or None when disabled."""
        return self.bb_to_sc(a_token, sigma) if self.enable_bb_to_sc else None

    def delta_a(self, h_packed, sigma):
        """SC -> BB residual for PXDesign's token features, or None when disabled."""
        return self.sc_to_bb(h_packed, sigma) if self.enable_sc_to_bb else None

    def set_phase(self, phase):
        """Freeze/unfreeze per the staged plan; returns what is trainable."""
        phases = {
            "frozen": (False, False),
            "bb_to_sc": (True, False),  # phase 1: train A_BS only
            "sc_to_bb": (False, True),  # phase 2: train A_SB only
            "joint": (True, True),  # phase 3: both, alternating losses
        }
        if phase not in phases:
            raise ValueError(f"Unknown phase {phase!r}; choose from {sorted(phases)}")
        train_bs, train_sb = phases[phase]
        self.bb_to_sc.requires_grad_(train_bs)
        self.sc_to_bb.requires_grad_(train_sb)
        self.phase = phase
        return dict(
            phase=phase,
            bb_to_sc=train_bs,
            sc_to_bb=train_sb,
            trainable_parameters=sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        )

    @torch.no_grad()
    def is_identity(self):
        """True while both directions are exact no-ops (phase 0 equivalence)."""
        return self.bb_to_sc.is_identity() and self.sc_to_bb.is_identity()

    def identity(self):
        return dict(
            d_backbone=self.d_backbone,
            d_fampnn=self.d_fampnn,
            enable_bb_to_sc=self.enable_bb_to_sc,
            enable_sc_to_bb=self.enable_sc_to_bb,
            phase=getattr(self, "phase", None),
            zero_initialized=self.is_identity(),
            parameters=sum(p.numel() for p in self.parameters()),
        )
