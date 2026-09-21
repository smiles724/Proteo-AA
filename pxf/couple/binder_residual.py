"""Scatter the BB->SC residual onto the binder rows only.

``bs_policy.residual`` answers "where does the residual come from and how
strongly is it applied", and it has no notion of chain roles because the
unconditional co-design path it was written for has none: every row is
designed. Binder design does have roles, and the residual must reach exactly
one of them.

    delta_h = binder_mask * gate(sigma_B) * A_BS(a_token, sigma_B)

Why the mask is not a detail. The target's sequence and side chains are held
fixed for the entire decode, so a residual on a target row cannot change what
that row *is* -- but ``h_V`` is the input to a graph encoder, so it still
changes the messages the target sends to the binder. An unmasked residual
therefore perturbs the design through a second, unintended channel, and the
arm stops being "A_BS informs the binder's packing" and becomes "A_BS perturbs
the whole complex representation". Both might help. Only one is the hypothesis.

The masking is enforced, not merely applied: :func:`binder_masked_residual`
checks the result is bit-exactly zero on every target row before returning it,
because a mask that is silently the wrong shape or the wrong orientation
broadcasts cleanly and produces a plausible, wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from pxf.couple.bs_policy import residual


@dataclass(frozen=True)
class ChainRoles:
    """Which token rows are the fixed target and which are the designed binder.

    ``binder`` is a ``[L]`` bool tensor. It is the single source of truth for
    the split and is what every downstream mask is derived from -- the
    ``fixed_sequence_mask`` handed to the designer, the rows the residual
    reaches, and the rows a metric is computed over. Deriving them separately
    is how they drift apart.
    """

    binder: torch.Tensor

    def __post_init__(self) -> None:
        if self.binder.dtype != torch.bool:
            raise TypeError(f"binder mask must be bool, got {self.binder.dtype}")
        if self.binder.dim() != 1:
            raise ValueError(f"binder mask must be [L], got {tuple(self.binder.shape)}")
        if not bool(self.binder.any()):
            raise ValueError("binder mask selects no rows; there is nothing to design")
        if bool(self.binder.all()):
            raise ValueError(
                "binder mask selects every row; this is a monomer co-design run, "
                "not binder design -- use scripts/codesign_uncond.py"
            )

    @property
    def length(self) -> int:
        return int(self.binder.shape[0])

    @property
    def n_binder(self) -> int:
        return int(self.binder.sum())

    @property
    def n_target(self) -> int:
        return self.length - self.n_binder

    @property
    def target(self) -> torch.Tensor:
        return ~self.binder

    def is_contiguous_tail(self) -> bool:
        """True when the binder is a suffix of the token axis.

        Worth recording rather than assuming: several places downstream take
        the binder as ``[-n_binder:]`` for speed, and that is only correct for
        a tail. A target built from several chains can put the binder
        elsewhere depending on how the complex was assembled.
        """
        indices = torch.nonzero(self.binder, as_tuple=False).reshape(-1)
        expected = torch.arange(
            self.length - self.n_binder, self.length, device=indices.device
        )
        return bool(indices.shape == expected.shape and torch.equal(indices, expected))

    def fixed_sequence_mask(self) -> torch.Tensor:
        """``[L]`` long, 1 where the designer must hold the identity: the target."""
        return self.target.long()

    def identity(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "n_target": self.n_target,
            "n_binder": self.n_binder,
            "binder_is_contiguous_tail": self.is_contiguous_tail(),
        }

    @classmethod
    def from_lengths(cls, n_target: int, n_binder: int) -> "ChainRoles":
        mask = torch.zeros(n_target + n_binder, dtype=torch.bool)
        mask[n_target:] = True
        return cls(binder=mask)

    @classmethod
    def from_chain_index(cls, chain_index: torch.Tensor, binder_chain: int) -> "ChainRoles":
        return cls(binder=(chain_index.reshape(-1) == int(binder_chain)))


def binder_masked_residual(
    adapters,
    source: str,
    *,
    roles: ChainRoles,
    a_token: Optional[torch.Tensor] = None,
    sigma=None,
    mean=None,
    gate=None,
    device=None,
    dtype=None,
) -> Optional[torch.Tensor]:
    """``[1, L, c_h_V]`` residual, zero on target rows, or ``None`` for bypass.

    ``None`` propagates from ``bs_policy.residual`` unchanged: the bypass arm
    and a gate that closed to exactly zero both mean "take the uncoupled path",
    and returning a zero tensor instead would route the arm through
    ``_add_residual`` and make it equal only up to float arithmetic.
    """
    delta = residual(
        adapters,
        source,
        a_token=a_token,
        sigma=sigma,
        length=roles.length,
        mean=mean,
        gate=gate,
        device=device,
        dtype=dtype,
    )
    if delta is None:
        return None

    if delta.dim() == 2:
        delta = delta[None]
    if delta.dim() != 3:
        raise ValueError(f"expected a [B, L, C] residual, got {tuple(delta.shape)}")
    if delta.shape[-2] != roles.length:
        raise ValueError(
            f"residual covers {delta.shape[-2]} rows but the complex has "
            f"{roles.length}; the adapter was built for a different structure"
        )

    # Finiteness is checked BEFORE masking, and the order is not cosmetic:
    # NaN * 0 is NaN, so a non-finite adapter output survives the mask and
    # trips the target-rows-are-zero assertion instead. The run would then
    # abort blaming the mask orientation for a numerically broken adapter.
    if not bool(torch.isfinite(delta).all()):
        raise AssertionError(
            "residual contains non-finite values before masking; the adapter "
            "output is broken, not the mask"
        )

    mask = roles.binder.to(device=delta.device).reshape(1, -1, 1)
    masked = delta * mask

    # Enforced, not assumed. A [L] mask and a [B] mask both broadcast against
    # [B, L, C] without error when B == L, and a transposed mask produces a
    # residual that is plausible and wrong.
    if bool(masked[:, roles.target.to(masked.device), :].abs().sum() != 0):
        raise AssertionError(
            "masking failed: residual is non-zero on target rows. Check the "
            "binder mask orientation against the token axis."
        )
    if bool(masked.abs().sum() == 0):
        raise AssertionError(
            "residual is identically zero on the binder rows too. The coupled "
            "arm would be a duplicate of the uncoupled one; refusing to report "
            "that as a null result."
        )
    return masked


def describe_residual(
    delta: Optional[torch.Tensor],
    roles: ChainRoles,
    *,
    h_v: Optional[torch.Tensor] = None,
    gate_value: Optional[float] = None,
) -> dict[str, Any]:
    """Numbers that say whether the residual can plausibly do anything.

    The one that matters is ``relative_norm``: the residual's per-row norm
    against ``h_V``'s. A residual three orders of magnitude below the signal it
    is added to is arithmetically present and practically inert, and that is a
    far more likely explanation of a null result than "coupling does not
    help". Reporting it costs nothing and forecloses the wrong conclusion.
    """
    out: dict[str, Any] = {
        "applied": delta is not None,
        "gate_value": gate_value,
        **roles.identity(),
    }
    if delta is None:
        return out

    binder_rows = delta[:, roles.binder.to(delta.device), :]
    per_row = binder_rows.norm(dim=-1)
    out.update({
        "binder_row_norm_mean": float(per_row.mean()),
        "binder_row_norm_max": float(per_row.max()),
        "target_row_norm_max": float(
            delta[:, roles.target.to(delta.device), :].norm(dim=-1).max()
        ),
        "n_rows_touched": int((delta.norm(dim=-1) > 0).sum()),
        "finite": bool(torch.isfinite(delta).all()),
    })
    if h_v is not None:
        reference = h_v
        if reference.dim() == 2:
            reference = reference[None]
        h_rows = reference[:, roles.binder.to(reference.device), :]
        h_norm = float(h_rows.norm(dim=-1).mean())
        out["h_v_row_norm_mean"] = h_norm
        out["relative_norm"] = (
            float(per_row.mean()) / h_norm if h_norm > 0 else None
        )
    return out
