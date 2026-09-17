"""Where the BB->SC residual comes from, and how much of it is applied.

Phase 1's measured behaviour is two-sided: ``A_BS`` helps when the backbone is
badly noised and hurts when it is nearly native, and on out-of-distribution
panels a donor protein's token features reproduce most of both. Closing the
phase therefore needs two knobs kept separate:

* **where the residual comes from** -- matched ``a_token``, a shared mean, a
  zero input, or a donor -- which is what tests whether the useful content is
  sample-specific;
* **how strongly it is applied as a function of sigma_B** -- the gate -- which
  is what repairs the low-noise harm without retraining.

Holding the gate fixed and varying the source measures representation content.
Holding the source fixed and varying the gate measures the repair. Mixing the
two produces numbers that cannot be attributed to either.

Nothing here changes ``ResidualAdapter`` or any ``bb_to_sc.*`` weight name: the
gate multiplies the adapter's output from outside, so the 20k checkpoint loads
unchanged and the gate-one setting reproduces it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Residual sources. `shuffled` is applied by the caller through the cycle's
# a_token_override rather than here, because it needs a donor from a different
# structure and this function sees only one.
SOURCES = ("none", "matched", "mean", "zero_input")


@dataclass(frozen=True)
class Gate:
    """A smoothstep in log sigma: zero at or below ``sigma_off``, one at or above ``sigma_on``.

    Smoothstep rather than a hard switch so the residual does not jump
    discontinuously across a sigma the sampler actually visits, and in log
    sigma because that is the space the adapter is conditioned in.

    ``sigma_off`` is a floor, not a threshold: the gate is *exactly* zero there,
    so a run at that noise level reproduces the uncoupled arm bit-for-bit
    rather than approximately.
    """

    name: str
    sigma_off: float
    sigma_on: float

    def __post_init__(self):
        if not 0 < self.sigma_off < self.sigma_on:
            raise ValueError(
                f"gate {self.name}: need 0 < sigma_off < sigma_on, got "
                f"{self.sigma_off} and {self.sigma_on}"
            )

    def __call__(self, sigma):
        """``g(sigma)``, elementwise, for a float or a tensor."""
        lo, hi = math.log(self.sigma_off), math.log(self.sigma_on)
        if torch.is_tensor(sigma):
            u = ((torch.log(sigma.clamp_min(1e-12)) - lo) / (hi - lo)).clamp(0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)
        u = min(max((math.log(max(float(sigma), 1e-12)) - lo) / (hi - lo), 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def identity(self):
        return dict(name=self.name, sigma_off=self.sigma_off, sigma_on=self.sigma_on)


# The development sweep: progressively stronger protection of the clean end.
# Named rather than passed as raw numbers so a run records which was used.
GATES = {
    "A": Gate("A", 0.010, 0.429),
    "B": Gate("B", 0.082, 0.429),
    "C": Gate("C", 0.082, 1.642),
    # `one` is the ungated original: the 20k adapter exactly as trained.
    "one": Gate("one", 1e-9, 2e-9),
}


def gate_by_name(name):
    if name in (None, "none", "off"):
        return None
    if name not in GATES:
        raise ValueError(f"unknown gate {name!r}; choose from {sorted(GATES)}")
    return GATES[name]


class MeanResidual:
    """``mu(sigma)``: one shared vector per noise level, interpolated in log sigma.

    Estimated once over *training* proteins and frozen. Estimating it from an
    evaluation panel would let the control peek at the set it is scored on,
    which is the specific failure this arm exists to rule out.

    Interpolation is linear in log sigma and clamped at the knots: the schedule
    only ever visits noise levels inside the training window, and extrapolating
    a mean residual past its support would invent behaviour the estimate does
    not contain.
    """

    def __init__(self, sigmas, vectors, *, provenance=None):
        if len(sigmas) != len(vectors):
            raise ValueError(f"{len(sigmas)} sigma knots but {len(vectors)} vectors")
        if not sigmas:
            raise ValueError("a mean residual needs at least one sigma knot")
        order = sorted(range(len(sigmas)), key=lambda i: float(sigmas[i]))
        self.sigmas = [float(sigmas[i]) for i in order]
        self.vectors = torch.stack([torch.as_tensor(vectors[i]).reshape(-1) for i in order])
        self.provenance = provenance or {}

    @classmethod
    def from_json(cls, path):
        import json
        import pathlib

        blob = json.loads(pathlib.Path(path).read_text())
        return cls(
            blob["sigmas"],
            [torch.tensor(v) for v in blob["vectors"]],
            provenance=blob.get("provenance"),
        )

    def at(self, sigma, *, device=None, dtype=None):
        """``[c_h_V]`` for one sigma."""
        value = float(sigma)
        knots = self.vectors.to(device=device, dtype=dtype)
        if len(self.sigmas) == 1 or value <= self.sigmas[0]:
            return knots[0]
        if value >= self.sigmas[-1]:
            return knots[-1]
        upper = next(i for i, s in enumerate(self.sigmas) if s >= value)
        lower = upper - 1
        lo, hi = math.log(self.sigmas[lower]), math.log(self.sigmas[upper])
        weight = (math.log(value) - lo) / (hi - lo)
        return knots[lower] * (1.0 - weight) + knots[upper] * weight

    def identity(self):
        return dict(
            sigmas=list(self.sigmas),
            width=int(self.vectors.shape[-1]),
            **{k: v for k, v in self.provenance.items()},
        )


def residual(
    adapters,
    source,
    *,
    a_token=None,
    sigma=None,
    length=None,
    mean=None,
    gate=None,
):
    """The BB->SC residual for one arm, gated.

    Returns ``None`` for the bypass arm rather than a zero tensor, so the
    uncoupled packing path is the *same* path rather than an equivalent one --
    an all-zero residual still takes a different branch through
    ``with_residual`` and would only be equal up to float arithmetic.
    """
    if source == "none":
        return None
    if source == "matched":
        if a_token is None:
            raise ValueError("the matched source needs a_token")
        delta = adapters.delta_h(a_token, sigma)
    elif source == "zero_input":
        if a_token is None:
            raise ValueError("the zero-input source needs a_token for its shape")
        # Deliberately not the same as bypass: normalization offsets, biases and
        # the sigma embedding all make A(0, sigma) non-zero, and how much of the
        # effect that alone explains is the question this arm answers.
        delta = adapters.delta_h(torch.zeros_like(a_token), sigma)
    elif source == "mean":
        if mean is None:
            raise ValueError("the mean source needs a MeanResidual")
        if length is None:
            raise ValueError("the mean source needs the residue count to broadcast to")
        vector = mean.at(
            sigma if not torch.is_tensor(sigma) else float(sigma.reshape(-1)[0])
        )
        delta = vector.reshape(1, 1, -1).expand(1, int(length), -1)
    else:
        raise ValueError(f"unknown residual source {source!r}; choose from {SOURCES}")

    if delta is None or gate is None:
        return delta
    scale = gate(float(sigma.reshape(-1)[0]) if torch.is_tensor(sigma) else float(sigma))
    if scale == 0.0:
        # Exactly the bypass arm, not a scaled-to-zero approximation of it.
        return None
    return delta * scale
