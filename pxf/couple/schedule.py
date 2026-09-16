"""Where in the backbone trajectory the coupling adapters are trained.

Both adapters are conditioned on the backbone noise level,

    A(z, sigma_B) = W_out SiLU( W_in [ LN(z), e(log sigma_B) ] ),

so the distribution ``sigma_B`` is drawn from during training decides what that
conditioning can mean. Training at a single ``sigma_B`` leaves
:class:`~pxf.couple.adapters.NoiseEmbedding` with a constant input: the adapter
still works, but only at that one noise level, and any deployment that invokes
coupling across several denoising steps is then extrapolating.

So ``sigma_B`` is sampled from the interval the coupling is actually intended to
run in:

    PXDesign trajectory, 400 steps, sigma_data = 16 A

      step      0        200       280     320      360     400
      sigma  2560       56.0      5.06    1.02     0.126   0.0064
             |-----------|----------|-------|--------|-------|
              coupling off          [ ---- training window ---- ]

The default window is ``[0.01, 5.0]`` A, the last ~30% of the trajectory: from
the point where the fold is decided but the geometry is still imprecise, down to
the final step. Before that the backbone is barely determined and side-chain
evidence has nothing to attach to.

``mode="trajectory"`` (the default) samples a *step index* uniformly from the
window and returns the schedule's exact ``sigma`` at that step, so every noise
level the adapter trains at is one the sampler will really visit.
``mode="loguniform"`` samples continuously and uniformly in ``log sigma`` over the
same interval, which covers between-step values as well. ``mode="fixed"`` pins a
single value and exists for deployment-matched ablations -- it is the degenerate
case this module was written to stop being the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# PXDesign's published inference scheduler (pxdesign/configs/configs_base.py:
# inference_noise_scheduler + sample_diffusion.N_step). Protenix's defaults are
# identical apart from N_step.
PXDESIGN_S_MAX = 160.0
PXDESIGN_S_MIN = 4e-4
PXDESIGN_RHO = 7.0
PXDESIGN_SIGMA_DATA = 16.0
PXDESIGN_N_STEP = 400

# The intended coupling window, in Angstroms of per-coordinate noise.
DEFAULT_SIGMA_MIN = 0.01
DEFAULT_SIGMA_MAX = 5.0

MODES = ("trajectory", "loguniform", "fixed")


def karras_sigmas(
    n_step=PXDESIGN_N_STEP,
    *,
    s_max=PXDESIGN_S_MAX,
    s_min=PXDESIGN_S_MIN,
    rho=PXDESIGN_RHO,
    sigma_data=PXDESIGN_SIGMA_DATA,
    device=None,
    dtype=torch.float64,
):
    """The ``n_step + 1`` noise levels PXDesign denoises through, descending.

    A transcription of ``InferenceNoiseScheduler.__call__``, kept here so the
    training distribution is defined by the same formula the sampler uses rather
    than by a guess at its range.
    """
    steps = torch.arange(int(n_step) + 1, device=device, dtype=dtype)
    a, b = s_max ** (1.0 / rho), s_min ** (1.0 / rho)
    return sigma_data * (a + (steps / float(n_step)) * (b - a)) ** rho


@dataclass
class CouplingNoiseSchedule:
    """The ``sigma_B`` distribution a coupling run trains against.

    Attributes:
        mode: one of :data:`MODES`.
        sigma_min, sigma_max: the window, in Angstroms (ignored when fixed).
        sigma: the single value used by ``mode="fixed"``.
        n_step, s_max, s_min, rho, sigma_data: the backbone sampler's schedule,
            defaulting to PXDesign's published one. ``sigma_data`` should match
            the donor; pass ``PXDesignBackboneDriver.sigma_data`` to be exact.
    """

    mode: str = "trajectory"
    sigma_min: float = DEFAULT_SIGMA_MIN
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma: float = 1.0
    n_step: int = PXDESIGN_N_STEP
    s_max: float = PXDESIGN_S_MAX
    s_min: float = PXDESIGN_S_MIN
    rho: float = PXDESIGN_RHO
    sigma_data: float = PXDESIGN_SIGMA_DATA
    _window: torch.Tensor = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown sigma mode {self.mode!r}; choose from {list(MODES)}")
        self.sigma_min = float(self.sigma_min)
        self.sigma_max = float(self.sigma_max)
        self.sigma = float(self.sigma)
        self.n_step = int(self.n_step)

        if self.mode == "fixed":
            if not self.sigma > 0:
                raise ValueError(f"--sigma must be positive, got {self.sigma}")
            return

        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                "the coupling window needs 0 < sigma_min < sigma_max, got "
                f"[{self.sigma_min}, {self.sigma_max}]"
            )
        sigmas = self.trajectory()
        inside = sigmas[(sigmas >= self.sigma_min) & (sigmas <= self.sigma_max)]
        if self.mode == "trajectory" and inside.numel() < 2:
            # A window this narrow makes the noise conditioning constant, which
            # is the failure this schedule exists to prevent -- so it is an error
            # rather than a silently degenerate run. Say "fixed" if you mean it.
            raise ValueError(
                f"only {inside.numel()} of the {self.n_step + 1} trajectory steps fall "
                f"in [{self.sigma_min}, {self.sigma_max}], so sigma_B would be "
                "effectively constant and the noise embedding would learn nothing. "
                "Widen the window, use mode='loguniform', or say mode='fixed'."
            )
        self._window = inside.to(torch.float32)

    def trajectory(self, device=None):
        """The full descending schedule, ``[n_step + 1]``."""
        return karras_sigmas(
            self.n_step,
            s_max=self.s_max,
            s_min=self.s_min,
            rho=self.rho,
            sigma_data=self.sigma_data,
            device=device,
        )

    def window_steps(self):
        """``(first, last)`` trajectory step indices inside the window."""
        sigmas = self.trajectory()
        inside = torch.nonzero(
            (sigmas >= self.sigma_min) & (sigmas <= self.sigma_max), as_tuple=False
        ).flatten()
        if inside.numel() == 0:
            return None, None
        return int(inside[0]), int(inside[-1])

    def sample(self, n=1, *, generator=None, device=None):
        """Draw ``n`` backbone noise levels, ``[n]`` float32 on ``device``."""
        n = int(n)
        if self.mode == "fixed":
            out = torch.full((n,), self.sigma, dtype=torch.float32)
        elif self.mode == "trajectory":
            # Uniform over the window's steps, so the training density matches
            # how much sampler time is spent at each noise level.
            index = torch.randint(
                self._window.numel(), (n,), generator=generator, dtype=torch.long
            )
            out = self._window[index]
        else:  # loguniform
            lo, hi = math.log(self.sigma_min), math.log(self.sigma_max)
            out = torch.exp(torch.rand(n, generator=generator) * (hi - lo) + lo).float()
        return out.to(device) if device is not None else out

    def identity(self):
        """A JSON-safe record of the distribution, for run configs and checkpoints."""
        record = dict(
            mode=self.mode,
            n_step=self.n_step,
            s_max=self.s_max,
            s_min=self.s_min,
            rho=self.rho,
            sigma_data=self.sigma_data,
        )
        if self.mode == "fixed":
            record.update(sigma=self.sigma, distinct_values=1)
            return record
        first, last = self.window_steps()
        record.update(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            window_steps=[first, last],
            window_fraction_of_trajectory=(
                None if first is None else (last - first + 1) / (self.n_step + 1)
            ),
            distinct_values=(
                int(self._window.numel()) if self.mode == "trajectory" else None
            ),
        )
        return record

    def describe(self):
        """One line for the log, so a run's noise range is visible at a glance."""
        if self.mode == "fixed":
            return f"sigma_B fixed at {self.sigma:g} (noise conditioning is constant)"
        first, last = self.window_steps()
        return (
            f"sigma_B ~ {self.mode} over [{self.sigma_min:g}, {self.sigma_max:g}] A "
            f"= steps {first}-{last} of {self.n_step} "
            f"({(last - first + 1) / (self.n_step + 1):.0%} of the trajectory, late end)"
        )


def from_config(spec=None, **overrides):
    """Build a schedule from a config mapping plus command-line overrides.

    ``None`` overrides are dropped, so an unset flag never shadows the config.
    """
    merged = dict(spec or {})
    merged.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(merged) - {f for f in CouplingNoiseSchedule.__dataclass_fields__}
    if unknown:
        raise ValueError(f"Unknown sigma schedule keys: {sorted(unknown)}")
    mode = merged.get("mode", CouplingNoiseSchedule.mode)
    if mode != "fixed" and "sigma" in merged:
        # A leftover --sigma from before this module existed would otherwise be
        # accepted and ignored, and the run would look configured when it wasn't.
        raise ValueError(
            f"sigma={merged['sigma']} was given but mode is {mode!r}, which samples "
            "from [sigma_min, sigma_max] and never reads it. Pass mode='fixed' to "
            "train at one noise level, or set the window instead."
        )
    if mode == "fixed":
        for key in ("sigma_min", "sigma_max"):
            merged.pop(key, None)  # a window is meaningless for a single value
    return CouplingNoiseSchedule(**merged)
