"""Named, replayable noise draws for joint refinement.

The donor's interpolant draws its training timesteps and its Gaussian from the
*global* RNG: ``EDM.sample_timestep`` calls ``torch.randn(n, device=...)`` and
``noise_x`` calls ``torch.randn_like(x)``, neither of which takes a generator.
``diffusion_loss(generator=...)`` therefore controls only the self-conditioning
coin. That is enough for a single-arm run seeded once at the top, and not enough
here: several arms must see *the same* backbone and side-chain noise so their
difference is the objective rather than the draw, and adding a diagnostic to one
arm must not shift another arm's stream.

So the draws are reproduced here against named generators. Reproduced, not
reimplemented differently: :func:`draw_sidechain_noise` follows
``EDM.sample_timestep`` term for term, including the ``sigma_data`` scaling and
the ``sigma_inv`` round trip, and is checked against the donor's own output
under a shared seed. A schedule this module has not been taught is an error
rather than a silent substitution -- the schedule lives in the checkpoint, so a
different donor can change it without anything else changing.

Two deliberate properties:

**Draws are made on the CPU and moved.** A CUDA generator produces a different
stream from a CPU one for the same seed, so a run replayed on another device
would diverge. The cost is one host-to-device copy per draw.

**Keys never include arm identity or loop position.** A key is
``(base seed, sample id, stream name, occurrence, sigma)``; two arms at the same
occurrence of the same sample therefore draw the same noise, which is what makes
their comparison paired.
"""

import hashlib
from dataclasses import dataclass

import torch

# Schedules this module can reproduce exactly. Anything else is refused: the
# point is bit-parity with the donor, and a "close enough" substitute would
# quietly change the noise distribution the model is trained against.
SUPPORTED_SCHEDULES = ("lognormal", "uniform_t", "constant_t")

STREAMS = (
    # "backbone_sigma" picks the noise LEVEL; "backbone_noise" draws the
    # perturbation at that level. They are separate streams on purpose: sharing
    # one would tie which sigma an example gets to the noise drawn at it.
    "backbone_sigma",
    "backbone_noise",
    "sidechain_time",
    "sidechain_noise",
    "self_conditioning",
    "augmentation",
)


class UnsupportedSchedule(NotImplementedError):
    """The donor's training noise schedule has no reproduction here."""


def stream_seed(base, sample_id, stream, *, occurrence=0, sigma=None):
    """A stable 63-bit seed for one (sample, stream, occurrence) draw.

    Hashed rather than arithmetic so that adding a stream cannot collide with an
    existing one, and so the value does not depend on the order streams are
    requested in. ``sigma`` is quantized before hashing: it reaches this as a
    float that has already been through a schedule, and an exact bit pattern
    would make the key depend on arithmetic order.
    """
    if stream not in STREAMS:
        raise ValueError(f"Unknown stream {stream!r}; choose from {list(STREAMS)}")
    parts = [str(int(base)), str(sample_id), stream, str(int(occurrence))]
    if sigma is not None:
        parts.append(f"{float(sigma):.6e}")
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generator_for(base, sample_id, stream, *, occurrence=0, sigma=None):
    """A CPU generator seeded by :func:`stream_seed`."""
    return torch.Generator().manual_seed(
        stream_seed(base, sample_id, stream, occurrence=occurrence, sigma=sigma)
    )


def _cpu_randn(shape, generator, *, device=None, dtype=torch.float32):
    """Draw on the CPU, then move: a seed must mean the same thing on any device."""
    values = torch.randn(*shape, generator=generator, dtype=dtype)
    return values if device is None else values.to(device)


# ---- backbone ---------------------------------------------------------------


def draw_backbone_noise(shape, generator, *, device=None, dtype=torch.float32):
    """Unit Gaussian for ``B_t = B* + sigma_B * eps``, drawn reproducibly."""
    return _cpu_randn(tuple(shape), generator, device=device, dtype=dtype)


# ---- side chain -------------------------------------------------------------


@dataclass
class SidechainNoise:
    """An explicit ``(t, epsilon)`` pair for one clone axis.

    Applying it reproduces ``EDM.forward`` without touching the global RNG:
    ``x_noised = x + sigma(t) * epsilon``, ``x_target = x`` (EDM predicts the
    clean sample directly), ``weight = 1 / c_out(sigma(t))^2``.
    """

    t: torch.Tensor  # [(m b)]
    epsilon: torch.Tensor  # [(m b), L, 33, 3]

    def to(self, device):
        return SidechainNoise(t=self.t.to(device), epsilon=self.epsilon.to(device))

    def apply(self, interpolant, x1):
        """``(x_noised, x_target, t, loss_weight)`` for a clean target ``x1``."""
        if self.epsilon.shape != x1.shape:
            raise ValueError(
                f"noise shaped {tuple(self.epsilon.shape)} for a target shaped "
                f"{tuple(x1.shape)}"
            )
        if self.t.shape[0] != x1.shape[0]:
            raise ValueError(
                f"{self.t.shape[0]} timesteps for {x1.shape[0]} clones"
            )
        t = self.t.to(device=x1.device, dtype=torch.float32)
        sigma = interpolant.sigma(t).reshape(-1, 1, 1, 1)
        noised = x1 + self.epsilon.to(x1.device, x1.dtype) * sigma.to(x1.dtype)
        return noised, x1, t, interpolant.get_loss_weight(t)


def schedule_of(interpolant):
    """The donor's training noise schedule name, refused if unreproducible."""
    name = str(interpolant.training_noise_schedule)
    if name not in SUPPORTED_SCHEDULES:
        raise UnsupportedSchedule(
            f"the loaded donor trains its side-chain diffusion with the {name!r} "
            f"schedule, which this module cannot reproduce exactly (it knows "
            f"{list(SUPPORTED_SCHEDULES)}). Teach it that schedule rather than "
            "substituting another: the distribution is part of the objective"
        )
    return name


def draw_sidechain_time(interpolant, n, generator, *, device=None):
    """``EDM.sample_timestep`` term for term, against an explicit generator."""
    name = schedule_of(interpolant)
    cfg = interpolant.training_noise_cfg
    if name == "constant_t":
        t = torch.ones(n, dtype=torch.float32) * float(cfg.t)
    elif name == "uniform_t":
        low, high = float(cfg.t_min), float(cfg.t_max)
        t = torch.rand(n, generator=generator, dtype=torch.float32) * (high - low) + low
    else:  # lognormal
        log_sigmas = (
            torch.randn(n, generator=generator, dtype=torch.float32) * float(cfg.psigma_std)
            + float(cfg.psigma_mean)
        )
        sigma_data = interpolant.sigma_data.detach().to("cpu", torch.float32)
        sigmas = sigma_data * torch.exp(log_sigmas)
        t = interpolant.sigma_inv(sigmas.to(interpolant.sigma_data.device)).to("cpu")
    return t if device is None else t.to(device)


def draw_sidechain_noise(interpolant, shape, generator, *, device=None, t=None):
    """A :class:`SidechainNoise` for a ``[(m b), L, 33, 3]`` target.

    One generator drives both the time and the Gaussian, in that order, so a
    single key reproduces the whole draw.
    """
    shape = tuple(shape)
    if len(shape) != 4:
        raise ValueError(f"expected [(m b), L, A, 3], got {shape}")
    if t is None:
        t = draw_sidechain_time(interpolant, shape[0], generator)
    epsilon = _cpu_randn(shape, generator)
    noise = SidechainNoise(t=t.to("cpu", torch.float32), epsilon=epsilon)
    return noise if device is None else noise.to(device)


def sidechain_noise_for(
    interpolant, shape, *, base, sample_id, occurrence=0, sigma=None, device=None
):
    """The named-key convenience form of :func:`draw_sidechain_noise`."""
    generator = generator_for(
        base, sample_id, "sidechain_time", occurrence=occurrence, sigma=sigma
    )
    return draw_sidechain_noise(interpolant, shape, generator, device=device)


def identity(*, base, streams=STREAMS):
    """What a run record needs to say about its random scheme."""
    return dict(
        base_seed=int(base),
        streams=list(streams),
        key="sha256(base|sample_id|stream|occurrence|sigma)",
        drawn_on="cpu",
        note="arm identity and loop position are deliberately not part of a key",
    )
