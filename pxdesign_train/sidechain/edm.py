"""EDM diffusion for the Side-Chain Module.

WHAT THIS FIXES, and why it is a separate module.

`SideChainModule` is not a diffusion model. It is a one-shot regressor that has
only ever seen ONE noise level:

  * `template_init_local` builds `y_T = mu_ideal[a, chi(a, phi, psi)] + sigma_T*eps`
    with `sigma_T = 0.3 A`, fixed, never sampled.
  * Its time embedding is dead in Stage II (`t = ones`, a constant, so `w_t`
    contributes a learned bias and nothing else) and in Stage III it is fed the
    BACKBONE's sigma, not its own.
  * There is no preconditioning: `self.out` regresses x0 directly.

The number that motivates this file: the input starts **2.18 A** from the target
(that is the measured ideal-rotamer template baseline, `atom_weighted_rmse` over
491 proteins), while the noise the model is told about is **0.3 A**. Seven times
smaller -- and the real gap is not Gaussian at all, it is dominated by discrete
chi flips where the modal rotamer picked the wrong branch. So the model is handed
a start far from the target and informed it is in a low-noise regime.

That mismatch is a good explanation for the measured A/B: `template_residual`
defines the task as "learn a small continuous correction" when the true
correction is multi-modal, which is easy to memorise on the training set and does
not transfer -- the local arm's validation stalled at step 12k while the global
arm, with no template anchor, was still improving at 50k.

This module supplies the three missing pieces, behind a config flag, leaving the
one-step path untouched:

  A1  sigma is SAMPLED (log-normal, EDM-style) over a range that covers the real
      2-3 A error scale, and log(sigma) is what the time embedding receives -- so
      the channel carries information instead of a constant.
  A3  a side-chain `sigma_data`. The backbone diffusion uses 16.0, appropriate for
      whole-protein coordinates spanning tens of Angstrom; side-chain geometry in
      the residue frame has a standard deviation near 2 A, and reusing 16 would
      place every side-chain sigma in the "essentially clean" corner of the
      preconditioning where c_skip ~ 1 and the network is barely consulted.
  A2  a reverse loop that CARRIES STATE. Today the sampler calls S_phi once per
      backbone step and re-initialises from the template every time, discarding
      the previous estimate; N independent one-step decodes, not a trajectory.

Karras et al. 2022 ("Elucidating the Design Space of Diffusion-Based Generative
Models"), Table 1, EDM column, is the parameterisation used throughout.

WHERE THE PRECONDITIONING IS APPLIED. EDM assumes data that is centred with
standard deviation `sigma_data`. Global protein coordinates are neither. So the
scalings are computed in the CA-centred frame, where side-chain geometry really
does look like that, and the module keeps receiving global coordinates -- the
single-frame contract is unchanged, and the cross-residue distance bias keeps
seeing true Angstrom.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


# Side-chain coordinates in the residue frame span roughly 0-6 A with a standard
# deviation near 2 A (ARG's NH1/NH2 are the far tail; ALA's CB the near one).
# This is an ESTIMATE, not a measurement: to pin it, take the std of
# `sc_gt_local[sc_atom_mask]` over a few hundred training items. It is a config
# value precisely so that measurement can replace it without touching code.
DEFAULT_SIGMA_DATA = 2.0

# EDM's log-normal sigma sampler, retuned for the range above. exp(P_mean) is the
# median sigma; with P_mean = ln(0.8) and P_std = 1.0 the middle 68% of draws land
# in ~0.3-2.2 A, which brackets both the template's 2.18 A starting error and the
# sub-Angstrom regime the model has to finish in.
DEFAULT_P_MEAN = math.log(0.8)
DEFAULT_P_STD = 1.0


def edm_scalings(sigma: torch.Tensor, sigma_data: float = DEFAULT_SIGMA_DATA):
    """Karras Table 1 (EDM): returns (c_skip, c_in, c_out, c_noise).

    `sigma` is [...]; every output broadcasts against it.

        D(x; sigma) = c_skip * x + c_out * F(c_in * x, c_noise)

    The point of the parameterisation is that F's input and target both stay unit
    variance across the whole sigma range, so one network can serve every noise
    level. Without it a network trained at sigma=0.3 and evaluated at sigma=3 is
    being asked to extrapolate on both.
    """
    s2, d2 = sigma**2, sigma_data**2
    c_skip = d2 / (s2 + d2)
    c_out = sigma * sigma_data / torch.sqrt(s2 + d2)
    c_in = 1.0 / torch.sqrt(s2 + d2)
    c_noise = torch.log(sigma.clamp_min(1e-8)) / 4.0
    return c_skip, c_in, c_out, c_noise


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float = DEFAULT_SIGMA_DATA):
    """lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.

    This is the weight that makes the effective training target uniform across
    sigma. It is exactly 1 / c_out^2, i.e. it undoes the output scaling, so a
    plain MSE on the DENOISED coordinates weighted by lambda equals the unit-
    weighted MSE on F's own output. Omitting it makes high-sigma draws dominate
    the gradient, which is the usual way an EDM implementation quietly fails.
    """
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data).clamp_min(1e-8) ** 2


class SideChainNoiseSampler:
    """A1: log-normal sigma, EDM-style, with a hard range clamp.

    The clamp matters here in a way it does not for the backbone: side-chain
    geometry is bounded (a side chain is at most ~6 A long), so a sigma far above
    that range produces a target the network cannot use and a sigma far below it
    produces one it cannot distinguish from the clean structure.
    """

    def __init__(
        self,
        p_mean: float = DEFAULT_P_MEAN,
        p_std: float = DEFAULT_P_STD,
        sigma_min: float = 0.05,
        sigma_max: float = 4.0,
    ) -> None:
        if not sigma_min < sigma_max:
            raise ValueError(f"sigma_min={sigma_min} must be below sigma_max={sigma_max}")
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, size, device=None, dtype=torch.float32,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        n = torch.randn(size, device=device, dtype=dtype, generator=generator)
        sigma = torch.exp(self.p_mean + self.p_std * n)
        return sigma.clamp(self.sigma_min, self.sigma_max)

    def schedule(self, n_steps: int, rho: float = 7.0, device=None,
                 dtype=torch.float32) -> torch.Tensor:
        """Karras sigma schedule, sigma_max -> sigma_min -> 0, for the reverse loop.

        `rho=7` is Karras' default: it spends more steps at low sigma, where the
        remaining error is the chi-angle fine structure that actually needs them.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        i = torch.arange(n_steps, device=device, dtype=dtype)
        inv = 1.0 / rho
        sig = (
            self.sigma_max**inv
            + i / max(1, n_steps - 1) * (self.sigma_min**inv - self.sigma_max**inv)
        ) ** rho
        return torch.cat([sig, torch.zeros(1, device=device, dtype=dtype)])


class SideChainEDM:
    """Wraps a raw `SideChainModule` as a preconditioned EDM denoiser.

    Deliberately NOT an nn.Module: it owns no parameters, so a checkpoint trained
    one-step and a checkpoint trained under EDM hold the identical state dict.
    What differs is only how the network is called and what it is asked to
    predict -- which is exactly why `sidechain.edm` has to be a recorded arch
    switch, not an inferable one.
    """

    def __init__(self, module, sigma_data: float = DEFAULT_SIGMA_DATA) -> None:
        if getattr(module, "template_residual", False):
            raise ValueError(
                "sidechain.edm and sidechain.template_residual both add a skip from "
                "the input to the output -- c_skip(sigma)*x and noisy_local "
                "respectively -- so enabling both double-counts the template and "
                "makes the effective residual sigma-dependent in a way nothing "
                "accounts for. EDM's c_skip already IS the template residual, with "
                "the right sigma-dependent magnitude; turn template_residual off."
            )
        self.module = module
        self.sigma_data = float(sigma_data)

    def denoise(
        self,
        x_noisy: torch.Tensor,        # [B, L, A, 3] GLOBAL, noised side-chain atoms
        sigma: torch.Tensor,          # [B] or scalar
        ca_coords: torch.Tensor,      # [B, L, 3] GLOBAL, the centring origin
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return the denoised x0 in GLOBAL coordinates.

        `*args` / `**kwargs` are forwarded to `SideChainModule.forward` verbatim
        (h_res, restype_logits, atom_name_ids, atom_mask, frames, bb_coords, ...),
        so this wrapper never has to track that signature.
        """
        sigma = torch.as_tensor(sigma, device=x_noisy.device, dtype=x_noisy.dtype)
        while sigma.dim() < 1:
            sigma = sigma[None]
        s = sigma.view(-1, *([1] * (x_noisy.dim() - 1)))     # [B,1,1,1]
        c_skip, c_in, c_out, c_noise = edm_scalings(s, self.sigma_data)

        # Centre before scaling: the scalings assume data with std sigma_data about
        # zero, and a global coordinate carries the residue's absolute position.
        centre = ca_coords[:, :, None, :].to(x_noisy.dtype)
        y = x_noisy - centre

        # The network still receives GLOBAL coordinates -- its cross-residue
        # distance bias must see true Angstrom -- and `coord_scale` applies c_in to
        # the centred per-atom embedding only, which is where EDM's input scaling
        # belongs.
        f = self.module(
            *args,
            noisy_coords=x_noisy,
            ca_coords=ca_coords,
            t=c_noise.reshape(-1),
            coord_scale=c_in.reshape(-1),
            **kwargs,
        )
        f_out = f[0] if isinstance(f, tuple) else f
        rest = f[1:] if isinstance(f, tuple) else ()

        # F predicts in the same centred frame it was scaled in.
        y0 = c_skip * y + c_out * (f_out - centre)
        x0 = y0 + centre
        return (x0, *rest) if rest else x0


@torch.no_grad()
def sidechain_reverse_loop(
    denoiser: SideChainEDM,
    x_init: torch.Tensor,             # [B, L, A, 3] GLOBAL, at sigma_schedule[0]
    sigma_schedule: torch.Tensor,     # [n_steps + 1], descending, ending at 0
    ca_coords: torch.Tensor,
    *args,
    atom_mask: Optional[torch.Tensor] = None,
    return_aux: bool = False,
    **kwargs,
):
    """A2: deterministic Euler reverse loop that CARRIES the estimate forward.

    Each step's input is the previous step's output, not a fresh draw from the
    template. That is the whole difference from what the sampler does today: it
    calls S_phi once per backbone step and re-initialises from the template every
    time, so N calls buy exactly one call's worth of refinement.

    Deterministic (no churn / stochastic sampling), matching the correctness-first
    posture of `cogenerate`. Quality tuning is deliberately out of scope.
    """
    x = x_init
    aux = ()
    for i in range(len(sigma_schedule) - 1):
        s_cur = sigma_schedule[i]
        s_next = sigma_schedule[i + 1]
        b = x.shape[0]
        sig = s_cur.expand(b) if s_cur.dim() == 0 else s_cur
        out = denoiser.denoise(x, sig, ca_coords, *args, **kwargs)
        if isinstance(out, tuple):
            x0, aux = out[0], out[1:]
        else:
            x0 = out
        # Euler step along dx/dsigma = (x - x0) / sigma.
        d = (x - x0) / s_cur.clamp_min(1e-8)
        x = x + (s_next - s_cur) * d
        if atom_mask is not None:
            x = x * atom_mask[..., None].to(x.dtype)
    # The caller needs atom_feats (h_res' pooling) and bb_feats (q_direct) from the
    # LAST denoise, not a fresh extra call at sigma=0.
    return (x, aux) if return_aux else x


def noise_sidechains(
    x_clean_local: torch.Tensor,      # [B, L, A, 3] residue-LOCAL clean geometry
    sigma: torch.Tensor,              # [B]
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """x_sigma = x_0 + sigma * eps, applied in the residue frame.

    The noise is isotropic in the LOCAL frame on purpose: that is where the data
    distribution this sigma is calibrated against lives. Adding it in the global
    frame would be numerically identical (a rotation of white noise is white
    noise) but would invite the reader to assume sigma is on the global scale,
    which is the confusion this whole module exists to remove.
    """
    s = sigma.view(-1, *([1] * (x_clean_local.dim() - 1)))
    eps = torch.randn(
        x_clean_local.shape, device=x_clean_local.device,
        dtype=x_clean_local.dtype, generator=generator,
    )
    return x_clean_local + s * eps
