"""
Training-side noise sampler + one-step training forward for PXDesign-d.

PXDesign's released `pxdesign/model/generator.py` ships only the inference noise
schedule (`InferenceNoiseScheduler`) and the multi-step reverse diffusion loop.
For training we need:

1. A log-normal noise *sampler* (EDM-style) — identical to Protenix's
   `TrainingNoiseSampler`. We re-import it from protenix to stay in lockstep.
2. A one-step training call: noise a GT coordinate, denoise once, return both
   the prediction and the sampled sigma so the loss can σ-mask LDDT/distogram.

Note: `protenix.model.generator.sample_diffusion_training` requires `pair_z`,
`p_lm`, `c_l` — Protenix-only features that PXDesign doesn't compute. PXDesign's
inference call (see `pxdesign/model/pxdesign.py:168`) omits them. We follow that
convention here.
"""
from typing import Any, Callable, Optional

import torch
from protenix.model.generator import TrainingNoiseSampler  # noqa: F401 — re-exported
from protenix.model.utils import centre_random_augmentation


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    precomputed_input: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One denoising step for training.

    For each of N_sample augmentations of the GT structure we:
      1. centre-random-augment the GT coords
      2. draw an independent σ from the EDM log-normal sampler
      3. add Gaussian noise of scale σ
      4. call denoise_net once

    Args:
        noise_sampler: EDM log-normal sampler producing per-sample σ.
        denoise_net: `DiffusionModule` (or any callable with PXDesign's inference
            denoise-step signature: x_noisy, t_hat_noise_level, input_feature_dict,
            s_inputs, s_trunk, z_trunk, chunk_size, inplace_safe).
        label_dict: must contain
            "coordinate":      [..., N_atom, 3]   ground-truth coords
            "coordinate_mask": [..., N_atom]      valid-atom mask
        input_feature_dict: PXDesign feature dict (restype, conditional_templ, ...).
        s_inputs / s_trunk / z_trunk: from `DesignConditionEmbedder`.
        N_sample: how many independent (rotation, noise) draws per macro-batch
            item. The PXDesign report's "diffusion batch size 8" maps to N_sample=8.
        precomputed_input: `(x_gt_aug, sigma, x_input)` supplied by the caller.
            Skips steps 1-3 and denoises `x_input` directly. Stage III uses this
            to feed the first backbone pass's prediction into the refinement pass
            in the same coordinate frame, with an explicit caller-provided fixed
            refinement condition.
            None keeps the ordinary independently sampled training behaviour.

    Returns:
        x_gt_aug   [..., N_sample, N_atom, 3]   rotated/centred GT coords
        x_denoised [..., N_sample, N_atom, 3]   model prediction
        sigma      [..., N_sample]              sampled σ, or the supplied condition
        x_input    [..., N_sample, N_atom, 3]   what the net actually received;
            this is `x_noisy` for an ordinary sampled pass, but may be a previous
            pass's prediction when `precomputed_input` is used
    """
    batch_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype

    if precomputed_input is not None:
        # Stage III refinement starts from B_pre's prediction, not from another
        # noisy draw and not from B_pre's original noisy input. Keep the first
        # pass's augmented target and sigma so the coordinate frame and diffusion
        # conditioning remain aligned. Deliberately draw nothing here.
        x_gt_aug, sigma, x_input = precomputed_input
    else:
        # 1. Augment: random rotation + translation of GT, then centre on the masked CoM.
        x_gt_aug = centre_random_augmentation(
            x_input_coords=label_dict["coordinate"],
            N_sample=N_sample,
            mask=label_dict["coordinate_mask"],
        ).to(dtype)  # [..., N_sample, N_atom, 3]

        # 2. Sample σ per (batch, sample) from EDM log-normal.
        sigma = noise_sampler(size=(*batch_shape, N_sample), device=device).to(dtype)

        # 3. Add Gaussian noise of scale σ.
        noise = torch.randn_like(x_gt_aug, dtype=dtype) * sigma[..., None, None]
        x_input = x_gt_aug + noise

    # 4. Denoise.  pair_z / p_lm / c_l are None — DiffusionModule computes them.
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_input,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=None,
            p_lm=None,
            c_l=None,
            inplace_safe=inplace_safe,
        )
    else:
        chunks = []
        n_chunks = (N_sample + diffusion_chunk_size - 1) // diffusion_chunk_size
        for i in range(n_chunks):
            lo, hi = i * diffusion_chunk_size, min((i + 1) * diffusion_chunk_size, N_sample)
            x_chunk = denoise_net(
                x_noisy=x_input[..., lo:hi, :, :],
                t_hat_noise_level=sigma[..., lo:hi],
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=None,
                p_lm=None,
                c_l=None,
                inplace_safe=inplace_safe,
            )
            chunks.append(x_chunk)
        x_denoised = torch.cat(chunks, dim=-3)

    return x_gt_aug, x_denoised, sigma, x_input
