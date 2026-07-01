"""
ProtenixDesignTrain — adds a training forward to PXDesign's `ProtenixDesign`.

Released [`pxdesign/model/pxdesign.py`](../../PXDesign/pxdesign/model/pxdesign.py)
hard-asserts `mode == "inference"` and ships no training plumbing. We subclass
it here so the inference path is unchanged but a `mode="train"` branch becomes
available, calling `sample_diffusion_training` once per macro-batch item.

The released config block `model.design_distogram_head` (c_z=128, no_bins=64)
is also enabled here — see `heads.DesignDistogramHead`. The diffusion-token
variant (`design_diffusion_distogram`, c_z=768) needs a DiffusionModule hook
to surface per-step token embeddings; that wiring is deferred (see heads.py).
"""
from typing import Any, Optional

import torch

from pxdesign.model.pxdesign import ProtenixDesign
from protenix.model.protenix import update_input_feature_dict

from pxdesign_train.generator import TrainingNoiseSampler, sample_diffusion_training
from pxdesign_train.heads import (
    DesignDiffusionDistogramHead,
    DesignDistogramHead,
    DesignResidueTypeHead,
)


class ProtenixDesignTrain(ProtenixDesign):
    """`ProtenixDesign` + training forward + distogram heads.

    Args (added on top of parent's `configs`):
        configs.training_noise_sampler: dict for `TrainingNoiseSampler`
            (p_mean, p_std, sigma_data). Defaults match Protenix's training.
        configs.training.diffusion_batch_size: N_sample for train-time denoising.
            Report value: 8.
        configs.enable_distogram_head: whether to instantiate the conditioning-z
            distogram head.
        configs.enable_diffusion_distogram_head: whether to instantiate the
            c_token=768 head (params only; not yet wired into forward).
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)
        ns_cfg = getattr(configs, "training_noise_sampler", None) or {
            "p_mean": -1.2,
            "p_std": 1.5,
            "sigma_data": configs.sigma_data,
        }
        # configs from Protenix's loader are namespace-like; use dict access.
        if not isinstance(ns_cfg, dict):
            ns_cfg = {k: getattr(ns_cfg, k) for k in ("p_mean", "p_std", "sigma_data")}
        self.training_noise_sampler = TrainingNoiseSampler(**ns_cfg)

        self.enable_distogram_head = getattr(configs, "enable_distogram_head", True)
        self.enable_diffusion_distogram_head = getattr(
            configs, "enable_diffusion_distogram_head", False
        )
        self.enable_residue_type_head = getattr(
            configs, "enable_residue_type_head", False
        )

        if self.enable_distogram_head:
            self.design_distogram_head = DesignDistogramHead(
                **configs.model.design_distogram_head
            )
        if self.enable_diffusion_distogram_head:
            # Parameters only — not yet wired into the train forward.
            self.design_diffusion_distogram = DesignDiffusionDistogramHead(
                c_token=configs.model.design_diffusion_distogram.c_z,
                no_bins=configs.model.design_diffusion_distogram.no_bins,
            )
        if self.enable_residue_type_head:
            res_cfg = getattr(configs, "residue_type", None)
            vocab_size = getattr(res_cfg, "vocab_size", 20) if res_cfg is not None else 20
            # The AA head reads `s_inputs` (the real per-token conditioning
            # embedding). PXDesign's `s_trunk`/`s` is a zero placeholder, so we
            # size the head to c_s_inputs (449), NOT c_s (384).
            c_in = getattr(configs, "c_s_inputs", None)
            if c_in is None:
                c_in = getattr(getattr(configs, "model", object()), "c_s_inputs", 449)
            use_time = bool(getattr(res_cfg, "use_time_embedding", True)) if res_cfg is not None else True
            self.design_residue_type_head = DesignResidueTypeHead(
                c_s=c_in,
                no_bins=vocab_size,
                use_time=use_time,
            )

    def _train_forward(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_sample: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        # Default to the report's diffusion batch size.
        if N_sample is None:
            N_sample = getattr(
                getattr(self.configs, "training", object()),
                "diffusion_batch_size",
                8,
            )

        # Compute relp, d_lm, v_lm, pad_info needed by DiffusionModule / AtomAttentionEncoder.
        input_feature_dict = self.diffusion_module.diffusion_conditioning.relpe.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        s_inputs, s, z = self.get_condition_embedding(
            input_feature_dict=input_feature_dict,
            chunk_size=chunk_size,
        )

        # 2. One-step denoising under EDM training noise.
        x_gt_aug, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=self.training_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
        )

        out = {
            "x_gt_aug": x_gt_aug,
            "x_denoised": x_denoised,
            "sigma": sigma,
            # `s` (s_trunk) is a zero placeholder; the real per-token state is
            # s_inputs. Expose that as the h_res candidate for the future
            # side-chain interface.
            "h_res_candidate": s_inputs,
            "z_pair_candidate": z,
        }

        # 3. Distogram on conditioning pair z, when enabled.
        if self.enable_distogram_head:
            out["distogram_logits"] = self.design_distogram_head(z)
        if self.enable_residue_type_head:
            # Read s_inputs (NOT the zero s_trunk), conditioned on the discrete
            # masked-diffusion time aa_t when present.
            out["aa_logits"] = self.design_residue_type_head(
                s_inputs, aa_t=input_feature_dict.get("aa_t")
            )

        return out

    def predict_aa(
        self,
        input_feature_dict: dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Lightweight AA-only forward for inference-time masked-diffusion
        sampling: computes the conditioning embedding and the residue-type
        logits WITHOUT running the (expensive) coordinate diffusion. Returns
        ``aa_logits`` of shape ``[..., N_token, no_bins]``.
        """
        assert self.enable_residue_type_head, "residue_type head is not enabled"
        input_feature_dict = self.diffusion_module.diffusion_conditioning.relpe.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)
        s_inputs, _s, _z = self.get_condition_embedding(
            input_feature_dict=input_feature_dict, chunk_size=chunk_size
        )
        return self.design_residue_type_head(
            s_inputs, aa_t=input_feature_dict.get("aa_t")
        )

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: Optional[dict[str, Any]] = None,
        mode: str = "inference",
    ) -> dict[str, torch.Tensor]:
        if mode == "train":
            assert label_dict is not None, (
                "mode='train' requires a label_dict with 'coordinate' and 'coordinate_mask'."
            )
            chunk_size = self.configs.infer_setting.chunk_size
            return self._train_forward(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                chunk_size=chunk_size,
            )

        # Delegate to the parent's inference path unchanged.
        return super().forward(input_feature_dict=input_feature_dict, mode=mode)
