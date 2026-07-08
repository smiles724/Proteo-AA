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
import logging
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
from pxdesign_train.sidechain.module import SideChainModule
from pxdesign_train.sidechain.feedback import HResFeedback
from pxdesign_train.sidechain.init import gaussian_init_local
from pxdesign_train.sidechain.coevolution import HResInjector
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.sidechain.physical import physical_loss


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
            use_time = bool(getattr(res_cfg, "use_time_embedding", True)) if res_cfg is not None else True
            # input_source selects which per-token representation the AA head reads:
            #   "s_inputs"          — outer conditioning embedding (449), structure-blind, sigma-free (default)
            #   "diffusion_internal"— a_token captured from DiffusionModule.layernorm_a
            #                         (c_token), structure- and sigma-aware
            self.aa_input_source = (
                getattr(res_cfg, "input_source", "s_inputs") if res_cfg is not None else "s_inputs"
            )
            # diffusion_internal hardening knobs:
            #   trunk_grad_scale: scale AA-loss gradient flowing into the coord
            #     trunk (1.0 = full, 0.0 = stop-grad → coords untouched by AA).
            #   internal_reduce: how to collapse the N_sample axis of a_token
            #     ("mean" or "low_sigma" = pick the least-noisy sample).
            self.aa_trunk_grad_scale = (
                float(getattr(res_cfg, "trunk_grad_scale", 1.0)) if res_cfg is not None else 1.0
            )
            self.aa_internal_reduce = (
                getattr(res_cfg, "internal_reduce", "mean") if res_cfg is not None else "mean"
            )
            if self.aa_input_source == "diffusion_internal":
                # a_token dim = the DiffusionModule's own c_token (read from the
                # live module — it differs from the global c_token=384).
                c_in = getattr(self.diffusion_module, "c_token", None) or getattr(
                    configs, "c_token", 768
                )
            else:
                c_in = getattr(configs, "c_s_inputs", None)
                if c_in is None:
                    c_in = getattr(getattr(configs, "model", object()), "c_s_inputs", 449)
            self.design_residue_type_head = DesignResidueTypeHead(
                c_s=c_in, no_bins=vocab_size, use_time=use_time,
            )
            # Capture the internal per-token representation via a forward
            # hook — NO edit to the Protenix/PXDesign submodule source.
            self._a_token_cache = None
            if self.aa_input_source == "diffusion_internal":
                def _capture_a_token(_module, _inp, out):
                    self._a_token_cache = out
                self.diffusion_module.layernorm_a.register_forward_hook(_capture_a_token)

        # ---- Side-Chain Module S_phi (Stage II-A) ----
        self.enable_sidechain = getattr(configs, "enable_sidechain", False)
        if self.enable_sidechain:
            assert self.enable_residue_type_head, (
                "enable_sidechain requires enable_residue_type_head (S_phi conditions "
                "on the residue-type logits and reads the same h_res=a_token)."
            )
            sc_cfg = getattr(configs, "sidechain", None)
            # h_res dim == the representation the AA head reads (a_token c_token
            # for diffusion_internal, else s_inputs dim).
            self.sc_c_res = c_in
            self.sc_init_sigma = float(getattr(sc_cfg, "init_sigma", 1.0)) if sc_cfg is not None else 1.0
            self.sc_detach_feedback = bool(getattr(sc_cfg, "detach_feedback", False)) if sc_cfg is not None else False
            # Stage III routing: supervise coord loss only where predicted type
            # matches GT; mismatched residues fall back to physical loss.
            self.sc_route_by_type = bool(getattr(sc_cfg, "route_by_type", False)) if sc_cfg is not None else False
            sc_grad_scale = float(getattr(sc_cfg, "trunk_grad_scale", 1.0)) if sc_cfg is not None else 1.0
            c_atom = int(getattr(sc_cfg, "c_atom", 128)) if sc_cfg is not None else 128
            self.sidechain_module = SideChainModule(
                c_res=self.sc_c_res, c_atom=c_atom, n_type=vocab_size,
                trunk_grad_scale=sc_grad_scale,
            )
            self.sidechain_feedback = HResFeedback(c_atom=c_atom, c_res=self.sc_c_res)

            # ---- Cycle closure (Stage II-B co-evolution) ----
            # Reuse B_theta for a side-chain-informed refinement pass by injecting
            # h_res' into the (currently-zero) token trunk s_trunk (dim c_s).
            self.enable_coevolution = getattr(configs, "enable_coevolution", False)
            if self.enable_coevolution:
                c_trunk = int(getattr(configs, "c_s", 384))
                self.hres_injector = HResInjector(c_hres=self.sc_c_res, c_trunk=c_trunk)
                self.w_bb_post = float(getattr(sc_cfg, "weight_bb_post", 1.0)) if sc_cfg is not None else 1.0
                self.w_aa_post = float(getattr(sc_cfg, "weight_aa_post", 1.0)) if sc_cfg is not None else 1.0
        else:
            self.enable_coevolution = False

    def _reduce_a_token(self, a: torch.Tensor, sigma: Optional[torch.Tensor]) -> torch.Tensor:
        """Collapse a_token's N_sample axis (dim -3).

        a: [..., N_sample, N_token, c_token]; sigma: [..., N_sample].
        "mean" averages all noise draws; "low_sigma" picks the least-noisy
        (smallest-sigma) draw, giving a cleaner, less sigma-mixed signal.
        """
        if self.aa_internal_reduce == "low_sigma" and sigma is not None:
            idx = sigma.argmin(dim=-1)  # [...]
            idx_e = idx[..., None, None, None].expand(*idx.shape, 1, a.shape[-2], a.shape[-1])
            return a.gather(dim=-3, index=idx_e).squeeze(-3)
        return a.mean(dim=-3)

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
            # h_res candidate for the future side-chain / h_res interface.
            # Overwritten below with the actual representation the AA head reads
            # (structure-aware a_token under diffusion_internal); s_inputs is only
            # the fallback when the residue-type head is disabled.
            "h_res_candidate": s_inputs,
            "z_pair_candidate": z,
        }

        # 3. Distogram on conditioning pair z, when enabled.
        if self.enable_distogram_head:
            out["distogram_logits"] = self.design_distogram_head(z)
        if self.enable_residue_type_head:
            aa_t = input_feature_dict.get("aa_t")
            token_repr = s_inputs
            if self.aa_input_source == "diffusion_internal":
                a = self._a_token_cache  # [..., N_sample, N_token, c_token]
                if a is None:
                    # Hook never fired (e.g. checkpointing edge case) — fall back.
                    logging.getLogger(__name__).warning(
                        "diffusion_internal: a_token not captured; falling back to s_inputs"
                    )
                else:
                    token_repr = self._reduce_a_token(a, sigma).to(s_inputs.dtype)
                    # Gradient control: scale the AA gradient flowing into the
                    # shared coord trunk (protects coordinates).
                    g = self.aa_trunk_grad_scale
                    if g != 1.0:
                        token_repr = g * token_repr + (1.0 - g) * token_repr.detach()
                    out["a_token_shape"] = torch.tensor(list(a.shape))
            # The representation the AA head reads IS the h_res candidate:
            # structure-aware a_token for diffusion_internal, s_inputs for the
            # baseline. This is what a future h_res module should consume.
            out["h_res_candidate"] = token_repr
            out["aa_logits"] = self.design_residue_type_head(token_repr, aa_t=aa_t)

        # 4. Side-Chain Module (Stage II-A): one-step local-frame decode + feedback.
        if self.enable_sidechain and "sc_atom_name_ids" in input_feature_dict:
            h_res = out["h_res_candidate"]          # [N_token, c_res] or [B, N_token, c_res]
            aa_logits = out["aa_logits"]            # matching leading dims
            squeeze = h_res.dim() == 2              # batch=1 collapsed upstream
            if squeeze:
                h_res = h_res.unsqueeze(0)
                aa_logits = aa_logits.unsqueeze(0)
            sc_ids = input_feature_dict["sc_atom_name_ids"]
            sc_mask = input_feature_dict["sc_atom_mask"]
            if sc_ids.dim() == 2:                   # add batch dim
                sc_ids = sc_ids.unsqueeze(0)
                sc_mask = sc_mask.unsqueeze(0)
            B = h_res.shape[0]
            if sc_ids.shape[0] != B:
                sc_ids = sc_ids.expand(B, -1, -1)
                sc_mask = sc_mask.expand(B, -1, -1)
            sc_ids = sc_ids.to(h_res.device).long()
            sc_mask = sc_mask.to(h_res.device).bool()
            noisy = gaussian_init_local(
                sc_mask.detach().cpu(), sigma=self.sc_init_sigma
            ).to(h_res.device).to(h_res.dtype)
            t = torch.ones(B, device=h_res.device)
            ca = input_feature_dict.get("sc_frame_t")
            if ca is not None:
                ca = ca.to(h_res.device).float()
                if ca.dim() == 2:
                    ca = ca.unsqueeze(0)
                if ca.shape[0] != B:
                    ca = ca.expand(B, -1, -1)
            y0_local, atom_feats = self.sidechain_module(
                h_res, aa_logits, sc_ids, sc_mask, noisy, t, ca_coords=ca,
            )
            h_res_prime = self.sidechain_feedback(
                atom_feats, sc_mask, h_res, detach=self.sc_detach_feedback,
            )
            if squeeze:                             # restore the collapsed batch dim
                y0_local = y0_local.squeeze(0)
                sc_mask = sc_mask.squeeze(0)
                h_res_prime = h_res_prime.squeeze(0)
            out["sc_pred_local"] = y0_local
            out["sc_atom_mask"] = sc_mask
            out["h_res_prime"] = h_res_prime

            # Physical regularization (clash + contact) on predicted GLOBAL
            # side-chain coords: local -> global via the residue frames, then a
            # coordinate-only physical loss (no ideal-geometry tables needed).
            fR = input_feature_dict.get("sc_frame_R")
            ft = input_feature_dict.get("sc_frame_t")
            bb = input_feature_dict.get("sc_bb_coords")
            if fR is not None and ft is not None and bb is not None:
                y_l = y0_local if y0_local.dim() == 4 else y0_local.unsqueeze(0)  # [B,L,A,3]
                fR = fR.to(y_l.device).float()
                ft = ft.to(y_l.device).float()
                bb = bb.to(y_l.device).float()
                if fR.dim() == 3:
                    fR, ft, bb = fR.unsqueeze(0), ft.unsqueeze(0), bb.unsqueeze(0)
                B_, L_, A_ = y_l.shape[0], y_l.shape[1], y_l.shape[2]
                y_g = to_global(y_l.float(), fR, ft)                     # [B,L,A,3]
                m = sc_mask if sc_mask.dim() == 3 else sc_mask.unsqueeze(0)
                phys = physical_loss(
                    y_g.reshape(B_, L_ * A_, 3),
                    backbone_coords=bb.reshape(B_, L_ * bb.shape[2], 3),
                    valid_mask=m.reshape(B_, L_ * A_),
                )
                out["sc_phys_val"] = phys["total"]

            # Stage III type routing: coord loss only where predicted AA == GT.
            if self.sc_route_by_type:
                aa_clean = input_feature_dict.get("aa_clean")
                if aa_clean is not None:
                    pred = aa_logits.argmax(dim=-1)
                    if pred.dim() > 1:
                        pred = pred[0]
                    out["sc_type_match"] = (pred == aa_clean.to(pred.device))

        # 5. Cycle closure (Stage II-B): reuse B_theta to refine backbone/type
        #    using the side-chain-informed h_res'. h_res' is injected into the
        #    (zero) token trunk s_trunk, then the backbone denoise is re-run.
        if getattr(self, "enable_coevolution", False) and "h_res_prime" in out:
            h_res_prime = out["h_res_prime"]
            s_trunk_refine = s + self.hres_injector(h_res_prime).to(s.dtype)
            x_gt_aug_post, x_denoised_post, sigma_post = sample_diffusion_training(
                noise_sampler=self.training_noise_sampler,
                denoise_net=self.diffusion_module,
                label_dict=label_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk_refine,
                z_trunk=z,
                N_sample=N_sample,
            )
            out["post_pred_coordinate"] = x_denoised_post
            out["post_gt_coordinate_aug"] = x_gt_aug_post
            out["post_sigma"] = sigma_post
            # Refined AA logits from the side-chain-aware refinement pass.
            if self.enable_residue_type_head:
                a_post = self._a_token_cache
                if a_post is not None:
                    tok_post = self._reduce_a_token(a_post, sigma_post).to(s_inputs.dtype)
                    g = self.aa_trunk_grad_scale
                    if g != 1.0:
                        tok_post = g * tok_post + (1.0 - g) * tok_post.detach()
                    out["post_aa_logits"] = self.design_residue_type_head(
                        tok_post, aa_t=input_feature_dict.get("aa_t")
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
        if getattr(self, "aa_input_source", "s_inputs") == "diffusion_internal":
            raise NotImplementedError(
                "predict_aa (AA-only sampler path) requires input_source='s_inputs'; "
                "diffusion_internal needs a coordinate forward to populate a_token."
            )
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
