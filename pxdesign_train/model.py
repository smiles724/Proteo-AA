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
            # M2: only emit a supervised AA-refinement head (post_aa_logits) when
            # the side-chain atom set is instantiated from PREDICTED type. Under
            # GT-type teacher-forcing, GT atom composition (atom names/counts)
            # uniquely encodes residue identity and would leak through h_res' into
            # post_aa_logits — an identity->identity shortcut. Default False keeps
            # the leak closed by simply not supervising post_aa in that regime.
            self.sc_predicted_mask = bool(getattr(sc_cfg, "predicted_mask", False)) if sc_cfg is not None else False
            # Per-sigma alignment: S_phi reads per-sigma h_res/aa_logits/sigma
            # (flattened [B*N_sample, L, C]) rather than a reduced h_res. Warmup
            # can turn this off for the single-baseline path.
            self.sc_per_sigma = bool(getattr(sc_cfg, "per_sigma", True)) if sc_cfg is not None else True
            # Paper Stage II-B: build side-chain frames from the PREDICTED backbone
            # (x_denoised) + a stop-grad global pseudo-target aux loss. Warmup uses
            # GT frames (predicted_frame=False).
            self.sc_predicted_frame = bool(getattr(sc_cfg, "predicted_frame", True)) if sc_cfg is not None else True
            self.sc_weight_global = float(getattr(sc_cfg, "weight_sc_global", 0.5)) if sc_cfg is not None else 0.5
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
            a_full = None  # per-sample (per-sigma) representation, if available
            if self.aa_input_source == "diffusion_internal":
                a = self._a_token_cache  # [..., N_sample, N_token, c_token]
                if a is None:
                    # Hook never fired (e.g. checkpointing edge case) — fall back.
                    logging.getLogger(__name__).warning(
                        "diffusion_internal: a_token not captured; falling back to s_inputs"
                    )
                else:
                    # Gradient control: scale the AA gradient flowing into the
                    # shared coord trunk (protects coordinates). Applied to BOTH
                    # the reduced repr (for h_res/S_phi) and the per-sample repr
                    # (for the AA loss).
                    g = self.aa_trunk_grad_scale

                    def _gscale(x):
                        return x if g == 1.0 else g * x + (1.0 - g) * x.detach()

                    # Reduced repr: what h_res / S_phi consume (one coherent
                    # representation, mean or low-sigma per `internal_reduce`).
                    token_repr = _gscale(self._reduce_a_token(a, sigma).to(s_inputs.dtype))
                    # Per-sample repr: KEEP the N_sample axis so the AA loss is
                    # computed per noise draw (per sigma) and averaged — no
                    # reduce-then-predict blur, and it matches how cogenerate
                    # queries the head across the sigma trajectory at inference.
                    a_full = _gscale(a.to(s_inputs.dtype))
                    # Per-sigma backbone state exposed for the side-chain / cycle
                    # path: h_res_sigma [.., N_sample, N_token, C] aligns row-for-
                    # row with out["sigma"] [.., N_sample] and out["aa_logits"].
                    out["h_res_sigma"] = a_full
                    out["a_token_shape"] = torch.tensor(list(a.shape))
            # The representation h_res / S_phi read (one coherent state). For the
            # s_inputs baseline this is just s_inputs (no per-sample axis exists).
            out["h_res_candidate"] = token_repr
            # Reduced logits for S_phi / type routing (single [..., N_token, 20]).
            out["aa_logits_reduced"] = self.design_residue_type_head(token_repr, aa_t=aa_t)
            # AA-loss logits: per-sample under diffusion_internal ([..., N_sample,
            # N_token, 20]); the reduced logits for the sigma-free s_inputs baseline.
            out["aa_logits"] = (
                self.design_residue_type_head(a_full, aa_t=aa_t)
                if a_full is not None else out["aa_logits_reduced"]
            )

        # 4. Side-Chain Module (Stage II-A): one-step global-coordinate decode + feedback.
        if self.enable_sidechain and "sc_atom_name_ids" in input_feature_dict:
            # Per-sigma (Stage II-B main line) vs single reduced-h_res (warmup).
            # In per-sigma mode each S_phi batch row corresponds to ONE specific
            # sigma: we flatten [.., N_sample, L, C] -> [B*N_sample, L, C] and feed
            # the matching per-sample aa_logits + real sigma. The per-token side-
            # chain tensors (ids / mask / frames) are tiled to the flattened batch
            # by the existing `expand(B, ...)` logic below. NOT a reduce/low-sigma
            # mixed representation.
            use_per_sigma = self.sc_per_sigma and out.get("h_res_sigma") is not None
            sigma_flat = None
            if use_per_sigma:
                hs = out["h_res_sigma"]             # [.., N_sample, L, C]
                al = out["aa_logits"]               # [.., N_sample, L, 20] (per-sigma)
                sample_shape = hs.shape[:-3]
                n_sample = hs.shape[-3]
                L_, C_ = hs.shape[-2], hs.shape[-1]
                h_res = hs.reshape(-1, L_, C_)       # [B*N_sample, L, C]
                aa_logits = al.reshape(-1, L_, al.shape[-1])
                sigma_flat = out["sigma"].reshape(-1)  # [B*N_sample], row-aligned
                squeeze = False

                def _tile_per_sigma(x: torch.Tensor, trailing_ndim: int) -> torch.Tensor:
                    """Tile per-item side-chain tensors over the sigma axis.

                    h_res/aa_logits are flattened in row-major order from
                    [..., N_sample, L, C] to [prod(...)*N_sample, L, C]. This helper
                    applies the same layout to tensors that only carry the per-item
                    leading dims, e.g. [B, L, A] -> [B*N_sample, L, A].
                    """
                    trail = x.shape[-trailing_ndim:]
                    flat_B = h_res.shape[0]
                    if x.dim() >= trailing_ndim + 1 and x.shape[:-trailing_ndim] == (flat_B,):
                        return x
                    if x.dim() == trailing_ndim:
                        base = x.reshape(*((1,) * len(sample_shape)), *trail)
                    else:
                        base = x
                    if base.shape[:-trailing_ndim] != sample_shape:
                        base = base.expand(*sample_shape, *trail)
                    expanded = base.unsqueeze(len(sample_shape)).expand(
                        *sample_shape, n_sample, *trail
                    )
                    return expanded.reshape(flat_B, *trail)
            else:
                h_res = out["h_res_candidate"]      # [N_token, c_res] or [B, N_token, c_res]
                # S_phi conditions on the REDUCED AA distribution (warmup baseline).
                aa_logits = out["aa_logits_reduced"]
                squeeze = h_res.dim() == 2          # batch=1 collapsed upstream
                if squeeze:
                    h_res = h_res.unsqueeze(0)
                    aa_logits = aa_logits.unsqueeze(0)
            sc_ids = input_feature_dict["sc_atom_name_ids"]
            sc_mask = input_feature_dict["sc_atom_mask"]
            # Stage III predicted-mask branch: instantiate the side-chain atom set
            # from the PREDICTED residue type (argmax of the reduced logits) instead
            # of the GT type. This is what makes post_aa safe to supervise (M2): the
            # atom composition no longer carries GT identity. Matched residues still
            # align with sc_gt_local (same atoms/order); mismatched ones are routed
            # to physical-only via sc_type_match below.
            if getattr(self, "sc_predicted_mask", False):
                from pxdesign_train.sidechain.instantiate import instantiate_from_type_indices
                ptype = out["aa_logits_reduced"].argmax(dim=-1)   # [L] or [B, L]
                if ptype.dim() > 1:
                    # Batch>1 not yet supported here: predicted-type instantiation +
                    # routing use item 0. Fine for the current batch_size=1 trainer;
                    # warn so it isn't a silent bug if macro-batch grows (see
                    # docs/method_status.md).
                    if ptype.shape[0] > 1 and not getattr(self, "_warned_predmask_batch", False):
                        logging.getLogger(__name__).warning(
                            "predicted_mask: batch>1 detected; per-item atom-set "
                            "instantiation/routing not implemented — using item 0."
                        )
                        self._warned_predmask_batch = True
                    ptype = ptype[0]
                pids, pmask = instantiate_from_type_indices(ptype)
                sc_ids = pids.to(sc_ids.device)
                sc_mask = pmask.to(sc_mask.device)
            if use_per_sigma:
                sc_ids = _tile_per_sigma(sc_ids, trailing_ndim=2)
                sc_mask = _tile_per_sigma(sc_mask, trailing_ndim=2)
            else:
                if sc_ids.dim() == 2:                   # add batch dim
                    sc_ids = sc_ids.unsqueeze(0)
                    sc_mask = sc_mask.unsqueeze(0)
                B = h_res.shape[0]
                if sc_ids.shape[0] != B:
                    sc_ids = sc_ids.expand(B, -1, -1)
                    sc_mask = sc_mask.expand(B, -1, -1)
            sc_ids = sc_ids.to(h_res.device).long()
            sc_mask = sc_mask.to(h_res.device).bool()
            B = h_res.shape[0]
            noisy_init = gaussian_init_local(
                sc_mask.detach().cpu(), sigma=self.sc_init_sigma
            ).to(h_res.device).to(h_res.dtype)
            # Sigma-embedding for S_phi's time input: the REAL per-sample noise
            # level (EDM c_noise = 0.25*ln sigma) when per-sigma; a constant for
            # the reduced warmup baseline (no single sigma to attach).
            if sigma_flat is not None:
                t = 0.25 * sigma_flat.to(h_res.device).clamp_min(1e-4).log()
            else:
                t = torch.ones(B, device=h_res.device)
            # Paper Stage II-B: PREDICTED-backbone frames F_hat from x_denoised
            # (x_hat_0). Falls back to GT frames when disabled / unavailable
            # (Stage II-A warmup). R_hat/t_hat are [B*N_sample, L, 3, 3]/[.., L, 3].
            R_hat = t_hat = bb_pred = None
            if getattr(self, "sc_predicted_frame", False) and use_per_sigma:
                bb_idx = input_feature_dict.get("sc_bb_atom_idx")
                xden = out.get("x_denoised")
                if bb_idx is not None and xden is not None:
                    from pxdesign_train.sidechain.frames import frames_from_backbone_index
                    bb_idx = bb_idx.to(h_res.device).long()
                    xden_flat = xden.reshape(-1, xden.shape[-2], xden.shape[-1]).to(h_res.device).float()
                    R_hat, t_hat, _fvalid = frames_from_backbone_index(xden_flat, bb_idx)
                    bb_pred = xden_flat[:, bb_idx.clamp_min(0), :]   # [B*N_sample, L, 3, 3] (N,CA,C)

            # Active frame for S_phi context, global initialization, physical
            # regularization, and coordinate supervision. Under Stage II-B this
            # is the predicted frame F_hat; warmup falls back to GT frames.
            fR, ft, bb = R_hat, t_hat, bb_pred
            if fR is None:
                fR = input_feature_dict.get("sc_frame_R")
                ft = input_feature_dict.get("sc_frame_t")
                bb = input_feature_dict.get("sc_bb_coords")
                if fR is not None and ft is not None and bb is not None:
                    fR = fR.to(h_res.device).float()
                    ft = ft.to(h_res.device).float()
                    bb = bb.to(h_res.device).float()
                    if use_per_sigma:
                        fR = _tile_per_sigma(fR, trailing_ndim=3)
                        ft = _tile_per_sigma(ft, trailing_ndim=2)
                        bb = _tile_per_sigma(bb, trailing_ndim=3)
                    elif fR.dim() == 3:
                        fR, ft, bb = fR.unsqueeze(0), ft.unsqueeze(0), bb.unsqueeze(0)
                    if fR.shape[0] != B:
                        fR = fR.expand(B, -1, -1, -1)
                        ft = ft.expand(B, -1, -1)
                        bb = bb.expand(B, -1, -1, -1)

            ca = ft
            if ca is None:
                ca = input_feature_dict.get("sc_frame_t")
            if ca is not None and ca is not ft:
                ca = ca.to(h_res.device).float()
                if use_per_sigma:
                    ca = _tile_per_sigma(ca, trailing_ndim=2)
                else:
                    if ca.dim() == 2:
                        ca = ca.unsqueeze(0)
                    if ca.shape[0] != B:
                        ca = ca.expand(B, -1, -1)

            # S_phi consumes global coordinates in the same active frame used for
            # supervision. Detaching the frame here keeps the side-chain coord
            # objective from nudging the predicted backbone frame through the
            # initialization path.
            if fR is not None and ft is not None:
                noisy = to_global(noisy_init.float(), fR.detach(), ft.detach()).to(h_res.dtype)
            else:
                noisy = noisy_init

            ca_for_module = ca.detach() if ca is not None else None
            y0_global, atom_feats = self.sidechain_module(
                h_res, aa_logits, sc_ids, sc_mask, noisy, t, ca_coords=ca_for_module,
            )
            h_res_prime = self.sidechain_feedback(
                atom_feats, sc_mask, h_res, detach=self.sc_detach_feedback,
            )
            if squeeze:                             # restore the collapsed batch dim
                y0_global = y0_global.squeeze(0)
                sc_mask = sc_mask.squeeze(0)
                h_res_prime = h_res_prime.squeeze(0)
                if fR is not None:
                    fR = fR.squeeze(0)
                    ft = ft.squeeze(0)
                    bb = bb.squeeze(0) if bb is not None else None
            out["sc_pred_global"] = y0_global
            if fR is not None and ft is not None:
                out["sc_frame_R"] = fR
                out["sc_frame_t"] = ft
            out["sc_atom_mask"] = sc_mask
            out["h_res_prime"] = h_res_prime        # per-sigma [B*N_sample, L, C] when per-sigma
            # Reduced h_res' for the cycle injection: the Protenix diffusion shares
            # s_trunk across noise draws (no per-sample s_trunk), so per-sigma h_res'
            # cannot be injected per-sigma without a submodule change. We average
            # over the sigma axis for the injection (documented limitation); the
            # per-sigma h_res' above is still available for other consumers.
            if use_per_sigma:
                out["h_res_prime_reduced"] = (
                    h_res_prime.reshape(-1, n_sample, h_res_prime.shape[-2], h_res_prime.shape[-1])
                    .mean(dim=1)
                )
                if out["h_res_prime_reduced"].shape[0] == 1:
                    out["h_res_prime_reduced"] = out["h_res_prime_reduced"].squeeze(0)
            else:
                out["h_res_prime_reduced"] = h_res_prime

            # Physical regularization (clash + contact) on predicted GLOBAL
            # side-chain coords (S_phi output is already global).
            y_g = y0_global if y0_global.dim() == 4 else y0_global.unsqueeze(0)  # [B,L,A,3]
            m = sc_mask if sc_mask.dim() == 3 else sc_mask.unsqueeze(0)
            if fR is not None and ft is not None and bb is not None:
                B_, L_, A_ = y_g.shape[0], y_g.shape[1], y_g.shape[2]
                # Broadcast per-token frames/backbone to the (possibly per-sigma)
                # batch B_ so the reshape below is well-formed for B*N_sample rows.
                if fR.shape[0] != B_:
                    fR = fR.expand(B_, -1, -1, -1)
                    ft = ft.expand(B_, -1, -1)
                    bb = bb.expand(B_, -1, -1, -1)
                phys = physical_loss(
                    y_g.float().reshape(B_, L_ * A_, 3),
                    backbone_coords=bb.reshape(B_, L_ * bb.shape[2], 3),
                    valid_mask=m.reshape(B_, L_ * A_),
                )
                out["sc_phys_val"] = phys["total"]

            # Stage III type routing: coord loss only where predicted AA == GT
            # (physical-only elsewhere). Always on under predicted_mask so the
            # coord loss is not applied to residues whose predicted atom set differs.
            if self.sc_route_by_type or getattr(self, "sc_predicted_mask", False):
                aa_clean = input_feature_dict.get("aa_clean")
                if aa_clean is not None:
                    # Use the REDUCED predicted type — the same one that
                    # instantiated the predicted mask — so routing is consistent.
                    pred = out["aa_logits_reduced"].argmax(dim=-1)   # [L] or [B, L]
                    if pred.dim() > 1:
                        pred = pred[0]
                    aa_clean = aa_clean.to(pred.device)
                    if aa_clean.dim() > 1:
                        aa_clean = aa_clean[0]
                    tm = (pred == aa_clean)                          # [L]
                    if use_per_sigma:
                        tm = _tile_per_sigma(tm, trailing_ndim=1)
                    out["sc_type_match"] = tm

        # 5. Cycle closure (Stage II-B): reuse B_theta to refine backbone/type
        #    using the side-chain-informed h_res'. h_res' is injected into the
        #    (zero) token trunk s_trunk, then the backbone denoise is re-run.
        #    NOTE: s_trunk is sample-shared in the Protenix diffusion (the N_sample
        #    axis is created inside the module), so we inject the sigma-REDUCED
        #    h_res' here. True per-sigma feedback needs a per-sample s_trunk
        #    (submodule change) — see README. post_aa stays gated (M2).
        if getattr(self, "enable_coevolution", False) and "h_res_prime_reduced" in out:
            h_res_prime = out["h_res_prime_reduced"]
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
            # M2: ONLY when side chains were instantiated from predicted type.
            # With GT-type teacher-forcing, h_res' carries GT atom composition,
            # so supervising post_aa here would be an identity leak — skip it.
            if self.enable_residue_type_head and not self.sc_predicted_mask:
                if not getattr(self, "_warned_post_aa_skip", False):
                    logging.getLogger(__name__).warning(
                        "coevolution: post_aa_logits NOT emitted because "
                        "sidechain.predicted_mask=False (GT atom composition would "
                        "leak residue identity into the AA-refinement objective). "
                        "Set sidechain.predicted_mask=True once S_phi instantiates "
                        "the atom set from predicted type."
                    )
                    self._warned_post_aa_skip = True
            if self.enable_residue_type_head and self.sc_predicted_mask:
                a_post = self._a_token_cache
                if a_post is not None:
                    g = self.aa_trunk_grad_scale
                    # Per-sample (per-sigma) refined logits, same treatment as the
                    # primary AA loss; the loss averages over the N_sample axis.
                    a_post = a_post.to(s_inputs.dtype)
                    if g != 1.0:
                        a_post = g * a_post + (1.0 - g) * a_post.detach()
                    out["post_aa_logits"] = self.design_residue_type_head(
                        a_post, aa_t=input_feature_dict.get("aa_t")
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
