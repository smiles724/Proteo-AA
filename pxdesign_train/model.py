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
import math
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
from pxdesign_train.sidechain.init import (
    DEFAULT_SIGMA_T,
    gaussian_init_local,
    template_init_local,
    templates_available,
)
from pxdesign_train.sidechain.coevolution import ATokenFusion, HResInjector, QAtomFusion
from pxdesign_train.sidechain.frames import to_global, to_local
from pxdesign_train.sidechain.physical import physical_loss


def _tile_per_sample(
    x: torch.Tensor,
    trailing_ndim: int,
    sample_shape: torch.Size,
    n_sample: int,
    flat_B: int,
) -> torch.Tensor:
    """Tile a per-ITEM tensor over the sigma (N_sample) axis.

    h_res / aa_logits are flattened row-major from ``[*sample_shape, N_sample, L, C]``
    to ``[prod(sample_shape) * N_sample, L, C]``. This applies the SAME layout to
    tensors that only carry the per-item leading dims, e.g. ``[B, L, A] ->
    [B*N_sample, L, A]``, so row ``b*N_sample + s`` of the result belongs to item
    ``b`` — per item, never item-0 broadcast to everybody.

    ``trailing_ndim`` is the number of trailing (non-batch) dims of ``x``:
    2 for ``[B, L, A]`` atom ids/masks, 1 for a per-token ``[B, L]`` residue type.
    A tensor with no leading dims (``[L, A]``) is treated as shared across items.
    """
    trail = x.shape[-trailing_ndim:]
    if x.dim() >= trailing_ndim + 1 and x.shape[:-trailing_ndim] == (flat_B,):
        return x                                    # already flattened
    if x.dim() == trailing_ndim:
        base = x.reshape(*((1,) * len(sample_shape)), *trail)
    else:
        base = x
    if base.shape[:-trailing_ndim] != sample_shape:
        base = base.expand(*sample_shape, *trail)
    expanded = base.unsqueeze(len(sample_shape)).expand(*sample_shape, n_sample, *trail)
    return expanded.reshape(flat_B, *trail)


def _route_aa_logits_to_sidechain(
    aa_logits: torch.Tensor, *, detach: bool
) -> torch.Tensor:
    """Optionally close the side-chain loss gradient path into the AA head."""
    return aa_logits.detach() if detach else aa_logits


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
        # Capture (and, under sidechain.a_direct, REPLACE) the internal per-token
        # representation via a forward hook — NO edit to the Protenix/PXDesign
        # submodule source. Registered below, after the side-chain switches are
        # read, because a_direct also needs it.
        self._a_token_cache = None
        # Pass-scoped injection state (see _a_token_forward_hook):
        #   _a_direct_active — True only inside the refinement diffusion call.
        #   _a_sc_cache      — the per-token side-chain summary a_sc from round 1.
        self._a_direct_active = False
        self._a_sc_cache = None
        # Pass-scoped ATOM-level (q) injection state (see _q_skip_decoder_pre_hook):
        #   _q_direct_active — True only inside the refinement diffusion call.
        #   _q_sc_cache      — S_phi's features for the 4 backbone atoms (q_sc_bb),
        #                      sigma-reduced, from round 1. NEVER mutated.
        #   _q_bb_idx_cache  — the (N, CA, C, O) atom indices those rows belong to.
        #   _q_skip_cache    — the Backbone Module's own per-atom q (read side).
        self._q_direct_active = False
        self._q_sc_cache = None
        self._q_bb_idx_cache = None
        self._q_skip_cache = None
        # CALL-KEYED injection. A time-varying flag CANNOT work here: Protenix runs the
        # atom decoder inside torch.utils.checkpoint (use_fine_grained_checkpoint=True,
        # blocks_per_ckpt=1 — the DEFAULT training path), so the decoder is RE-RUN during
        # backward. A flag cleared in `finally` makes the recompute take a different branch
        # than the forward -> "CheckpointError: a different number of tensors was saved
        # during the original forward and recomputation", and QAtomFusion/S_phi get NO
        # gradient. Keeping the flag armed through backward is equally wrong: the FIRST
        # pass's decoder also recomputes, and would then inject too.
        # So we remember WHICH decoder call injected, by the identity of its q_skip tensor.
        # checkpoint(use_reentrant=False) saves the region's inputs, so the recompute sees
        # the SAME tensor object -> forward and recompute make the SAME decision.
        # The dict holds the tensors (not bare ids) so they cannot be GC'd and id-reused.
        self._q_inject_calls: dict = {}

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
            # Overleaf paragraph 221: template-anchored init. y_T = mu_ideal[a] + sigma_T eps,
            # then x_T = F_hat y_T. The ideal template is anisotropic, so the global
            # initialization encodes the backbone orientation; the old isotropic Gaussian
            # is rotation-invariant and encodes nothing. template_init=False restores it.
            self.sc_template_init = bool(getattr(sc_cfg, "template_init", True)) if sc_cfg is not None else True
            self.sc_frame_aware_head = bool(getattr(sc_cfg, "frame_aware_head", False)) if sc_cfg is not None else False
            # Renamed from `local_coord_input`: the coordinate CONTRACT is now
            # single-frame (S_phi is handed global coordinates, always), and this
            # switch only decides whether the per-atom embedding sees them recentred
            # on the residue CA. Rotation into a residue frame no longer happens on
            # any input tensor, which is what made the old switch able to desync the
            # embedding from the cross-residue geometry.
            if sc_cfg is not None and getattr(sc_cfg, "local_coord_input", None) is not None:
                raise ValueError(
                    "sidechain.local_coord_input has been replaced by "
                    "sidechain.centre_coord_input. The old switch rotated S_phi's "
                    "input into per-residue frames, which silently broke the "
                    "cross-residue distance bias; the new one only subtracts CA. "
                    "They are different parameterizations -- a checkpoint trained "
                    "under one cannot be warm-started under the other."
                )
            self.sc_centre_coord_input = bool(getattr(sc_cfg, "centre_coord_input", False)) if sc_cfg is not None else False
            self.sc_template_residual = bool(getattr(sc_cfg, "template_residual", False)) if sc_cfg is not None else False
            if self.sc_template_residual and not self.sc_frame_aware_head:
                raise ValueError(
                    "sidechain.template_residual requires frame_aware_head=True: the "
                    "residual base is a residue-local template, so the head has to "
                    "emit local offsets for the two to be commensurate"
                )
            self.sc_init_sigma_T = float(getattr(sc_cfg, "init_sigma_T", DEFAULT_SIGMA_T)) if sc_cfg is not None else DEFAULT_SIGMA_T
            # ---- EDM side-chain diffusion (sidechain.edm) ----
            self.sc_edm = bool(getattr(sc_cfg, "edm", False)) if sc_cfg is not None else False
            if self.sc_edm:
                from pxdesign_train.sidechain.edm import (
                    SideChainEDM, SideChainNoiseSampler,
                )
                if not self.sc_centre_coord_input:
                    raise ValueError(
                        "sidechain.edm requires centre_coord_input=True. EDM scales the "
                        "network's coordinate input by c_in(sigma), and that scaling is "
                        "only meaningful on data centred near zero -- applied to a raw "
                        "global coordinate it just modulates the residue's absolute "
                        "position, which is tens to hundreds of Angstrom and swamps the "
                        "~4 A of side-chain geometry the scaling is for."
                    )
                self.sc_noise_sampler = SideChainNoiseSampler(
                    p_mean=float(getattr(sc_cfg, "edm_p_mean", -0.223)),
                    p_std=float(getattr(sc_cfg, "edm_p_std", 1.0)),
                    sigma_min=float(getattr(sc_cfg, "edm_sigma_min", 0.05)),
                    sigma_max=float(getattr(sc_cfg, "edm_sigma_max", 4.0)),
                )
                self.sc_edm_sigma_data = float(getattr(sc_cfg, "edm_sigma_data", 2.0))
                self.sc_edm_infer_steps = int(getattr(sc_cfg, "edm_infer_steps", 8))
            if self.sc_template_init and not templates_available():
                logging.getLogger(__name__).warning(
                    "sidechain.template_init=True but pxdesign_train.sidechain.templates "
                    "is not importable; falling back to isotropic Gaussian init."
                )
                self.sc_template_init = False
            # 0714 appendix: which mu_ideal construction to use.
            #   dunbrack      BuildSC with chi ~ Categorical(p(r | a, phi, psi))   [default]
            #   dunbrack_mode BuildSC with chi = argmax_r p(r | a, phi, psi)
            #   ccd           the static one-conformer CCD table (pre-0714 baseline)
            if self.sc_template_init:
                from pxdesign_train.sidechain import rotamers, templates as _templates

                name = str(getattr(sc_cfg, "template_provider", "dunbrack")) if sc_cfg is not None else "dunbrack"
                if name.startswith("dunbrack") and not rotamers.available():
                    logging.getLogger(__name__).warning(
                        "sidechain.template_provider=%s but the rotamer library is not "
                        "built; run `python scripts/sidechain/build_rotamer_library.py --download`. "
                        "Falling back to the static CCD template.",
                        name,
                    )
                    name = "ccd"
                _templates.set_provider_by_name(name)
                self.sc_template_provider = name
            # 0722 L_compat arm for type-MISMATCHED residues: none | clash | legacy | compat.
            from pxdesign_train.sidechain.physical import MISMATCH_ARMS

            self.sc_mismatch_loss = (
                str(getattr(sc_cfg, "mismatch_loss", "clash")) if sc_cfg is not None else "clash"
            )
            if self.sc_mismatch_loss not in MISMATCH_ARMS:
                raise ValueError(
                    f"sidechain.mismatch_loss={self.sc_mismatch_loss!r} is not one of {MISMATCH_ARMS}"
                )
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
            self.sc_force_gt_type_logits = bool(getattr(sc_cfg, "force_gt_type_logits", False)) if sc_cfg is not None else False
            # Optimization ablation: S_phi still receives the exact same AA
            # probabilities, but B_post/sc losses cannot use that route to update
            # the classifier.  The primary AA CE keeps its normal graph through
            # out["aa_logits"].
            self.sc_detach_type_logits_for_sidechain = bool(
                getattr(sc_cfg, "detach_type_logits_for_sidechain", False)
            ) if sc_cfg is not None else False
            # Per-sigma alignment: S_phi reads per-sigma h_res/aa_logits/sigma
            # (flattened [B*N_sample, L, C]) rather than a reduced h_res. Warmup
            # can turn this off for the single-baseline path.
            self.sc_per_sigma = bool(getattr(sc_cfg, "per_sigma", True)) if sc_cfg is not None else True
            # Paper Stage II-B: build side-chain frames from the PREDICTED backbone
            # (x_denoised) + a stop-grad global pseudo-target aux loss. Warmup uses
            # GT frames (predicted_frame=False).
            self.sc_predicted_frame = bool(getattr(sc_cfg, "predicted_frame", True)) if sc_cfg is not None else True
            self.sc_refinement_sigma = float(getattr(sc_cfg, "refinement_sigma", 2.0)) if sc_cfg is not None else 2.0
            if self.sc_refinement_sigma <= 0:
                raise ValueError(
                    "sidechain.refinement_sigma must be > 0, got "
                    f"{self.sc_refinement_sigma}"
                )
            self.sc_weight_global = float(getattr(sc_cfg, "weight_sc_global", 0.5)) if sc_cfg is not None else 0.5
            sc_grad_scale = float(getattr(sc_cfg, "trunk_grad_scale", 1.0)) if sc_cfg is not None else 1.0
            sc_arch = str(getattr(sc_cfg, "architecture", "light")) if sc_cfg is not None else "light"
            dm_cfg = getattr(getattr(configs, "model", None), "diffusion_module", None)
            dm_tr = getattr(dm_cfg, "transformer", None)
            if sc_arch == "diffusion_config":
                c_atom_default = int(getattr(dm_cfg, "c_token", 768)) if dm_cfg is not None else 768
                n_blocks_default = int(getattr(dm_tr, "n_blocks", 16)) if dm_tr is not None else 16
                n_heads_default = int(getattr(dm_tr, "n_heads", 16)) if dm_tr is not None else 16
                c_atom = int(getattr(sc_cfg, "c_atom", c_atom_default))
                n_blocks = int(getattr(sc_cfg, "n_blocks", n_blocks_default))
                n_heads = int(getattr(sc_cfg, "n_heads", n_heads_default))
                n_cross_blocks = int(getattr(sc_cfg, "n_cross_blocks", n_blocks))
            elif sc_arch == "light":
                c_atom = int(getattr(sc_cfg, "c_atom", 128)) if sc_cfg is not None else 128
                n_blocks = int(getattr(sc_cfg, "n_blocks", 2)) if sc_cfg is not None else 2
                n_heads = int(getattr(sc_cfg, "n_heads", 4)) if sc_cfg is not None else 4
                n_cross_blocks = int(getattr(sc_cfg, "n_cross_blocks", 1)) if sc_cfg is not None else 1
            else:
                raise ValueError(
                    f"sidechain.architecture={sc_arch!r} must be 'diffusion_config' or 'light'"
                )
            ff_mult = int(getattr(sc_cfg, "ff_mult", 2)) if sc_cfg is not None else 2
            self.sc_architecture = sc_arch
            self.sc_n_blocks = n_blocks
            self.sc_n_heads = n_heads
            self.sc_n_cross_blocks = n_cross_blocks
            # ---- B<->S interconnection (sidechain.a_bs_concat / q_bs) ----
            # Read-only forward-input fusions: S_phi consumes the Backbone
            # Module's features going INTO this round's call. Independent of
            # the feedback channels (a_direct / q_direct) below, which instead
            # write B's features FROM S_phi's output in the coevolution
            # refinement pass. c_q (the Backbone Module's per-atom width) is
            # needed here — a SideChainModule constructor arg — computed the
            # same way the q_direct branch below computes its own copy.
            self.sc_a_bs_concat = bool(getattr(sc_cfg, "a_bs_concat", False)) if sc_cfg is not None else False
            self.sc_q_bs = bool(getattr(sc_cfg, "q_bs", False)) if sc_cfg is not None else False
            c_q = int(
                getattr(self.diffusion_module.atom_attention_encoder, "c_atom", None)
                or 128
            )
            self.sc_cross_neighbors = (
                int(getattr(sc_cfg, "cross_neighbors", 16)) if sc_cfg is not None else 16
            )
            self.sidechain_module = SideChainModule(
                c_res=self.sc_c_res, c_atom=c_atom, n_type=vocab_size,
                n_blocks=n_blocks, n_heads=n_heads, n_cross_blocks=n_cross_blocks,
                ff_mult=ff_mult, trunk_grad_scale=sc_grad_scale,
                a_bs_concat=self.sc_a_bs_concat, q_bs=self.sc_q_bs, c_q=c_q,
                cross_neighbors=self.sc_cross_neighbors,
                template_residual=self.sc_template_residual,
                centre_coord_input=self.sc_centre_coord_input,
            )
            if self.sc_edm:
                from pxdesign_train.sidechain.edm import SideChainEDM

                # Holds no parameters, so the state dict is identical either way --
                # which is exactly why `edm` must be a RECORDED arch switch: nothing
                # about the weights reveals which objective produced them.
                self.sc_edm_denoiser = SideChainEDM(
                    self.sidechain_module, sigma_data=self.sc_edm_sigma_data
                )
            self.sidechain_feedback = HResFeedback(c_atom=c_atom, c_res=self.sc_c_res)

            # ---- Cycle closure (Stage II-B co-evolution) ----
            # Reuse B_theta for a side-chain-informed refinement pass by injecting
            # h_res' into the (currently-zero) token trunk s_trunk (dim c_s).
            self.enable_coevolution = getattr(configs, "enable_coevolution", False)
            if self.enable_coevolution:
                c_trunk = int(getattr(configs, "c_s", 384))
                self.hres_injector = HResInjector(c_hres=self.sc_c_res, c_trunk=c_trunk)
                # The shared DiffusionModule sees two semantically different modes:
                # B_pre receives x_sigma at a sampled physical noise level, while
                # B_post receives B_pre's x_hat_0 under a fixed refinement sigma.
                # Give the refinement call an explicit identity instead of asking
                # the network to infer the pass from coordinate statistics. Zero-init
                # preserves warm-start behaviour; the BB phase learns the marker.
                self.refinement_pass_embedding = torch.nn.Parameter(
                    torch.zeros(c_trunk)
                )
                self.w_bb_post = float(getattr(sc_cfg, "weight_bb_post", 1.0)) if sc_cfg is not None else 1.0
                self.w_aa_post = float(getattr(sc_cfg, "weight_aa_post", 1.0)) if sc_cfg is not None else 1.0

            # ---- DIRECT a-level feedback (sidechain.a_direct) ----
            # Direct token-level fusion: a'_bb = a_bb + MLP(concat(a_bb, a_sc)).
            # The indirect path above projects h_res' into s_trunk and the
            # DiffusionModule then RECOMPUTES a_token from scratch, so the fused
            # representation never *is* the next round's token. With a_direct we
            # additionally fuse at the a_token level and keep the previous
            # backbone token as the residual base, by REPLACING the output of
            # diffusion_module.layernorm_a in the refinement pass (forward hook).
            # a_sc only exists after round 1, so this fires in the refinement pass
            # only. Ablation arm — default False, indirect path unchanged.
            self.sc_a_direct = bool(getattr(sc_cfg, "a_direct", False)) if sc_cfg is not None else False
            if self.sc_a_direct and not self.enable_coevolution:
                logging.getLogger(__name__).warning(
                    "sidechain.a_direct=True but enable_coevolution=False; there is "
                    "no refinement pass to inject into — disabling a_direct."
                )
                self.sc_a_direct = False
            # ---- SAME fusion, EARLIER injection point (sidechain.a_direct_pre) ----
            # `a_direct` above hooks `layernorm_a`, which runs AFTER the
            # DiffusionTransformer — i.e. after the only GLOBAL cross-residue
            # attention in the module. Everything downstream of it (the atom
            # decoder) is SEQUENCE-local (n_queries=32 / n_keys=128 over the atom
            # index), so a side chain injected there can reach residues that are
            # near in sequence but never one that is near in SPACE. Side-chain
            # packing is a spatial effect: a bulky side chain displaces whatever
            # is beside it in 3D, which is usually far away in sequence. So the
            # late injection point cannot express the very interaction that
            # motivates the co-evolution channel.
            # `a_direct_pre` injects the same concat+MLP+residual fusion into the
            # DiffusionTransformer's `a` argument instead, i.e. BEFORE the global
            # attention. Separate fusion weights from `a_direct` so the two
            # injection points stay independent ablation arms.
            self.sc_a_direct_pre = (
                bool(getattr(sc_cfg, "a_direct_pre", False)) if sc_cfg is not None else False
            )
            if self.sc_a_direct_pre and not self.enable_coevolution:
                logging.getLogger(__name__).warning(
                    "sidechain.a_direct_pre=True but enable_coevolution=False; there "
                    "is no refinement pass to inject into — disabling a_direct_pre."
                )
                self.sc_a_direct_pre = False
            if self.sc_a_direct or self.sc_a_direct_pre:
                c_token_diff = int(
                    getattr(self.diffusion_module, "c_token", None)
                    or getattr(configs, "c_token", 768)
                )
                zero_init = bool(getattr(sc_cfg, "a_direct_zero_init", True))
                if self.sc_a_direct:
                    self.a_token_fusion = ATokenFusion(
                        c_token=c_token_diff, c_atom=c_atom, zero_init=zero_init,
                    )
                if self.sc_a_direct_pre:
                    self.a_token_fusion_pre = ATokenFusion(
                        c_token=c_token_diff, c_atom=c_atom, zero_init=zero_init,
                    )

            # ---- DIRECT q-level ATOM feedback (sidechain.q_direct) ----
            # 14-slot design: the side-chain module keeps backbone atoms in its
            # representation, updates those slots through side-chain attention, and
            # passes the resulting four backbone-atom features back to the backbone module.
            #     q'_bb = q_bb + MLP(concat(q_bb, W q_sc_bb))
            # q_bb are the Backbone Module's per-atom features (AtomAttentionEncoder
            # q_skip) of a residue's N/CA/C/O; q_sc_bb are S_phi's features for the
            # SAME four atoms (its ATOM14 slots 0..3). We write q'_bb back into the
            # atom decoder's skip connection (forward-pre-hook on
            # atom_attention_decoder), leaving every other atom row untouched.
            # Independent of a_direct -> the no / a / q / a+q ablation.
            self.sc_q_direct = bool(getattr(sc_cfg, "q_direct", False)) if sc_cfg is not None else False
            # 14-slot S_phi (4 backbone context atoms in the intra-residue attention) is a
            # SEPARATE change from the q feedback channel. Coupling them confounds the
            # ablation: enabling q_direct also flips S_phi to 14 slots (measured: it moves
            # side-chain coords by up to 0.15 A and atom feats by 0.86 even with a
            # zero-init fusion), so a "q" arm would really measure "q + bb-context".
            # q_direct REQUIRES bb_context (it needs S_phi's backbone-slot features), but
            # bb_context alone is the CONTROL arm that isolates the q channel.
            self.sc_bb_context = bool(getattr(sc_cfg, "bb_context", False)) if sc_cfg is not None else False
            self.sc_hres_inject = bool(getattr(sc_cfg, "hres_inject", True)) if sc_cfg is not None else True
            # Context (receptor/motif/ligand) awareness — see configs_train.py.
            self.sc_context_aware = bool(getattr(sc_cfg, "context_aware", True)) if sc_cfg is not None else True
            self.sc_pack_weight = float(getattr(sc_cfg, "pack_loss", 0.0)) if sc_cfg is not None else 0.0
            self.sc_pack_arm = str(getattr(sc_cfg, "pack_arm", "clash")) if sc_cfg is not None else "clash"
            self.sc_type_logits_input = bool(getattr(sc_cfg, "type_logits_input", True)) if sc_cfg is not None else True
            self.sc_context_radius = float(getattr(sc_cfg, "context_radius", 10.0)) if sc_cfg is not None else 10.0
            self.sc_context_max_atoms = int(getattr(sc_cfg, "context_max_atoms", 4096)) if sc_cfg is not None else 4096
            if self.sc_q_direct and not self.sc_bb_context:
                logging.getLogger(__name__).info(
                    "sidechain.q_direct=True implies bb_context=True (S_phi needs the "
                    "14-slot axis to produce backbone-atom features); enabling it."
                )
                self.sc_bb_context = True
            if self.sc_q_bs and not self.sc_bb_context:
                logging.getLogger(__name__).info(
                    "sidechain.q_bs=True implies bb_context=True (S_phi needs the "
                    "14-slot axis to receive the Backbone Module's q for those "
                    "atoms); enabling it."
                )
                self.sc_bb_context = True
            if self.sc_q_direct and not self.enable_coevolution:
                logging.getLogger(__name__).warning(
                    "sidechain.q_direct=True but enable_coevolution=False; there is "
                    "no refinement pass to inject into — disabling q_direct."
                )
                self.sc_q_direct = False
            if self.sc_q_direct:
                # c_atom on the BACKBONE side (Protenix AtomAttentionEncoder q dim).
                c_q = int(
                    getattr(self.diffusion_module.atom_attention_encoder, "c_atom", None)
                    or 128
                )
                self.q_atom_fusion = QAtomFusion(
                    c_q=c_q, c_atom=c_atom,
                    zero_init=bool(getattr(sc_cfg, "q_direct_zero_init", True)),
                )
        else:
            self.enable_coevolution = False
            self.sc_a_direct = False
            self.sc_q_direct = False
            self.sc_a_bs_concat = False
            self.sc_q_bs = False

        # One hook, two jobs (capture for the AA head, replace for a_direct).
        # A forward hook that returns non-None REPLACES the module's output, which
        # is how a'_bb becomes the token the atom decoder actually consumes.
        if (
            self.enable_residue_type_head
            and self.aa_input_source == "diffusion_internal"
        ) or self.sc_a_direct:
            self.diffusion_module.layernorm_a.register_forward_hook(
                self._a_token_forward_hook
            )
        if getattr(self, "sc_a_direct_pre", False):
            # Pre-hook (with_kwargs) on the DiffusionTransformer: rewriting its
            # `a` argument injects the side chain BEFORE the global cross-residue
            # attention, unlike the layernorm_a hook which lands after it.
            self.diffusion_module.diffusion_transformer.register_forward_pre_hook(
                self._a_token_pre_attention_hook, with_kwargs=True
            )
        if self.sc_q_direct or self.sc_q_bs:
            # READ side: q_skip is the AtomAttentionEncoder's 2nd output. Shared by
            # q_direct (S->B feedback, consumed in the refinement pass's decoder
            # pre-hook below) and q_bs (B->S forward input, gathered at the
            # side-chain forward call site into bb_q).
            self.diffusion_module.atom_attention_encoder.register_forward_hook(
                self._q_skip_encoder_hook
            )
        if self.sc_q_direct:
            # WRITE side: a forward-PRE-hook with_kwargs=True may return a replaced
            # (args, kwargs), which is how q'_bb becomes the q_skip the atom decoder
            # actually consumes. No submodule edit anywhere.
            self.diffusion_module.atom_attention_decoder.register_forward_pre_hook(
                self._q_skip_decoder_pre_hook, with_kwargs=True
            )

    def _a_token_forward_hook(self, _module, _inp, out: torch.Tensor):
        """Forward hook on `DiffusionModule.layernorm_a`.

        1. Caches a_token for the `diffusion_internal` AA head (unchanged).
        2. Under `sidechain.a_direct`, and ONLY while `_a_direct_active` is set
           (i.e. inside the refinement diffusion call) and a side-chain summary
           from round 1 exists, returns the fused token a'_bb, which REPLACES
           layernorm_a's output for the rest of that DiffusionModule forward.

        The hook fires on every layernorm_a call — both passes, and again under
        activation-checkpoint recomputation. It is idempotent: the returned value
        is a pure function of `out` and the (never mutated) cached `a_sc`, so a
        repeated call recomputes the same a'_bb instead of compounding a residual.
        """
        cache = (
            self.enable_residue_type_head
            and self.aa_input_source == "diffusion_internal"
        )
        fused = None
        if getattr(self, "_a_direct_active", False):     # refinement pass only
            a_sc = getattr(self, "_a_sc_cache", None)    # None on the first pass
            if a_sc is not None:
                a_sc = self._align_a_sc(a_sc, out)
                if a_sc is not None:
                    fused = self.a_token_fusion(out, a_sc)
        if cache:
            # Cache what the atom decoder actually consumes, so post_aa_logits
            # reads a'_bb (not the pre-fusion token) when a_direct is on.
            self._a_token_cache = out if fused is None else fused
        return fused                                     # None => output unchanged

    def _a_token_pre_attention_hook(self, _module, args, kwargs):
        """Forward-PRE hook on `DiffusionModule.diffusion_transformer`.

        Fuses the side-chain summary into the token representation BEFORE the
        global cross-residue attention, so the side chain can influence the
        backbone of residues that are close in space but distant in sequence —
        which the post-attention `a_direct` hook structurally cannot do.

        Only fires inside the refinement pass (`_a_direct_active`), the only pass
        where `a_sc` exists. Idempotent under activation-checkpoint recompute:
        the result is a pure function of the incoming `a` and the never-mutated
        `a_sc` cache, so a replay recomputes the same value rather than stacking
        a second residual.

        The DiffusionModule calls this submodule with keyword arguments only; if
        that ever changes, `a` is absent from kwargs and we no-op rather than
        silently injecting into the wrong tensor.
        """
        if not getattr(self, "_a_direct_active", False):
            return None
        a_sc = getattr(self, "_a_sc_cache", None)
        if a_sc is None:
            return None
        a_bb = kwargs.get("a")
        if a_bb is None:
            self._warn_once(
                "_warned_a_direct_pre_positional",
                "a_direct_pre: DiffusionTransformer was called without a keyword "
                "`a` argument; skipping the pre-attention injection.",
            )
            return None
        aligned = self._align_a_sc(a_sc, a_bb)
        if aligned is None:
            return None
        kwargs = dict(kwargs)
        kwargs["a"] = self.a_token_fusion_pre(a_bb, aligned)
        return args, kwargs

    def _align_a_sc(self, a_sc: torch.Tensor, out: torch.Tensor) -> Optional[torch.Tensor]:
        """Broadcast the per-token side-chain summary onto a_token's shape.

        a_sc: [*batch, N_token, c_atom] (sigma-reduced);
        out:  [*batch, N_sample, N_token, c_token]  ->  [*batch, N_sample, N_token, c_atom].
        Returns None (and warns once) if the shapes cannot be reconciled, so a
        shape surprise degrades to "no injection" rather than a wrong fusion.
        """
        a_sc = a_sc.to(device=out.device)
        while a_sc.dim() < out.dim() - 1:
            a_sc = a_sc.unsqueeze(0)
        ok = a_sc.dim() == out.dim() - 1 and a_sc.shape[-2] == out.shape[-2]
        if ok:
            ok = all(
                s in (1, t) for s, t in zip(a_sc.shape[:-2], out.shape[:-3])
            )
        if not ok:
            if not getattr(self, "_warned_a_direct_shape", False):
                logging.getLogger(__name__).warning(
                    "a_direct: cannot align a_sc %s with a_token %s — skipping "
                    "the direct injection for this call.",
                    tuple(a_sc.shape), tuple(out.shape),
                )
                self._warned_a_direct_shape = True
            return None
        # [*batch, N_token, c_atom] -> [*batch, 1, N_token, c_atom] -> expand N_sample
        return a_sc.unsqueeze(-3).expand(*out.shape[:-1], a_sc.shape[-1])

    # ------------------------------------------------------------------
    # ATOM-level (q) side-chain -> backbone feedback (sidechain.q_direct)
    # ------------------------------------------------------------------

    def _q_skip_encoder_hook(self, _module, _inp, out):
        """Forward hook on `DiffusionModule.atom_attention_encoder`.

        The encoder returns ``(a_token, q_skip, c_skip, p_skip)``; ``q_skip`` is the
        per-atom representation [..., N_sample, N_atom, c_atom] that is carried
        around the token trunk and added back inside the atom decoder. We only READ
        it here (returning None leaves the encoder's output untouched).

        The cache is a read-side handle: it is what a residue's four backbone atom
        rows look like on the Backbone Module side, and it is what an S_phi that
        wants q_bb as an *input* feature would consume. The FUSION itself does NOT
        use this cache — it uses the q_skip that arrives at the decoder pre-hook in
        the refinement pass, exactly as `a_direct` uses the refinement pass's own
        freshly recomputed a_bb as the residual base. That also makes the write side
        idempotent under activation-checkpoint recomputation (this hook may fire
        several times; overwriting the cache with the same tensor is harmless).
        """
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            self._q_skip_cache = out[1]
        return None

    def _q_skip_decoder_pre_hook(self, _module, args, kwargs):
        """Forward-PRE-hook on `DiffusionModule.atom_attention_decoder`.

        Replaces `q_skip` with the version whose 4 backbone-atom rows per binder
        residue are the fused q'_bb. Every other atom row — receptor atoms, binder
        side-chain atoms, tokens with no resolved frame — is passed through byte-for-
        byte (they are literally the same rows of the incoming tensor).

        CALL-KEYED (not flag-keyed): `_q_direct_active` marks only the FORWARD of the
        refinement pass. Protenix runs this decoder inside torch.utils.checkpoint on the
        DEFAULT training config, so it is RE-RUN during backward, when the flag is already
        down. We therefore remember which decoder call injected, by the identity of its
        q_skip tensor, so forward and recomputation make the SAME decision. The first
        pass never injects — in the forward (flag down, no cache) nor in its recompute
        (its q_skip was never recorded).

        Handles BOTH call forms Protenix uses:
          * kwargs      — the normal `self.atom_attention_decoder(..., q_skip=...)`;
          * POSITIONAL  — the fine-grained-checkpoint branch calls the decoder through
            `checkpoint_fn(self.atom_attention_decoder, atom_to_token_idx, a_token,
            q_skip, ...)`, i.e. q_skip is positional arg index 2.

        Returns None (leave the call untouched) whenever anything is missing or the
        shapes cannot be reconciled — a surprise degrades to "no injection", never to
        a wrong fusion.
        """
        q_sc = getattr(self, "_q_sc_cache", None)
        bb_idx = getattr(self, "_q_bb_idx_cache", None)
        if q_sc is None or bb_idx is None:
            return None                       # first pass: S_phi has not run yet

        pos = None
        if "q_skip" in kwargs:
            q_in = kwargs["q_skip"]
        elif len(args) > 2 and torch.is_tensor(args[2]):
            pos, q_in = 2, args[2]
        else:
            return None

        # CALL-KEYED decision (see _q_inject_calls in __init__). `_q_direct_active` marks
        # only the FORWARD of the refinement pass; on backward recomputation it is already
        # down, so we must not consult it — we consult whether THIS decoder call (identified
        # by its q_skip tensor) injected during the forward.
        calls = getattr(self, "_q_inject_calls", None)
        if calls is None:
            calls = self._q_inject_calls = {}
        key = id(q_in)
        if key in calls:
            pass                                   # recompute of an injected call -> inject
        elif getattr(self, "_q_direct_active", False):
            # Only register under grad: the registry exists solely so the backward
            # RECOMPUTE of a checkpointed decoder call reaches the same decision as its
            # forward. Under no_grad (inference) there is no recompute, so registering
            # would just strong-reference every step's q_skip for the whole sampling run.
            if torch.is_grad_enabled():
                calls[key] = q_in                  # forward of the refinement pass
        else:
            return None                            # first pass (or unarmed) -> never inject

        q_new = self._fuse_q_backbone_atoms(q_in, q_sc, bb_idx)
        if q_new is None:
            return None
        if pos is None:
            kwargs = dict(kwargs)
            kwargs["q_skip"] = q_new
        else:
            args = list(args)
            args[pos] = q_new
            args = tuple(args)
        return args, kwargs

    def _fuse_q_backbone_atoms(
        self, q_skip: torch.Tensor, q_sc: torch.Tensor, bb_idx: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """q'_bb = q_bb + MLP(concat(q_bb, W q_sc_bb)), scattered back into q_skip.

        q_skip: [..., N_sample, N_atom, c_q]  — the LIVE q of this (refinement) pass.
        q_sc:   [..., L, 4, c_atom]           — S_phi's backbone-slot features (round 1).
        bb_idx: [..., L, 4] long              — (N, CA, C, O) atom indices; -1 = absent.

        PURE FUNCTION of (q_skip, q_sc, bb_idx): nothing is accumulated and nothing
        cached is mutated, so a repeated pre-hook call (checkpoint recompute) lands
        on exactly the same tensor.
        """
        n_atom, c_q = q_skip.shape[-2], q_skip.shape[-1]
        lead = q_skip.shape[:-2]                       # (*item_dims, N_sample)
        q_sc = q_sc.to(device=q_skip.device)
        bb_idx = bb_idx.to(device=q_skip.device).long()
        while q_sc.dim() < q_skip.dim():               # [L,4,c] -> [1,...,L,4,c]
            q_sc = q_sc.unsqueeze(0)
        while bb_idx.dim() < q_skip.dim() - 1:         # [L,4]   -> [1,...,L,4]
            bb_idx = bb_idx.unsqueeze(0)
        n_res, n_bb = q_sc.shape[-3], q_sc.shape[-2]
        ok = (
            q_sc.dim() == q_skip.dim()
            and bb_idx.dim() == q_skip.dim() - 1
            and tuple(bb_idx.shape[-2:]) == (n_res, n_bb)
            and all(s in (1, t) for s, t in zip(q_sc.shape[:-3], lead[:-1]))
            and all(s in (1, t) for s, t in zip(bb_idx.shape[:-2], lead[:-1]))
        )
        if not ok:
            if not getattr(self, "_warned_q_direct_shape", False):
                logging.getLogger(__name__).warning(
                    "q_direct: cannot align q_sc %s / bb_idx %s with q_skip %s — "
                    "skipping the atom-level injection for this call.",
                    tuple(q_sc.shape), tuple(bb_idx.shape), tuple(q_skip.shape),
                )
                self._warned_q_direct_shape = True
            return None

        # Broadcast the per-item, sigma-reduced side-chain rows over the N_sample axis
        # (the refinement pass draws FRESH sigmas, so its row s carries no
        # correspondence to round-1 row s — same reduction a_direct makes).
        tgt = (*lead, n_res, n_bb)
        q_sc = q_sc.unsqueeze(-4).expand(*tgt, q_sc.shape[-1])
        bb_idx = bb_idx.unsqueeze(-3).expand(*tgt)

        flat_b = math.prod(lead) if lead else 1
        m = n_res * n_bb
        q_flat = q_skip.reshape(flat_b, n_atom, c_q)
        idx = bb_idx.reshape(flat_b, m)
        valid = idx >= 0                                            # [B, L*4]
        gather_idx = idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, c_q)
        q_bb = q_flat.gather(1, gather_idx) * valid.unsqueeze(-1).to(q_flat.dtype)
        fused = self.q_atom_fusion(
            q_bb.reshape(flat_b, n_res, n_bb, c_q),
            q_sc.reshape(flat_b, n_res, n_bb, q_sc.shape[-1]),
        ).reshape(flat_b, m, c_q)

        # Scatter q'_bb back at the atoms it came from. Absent atoms (-1, e.g. an
        # unresolved O, or any non-binder token) are routed to a TRASH row appended
        # at index n_atom and dropped: clamping them to 0 instead would collide with
        # residue 0's real N atom and non-deterministically overwrite it.
        trash = torch.full_like(idx, n_atom)
        scatter_idx = torch.where(valid, idx, trash).unsqueeze(-1).expand(-1, -1, c_q)
        q_pad = torch.cat([q_flat, q_flat.new_zeros(flat_b, 1, c_q)], dim=1)
        q_pad = q_pad.scatter(1, scatter_idx, fused)
        return q_pad[:, :n_atom, :].reshape(q_skip.shape)

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

    def _template_phi_psi(
        self, input_feature_dict, out, n_rows: int, n_token: int, use_predicted: bool
    ):
        """(phi, psi) for the template's rotamer lookup — 0714 appendix, Step 2.

        `use_predicted` mirrors the ACTIVE FRAME's choice of backbone and must not be
        decided independently: mu_ideal is posed in that frame, so the rotamer has to be
        conditioned on the same backbone the frame is built from. Stage II-B uses the
        predicted x_hat_0 (what the appendix specifies, and what inference has); the
        Stage II-A warmup has no predicted frame and uses the GT backbone, which is that
        stage's own definition ("ideal templates in the ground-truth local frames"). The
        leakage rule is about side-chain coordinates, not the backbone, so both are sound.

        Returns CPU tensors [n_rows, n_token] in radians (NaN where undefined), or
        (None, None) if no backbone is available — the provider then falls back to the
        backbone-independent marginal rather than failing the step.
        """
        from pxdesign_train.sidechain.frames import backbone_phi_psi, phi_psi_from_ncac

        ri = input_feature_dict.get("residue_index")
        ai = input_feature_dict.get("asym_id")
        if ri is None or ai is None:
            self._warn_once(
                "_warned_no_phipsi_feats",
                "template_init: residue_index/asym_id absent; the rotamer lookup "
                "falls back to the backbone-independent marginal.",
            )
            return None, None
        ri = ri.detach().cpu().reshape(-1)[:n_token]
        ai = ai.detach().cpu().reshape(-1)[:n_token]

        bb_idx = input_feature_dict.get("sc_bb_atom_idx")
        if bb_idx is None:
            return None, None
        bb_idx = bb_idx.detach().cpu().long().reshape(-1, bb_idx.shape[-1])[:n_token]
        # The featurizer fills the backbone slots for BINDER tokens only; everything else
        # is -1 (and sc_bb_coords is all-zero there). Without this mask the GT branch would
        # read those zeros as real coordinates, find a "peptide bond" of length 0, and hand
        # back a confident-looking dihedral of three coincident points.
        have = (bb_idx[:, :3] >= 0).all(dim=-1)

        xden = out.get("x_denoised")
        phi = psi = None
        if use_predicted and xden is not None:
            x = xden.detach().cpu().float()
            x = x.reshape(-1, x.shape[-2], x.shape[-1])          # [B*N_sample, N_atom, 3]
            phi, psi = backbone_phi_psi(x, bb_idx, ri, ai)
        else:
            gt = input_feature_dict.get("sc_bb_coords")          # [.., L, 4, 3] (N,CA,C,O)
            if gt is not None:
                g = gt.detach().cpu().float()
                g = g.reshape(-1, g.shape[-3], g.shape[-2], g.shape[-1])
                phi, psi = phi_psi_from_ncac(
                    g[..., 0, :], g[..., 1, :], g[..., 2, :], ri, ai, have=have
                )
        if phi is None:
            return None, None

        # Match the [B*N_sample, L] tiling that sc_type_idx / sc_mask already have.
        if phi.shape[0] != n_rows:
            if phi.shape[0] == 1:
                phi = phi.expand(n_rows, -1)
                psi = psi.expand(n_rows, -1)
            elif n_rows % phi.shape[0] == 0:
                rep = n_rows // phi.shape[0]
                phi = phi.repeat_interleave(rep, dim=0)
                psi = psi.repeat_interleave(rep, dim=0)
            else:
                self._warn_once(
                    "_warned_phipsi_shape",
                    f"template_init: phi/psi rows {phi.shape[0]} do not tile to "
                    f"{n_rows}; falling back to the marginal rotamer distribution.",
                )
                return None, None
        return phi.contiguous(), psi.contiguous()

    def _warn_once(self, flag: str, msg: str) -> None:
        if not getattr(self, flag, False):
            logging.getLogger(__name__).warning(msg)
            setattr(self, flag, True)

    def _mismatch_subject_mask(self, out, valid_flat, B_, L_, A_):
        """Side-chain atoms the mismatch regularizer is ABOUT: 0722's I_unmatched.

        Returns [B_, L_*A_] bool = valid AND belongs to a residue whose predicted
        type differs from GT.

        With no type-match mask the answer is an EMPTY subject set, not "everything".
        Those are the teacher-forced stages (II warmup, III joint): the atom set comes
        from a_GT, so by construction nothing is mismatched, and the Stage III objective
        in 0722 contains no L_compat term at all. Scoring every residue there would
        reintroduce the pre-0722 behaviour precisely where the paper says the coordinate
        loss should be the only side-chain supervision.
        """
        tm = out.get("sc_type_match")
        if tm is None:
            if getattr(self, "sc_predicted_mask", False):
                self._warn_once(
                    "_warned_no_typematch",
                    "mismatch_loss: predicted_mask is on but no type-match mask was "
                    "produced (aa_clean missing?), so no residue can be identified as "
                    "mismatched and the regularizer contributes 0.",
                )
            return torch.zeros_like(valid_flat)
        tm = tm.bool()
        if tm.dim() == 1:
            tm = tm.unsqueeze(0)
        if tm.shape[0] != B_:
            if tm.shape[0] == 1:
                tm = tm.expand(B_, -1)
            elif B_ % tm.shape[0] == 0:
                tm = tm.repeat_interleave(B_ // tm.shape[0], dim=0)
            else:
                self._warn_once(
                    "_warned_mismatch_shape",
                    f"mismatch_loss: type-match rows {tm.shape[0]} do not tile to {B_}; "
                    "the regularizer contributes 0 for this step rather than guessing "
                    "which residues are mismatched.",
                )
                return torch.zeros_like(valid_flat)
        assert tm.shape[-1] >= L_, (
            f"mismatch_loss: type-match covers {tm.shape[-1]} tokens but the side-chain "
            f"axis has {L_} — the two came from different token sets, so the mask would "
            "silently regularize the wrong residues."
        )
        mismatch = ~tm[..., :L_]                                  # [B_, L_]
        return valid_flat & mismatch[..., None].expand(B_, L_, A_).reshape(B_, L_ * A_)

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

        # Side-chain state is per-forward: never let a_sc from the previous step
        # (or a previous item) leak into this one, and make sure the first pass
        # runs with the injection off.
        self._a_sc_cache = None
        self._a_direct_active = False
        self._q_inject_calls = {}      # per-forward; backward of THIS step still sees it
        self._q_sc_cache = None
        self._q_bb_idx_cache = None
        self._q_direct_active = False

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
        x_gt_aug, x_denoised, sigma, x_noisy = sample_diffusion_training(
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
                    See module-level `_tile_per_sample` (hoisted so it is testable)."""
                    return _tile_per_sample(
                        x, trailing_ndim, sample_shape, n_sample, h_res.shape[0]
                    )
            else:
                h_res = out["h_res_candidate"]      # [N_token, c_res] or [B, N_token, c_res]
                # S_phi conditions on the REDUCED AA distribution (warmup baseline).
                aa_logits = out["aa_logits_reduced"]
                squeeze = h_res.dim() == 2          # batch=1 collapsed upstream
                if squeeze:
                    h_res = h_res.unsqueeze(0)
                    aa_logits = aa_logits.unsqueeze(0)
            sc_ids = input_feature_dict["sc_atom_name_ids"]
            # SUPERVISION mask (chemistry AND resolved AND plausible) -- losses only.
            sc_mask = input_feature_dict["sc_atom_mask"]
            # CHEMISTRY mask (does this residue type own this slot) -- everything the
            # FORWARD does: attention keys, output zeroing, pooling, template init.
            # A model whose key set depends on crystallographic completeness cannot be
            # reproduced at inference, where no `is_resolved` exists. Falls back to the
            # supervision mask for feature dicts built before the split.
            sc_slot = input_feature_dict.get("sc_slot_mask")
            if sc_slot is None:
                sc_slot = sc_mask
            # Stage III predicted-mask branch: instantiate the side-chain atom set
            # from the PREDICTED residue type (argmax of the reduced logits) instead
            # of the GT type. This is what makes post_aa safe to supervise (M2): the
            # atom composition no longer carries GT identity. Matched residues still
            # align with sc_gt_local (same atoms/order); mismatched ones are routed
            # to physical-only via sc_type_match below.
            # Residue-type source for the template-anchored init (paragraph 221).
            # It MUST be the same type that produced sc_ids / sc_mask below:
            # predicted type under sidechain.predicted_mask, GT type (aa_clean)
            # under atom-mask teacher forcing. [L] long, STD_AA_3 order.
            sc_type_idx = None
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
                # Predicted-type instantiation carries no resolution information, so
                # chemistry and supervision coincide on this branch.
                sc_mask = pmask.to(sc_mask.device)
                sc_slot = sc_mask
                sc_type_idx = ptype                              # [L] predicted type
            else:
                # Teacher forcing: sc_ids / sc_mask come from the GT residue type,
                # so the init template must use that same GT type (aa_clean).
                # Keep the ITEM axis here ([B, L] stays [B, L]): it is tiled per
                # item below, exactly like sc_ids / sc_mask. Collapsing it to
                # item 0 and broadcasting would give items 1..B-1 item-0's
                # residue templates under their own atom masks.
                aa_clean_t = input_feature_dict.get("aa_clean")
                if aa_clean_t is not None:
                    sc_type_idx = aa_clean_t.long()
            if use_per_sigma:
                sc_ids = _tile_per_sigma(sc_ids, trailing_ndim=2)
                sc_mask = _tile_per_sigma(sc_mask, trailing_ndim=2)
                sc_slot = _tile_per_sigma(sc_slot, trailing_ndim=2)
                if sc_type_idx is not None:
                    # trailing_ndim=1: the type is per TOKEN ([B, L] / [L]), so the
                    # same row-major tiling as the [B, L, A] mask above lines type
                    # row b*N_sample+s up with mask row b*N_sample+s.
                    sc_type_idx = _tile_per_sigma(sc_type_idx, trailing_ndim=1)
            else:
                if sc_ids.dim() == 2:                   # add batch dim
                    sc_ids = sc_ids.unsqueeze(0)
                    sc_mask = sc_mask.unsqueeze(0)
                    sc_slot = sc_slot.unsqueeze(0)
                B = h_res.shape[0]
                if sc_ids.shape[0] != B:
                    sc_ids = sc_ids.expand(B, -1, -1)
                    sc_mask = sc_mask.expand(B, -1, -1)
                    sc_slot = sc_slot.expand(B, -1, -1)
                if sc_type_idx is not None:
                    if sc_type_idx.dim() == 1:          # [L] -> [1, L]
                        sc_type_idx = sc_type_idx.unsqueeze(0)
                    if sc_type_idx.shape[0] != B:
                        sc_type_idx = sc_type_idx.expand(B, -1)
            sc_ids = sc_ids.to(h_res.device).long()
            sc_mask = sc_mask.to(h_res.device).bool()
            sc_slot = sc_slot.to(h_res.device).bool()
            if sc_type_idx is not None:
                sc_type_idx = sc_type_idx.to(h_res.device).long()
            if not getattr(self, "sc_type_logits_input", True):
                # Ablation: hold shapes and parameters fixed, remove the information.
                # A uniform distribution makes w_aa's contribution the mean of its 20
                # columns -- a constant bias -- so any change in the loss is exactly
                # what this channel was carrying.
                aa_logits = torch.zeros_like(aa_logits)
            if getattr(self, "sc_force_gt_type_logits", False) and sc_type_idx is not None:
                n_type = aa_logits.shape[-1]
                valid_type = (sc_type_idx >= 0) & (sc_type_idx < n_type)
                gt_logits = torch.full(
                    (*sc_type_idx.shape, n_type),
                    -20.0,
                    device=h_res.device,
                    dtype=aa_logits.dtype,
                )
                gt_logits.scatter_(-1, sc_type_idx.clamp(0, n_type - 1)[..., None], 20.0)
                aa_logits = torch.where(valid_type[..., None], gt_logits, aa_logits)
            aa_logits = _route_aa_logits_to_sidechain(
                aa_logits,
                detach=getattr(self, "sc_detach_type_logits_for_sidechain", False),
            )
            B = h_res.shape[0]
            # Overleaf paragraph 221: y_T = mu_ideal[a_i, j] + sigma_T eps  (the
            # x_T = F_hat y_T half is the to_global(...) call below). sc_type_idx and
            # sc_mask are tiled to the SAME flattened [B*N_sample, ...] layout above,
            # so row r of the type table is row r of the atom mask (per item, not
            # item 0 broadcast), and each sigma row still draws its own eps.
            use_template_init = getattr(self, "sc_template_init", False) and sc_type_idx is not None
            if use_template_init:
                mask_cpu = sc_slot.detach().cpu()
                tix = sc_type_idx.detach().cpu().long()
                assert tix.shape == mask_cpu.shape[:-1], (
                    f"template init: type {tuple(tix.shape)} not aligned with atom "
                    f"mask {tuple(mask_cpu.shape)} — per-item tiling is broken"
                )
                # 0714 appendix Step 2: the rotamer is conditioned on the backbone's
                # dihedrals. phi needs residue i-1's C and psi residue i+1's N, so this
                # has to be computed here, over the whole chain — a per-residue template
                # lookup cannot do it.
                #
                # It must come from the SAME backbone that builds F_hat below, because
                # mu_ideal is expressed in that frame: conditioning the rotamer on the
                # predicted backbone while posing it in a GT frame (or vice versa) puts
                # the side chain in a frame belonging to a different structure. So mirror
                # the frame's own choice exactly — predicted x_hat_0 under Stage II-B,
                # GT backbone in the Stage II-A warmup, which is also why the row counts
                # line up (predicted: B*N_sample rows; GT: B rows).
                use_pred_bb = (
                    bool(getattr(self, "sc_predicted_frame", False))
                    and use_per_sigma
                    and out.get("x_denoised") is not None
                )
                phi, psi = self._template_phi_psi(
                    input_feature_dict, out, n_rows=tix.shape[0], n_token=tix.shape[-1],
                    use_predicted=use_pred_bb,
                )
                # Say ONCE whether the backbone conditioning is actually live: a silent
                # fall-back to the marginal looks exactly like a working run. Measured over
                # the tokens that HAVE a side chain to initialize -- the featurizer resolves
                # backbone atoms for binder tokens only, so a fraction over all tokens would
                # just report the binder's share of the crop and tell us nothing.
                if not getattr(self, "_logged_phipsi", False):
                    self._logged_phipsi = True
                    sc_tok = mask_cpu.any(-1)
                    cov = (
                        float((torch.isfinite(phi) & sc_tok).sum() / sc_tok.sum().clamp_min(1))
                        if phi is not None else 0.0
                    )
                    logging.getLogger(__name__).info(
                        "template_init: provider=%s, phi/psi from %s, backbone-conditioned "
                        "for %.1f%% of side-chain-bearing tokens (rest: marginal)",
                        getattr(self, "sc_template_provider", "?"),
                        "predicted x_hat_0" if use_pred_bb else "GT backbone",
                        100.0 * cov,
                    )
                noisy_init = template_init_local(
                    tix, mask_cpu,
                    # Under EDM the perturbation is the SAMPLED sigma below, so the
                    # template arrives clean; sigma_T's fixed 0.3 A would otherwise
                    # sit underneath it as an unaccounted noise floor.
                    sigma_T=(0.0 if getattr(self, "sc_edm", False) else self.sc_init_sigma_T),
                    phi=phi, psi=psi,
                )
            else:
                if getattr(self, "sc_template_init", False) and not getattr(self, "_warned_no_type_src", False):
                    logging.getLogger(__name__).warning(
                        "sidechain.template_init=True but no residue-type source "
                        "(aa_clean) is available; falling back to Gaussian init."
                    )
                    self._warned_no_type_src = True
                noisy_init = gaussian_init_local(
                    sc_slot.detach().cpu(), sigma=self.sc_init_sigma
                )
            noisy_init = noisy_init.to(h_res.device).to(h_res.dtype)
            sc_sigma = None
            if getattr(self, "sc_edm", False):
                from pxdesign_train.sidechain.edm import noise_sidechains

                # One sigma per ROW of the flattened batch, so under per_sigma each
                # backbone noise draw gets its own independent side-chain noise level
                # rather than inheriting the backbone's.
                if getattr(self, "sc_edm_eval", False):
                    # EVAL PROTOCOL: start deterministically at sigma_max and walk
                    # the Karras schedule down, which is how this model is actually
                    # used. Sampling a random sigma per item -- correct for training
                    # -- makes the validation number a draw from a distribution
                    # rather than a measurement, and lambda(sigma) weighting makes it
                    # incomparable to an arm trained at a fixed sigma.
                    sc_sigma = torch.full(
                        (h_res.shape[0],), self.sc_noise_sampler.sigma_max,
                        device=h_res.device, dtype=torch.float32,
                    )
                else:
                    sc_sigma = self.sc_noise_sampler(
                        (h_res.shape[0],), device=h_res.device, dtype=torch.float32
                    )
                # WHAT GETS NOISED, and why it is the GT and not the template.
                #
                # EDM's whole parameterisation assumes x_sigma = x_0 + sigma*eps with
                # x_0 the TARGET. Noising the template instead breaks it in the worst
                # possible place: the template sits ~2.18 A from the target (measured),
                # so at small sigma c_skip -> 1 forces the denoiser to output the
                # template and be wrong by that much, while lambda(sigma) -> infinity
                # puts almost all the loss weight exactly there. The model would be
                # graded hardest on the regime where it is structurally forbidden to
                # be right.
                #
                # This is not GT leakage: noising your own target and learning to undo
                # it IS diffusion training. The model never sees a clean target at
                # inference -- it starts from the template at sigma_max, where
                # c_skip ~ 0 and the template's bias is small next to 4 A of noise.
                #
                # Under the eval protocol there IS no target to noise, so the template
                # stays the starting point, which is exactly the inference behaviour.
                # DIAGNOSTIC ONLY (sc_edm_eval_from_gt): start the reverse loop from
                # the noised TARGET instead of the noised template. This is not a
                # legitimate inference protocol -- it uses the answer -- and exists
                # solely to separate two explanations for the EDM arm's gap:
                #   (A) train/inference mismatch: it trains on GT+sigma*eps and is
                #       deployed from template+sigma*eps, so the starting
                #       distribution differs. Then this score improves a lot.
                #   (B) the model simply has not learned the task well, e.g. because
                #       it never sees the rotamer prior the one-step arms get free.
                #       Then this score is about the same.
                if getattr(self, "sc_edm_eval", False) and not getattr(
                    self, "sc_edm_eval_from_gt", False
                ):
                    x0_local = noisy_init.float()
                else:
                    gt_local_t = input_feature_dict.get("sc_gt_local")
                    if gt_local_t is None:
                        raise ValueError(
                            "sidechain.edm needs sc_gt_local to build the diffusion "
                            "input; without it the only thing available to noise is "
                            "the template, which mis-specifies the objective."
                        )
                    x0_local = gt_local_t.to(h_res.device).float()
                    if use_per_sigma:
                        x0_local = _tile_per_sigma(x0_local, trailing_ndim=3)
                    else:
                        if x0_local.dim() == 3:
                            x0_local = x0_local.unsqueeze(0)
                        if x0_local.shape[0] != h_res.shape[0]:
                            x0_local = x0_local.expand(h_res.shape[0], -1, -1, -1)
                noisy_init = noise_sidechains(x0_local, sc_sigma).to(h_res.dtype)
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
                    # sc_bb_atom_idx is [L, 4] = (N, CA, C, O). Frames and the
                    # physical-loss backbone reference use the three frame atoms
                    # only — slicing here keeps behavior identical to the former
                    # 3-wide feature (a 4-wide gather would silently add O, and an
                    # `.all(-1)` validity test over 4 columns would drop every token
                    # whose O is unresolved).
                    bb_idx = bb_idx[..., :3]
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

            # Validity of every row of `bb`. `bb` is a per-TOKEN table, so its
            # non-binder rows are NOT absent — they are (0,0,0) on the GT path
            # (sc_bb_coords is zero-filled there) and a copy of atom 0 on the
            # predicted path (bb_idx.clamp_min(0)). Anything that reduces over that
            # axis (contact_loss's `min`) will happily select them unless masked.
            bb_valid = None
            if bb is not None:
                _bbi = input_feature_dict.get("sc_bb_atom_idx")
                if _bbi is not None:
                    _bbi = _bbi.to(h_res.device).long()[..., : bb.shape[-2]]
                    if use_per_sigma:
                        _bbi = _tile_per_sigma(_bbi, trailing_ndim=2)
                    else:
                        if _bbi.dim() == 2:
                            _bbi = _bbi.unsqueeze(0)
                        if _bbi.shape[0] != B:
                            _bbi = _bbi.expand(B, -1, -1)
                    bb_valid = _bbi >= 0                      # [B, L, K] bool

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

            # SINGLE FRAME CONTRACT: everything handed to S_phi is global. The
            # template init is generated residue-locally (that is what a rotamer
            # library gives you), so it is mapped out here, once, and nothing
            # downstream has to know it was ever local.
            #
            # The scale problem this used to solve -- a raw global coordinate carries
            # t_CA, tens to hundreds of Angstrom, on top of ~4 A of side-chain
            # geometry, and W_xyz cannot separate them -- is now handled inside the
            # module by `centre_coord_input`, which subtracts CA for the embedding
            # only. That is a pure translation, so it cannot desynchronise from the
            # coordinates used for cross-residue distances the way a residue-local
            # input tensor did.
            #
            # Detaching the frame keeps the side-chain coord objective from nudging
            # the predicted backbone frame through the initialization path.
            if fR is not None and ft is not None:
                noisy = to_global(noisy_init.float(), fR.detach(), ft.detach()).to(h_res.dtype)
            else:
                noisy = noisy_init

            ca_for_module = ca.detach() if ca is not None else None

            # ---- CONTEXT: receptor / motif / ligand (SPEC, see configs_train.py) ----
            # S_phi must be able to attend to the thing it packs against, and the
            # clash/contact terms must score against it. Both need the atoms S_phi does
            # NOT own, expressed in the SAME frame as its predicted side chains — so we
            # take them from x_denoised (the augmented frame), never from the
            # featurizer's un-augmented coordinates. All of it is stop-grad: the
            # receptor is fixed conditioning, not something the side-chain loss may move.
            #
            # Gated on use_per_sigma for the same reason predicted frames are: that is
            # the path where x_denoised's rows line up with S_phi's rows. The Stage II-A
            # warmup (single reduced baseline, GT frames) runs without context.
            ctx_tok = None                     # [B, L] bool — token is context, not binder
            ctx_atoms = None                   # (coords [B,M,3], mask [B,M], group [B,M])
            xden_ctx = out.get("x_denoised")
            if (
                getattr(self, "sc_context_aware", False)
                and use_per_sigma
                and xden_ctx is not None
                and input_feature_dict.get("sc_token_center_idx") is not None
                and input_feature_dict.get("atom_to_token_idx") is not None
                and input_feature_dict.get("sc_bb_atom_idx") is not None
            ):
                from pxdesign_train.sidechain.physical import build_sidechain_context

                ca_for_module, ctx_tok, ctx_atoms = build_sidechain_context(
                    xyz=xden_ctx.reshape(
                        -1, xden_ctx.shape[-2], xden_ctx.shape[-1]
                    ).to(h_res.device).float(),                               # [B, N_atom, 3]
                    center_idx=_tile_per_sigma(
                        input_feature_dict["sc_token_center_idx"].to(h_res.device).long(),
                        trailing_ndim=1,
                    ),                                                        # [B, L]
                    atom_to_token=_tile_per_sigma(
                        input_feature_dict["atom_to_token_idx"].to(h_res.device).long(),
                        trailing_ndim=1,
                    ),                                                        # [B, N_atom]
                    bb_atom_idx=_tile_per_sigma(
                        input_feature_dict["sc_bb_atom_idx"].to(h_res.device).long(),
                        trailing_ndim=2,
                    ),                                                        # [B, L, 4]
                    ca=ca_for_module,
                    radius=float(getattr(self, "sc_context_radius", 10.0)),
                    max_atoms=int(getattr(self, "sc_context_max_atoms", 4096)),
                    # Drop the binder's own scrubbed side-chain rows (coords pinned to
                    # Cα by the featurizer) so they cannot be selected as context —
                    # otherwise phantom Cα-piled atoms pollute the mismatch
                    # clash/contact context set.
                    exclude_atom_mask=(
                        _tile_per_sigma(
                            input_feature_dict["design_sidechain_atom_mask"].to(h_res.device),
                            trailing_ndim=1,
                        )
                        if input_feature_dict.get("design_sidechain_atom_mask") is not None
                        else None
                    ),
                )

            # Frame-aware head (sidechain.frame_aware_head): hand S_phi the SAME stop-grad
            # rigid frame the target is built on, so it regresses rotation-invariant local
            # offsets and the known transform does the rotating. Output stays global.
            _fa = getattr(self, "sc_frame_aware_head", False) and fR is not None and ft is not None

            # ---- ATOM-level (q) feedback: give S_phi its 4 backbone context slots ----
            # 14-slot context: S_phi attends over ATOM14 = (N, CA, C, O) +
            # 10 side-chain slots. The backbone slots are CONTEXT ONLY (known coords,
            # never denoised, never supervised); their post-attention features are the
            # q_sc_bb handed back to the Backbone Module's 4 atom rows.
            #
            # WHAT WE FEED S_phi: the residue's four backbone atoms in its own LOCAL
            # frame, gathered by NAME from sc_bb_atom_idx (N, CA, C, O). The coords come
            # from the PREDICTED backbone x_denoised when Stage II-B predicted frames are
            # active (GT sc_bb_coords only in the Stage II-A warmup fallback) — backbone
            # coordinates are what S_phi conditions on, so this is inside the leakage
            # rule; NO GT side-chain coordinate enters. Both the coords and the frame are
            # stop-grad, so the side-chain objective cannot nudge the backbone through
            # this new path (it reaches the backbone only through q'_bb, on purpose).
            # By default we do NOT feed q_bb (the Backbone Module's own atom
            # features) into S_phi — the backbone signal already reaches S_phi
            # through h_res. `sidechain.q_bs` opts into that extra arm: it
            # gathers this round's q_skip (cached read-side by
            # `_q_skip_encoder_hook`) at the SAME four backbone-atom indices and
            # feeds it in as `bb_q` below, ablation-gated and independent of
            # q_direct's S->B write-back.
            bb_local = sc_res_mask = None
            q_bb_idx = None
            bb_q = None
            if getattr(self, "sc_bb_context", False) and fR is not None and ft is not None:
                idx4 = input_feature_dict.get("sc_bb_atom_idx")
                if idx4 is not None:
                    idx4 = idx4.to(h_res.device).long()             # [L,4] or [B_item,L,4]
                    q_bb_idx = idx4
                    if use_per_sigma:
                        idx4 = _tile_per_sigma(idx4, trailing_ndim=2)
                    else:
                        if idx4.dim() == 2:
                            idx4 = idx4.unsqueeze(0)
                        if idx4.shape[0] != B:
                            idx4 = idx4.expand(B, -1, -1)
                    L4 = idx4.shape[-2]
                    valid4 = idx4 >= 0                              # [B, L, 4]
                    bb4 = None
                    _xden = out.get("x_denoised")
                    if R_hat is not None and _xden is not None:
                        _xf = _xden.reshape(-1, _xden.shape[-2], _xden.shape[-1])
                        _xf = _xf.to(h_res.device).float()
                        if _xf.shape[0] == B:
                            gi = idx4.clamp_min(0).reshape(B, L4 * 4, 1).expand(-1, -1, 3)
                            bb4 = _xf.gather(1, gi).reshape(B, L4, 4, 3)
                    if bb4 is None:                                 # warmup: GT backbone
                        gtbb = input_feature_dict.get("sc_bb_coords")
                        if gtbb is not None:
                            gtbb = gtbb.to(h_res.device).float()
                            if use_per_sigma:
                                gtbb = _tile_per_sigma(gtbb, trailing_ndim=3)
                            else:
                                if gtbb.dim() == 3:
                                    gtbb = gtbb.unsqueeze(0)
                                if gtbb.shape[0] != B:
                                    gtbb = gtbb.expand(B, -1, -1, -1)
                            bb4 = gtbb
                    if bb4 is not None:
                        v4 = valid4[..., None].to(bb4.dtype)
                        bb_global = (bb4 * v4).detach()
                        # An absent atom (an unresolved O) is left at the origin by
                        # the `* v4` above — a bounded fallback; its fused row is never
                        # scattered back, because its index is -1. These four slots are
                        # concatenated to the ten side-chain slots and read by ONE
                        # Linear, so they carry the same (global) frame as those.
                        bb_local = bb_global.to(h_res.dtype)
                        sc_res_mask = valid4[..., :3].all(dim=-1)   # frame atoms present

                    # ---- B->S gather (sidechain.q_bs): bb_q from this round's cached
                    # encoder q_skip, at the SAME (N, CA, C, O) atom indices used for
                    # bb_local above. Read-only — no scatter-back (that is q_direct's
                    # job, in the LATER refinement pass's decoder pre-hook). Mirrors
                    # the gather portion of `_fuse_q_backbone_atoms`: gather rows by
                    # bb_idx.clamp_min(0), zero out where bb_idx < 0. `idx4` here is
                    # already tiled to THIS round's flattened batch B (same tiling
                    # `_fuse_q_backbone_atoms` does per-sigma via its own broadcast),
                    # so no separate re-tiling from `q_bb_idx` is needed.
                    if getattr(self, "sc_q_bs", False) and self._q_skip_cache is not None:
                        q_skip_c = self._q_skip_cache
                        n_atom_q, c_q_ = q_skip_c.shape[-2], q_skip_c.shape[-1]
                        q_flat = q_skip_c.reshape(-1, n_atom_q, c_q_).to(h_res.device)
                        if q_flat.shape[0] == B:
                            idxq = idx4.reshape(B, L4 * 4)
                            validq = idxq >= 0
                            gather_idx = idxq.clamp_min(0).unsqueeze(-1).expand(-1, -1, c_q_)
                            bb_q = q_flat.gather(1, gather_idx) * validq.unsqueeze(-1).to(q_flat.dtype)
                            bb_q = bb_q.reshape(B, L4, 4, c_q_).to(h_res.dtype)
                        elif not getattr(self, "_warned_q_bs_shape", False):
                            logging.getLogger(__name__).warning(
                                "sidechain.q_bs=True but q_skip_cache batch %d != "
                                "side-chain batch %d (likely sc_per_sigma=False) — "
                                "skipping the bb_q gather for this call.",
                                q_flat.shape[0], B,
                            )
                            self._warned_q_bs_shape = True

            sc_kwargs = {}
            if bb_local is not None:
                sc_kwargs = {"bb_coords": bb_local, "res_mask": sc_res_mask}
            if bb_q is not None:
                sc_kwargs["bb_q"] = bb_q
            if sc_sigma is not None and ca_for_module is None:
                # Refuse rather than fall through. A silent fallback here would run
                # the one-step path while every log line, checkpoint record and
                # report still said EDM -- the failure is invisible until someone
                # compares two runs that were never the same experiment.
                raise ValueError(
                    "sidechain.edm is on but no ca_coords reached S_phi, so the EDM "
                    "preconditioning has no centre to work in. ca comes from the "
                    "frame translation (sc_frame_t); if that is absent the run is "
                    "not doing what its config says."
                )
            if sc_sigma is not None:
                if not getattr(self, "_logged_edm", False):
                    self._logged_edm = True
                    logging.getLogger(__name__).info(
                        "sidechain EDM ACTIVE: sigma_data=%.3g, sigma in [%.3g, %.3g], "
                        "median %.3g; time embedding receives log(sigma)/4, "
                        "loss weighted by lambda(sigma)=1/c_out^2",
                        self.sc_edm_sigma_data,
                        self.sc_noise_sampler.sigma_min,
                        self.sc_noise_sampler.sigma_max,
                        float(torch.tensor(self.sc_noise_sampler.p_mean).exp()),
                    )
                # EDM: the wrapper supplies `t` (= c_noise) and `coord_scale` (= c_in),
                # so the time channel finally carries the noise level instead of the
                # constant it has been fed since Stage II began.
                _edm_common = dict(
                    frame_R=(fR.detach() if _fa else None),
                    frame_t=(ft.detach() if _fa else None),
                    ctx_mask=ctx_tok,
                    **sc_kwargs,
                )
                if getattr(self, "sc_edm_eval", False):
                    from pxdesign_train.sidechain.edm import sidechain_reverse_loop

                    sch = self.sc_noise_sampler.schedule(
                        self.sc_edm_infer_steps,
                        device=h_res.device, dtype=torch.float32,
                    )
                    _y, _aux = sidechain_reverse_loop(
                        self.sc_edm_denoiser, noisy, sch, ca_for_module,
                        h_res, aa_logits, sc_ids, sc_slot,
                        atom_mask=sc_slot, return_aux=True, **_edm_common,
                    )
                    sc_out = (_y, *_aux)
                else:
                    sc_out = self.sc_edm_denoiser.denoise(
                        noisy, sc_sigma, ca_for_module,
                        h_res, aa_logits, sc_ids, sc_slot,
                        **_edm_common,
                    )
            else:
                sc_out = self.sidechain_module(
                    h_res, aa_logits, sc_ids, sc_slot, noisy, t, ca_coords=ca_for_module,
                    frame_R=(fR.detach() if _fa else None),
                    frame_t=(ft.detach() if _fa else None),
                    ctx_mask=ctx_tok,
                    **sc_kwargs,
                )
            # 3-tuple only when we opted into the 14-slot axis (bb_local given).
            bb_feats = None
            if len(sc_out) == 3:
                y0_global, atom_feats, bb_feats = sc_out
            else:
                y0_global, atom_feats = sc_out
            h_res_prime = self.sidechain_feedback(
                atom_feats, sc_slot, h_res, detach=self.sc_detach_feedback,
            )
            # DIRECT a-level feedback: a_sc = the SAME pooled side-chain atom
            # feature HResFeedback consumes (masked mean over the residue's atoms),
            # cached for the refinement pass's a_token fusion. Reduced over the
            # sigma axis like h_res_prime_reduced: the refinement pass draws FRESH
            # sigmas, so its row s carries no correspondence to round-1 row s —
            # pretending otherwise would pair a_sc with the wrong noise level.
            if getattr(self, "sc_a_direct", False):
                a_sc = self.a_token_fusion.pool(
                    atom_feats.detach() if self.sc_detach_feedback else atom_feats,
                    sc_slot,
                )                                   # [B*N_sample, L, c_atom] (per-sigma)
                if use_per_sigma:
                    a_sc = a_sc.reshape(
                        *sample_shape, n_sample, a_sc.shape[-2], a_sc.shape[-1]
                    ).mean(dim=len(sample_shape))   # [*sample_shape, L, c_atom]
                elif squeeze:
                    a_sc = a_sc.squeeze(0)
                self._a_sc_cache = a_sc
            # DIRECT q-level (atom) feedback: q_sc_bb = S_phi's post-attention features
            # for the residue's OWN 4 backbone slots. Reduced over the sigma axis for
            # the same reason a_sc is (the refinement pass redraws sigma), and paired
            # with the per-ITEM (N, CA, C, O) atom indices those rows must be written
            # back to. Both are cached read-only; the fusion itself happens in the
            # decoder pre-hook against that pass's LIVE q_skip, so it is idempotent.
            if getattr(self, "sc_q_direct", False) and bb_feats is not None:
                q_sc = bb_feats.detach() if self.sc_detach_feedback else bb_feats
                if use_per_sigma:
                    q_sc = q_sc.reshape(
                        *sample_shape, n_sample, *q_sc.shape[-3:]
                    ).mean(dim=len(sample_shape))   # [*sample_shape, L, 4, c_atom]
                elif squeeze:
                    q_sc = q_sc.squeeze(0)
                self._q_sc_cache = q_sc
                self._q_bb_idx_cache = q_bb_idx     # [L,4] / [B_item,L,4], per ITEM
            if squeeze:                             # restore the collapsed batch dim
                y0_global = y0_global.squeeze(0)
                sc_mask = sc_mask.squeeze(0)
                sc_slot = sc_slot.squeeze(0)
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
            if sc_sigma is not None:
                from pxdesign_train.sidechain.edm import edm_loss_weight

                out["sc_sigma"] = sc_sigma.detach()
                # lambda(sigma) = 1/c_out^2. Without it the high-sigma draws dominate
                # the gradient and the model quietly optimises the easy end.
                #
                # NOT applied under the eval protocol: there the whole point is a
                # number commensurate with an arm that never had a sigma to weight
                # by, so the reported loss must be the plain masked MSE.
                if not getattr(self, "sc_edm_eval", False):
                    out["sc_loss_weight"] = edm_loss_weight(
                        sc_sigma, self.sc_edm_sigma_data
                    )
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
            #
            # The context set (backbone atoms + receptor/motif/ligand, radius-filtered
            # and MASKED) is what both terms score against. It replaces the old
            # `backbone_coords=bb.reshape(...)` reference, which was binder-only AND
            # unmasked: `bb` is a per-TOKEN table, so every receptor token contributed
            # phantom atoms — (0,0,0) on the GT-frame path (sc_bb_coords is zero there)
            # and a duplicate of atom 0 on the predicted path (bb_idx.clamp_min(0)).
            # `contact_loss` takes a min over that axis, so those phantoms silently
            # zeroed the runaway penalty for any side-chain atom near them.
            y_g = y0_global if y0_global.dim() == 4 else y0_global.unsqueeze(0)  # [B,L,A,3]
            m = sc_slot if sc_slot.dim() == 3 else sc_slot.unsqueeze(0)
            if fR is not None and ft is not None and ctx_atoms is None and bb is not None and bb_valid is not None:
                # Stage II-A warmup fallback: no receptor. GT frames live in the RAW,
                # un-augmented coordinate frame, while x_denoised (our only source of
                # receptor atoms) lives in the AUGMENTED one — scoring one against the
                # other is a frame bug, not a conservative approximation. So warmup keeps
                # the binder's own backbone as context, now MASKED (this is where the
                # phantom atoms used to enter).
                bb_ctx = bb.unsqueeze(0) if bb.dim() == 3 else bb
                bb_valid_ctx = bb_valid.unsqueeze(0) if bb_valid.dim() == 2 else bb_valid
                B_, L_, K_ = bb_ctx.shape[0], bb_ctx.shape[1], bb_ctx.shape[2]
                y_B = y_g.shape[0]
                if B_ != y_B:
                    if B_ != 1:
                        raise ValueError(
                            f"cannot align side-chain context batch {B_} with predictions {y_B}"
                        )
                    bb_ctx = bb_ctx.expand(y_B, -1, -1, -1)
                    bb_valid_ctx = bb_valid_ctx.expand(y_B, -1, -1)
                    B_ = y_B
                if bb_valid_ctx.shape[0] != B_:
                    bb_valid_ctx = bb_valid_ctx.expand(B_, -1, -1)
                _g = torch.arange(L_, device=y_g.device)[:, None].expand(L_, K_)
                ctx_atoms = (
                    bb_ctx.reshape(B_, L_ * K_, 3).detach(),
                    bb_valid_ctx.reshape(B_, L_ * K_),
                    _g.reshape(1, L_ * K_).expand(B_, -1),
                )
            # Stage III/IV type routing: coord loss only where predicted AA == GT
            # (mismatch regularizer elsewhere). Always on under predicted_mask so the
            # coord loss is not applied to residues whose predicted atom set differs.
            #
            # Computed BEFORE the mismatch regularizer, because that term needs the
            # same mask: 0722 scopes it to I_unmatched = {i : a_hat_i != a_i^GT}.
            # The primary side-chain constraint comes from correctly typed residues;
            # wrong-type residues get this auxiliary term, or nothing when
            # mismatch_loss="none".
            if self.sc_route_by_type or getattr(self, "sc_predicted_mask", False):
                aa_clean = input_feature_dict.get("aa_clean")
                if aa_clean is not None:
                    # Use the REDUCED predicted type — the same one that
                    # instantiated the predicted mask — so routing is consistent.
                    pred = out["aa_logits_reduced"].argmax(dim=-1)   # [L] or [B, L]
                    aa_clean = aa_clean.to(pred.device)
                    # Keep the item axis: collapsing to item 0 here would gate item 1..B-1's
                    # side-chain coordinate loss with item 0's (pred == GT) pattern — the same
                    # batch bug the init path just fixed, silently, on the same aa_clean tensor.
                    if pred.dim() == 1 and aa_clean.dim() > 1:
                        aa_clean = aa_clean[0]
                    elif aa_clean.dim() == 1 and pred.dim() > 1:
                        aa_clean = aa_clean.unsqueeze(0).expand_as(pred)
                    tm = (pred == aa_clean)                          # [L] or [B, L]
                    if use_per_sigma:
                        tm = _tile_per_sigma(tm, trailing_ndim=1)
                    out["sc_type_match"] = tm

            # Mismatched-residue regularizer (0722 L_compat; arm chosen by
            # sidechain.mismatch_loss). Scoped to residues whose PREDICTED type
            # differs from GT: where the type is right, the coordinate loss already
            # supervises the full local geometry and 0722 explicitly wants no extra
            # geometric term there.
            #
            # Under teacher forcing (Stage II/III) every residue matches by
            # construction, so the subject set is empty and this contributes 0 —
            # which is exactly the Stage III objective, whose equation has no
            # L_compat term at all. It only becomes live under predicted masks
            # (Stage IV).
            if (
                fR is not None and ft is not None and ctx_atoms is not None
                and self.sc_mismatch_loss != "none"
            ):
                B_, L_, A_ = y_g.shape[0], y_g.shape[1], y_g.shape[2]
                ctx_xyz, ctx_m, ctx_g = ctx_atoms
                # Residue id of each side-chain atom, so the clash term can drop
                # same-residue (bonded) side-chain<->backbone pairs.
                sc_group = (
                    torch.arange(L_, device=y_g.device)[:, None]
                    .expand(L_, A_)
                    .reshape(1, L_ * A_)
                    .expand(B_, -1)
                )
                valid_flat = m.reshape(B_, L_ * A_)
                subject_flat = self._mismatch_subject_mask(out, valid_flat, B_, L_, A_)
                phys = physical_loss(
                    y_g.float().reshape(B_, L_ * A_, 3),
                    context_coords=ctx_xyz,
                    context_mask=ctx_m,
                    context_group_id=ctx_g,
                    group_id=sc_group,
                    valid_mask=valid_flat,
                    subject_mask=subject_flat,
                    arm=self.sc_mismatch_loss,
                )
                out["sc_phys_val"] = phys["total"]
                out["sc_phys_clash"] = phys["clash"].detach()
                out["sc_phys_contact"] = phys["contact"].detach()

            # GENERAL packing term (sidechain.pack_loss), distinct from L_compat above.
            #
            # `sc_phys` is 0722's mismatch regularizer: `_mismatch_subject_mask` scopes
            # it to residues whose PREDICTED type differs from GT, so under teacher
            # forcing -- Stage II warmup and Stage III -- the subject set is empty and
            # it contributes exactly zero. That is deliberate for L_compat, but it also
            # means the objective contains NO steric supervision at all in those stages:
            # the coordinate loss is a per-residue masked MSE in a local frame, which is
            # structurally blind to two side chains occupying the same space. Two
            # residues can each be scored perfectly and still interpenetrate.
            #
            # So this scores the SAME clash/contact terms over every supervised
            # side-chain atom. Default weight 0 -- switching it on changes the
            # objective, and Stage III/IV comparisons must not shift underneath us --
            # but it is the only channel through which packing can be optimised.
            if (
                fR is not None and ft is not None and ctx_atoms is not None
                and float(getattr(self, "sc_pack_weight", 0.0)) > 0.0
            ):
                B_, L_, A_ = y_g.shape[0], y_g.shape[1], y_g.shape[2]
                ctx_xyz, ctx_m, ctx_g = ctx_atoms
                sc_group = (
                    torch.arange(L_, device=y_g.device)[:, None]
                    .expand(L_, A_)
                    .reshape(1, L_ * A_)
                    .expand(B_, -1)
                )
                valid_flat = m.reshape(B_, L_ * A_)
                pack = physical_loss(
                    y_g.float().reshape(B_, L_ * A_, 3),
                    context_coords=ctx_xyz,
                    context_mask=ctx_m,
                    context_group_id=ctx_g,
                    group_id=sc_group,
                    valid_mask=valid_flat,
                    subject_mask=valid_flat,     # every supervised atom, not just mismatches
                    # Steric exclusion only. 0722 calls clash the one
                    # reliable term, and it is precisely what a per-residue
                    # local-frame MSE cannot see; the contact hinge is
                    # position anchoring, which the coordinate loss already does.
                    arm=str(getattr(self, "sc_pack_arm", "clash")),
                )
                out["sc_pack_val"] = pack["total"]
                out["sc_pack_clash"] = pack["clash"].detach()
                out["sc_pack_contact"] = pack["contact"].detach()

        # 5. Cycle closure (Stage II-B): reuse B_theta to refine backbone/type
        #    using the side-chain-informed h_res'. h_res' is injected into the
        #    (zero) token trunk s_trunk, then the backbone denoise is re-run.
        #    NOTE: s_trunk is sample-shared in the Protenix diffusion (the N_sample
        #    axis is created inside the module), so we inject the sigma-REDUCED
        #    h_res' here. True per-sigma feedback needs a per-sample s_trunk
        #    (submodule change) — see README. post_aa stays gated (M2).
        if getattr(self, "enable_coevolution", False) and "h_res_prime_reduced" in out:
            h_res_prime = out["h_res_prime_reduced"]
            # INDIRECT token-level feedback (sidechain.hres_inject, default ON = today's
            # behaviour): h_res' -> HResInjector -> s_trunk, and the DiffusionModule then
            # recomputes a_token from it. Turning this OFF is what makes a TRUE no-feedback
            # control possible: the refinement pass still runs (B_theta is still called a
            # second time), but it carries NO side-chain information at all. That is the
            # arm that answers "does the co-evolution channel buy anything?" -- it is NOT
            # the same as enable_coevolution=False, which removes the refinement pass
            # entirely and would confound "second pass" with "side-chain feedback".
            # Explicit pass identity is independent of the feedback ablation:
            # even a no-feedback B_post still receives x_hat_0 rather than x_sigma.
            s_trunk_refine = s + self.refinement_pass_embedding.to(s.dtype)
            if getattr(self, "sc_hres_inject", True):
                s_trunk_refine = (
                    s_trunk_refine + self.hres_injector(h_res_prime).to(s.dtype)
                )
            # DIRECT a-level feedback (sidechain.a_direct): arm the layernorm_a hook
            # for the duration of THIS call only. The first pass above ran with the
            # flag down (and with _a_sc_cache=None), so a'_bb = a_bb + MLP(...) can
            # only happen here, in the refinement pass — which is also the only pass
            # where a_sc exists. The finally-clause disarms it even if the diffusion
            # call raises, so a later first pass can never inherit a live flag.
            self._a_direct_active = bool(
                getattr(self, "sc_a_direct", False) and self._a_sc_cache is not None
            )
            # Same pass scoping for the ATOM-level channel: the decoder pre-hook only
            # rewrites q_skip while this flag is up, i.e. inside the refinement call —
            # the only pass where q_sc_bb exists ("only available after the first-round").
            self._q_direct_active = bool(
                getattr(self, "sc_q_direct", False)
                and self._q_sc_cache is not None
                and self._q_bb_idx_cache is not None
            )
            try:
                # Continue from the structure produced by B_pre. This is iterative
                # refinement, not a second prediction of the original noisy state:
                #
                #   x_noisy -> B_pre -> x_denoised -> B_post -> x_denoised_post
                #
                # B_post also receives the side-chain feedback through
                # s_trunk_refine and the armed a/q hooks. Reusing x_gt_aug preserves
                # the first pass's coordinate frame. Do NOT reuse B_pre's randomly
                # sampled physical sigma here: B_post's input is already x_hat_0,
                # not x_sigma. Train and inference instead share one explicit
                # refinement sigma so they learn/execute the same transition.
                # `torch.full_like` below changes only DiffusionModule conditioning
                # and EDM c_skip/c_out; it does NOT add noise to x_denoised. The
                # coordinate input remains B_pre's prediction exactly.
                refinement_sigma = torch.full_like(
                    sigma, self.sc_refinement_sigma
                )
                x_gt_aug_post, x_denoised_post, sigma_post, _ = sample_diffusion_training(
                    noise_sampler=self.training_noise_sampler,
                    denoise_net=self.diffusion_module,
                    label_dict=label_dict,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk_refine,
                    z_trunk=z,
                    N_sample=N_sample,
                    precomputed_input=(x_gt_aug, refinement_sigma, x_denoised),
                )
            finally:
                self._a_direct_active = False
                self._q_direct_active = False
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
