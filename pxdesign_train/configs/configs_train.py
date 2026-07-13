"""
Training configs for PXDesign-d.

Numbers come straight from the technical report (Appendix C, p. 24). We extend
PXDesign's released `configs_base.py` rather than redefining the model — the
architecture is identical, only training-side knobs are added.
"""
from copy import deepcopy

from pxdesign.configs.configs_base import configs as base_configs


training_configs = deepcopy(base_configs)

# Override loss-relevant model knobs that the inference build set conservatively.
training_configs["load_strict"] = False  # we add heads not in the inference ckpt

# Two new model flags consumed by ProtenixDesignTrain.
training_configs["enable_distogram_head"] = True
training_configs["enable_diffusion_distogram_head"] = False  # head built but unused for now
training_configs["enable_residue_type_head"] = True

training_configs["residue_type"] = {
    "vocab_size": 20,
    "ignore_index": -100,
    "loss_on_design_only": True,
    "mask_mode": "time_dependent",
    "mask_prob": 1.0,
    "mask_min_prob": 0.0,
    "mask_max_prob": 1.0,
    # Feed the discrete masked-diffusion time aa_t into the AA head.
    "use_time_embedding": True,
    # Representation the AA head reads. DEFAULT = "diffusion_internal": the
    # a_token AFTER DiffusionModule's full token self-attention (layernorm_a) —
    # it has cross-token context AND is conditioned on the binder's own noisy
    # backbone (r_noisy) + target, which is required for structure-aware residue
    # prediction. "s_inputs" is kept as a structure-blind,
    # cross-token-free baseline/ablation only.
    "input_source": "diffusion_internal",
    # diffusion_internal controls:
    #   trunk_grad_scale: AA-loss gradient into the coord trunk. 1.0 = full
    #     co-design coupling (backbone becomes sequence-aware); lower it only
    #     if the clean-eval shows structure degradation. 0.0 = stop-grad (protect
    #     structure, no co-design coupling).
    "trunk_grad_scale": 1.0,
    # `internal_reduce` controls the single reduced representation used by the
    # h_res_candidate / warmup / cycle-reduction path ("mean" | "low_sigma").
    # The AA LOSS is computed per-sample (per-sigma) and averaged — it does NOT
    # reduce first, so this knob no longer affects the AA training target.
    # When sidechain.per_sigma=True, S_phi also consumes the per-sigma h_res/logits
    # path rather than this reduced h_res_candidate.
    "internal_reduce": "mean",
}

# EDM training noise sampler.
training_configs["training_noise_sampler"] = {
    "p_mean": -1.2,
    "p_std": 1.5,
    "sigma_data": training_configs["sigma_data"],  # 16.0
}

# Training-side hyperparameters from the report.
training_configs["training"] = {
    # The macro-batch is 64 examples; "diffusion batch size 8" is N_sample per
    # example, i.e. how many (rotation, noise) draws we evaluate per item.
    "macro_batch_size": 64,
    "diffusion_batch_size": 8,
    "crop_size": 640,
    "max_steps": 100000,        # ballpark; not stated in report
    "warmup_steps": 2000,       # mirrors Protenix's demo
    "lr": 5e-4,
    "weight_decay": 0.0,
    "ema_decay": 0.999,
    "checkpoint_interval": 400,
    "log_interval": 50,
    "eval_interval": 400,
}

# Loss weights from eq. 4 of the report.
training_configs["loss"] = {
    "weight_mse": 4.0,
    "weight_lddt": 1.0,
    "weight_disto": 0.03,
    "weight_aa": 1.0,
    # MDLM / absorbing-diffusion time weighting (1/t) for the AA CE. When
    # False the AA term is a plain masked-LM mean CE.
    "aa_time_weighting": True,
    "sigma_low_threshold": 4.0,  # σ below this gates LDDT and distogram terms
    "no_bins": training_configs["no_bins"],
    "min_bin": 2.3125,
    "max_bin": 21.6875,
    "lddt_radius": 15.0,
    "align_before_mse": True,
    # Side-chain terms (Stage II-A onwards). Previously these lived only as
    # PXDesignLoss defaults and were NOT plumbed from config; now透传 (M5).
    # Main S_phi coordinate term. In the current path S_phi emits global
    # coordinates and this weights the predicted-frame-aligned global MSE.
    "weight_sc_local": 1.0,
    "weight_sc_phys": 0.1,
    # Legacy local-output aux weight. The global-output path uses the
    # predicted-frame pseudo-target as the primary coordinate term above.
    "weight_sc_global": 0.5,
    # Post-refinement (Stage II-B cycle closure) term weights.
    "weight_bb_post": 1.0,
    "weight_aa_post": 1.0,
}

# Side-Chain Module knobs (consumed by ProtenixDesignTrain when
# enable_sidechain=True). Kept off by default; finetune scripts opt in.
training_configs["sidechain"] = {
    "init_sigma": 1.0,
    # Overleaf paragraph 221 (template-anchored leakage-free initialization):
    # start side-chain denoising from the type-conditioned IDEAL template
    # perturbed by sigma_T, rather than isotropic Gaussian noise. An isotropic
    # Gaussian is rotation-invariant, so pushing it through the predicted frame
    # F_hat carries no backbone-orientation information and S_phi cannot learn
    # where to place atoms in GLOBAL space. The (anisotropic) template does.
    # False restores the old Gaussian init for A/B.
    "template_init": True,
    # ABLATION CANDIDATE -- default OFF, i.e. Yifei's active path
    #     x0_global = MLP(atom_feats) + ca_coords          (CA-anchored global head)
    # Turning it ON gives
    #     x0_global = F_hat . MLP(atom_feats)              (regress LOCAL offsets, let the
    #                                                       KNOWN stop-grad frame rotate them)
    # The output space is global either way, so BOTH satisfy Overleaf par.204; the paper does
    # not mandate a head parameterisation (its appendix explicitly allows equivariant nets),
    # so this is a TRAINING-STABILITY assumption, not a spec requirement -- hence default OFF
    # until real data says otherwise.
    #
    # What we measured, so nobody flips this blindly (1cse chain B, single-structure
    # memorization, 400 steps, sc_local; old local-output baseline = 0.51):
    #     OFF (CA-anchored) + isotropic gaussian init ............ 4.05
    #     OFF + template init ................................... 3.84
    #     ON  + template init ................................... 2.22
    #     ON  + template init + local_coord_input ............... 0.557
    # Hypothesis for the gap: with random rotation augmentation the CA-anchored MLP must
    # infer R from its own input AND apply it to its output -- a bilinear op an MLP
    # approximates poorly. A single-structure memorization run cannot settle this; real data
    # can. Open it as an ablation arm if training stalls.
    "frame_aware_head": False,
    # ABLATION CANDIDATE -- default OFF, i.e. Yifei's active path: S_phi's own noisy
    # side-chain atoms are fed as RAW GLOBAL coordinates, x = F_hat.(mu + sigma.eps).
    # Turning it ON feeds them in the residue-LOCAL frame (translation-free).
    # Overleaf's appendix calls the global-coordinate atom feature "optional", so neither is
    # a spec violation -- this too is a training-stability assumption.
    #
    # The concern (untested on real data): the global form carries t_CA, the residue's
    # ABSOLUTE position (tens of Angstrom, different per residue), on top of a ~4 A side-chain
    # geometry, so the linear coord embedding W_xyz sees mostly "where is this residue" rather
    # than "what shape is this side chain". Measured contribution on the memorization smoke
    # (with frame_aware_head ON): 2.22 -> 0.557. Open as an ablation arm if training stalls.
    "local_coord_input": False,
    # Template perturbation scale (Angstrom, per coordinate). Keep it small
    # relative to side-chain bond lengths (~1.5 A): a large sigma_T destroys the
    # template anisotropy that carries the orientation.
    "init_sigma_T": 0.3,
    "c_atom": 128,
    "trunk_grad_scale": 1.0,
    "detach_feedback": False,
    "route_by_type": False,
    # Per-sigma alignment: feed S_phi a per-sigma h_res / aa_logits / sigma
    # (flattened to [B*N_sample, L, C]) instead of a mean/low-sigma-reduced h_res.
    # This is the intended joint-training main line. Stage II-A warmup may set
    # this False to use the single reduced-h_res baseline (labeled as warmup).
    "per_sigma": True,
    # Paper Stage II-B: S_phi conditions on the PREDICTED backbone. When True
    # (and per_sigma), side-chain frames F_hat are built from x_denoised (x_hat_0)
    # rather than the GT backbone, and the coordinate loss compares S_phi's global
    # output to stopgrad(F_hat) y_gt_local. Warmup (Stage II-A) uses GT frames.
    "predicted_frame": True,
    "weight_sc_global": 0.5,   # legacy local-output aux weight
    # M1: exclude binder side-chain atoms from the backbone (L_bb) target so
    # B_theta is backbone-only and S_phi is the sole side-chain generator.
    "backbone_only_binder": True,
    # M2: only emit/supervise the AA-refinement logits (post_aa) when the
    # side-chain atom set is instantiated from PREDICTED type. Under GT-type
    # teacher-forcing (the current default) GT atom composition would leak
    # residue identity into post_aa via h_res', so we do NOT supervise it.
    "predicted_mask": False,
    # DIRECT a-level side-chain -> backbone feedback (FangWu's slide):
    #     a'_bb = a_bb + MLP(concat(a_bb, a_sc))
    # The default (indirect) path projects h_res' into s_trunk and lets the
    # DiffusionModule recompute a_token from scratch, so the fused representation
    # never *is* the next round's token. With a_direct=True the fusion happens at
    # the a_token level itself (a forward hook on DiffusionModule.layernorm_a
    # replaces its output) and KEEPS the previous backbone token as the residual
    # base. a_sc only exists after round 1, so it fires ONLY in the refinement
    # pass (requires enable_coevolution). Ablation arm: default False, and the
    # residual branch is zero-initialised, so turning it on is a no-op at step 0.
    "a_direct": False,
    "a_direct_zero_init": True,
    # DIRECT q-level (ATOM-level) side-chain -> backbone feedback (FangWu's slide,
    # "Interconnection between Backbone Module and Side-chain Module"):
    #     q'_bb = q_bb + MLP(concat(q_bb, W q_sc_bb))
    # a_direct closes the loop at the TOKEN level (one vector per residue). q_direct
    # closes it at the ATOM level: S_phi keeps all 14 ATOM14 slots — (N, CA, C, O) +
    # 10 side-chain slots — and "by changing the last 10 it adjusts the first 4"; those
    # 4 per-atom features are fused into the Backbone Module's per-atom q (its
    # AtomAttentionEncoder q_skip) for the SAME 4 atoms, via a forward-pre-hook on
    # DiffusionModule.atom_attention_decoder. Every other atom row (receptor, binder
    # side-chain atoms) passes through untouched.
    # q_sc_bb only exists after round 1, so it fires ONLY in the refinement pass
    # (requires enable_coevolution). Independent of a_direct -> the intended ablation
    # is no / a-only / q-only / a+q. Default False, residual branch zero-initialised,
    # so turning it on is an exact no-op at step 0.
    # 14-slot S_phi: the residue's 4 backbone atoms (N,CA,C,O) join the intra-residue
    # attention as CONTEXT (never denoised, never supervised) so the side chain can move
    # their features. This is the PREREQUISITE for q_direct, and also the CONTROL arm that
    # separates "S_phi sees the backbone" from "the q feedback channel".
    # INDIRECT token-level feedback: h_res' -> HResInjector -> s_trunk (today's path).
    # Set False for the TRUE no-feedback control arm: the refinement pass still runs,
    # but carries no side-chain information. Not the same as enable_coevolution=False,
    # which removes the second pass entirely.
    "hres_inject": True,
    "bb_context": False,
    "q_direct": False,
    "q_direct_zero_init": True,
    "weight_bb_post": 1.0,
    "weight_aa_post": 1.0,
}

# Two-stage curriculum (data source weights — to be consumed by the dataloader).
training_configs["curriculum"] = {
    "stage1": {
        # Pre-training: monomer-only distillation data dominates.
        "weights": {"afdb_monomer": 0.5, "mgnify_monomer": 0.4, "pdb_complex": 0.1},
        "max_steps": 50000,
    },
    "stage2": {
        # Conditioned design: shift toward PDB complexes.
        "weights": {"afdb_monomer": 0.1, "mgnify_monomer": 0.1, "pdb_complex": 0.8},
        "max_steps": 50000,
    },
}
