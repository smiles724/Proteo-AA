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
    "weight_sc_local": 1.0,
    "weight_sc_phys": 0.1,
    # Predicted-frame stop-grad pseudo-target aux (paper Stage II-B).
    "weight_sc_global": 0.5,
    # Post-refinement (Stage II-B cycle closure) term weights.
    "weight_bb_post": 1.0,
    "weight_aa_post": 1.0,
}

# Side-Chain Module knobs (consumed by ProtenixDesignTrain when
# enable_sidechain=True). Kept off by default; finetune scripts opt in.
training_configs["sidechain"] = {
    "init_sigma": 1.0,
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
    # rather than the GT backbone, and a stop-grad global pseudo-target aux loss
    # is added. Warmup (Stage II-A) uses GT frames -> set False.
    "predicted_frame": True,
    "weight_sc_global": 0.5,   # weight of the predicted-frame pseudo-target aux loss
    # M1: exclude binder side-chain atoms from the backbone (L_bb) target so
    # B_theta is backbone-only and S_phi is the sole side-chain generator.
    "backbone_only_binder": True,
    # M2: only emit/supervise the AA-refinement logits (post_aa) when the
    # side-chain atom set is instantiated from PREDICTED type. Under GT-type
    # teacher-forcing (the current default) GT atom composition would leak
    # residue identity into post_aa via h_res', so we do NOT supervise it.
    "predicted_mask": False,
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
