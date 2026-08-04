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
    # DEFAULT = "all" (complete-mask / predict-all): the design region is always
    # fully masked, so sequence is a pure prediction target and the model learns
    # p(a, x0^bb | x_t^bb, C) at every timestep. This matches cogenerate's default
    # path and avoids a separate sequence unmasking schedule.
    "mask_mode": "all",
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

SC_ABLATION_ARMS = {
    # true control: refinement pass runs, no side-chain info reaches the backbone
    "no":            dict(hres_inject=False, a_direct=False, bb_context=False, q_direct=False),
    # current/default indirect channel: h_res' -> s_trunk -> a_token recomputed
    "a-indirect":    dict(hres_inject=True,  a_direct=False, bb_context=False, q_direct=False),
    # token-level concat/fusion: a'_bb = a_bb + MLP([a_bb, a_sc])
    "a-direct":      dict(hres_inject=False, a_direct=True,  bb_context=False, q_direct=False),
    # q control: S_phi sees 4 backbone context atoms, but no q feedback is written back
    "bbctx":         dict(hres_inject=False, a_direct=False, bb_context=True,  q_direct=False),
    # atom-level concat/fusion: q'_bb = q_bb + MLP([q_bb, q_sc_bb])
    "q":             dict(hres_inject=False, a_direct=False, bb_context=True,  q_direct=True),
    # both explicit concat/fusion channels
    "a-direct+q":    dict(hres_inject=False, a_direct=True,  bb_context=True,  q_direct=True),
}


def apply_sidechain_ablation_arm(configs, arm: str):
    """Apply a named side-chain feedback ablation arm to a config object/dict."""
    if arm in (None, "", "default"):
        return configs
    if arm not in SC_ABLATION_ARMS:
        raise ValueError(f"unknown side-chain ablation arm {arm!r}; choose one of {sorted(SC_ABLATION_ARMS)}")
    sc = configs["sidechain"] if isinstance(configs, dict) else configs.sidechain
    for key, value in SC_ABLATION_ARMS[arm].items():
        if isinstance(sc, dict):
            sc[key] = value
        else:
            setattr(sc, key, value)
    return configs


# Side-Chain Module knobs (consumed by ProtenixDesignTrain when
# enable_sidechain=True). Kept off by default; finetune scripts opt in.
training_configs["sidechain"] = {
    # S_phi architecture. The main line now matches the backbone diffusion
    # transformer's width/depth configuration, because this is a full-atom model
    # rather than a small side-chain probe. Keep mechanism ablations at the same
    # capacity; future capacity runs should change the numeric fields below.
    "architecture": "diffusion_config",
    "c_atom": 768,
    "n_blocks": 16,
    "n_heads": 16,
    "n_cross_blocks": 16,
    "ff_mult": 2,
    "init_sigma": 1.0,
    # Receptor / motif / ligand context. Spec (Overleaf requires it in 6 places), not an
    # option; a switch only so Stage II-A warmup (GT frames) can skip it. See
    # docs/sidechain_config_notes.md.
    "context_aware": True,
    "context_radius": 10.0,      # A; atoms beyond this from any binder CA are dropped
    "context_max_atoms": 4096,   # hard cap on the context set (memory bound)
    # Overleaf par.221: init from the type-conditioned ideal template + sigma_T noise,
    # not isotropic Gaussian (mu_ideal == 0). False restores the Gaussian A/B baseline.
    # docs/sidechain_config_notes.md.
    "template_init": True,
    # mu_ideal construction: "dunbrack_mode" (argmax) | "dunbrack" (sampled) | "ccd" (static).
    # Template RMSD to true side chain: gaussian 2.89 / ccd 1.66 / dunbrack 1.49 /
    # dunbrack_mode 1.28 A. docs/sidechain_config_notes.md.
    "template_provider": "dunbrack_mode",
    # Ablation candidate, default OFF (CA-anchored head x0 = MLP + ca_coords). ON:
    # x0 = F_hat.MLP (regress local offsets, known frame rotates). docs/sidechain_config_notes.md.
    "frame_aware_head": False,
    # Ablation candidate, default OFF (feed S_phi's noisy coords as raw global). ON: feed them
    # in the residue-local frame (translation-free). docs/sidechain_config_notes.md.
    "local_coord_input": False,
    # Template perturbation scale (Angstrom, per coordinate). Keep it small
    # relative to side-chain bond lengths (~1.5 A): a large sigma_T destroys the
    # template anisotropy that carries the orientation.
    "init_sigma_T": 0.3,
    "trunk_grad_scale": 1.0,
    "detach_feedback": False,
    "route_by_type": False,
    # ---- What to do with residues whose PREDICTED aa type is WRONG ----
    # 0722 renamed the old "physical regularization" to CONTEXT-AWARE regularization
    # and changed its contents: L_compat = clash + pack + hbond. bond / angle /
    # rotamer are GONE, and explicitly so -- when the type is right, the coordinate
    # loss already supervises the complete local geometry "without introducing
    # additional bond-length, bond-angle, or rotamer losses".
    #
    # The mismatch branch is intentionally an ablation switch. The primary
    # side-chain constraint comes from correctly typed residues with coordinate
    # supervision; for wrong-type residues these context-aware terms are weak
    # auxiliary signals and may bias the sequence distribution. The 0722
    # Limitations section makes the same point: they can be useful but need
    # careful weighting and ablation, and may be better suited to reranking than
    # as a replacement for coordinate-supervised side-chain learning.
    #
    #   "none"    no term. Wrong-type side chains are unconstrained (they are still
    #             generated, and still feed h_res' back into B_theta).
    #   "clash"   steric only. DEFAULT: it is the one term 0722 calls "reliable for
    #             rejecting severe overlaps", and being purely repulsive it is the
    #             hardest to game -- the attractive terms (pack/hbond) are where the
    #             exploitation risk lives.
    #   "legacy"  clash + the pre-0722 nearest-context hinge. Reproduces earlier runs.
    #   "compat"  0722 App. 4.9 in full (softplus/vdW clash + pack + hbond).
    #             NOT IMPLEMENTED -- raises. That is the next work item.
    #
    # For reference, these are differentiable simplifications of standard Rosetta
    # energy terms (clash ~ fa_rep, pack ~ fa_atr/fa_sol, hbond ~ hbond_sc), so the
    # concepts are not unprecedented even though the smooth forms in 4.9 are bespoke.
    #
    # SCOPE: this applies ONLY to residues with a_hat_i != a_i^GT. Under teacher
    # forcing (Stage II/III) nothing mismatches, so the term is 0 -- matching the
    # Stage III objective, which contains no L_compat. It goes live in Stage IV.
    # Before 2026-07-22 the code applied clash+contact to EVERY residue, including
    # correctly-typed ones; that contradicted the spec and is fixed.
    "mismatch_loss": "clash",
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
    # Stage II-A side-chain-only warm-up: S_phi should condition on GT residue
    # identity/type masks, not a frozen/untrained AA head. Joint stages keep this
    # False so S_phi consumes predicted residue-type logits.
    "force_gt_type_logits": False,
    # DIRECT a-level side-chain -> backbone feedback:
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
    # DIRECT q-level (ATOM-level) side-chain -> backbone feedback:
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
