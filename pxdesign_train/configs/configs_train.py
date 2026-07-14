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
    "init_sigma": 1.0,
    # ---- Context (receptor / motif / ligand) awareness ----
    # SPEC, not an option. The paper requires it in six places, e.g.
    #   - "Operating in the global frame allows side-chain atoms to directly attend
    #     to neighboring residues, receptor atoms, fixed motifs, ligands, and other
    #     spatial context.";
    #   - clash covers "side-chain--backbone, side-chain--side-chain, and
    #     side-chain--context atom pairs";
    #   - contact "promotes compatible interactions with ... receptor atoms ...";
    #   - the appendix says our own default S_phi's "inter-residue and context
    #     attention capture ... side-chain--receptor interactions".
    # Before this, S_phi could see NONE of it: every non-binder token was an
    # all-masked key in the cross-residue attention, clash scored side-chain <->
    # side-chain pairs only, and contact's reference was binder-only (and unmasked,
    # so its phantom rows silently zeroed the penalty). Emitting GLOBAL coordinates
    # was necessary for receptor awareness but never sufficient — no path carried
    # the receptor in.
    # A switch so it can be ablated, and so the Stage II-A warmup can run without it:
    # warmup uses GT frames, which live in the RAW coordinate frame, while x_denoised
    # (our only source of receptor atoms) lives in the AUGMENTED one — scoring one
    # against the other would be a frame bug, not a conservative approximation.
    "context_aware": True,
    "context_radius": 10.0,      # A; atoms beyond this from any binder CA are dropped
    "context_max_atoms": 4096,   # hard cap on the context set (memory bound)
    # Overleaf paragraph 221 (template-anchored leakage-free initialization):
    # start side-chain denoising from the type-conditioned IDEAL template
    # perturbed by sigma_T, rather than isotropic Gaussian noise. An isotropic
    # Gaussian is rotation-invariant, so pushing it through the predicted frame
    # F_hat carries no backbone-orientation information and S_phi cannot learn
    # where to place atoms in GLOBAL space. The (anisotropic) template does.
    # ============================== STATUS: ENABLED ==============================
    # Overleaf par.221 specifies this as a FORMULA, and the 0714 appendix
    # ("Residue-Specific Side-Chain Template Construction") specifies how mu_ideal is
    # actually built. Yifei's code (069645a) implements par.221's equation with
    # mu_ideal == 0: its only initializer is gaussian_init_local(mask, sigma), whose
    # signature never even receives the residue type, so it structurally cannot produce
    # a residue-specific template.
    #
    # THIS WAS OFF UNTIL 2026-07-14, waiting for one thing. In the 2026-07-09 meeting
    # FangWu asked Jiaming to research how the ideal template should be built --
    # "从统计分析上去做...还是说已经有一些比较成熟的方法" -- because the CCD table has the
    # right bond lengths and angles but ONE ARBITRARY chi, which is 2-3 A from real side
    # chains on the multi-chi residues. That research landed as the 0714 appendix:
    # a mature method exists, namely a BACKBONE-DEPENDENT ROTAMER LIBRARY (Dunbrack
    # BBDEP2010). It is now implemented, so the blocker is gone and the spec'd
    # initialization is the default.
    #
    # WHAT RUNS NOW (0714 appendix, Steps 1-3):
    #   Step 1  chi_constants.py   A_sc, K_i and G_ideal (connectivity, bond lengths/angles,
    #                              rigid groups) from the CCD + Protenix's AF chi tables.
    #   Step 2  rotamers.py        chi ~ p(r | a_hat, phi_hat, psi_hat) from BBDEP2010,
    #                              phi/psi being dihedrals of the PREDICTED backbone.
    #   Step 3  buildsc.py         BuildSC: pose G_ideal at those torsions, in the local
    #                              frame. Bond lengths/angles are preserved exactly.
    #   init.py                    y_T = mu_ideal[a, chi, j] + sigma_T * eps
    #   model.py / cogenerate.py   x_T = F_hat y_T  (training and sampling mirror each other)
    #
    # Measured on 2790 residues of 33 real chains, mean local-frame distance of the
    # initialization from the true side chain: 2.887 A (Gaussian, i.e. what runs today)
    # -> 1.277 A (this). scripts/eval_template_quality.py regenerates the numbers.
    #
    # Set to False to restore Yifei's isotropic Gaussian init (the A/B baseline).
    # ============================================================================
    "template_init": True,
    # Which mu_ideal construction to use:
    #   "dunbrack_mode" BuildSC, chi = argmax_r p(r | a, phi, psi)       [default]
    #   "dunbrack"      BuildSC, chi ~ Categorical(p(r | a, phi, psi))
    #   "ccd"           the static one-conformer CCD table (pre-0714 baseline)
    #
    # The appendix permits deterministic OR sampled selection and does not choose. We
    # measured both on 2790 real residues (scripts/eval_template_quality.py), mean
    # local-frame RMSD of the template from the true side chain:
    #
    #     gaussian (mu=0, the pre-0714 default) .. 2.887 A     chi1 recovery   n/a
    #     ccd      (static, one arbitrary chi) ... 1.662 A                    46.3 %
    #     dunbrack (sampled) .................... 1.487 A                    61.0 %
    #     dunbrack_mode (argmax) ................ 1.277 A                    68.7 %
    #     oracle   (true chi; lower bound) ...... 0.333 A                   100.0 %
    #
    # Mode wins on 18 of 19 side-chain-bearing types. Sampling is WORSE than the old
    # static table on the high-entropy residues (GLN -19%, GLU -16%, LYS -6%, ARG -4%),
    # because their rotamer distributions are nearly flat (GLN's modal rotamer carries
    # only 12% of the mass over 108 rotamers), so a draw is usually far from the truth.
    # As an INITIALIZATION, lower error is the whole point, so mode is the default.
    #
    # OPEN: sampling is the more natural prior for a *generative* model -- it makes the
    # init distribution match p(r | a, phi, psi) instead of collapsing it to the mode.
    # That argument is about sample diversity, which this geometric benchmark cannot
    # measure; it needs a training A/B. Nothing here forecloses it -- flip this string.
    "template_provider": "dunbrack_mode",
    # ABLATION CANDIDATE -- default OFF, i.e. the CA-anchored global-head baseline.
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
    # ABLATION CANDIDATE -- default OFF, i.e. S_phi's own noisy
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
