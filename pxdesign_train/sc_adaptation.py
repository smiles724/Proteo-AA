"""SC-only phase contracts and explicit coordinate-source dispatch.

The protocol is opt-in so historical checkpoints retain their original forward.
Reconstruction labels are local-geometry pseudo-targets, never free-sample labels.
"""
from dataclasses import replace
import math
import torch
import torch.nn.functional as F

PROTOCOL = "sc_only_v1"
DEFAULTS = dict(adaptation_protocol="legacy", native_fraction=0.5,
    paired_fraction=0.5, full_sample_fraction=0.0,
    reconstruction_sigmas="0.4,1,2,4", reconstruction_max_ca_error=3.0,
    reconstruction_max_bond_error=0.3)


def configure_runtime(config):
    """Honor this protocol's numeric settings in training and checkpoint evaluation."""
    if getattr(getattr(config, "stage4", None), "adaptation_protocol", "legacy") == PROTOCOL:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        config.training.deterministic_algorithms = True


def validate_phase(config):
    """Resolve the whole destination objective without changing SC architecture."""
    cfg = config.stage4
    phase = str(cfg.phase)
    if phase not in ("sc_warmup", "sc_complex_adapt", "sc_adapt"):
        raise ValueError("SC-only recipes require an SC-only phase")
    for key, value in DEFAULTS.items():
        if key not in cfg:
            cfg[key] = value
    if cfg.train_rounds or cfg.inference_rounds or cfg.sc_to_aa or cfg.sc_to_bb or cfg.backbone_refinement_enabled:
        raise ValueError("SC-only adaptation requires zero rounds and disabled feedback/refinement")
    if not cfg.packing_enabled or int(config.training.diffusion_batch_size) != 1:
        raise ValueError("SC-only adaptation requires packing and one diffusion sample")
    if cfg.initial_target_policy != "joint":
        raise ValueError("This adaptation protocol preserves the joint reconstructed complex")
    if not math.isfinite(float(cfg.weight_physical)) or cfg.weight_physical < 0:
        raise ValueError("Physical coefficient must be finite and nonnegative")
    if phase == "sc_warmup" and cfg.weight_physical:
        raise ValueError("sc_warmup forbids physical loss")
    if cfg.weight_aa_pre or cfg.weight_aa_revision:
        raise ValueError("SC-only phases require zero AA objective coefficients")
    for key in ("weight_mse", "weight_lddt", "weight_disto", "weight_bb_post", "weight_aa", "weight_aa_post"):
        if key in config.loss:
            config.loss[key] = 0.0
    native = phase != "sc_adapt"
    config.sidechain.predicted_frame = not native
    config.sidechain.predicted_mask = not native
    config.sidechain.force_gt_type_logits = native
    if config.sidechain.edm:
        raise ValueError("SC-only adaptation preserves the one-step edm=false packer")
    # The packer returns an unweighted term; the trainer applies the coefficient once.
    config.sidechain.pack_loss = float(cfg.weight_physical)
    config.sidechain.pack_arm = "clash"
    fractions = [float(cfg[k]) for k in ("native_fraction", "paired_fraction", "full_sample_fraction")]
    if any(not math.isfinite(x) or x < 0 for x in fractions) or abs(sum(fractions)-1) > 1e-8:
        raise ValueError("Backbone-source fractions must be nonnegative and sum to one")
    if fractions[2] > 0.1 + 1e-8:
        raise ValueError("Experimental full samples are limited to 10% with supervised replay")
    sigmas = [float(x) for x in str(cfg.reconstruction_sigmas).split(",")]
    if not sigmas or any(not math.isfinite(x) or x <= 0 for x in sigmas):
        raise ValueError("Reconstruction sigma panel must be finite and positive")
    if not cfg.native_sc_augmentation:
        raise ValueError("These adaptation recipes require explicit native rigid augmentation")
    return config


def choose_source(cfg, *, seed):
    rng = torch.Generator().manual_seed(int(seed))
    weights = torch.tensor([cfg.native_fraction, cfg.paired_fraction, cfg.full_sample_fraction])
    return ("native", "paired_reconstruction", "full_sample")[int(torch.multinomial(weights, 1, generator=rng))]


def check_source(feat, labels):
    source = feat.get("backbone_source")
    if source not in ("native", "paired_reconstruction", "full_sample"):
        raise ValueError("Explicit backbone_source is required for SC adaptation")
    if source == "full_sample":
        forbidden = ("aa_clean", "sc_gt_local", "sc_atom_mask", "sc_observed_mask", "sc_loss_mask")
        if labels or any(key in feat for key in forbidden):
            raise ValueError("Full samples must not carry native AA/SC labels")
        provenance = feat.get("backbone_provenance", {})
        for key in ("seed", "checkpoint_sha256", "sampler", "steps", "target_policy"):
            if key not in provenance:
                raise ValueError(f"Cached full sample lacks provenance: {key}")
        if provenance["sampler"] != "pxdesign_native" or provenance["target_policy"] != "joint":
            raise ValueError("Full-sample adaptation requires the official joint native sampler")
        if "cached_backbone_xyz" not in feat:
            raise ValueError("Full samples require cached coordinates")
    else:
        for key in ("aa_clean", "sc_gt_local", "sc_atom_mask", "sc_frame_valid"):
            if key not in feat:
                raise ValueError(f"Paired/native adaptation requires {key}")
        if "coordinate" not in labels or "coordinate_mask" not in labels:
            raise ValueError("Paired/native adaptation requires native coordinates and observations")
    return source


def paired_quality(feat, native_xyz, xyz, cfg):
    """Per-residue gate: observed valid frames, CA displacement and NCAC geometry.

    Coordinates share one augmentation frame. No independent alignment or target
    replacement hides poor reconstruction. Report exclusions instead of retrying.
    """
    idx = feat["aa_bb_atom_idx"].long()[..., :3]
    bb = xyz[idx.clamp_min(0)].float()
    native_bb = native_xyz[idx.clamp_min(0)].float()
    ca_error = (bb[..., 1, :] - native_bb[..., 1, :]).norm(dim=-1)
    bonds = torch.stack(((bb[..., 0, :]-bb[..., 1, :]).norm(dim=-1),
                         (bb[..., 2, :]-bb[..., 1, :]).norm(dim=-1)), -1)
    bond_error = (bonds - bonds.new_tensor([1.46, 1.53])).abs().amax(-1)
    valid = feat["sc_frame_valid"].bool() & (idx >= 0).all(-1)
    valid &= torch.isfinite(bb).all(dim=(-1, -2))
    valid &= ca_error <= float(cfg.reconstruction_max_ca_error)
    valid &= bond_error <= float(cfg.reconstruction_max_bond_error)
    return valid, ca_error, bond_error


def adaptation_forward(model, feat, labels, s_inputs, s_trunk, z_trunk):
    from .stage4 import capture_packing_features, make_state, ProteoAACycle, cycle_config
    from .codesign import decode
    from .sidechain.losses import sidechain_global_frame_aligned_loss
    source = check_source(feat, labels)
    cfg = model.configs.stage4
    seed = int(feat.get("input_seed", 0))
    device = s_inputs.device
    rng = torch.Generator(device=device).manual_seed(seed + 101)
    sigma = float(cfg.feature_sigma)
    with torch.no_grad():
        if source == "full_sample":
            xyz = feat["cached_backbone_xyz"].detach().float()
            if xyz.ndim != 2 or xyz.shape[-1] != 3:
                raise ValueError("Cached coordinates must have shape [atom,3]")
        else:
            xyz = labels["coordinate"].detach().float()
            observed = labels["coordinate_mask"].bool() & torch.isfinite(xyz).all(-1)
            xyz = torch.where(observed[...,None], xyz, 0.)
            if source == "paired_reconstruction":
                panel = [float(x) for x in cfg.reconstruction_sigmas.split(",")]
                sigma = float(feat.get("reconstruction_sigma", panel[int(torch.randint(len(panel), (), generator=rng, device=device))]))
                if sigma <= 0 or not math.isfinite(sigma):
                    raise ValueError("Reconstruction sigma must be positive and finite")
                noisy = xyz + sigma * torch.randn(xyz.shape, device=device, generator=rng)
                prediction = model.diffusion_module(x_noisy=noisy[None], t_hat_noise_level=xyz.new_tensor([sigma]),
                    input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                    pair_z=None, p_lm=None, c_l=None)
                if prediction.shape != (1, *xyz.shape):
                    raise ValueError("Paired reconstruction must return exactly one [atom,3] sample")
                xyz = prediction[0].float()
        features = capture_packing_features(model, feat, xyz[None], s_inputs, s_trunk, z_trunk)
        state = make_state(model, feat, xyz[None], features["sigma"][0], xyz[None],
                           target_policy="joint", features=features)
        state, _ = decode(state, model.aa_head, cycle_config(model),
            generator=torch.Generator(device=device).manual_seed(seed+211),
            sequence_generator=torch.Generator(device=device).manual_seed(seed+307))
    runtime = ProteoAACycle(model, feat, s_inputs, s_trunk, z_trunk)
    state = runtime.pack(state)
    generated = runtime.last_pack
    physical = generated.get("sc_pack_val", generated["sc_pred_global"].sum()*0.).float()
    if float(cfg.weight_physical) > 0 and "sc_pack_val" not in generated:
        raise RuntimeError("Generated physical objective requested but packer did not compute it")
    metrics = {}
    if not model.training:
        from .sidechain.metrics import diagnose_packing
        metrics = {"generated/"+key:value for key,value in diagnose_packing(feat, generated, state.assigned_aa, xyz).items()}
    aux_loss = generated["sc_pred_global"].float().sum()*0.
    count = torch.zeros((), device=device)
    excluded = count.clone()
    ca_error = count.clone()
    if source != "full_sample":
        native = feat["aa_clean"].long()
        canonical = (native >= 0) & (native < 20)
        eligible = feat["sc_frame_valid"].bool() & canonical
        if source == "paired_reconstruction":
            quality, ca, _ = paired_quality(feat, labels["coordinate"], xyz, cfg)
            eligible &= quality
            design = feat["design_token_mask"].bool()
            ca_idx = feat["aa_bb_atom_idx"][...,1].long()
            ca_valid = design & (ca_idx >= 0) & labels["coordinate_mask"].bool()[ca_idx.clamp_min(0)] & torch.isfinite(ca)
            ca_error = ca[ca_valid].mean() if ca_valid.any() else xyz.new_zeros(())
        design = feat["design_token_mask"].bool()
        excluded = (design & ~eligible).sum()
        # A separate runtime owns the auxiliary tensors. Never mutate generated state.
        aux_feat = dict(feat, design_token_mask=design & canonical, sc_init_seed=seed+503)
        aux_types = torch.where(state.design_mask, native.clamp(0, 19)[None, None], state.assigned_aa)
        aux_features = dict(features, aa_logits=F.one_hot(aux_types.clamp(0,19),20).float()*40.-20.)
        aux_state = replace(state, backbone_xyz=state.backbone_xyz.detach(),
            backbone_features=aux_features, assigned_aa=aux_types,
            design_mask=state.design_mask & canonical[None,None])
        auxiliary = ProteoAACycle(model, aux_feat, s_inputs, s_trunk, z_trunk)
        auxiliary.pack(aux_state)
        pack = auxiliary.last_pack
        mask = feat["sc_atom_mask"].bool() & eligible[...,None] & design[...,None] & pack["sc_generation_mask"]
        aux_loss = sidechain_global_frame_aligned_loss(pack["sc_pred_global"].float(),
            feat["sc_gt_local"].float(), pack["sc_frame_R"].float(), pack["sc_frame_t"].float(), mask)
        count = mask.sum()
        if not model.training:
            metrics.update({"auxiliary/"+key:value for key,value in diagnose_packing(feat, pack, native, xyz, observed=mask).items()})
    return dict(sc_adaptation=True, sc_aux=aux_loss, sc_physical=physical,
        packing_metrics=metrics,
        sc_observed_atoms=count, sc_excluded_residues=excluded, reconstruction_ca_error=ca_error,
        codesign_state=state, generated_pack=generated,
        protocol=dict(backbone_source=source, reconstruction_sigma=sigma if source == "paired_reconstruction" else None,
            backbone_checkpoint_sha256=getattr(model, "component_origins", {}).get("backbone", {}).get("sha256"),
            input_seed=seed, feature_sigma=float(cfg.feature_sigma), feature_convention=features["convention"],
            coordinate_targets="none" if source == "full_sample" else "native_local" if source == "native" else "transported_native_pseudo_targets",
            sequence_source="frozen_fampnn", target_source=source, feedback=False))
