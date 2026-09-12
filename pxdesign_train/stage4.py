"""Proteo-AA bindings for the shared FaMPNN revision loop."""
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from .codesign import CoDesignState, CycleConfig, run_cycle
from .sidechain.frames import gather_backbone
from .sidechain.instantiate import instantiate_from_type_indices
from .sidechain.losses import sidechain_global_frame_aligned_loss


def cycle_config(model, rounds=None, temperature=None):
    config = model.configs.stage4
    return CycleConfig(rounds=int(config.train_rounds if rounds is None else rounds),
        decode_blocks=int(config.decode_blocks), query_fraction=float(config.query_fraction),
        whole_mask_probability=float(config.whole_mask_probability),
        temperature=float(config.temperature if temperature is None else temperature),
        sc_to_aa=bool(config.sc_to_aa), sc_to_bb=bool(config.sc_to_bb),
        packing_enabled=bool(getattr(config, "packing_enabled", True)),
        backbone_refinement_enabled=bool(getattr(config, "backbone_refinement_enabled", True)))


def _expand(x, samples, trailing):
    # Current PXDesign trainer calls one item at a time. Fail rather than flatten
    # distinct items into the sample axis; the core CoDesignState supports both.
    if x.ndim != trailing:
        raise ValueError(f"Stage IV PXDesign binding requires one unbatched item; got {tuple(x.shape)}")
    return x[None, None].expand(1, samples, *x.shape)


def make_state(model, feat, xyz, sigma, fixed_xyz, *, target_policy=None, features=None):
    if xyz.ndim != 3:
        raise ValueError("Stage IV PXDesign binding expects [sample,atom,xyz]; use item-at-a-time loader")
    if xyz.shape[0] != 1:
        raise ValueError("PXDesign pack/refine binding supports one diffusion sample per call; use gradient accumulation")
    samples, length = xyz.shape[0], feat["design_token_mask"].shape[-1]
    expand = lambda value, trailing: _expand(value.to(xyz.device), samples, trailing)
    required = ("aa_bb_atom_idx", "aa_fixed_atom37_idx", "aa_fixed_aatype", "aa_residue_mask", "fixed_atom_mask")
    for key in required:
        if key not in feat:
            raise ValueError(f"Stage IV requires strict all-residue feature {key}")
    design = expand(feat["design_token_mask"].bool(), 1)
    residue_mask = expand(feat["aa_residue_mask"].bool(), 1)
    if (design & ~residue_mask).any():
        raise ValueError("FaMPNN design positions must be protein residues")
    fixed_mask = expand(feat["fixed_atom_mask"].bool(), 1)
    # Freeze every non-design coordinate, including unresolved fixed placeholders.
    atom_design = feat["design_token_mask"].bool()[feat["atom_to_token_idx"].long()]
    fixed_mask = expand(~atom_design, 1)
    policy = target_policy or getattr(model.configs.stage4, "initial_target_policy", "fixed_context")
    if policy not in ("joint", "fixed_context"):
        raise ValueError(f"Unknown target policy {policy}")
    fixed_xyz = xyz.clone() if policy == "joint" else fixed_xyz
    xyz = torch.where(fixed_mask[0, ..., None], fixed_xyz, xyz)
    fixed_context, fixed_context_mask = gather_backbone(fixed_xyz[None], expand(feat["aa_fixed_atom37_idx"], 2))
    assigned = expand(feat["aa_fixed_aatype"].long(), 1)
    assigned = torch.where(design, 20, assigned)
    if features is None:
        h = model.backbone_features(sigma) if model.enable_sidechain else xyz.new_zeros(samples, length, 0)
        features = dict(h=h[None], sigma=sigma[None], q=model._q_skip_cache)
    h = features["h"][0]
    if h.ndim != 3 or h.shape[:2] != (samples, length):
        raise ValueError(f"Backbone features have incompatible item/sample axes: {tuple(h.shape)}")
    shape = (1, samples, length, 10)
    return CoDesignState(backbone_xyz=xyz[None], backbone_features=features,
        bb_atom_idx=expand(feat["aa_bb_atom_idx"], 2), assigned_aa=assigned,
        sc_xyz=xyz.new_zeros(*shape, 3), sc_atom_name_ids=torch.zeros(shape, device=xyz.device, dtype=torch.long),
        generation_mask=torch.zeros(shape, device=xyz.device, dtype=torch.bool),
        fixed_context_xyz=fixed_context, fixed_context_mask=fixed_context_mask,
        fixed_atom_xyz=fixed_xyz[None], fixed_atom_mask=fixed_mask, residue_mask=residue_mask,
        design_mask=design, query_mask=design, seq_visible=~design & residue_mask,
        sc_visible=~design & residue_mask, residue_index=expand(feat["residue_index"], 1),
        chain_index=expand(feat["asym_id"], 1),
        protocol=dict(initial_target_policy=policy, subsequent_target="generated_complex" if policy == "joint" else "supplied_target",
                      frame="sampler_returned" if policy == "joint" else "supplied_context"))


class ProteoAACycle:
    def __init__(self, model, feat, s_inputs, s_trunk, z_trunk):
        self.model, self.feat = model, feat
        self.s_inputs, self.s_trunk, self.z_trunk = s_inputs, s_trunk, z_trunk
        self.last_pack = None

    def pack(self, state):
        model = self.model
        if state.backbone_xyz.shape[:2] != (1, 1):
            raise ValueError("Packing requires one item and one sample per call")
        if model.sc_edm or not model.sc_predicted_frame or not model.sc_per_sigma:
            raise ValueError("Stage IV requires one-step packing with predicted frames and per-sample features")
        feat = dict(self.feat)
        # The generated branch has no label-derived inventory or observation mask.
        # Unknown supervision labels cannot change its forward or physical losses.
        ids, chemistry = instantiate_from_type_indices(state.assigned_aa[0, 0])
        feat.update(sc_atom_name_ids=ids, sc_slot_mask=chemistry,
                    sc_atom_mask=torch.zeros_like(chemistry))
        for key in ("aa_clean", "sc_gt_local", "sc_frame_R", "sc_frame_t", "sc_bb_coords",
                    "sc_frame_valid", "sc_bb_observed_mask", "sc_observed_mask", "sc_loss_mask", "sc_chemical_mask"):
            feat.pop(key, None)
        h = state.backbone_features["h"]
        logits = state.backbone_features.get("aa_logits")
        if logits is None:
            logits = torch.full((*state.assigned_aa.shape, 20), -20., device=h.device)
            logits = logits.scatter(-1, state.assigned_aa.clamp(0,19)[..., None], 20.)
        out = dict(h_res_sigma=h, h_res_candidate=h.mean(-3), aa_logits=logits,
            aa_logits_reduced=logits.mean(-3), assigned_aa=state.assigned_aa.reshape(-1, h.shape[-2]),
            sigma=state.backbone_features["sigma"], x_denoised=state.backbone_xyz)
        model._q_skip_cache = state.backbone_features.get("q")
        model.pack_backbone_state(feat, out)
        self.last_pack = out
        mask = out["sc_generation_mask"].reshape_as(state.generation_mask)
        ids = out["sc_atom_name_ids"].reshape_as(state.sc_atom_name_ids)
        ids = torch.where(mask, ids, 0)
        xyz = out["sc_pred_global"].reshape_as(state.sc_xyz)
        return state.updated(sc_xyz=torch.where(mask[..., None], xyz, 0.), sc_atom_name_ids=ids,
            generation_mask=mask, sc_visible=(state.seq_visible & ~state.design_mask) | (mask.any(-1) & state.seq_visible),
            feedback=dict(h=out["h_res_prime_reduced"],
                          a=model._a_sc_cache[0] if model._a_sc_cache is not None else None,
                          q=model._q_sc_cache[0] if model._q_sc_cache is not None else None,
                          q_idx=model._q_bb_idx_cache))

    def refine(self, state, enabled):
        model, feedback = self.model, state.feedback
        if state.backbone_xyz.shape[:2] != (1, 1):
            raise ValueError("Refinement requires one item and one sample per call")
        trunk = self.s_trunk + getattr(model, "refinement_pass_embedding", self.s_trunk.new_zeros(self.s_trunk.shape[-1])).to(self.s_trunk.dtype)
        if enabled and model.sc_hres_inject:
            trunk = trunk + model.hres_injector(feedback["h"]).to(trunk.dtype)
        model._a_sc_cache = feedback["a"] if enabled else None
        model._q_sc_cache = feedback["q"] if enabled else None
        model._q_bb_idx_cache = feedback["q_idx"] if enabled else None
        model._a_direct_active = enabled and (model.sc_a_direct or getattr(model, "sc_a_direct_pre", False))
        model._q_direct_active = enabled and model.sc_q_direct
        sigma = torch.full_like(state.backbone_features["sigma"][0], getattr(model, "sc_refinement_sigma", 0.4))
        try:
            xyz = model.diffusion_module(x_noisy=state.backbone_xyz[0], t_hat_noise_level=sigma,
                input_feature_dict=self.feat, s_inputs=self.s_inputs, s_trunk=trunk,
                z_trunk=self.z_trunk, pair_z=None, p_lm=None, c_l=None)
        finally:
            model._a_direct_active = model._q_direct_active = False
        xyz = torch.where(state.fixed_atom_mask[0,...,None], state.fixed_atom_xyz[0], xyz)
        features = capture_packing_features(model, self.feat, xyz, self.s_inputs, self.s_trunk, self.z_trunk)
        return state.on_new_backbone(xyz[None], features)


def masked_aa_objective(records, native_aa):
    if not records:
        zero = torch.zeros((), device=native_aa.device, dtype=torch.float32)
        return zero, zero
    numerator = records[0][0].sum() * 0.
    denominator = numerator.detach()
    correct = numerator.detach()
    for logits, query in records:
        labels = native_aa.expand_as(query)
        valid = query & (labels >= 0) & (labels < 20)
        ce = F.cross_entropy(logits.float().reshape(-1,20), labels.clamp(0,19).reshape(-1), reduction="none").reshape_as(query)
        numerator = numerator + torch.where(valid, ce, 0.).sum()
        denominator = denominator + valid.sum()
        correct = correct + ((logits.argmax(-1) == labels) & valid).sum()
    return numerator / denominator.clamp_min(1), correct / denominator.clamp_min(1)



def supervised_sc_forward(model, feat, labels, s_inputs, s_trunk, z_trunk):
    """Native-type packing in native frames; no sequence decoding or BB denoising target."""
    if model.sc_predicted_frame or model.sc_predicted_mask:
        raise ValueError("Supervised SC phases require GT frames and GT atom inventories")
    cfg=model.configs.stage4
    if cfg.train_rounds or cfg.sc_to_aa or cfg.sc_to_bb or cfg.backbone_refinement_enabled:
        raise ValueError("Supervised SC phases do not enable revisions or feedback")
    for key in ("aa_clean", "sc_gt_local", "sc_frame_R", "sc_frame_t", "sc_bb_coords", "sc_atom_mask",
                "sc_frame_valid", "sc_chemical_mask", "sc_bb_observed_mask"):
        if key not in feat: raise ValueError(f"Supervised SC requires native {key}")
    native=feat["aa_clean"].long()
    design=feat["design_token_mask"].bool()
    canonical=(native>=0)&(native<20)
    # Real monomer data can contain UNK or modified residues. They have no
    # unambiguous canonical atom inventory or type target, so retain them as
    # structural context while excluding them from SC ownership/supervision.
    sc_design=design&canonical
    xyz=labels["coordinate"].detach()[None]
    # The official feature pass sees XPB-masked backbone inputs. Native types
    # enter only the separate SC call below, never the backbone or FAMPNN.
    with torch.no_grad():
        features=capture_packing_features(model,feat,xyz,s_inputs,s_trunk,z_trunk)
    logits=torch.where(canonical[..., None], F.one_hot(native.clamp(0,19),20).float()*40.-20., 0.)[None,None]
    pack=dict(h_res_sigma=features["h"],h_res_candidate=features["h"].mean(-3),
        aa_logits=logits,aa_logits_reduced=logits.mean(-3),sigma=features["sigma"],
        x_denoised=xyz[None],assigned_aa=native[None],backbone_source="native")
    model._q_skip_cache=features.get("q")
    pack_feat=dict(feat,design_token_mask=sc_design)
    model.pack_backbone_state(pack_feat,pack)
    mask=feat["sc_atom_mask"].bool() & feat["sc_frame_valid"].bool()[..., None] & pack["sc_generation_mask"]
    mse=sidechain_global_frame_aligned_loss(pack["sc_pred_global"].float(),feat["sc_gt_local"].float(),
        pack["sc_frame_R"].float(),pack["sc_frame_t"].float(),mask)
    return dict(supervised_sc=True,sc_gt_mse=mse,sc_observed_atoms=mask.sum(),
        sc_skipped_noncanonical=(design&~canonical).sum(),
        sc_invalid_native_frames=(design&~feat["sc_frame_valid"].bool()).sum(),
        sc_chemical_mask=pack["sc_chemical_mask"],sc_model_mask=pack["sc_model_mask"],sc_loss_mask=mask,
        sc_pred_global=pack["sc_pred_global"],sc_atom_mask=mask,
        sc_frame_R=pack["sc_frame_R"],sc_frame_t=pack["sc_frame_t"],
        sc_input_backbone=xyz,sc_input_types=native,feature_xyz=features["feature_xyz"],
        protocol=dict(phase=str(cfg.phase),backbone_source="native",sequence_source="native",
            coordinate_targets="native_sidechains",feedback=False,feature_convention=features["convention"],
            mask_contract="chemical_model_observed_v1"))

def training_forward(model, feat, out, s_inputs, s_trunk, z_trunk):
    if getattr(model.configs.stage4, "initial_target_policy", "joint") == "fixed_context":
        atom_design = feat["design_token_mask"].bool()[feat["atom_to_token_idx"].long()]
        out["x_denoised"] = torch.where(atom_design[None,:,None], out["x_denoised"], out["x_gt_aug"])
    features = capture_packing_features(model, feat, out["x_denoised"], s_inputs, s_trunk, z_trunk)
    state = make_state(model, feat, out["x_denoised"], features["sigma"][0], out["x_gt_aug"], features=features)
    # Ensure pre-BB supervision uses the same fixed receptor as every revision.
    out["x_denoised"] = state.backbone_xyz[0]
    runtime = ProteoAACycle(model, feat, s_inputs, s_trunk, z_trunk)
    query_seed = int(torch.randint(2**31 - 1, ()).item())
    query_rng = torch.Generator(device=state.backbone_xyz.device).manual_seed(query_seed)
    sequence_rng = torch.Generator(device=state.backbone_xyz.device).manual_seed(query_seed + 1)
    state, records = run_cycle(state, model.aa_head, runtime.pack, runtime.refine, cycle_config(model),
                              generator=query_rng, sequence_generator=sequence_rng)
    generated_pack = runtime.last_pack
    native = feat["aa_clean"].to(state.assigned_aa.device)[None,None]
    out["stage4_aa_pre"], out["stage4_recovery_pre"] = masked_aa_objective(records["initial"], native)
    out["stage4_aa_revision"], out["stage4_recovery_revision"] = masked_aa_objective(records["revisions"], native)
    generated_pack = generated_pack or {}
    out["stage4_phys"] = generated_pack.get("sc_pack_val", out["x_denoised"].sum() * 0.)
    # Native-AA auxiliary branch: fresh template init and detached BB features;
    # its outputs never enter the generated loop or any AA context.
    if not model.enable_sidechain or not cycle_config(model).packing_enabled:
        out["stage4_sc_aux"] = out["x_denoised"].sum() * 0.
        out.update(post_pred_coordinate=state.backbone_xyz[0], post_gt_coordinate_aug=out["x_gt_aug"],
                   codesign_state=state, codesign_trace=records["trace"])
        return out
    def detach(value):
        if torch.is_tensor(value): return value.detach()
        if isinstance(value, dict): return {k:detach(v) for k,v in value.items()}
        if isinstance(value, tuple): return tuple(detach(v) for v in value)
        if isinstance(value, list): return [detach(v) for v in value]
        return value
    aux_features = detach(state.backbone_features)
    native_types = torch.where(state.design_mask, native.expand_as(state.assigned_aa), state.assigned_aa)
    aux_features["aa_logits"] = F.one_hot(native_types.clamp(0,19), 20).to(state.backbone_xyz.dtype) * 40. - 20.
    aux_state = replace(state, backbone_xyz=state.backbone_xyz.detach(), backbone_features=aux_features,
                        assigned_aa=native_types)
    runtime.pack(aux_state)
    aux = runtime.last_pack
    mask = feat["sc_atom_mask"].bool() & aux["sc_generation_mask"]
    out["stage4_sc_aux"] = sidechain_global_frame_aligned_loss(aux["sc_pred_global"],
        feat["sc_gt_local"], aux["sc_frame_R"], aux["sc_frame_t"], mask)
    out.update(post_pred_coordinate=state.backbone_xyz[0], post_gt_coordinate_aug=out["x_gt_aug"],
        codesign_state=state, codesign_trace=records["trace"])
    return out


def apply_phase(model):
    """Frozen parameters retain input-coordinate gradients and stay in eval mode."""
    from .checkpoints import FEEDBACK_PREFIXES, BACKBONE_PREFIXES
    cfg = model.configs.stage4
    phase = str(cfg.phase)
    phases = ("baseline", "sc_warmup", "sc_complex_adapt", "sc_adapt", "feedback_adapt", "aa_adapt", "joint_adapt", "IV-0", "IV-A", "IV-B", "IV-C")
    if phase not in phases:
        raise ValueError(f"Unknown phase {phase}")
    bb = tuple(p for p in (cfg.bb_trainable_prefixes or ()) if p)
    if any(not prefix.startswith(BACKBONE_PREFIXES) for prefix in bb):
        raise ValueError("Backbone trainable prefixes must select the backbone or condition encoder")
    selected_feedback = tuple(getattr(cfg, "feedback_trainable_prefixes", FEEDBACK_PREFIXES))
    if any(not prefix.startswith(FEEDBACK_PREFIXES) for prefix in selected_feedback):
        raise ValueError("Feedback prefixes must select feedback modules")
    for name, param in model.named_parameters():
        aa = name.startswith("aa_head.")
        sc = name.startswith("sidechain_module.")
        feedback = bool(selected_feedback and name.startswith(selected_feedback))
        if phase in ("baseline", "IV-0"):
            enabled = False
        elif phase in ("sc_warmup", "sc_complex_adapt", "sc_adapt"):
            enabled = sc
        elif phase == "feedback_adapt":
            enabled = feedback or (sc and bool(getattr(cfg, "train_sc", False)))
        elif phase in ("aa_adapt", "IV-A"):
            enabled = aa or (sc and phase == "aa_adapt" and bool(getattr(cfg, "train_sc", False)))
        else:
            enabled = aa or sc or feedback or bool(bb and name.startswith(bb))
        param.requires_grad_(enabled)
    for module in model.children():
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()


def optimizer_groups(model):
    from .checkpoints import FEEDBACK_PREFIXES
    cfg = model.configs.stage4
    groups = {name: dict(name=name, params=[], lr=float(getattr(cfg, name + "_lr", cfg.bb_lr)))
              for name in ("aa", "sc", "feedback", "bb")}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = "aa" if name.startswith("aa_head.") else "sc" if name.startswith("sidechain_module.") else "feedback" if name.startswith(FEEDBACK_PREFIXES) else "bb"
            groups[group]["params"].append(param)
    return [group for group in groups.values() if group["params"]]


def capture_packing_features(model, feat, xyz, s_inputs, s_trunk, z_trunk):
    """Fresh deterministic feature convention shared by adaptation and inference.

    Denoise the current backbone at a positive *conditioning* sigma, capture h/q,
    discard predicted coordinates, and retain the exact input backbone/frame.
    This sigma is not a measured physical corruption level.
    """
    sigma = xyz.new_full(xyz.shape[:-2], float(getattr(model.configs.stage4, "feature_sigma", 0.4)))
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("Feature conditioning sigma must be positive and finite")
    if model.enable_sidechain:
        model._a_direct_active = model._q_direct_active = False
        model.diffusion_module(x_noisy=xyz, t_hat_noise_level=sigma,
            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            pair_z=None, p_lm=None, c_l=None)
        h = model.backbone_features(sigma)
    else:
        h = xyz.new_zeros(*xyz.shape[:-2], feat["design_token_mask"].numel(), 0)
    return dict(h=h[None], sigma=sigma[None], q=model._q_skip_cache,
                feature_xyz=xyz[None], feature_frame="current_backbone_global", convention="fresh_positive_sigma_v1")


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    paths = sorted([root/"scripts/training/train_protenix_monomer.py",
        *root.glob("pxdesign_train/**/*.py"), *root.glob("scripts/training/*stage4*"),
        *root.glob("scripts/evaluation/*stage4*"), *root.glob("scripts/utilities/*stage4*")])
    checksum = hashlib.sha256()
    for path in paths:
        checksum.update(str(path.relative_to(root)).encode())
        checksum.update(path.read_bytes())
    return dict(proteoaa_revision=revision, implementation_sha256=checksum.hexdigest())


def checkpoint_identity(model):
    cfg = model.configs.stage4
    return dict(**model.aa_head.identity, **implementation_identity(), phase=str(cfg.phase),
        sc_mask_contract="chemical_model_observed_v1",
        cycle=asdict(cycle_config(model)), inference_rounds=int(cfg.inference_rounds),
        mask_policy="query-X-hide-SC-before-encoding-v1", sidechain_edm=False,
        optimizer_policy="joint-AA-SC-BB-v1",
        optimizer={key: float(getattr(cfg, key)) for key in ("aa_lr", "sc_lr", "bb_lr")},
        bb_trainable_prefixes=list(cfg.bb_trainable_prefixes or ()),
        objectives={key: float(getattr(cfg,key)) for key in ("weight_aa_pre", "weight_aa_revision", "weight_sc_aux", "weight_physical")},
        template_provider=str(getattr(model, "sc_template_provider", "none")),
        diffusion_samples=int(model.configs.training.diffusion_batch_size),
        feedback={key: bool(getattr(model, key, False)) for key in ("sc_a_direct", "sc_a_direct_pre", "sc_q_direct", "sc_hres_inject")})


@torch.no_grad()
def generate(model, input_feature_dict, N_step=400, temperature=0.0,
             refinement_steps=None, seed=0, allow_one_round_ablation=False,
             backbone_sampler=None, initial_target_policy=None, packing_enabled=None,
             backbone_refinement_enabled=None):
    from protenix.model.protenix import update_input_feature_dict
    from .structure import assemble_atoms
    from .initial_sampling import sample_initial, RandomStream
    config = model.configs.stage4
    rounds = int(config.inference_rounds if refinement_steps is None else refinement_steps)
    if rounds < 0 or N_step < 1:
        raise ValueError("Rounds must be nonnegative and backbone steps positive")
    sampler = backbone_sampler or getattr(config, "backbone_sampler", "pxdesign_native")
    policy = initial_target_policy or getattr(config, "initial_target_policy", "joint")
    cfg = cycle_config(model, rounds=rounds, temperature=temperature)
    cfg = replace(cfg,
        packing_enabled=cfg.packing_enabled if packing_enabled is None else packing_enabled,
        backbone_refinement_enabled=cfg.backbone_refinement_enabled if backbone_refinement_enabled is None else backbone_refinement_enabled)
    if cfg.packing_enabled and not model.enable_sidechain:
        raise ValueError("Packing is enabled but the model has no SC module")
    model.eval()
    backbone_rng, pack_rng, feature_rng = RandomStream(seed), RandomStream(seed+3), RandomStream(seed+4)
    feat = dict(input_feature_dict)
    for key in ("aa_clean", "sc_gt_local", "aa_loss_mask", "aa_corruption_mask", "sc_atom_mask", "sc_slot_mask", "sc_atom_name_ids", "sc_frame_R", "sc_frame_t", "sc_bb_coords",
                "sc_frame_valid", "sc_bb_observed_mask", "sc_observed_mask", "sc_loss_mask", "sc_chemical_mask"):
        feat.pop(key, None)
    design = feat["design_token_mask"].bool()
    from .sampler import build_aa20_to_restype36
    _, xpb = build_aa20_to_restype36()
    feat["restype"] = feat["restype"].clone()
    feat["restype"][design] = 0.
    feat["restype"][design, xpb] = 1.
    with backbone_rng.use():
        feat = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
        feat = update_input_feature_dict(feat)
        model._a_sc_cache = model._q_sc_cache = None
        model._a_direct_active = model._q_direct_active = False
        model._q_inject_calls = {}
        s_inputs, s_trunk, z_trunk = model.get_condition_embedding(feat)
        schedule = model.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype)
        xyz, captured = sample_initial(model, feat, s_inputs, s_trunk, z_trunk, schedule,
                                       sampler=sampler, target_policy=policy)
    # An explicit fresh pass preserves xyz exactly. Native cached features are
    # retained as observations, never mislabeled as final-coordinate features.
    with feature_rng.use():
        features = capture_packing_features(model, feat, xyz, s_inputs, s_trunk, z_trunk)
    reference = xyz if policy == "joint" else feat["fixed_atom_xyz"].to(xyz)[None]
    state = make_state(model, feat, xyz, features["sigma"][0], reference, target_policy=policy, features=features)
    state = replace(state, protocol=dict(state.protocol, backbone_sampler=sampler, packing_enabled=cfg.packing_enabled))
    runtime = ProteoAACycle(model, feat, s_inputs, s_trunk, z_trunk)
    def pack(state):
        with pack_rng.use():
            return runtime.pack(state)
    def refine(state, enabled):
        with feature_rng.use():
            return runtime.refine(state, enabled)
    query_rng = torch.Generator(device=xyz.device).manual_seed(seed+1)
    sequence_rng = torch.Generator(device=xyz.device).manual_seed(seed+2)
    state, records = run_cycle(state, model.aa_head, pack, refine, cfg,
                              generator=query_rng, sequence_generator=sequence_rng)
    atoms = assemble_atoms(state, feat)
    logits = state.backbone_features.get("aa_logits")
    # Refinement replaces backbone features; gather committed logits from the
    # final decoding calls if the final backbone no longer carries them.
    if logits is None:
        logits = torch.zeros(*state.assigned_aa.shape, 20, device=xyz.device)
        for value, mask in records["initial"] + records["revisions"]:
            logits = torch.where(mask[...,None], value, logits)
    metadata = dict(seed=seed, temperature=temperature, cycle=asdict(cfg), protocol=state.protocol,
        component_origins=getattr(model, "component_origins", {}), stop_reason=records["stop_reason"],
        rng_streams=dict(backbone=seed, query=seed+1, sequence=seed+2, sidechain=seed+3, feature=seed+4),
        feature_convention="fresh_positive_sigma_v1", feature_sigma=float(features["sigma"].flatten()[0]),
        rounds=[{k:v for k,v in row.items() if k not in ("query_mask", "aa_changes")} for row in records["trace"]])
    return dict(coordinate=state.backbone_xyz[0,0], sequence=torch.where(design,state.assigned_aa[0,0],-1),
        aa_logits=logits[0,0], aa_probs=logits[0,0].softmax(-1),
        sidechain=dict(coordinate=state.sc_xyz[0,0], atom_name_ids=state.sc_atom_name_ids[0,0], mask=state.generation_mask[0,0]),
        has_full_atom_sidechain=cfg.packing_enabled, sampler_mode=sampler,
        state=state, atoms=atoms, trajectory=records["trace"], aa_records=records, metadata=metadata,
        initial_coordinate=xyz[0], native_observation=captured)
