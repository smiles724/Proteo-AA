"""Proteo-AA bindings for the shared FaMPNN revision loop."""
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from .aa import uses_codesign  # noqa: F401 - re-exported for stage4 callers
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
        sc_to_aa=bool(config.sc_to_aa), sc_to_bb=bool(config.sc_to_bb))


def _expand(x, samples, trailing):
    # Current PXDesign trainer calls one item at a time. Fail rather than flatten
    # distinct items into the sample axis; the core CoDesignState supports both.
    if x.ndim != trailing:
        raise ValueError(f"Stage IV PXDesign binding requires one unbatched item; got {tuple(x.shape)}")
    return x[None, None].expand(1, samples, *x.shape)


def make_state(model, feat, xyz, sigma, fixed_xyz):
    if xyz.ndim != 3:
        raise ValueError("Stage IV PXDesign binding expects [sample,atom,xyz]; use item-at-a-time loader")
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
    xyz = torch.where(fixed_mask[0, ..., None], fixed_xyz, xyz)
    fixed_context, fixed_context_mask = gather_backbone(fixed_xyz[None], expand(feat["aa_fixed_atom37_idx"], 2))
    assigned = expand(feat["aa_fixed_aatype"].long(), 1)
    assigned = torch.where(design, 20, assigned)
    h = model.backbone_features(sigma)
    if h.ndim != 3 or h.shape[:2] != (samples, length):
        raise ValueError(f"Backbone features have incompatible item/sample axes: {tuple(h.shape)}")
    shape = (1, samples, length, 10)
    return CoDesignState(backbone_xyz=xyz[None], backbone_features={"h": h[None], "sigma": sigma[None], "q": model._q_skip_cache},
        bb_atom_idx=expand(feat["aa_bb_atom_idx"], 2), assigned_aa=assigned,
        sc_xyz=xyz.new_zeros(*shape, 3), sc_atom_name_ids=torch.zeros(shape, device=xyz.device, dtype=torch.long),
        generation_mask=torch.zeros(shape, device=xyz.device, dtype=torch.bool),
        fixed_context_xyz=fixed_context, fixed_context_mask=fixed_context_mask,
        fixed_atom_xyz=fixed_xyz[None], fixed_atom_mask=fixed_mask, residue_mask=residue_mask,
        design_mask=design, query_mask=design, seq_visible=~design & residue_mask,
        sc_visible=~design & residue_mask, residue_index=expand(feat["residue_index"], 1),
        chain_index=expand(feat["asym_id"], 1))


class ProteoAACycle:
    def __init__(self, model, feat, s_inputs, s_trunk, z_trunk):
        self.model, self.feat = model, feat
        self.s_inputs, self.s_trunk, self.z_trunk = s_inputs, s_trunk, z_trunk
        self.last_pack = None

    def pack(self, state):
        model = self.model
        if model.sc_edm or not model.sc_predicted_frame or not model.sc_per_sigma:
            raise ValueError("Stage IV requires one-step packing with predicted frames and per-sample features")
        feat = dict(self.feat)
        # The generated branch has no label-derived inventory or observation mask.
        # Unknown supervision labels cannot change its forward or physical losses.
        ids, chemistry = instantiate_from_type_indices(state.assigned_aa[0, 0])
        feat.update(sc_atom_name_ids=ids, sc_slot_mask=chemistry,
                    sc_atom_mask=torch.zeros_like(chemistry))
        for key in ("aa_clean", "sc_gt_local", "sc_frame_R", "sc_frame_t", "sc_bb_coords"):
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
        trunk = self.s_trunk + model.refinement_pass_embedding.to(self.s_trunk.dtype)
        if enabled and model.sc_hres_inject:
            trunk = trunk + model.hres_injector(feedback["h"]).to(trunk.dtype)
        model._a_sc_cache = feedback["a"] if enabled else None
        model._q_sc_cache = feedback["q"] if enabled else None
        model._q_bb_idx_cache = feedback["q_idx"] if enabled else None
        model._a_direct_active = enabled and (model.sc_a_direct or getattr(model, "sc_a_direct_pre", False))
        model._q_direct_active = enabled and model.sc_q_direct
        sigma = torch.full_like(state.backbone_features["sigma"][0], model.sc_refinement_sigma)
        try:
            xyz = model.diffusion_module(x_noisy=state.backbone_xyz[0], t_hat_noise_level=sigma,
                input_feature_dict=self.feat, s_inputs=self.s_inputs, s_trunk=trunk,
                z_trunk=self.z_trunk, pair_z=None, p_lm=None, c_l=None)
        finally:
            model._a_direct_active = model._q_direct_active = False
        features = dict(h=model.backbone_features(sigma)[None], sigma=sigma[None], q=model._q_skip_cache)
        return state.on_new_backbone(xyz[None], features)


def masked_aa_objective(records, native_aa):
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


def training_forward(model, feat, out, s_inputs, s_trunk, z_trunk):
    state = make_state(model, feat, out["x_denoised"], out["sigma"], out["x_gt_aug"])
    # Ensure pre-BB supervision uses the same fixed receptor as every revision.
    out["x_denoised"] = state.backbone_xyz[0]
    runtime = ProteoAACycle(model, feat, s_inputs, s_trunk, z_trunk)
    state, records = run_cycle(state, model.aa_head, runtime.pack, runtime.refine, cycle_config(model))
    generated_pack = runtime.last_pack
    native = feat["aa_clean"].to(state.assigned_aa.device)[None,None]
    out["stage4_aa_pre"], out["stage4_recovery_pre"] = masked_aa_objective(records["initial"], native)
    out["stage4_aa_revision"], out["stage4_recovery_revision"] = masked_aa_objective(records["revisions"], native)
    out["stage4_phys"] = generated_pack.get("sc_pack_val", out["x_denoised"].sum() * 0.)
    # Native-AA auxiliary branch: fresh template init and detached BB features;
    # its outputs never enter the generated loop or any AA context.
    aux_features = {key: value.detach() if torch.is_tensor(value) else value for key,value in state.backbone_features.items()}
    aux_state = replace(state, backbone_xyz=state.backbone_xyz.detach(), backbone_features=aux_features,
                        assigned_aa=torch.where(state.design_mask, native.expand_as(state.assigned_aa), state.assigned_aa))
    runtime.pack(aux_state)
    aux = runtime.last_pack
    mask = feat["sc_atom_mask"].bool() & aux["sc_generation_mask"]
    out["stage4_sc_aux"] = sidechain_global_frame_aligned_loss(aux["sc_pred_global"],
        feat["sc_gt_local"], aux["sc_frame_R"], aux["sc_frame_t"], mask)
    out.update(post_pred_coordinate=state.backbone_xyz[0], post_gt_coordinate_aug=out["x_gt_aug"],
        codesign_state=state, codesign_trace=records["trace"])
    return out


# Packer, feedback channels and fusion: the generator side of the cycle,
# trained together in every phase that trains anything but the head.
GENERATOR_PREFIXES = ("sidechain_module.", "sidechain_feedback.", "hres_injector.",
                      "a_token_fusion", "q_atom_fusion", "refinement_pass_embedding")
PHASES = ("IV-0", "IV-A", "IV-F", "IV-B", "IV-C")


def apply_phase(model):
    """Reapply after model.train(): frozen module mode and autograd are distinct."""
    phase = str(model.configs.stage4.phase)
    if phase not in PHASES:
        raise ValueError(f"Unknown Stage IV phase {phase}")
    bb_prefixes = tuple(model.configs.stage4.bb_trainable_prefixes)
    generator = GENERATOR_PREFIXES + bb_prefixes if bb_prefixes else GENERATOR_PREFIXES
    for name, param in model.named_parameters():
        if phase == "IV-0":
            enabled = False
        elif phase == "IV-A":
            enabled = name.startswith("aa_head.")
        elif phase == "IV-F":
            # Frozen sequence head, generator trained against it. `requires_grad
            # = False` on the head does NOT stop gradient: the AA cross-entropy
            # still reaches the backbone through the head's coordinate inputs,
            # which is the whole objective here -- produce backbones the frozen
            # designer reads as native. Losing that route leaves the packer
            # training alone and the backbone receiving no sequence signal, so
            # it is asserted, not assumed (test_frozen_head_still_passes_gradient).
            enabled = name.startswith(generator)
        else:
            enabled = name.startswith(("aa_head.",) + generator)
        param.requires_grad_(enabled)
    for module in model.children():
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()


def optimizer_groups(model):
    cfg = model.configs.stage4
    groups = {name: dict(name=name, params=[], lr=float(getattr(cfg, name + "_lr"))) for name in ("aa", "sc", "bb")}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = "aa" if name.startswith("aa_head.") else "sc" if name.startswith("sidechain_module.") else "bb"
            groups[group]["params"].append(param)
    return [group for group in groups.values() if group["params"]]


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
        cycle=asdict(cycle_config(model)), inference_rounds=int(cfg.inference_rounds),
        mask_policy="query-X-hide-SC-before-encoding-v1", sidechain_edm=False,
        optimizer_policy="joint-AA-SC-BB-v1",
        optimizer={key: float(getattr(cfg, key)) for key in ("aa_lr", "sc_lr", "bb_lr")},
        bb_trainable_prefixes=list(cfg.bb_trainable_prefixes),
        objectives={key: float(getattr(cfg,key)) for key in ("weight_aa_pre", "weight_aa_revision", "weight_sc_aux", "weight_physical")},
        template_provider=str(model.sc_template_provider),
        diffusion_samples=int(model.configs.training.diffusion_batch_size),
        feedback={key: bool(getattr(model, key, False)) for key in ("sc_a_direct", "sc_a_direct_pre", "sc_q_direct", "sc_hres_inject")})


@torch.no_grad()
def generate(model, input_feature_dict, N_step=20, temperature=0.0,
             refinement_steps=3, seed=0, allow_one_round_ablation=False):
    from protenix.model.protenix import update_input_feature_dict
    from .structure import assemble_atoms
    if refinement_steps < 1 or (refinement_steps < 2 and not allow_one_round_ablation):
        raise ValueError("Stage IV inference requires at least two rounds; one round needs an explicit ablation opt-in")
    if N_step < 1:
        raise ValueError("Backbone generation requires at least one diffusion step")
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    feat = dict(input_feature_dict)
    for key in ("aa_clean", "sc_gt_local", "aa_loss_mask", "aa_corruption_mask", "sc_atom_mask", "sc_slot_mask", "sc_atom_name_ids", "sc_frame_R", "sc_frame_t", "sc_bb_coords"):
        feat.pop(key, None)
    design = feat["design_token_mask"].bool()
    from .sampler import build_aa20_to_restype36
    _, xpb = build_aa20_to_restype36()
    feat["restype"] = feat["restype"].clone()
    feat["restype"][design] = 0.
    feat["restype"][design, xpb] = 1.
    feat = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
    feat = update_input_feature_dict(feat)
    model._a_sc_cache = model._q_sc_cache = None
    model._a_direct_active = model._q_direct_active = False
    model._q_inject_calls = {}
    s_inputs, s_trunk, z_trunk = model.get_condition_embedding(feat)
    schedule = model.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype)
    n_atom = feat["atom_to_token_idx"].numel()
    xyz = schedule[0] * torch.randn(1,n_atom,3,device=s_inputs.device,dtype=s_inputs.dtype)
    fixed_xyz = feat["fixed_atom_xyz"].to(xyz)[None]
    atom_design = design[feat["atom_to_token_idx"].long()]
    xyz = torch.where(atom_design[None,:,None], xyz, fixed_xyz)
    for current, following in zip(schedule[:-1],schedule[1:]):
        sigma = current.reshape(1)
        denoised = model.diffusion_module(x_noisy=xyz,t_hat_noise_level=sigma,
            input_feature_dict=feat,s_inputs=s_inputs,s_trunk=s_trunk,z_trunk=z_trunk,
            pair_z=None,p_lm=None,c_l=None)
        xyz = xyz + (following-current) * (xyz-denoised) / current
        xyz = torch.where(atom_design[None,:,None],xyz,fixed_xyz)
    xyz = torch.where(atom_design[None,:,None],denoised,fixed_xyz)
    state = make_state(model,feat,xyz,sigma,fixed_xyz)
    runtime = ProteoAACycle(model,feat,s_inputs,s_trunk,z_trunk)
    cfg = cycle_config(model,rounds=refinement_steps,temperature=temperature)
    # Keep query/block/commit draws independent of template and head augmentation
    # RNG so feedback arms use identical query masks under the same seed.
    cycle_rng = torch.Generator(device=xyz.device).manual_seed(seed)
    state, records = run_cycle(state,model.aa_head,runtime.pack,runtime.refine,cfg,generator=cycle_rng)
    atoms = assemble_atoms(state,feat)
    return dict(coordinate=state.backbone_xyz[0], sequence=torch.where(design,state.assigned_aa[0,0],-1),
        state=state, atoms=atoms, trajectory=records["trace"], aa_records=records,
        metadata=dict(seed=seed,temperature=temperature,cycle=asdict(cfg),stop_reason=records["stop_reason"],
            rounds=[{k:v for k,v in row.items() if k not in ("query_mask", "aa_changes")} for row in records["trace"]]))
