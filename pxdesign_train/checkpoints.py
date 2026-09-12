"""Component-scoped initialization and self-contained integrated checkpoints.

A donor is never a permissive full-model load. Validate every requested tensor
before mutating the model; full resume uses the complete saved architecture.
"""
from pathlib import Path
import hashlib
import random
import subprocess
import numpy as np
import torch

SCHEMA_VERSION = 1
BACKBONE_PREFIXES = ("design_condition_embedder.", "diffusion_module.")
SC_PREFIXES = ("sidechain_module.",)
FEEDBACK_PREFIXES = ("sidechain_feedback.", "hres_injector.", "a_token_fusion", "q_atom_fusion", "refinement_pass_embedding")
SC_LAYOUT_KEYS = ("bb_context", "centre_coord_input", "frame_aware_head", "template_residual", "type_logits_input", "edm", "a_bs_concat", "q_bs")
# Same one-step architecture as the validated donor, with no donor weights.
SCRATCH_SC_LAYOUT = dict(bb_context=True, centre_coord_input=True, frame_aware_head=False,
    template_residual=False, type_logits_input=True, edm=False, a_bs_concat=True, q_bs=False)


def unwrap(model):
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def normalize_state(state):
    result = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError(f"Ambiguous checkpoint key after prefix normalization: {key}")
        result[key] = value
    return result


def read_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def tensor_state(checkpoint):
    return normalize_state(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))


def runtime_sources():
    from pxdesign.model import pxdesign
    from protenix.model import generator
    roots = {"pxdesign": (Path(pxdesign.__file__).resolve().parents[2], "f788441313c84c3074fe9596ac2433f96b15c763"),
             "protenix": (Path(generator.__file__).resolve().parents[2], "c3bfc365b3e1341a11935eddfe7bfdc308092147")}
    record = {}
    for name, (root, expected) in roots.items():
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != expected:
            raise ValueError(f"{name} source revision {revision} differs from validated {expected}")
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", name])
        if diff:
            patch = Path(__file__).resolve().parents[1]/"patches/pxdesign-embedders-protenix-2.0.patch"
            if name != "pxdesign" or diff.strip() != patch.read_bytes().strip():
                raise ValueError(f"Unrecorded {name} source changes; only the checked-in compatibility patch is allowed")
        record[name] = dict(revision=revision, patch_sha256=hashlib.sha256(diff).hexdigest(), path=str(root))
    return record


def source_record(path, checkpoint):
    with Path(path).open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(Path(path).resolve()), sha256=sha,
                revision=checkpoint.get("revision", checkpoint.get("source_revision")), step=checkpoint.get("step"))


def component_state(model, checkpoint, prefixes):
    expected = {k: v for k, v in unwrap(model).state_dict().items() if k.startswith(prefixes)}
    for prefix in prefixes:
        if not any(k.startswith(prefix) for k in expected):
            raise ValueError(f"Model has no component {prefix}")
    supplied = {k: v for k, v in tensor_state(checkpoint).items() if k.startswith(prefixes)}
    missing = sorted(expected.keys() - supplied.keys())
    extra = sorted(supplied.keys() - expected.keys())
    shapes = [k for k in expected.keys() & supplied.keys()
              if not torch.is_tensor(supplied[k]) or expected[k].shape != supplied[k].shape]
    if missing or extra or shapes:
        raise ValueError(f"Incomplete/incompatible component {prefixes}: missing={missing[:12]}, unexpected={extra[:12]}, shapes={shapes[:12]}")
    return supplied


def check_sc_layout(model, checkpoint):
    saved = checkpoint.get("sidechain_arch")
    if not saved or any(key not in saved for key in SC_LAYOUT_KEYS):
        raise ValueError("SC donor must record its complete sidechain_arch")
    if saved["edm"] or "local_coord_input" in saved:
        raise ValueError("SC donor requires edm=false and the global-coordinate frame convention")
    current = unwrap(model).configs.sidechain
    mismatched = [k for k in SC_LAYOUT_KEYS if bool(getattr(current, k, False)) != bool(saved[k])]
    if mismatched:
        raise ValueError(f"SC donor architecture mismatch: {mismatched}")
    return dict(saved)


def compose_components(model, *, backbone_checkpoint, sidechain_checkpoint=None, sidechain_init="checkpoint"):
    model = unwrap(model)
    if sidechain_init not in ("checkpoint", "scratch"):
        raise ValueError(f"Unknown SC initialization {sidechain_init}")
    if sidechain_init == "scratch" and sidechain_checkpoint:
        raise ValueError("Scratch SC initialization cannot also load an SC donor")
    if sidechain_init == "scratch" and not getattr(model, "enable_sidechain", False):
        raise ValueError("Scratch SC initialization requires an SC module")
    if getattr(model, "aa_backend", None) != "fampnn":
        raise ValueError("Component composition requires the strictly initialized FAMPNN adapter")
    # All validation precedes any writes; FAMPNN was strictly initialized by its adapter.
    backbone = read_checkpoint(backbone_checkpoint)
    state = component_state(model, backbone, BACKBONE_PREFIXES)
    sources = runtime_sources()
    origins = dict(backbone=source_record(backbone_checkpoint, backbone), fampnn=dict(model.aa_head.identity), feedback=dict(origin="fresh"))
    if sidechain_checkpoint:
        sidechain = read_checkpoint(sidechain_checkpoint)
        arch = check_sc_layout(model, sidechain)
        state.update(component_state(model, sidechain, SC_PREFIXES))
        origins["sidechain"] = dict(**source_record(sidechain_checkpoint, sidechain), architecture=arch,
            atom_vocabulary="Proteo-AA-37-append-only", frame="global-input-CA-Gram-Schmidt", edm=False)
    elif sidechain_init == "scratch":
        arch = {key:bool(getattr(model.configs.sidechain,key)) for key in SC_LAYOUT_KEYS}
        if arch["edm"]:
            raise ValueError("Scratch SC initialization requires one-step packing (edm=false)")
        origins["sidechain"] = dict(origin="scratch", initialization="module_constructors", seed=int(torch.initial_seed()),
            architecture=arch, atom_vocabulary="Proteo-AA-37-append-only", frame="global-input-CA-Gram-Schmidt", edm=False)
    elif getattr(model, "enable_sidechain", False):
        raise ValueError("Packing requires --sidechain-checkpoint or explicit --sidechain-init scratch")
    model.load_state_dict(state, strict=False)
    origins["backbone"]["runtime_sources"] = sources
    model.component_origins = origins
    return origins


def plain_config(config):
    if isinstance(config, dict) or hasattr(config, "items"):
        return {str(k): plain_config(v) for k,v in config.items()}
    if isinstance(config, (list, tuple)):
        return [plain_config(v) for v in config]
    if hasattr(config, "__dict__"):
        return {k: plain_config(v) for k,v in vars(config).items() if not k.startswith("_")}
    return config


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def integrated_record(model):
    model = unwrap(model)
    from .stage4 import implementation_identity
    return dict(schema_version=SCHEMA_VERSION, effective_config=plain_config(model.configs),
        component_origins=model.component_origins, fampnn_identity=dict(model.aa_head.identity),
        trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
        implementation=implementation_identity(), rng=rng_state())


def config_from_checkpoint(checkpoint):
    from ml_collections import ConfigDict
    record = checkpoint.get("integrated")
    if not record or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("A complete integrated checkpoint is required")
    config = ConfigDict(record["effective_config"])
    config.residue_type.fampnn_identity = record["fampnn_identity"]
    return config


def restore_model(model, checkpoint, *, weights="raw"):
    model = unwrap(model)
    config_from_checkpoint(checkpoint)  # schema validation
    if weights not in ("raw", "ema"):
        raise ValueError("Select raw or ema weights explicitly")
    state = tensor_state(checkpoint)
    if weights == "ema":
        if not checkpoint.get("ema"):
            raise ValueError("Checkpoint has no EMA state")
        state.update(normalize_state(checkpoint["ema"]["shadow"]))
    mapping = "aa_head.canonical_indices"
    expected_mapping = model.state_dict().get(mapping)
    if expected_mapping is not None and (mapping not in state or not torch.equal(expected_mapping.cpu(), state[mapping].cpu())):
        raise ValueError("Checkpoint AA mapping differs from canonical identities")
    model.load_state_dict(state, strict=True)
    model.component_origins = checkpoint["integrated"]["component_origins"]
    return model


def evaluation_model(path, *, device="cpu", weights="raw"):
    from .model import ProtenixDesignTrain
    checkpoint = read_checkpoint(path)
    model = ProtenixDesignTrain(config_from_checkpoint(checkpoint))
    restore_model(model, checkpoint, weights=weights)
    return model.to(device).eval()


def transition_config(checkpoint, *, phase, stage4_overrides=None, training_overrides=None, loss_overrides=None):
    """Warm-start a new phase from the saved architecture, with fresh optimizers."""
    config = config_from_checkpoint(checkpoint)
    config.training.backbone_checkpoint = ''
    config.training.sidechain_checkpoint = ''
    config.training.resume_checkpoint = ''
    config.training.warm_start_checkpoint = ''
    previous_phase = config.stage4.phase
    config.stage4.phase = phase
    for section, values in ((config.stage4, stage4_overrides), (config.training, training_overrides), (config.loss, loss_overrides)):
        for key,value in (values or {}).items():
            if key not in section:
                raise ValueError(f'Unknown phase-transition setting {key}')
            setattr(section,key,value)
    if phase in ("sc_warmup", "sc_complex_adapt"):
        if config.stage4.train_rounds or config.stage4.sc_to_aa or config.stage4.sc_to_bb or config.stage4.backbone_refinement_enabled:
            raise ValueError("Supervised SC phases require zero revisions and disabled feedback")
        config.sidechain.predicted_frame=False
        config.sidechain.predicted_mask=False
        config.sidechain.force_gt_type_logits=True
        config.stage4.weight_aa_pre=config.stage4.weight_aa_revision=config.stage4.weight_physical=0.
        config.loss.weight_bb_post=0.
    elif previous_phase in ("sc_warmup", "sc_complex_adapt") and phase in ("sc_adapt", "feedback_adapt", "aa_adapt", "joint_adapt"):
        config.sidechain.predicted_frame=True
        config.sidechain.predicted_mask=True
        config.sidechain.force_gt_type_logits=False
        # Restore generated-path objectives explicitly after pure packing.
        for key,default in (("weight_aa_pre",1.),("weight_aa_revision",1.),("weight_physical",0.1)):
            setattr(config.stage4,key,(stage4_overrides or {}).get(key,default))
    if phase == 'feedback_adapt' and (config.stage4.train_rounds < 1 or not config.stage4.backbone_refinement_enabled or not config.stage4.sc_to_bb):
        raise ValueError('Feedback adaptation requires explicit revision rounds, backbone refinement, and SC-to-BB')
    return config
