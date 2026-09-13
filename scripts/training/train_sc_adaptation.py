#!/usr/bin/env python3
"""Constant-mixture SC-only adaptation; explicit accepted warm start or exact resume.

No architecture switches or donor overlays are accepted. Run --help for the
supported overrides; argparse rejects everything else instead of dropping it.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

import train_protenix_monomer as base


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--accepted-checkpoint")
    source.add_argument("--resume-checkpoint")
    p.add_argument("--phase", choices=["sc_complex_adapt", "sc_adapt"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true", help="Resolve config, fingerprint/audit data; do not construct model")
    p.add_argument("--export-validation-inputs", action="store_true", help="Export audited native panels for separate full-sample caching")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    # None distinguishes an explicit override from an exact-resume default.
    for name, typ in (("seed", int), ("max-steps", int), ("sc-lr", float),
                      ("warmup-steps", int), ("accumulation", int), ("num-workers", int),
                      ("physical-weight", float), ("monomer-fraction", float),
                      ("native-fraction", float), ("full-sample-fraction", float),
                      ("eval-samples", int), ("eval-interval", int), ("checkpoint-interval", int),
                      ("reconstruction-max-ca-error", float), ("reconstruction-max-bond-error", float)):
        p.add_argument("--"+name, type=typ)
    for name in ("data-root", "source-index", "eval-source-index", "pinder-root", "pinder-manifest",
                 "pinder-cif-cache", "pinder-pdb-cache", "pinder-archive", "reconstruction-sigmas",
                 "full-sample-train-cache", "full-sample-validation-cache"):
        p.add_argument("--"+name)
    return p


def resolve(options):
    """Resolve saved config BEFORE touching datasets. Return effective recipe."""
    from pxdesign_train.checkpoints import read_checkpoint, config_from_checkpoint, transition_config, SC_LAYOUT_KEYS
    from pxdesign_train.sc_adaptation import validate_phase, PROTOCOL
    from pxdesign_train.runner.sc_stream import sha256_file
    path = options.resume_checkpoint or options.accepted_checkpoint
    checkpoint = read_checkpoint(path)
    saved = config_from_checkpoint(checkpoint)
    if options.resume_checkpoint:
        recipe = dict(saved.training.sc_adaptation_recipe)
        for key, value in vars(options).items():
            if key not in ("output_dir", "dry_run", "export_validation_inputs", "device", "resume_checkpoint", "accepted_checkpoint") and value is not None:
                if recipe.get(key) != value:
                    raise ValueError(f"Exact resume cannot override {key}; start a new phase instead")
        config = saved
        config.training.resume_checkpoint = str(Path(path).resolve())
        config.training.warm_start_checkpoint = ""
    else:
        if options.phase is None:
            raise ValueError("Warm start requires --phase")
        previous = "sc_warmup" if options.phase == "sc_complex_adapt" else "sc_complex_adapt"
        if saved.stage4.phase != previous:
            raise ValueError(f"{options.phase} requires an accepted {previous} checkpoint, got {saved.stage4.phase}")
        if not saved.stage4.native_sc_augmentation:
            raise ValueError("Accepted donor must record native rigid augmentation")
        recipe = dict(phase=options.phase, seed=42, max_steps=1000, sc_lr=1e-5,
            warmup_steps=500, accumulation=8, num_workers=4,
            physical_weight=0. if options.phase == "sc_complex_adapt" else .01,
            monomer_fraction=.75 if options.phase == "sc_complex_adapt" else .5,
            native_fraction=1. if options.phase == "sc_complex_adapt" else .5,
            full_sample_fraction=0., eval_samples=64, eval_interval=1000, checkpoint_interval=500,
            reconstruction_sigmas="0.4,1,2,4", reconstruction_max_ca_error=3., reconstruction_max_bond_error=.3,
            full_sample_train_cache="", full_sample_validation_cache="")
        for key, value in vars(options).items():
            if key not in ("output_dir", "dry_run", "export_validation_inputs", "device", "resume_checkpoint", "accepted_checkpoint") and value is not None:
                recipe[key] = value
        if not 0 < recipe["monomer_fraction"] < 1:
            raise ValueError("Adaptation recipes require both monomer and PINDER sources")
        if recipe["phase"] == "sc_complex_adapt" and (recipe["native_fraction"] != 1 or recipe["full_sample_fraction"]):
            raise ValueError("Native-complex adaptation cannot use reconstructed/full-sample inputs")
        config = transition_config(checkpoint, phase=recipe["phase"], stage4_overrides=dict(
            adaptation_protocol=PROTOCOL, train_rounds=0, inference_rounds=0,
            sc_to_aa=False, sc_to_bb=False, backbone_refinement_enabled=False,
            packing_enabled=True, initial_target_policy="joint", backbone_sampler="pxdesign_native",
            native_sc_augmentation=True, feature_sigma=.4, decode_blocks=4, temperature=0.,
            weight_aa_pre=0., weight_aa_revision=0., weight_sc_aux=1., weight_physical=recipe["physical_weight"],
            sc_lr=recipe["sc_lr"], native_fraction=recipe["native_fraction"],
            paired_fraction=1-recipe["native_fraction"]-recipe["full_sample_fraction"],
            full_sample_fraction=recipe["full_sample_fraction"], reconstruction_sigmas=recipe["reconstruction_sigmas"],
            reconstruction_max_ca_error=recipe["reconstruction_max_ca_error"],
            reconstruction_max_bond_error=recipe["reconstruction_max_bond_error"]))
        config.seed, config.dtype = recipe["seed"], "bf16"
        config.training.update(dict(warm_start_checkpoint=str(Path(path).resolve()),
            max_steps=recipe["max_steps"], lr=recipe["sc_lr"], warmup_steps=recipe["warmup_steps"],
            iters_to_accumulate=recipe["accumulation"], grad_clip_norm=1., crop_size=384,
            diffusion_batch_size=1, eval_interval=recipe["eval_interval"], checkpoint_interval=recipe["checkpoint_interval"],
            num_workers=recipe["num_workers"], log_interval=50, ema_decay=0.,
            accepted_parent=dict(path=str(Path(path).resolve()), sha256=sha256_file(path), phase=previous)))
    validate_phase(config)
    for key in SC_LAYOUT_KEYS:
        if key not in checkpoint["sidechain_arch"] or bool(config.sidechain[key]) != bool(checkpoint["sidechain_arch"][key]):
            raise ValueError(f"Accepted SC architecture mismatch: {key}")
    if recipe["max_steps"] < 1 or recipe["accumulation"] < 1 or recipe["sc_lr"] <= 0:
        raise ValueError("Step budget, accumulation and SC LR must be positive")
    if recipe["full_sample_fraction"] and not recipe["physical_weight"]:
        raise ValueError("Unlabeled full-sample training would have no objective at physical weight zero")
    return config, recipe


def legacy_arguments(recipe, output_dir):
    # Reuse established strict data builders without their training defaults.
    previous = sys.argv
    try:
        sys.argv = [previous[0]]
        args = base.parse_args()
    finally:
        sys.argv = previous
    args.training_stage = "stage4_fampnn"
    args.stage4_phase = recipe["phase"]
    args.output_dir = str(output_dir)
    args.data_mode, args.complex_provider = "mixed_monomer_complex", "pinder"
    args.crop_size = args.max_n_token = 384
    args.complex_max_n_token = 640
    args.stage2_start_monomer_frac = args.stage2_end_monomer_frac = recipe["monomer_fraction"]
    args.curriculum_stage1_end_step = args.curriculum_stage2_start_step = 0
    args.pinder_complex_frac = 1.
    args.ref_pos_augment = False
    args.max_crop_retries = 64
    args.eval_seed = recipe["seed"] + 1000003
    for key in ("seed", "data_root", "source_index", "eval_source_index", "pinder_root", "pinder_manifest",
                "pinder_cif_cache", "pinder_pdb_cache", "pinder_archive", "eval_samples", "eval_interval"):
        if key in recipe:
            setattr(args, key, recipe[key])
    base.apply_training_stage_args(args)
    args.rebuild_eval_index = True
    return args


def build_data(config, recipe, output):
    from pxdesign_train.runner.sc_stream import fingerprint, SCStream, FullSampleCache, CoordinatePanel
    from pxdesign_train.runner.sc_partitions import prepare_partitions
    args = legacy_arguments(recipe, output)
    for key, subdir in (("pinder_cif_cache", "pinder_cif_cache"), ("pinder_pdb_cache", "pinder_pdb_cache")):
        if key not in recipe:
            setattr(args, key, str(output/subdir))
    source = Path(args.source_index) if args.source_index else base._source_index_path(Path(args.data_root))
    manifest = Path(args.pinder_manifest) if args.pinder_manifest else Path(args.pinder_root)/"indices/pinder_ppi_complex.parquet"
    # Paths/defaults become part of the effective recipe, including exact resume.
    for key in ("data_root", "pinder_root", "pinder_cif_cache", "pinder_pdb_cache", "pinder_archive"):
        recipe[key] = str(getattr(args, key))
    recipe["source_index"], recipe["pinder_manifest"] = str(source.resolve()), str(manifest.resolve())
    cache = output/"cache"
    cache.mkdir(parents=True, exist_ok=True)
    # Filter name includes source content + all filtering knobs: stale indices cannot match.
    eval_source = Path(args.eval_source_index) if args.eval_source_index else base._recent_index_path(Path(args.data_root))
    recipe["eval_source_index"] = str(eval_source.resolve())
    inputs = fingerprint([source, manifest, eval_source], dict(crop=384, monomer_min=args.min_n_token,
        binder_fraction=args.complex_max_binder_fraction, complex_max_tokens=args.complex_max_n_token))
    filtered = cache/f"monomers-{inputs['sha256'][:16]}.csv.gz"
    base.build_monomer_index(source_index=source, output_index=filtered,
        min_n_token=args.min_n_token, max_n_token=384, limit=0, rebuild=False)
    eval_loader, _, eval_index = base.build_eval_dataloader(args, output)
    if eval_loader is None:
        raise ValueError("SC adaptation requires held-out native validation")
    all_validation = cache/"all_validation_monomers.csv.gz"
    base.build_monomer_index(source_index=eval_source, output_index=all_validation,
        min_n_token=args.min_n_token, max_n_token=384, limit=0, rebuild=True)
    filtered, manifest, audit = prepare_partitions(filtered, all_validation, manifest, cache)
    components, counts = base.build_mixed_components(args, filtered, None, manifest)
    sampling_policy = {name:("inverse_cluster_after_filtering" if getattr(ds.provider, "cluster_ids", None) is not None else "uniform")
        for name,ds in zip(components.train_dataset.source_names, components.train_dataset.datasets)}
    panels = base.build_stage4_binder_validation(args, None, manifest)
    panels["monomer_retention"] = eval_loader
    paths = [filtered, Path(eval_index), all_validation, manifest]
    for key in ("full_sample_train_cache", "full_sample_validation_cache"):
        if recipe.get(key):
            paths.append(recipe[key])
    input_identity = dict(hashes=sorted(inputs["files"].values()), settings=inputs["settings"])
    data_identity = fingerprint(paths, dict(source_inputs=input_identity, recipe=recipe, overlap_audit=audit, sampling_policy=sampling_policy))
    # Output directories differ on resume; compare file contents/settings, not cache paths.
    identity = dict(hashes=sorted(data_identity["files"].values()), settings=data_identity["settings"])
    saved = getattr(config.training, "sc_data_identity", None)
    from pxdesign_train.checkpoints import plain_config
    if saved is not None and config.training.resume_checkpoint and plain_config(saved) != identity:
        raise ValueError("Exact resume dataset/cache fingerprint differs")
    config.training.sc_data_identity = identity
    config.training.sc_adaptation_recipe = recipe
    config.training.entrypoint_arguments = dict(recipe, entrypoint="train_sc_adaptation.py")
    # The accepted integrated donor records the exact official backbone hash.
    from pxdesign_train.checkpoints import read_checkpoint
    donor = read_checkpoint(config.training.resume_checkpoint or config.training.warm_start_checkpoint)
    bb_hash = donor["integrated"]["component_origins"]["backbone"]["sha256"]
    full_train = FullSampleCache(recipe["full_sample_train_cache"], checkpoint_sha256=bb_hash, partition="train") if recipe.get("full_sample_train_cache") else None
    components.train_dataset = SCStream(components.train_dataset, components.schedule, config.stage4,
        seed=config.seed, microsteps=config.training.max_steps*config.training.iters_to_accumulate, full_samples=full_train)
    components.eval_dataloader = None
    components.named_eval_dataloaders = {}
    for name, loader in panels.items():
        components.named_eval_dataloaders["native/"+name] = CoordinatePanel(loader, "native")
        if recipe["phase"] == "sc_adapt":
            for sigma in (float(x) for x in recipe["reconstruction_sigmas"].split(",")):
                components.named_eval_dataloaders[f"paired_sigma{sigma:g}/{name}"] = CoordinatePanel(loader, "paired_reconstruction", sigma)
    if recipe.get("full_sample_validation_cache"):
        if recipe["phase"] != "sc_adapt":
            raise ValueError("Full-sample panels belong to sc_adapt validation")
        components.named_eval_dataloaders["full_sample_400"] = FullSampleCache(
            recipe["full_sample_validation_cache"], checkpoint_sha256=bb_hash, partition="validation")
    (output/"data_audit.json").write_text(json.dumps(dict(identity=identity, counts=counts), indent=2, default=str))
    return components


def main():
    options = parser().parse_args()
    # Protenix resolves its CCD paths at import time. Establish the data root
    # before checkpoint/config imports can transitively import Protenix.
    bootstrap_data_root = options.data_root or os.environ.get(
        "PROTENIX_ROOT_DIR", "/hai/scratch/yfsun/protenix_data"
    )
    os.environ.setdefault("PROTENIX_ROOT_DIR", bootstrap_data_root)
    os.environ.setdefault("PROTENIX_DATA_ROOT_DIR", str(Path(bootstrap_data_root) / "common"))
    # Bootstrap imports only; do not construct data or a model before resolution.
    args = legacy_arguments(dict(phase=options.phase or "sc_adapt", monomer_fraction=.5, seed=42), options.output_dir)
    base._bootstrap_paths(args)
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    logging.basicConfig(level=logging.INFO)
    config, recipe = resolve(options)
    from pxdesign_train.runner.sc_stream import seed_all
    seed_all(int(config.seed))
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    components = build_data(config, recipe, output)
    if options.export_validation_inputs:
        import torch
        cache = output/"native_validation_inputs"; cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, loader in components.named_eval_dataloaders.items():
            if not name.startswith("native/"): continue
            for batch in loader:
                path = cache/f"input-{len(rows):05d}.pt"
                torch.save(batch, path)
                rows.append(dict(path=path.name, sample_id=batch["sample_id"], source=name))
        (cache/"manifest.json").write_text(json.dumps(dict(partition="validation", items=rows), indent=2))
    (output/"resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
    print(json.dumps(dict(phase=config.stage4.phase, recipe=recipe, trainable=["sidechain_module.*"],
        architecture=dict(config.sidechain), validation_panels=list(components.named_eval_dataloaders)), indent=2, default=str))
    if options.dry_run:
        return
    if recipe["phase"] == "sc_adapt" and not recipe.get("full_sample_validation_cache"):
        raise ValueError("sc_adapt launch requires a separate full-400-step validation cache")
    from pxdesign_train.runner.train import train_from_components
    import torch
    train_from_components(configs=config, components=components, device=torch.device(options.device),
        checkpoint_dir=str(output/"checkpoints"), max_steps=int(config.training.max_steps))


if __name__ == "__main__":
    main()
