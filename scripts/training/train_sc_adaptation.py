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


_REPAIR_RUNTIME_RECIPE_KEYS = {
    'max_steps', 'sc_lr', 'warmup_steps', 'num_workers', 'physical_weight',
    'eval_interval', 'checkpoint_interval', 'donor_weights', 'repair_arm',
    'geometry_ramp_steps', 'weight_bond_sc', 'weight_bond_attach',
    'weight_angle_sc', 'weight_angle_attach',
}


def repair_data_recipe(recipe):
    """Keep only settings that can change sampled train/evaluation examples."""
    return {key: value for key, value in dict(recipe).items()
            if key not in _REPAIR_RUNTIME_RECIPE_KEYS}


def normalized_repair_data_identity(identity):
    """Normalize identities written before runtime budgets were separated."""
    from pxdesign_train.checkpoints import plain_config
    value = plain_config(identity)
    settings = value.get('settings', {})
    if 'recipe' in settings:
        settings['recipe'] = repair_data_recipe(settings['recipe'])
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--accepted-checkpoint")
    source.add_argument("--resume-checkpoint")
    p.add_argument("--phase", choices=["sc_geometry_repair", "sc_complex_adapt", "sc_adapt"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--donor-weights", choices=["ema","raw"])
    p.add_argument("--repair-arm", choices=["A","B","C","D","E","F"])
    p.add_argument("--calibration-path")
    p.add_argument("--final-test-index")
    p.add_argument("--repair-acceptance")
    p.add_argument("--repair-final-test")
    p.add_argument("--geometry-ramp-steps", type=int)
    for name in ("bond-sc","bond-attach","angle-sc","angle-attach"):
        p.add_argument("--weight-"+name,type=float)
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
                if key == 'max_steps':
                    if value <= int(checkpoint['step']):
                        raise ValueError("Extended repair budget must exceed the resumed checkpoint step")
                    recipe[key] = value
                    continue
                if recipe.get(key) != value:
                    raise ValueError(f"Exact resume cannot override {key}; start a new phase instead")
        config = saved
        config.training.max_steps = recipe['max_steps']
        config.training.resume_checkpoint = str(Path(path).resolve())
        config.training.warm_start_checkpoint = ""
    else:
        if options.phase is None:
            raise ValueError("Warm start requires --phase")
        previous = {"sc_geometry_repair":"sc_warmup", "sc_complex_adapt":"sc_geometry_repair", "sc_adapt":"sc_complex_adapt"}[options.phase]
        if saved.stage4.phase != previous:
            raise ValueError(f"{options.phase} requires an accepted {previous} checkpoint, got {saved.stage4.phase}")
        if options.phase == "sc_complex_adapt":
            if not options.repair_acceptance:
                raise ValueError("sc_complex_adapt requires --repair-acceptance from the fixed-panel selector")
            if not options.repair_final_test:
                raise ValueError("sc_complex_adapt requires --repair-final-test from post-selection evaluation")
            acceptance = json.loads(Path(options.repair_acceptance).read_text())
            if (acceptance.get('schema') not in {
                    'sc_geometry_repair_acceptance_v1', 'sc_geometry_repair_acceptance_v2'}
                    or not acceptance.get('approved')):
                raise ValueError("Repair acceptance artifact is missing approval")
            if acceptance.get('selected', {}).get('checkpoint_sha256') != sha256_file(path):
                raise ValueError("Repair acceptance selects a different checkpoint")
            final_test = json.loads(Path(options.repair_final_test).read_text())
            if (final_test.get('schema') != 'sc_geometry_repair_final_test_v1'
                    or not final_test.get('completed')):
                raise ValueError("Repair final-test artifact is incomplete")
            if final_test.get('selected_checkpoint_sha256') != sha256_file(path):
                raise ValueError("Repair final-test artifact evaluated a different checkpoint")
            if final_test.get('acceptance_sha256') != sha256_file(options.repair_acceptance):
                raise ValueError("Repair final-test artifact belongs to a different acceptance decision")
            selected_recipe = dict(saved.training.sc_adaptation_recipe)
            if (final_test.get('final_test_manifest_sha256')
                    != sha256_file(selected_recipe['final_test_index'])):
                raise ValueError("Repair final-test manifest differs from the selected checkpoint")
        if not saved.stage4.native_sc_augmentation:
            raise ValueError("Accepted donor must record native rigid augmentation")
        recipe = dict(phase=options.phase, seed=42, max_steps=1000, sc_lr=1e-5,
            warmup_steps=500, accumulation=8, num_workers=4,
            physical_weight=0. if options.phase == "sc_complex_adapt" else .01,
            monomer_fraction=.75 if options.phase == "sc_complex_adapt" else .5,
            native_fraction=1. if options.phase == "sc_complex_adapt" else .5,
            full_sample_fraction=0., eval_samples=64, eval_interval=1000, checkpoint_interval=500,
            reconstruction_sigmas="0.4,1,2,4", reconstruction_max_ca_error=3., reconstruction_max_bond_error=.3,
            full_sample_train_cache="", full_sample_validation_cache="", donor_weights="raw",
            repair_arm="C", calibration_path="", final_test_index="", geometry_ramp_steps=200,
            weight_bond_sc=0., weight_bond_attach=0., weight_angle_sc=0., weight_angle_attach=0.)
        if options.phase == "sc_geometry_repair":
            recipe.update(seed=int(saved.seed), max_steps=2000, warmup_steps=100,
                accumulation=int(saved.training.iters_to_accumulate), physical_weight=0.,
                monomer_fraction=1.,native_fraction=1.,eval_samples=491,eval_interval=500,
                donor_weights="ema")
            if options.donor_weights != "ema":
                raise ValueError("Repair requires explicit --donor-weights ema from step 46000")
            if int(checkpoint['step']) != 46000:
                raise ValueError("Initial repair requires the step 46000 donor")
        for key, value in vars(options).items():
            if key not in ("output_dir", "dry_run", "export_validation_inputs", "device", "resume_checkpoint", "accepted_checkpoint") and value is not None:
                recipe[key] = value
        if recipe["phase"] != "sc_geometry_repair" and not 0 < recipe["monomer_fraction"] < 1:
            raise ValueError("Adaptation recipes require both monomer and PINDER sources")
        if recipe["phase"] == "sc_complex_adapt" and (recipe["native_fraction"] != 1 or recipe["full_sample_fraction"]):
            raise ValueError("Native-complex adaptation cannot use reconstructed/full-sample inputs")
        repair = recipe["phase"] == "sc_geometry_repair"
        repair_overrides = {"weight_" + name: 0. for name in
            ("bond_sc", "bond_attach", "angle_sc", "angle_attach")}
        if repair:
            from pxdesign_train.sidechain.repair_calibration import load_calibration
            calibration_hash = sha256_file(recipe["calibration_path"])
            calibration = load_calibration(str(Path(recipe["calibration_path"]).resolve()), calibration_hash)
            if not recipe["final_test_index"]:
                raise ValueError("Repair requires a separate frozen --final-test-index")
            names = ("bond_sc","bond_attach","angle_sc","angle_attach")
            if recipe["repair_arm"] in ("A","B"):
                if any(recipe["weight_"+name] for name in names):
                    raise ValueError("Control arms A/B require zero geometry weights")
            elif any(recipe["weight_"+name] <= 0 for name in names):
                raise ValueError("Arm C requires four explicitly calibrated positive weights")
            repair_overrides = dict(symmetry_aware_coordinates=recipe["repair_arm"] != "A",
                geometry_calibration_path=str(Path(recipe["calibration_path"]).resolve()),
                geometry_calibration_sha256=calibration_hash,
                chemistry_registry_sha256=calibration['chemistry_registry_sha256'],
                geometry_ramp_steps=recipe["geometry_ramp_steps"],
                **{"weight_"+name:recipe["weight_"+name] for name in names})
        config = transition_config(checkpoint, phase=recipe["phase"], stage4_overrides=dict(
            **repair_overrides, monomer_fraction=recipe["monomer_fraction"],
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
            num_workers=recipe["num_workers"], log_interval=50, ema_decay=float(saved.training.ema_decay) if repair else 0.,
            accepted_parent=dict(path=str(Path(path).resolve()), sha256=sha256_file(path), phase=previous, weights=recipe["donor_weights"], step=int(checkpoint["step"]))))
        config.training.warm_start_weights = recipe["donor_weights"]
        if repair:
            config.ema_mutable_param_keywords = ["sidechain_module."]
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
    args.data_mode, args.complex_provider = ("monomer" if recipe["phase"] == "sc_geometry_repair" else "mixed_monomer_complex"), "pinder"
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
    if recipe["phase"] == "sc_geometry_repair":
        return build_repair_data(config, recipe, output, args)
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


def build_repair_data(config, recipe, output, args):
    """Monomer-only path: never construct, open or fingerprint a PINDER source."""
    import pandas as pd
    from pxdesign_train.runner.sc_stream import fingerprint, SCStream, CoordinatePanel, sha256_file
    from pxdesign_train.sidechain.repair_calibration import load_calibration
    cache = output/'cache'; cache.mkdir(parents=True,exist_ok=True)
    source = Path(args.source_index) if args.source_index else base._source_index_path(Path(args.data_root))
    validation = Path(args.eval_source_index) if args.eval_source_index else base._recent_index_path(Path(args.data_root))
    final = Path(recipe['final_test_index'])
    inputs = fingerprint([source,validation,final],dict(crop=384,min_tokens=args.min_n_token))
    filtered = cache/'native_train.csv.gz'
    base.build_monomer_index(source_index=source,output_index=filtered,min_n_token=args.min_n_token,
        max_n_token=384,limit=0,rebuild=True)
    train,val,test = pd.read_csv(filtered),pd.read_csv(validation),pd.read_csv(final)
    val_ids,test_ids = set(val.pdb_id.str.lower()),set(test.pdb_id.str.lower())
    if val_ids & test_ids: raise ValueError('Validation and final test overlap')
    keep = ~train.pdb_id.str.lower().isin(val_ids|test_ids)
    train = train[keep]
    if train.empty: raise ValueError('No monomer training items after exclusions')
    train.to_csv(filtered,index=False,compression=dict(method='gzip',mtime=0))
    calibration = load_calibration(str(config.stage4.geometry_calibration_path),str(config.stage4.geometry_calibration_sha256))
    if calibration['validation_manifest_sha256'] != sha256_file(validation) or calibration['final_test_manifest_sha256'] != sha256_file(final):
        raise ValueError('Held-out manifests differ from frozen calibration provenance')
    calibration_ids = set(calibration.get('pdb_ids',[]))
    if not calibration_ids or not calibration_ids <= set(train.pdb_id.str.lower()):
        raise ValueError('Calibration subset must belong to the audited training partition')
    if calibration_ids & (val_ids|test_ids): raise ValueError('Calibration overlaps held-out data')
    components,n_items = base.build_components(args,filtered)
    loader,_,eval_index = base.build_eval_dataloader(args,output)
    if loader is None: raise ValueError('Repair requires native validation')
    recipe.update(source_index=str(source.resolve()),eval_source_index=str(validation.resolve()),data_root=str(args.data_root))
    audit = dict(partition='native_monomer_only',train_items=n_items,validation_items=len(pd.read_csv(eval_index)),
        final_test_items=len(test),excluded_training_items=int((~keep).sum()),
        calibration_pdbs=len(calibration_ids),no_pdb_overlap=True,
        donor_pretraining_independence=False,homology_independence=False)
    data = fingerprint([filtered,Path(eval_index),final],dict(source_hashes=sorted(inputs['files'].values()),
        recipe=repair_data_recipe(recipe),audit=audit,calibration_sha256=config.stage4.geometry_calibration_sha256))
    identity = dict(hashes=sorted(data['files'].values()),settings=data['settings'])
    saved = getattr(config.training,'sc_data_identity',None)
    if (saved is not None and config.training.resume_checkpoint
            and normalized_repair_data_identity(saved) != normalized_repair_data_identity(identity)):
        raise ValueError('Exact resume monomer dataset fingerprint differs')
    config.training.sc_data_identity = identity
    config.training.sc_adaptation_recipe = recipe
    config.training.entrypoint_arguments = dict(recipe,entrypoint='train_sc_adaptation.py')
    components.train_dataset = SCStream(components.train_dataset,components.schedule,config.stage4,
        seed=config.seed,microsteps=config.training.max_steps*config.training.iters_to_accumulate)
    components.eval_dataloader = None
    components.named_eval_dataloaders = {'native/monomer_retention':CoordinatePanel(loader,'native',seed=1000003)}
    (output/'data_audit.json').write_text(json.dumps(dict(identity=identity,counts=audit),indent=2,default=str))
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
    if config.training.warm_start_checkpoint:
        (output/'starting_weights.json').write_text(json.dumps(dict(
            source_checkpoint=config.training.warm_start_checkpoint,
            source_sha256=config.training.accepted_parent['sha256'],
            source_step=config.training.accepted_parent['step'],
            weights=recipe['donor_weights'],
            optimizer='fresh', scheduler='fresh',
            ema_shadow='initialized_after_weight_load',
        ), indent=2))
    train_from_components(configs=config, components=components, device=torch.device(options.device),
        checkpoint_dir=str(output/"checkpoints"), max_steps=int(config.training.max_steps))


if __name__ == "__main__":
    main()
