#!/usr/bin/env python3
"""Real-complex GPU implementation gate (not a quality/acceptance experiment)."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Rigid SC checkpoint used only as a test fixture")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"scripts/training"))
    import train_sc_adaptation as driver
    from pxdesign_train.runner.trainer import PXDesignTrainer
    from pxdesign_train.checkpoints import transition_config, read_checkpoint, rng_state, restore_rng
    from pxdesign_train.stage4 import apply_phase, optimizer_groups
    from pxdesign_train.runner.sc_stream import seed_all
    output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    options = driver.parser().parse_args(["--accepted-checkpoint", a.checkpoint, "--phase", "sc_complex_adapt",
        "--output-dir", str(output), "--eval-samples", "1", "--physical-weight", ".01",
        "--accumulation", "1", "--max-steps", "2", "--warmup-steps", "0", "--num-workers", "0"])
    cfg, recipe = driver.resolve(options)
    seed_all(cfg.seed)
    components = driver.build_data(cfg, recipe, output)
    dataset = components.train_dataset.dataset
    pinder = next(ds for name,ds in zip(dataset.source_names,dataset.datasets) if "pinder" in name)
    from smoke_stage4_fampnn import select_supervised_batch
    batch, supervision = select_supervised_batch(pinder)
    print("SC_ADAPTATION_SMOKE complex", batch["sample_id"], supervision, flush=True)
    torch.save(batch, output/"complex.pt")
    trainer = PXDesignTrainer(cfg, components, device=torch.device("cuda"), checkpoint_dir=str(output/"checkpoints"))
    model = trainer.raw_model
    assert [group["name"] for group in optimizer_groups(model)] == ["sc"]
    def digest(model):
        digest = hashlib.sha256()
        for name,value in model.state_dict().items():
            if not name.startswith("sidechain_module."):
                digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    frozen = digest(model)
    sc_before = next(model.sidechain_module.parameters()).detach().clone()
    calls = []
    hook = model.aa_head.register_forward_pre_hook(lambda *args: calls.append(1))
    first = trainer.train_step(batch)
    assert not calls
    broken = copy.deepcopy(batch)
    feat, labels = broken["input_feature_dict"], broken["label_dict"]
    residue = int((feat["sc_frame_valid"] & (feat["sc_bb_atom_idx"][:,3] >= 0)).nonzero()[0])
    oxygen = int(feat["sc_bb_atom_idx"][residue,3])
    feat["sc_bb_observed_mask"][residue,3] = False
    feat["sc_bb_coords"][residue,3] = float("nan")
    labels["coordinate_mask"][oxygen] = 0
    labels["coordinate"][oxygen] = float("nan")
    second = trainer.train_step(broken)
    assert all(torch.isfinite(value) for value in second.values()) and not calls
    hook.remove()
    assert not torch.equal(sc_before, next(model.sidechain_module.parameters()))
    model.train(); apply_phase(model)
    tensor = trainer._to_device(batch)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rigid = model(input_feature_dict=tensor["input_feature_dict"], label_dict=tensor["label_dict"], mode="train")
    from pxdesign_train.sc_augmentation import transform_native_sc_inputs
    transformed, targets = transform_native_sc_inputs(tensor["input_feature_dict"], tensor["label_dict"], **rigid["native_rigid_transform"])
    torch.testing.assert_close(rigid["sc_input_backbone"][0], targets["coordinate"])
    torch.testing.assert_close(rigid["feature_xyz"][0,0], targets["coordinate"])
    torch.testing.assert_close(rigid["sc_frame_t"].reshape_as(transformed["sc_frame_t"]), transformed["sc_frame_t"])
    del rigid,tensor,transformed,targets
    # Measure the rotation-sensitive packer's consistency; this is a diagnostic,
    # not an assertion of equivariance or a checkpoint acceptance threshold.
    model.eval()
    tensor = trainer._to_device(batch)
    tensor["input_feature_dict"]["input_seed"] = 8001
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device="cuda")
    translation = torch.tensor([3.,-4.,2.],device="cuda")
    rotated_feat, rotated_labels = transform_native_sc_inputs(tensor["input_feature_dict"],tensor["label_dict"],rotation,translation)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        original = model(input_feature_dict=tensor["input_feature_dict"],label_dict=tensor["label_dict"],mode="train")
        rotated = model(input_feature_dict=rotated_feat,label_dict=rotated_labels,mode="train")
    mask = original["sc_model_mask"] & rotated["sc_model_mask"]
    aligned = (rotated["sc_pred_global"].float()-translation) @ rotation
    rotation_rmsd = float((aligned-original["sc_pred_global"].float()).square().sum(-1)[mask].mean().sqrt())
    assert torch.isfinite(torch.tensor(rotation_rmsd))
    del tensor,rotated_feat,rotated_labels,original,rotated
    native_metrics = trainer.evaluate()
    (output/"native_metrics_before_resume.json").write_text(json.dumps(native_metrics, indent=2))
    assert native_metrics
    assert digest(model) == frozen
    checkpoint = trainer.save_checkpoint("native_complex_gate")
    cfg.training.resume_checkpoint = checkpoint
    cfg.training.warm_start_checkpoint = ""
    del trainer,model; gc.collect(); torch.cuda.empty_cache()
    resume_options = driver.parser().parse_args(["--resume-checkpoint", checkpoint,
        "--output-dir", str(output/"resume_data")])
    cfg, resumed_recipe = driver.resolve(resume_options)
    (output/"resume_data").mkdir(parents=True, exist_ok=True)
    resumed_components = driver.build_data(cfg, resumed_recipe, output/"resume_data")
    assert len(resumed_components.train_dataset) == len(components.train_dataset)
    components = resumed_components
    resumed = PXDesignTrainer(cfg, components, device=torch.device("cuda"), checkpoint_dir=str(output/"resume"))
    assert resumed.step == 2 and resumed.global_step == 2
    assert digest(resumed.raw_model) == frozen
    native_roundtrip = resumed.evaluate()
    for key in native_metrics:
        assert abs(native_metrics[key]-native_roundtrip[key]) < 1e-6, (key,native_metrics[key],native_roundtrip[key])
    del resumed; gc.collect(); torch.cuda.empty_cache()
    next_cfg = transition_config(read_checkpoint(checkpoint), phase="sc_adapt", stage4_overrides=dict(
        weight_aa_pre=0., weight_aa_revision=0., weight_physical=.01,
        native_fraction=.5, paired_fraction=.5))
    next_cfg.training.warm_start_checkpoint = checkpoint
    adapted = PXDesignTrainer(next_cfg, components, device=torch.device("cuda"), checkpoint_dir=str(output/"adapt_checkpoints"))
    model = adapted.raw_model
    losses = {}
    for source in ("native", "paired_reconstruction"):
        current = copy.deepcopy(batch)
        current["input_feature_dict"].update(backbone_source=source, input_seed=101, reconstruction_sigma=.4)
        losses[source] = {key:float(value) for key,value in adapted.train_step(current).items()}
        model.eval()
        tensor = adapted._to_device(current)
        saved_rng = rng_state()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = model(input_feature_dict=tensor["input_feature_dict"], label_dict=tensor["label_dict"], mode="train")
        tensor["input_feature_dict"]["sc_atom_mask"].zero_()
        tensor["input_feature_dict"]["sc_gt_local"].fill_(float("nan"))
        restore_rng(saved_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            after = model(input_feature_dict=tensor["input_feature_dict"], label_dict=tensor["label_dict"], mode="train")
        torch.testing.assert_close(before["codesign_state"].sc_xyz, after["codesign_state"].sc_xyz)
        assert after["sc_aux"] == 0
        after["sc_physical"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        physical_gradient_norm = torch.stack([p.grad.float().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        assert physical_gradient_norm > 0
        losses[source]["physical_gradient_norm"] = float(physical_gradient_norm)
        model.zero_grad(set_to_none=True)
        del before,after
    # Interface test uses an unlabeled coordinate fixture, not a claim that a
    # short smoke has established quality of full 400-step generation.
    from cache_sc_backbones import unlabeled_features
    full = dict(input_feature_dict=unlabeled_features(batch["input_feature_dict"]), label_dict={},
                sample_id="unlabeled-interface-fixture", source_name="test_fixture")
    full["input_feature_dict"].update(backbone_source="full_sample", input_seed=501,
        cached_backbone_xyz=batch["label_dict"]["coordinate"],
        backbone_provenance=dict(seed=501, checkpoint_sha256=model.component_origins["backbone"]["sha256"],
            sampler="pxdesign_native", steps=400, target_policy="joint", test_fixture=True))
    losses["unlabeled_interface_fixture"] = {key:float(value) for key,value in adapted.train_step(full).items()}
    assert losses["unlabeled_interface_fixture"]["sc_aux"] == 0
    assert losses["unlabeled_interface_fixture"]["global_grad_norm"] > 0
    assert digest(model) == frozen
    for values in losses.values():
        assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    (output/"smoke_result.json").write_text(json.dumps(dict(status="passed", sample_id=batch["sample_id"],
        native_losses=[{key:float(value) for key,value in row.items()} for row in (first,second)],
        predicted_losses=losses, native_metrics=native_metrics,
        only_sc_updates=True, frozen_digest=frozen, save_resume=True, hidden_labels_invariant=True,
        complex_rotation_consistency_rmsd=rotation_rmsd,
        full_400_step_quality_validated=False), indent=2))
    print("SC_ADAPTATION_SMOKE passed", flush=True)


if __name__ == "__main__":
    main()
