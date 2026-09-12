#!/usr/bin/env python3
"""Cache official 400-step backbones from partitioned strict input batches.

Input manifest: JSON {partition: train|validation, items: [{path, sample_id}]}.
Split and audit inputs before this command; no native labels survive in output.
"""
import argparse
import json
from pathlib import Path
import sys
import torch


def unlabeled_features(feat):
    excluded = {"aa_clean", "aa_loss_mask", "aa_corruption_mask", "sc_gt_local", "sc_atom_mask",
        "sc_observed_mask", "sc_loss_mask", "sc_chemical_mask", "sc_slot_mask", "sc_atom_name_ids",
        "sc_frame_R", "sc_frame_t", "sc_bb_coords", "sc_frame_valid", "sc_bb_observed_mask",
        "sc_context_atom_mask", "sc_interface_mask", "_native_rigid_transform"}
    return {key:value for key,value in feat.items() if key not in excluded}


@torch.no_grad()
def sample(model, feat, seed):
    from pxdesign_train.initial_sampling import RandomStream, sample_initial
    from protenix.model.protenix import update_input_feature_dict
    feat = unlabeled_features(feat)
    with RandomStream(seed).use():
        feat = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
        feat = update_input_feature_dict(feat)
        model._a_sc_cache = model._q_sc_cache = None
        model._a_direct_active = model._q_direct_active = False
        model._q_inject_calls = {}
        s_inputs, s, z = model.get_condition_embedding(feat)
        schedule = model.inference_noise_scheduler(N_step=400, device=s.device, dtype=s.dtype)
        xyz, _ = sample_initial(model, feat, s_inputs, s, z, schedule,
            sampler="pxdesign_native", target_policy="joint")
    return xyz[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1701)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from pxdesign_train.checkpoints import evaluation_model, plain_config
    from pxdesign_train.runner.sc_stream import sha256_file
    from pxdesign_train.sc_adaptation import check_source
    model = evaluation_model(a.checkpoint, device="cuda", weights="raw").eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    manifest_path = Path(a.input_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("partition") not in ("train", "validation"):
        raise ValueError("Partitioned inputs are required before generating derivatives")
    output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoint_hash = sha256_file(a.checkpoint)
    def move(value, device):
        if torch.is_tensor(value): return value.to(device)
        if isinstance(value, dict): return {key:move(item,device) for key,item in value.items()}
        return value
    for index, item in enumerate(manifest["items"]):
        batch = torch.load(manifest_path.parent/item["path"], map_location="cpu", weights_only=False)
        feat = unlabeled_features(batch["input_feature_dict"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xyz = sample(model, move(feat, "cuda"), a.seed+index)
        feat.update(backbone_source="full_sample", cached_backbone_xyz=xyz.float().cpu(),
            backbone_provenance=dict(seed=a.seed+index, checkpoint_sha256=model.component_origins["backbone"]["sha256"],
                sampler="pxdesign_native", steps=400, target_policy="joint", parent_sample_id=item["sample_id"],
                integrated_checkpoint_sha256=checkpoint_hash, precision="bf16",
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                coordinate_frame="sampler_returned_joint_complex",
                partition=manifest["partition"], input_manifest_sha256=sha256_file(manifest_path)))
        check_source(feat, {})
        path = output/f"sample-{index:05d}.pt"
        torch.save(dict(input_feature_dict=feat, label_dict={}, sample_id=item["sample_id"],
            source_name=batch.get("source_name", "unknown"), source_provenance=batch.get("source_provenance", {})), path)
        rows.append(dict(path=path.name, sample_id=item["sample_id"], source_name=batch.get("source_name", "unknown"), sha256=sha256_file(path)))
        print(f"Cached {item['sample_id']} -> {path}", flush=True)
    (output/"manifest.json").write_text(json.dumps(dict(schema="sc_full_samples_v1", partition=manifest["partition"], items=rows,
        effective_config=plain_config(model.configs), component_origins=model.component_origins,
        integrated_checkpoint=dict(path=str(Path(a.checkpoint).resolve()),sha256=checkpoint_hash)), indent=2))


if __name__ == "__main__":
    main()
