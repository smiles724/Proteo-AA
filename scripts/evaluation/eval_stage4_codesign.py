#!/usr/bin/env python3
"""Stage IV generated-state evaluation on saved strict, held-out feature batches.

Input .pt files are trusted local batches from DesignSourceDataset. Targets are
used only for metrics; generate() removes them before executing the shared cycle.
A manifest supplies path, sample_id, split, cluster_id and source for each batch.
"""
import argparse
import csv
from dataclasses import fields, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--fampnn-checkpoint",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--training-clusters",required=True,help="JSON source -> list of training cluster IDs")
    parser.add_argument("--output",required=True)
    parser.add_argument("--rounds",type=int,nargs="+",default=[3])
    parser.add_argument("--arms",nargs="+",choices=["A","B","C","D"],default=["D"])
    parser.add_argument("--backbone-steps",type=int,default=20)
    parser.add_argument("--temperature",type=float,default=0.)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--allow-one-round-ablation",action="store_true")
    a=parser.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location("proteoaa_training_driver",root/"scripts/training/train_protenix_monomer.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from pxdesign_train.model import ProtenixDesignTrain
    from pxdesign_train.stage4 import generate, checkpoint_identity, masked_aa_objective
    from pxdesign_train.structure import write_mmcif
    from pxdesign_train.sidechain.frames import gather_backbone
    from pxdesign_train.sidechain.physical import clash_loss
    from pxdesign_train.aa.atom_mapping import AA_ORDER
    from pxdesign_train.runner.trainer import PXDesignTrainer
    args=driver.fill_missing_args(argparse.Namespace(training_stage="stage4_fampnn",load_checkpoint=a.checkpoint,
        fampnn_checkpoint=a.fampnn_checkpoint,stage4_inference_rounds=max(2,max(a.rounds)),warm_start_params_only=True))
    driver.apply_training_stage_args(args)
    config=driver.build_configs(args,torch.device(a.device))
    driver.adopt_feedback_channels_from_checkpoint(config,a.checkpoint)
    model=ProtenixDesignTrain(config).to(a.device).eval()
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    recorded=checkpoint.get("stage4_identity")
    expected=checkpoint_identity(model)
    if recorded is not None:
        for key in ("backend","upstream_revision","checkpoint_sha256","model_config","mapping_version"):
            if recorded.get(key) != expected.get(key): raise ValueError(f"Checkpoint {key} mismatch")
    # Reuse trainer compatibility guard without constructing an optimizer.
    carrier=object.__new__(PXDesignTrainer);carrier.configs=config;carrier._log=print
    carrier._check_sidechain_arch(checkpoint)
    weights={k.removeprefix("module."):v for k,v in checkpoint["model"].items() if not k.removeprefix("module.").startswith("design_residue_type_head.")}
    missing,unexpected=model.load_state_dict(weights,strict=False)
    permitted=("aa_head.",) if recorded is None else ()
    if unexpected or any(not k.startswith(permitted) for k in missing):
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    manifest_path=Path(a.manifest).resolve()
    manifest=list(csv.DictReader(manifest_path.open()))
    training=json.loads(Path(a.training_clusters).read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    state_rows=[]
    def device(value):
        if torch.is_tensor(value): return value.to(a.device)
        if isinstance(value,dict): return {k:device(v) for k,v in value.items()}
        return value
    for index,entry in enumerate(manifest):
        source=entry["source"]
        if entry["split"] not in ("val","test") or entry["cluster_id"] in set(training[source]):
            raise ValueError(f"Manifest is not held out: {entry}")
        batch=device(torch.load(manifest_path.parent/entry["path"],map_location="cpu",weights_only=False))
        feat=batch["input_feature_dict"]
        native=feat.get("aa_clean")
        native_bb, native_present = gather_backbone(batch["label_dict"]["coordinate"], feat["aa_bb_atom_idx"])
        design_native = feat["design_token_mask"].bool()
        fixed_native = feat["aa_residue_mask"].bool() & ~design_native & native_present[:,1]
        native_ca = native_bb[:,1]
        interface = design_native & (torch.cdist(native_ca,native_ca[fixed_native]).amin(-1)<8.) if fixed_native.any() else torch.zeros_like(design_native)
        for rounds in a.rounds:
            for arm in a.arms:
                model.configs.stage4.sc_to_bb=arm in ("B","D")
                model.configs.stage4.sc_to_aa=arm in ("C","D")
                result=generate(model,feat,N_step=a.backbone_steps,temperature=a.temperature,
                    refinement_steps=rounds,seed=a.seed+index,allow_one_round_ablation=a.allow_one_round_ablation)
                state=result["state"];design=state.design_mask[0,0];assigned=state.assigned_aa[0,0]
                prefix=f"sample{index:05d}_{arm}_r{rounds}"
                write_mmcif(result["atoms"],output/f"{prefix}.cif")
                (output/f"{prefix}.fasta").write_text(f">{entry['sample_id']}\n"+"".join(AA_ORDER[int(v)] for v in assigned[design])+"\n")
                row=dict(sample_id=entry["sample_id"],source=source,cluster_id=entry["cluster_id"],arm=arm,rounds=rounds,
                    stop_reason=result["metadata"]["stop_reason"],temperature=a.temperature,seed=a.seed+index,
                    design_residues=int(design.sum()),fixed_coordinate_max_error=float((state.backbone_xyz-state.fixed_atom_xyz)[state.fixed_atom_mask].abs().max()) if state.fixed_atom_mask.any() else 0.)
                bb,_=gather_backbone(state.backbone_xyz,state.bb_atom_idx)
                ca=bb[0,0,:,1]
                fixed=state.residue_mask[0,0] & ~design
                if native is not None:
                    for stage,key in (("initial","initial"),("revision","revisions")):
                        nll,recovery=masked_aa_objective(result["aa_records"][key],native[None,None])
                        row[f"conditional_nll_{stage}"]=float(nll)
                        row[f"conditional_recovery_{stage}"]=float(recovery)
                    for label,mask in (("all",design),("interface",interface),("noninterface",design & ~interface)):
                        valid=mask & (native >= 0) & (native < 20)
                        row[f"recovery_{label}"]=float((assigned[valid] == native[valid]).float().mean()) if valid.any() else None
                        row[f"recovery_count_{label}"]=int(valid.sum())
                row["composition"]=" ".join(map(str,torch.bincount(assigned[design],minlength=20).tolist()))
                row["clash_penalty"]=float(clash_loss(state.sc_xyz.reshape(1,-1,3),valid_mask=state.generation_mask.reshape(1,-1),
                    group_id=torch.arange(design.numel(),device=ca.device).repeat_interleave(10)[None],
                    context_coords=state.backbone_xyz.reshape(1,-1,3),context_mask=(design[feat["atom_to_token_idx"].long()] | feat["fixed_atom_mask"].bool())[None],
                    context_group_id=feat["atom_to_token_idx"][None]))
                same_chain = state.chain_index[0,0,1:] == state.chain_index[0,0,:-1]
                consecutive = state.residue_index[0,0,1:] == state.residue_index[0,0,:-1]+1
                peptide = design[1:] & design[:-1] & same_chain & consecutive
                cn = torch.linalg.vector_norm(bb[0,0,:-1,2]-bb[0,0,1:,0], dim=-1)
                row["peptide_cn_count"] = int(peptide.sum())
                row["peptide_cn_mean_angstrom"] = float(cn[peptide].mean()) if peptide.any() else None
                row["peptide_cn_max_angstrom"] = float(cn[peptide].max()) if peptide.any() else None
                if native is not None:
                    saved = replace(state, backbone_features={}, feedback=None)
                    saved = replace(saved, **{f.name:getattr(saved,f.name).detach().cpu() for f in fields(saved) if torch.is_tensor(getattr(saved,f.name))})
                    state_path = f"{prefix}_state.pt"
                    torch.save(dict(state=saved, targets=dict(native_aa=native.detach().cpu()[None,None])),output/state_path)
                    state_rows.append(dict(path=state_path, sample_id=entry["sample_id"], source=source, cluster_id=entry["cluster_id"], split=entry["split"], arm=arm, rounds=rounds))
                rows.append(row)
                (output/f"{prefix}.json").write_text(json.dumps(result["metadata"],indent=2))
    if rows:
        keys=sorted({key for row in rows for key in row})
        with (output/"metrics.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if state_rows:
        with (output/"states_manifest.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(state_rows[0]));writer.writeheader();writer.writerows(state_rows)
    (output/"protocol.json").write_text(json.dumps(dict(checkpoint_identity=expected,arguments=vars(a),
        interface_definition="native CA distance to fixed protein < 8 A; held fixed across arms",backbone_start="noise",
        split_scope="provided per-source cluster IDs; cross-source homology requires a shared clustering manifest"),indent=2))


if __name__ == "__main__": main()
