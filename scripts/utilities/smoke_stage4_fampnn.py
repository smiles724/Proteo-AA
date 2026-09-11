#!/usr/bin/env python3
"""Bounded real-model Stage IV test: gradients, update, save/resume and export.

Backend-agnostic: `--backend` selects which frozen sequence network is loaded.
Everything it asserts -- that the head updates, that the frozen parameters do
not, that all three feedback routes reach the packer, that a checkpoint round
trips, that generation exports -- is a property of the cycle, not of FaMPNN.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys
import torch


def select_supervised_batch(dataset):
    """Choose a bounded smoke item with observed design-side-chain targets."""
    for index in range(len(dataset)):
        batch = dataset[index]
        feat = batch["input_feature_dict"]
        observed = feat["sc_atom_mask"].bool() & feat["design_token_mask"].bool()[..., None]
        atom_count = int(observed.sum())
        if atom_count:
            return batch, dict(observed_sc_atoms=atom_count,
                               observed_sc_residues=int(observed.any(-1).sum()),
                               candidates_checked=index + 1)
    raise RuntimeError(
        "Stage IV smoke requires observed binder side-chain targets; "
        f"none of the {len(dataset)} candidate items can exercise sc_aux"
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--donor",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--backend",choices=["fampnn","ligandmpnn"],default="fampnn")
    parser.add_argument("--fampnn-checkpoint",default="")
    parser.add_argument("--ligandmpnn-checkpoint",default="")
    parser.add_argument("--ligandmpnn-source",default="")
    parser.add_argument("--cpu",action="store_true")
    a=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    import importlib.util
    spec=importlib.util.spec_from_file_location("proteoaa_training_driver",root/"scripts/training/train_protenix_monomer.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    import pandas as pd
    from pxdesign_train.runner import DesignSourceDataset, PinderPdbProvider
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents
    from pxdesign_train.data import CurriculumMultiDataset,CurriculumSchedule
    from pxdesign_train.stage4 import checkpoint_identity, generate
    from pxdesign_train.structure import write_mmcif
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    data=Path(a.data_root)
    torch.manual_seed(17)
    old=sys.argv
    sys.argv=[old[0]]
    args=driver.parse_args();sys.argv=old
    args.training_stage=f"stage4_{a.backend}";args.load_checkpoint=a.donor
    if a.backend == "fampnn":
        if not a.fampnn_checkpoint: parser.error("--fampnn-checkpoint is required for --backend fampnn")
        args.fampnn_checkpoint=a.fampnn_checkpoint
    else:
        if not (a.ligandmpnn_checkpoint and a.ligandmpnn_source):
            parser.error("--ligandmpnn-checkpoint and --ligandmpnn-source are required for --backend ligandmpnn")
        args.ligandmpnn_checkpoint=a.ligandmpnn_checkpoint
        args.ligandmpnn_source=a.ligandmpnn_source
    args.warm_start_params_only=True
    args.stage4_phase="IV-B";args.stage4_train_rounds=2;args.stage4_decode_blocks=2
    args.stage4_whole_mask_probability=0.;args.stage4_query_fraction=0.5
    args.crop_size=128;args.max_n_token=128;args.diffusion_batch_size=1
    args.num_workers=0;args.iters_to_accumulate=1;args.dtype="fp32"
    args.eval_interval=0;args.ema_decay=0.
    driver.apply_training_stage_args(args)
    device=torch.device("cpu" if a.cpu else "cuda")
    configs=driver.build_configs(args,device)
    assert configs.training.diffusion_batch_size == 1
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # Exactly one real complex; this job is an engineering smoke, not validation.
    columns=["pinder_id","pdb_path","converted_binder_chain","source_split","cluster_id","num_tokens"]
    frame=pd.read_parquet(data/"pinder/2024-02/indices/pinder_ppi_complex.parquet",columns=columns)
    frame=frame.loc[(frame.source_split == "train") & frame.num_tokens.between(48,128)].sort_values(["num_tokens", "pinder_id"]).head(16)
    manifest=output/"smoke_manifest.parquet";frame.to_parquet(manifest,index=False)
    provider=PinderPdbProvider(manifest,data/"pinder/2024-02",output/"cif_cache",
        archive_path=data/"pinder/2024-02/raw/pdbs.zip")
    dataset=DesignSourceDataset(provider,source_name="smoke_binder",crop_size=128,
        max_binder_fraction=0.75,hotspot_force_zero_prob=1.,aa_mask_mode="all",
        compute_sidechain=True,inference_safe_binder=True,backbone_only_binder=True,
        ref_pos_augment=False,seed=17)
    batch, supervision = select_supervised_batch(dataset)
    print("SMOKE sample",batch.get("sample_id"),"tokens",batch["input_feature_dict"]["design_token_mask"].numel(),"supervision",supervision,flush=True)
    multi=CurriculumMultiDataset([dataset],["smoke_binder"],[[1.]*len(dataset)])
    schedule=CurriculumSchedule(stage1={"smoke_binder":1.},stage2={"smoke_binder":1.},stage1_end_step=0,stage2_start_step=0,sources=["smoke_binder"])
    components=TrainerComponents(multi,schedule,train_samples_per_epoch=1)
    trainer=PXDesignTrainer(configs,components,device=device,checkpoint_dir=str(output/"checkpoints"),
        load_checkpoint_path=a.donor,checkpoint_params_only=True)
    assert not hasattr(trainer.raw_model,"design_residue_type_head")
    model=trainer.raw_model
    with Path(a.donor).open("rb") as stream:
        donor_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (output/"provenance.json").write_text(json.dumps(dict(donor=str(Path(a.donor).resolve()),
        donor_sha256=donor_sha, identity=checkpoint_identity(model),
        configuration=configs.to_dict(), arguments=vars(a)), indent=2, default=str))
    tensor_batch=trainer._to_device(batch)
    torch.save(batch,output/"smoke_batch.pt")
    model.train()
    from pxdesign_train.stage4 import apply_phase
    apply_phase(model)
    parameter=next(p for p in model.aa_head.parameters() if p.requires_grad)
    before=parameter.detach().clone()
    frozen_name,frozen_param=next((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    frozen_before=frozen_param.detach().clone()
    print("SMOKE train step",flush=True)
    loss=trainer.train_step(batch)
    assert all(torch.isfinite(value).all() for value in loss.values())
    assert not torch.equal(before,parameter)
    assert torch.equal(frozen_before,frozen_param),frozen_name
    # A freshly enabled zero-initialized feedback gate first learns its output
    # projection; inspect packer gradients after that first real optimizer step.
    print("SMOKE feedback gradients",flush=True)
    out=model(input_feature_dict=tensor_batch["input_feature_dict"],label_dict=tensor_batch["label_dict"],mode="train")
    sc=[p for p in model.sidechain_module.parameters() if p.requires_grad]
    gradient_metrics={}
    for name,objective in [("aa_to_sc",out["stage4_aa_revision"]),
                           ("bb_to_sc",out["post_pred_coordinate"].square().mean()),
                           ("sc_aux_to_sc",out["stage4_sc_aux"])]:
        gradients=torch.autograd.grad(objective,sc,retain_graph=True,allow_unused=True)
        total=sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
        assert total > 0 and all(torch.isfinite(g).all() for g in gradients if g is not None),(name,total)
        gradient_metrics[name]=total
    del out
    print("SMOKE save/resume",flush=True)
    checkpoint=trainer.save_checkpoint("smoke")
    trainer.load_checkpoint(checkpoint,params_only=False)
    assert trainer.step == 1
    print("SMOKE generation",flush=True)
    result=generate(model,tensor_batch["input_feature_dict"],N_step=3,refinement_steps=3,seed=17)
    write_mmcif(result["atoms"],output/"generated.cif")
    metrics=dict(status="passed",device=str(device),checkpoint=checkpoint,gradients=gradient_metrics,
        losses={k:float(v.detach()) for k,v in loss.items()},identity=checkpoint_identity(model),
        generation=result["metadata"],sample_id=batch.get("sample_id"),supervision=supervision)
    (output/"smoke_result.json").write_text(json.dumps(metrics,indent=2))
    print("STAGE4_SMOKE_PASSED",json.dumps(gradient_metrics),flush=True)


if __name__ == "__main__": main()
