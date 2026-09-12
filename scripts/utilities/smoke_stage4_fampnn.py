#!/usr/bin/env python3
"""Bounded real-model Stage IV test: gradients, update, save/resume and export."""
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
    parser.add_argument("--backbone-checkpoint",required=True)
    parser.add_argument("--sidechain-checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--fampnn-checkpoint",required=True)
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
    args.training_stage="stage4_fampnn";args.backbone_checkpoint=a.backbone_checkpoint;args.sidechain_checkpoint=a.sidechain_checkpoint
    args.fampnn_checkpoint=a.fampnn_checkpoint;args.warm_start_params_only=True
    args.stage4_phase="sc_adapt";args.stage4_train_rounds=0;args.stage4_decode_blocks=2
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
        checkpoint_params_only=True)
    assert not hasattr(trainer.raw_model,"design_residue_type_head")
    model=trainer.raw_model
    from pxdesign_train.checkpoints import read_checkpoint, component_state, BACKBONE_PREFIXES, SC_PREFIXES
    for path,prefixes in ((a.backbone_checkpoint,BACKBONE_PREFIXES),(a.sidechain_checkpoint,SC_PREFIXES)):
        expected=component_state(model,read_checkpoint(path),prefixes)
        actual=model.state_dict()
        assert all(torch.equal(actual[k].cpu(),v.to(actual[k].dtype)) for k,v in expected.items())
    def frozen_digest():
        digest=hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith(BACKBONE_PREFIXES + ("aa_head.",)):
                digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen_before=frozen_digest()
    tensor_batch=trainer._to_device(batch)
    torch.save(batch,output/"smoke_batch.pt")
    model.train()
    from pxdesign_train.stage4 import apply_phase
    apply_phase(model)
    parameter=next(p for p in model.sidechain_module.parameters() if p.requires_grad)
    before=parameter.detach().clone()
    print("SMOKE train step",flush=True)
    loss=trainer.train_step(batch)
    assert all(torch.isfinite(value).all() for value in loss.values())
    assert not torch.equal(before,parameter)
    assert frozen_before == frozen_digest()
    print("SMOKE component baseline generation and native trajectory parity",flush=True)
    baseline=generate(model,tensor_batch["input_feature_dict"],N_step=3,refinement_steps=0,seed=17,packing_enabled=False)
    packed=generate(model,tensor_batch["input_feature_dict"],N_step=3,refinement_steps=0,seed=17,packing_enabled=True)
    parity = dict(backbone_max_error=float((baseline["coordinate"]-packed["coordinate"]).abs().max()),
                  first_input_max_error=float((baseline["native_observation"]["first_input_xyz"]-packed["native_observation"]["first_input_xyz"]).abs().max()))
    print("SMOKE native parity",parity,flush=True)
    torch.save(dict(baseline=baseline["coordinate"].cpu(), packed=packed["coordinate"].cpu(),parity=parity),output/"native_parity.pt")
    assert parity["first_input_max_error"] == 0., parity
    # CUDA scatter reductions are not bitwise deterministic. The observed fp32
    # repeat error is ~2e-5 A; report it and require a bounded numerical parity.
    torch.testing.assert_close(baseline["coordinate"],packed["coordinate"],rtol=1e-5,atol=1e-4)
    assert torch.equal(packed["coordinate"],packed["initial_coordinate"])
    write_mmcif(packed["atoms"],output/"baseline_packed.cif")
    from torch.utils.data import DataLoader
    from pxdesign_train.runner.trainer import _identity_collate
    trainer.eval_dl=DataLoader([batch],batch_size=1,collate_fn=_identity_collate)
    validation=trainer.evaluate()
    gradient_metrics={}
    print("SMOKE save/resume",flush=True)
    checkpoint=trainer.save_checkpoint("smoke")
    trainer.load_checkpoint(checkpoint,params_only=False)
    assert trainer.step == 1
    # Reconstruct from the saved config without any donor paths being available.
    from pxdesign_train.checkpoints import evaluation_model
    restored=evaluation_model(checkpoint,device=device)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    del restored
    # Only after component validation: exercise feedback through frozen BB/AA.
    sys.argv=[old[0], '--warm-start-checkpoint', checkpoint,
        '--stage4-phase','feedback_adapt','--stage4-train-rounds','2',
        '--backbone-refinement-enabled','--stage4-sc-to-aa','--stage4-sc-to-bb',
        '--weight-refine','1']
    transition_args=driver.parse_args();sys.argv=old
    transition=driver.build_configs(transition_args,device)
    del trainer,model
    trainer=PXDesignTrainer(transition,components,device=device,checkpoint_dir=str(output/"feedback_checkpoints"))
    model=trainer.raw_model
    assert trainer.step==0 and model.configs.stage4.phase=='feedback_adapt'
    assert model.configs.loss.weight_bb_post==1.
    assert frozen_before == frozen_digest()
    model.train();apply_phase(model)
    out=model(input_feature_dict=tensor_batch["input_feature_dict"],label_dict=tensor_batch["label_dict"],mode="train")
    feedback=[p for p in model.parameters() if p.requires_grad]
    objective=out["post_pred_coordinate"].float().square().mean()
    gradients=torch.autograd.grad(objective,feedback,allow_unused=True)
    total=sum(float(g.detach().abs().sum()) for g in gradients if g is not None)
    assert total > 0 and all(torch.isfinite(g).all() for g in gradients if g is not None)
    gradient_metrics["refine_to_feedback"]=total
    del out
    # One bounded feedback update is an engineering gate, not quality evidence.
    trainer.train_step(batch)
    assert frozen_before == frozen_digest()
    feedback_checkpoint=trainer.save_checkpoint('feedback')
    sys.argv=[old[0], '--resume-checkpoint', feedback_checkpoint]
    resume_args=driver.parse_args();sys.argv=old
    resumed=PXDesignTrainer(driver.build_configs(resume_args,device),components,device=device)
    assert resumed.step==trainer.step==1
    assert resumed.raw_model.configs.stage4.phase=='feedback_adapt'
    assert resumed.raw_model.configs.loss.weight_bb_post==1.
    assert all(torch.equal(v,resumed.raw_model.state_dict()[k]) for k,v in model.state_dict().items())
    del resumed
    print("SMOKE generation",flush=True)
    result=generate(model,tensor_batch["input_feature_dict"],N_step=3,refinement_steps=3,seed=17)
    write_mmcif(result["atoms"],output/"generated.cif")
    metrics=dict(status="passed",device=str(device),checkpoint=checkpoint,feedback_checkpoint=feedback_checkpoint,phase_transition_and_resume=True,gradients=gradient_metrics,
        losses={k:float(v.detach()) for k,v in loss.items()},identity=checkpoint_identity(model),
        generation=result["metadata"],component_origins=model.component_origins,native_parity=parity,validation={k:float(v) for k,v in validation.items()},sample_id=batch.get("sample_id"),supervision=supervision)
    (output/"smoke_result.json").write_text(json.dumps(metrics,indent=2))
    print("STAGE4_SMOKE_PASSED",json.dumps(gradient_metrics),flush=True)


if __name__ == "__main__": main()
