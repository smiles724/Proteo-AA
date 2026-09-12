#!/usr/bin/env python3
"""Real monomer GT-packing gate for randomly initialized integrated SC."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--fampnn-checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--crop-size',type=int,default=384)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('training_driver',root/'scripts/training/train_protenix_monomer.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from smoke_stage4_fampnn import select_supervised_batch
    from pxdesign_train.runner.trainer import PXDesignTrainer
    from pxdesign_train.stage4 import apply_phase, optimizer_groups
    from pxdesign_train.checkpoints import BACKBONE_PREFIXES, component_state, read_checkpoint, evaluation_model
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],'--training-stage','stage4_fampnn','--stage4-phase','sc_warmup',
        '--sidechain-init','scratch','--backbone-checkpoint',a.backbone_checkpoint,
        '--fampnn-checkpoint',a.fampnn_checkpoint,'--data-root',a.data_root,
        '--data-mode','monomer','--crop-size',str(a.crop_size),'--max-n-token',str(a.crop_size),
        '--no-ref-pos-augment','--diffusion-batch-size','1','--iters-to-accumulate','1',
        '--dtype','bf16','--num-workers','0','--eval-num-workers','0','--eval-samples','2',
        '--eval-interval','1','--warmup-steps','0','--stage4-sc-lr','5e-5','--ema-decay','0',
        '--output-dir',str(output),'--train-samples-per-epoch','2']
    args=driver.parse_args();driver.apply_training_stage_args(args)
    torch.manual_seed(17);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    config=driver.build_configs(args,torch.device('cuda'))
    index=output/'monomer_smoke_index.csv.gz'
    driver.build_monomer_index(source_index=driver._source_index_path(Path(a.data_root)),
        output_index=index,min_n_token=a.crop_size-32,max_n_token=a.crop_size,limit=8,rebuild=True)
    components,count=driver.build_components(args,index)
    batch,supervision=select_supervised_batch(components.train_dataset.datasets[0])
    print('GT_SC_SMOKE sample',batch.get('sample_id'),supervision,flush=True)
    trainer=PXDesignTrainer(config,components,device=torch.device('cuda'),checkpoint_dir=str(output/'checkpoints'))
    model=trainer.raw_model
    assert model.component_origins['sidechain']['origin']=='scratch'
    assert {g['name'] for g in optimizer_groups(model)}=={'sc'}
    expected=component_state(model,read_checkpoint(a.backbone_checkpoint),BACKBONE_PREFIXES)
    actual=model.state_dict()
    assert all(torch.equal(actual[k].cpu(),v.to(actual[k].dtype)) for k,v in expected.items())
    del expected,actual
    def digest():
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith(BACKBONE_PREFIXES+('aa_head.',)):
                h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def forbidden(*args,**kwargs):
        raise AssertionError('GT warm-up must not decode FAMPNN or sample noisy backbones')
    model.aa_head.register_forward_pre_hook(forbidden)
    import pxdesign_train.model as sampling
    sampling.sample_diffusion_training=forbidden
    frozen=digest()
    tensor_batch=trainer._to_device(batch)
    feat=tensor_batch['input_feature_dict'];labels=tensor_batch['label_dict']
    restype=feat['restype'].clone()
    model.train();apply_phase(model)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(input_feature_dict=feat,label_dict=labels,mode='train')
    assert out['supervised_sc']
    assert torch.equal(out['sc_input_backbone'][0],labels['coordinate'])
    assert torch.equal(out['feature_xyz'][0,0],labels['coordinate'])
    assert torch.equal(out['sc_input_types'],feat['aa_clean'])
    torch.testing.assert_close(out['sc_frame_R'].reshape_as(feat['sc_frame_R']),feat['sc_frame_R'])
    torch.testing.assert_close(out['sc_frame_t'].reshape_as(feat['sc_frame_t']),feat['sc_frame_t'])
    assert torch.equal(restype,feat['restype'])
    protocol=out['protocol'];del out
    # Real-model regression: observation changes cannot affect the packer's
    # attention/output. An invalid native frame must gate every SC output/loss.
    bad_feat=dict(feat)
    bad_feat['sc_atom_mask']=feat['sc_atom_mask'].clone()
    bad_feat['sc_gt_local']=feat['sc_gt_local'].clone()
    bad_feat['sc_frame_valid']=feat['sc_frame_valid'].clone()
    bad_feat['sc_frame_R']=feat['sc_frame_R'].clone()
    residue=int((feat['sc_atom_mask'].any(-1)&feat['sc_frame_valid']).nonzero()[0])
    bad_feat['sc_frame_valid'][residue]=False
    bad_feat['sc_frame_R'][residue]=float('nan')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(73)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            base=model(input_feature_dict=bad_feat,label_dict=labels,mode='train')
        bad_feat['sc_atom_mask'].zero_()
        bad_feat['sc_gt_local'].fill_(float('nan'))
        torch.manual_seed(73)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model(input_feature_dict=bad_feat,label_dict=labels,mode='train')
        torch.testing.assert_close(base['sc_pred_global'],hidden['sc_pred_global'])
        assert not hidden['sc_model_mask'][0,residue].any()
        assert hidden['sc_chemical_mask'][0,residue].any()
        assert hidden['sc_gt_mse']==0 and torch.isfinite(hidden['sc_pred_global']).all()
        hidden['sc_pred_global'].float().square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    del base,hidden
    parameter=next(p for p in model.sidechain_module.parameters() if p.requires_grad)
    before=parameter.detach().clone()
    losses=[]
    for _ in range(2):
        loss=trainer.train_step(batch)
        assert all(torch.isfinite(v).all() for v in loss.values())
        losses.append({k:float(v) for k,v in loss.items()})
    assert not torch.equal(before,parameter)
    assert frozen==digest()
    # Exercise the real held-out monomer loader, not the training smoke batch.
    eval_loader,n_eval,eval_index=driver.build_eval_dataloader(args,output)
    trainer.eval_dl=eval_loader
    validation=trainer.evaluate()
    assert validation and not any('bb_rmsd' in k for k in validation)
    checkpoint=trainer.save_checkpoint('gt_warmup')
    trainer.load_checkpoint(checkpoint,params_only=False)
    assert trainer.step==2
    restored=evaluation_model(checkpoint,device=torch.device('cuda'))
    assert not restored.sc_predicted_frame and not restored.sc_predicted_mask
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    result=dict(status='passed',checkpoint=checkpoint,protocol=protocol,
        sample_id=batch.get('sample_id'),supervision=supervision,crop_size=a.crop_size,
        losses=losses,validation={k:float(v) for k,v in validation.items()},validation_rows=n_eval,
        pretrained_weights_unchanged=frozen==digest(),frozen_digest=frozen,
        native_frames_and_types_verified=True,no_backbone_training_sampler=True,no_fampnn_decoding=True,
        strict_backbone_restype_unchanged=True,save_resume_and_reconstruction=True,
        observation_independent_forward=True,invalid_native_frame_gated=True,
        masked_nan_forward_and_backward_finite=True,
        component_origins=model.component_origins)
    (output/'smoke_result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('GT_SC_WARMUP_SMOKE_PASSED',json.dumps(result,default=str),flush=True)


if __name__=='__main__': main()
