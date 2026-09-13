#!/usr/bin/env python3
"""Real native update/validation/checkpoint gate and SC gradient weight calibration."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch


def digest(model,sc=False):
    d=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith('sidechain_module.')==sc:
            d.update(name.encode());d.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return d.hexdigest()


def verify_frozen_ema_start(model, checkpoint_path):
    """Require every frozen parameter to equal the donor's materialized EMA."""
    from pxdesign_train.checkpoints import read_checkpoint, materialize_starting_weights
    donor = read_checkpoint(checkpoint_path)
    expected = materialize_starting_weights(donor, weights='ema', expected_step=46000)['model']
    checked = 0
    for name, parameter in model.named_parameters():
        if name.startswith('sidechain_module.'):
            continue
        if name not in expected or not torch.equal(parameter.detach().cpu(), expected[name].detach().cpu()):
            raise AssertionError(f'Frozen parameter does not equal donor EMA start: {name}')
        checked += 1
    if not checked:
        raise AssertionError('No frozen donor parameters were checked')
    del donor, expected
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--calibration-dir',required=True)
    p.add_argument('--calibration-sample-dir',help='Directory containing frozen native-*.pt calibration batches')
    p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    cal=Path(a.calibration_dir).resolve()
    samples=Path(a.calibration_sample_dir).resolve() if a.calibration_sample_dir else cal
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
    import train_sc_adaptation as driver
    from pxdesign_train.runner.trainer import PXDesignTrainer
    from pxdesign_train.checkpoints import read_checkpoint,materialize_starting_weights,config_from_checkpoint
    from pxdesign_train.runner.sc_stream import seed_all,sha256_file
    options=driver.parser().parse_args(['--accepted-checkpoint',a.checkpoint,'--phase','sc_geometry_repair',
        '--donor-weights','ema','--repair-arm','C','--output-dir',str(out),
        '--calibration-path',str(cal/'calibration.yaml'),'--source-index',str(cal/'train.csv.gz'),
        '--eval-source-index',str(cal/'validation.csv.gz'),'--final-test-index',str(cal/'final_test.csv.gz'),
        '--weight-bond-sc','.001','--weight-bond-attach','.001','--weight-angle-sc','.001','--weight-angle-attach','.001',
        '--accumulation','1','--max-steps','2','--num-workers','0','--eval-samples','1'])
    cfg,recipe=driver.resolve(options);seed_all(cfg.seed)
    components=driver.build_data(cfg,recipe,out)
    # Calibration batches are actual native training examples, not synthetic coordinates.
    manifest=json.loads((cal/'manifest.json').read_text())
    batches=[]
    for item in manifest['items']:
        sample_path=samples/item['path']
        if sha256_file(sample_path) != item['sha256']:
            raise ValueError(f'Calibration sample changed: {sample_path}')
        batch=torch.load(sample_path,map_location='cpu',weights_only=False)
        if batch['input_feature_dict']['aa_clean'].numel() <= 160:
            batch['input_feature_dict'].update(backbone_source='native',input_seed=item['seed'])
            batches.append(batch)
        if len(batches)==4: break
    if not batches: raise ValueError('No small native calibration batch')
    trainer=PXDesignTrainer(cfg,components,device=torch.device(a.device),checkpoint_dir=str(out/'checkpoints'))
    model=trainer.raw_model
    assert all(name.startswith('sidechain_module.') for name,param in model.named_parameters() if param.requires_grad)
    assert trainer.step==0 and trainer.global_step==0 and not trainer.optimizer.state
    frozen_ema_parameters_verified=verify_frozen_ema_start(model,a.checkpoint)
    if trainer.ema_wrapper:
        for name,param in trainer.model.named_parameters():
            torch.testing.assert_close(trainer.ema_wrapper.shadow[name],param,rtol=0,atol=0)
    frozen=digest(model);sc_before=digest(model,True)
    calls=[];hook=model.aa_head.register_forward_pre_hook(lambda *args:calls.append(1))
    params=[p for p in model.parameters() if p.requires_grad]
    norms=[]
    for batch in batches:
        tensor=trainer._to_device(batch)
        with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):
            result=model(input_feature_dict=tensor['input_feature_dict'],label_dict=tensor['label_dict'],mode='train')
        terms=dict(coord=result['sc_symmetry_mse'],**{k:v for k,v in result['sc_geometry'].items() if k!='counts'})
        row={}
        for name,value in terms.items():
            gradients=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
            if not all(torch.isfinite(g).all() for g in gradients if g is not None): raise ValueError('Nonfinite component gradients')
            row[name]=float(torch.stack([g.float().square().sum() for g in gradients if g is not None]).sum().sqrt())
        norms.append(row);print('GRADIENT_CALIBRATION',batch['sample_id'],row,flush=True)
        del tensor,result,terms,gradients
    means={k:sum(row[k] for row in norms)/len(norms) for k in norms[0]}
    weights={k:.25*means['coord']/means[k] for k in means if k!='coord'}
    if any(not torch.isfinite(torch.tensor(v)) or v<=0 for v in weights.values()): raise ValueError('Invalid measured weights')
    for name,value in weights.items(): cfg.stage4['weight_'+name]=value
    report=dict(weights=weights,mean_gradient_norms=means,per_batch=norms,
        samples=[b['sample_id'] for b in batches],definition='Each term starts at 0.25 of coordinate SC parameter gradient norm before ramp',
        source_checkpoint_sha256=sha256_file(a.checkpoint),source_step=46000,donor_weights='ema',
        frozen_policy='ema_start_then_frozen',frozen_ema_parameters_verified=frozen_ema_parameters_verified,
        calibration_sha256=sha256_file(cal/'calibration.yaml'))
    (out/'gradient_weights.json').write_text(json.dumps(report,indent=2))
    stream_batch=components.train_dataset[0]
    # Both disabled clash paths must be absent from the differentiable forward.
    with patch('pxdesign_train.sidechain.packing_objective.steric_clash_loss',side_effect=AssertionError('repair clash evaluated')), \
         patch('pxdesign_train.sidechain.physical.physical_loss',side_effect=AssertionError('legacy physical evaluated')):
        loss=trainer.train_step(stream_batch)
    assert trainer.step==1 and trainer.global_step==1 and not calls
    assert all(torch.isfinite(v) for v in loss.values())
    assert digest(model,True)!=sc_before and digest(model)==frozen
    validation=trainer.evaluate();assert validation and not calls
    assert digest(model)==frozen
    hook.remove()
    path=trainer.save_checkpoint('repair_gate')
    reloaded_cfg=config_from_checkpoint(read_checkpoint(path))
    reloaded_cfg.training.resume_checkpoint=path;reloaded_cfg.training.warm_start_checkpoint=''
    before_lr=trainer.optimizer.param_groups[0]['lr']
    del trainer,model;gc.collect()
    if a.device=='cuda': torch.cuda.empty_cache()
    resumed=PXDesignTrainer(reloaded_cfg,components,device=torch.device(a.device),checkpoint_dir=str(out/'reload'))
    assert resumed.step==1 and resumed.global_step==1
    assert resumed.optimizer.param_groups[0]['lr']==before_lr
    assert digest(resumed.raw_model)==frozen
    repeated=resumed.evaluate()
    for key,value in validation.items():
        if abs(value-repeated[key])>1e-6: raise AssertionError((key,value,repeated[key]))
    result=dict(passed=True,device=a.device,update={k:float(v) for k,v in loss.items()},
        validation=validation,frozen_sha256=frozen,checkpoint=path,weights=weights,
        update_sample_id=stream_batch['sample_id'],
        frozen_ema_parameters_verified=frozen_ema_parameters_verified,
        finite_sc_gradients=True,no_fampnn_decoding=True,no_clash_objective=True,exact_validation_reload=True)
    (out/'gate.json').write_text(json.dumps(result,indent=2))
    print('REPAIR_GATE_PASSED',str(out/'gate.json'),flush=True)


if __name__=='__main__': main()
