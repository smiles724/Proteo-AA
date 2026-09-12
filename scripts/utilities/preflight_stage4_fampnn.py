#!/usr/bin/env python3
"""Validate actual composed tensors and record resolved component provenance."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--backbone-checkpoint', default='')
    p.add_argument('--sidechain-checkpoint', default='')
    p.add_argument('--fampnn-checkpoint', default='')
    p.add_argument('--resume-checkpoint', default='')
    p.add_argument('--warm-start-checkpoint', default='')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', default='sc_adapt')
    p.add_argument('--train-rounds', type=int, default=0)
    p.add_argument('--inference-rounds', type=int, default=0)
    a, overrides = p.parse_known_args()
    root = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root))
    spec = importlib.util.spec_from_file_location('training_driver',root/'scripts/training/train_protenix_monomer.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    from pxdesign_train.model import ProtenixDesignTrain
    from pxdesign_train.checkpoints import compose_components, component_state, BACKBONE_PREFIXES, SC_PREFIXES, read_checkpoint
    from pxdesign_train.stage4 import apply_phase, implementation_identity
    original = sys.argv
    try:
        sys.argv = [original[0], '--training-stage','stage4_fampnn', '--stage4-phase',a.phase,
            '--backbone-checkpoint',a.backbone_checkpoint,'--sidechain-checkpoint',a.sidechain_checkpoint,
            '--fampnn-checkpoint',a.fampnn_checkpoint,'--resume-checkpoint',a.resume_checkpoint,
            '--warm-start-checkpoint',a.warm_start_checkpoint,'--diffusion-batch-size','1',
            '--stage4-train-rounds',str(a.train_rounds),'--stage4-inference-rounds',str(a.inference_rounds), *overrides]
        args = driver.parse_args()
    finally:
        sys.argv = original
    driver.apply_training_stage_args(args)
    config = driver.build_configs(args,torch.device('cpu'))
    model = ProtenixDesignTrain(config)
    if a.warm_start_checkpoint or a.resume_checkpoint:
        from pxdesign_train.checkpoints import restore_model
        restore_model(model,read_checkpoint(a.warm_start_checkpoint or a.resume_checkpoint))
    else:
        if not a.backbone_checkpoint or not a.fampnn_checkpoint:
            raise ValueError('Composition requires official backbone and pretrained FAMPNN checkpoints')
        compose_components(model,backbone_checkpoint=a.backbone_checkpoint,sidechain_checkpoint=a.sidechain_checkpoint or None)
        for path,prefixes in ((a.backbone_checkpoint,BACKBONE_PREFIXES),(a.sidechain_checkpoint,SC_PREFIXES)):
            if path:
                expected = component_state(model,read_checkpoint(path),prefixes)
                actual = model.state_dict()
                if any(not torch.equal(actual[k],v.to(actual[k].dtype)) for k,v in expected.items()):
                    raise AssertionError(f'Loaded tensors differ from donor: {path}')
    apply_phase(model)
    record = dict(**implementation_identity(), component_origins=model.component_origins,
        effective_config=model.configs.to_dict(), trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
        coverage=('complete integrated state' if a.warm_start_checkpoint or a.resume_checkpoint else 'complete keys and shapes; exact selected donor tensors'),
        validation_scope='Component tensors and configuration only; dataset contracts are checked by the data dry run and GPU smoke')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,default=str)+'\n')
    print(json.dumps(record,indent=2,default=str))
    print('COMPONENT_PREFLIGHT_OK',path,flush=True)


if __name__ == '__main__': main()
