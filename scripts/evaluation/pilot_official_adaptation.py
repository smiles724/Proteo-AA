#!/usr/bin/env python3
"""Bounded SC-then-feedback pilot; frozen pretrained networks, paired validation.

This estimates behavior on a small source-specific holdout. It never authorizes
pretrained-network updates or assigns native labels to free-generated binders.
"""
import argparse
import hashlib
import random
import numpy as np
import json
from pathlib import Path
import torch
import pandas as pd
from pxdesign_train.checkpoints import read_checkpoint, config_from_checkpoint, BACKBONE_PREFIXES
from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents
from pxdesign_train.runner import DesignSourceDataset, PinderPdbProvider
from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
from pxdesign_train.stage4 import apply_phase, training_forward
from pxdesign_train.initial_sampling import RandomStream
from protenix.model.protenix import update_input_feature_dict


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Validated integrated SC-adapt checkpoint')
    p.add_argument('--smoke-dir', required=True, help='Contains strict train batch and its source manifest')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps-per-phase', type=int, default=20)
    p.add_argument('--validation-items', type=int, default=2)
    p.add_argument('--seed', type=int, default=17)
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    data, output, smoke=Path(a.data_root),Path(a.output),Path(a.smoke_dir)
    output.mkdir(parents=True,exist_ok=True)
    train_batch=read_checkpoint(smoke/'smoke_batch.pt')
    train_manifest=pd.read_parquet(smoke/'smoke_manifest.parquet')
    train_clusters=set(train_manifest.cluster_id.astype(str))
    frame=pd.read_parquet(data/'pinder/2024-02/indices/pinder_ppi_complex.parquet')
    frame=frame[(frame.source_split=='val') & frame.num_tokens.between(48,256) & ~frame.cluster_id.astype(str).isin(train_clusters)]
    frame=frame.sort_values(['num_tokens','pinder_id']).drop_duplicates('cluster_id').head(a.validation_items)
    if len(frame) != a.validation_items:
        raise ValueError('Insufficient disjoint source validation clusters for the pilot')
    manifest=output/'validation_manifest.parquet';frame.to_parquet(manifest,index=False)
    provider=PinderPdbProvider(manifest,data/'pinder/2024-02',output/'cif_cache',archive_path=data/'pinder/2024-02/raw/pdbs.zip', split='val')
    dataset=DesignSourceDataset(provider,source_name='paired_holdout',crop_size=256,max_binder_fraction=.75,
        hotspot_force_zero_prob=1.,aa_mask_mode='all',compute_sidechain=True,inference_safe_binder=True,
        backbone_only_binder=True,ref_pos_augment=False,seed=81)
    multi=CurriculumMultiDataset([[train_batch]],['train'],[[1.]])
    schedule=CurriculumSchedule(stage1={'train':1.},stage2={'train':1.},stage1_end_step=0,stage2_start_step=0,sources=['train'])
    config=config_from_checkpoint(read_checkpoint(a.checkpoint))
    config.training.backbone_checkpoint=config.training.sidechain_checkpoint=config.training.resume_checkpoint=''
    config.training.warm_start_checkpoint=''
    config.training.warmup_steps=0
    config.training.ema_decay=0.;config.training.iters_to_accumulate=1
    config.stage4.phase='sc_adapt';config.stage4.train_rounds=0;config.stage4.backbone_refinement_enabled=False
    config.stage4.sc_to_aa=config.stage4.sc_to_bb=False
    trainer=PXDesignTrainer(config,TrainerComponents(multi,schedule,train_samples_per_epoch=1),
        device=torch.device('cuda'),checkpoint_dir=str(output/'checkpoints'),
        load_checkpoint_path=a.checkpoint,checkpoint_params_only=True)
    model=trainer.raw_model
    def frozen_hash():
        digest=hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith(BACKBONE_PREFIXES+('aa_head.',)):
                digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen=frozen_hash()
    # Construct paired denoising predictions once. Every arm and every phase
    # reuses these exact starting coordinates, target frames and conditioning.
    prepared=[]
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch=trainer._to_device(dataset[index])
            with RandomStream(100+index).use():
                base=model(input_feature_dict=batch['input_feature_dict'],label_dict=batch['label_dict'],mode='train')
            feat=model.diffusion_module.diffusion_conditioning.relpe.generate_relp(dict(batch['input_feature_dict']))
            feat=update_input_feature_dict(feat)
            embeddings=model.get_condition_embedding(feat)
            initial={k:base[k] for k in ('x_gt_aug','x_denoised','sigma')}
            prepared.append((batch,feat,embeddings,initial))
    arms={'no_revision':(0,False,False,False),'neither':(2,False,False,True),
          'sc_to_aa':(2,True,False,True),'sc_to_bb':(2,False,True,True),'both':(2,True,True,True)}
    evaluations=[]
    def evaluate(stage):
        model.eval()
        rows=[]
        with torch.no_grad():
            for index,(batch,feat,embeddings,initial) in enumerate(prepared):
                expected_queries=None
                for arm,(rounds,aa,bb,refine) in arms.items():
                    config.stage4.train_rounds=rounds;config.stage4.sc_to_aa=aa;config.stage4.sc_to_bb=bb
                    config.stage4.backbone_refinement_enabled=refine
                    with RandomStream(200+index).use():
                        out=training_forward(model,feat,dict(initial),*embeddings)
                    queries=[row['query_mask'] for row in out['codesign_trace']]
                    if rounds:
                        if expected_queries is None: expected_queries=queries
                        assert all(torch.equal(x,y) for x,y in zip(expected_queries,queries))
                    mask=batch['label_dict']['coordinate_mask'] * feat['backbone_loss_mask']
                    mse=trainer.loss_fn._mse_term(out['post_pred_coordinate'],initial['x_gt_aug'],mask).mean()
                    rows.append(dict(stage=stage,index=index,arm=arm,paired_refine_mse=float(mse),
                        aa_pre=float(out['stage4_aa_pre']),aa_revision=float(out['stage4_aa_revision']),
                        sc_aux=float(out['stage4_sc_aux']),physical=float(out['stage4_phys'])))
        evaluations.extend(rows)
        print('PILOT_EVAL',json.dumps(rows),flush=True)
    evaluate('start')
    losses=[]
    for phase in ('sc_adapt','feedback_adapt'):
        config.stage4.phase=phase
        config.stage4.train_rounds=0 if phase=='sc_adapt' else 2
        config.stage4.sc_to_aa=config.stage4.sc_to_bb=phase=='feedback_adapt'
        config.stage4.backbone_refinement_enabled=phase=='feedback_adapt'
        model.train();apply_phase(model);trainer._init_optimizer()
        config.loss.weight_bb_post=0. if phase=='sc_adapt' else 1.
        trainer._weight_bb_post=config.loss.weight_bb_post
        for step in range(a.steps_per_phase):
            result=trainer.train_step(train_batch)
            if any(not torch.isfinite(v).all() for v in result.values()):
                raise ValueError(f'Nonfinite {phase} step {step}')
            losses.append(dict(phase=phase,step=step,loss=float(result['loss'])))
        assert frozen_hash()==frozen
        trainer.save_checkpoint(phase)
        evaluate(phase)
    report=dict(status='complete',arguments=vars(a),evaluations=evaluations,training=losses,
        frozen_components_sha256=frozen,pretrained_weights_unchanged=True,starting_backbones_identical=True,
        validation_clusters=frame.cluster_id.astype(str).tolist(),training_clusters=sorted(train_clusters),
        scope='one training example, two source-held-out clusters; engineering pilot, not a design-quality study',
        eligible_for_backbone_updates=False)
    (output/'pilot_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ADAPTATION_PILOT_COMPLETE',output/'pilot_result.json',flush=True)


if __name__=='__main__': main()
