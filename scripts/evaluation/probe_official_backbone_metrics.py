#!/usr/bin/env python3
"""Measure the integrated frozen backbone with full native free generation."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import torch
from pxdesign_train.backbone_metrics import backbone_geometry
from pxdesign_train.checkpoints import evaluation_model, component_state, read_checkpoint, BACKBONE_PREFIXES
from pxdesign_train.stage4 import generate
from pxdesign_train.structure import write_mmcif


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--official-checkpoint',required=True)
    p.add_argument('--data-root',default='/hai/scratch/yfsun')
    p.add_argument('--output',required=True)
    p.add_argument('--samples-per-source',type=int,default=12)
    p.add_argument('--steps',type=int,nargs='+',default=[400])
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('driver',root/'scripts/training/train_protenix_monomer.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    old=sys.argv;sys.argv=[old[0]]
    args=driver.parse_args();sys.argv=old
    args.training_stage='stage4_fampnn';args.data_root=str(Path(a.data_root)/'protenix_data')
    args.crop_size=256;args.min_n_token=80;args.max_n_token=200;args.num_workers=0
    args.ref_pos_augment=False;args.max_crop_retries=32;args.seed=a.seed
    driver.apply_training_stage_args(args)
    index=output/'monomer_index.csv.gz'
    driver.build_monomer_index(source_index=driver._recent_index_path(Path(args.data_root)),output_index=index,
        min_n_token=80,max_n_token=200,limit=a.samples_per_source,rebuild=False)
    mono,_=driver.build_components(args,index)
    from pxdesign_train.runner import PinderPdbProvider,DesignSourceDataset
    frame=pd.read_parquet(Path(a.data_root)/'pinder/2024-02/indices/pinder_ppi_complex.parquet')
    train_clusters=set(frame.loc[frame.source_split=='train','cluster_id'].astype(str))
    frame=frame[(frame.source_split=='val') & frame.num_tokens.between(80,256) & ~frame.cluster_id.astype(str).isin(train_clusters)]
    frame=frame.sort_values(['num_tokens','pinder_id']).drop_duplicates('cluster_id').head(a.samples_per_source)
    if len(frame)!=a.samples_per_source: raise ValueError('Not enough disjoint PINDER validation clusters')
    manifest=output/'binder_manifest.parquet';frame.to_parquet(manifest,index=False)
    provider=PinderPdbProvider(manifest,Path(a.data_root)/'pinder/2024-02',output/'cif_cache',
        pdb_cache_dir=output/'pdb_cache',archive_path=Path(a.data_root)/'pinder/2024-02/raw/pdbs.zip',split='val')
    bind=DesignSourceDataset(provider,source_name='pinder_validation',crop_size=256,max_binder_fraction=.75,
        hotspot_force_zero_prob=1.,aa_mask_mode='all',compute_sidechain=True,inference_safe_binder=True,
        backbone_only_binder=True,ref_pos_augment=False,seed=a.seed)
    model=evaluation_model(a.checkpoint,device='cuda',weights='raw')
    donor=component_state(model,read_checkpoint(a.official_checkpoint),BACKBONE_PREFIXES)
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in donor.items())
    del donor
    rows=[];failures=[]
    for source,dataset in [('monomer',mono.train_dataset.datasets[0]),('binder',bind)]:
        seen=set()
        for i in range(min(a.samples_per_source,len(dataset))):
            try:
                batch=dataset[i]
                identity=str(batch.get('sample_id'))
                if identity in seen: raise ValueError('Retry returned a duplicate source sample')
                seen.add(identity)
                cpu=batch['input_feature_dict']
                topology=[cpu[k].cpu().numpy() for k in ('aa_bb_atom_idx','design_token_mask','asym_id','residue_index')]
                native=backbone_geometry(batch['label_dict']['coordinate'].cpu().numpy(),*topology,
                    observed=batch['label_dict']['coordinate_mask'].cpu().numpy())
                feat={k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()}
                for steps in a.steps:
                    result=generate(model,feat,N_step=steps,seed=a.seed+i,packing_enabled=False,refinement_steps=0,
                        backbone_sampler='pxdesign_native',initial_target_policy='joint',backbone_refinement_enabled=False)
                    xyz=result['coordinate'].detach().cpu().numpy()
                    if xyz.ndim != 2 or xyz.shape[-1] != 3:
                        raise ValueError(f'Public coordinates must be [atom,3], got {xyz.shape}')
                    stats=backbone_geometry(xyz,*topology)
                    if stats['ca_ca_nonfinite_count'] or not stats['ca_ca_count']: raise ValueError('Invalid generated geometry')
                    prefix=f'{source}_{i:03d}_{steps}'
                    write_mmcif(result['atoms'],output/f'{prefix}.cif')
                    row=dict(source=source,index=i,sample_id=identity,steps=steps,seed=a.seed+i,
                        generated=stats,native=native,protocol=result['metadata']['protocol'])
                    rows.append(row)
                    (output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print('BACKBONE_METRICS',json.dumps(row),flush=True)
            except Exception as e:
                failures.append(dict(source=source,index=i,error=f'{type(e).__name__}: {e}'))
                print('PROBE_FAILURE',failures[-1],flush=True)
    summary=[]
    for source in ('monomer','binder'):
        for steps in a.steps:
            selected=[r for r in rows if r['source']==source and r['steps']==steps]
            entry=dict(source=source,steps=steps,samples=len(selected))
            for kind in ('generated','native'):
                count=sum(r[kind]['ca_ca_count'] for r in selected)
                bad=sum(r[kind]['bad_bond_count'] for r in selected)
                entry[kind]=dict(bad_bond_count=bad,ca_ca_count=count,bad_bond_percent=100*bad/count if count else None,
                    sample_mean_bad_bond_percent=float(np.mean([100*r[kind]['bad_bond_fraction'] for r in selected if r[kind]['bad_bond_fraction'] is not None])) if selected else None)
            summary.append(entry)
    report=dict(arguments=vars(a),component_origins=model.component_origins,official_backbone_exact=True,
        profile='native joint free generation; no packing or refinement; FP32',
        definition='100 * count(|CA_i-CA_(i+1)| outside [3.5,4.1] A) / valid continuous design-residue CA pairs',
        split_scope='PINDER source validation clusters exclude training clusters; recentPDB monomers; no cross-source homology guarantee',
        summary=summary,failures=failures)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BACKBONE_SUMMARY',json.dumps(summary),flush=True)
    if failures: raise RuntimeError(f'{len(failures)} samples failed; see summary.json')


if __name__=='__main__': main()
