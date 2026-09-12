#!/usr/bin/env python3
"""Sample actual monomer warm-up crops and count missing backbone observations."""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--seed',type=int,default=83)
    p.add_argument('--data-root',default='/hai/scratch/yfsun/protenix_data')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('audit_training_driver',root/'scripts/training/train_protenix_monomer.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    sys.argv=[sys.argv[0],'--training-stage','stage4_fampnn','--stage4-phase','sc_warmup',
        '--sidechain-init','scratch','--data-mode','monomer','--data-root',a.data_root,
        '--crop-size','384','--max-n-token','1024','--no-ref-pos-augment',
        '--num-workers','0','--max-crop-retries','64','--seed',str(a.seed)]
    args=driver.parse_args();driver.apply_training_stage_args(args)
    torch.set_num_threads(2)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    components,_=driver.build_components(args,Path(a.index))
    dataset=components.train_dataset.datasets[0]
    selected=np.random.default_rng(a.seed).choice(len(dataset),min(a.samples,len(dataset)),replace=False)
    counts=dict(canonical_design_residues=0,valid_frames=0,missing_o=0,missing_o_valid_frame=0,
                missing_o_row=0,unobserved_o_row=0,missing_n=0,missing_ca=0,missing_c=0)
    examples=[];ids=[];failures=[]
    for i,index in enumerate(selected):
        try:
            batch=dataset[int(index)];f=batch['input_feature_dict']
            design=f['design_token_mask'].bool() & (f['aa_clean']>=0) & (f['aa_clean']<20)
            observed=f['sc_bb_observed_mask'].bool();valid=f['sc_frame_valid'].bool()&design
            missing=design & ~observed[:,3]
            counts['canonical_design_residues']+=int(design.sum())
            counts['valid_frames']+=int(valid.sum())
            counts['missing_o']+=int(missing.sum())
            counts['missing_o_valid_frame']+=int((missing&valid).sum())
            counts['missing_o_row']+=int((missing&(f['sc_bb_atom_idx'][:,3]<0)).sum())
            counts['unobserved_o_row']+=int((missing&(f['sc_bb_atom_idx'][:,3]>=0)).sum())
            for slot,key in enumerate(('missing_n','missing_ca','missing_c')):
                counts[key]+=int((design&~observed[:,slot]).sum())
            sample=str(batch.get('sample_id',index));ids.append(sample)
            if missing.any() and len(examples)<10:
                examples.append(dict(sample_id=sample,missing_o=int(missing.sum()),
                                     missing_o_valid_frame=int((missing&valid).sum())))
        except Exception as exc:
            failures.append(dict(index=int(index),error=str(exc)))
        if (i+1)%32==0: print('AUDIT_PROGRESS',i+1,counts,flush=True)
    result=dict(index=str(Path(a.index).resolve()),population_rows=len(dataset),requested_samples=len(selected),
        successful_crops=len(ids),unique_sample_ids=len(set(ids)),seed=a.seed,crop_size=384,
        sampling='uniform index rows without replacement; dataset crop/retry behavior retained; sample estimate',
        counts=counts,examples=examples,failures=failures,
        missing_o_percent=100*counts['missing_o']/max(counts['canonical_design_residues'],1),
        missing_o_valid_frame_percent=100*counts['missing_o_valid_frame']/max(counts['valid_frames'],1))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('SC_INPUT_AUDIT',json.dumps(result),flush=True)
    if not ids: raise RuntimeError('No actual training examples were audited')


if __name__=='__main__':main()
