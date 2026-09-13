#!/usr/bin/env python3
"""Freeze repair data partitions and a native training calibration cache.

A deterministic PDB-level holdout from the donor training index is reserved for
final repair evaluation. It is NOT independent of donor pretraining. Existing
final-test manifests can instead be supplied with --final-test-index.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-index',required=True);p.add_argument('--validation-index',required=True)
    p.add_argument('--final-test-index');p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=32);p.add_argument('--final-pdbs',type=int,default=128)
    p.add_argument('--data-root',default='/hai/scratch/yfsun/protenix_data')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('PROTENIX_ROOT_DIR',a.data_root);os.environ.setdefault('PROTENIX_DATA_ROOT_DIR',a.data_root+'/common')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
    import train_sc_adaptation as driver
    from pxdesign_train.runner.sc_stream import item_rng,sha256_file
    train=pd.read_csv(a.train_index);val=pd.read_csv(a.validation_index)
    ids=train.pdb_id.str.lower();val_ids=set(val.pdb_id.str.lower())
    train=train[~ids.isin(val_ids)];ids=train.pdb_id.str.lower()
    ordered=sorted(set(ids),key=lambda x:hashlib.sha256(('repair-partition-v1/'+x).encode()).hexdigest())
    final=pd.read_csv(a.final_test_index) if a.final_test_index else train[ids.isin(set(ordered[:a.final_pdbs]))]
    final_ids=set(final.pdb_id.str.lower())
    if final_ids & val_ids: raise ValueError('Final/validation overlap')
    train=train[~ids.isin(final_ids)]
    eligible=[x for x in ordered if x not in final_ids]
    selected=train[train.pdb_id.str.lower().isin(set(eligible[:a.samples]))].drop_duplicates('pdb_id').copy()
    selected=selected.sort_values(['pdb_id'])
    for name,frame in [('train',train),('validation',val),('final_test',final),('calibration_subset',selected)]:
        dest=out/(name+'.csv.gz')
        if dest.exists(): raise FileExistsError(dest)
        frame.to_csv(dest,index=False,compression=dict(method='gzip',mtime=0))
    args=driver.legacy_arguments(dict(phase='sc_geometry_repair',monomer_fraction=1.,seed=0,data_root=a.data_root),out)
    driver.base._bootstrap_paths(args)
    components,_=driver.base.build_components(args,out/'calibration_subset.csv.gz')
    items=[]
    for i in range(len(components.train_dataset)):
        seed=910000+i
        with item_rng(seed): batch=components.train_dataset[i]
        sample=str(batch['sample_id'])
        pdb=sample[:4].lower()
        if pdb not in set(selected.pdb_id.str.lower()):
            raise ValueError(f'Calibration retry escaped selected subset: {sample}')
        file=out/f'native-{i:04d}.pt';torch.save(batch,file)
        items.append(dict(path=file.name,sha256=sha256_file(file),sample_id=batch['sample_id'],pdb_id=pdb,seed=seed))
        print('CALIBRATION_NATIVE',i,sample,flush=True)
    manifest=dict(partition='train',items=items,validation_manifest_sha256=sha256_file(out/'validation.csv.gz'),
        final_test_manifest_sha256=sha256_file(out/'final_test.csv.gz'),
        train_manifest_sha256=sha256_file(out/'train.csv.gz'),donor_pretraining_independence=False,
        final_test_origin='supplied' if a.final_test_index else 'PDB holdout from donor training; repair-only holdout')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from calibrate_sc_geometry import calibrate,reference_comparison
    import yaml
    artifact=calibrate(out/'manifest.json');artifact['legacy_reference_comparison']=reference_comparison()
    with open(out/'calibration.yaml','x') as f: yaml.safe_dump(artifact,f,sort_keys=False)
    print(yaml.safe_dump(artifact,sort_keys=False),flush=True)


if __name__=='__main__': main()
