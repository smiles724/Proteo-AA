#!/usr/bin/env python3
"""Freeze native RMS deviations from fixed, hashed training-data native batches.

Input manifest: partition=train, items=[{path,sha256,sample_id,pdb_id}], plus
validation/final-test manifest hashes. Native coordinates are scored by the exact
registry and atom mapper used by the repair loss; no baked reference is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import yaml
from pxdesign_train.sidechain.repair_calibration import pack_native_geometry, validate_calibration
from pxdesign_train.sidechain.chemistry import registry_sha256, canonical_registry
from pxdesign_train.sidechain.packing_objective import TERM_CLASSES, geometry_values
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.runner.sc_stream import sha256_file


def calibrate(manifest, sample_root=None):
    manifest=Path(manifest)
    data=json.loads(manifest.read_text())
    if data.get('partition') != 'train' or not data['items']:
        raise ValueError('Calibration requires a nonempty fixed training subset')
    sums={name:[0,0.,0.] for name in TERM_CLASSES}
    ids=[]
    for item in data['items']:
        path=(Path(sample_root).resolve() if sample_root else manifest.parent)/item['path']
        if sha256_file(path) != item['sha256']: raise ValueError('Calibration sample changed')
        batch=torch.load(path,map_location='cpu',weights_only=False)
        if batch['sample_id'] != item['sample_id']: raise ValueError('Calibration sample ID mismatch')
        feat=batch['input_feature_dict']
        pred=to_global(feat['sc_gt_local'].float(),feat['sc_frame_R'].float(),feat['sc_frame_t'].float())
        coords,chem,observed=pack_native_geometry(feat,pred,feat['sc_chemical_mask'],native=True)
        for name,(kind,cls) in TERM_CLASSES.items():
            idx=getattr(chem,kind+'_idx')
            valid=getattr(chem,kind+'_valid') & (getattr(chem,kind+'_class')==cls)
            valid &= observed[torch.arange(coords.shape[0])[:,None,None],idx.clamp_min(0)].all(-1)
            value=geometry_values(coords,chem,kind,active=valid)
            if kind=='bond': error=value-chem.ideal_lengths
            else: error=value.acos()-chem.ideal_angles_rad
            selected=error[valid].double()
            if not torch.isfinite(selected).all(): raise ValueError('Nonfinite native geometry')
            n,s,q=sums[name]
            sums[name]=[n+selected.numel(),s+float(selected.sum()),q+float(selected.square().sum())]
        ids.append(item['pdb_id'].lower())
    result=dict(chemistry_registry_sha256=registry_sha256(),calibration_manifest_sha256=sha256_file(manifest),
        coordinate_units='angstrom',angle_units='radians',partition='train',scale_definition='native_rms_deviation',
        tolerance_multiplier=3.,cosine_scale_floor=1e-4,pdb_ids=sorted(set(ids)),
        validation_manifest_sha256=data['validation_manifest_sha256'],final_test_manifest_sha256=data['final_test_manifest_sha256'],
        classes={name:dict(count=n,signed_mean=s/n,rms=math.sqrt(q/n)) for name,(n,s,q) in sums.items()})
    validate_calibration(result)
    return result


def reference_comparison():
    """Audit differences against the old evaluator's rounded baked templates."""
    from pxdesign_train.sidechain.templates import IDEAL_SC_LOCAL
    from pxdesign_train.sidechain.chi_constants import IDEAL_BB_LOCAL
    from pxdesign_train.sidechain.instantiate import STD_AA_3,sidechain_atoms
    from pxdesign_train.sidechain.chemistry import _angle
    differences={'bond':[],'angle':[]}
    for i,name in enumerate(STD_AA_3):
        rec=canonical_registry()[name]
        names=['N','CA','C',*sidechain_atoms(name)]
        xyz=dict(zip(names,torch.cat((IDEAL_BB_LOCAL[i],IDEAL_SC_LOCAL[i]),0).double().tolist()))
        for atoms,ref in zip(rec.bonds,rec.ideal_lengths):
            if all(a in xyz for a in atoms): differences['bond'].append(abs(math.dist(*(xyz[a] for a in atoms))-ref))
        for atoms,ref in zip(rec.angles,rec.ideal_angles_rad):
            if all(a in xyz for a in atoms): differences['angle'].append(abs(_angle(*(xyz[a] for a in atoms))-ref))
    return {kind:dict(max_abs_difference=max(values),rms_difference=math.sqrt(sum(x*x for x in values)/len(values))) for kind,values in differences.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sample-root', help='Directory containing the manifest native-*.pt cache')
    a=p.parse_args()
    result=calibrate(a.manifest,a.sample_root);result['legacy_reference_comparison']=reference_comparison()
    with open(a.output,'x') as f: yaml.safe_dump(result,f,sort_keys=False)
    print(yaml.safe_dump(result,sort_keys=False))


if __name__=='__main__': main()
