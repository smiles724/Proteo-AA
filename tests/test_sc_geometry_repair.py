"""Repair objective contracts: chemistry, masks, reductions and warm starts."""
from dataclasses import replace
import math
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
from pxdesign_train.sidechain.chemistry import (canonical_registry,registry_sha256,build_packing_chemistry,
    GeometryConfig,PackedAtom,CovalentLink,SC_INTERNAL,SC_ATTACHMENT,CROSS_RESIDUE)
from pxdesign_train.sidechain.packing_objective import native_geometry_repair_loss,bond_violation_loss,angle_violation_loss
from pxdesign_train.sidechain.losses import sidechain_global_frame_aligned_loss
from pxdesign_train.sidechain.instantiate import STD_AA_3,sidechain_atoms


def calibration():
    return dict(chemistry_registry_sha256=registry_sha256(),calibration_manifest_sha256='a'*64,
        coordinate_units='angstrom',angle_units='radians',partition='train',scale_definition='native_rms_deviation',
        classes={k:dict(count=100,signed_mean=0.,rms=s) for k,s in
                 [('bond_sc',.0346),('bond_attach',.0094),('angle_sc',math.radians(2.06)),('angle_attach',math.radians(2.81))]})


def fixture(names=('PRO','PHE','ASP')):
    rows,xyz,ids=[],[],{}
    for i,name in enumerate(names):
        rec=canonical_registry()[name];uid=str(i);ids[uid]=name
        for atom,coord in zip(rec.atom_names,rec.ideal_xyz):
            rows.append(PackedAtom(uid,atom,generated=atom in sidechain_atoms(name)))
            xyz.append([x+10*i for x in coord])
    chem=build_packing_chemistry([ids],[rows],config=GeometryConfig(0.,1.,0.,1.))
    return torch.tensor([xyz],requires_grad=True),chem,rows


@pytest.mark.parametrize('name',STD_AA_3)
def test_all_canonical_ideals_have_zero_loss(name):
    xyz,chem,_=fixture((name,))
    terms=native_geometry_repair_loss(xyz,chem,calibration=calibration())
    for key in calibration()['classes']: assert float(terms[key]) == 0.


def test_pro_attachment_and_internal_classes():
    xyz,chem,rows=fixture(('PRO',))
    bonds={tuple(rows[i].atom_name for i in pair):int(cls) for pair,cls in zip(chem.bond_idx[0],chem.bond_class[0])}
    assert bonds[('CA','CB')]==SC_ATTACHMENT
    assert bonds[('CD','N')]==SC_ATTACHMENT
    assert sum(x==SC_ATTACHMENT for x in bonds.values())==2
    assert sum(x==SC_INTERNAL for x in bonds.values())==2
    for idx,cls in zip(chem.angle_idx[0],chem.angle_class[0]):
        sc=[rows[i].generated for i in idx]
        assert any(sc)
        assert int(cls)==(SC_INTERNAL if all(sc) else SC_ATTACHMENT)


@pytest.mark.parametrize('term',list(calibration()['classes']))
def test_each_term_has_finite_sc_only_gradients(term):
    xyz,chem,_=fixture()
    torch.manual_seed(2)
    displaced=(xyz.detach()+.4*torch.randn_like(xyz)).requires_grad_()
    terms=native_geometry_repair_loss(displaced,chem,calibration=calibration(),terms=[term])
    assert float(terms[term])>0
    terms[term].backward()
    assert torch.isfinite(displaced.grad).all()
    assert displaced.grad[chem.subject_mask].norm()>0
    assert displaced.grad[~chem.subject_mask].count_nonzero()==0
    assert terms['counts'][term]['items']==1


def test_explicit_cross_link_stays_separate():
    _,_,rows=fixture(('CYS','CYS'))
    chem=build_packing_chemistry([{'0':'CYS','1':'CYS'}],[rows],config=GeometryConfig(0,1,0,1),
        covalent_links=[[CovalentLink(('0','SG'),('1','SG'),2.03)]])
    assert int((chem.bond_class==CROSS_RESIDUE).sum())==1
    assert int((chem.bond_class==SC_ATTACHMENT).sum())==2


def test_nonfinite_generated_is_error_even_when_term_disabled():
    xyz,chem,_=fixture()
    xyz=xyz.detach();xyz[chem.subject_mask]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        native_geometry_repair_loss(xyz,chem,calibration=calibration(),terms=['bond_sc'])


def test_invalid_anchor_only_removes_its_constraints():
    xyz,chem,rows=fixture(('PRO',))
    rows=[replace(a,valid=False) if a.atom_name=='N' else a for a in rows]
    c=build_packing_chemistry([{'0':'PRO'}],[rows],config=GeometryConfig(0,1,0,1))
    assert (c.bond_class==SC_INTERNAL).sum()==2
    assert (c.bond_class==SC_ATTACHMENT).sum()==2
    assert (c.bond_valid & (c.bond_class==SC_ATTACHMENT)).sum()==1
    # Exclusions still connect CD to CA through the absent N.
    keys={a.atom_name:i for i,a in enumerate(rows)}
    assert tuple(sorted([keys['CA'],keys['CD']])) in map(tuple,c.excluded_pairs[0].tolist())


def test_residue_then_item_reduction_and_term_mask():
    xyz=torch.tensor([[[0.,0,0],[2.,0,0],[5.,0,0],[7.,0,0],[9.,0,0]]],requires_grad=True)
    idx=torch.tensor([[0,1],[2,3],[3,4]])
    common=dict(valid_mask=torch.ones(1,5,dtype=torch.bool),subject_mask=torch.ones(1,5,dtype=torch.bool),
                group_id=torch.tensor([[5,5,99,99,99]]))
    # First residue penalty 1, second mean (4+4)/2 = 4; result 2.5.
    loss,counts=bond_violation_loss(xyz,idx,torch.tensor([1.,0.0001,0.0001]),tolerance=0,scale=1,
                                   return_counts=True,**common)
    assert float(loss)==pytest.approx((1+(2-.0001)**2)/2)
    assert counts['constraints']==3 and counts['residues']==2
    masked=bond_violation_loss(xyz,idx,1.,tolerance=0,scale=1,term_mask=torch.tensor([True,False,False]),**common)
    assert float(masked)==1.


def test_zero_angle_arm_remains_in_denominator_with_finite_penalty():
    coords=torch.zeros(1,3,3,requires_grad=True)
    loss,counts=angle_violation_loss(coords,torch.tensor([[0,1,2]]),.5,1.,scale=.1,
        valid_mask=torch.ones(1,3,dtype=torch.bool),subject_mask=torch.ones(1,3,dtype=torch.bool),
        group_id=torch.zeros(1,3,dtype=torch.long),return_counts=True)
    assert torch.isfinite(loss) and loss>0
    assert int(counts['constraints'])==1
    loss.backward()
    assert torch.isfinite(coords.grad).all()


def test_symmetry_couples_aromatic_branches_and_keeps_observed_denominator():
    target=torch.randn(1,10,3)
    names=sidechain_atoms('PHE');perm=list(range(10))
    for a,b in [('CD1','CD2'),('CE1','CE2')]:
        i,j=names.index(a),names.index(b);perm[i],perm[j]=perm[j],perm[i]
    pred=target[:,perm].clone().requires_grad_()
    mask=torch.zeros(1,10,dtype=torch.bool);mask[:,:7]=True;mask[:,names.index('CE2')]=False
    kwargs=dict(frame_R=torch.eye(3)[None],frame_t=torch.zeros(1,3),mask=mask)
    ordinary=sidechain_global_frame_aligned_loss(pred,target,**kwargs)
    symmetry=sidechain_global_frame_aligned_loss(pred,target,**kwargs,symmetry_aware=True,residue_types=torch.tensor([13]))
    assert ordinary>0 and symmetry==0
    # One-branch-only swap cannot be independently corrected.
    broken=pred.detach().clone();i,j=names.index('CD1'),names.index('CD2');broken[:,[i,j]]=broken[:,[j,i]]
    assert sidechain_global_frame_aligned_loss(broken,target,**kwargs,symmetry_aware=True,residue_types=torch.tensor([13]))>0
    identity=sidechain_global_frame_aligned_loss(pred,target,**kwargs,symmetry_aware=True,residue_types=torch.tensor([2]))
    torch.testing.assert_close(identity,ordinary)


def test_calibration_identity_is_enforced():
    xyz,chem,_=fixture();bad=dict(calibration(),chemistry_registry_sha256='b'*64)
    with pytest.raises(ValueError,match='SHA256'): native_geometry_repair_loss(xyz,chem,calibration=bad)


def test_repair_never_evaluates_clashes_or_combined_helper():
    xyz,chem,_=fixture()
    with patch('pxdesign_train.sidechain.packing_objective.steric_clash_loss',side_effect=AssertionError('clash evaluated')), \
         patch('pxdesign_train.sidechain.packing_objective.packing_geometry_loss',side_effect=AssertionError('combined evaluated')):
        result=native_geometry_repair_loss(xyz,chem,calibration=calibration())
        assert len(result)==5


def test_all_donor_ema_weights_are_materialized():
    from pxdesign_train.checkpoints import materialize_starting_weights
    source=dict(step=46000,model={'sidechain_module.w':torch.tensor([2.]),'aa_head.w':torch.tensor([3.])},
        integrated={'trainable_parameters':['sidechain_module.w']},
        ema={'shadow':{'sidechain_module.w':torch.tensor([1.]),'aa_head.w':torch.tensor([3.000001])}},optimizer={'old':True})
    start=materialize_starting_weights(source,weights='ema',expected_step=46000)
    assert start['model']['sidechain_module.w']==1.
    assert start['model']['aa_head.w']==torch.tensor([3.000001])
    assert 'optimizer' not in start and 'ema' not in start
    assert source['model']['sidechain_module.w']==2.


def test_repair_data_identity_ignores_extension_budget_but_keeps_seed():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'training'))
    import train_sc_adaptation as driver
    old=dict(hashes=['h'],settings=dict(recipe=dict(max_steps=2000,sc_lr=1e-5,
        eval_interval=500,checkpoint_interval=500,seed=42,accumulation=8,
        source_index='train.csv.gz')))
    extended=dict(hashes=['h'],settings=dict(recipe=dict(max_steps=5000,sc_lr=1e-5,
        eval_interval=500,checkpoint_interval=500,seed=42,accumulation=8,
        source_index='train.csv.gz')))
    assert driver.normalized_repair_data_identity(old)==driver.normalized_repair_data_identity(extended)
    extended['settings']['recipe']['seed']=43
    assert driver.normalized_repair_data_identity(old)!=driver.normalized_repair_data_identity(extended)


def test_selector_requires_internal_bond_improvement_and_bounds_other_classes():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'utilities'))
    from select_sc_geometry_repair import TERMS,classwise_acceptance
    donor={name:dict(rate=.20) for name in TERMS}
    control={name:dict(rate=.25) for name in TERMS}
    candidate={name:dict(rate=.19) for name in TERMS}
    assert all(row['passed'] for row in classwise_acceptance(donor,control,candidate,.02).values())
    candidate['bond_sc']['rate']=.21
    assert not classwise_acceptance(donor,control,candidate,.02)['bond_sc']['passed']
    candidate['bond_sc']['rate']=.19
    candidate['angle_attach']['rate']=.221
    assert not classwise_acceptance(donor,control,candidate,.02)['angle_attach']['passed']


def test_v2_strength_arms_are_explicit_cli_choices():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'training'))
    import train_sc_adaptation as driver
    for arm in ('D','E'):
        options=driver.parser().parse_args(['--accepted-checkpoint','donor.pt',
            '--phase','sc_geometry_repair','--repair-arm',arm,'--output-dir','out'])
        assert options.repair_arm==arm
