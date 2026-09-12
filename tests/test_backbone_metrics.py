import numpy as np
from pxdesign_train.backbone_metrics import backbone_geometry


def fixture():
    rows=np.arange(20).reshape(5,4)
    xyz=np.zeros((20,3))
    xyz[rows[:,1],0]=[0.,3.8,8.3,100.,103.8]
    return xyz,rows


def test_bad_ca_spacing_does_not_join_chains_or_gaps():
    xyz,rows=fixture()
    result=backbone_geometry(xyz,rows,np.ones(5,bool),[1,1,1,2,2],[1,2,3,1,2])
    assert result['ca_ca_count']==3 and result['bad_bond_count']==1
    assert result['bad_bond_fraction']==1/3
    result=backbone_geometry(xyz,rows,np.ones(5,bool),[1]*5,[1,2,4,8,9])
    assert result['ca_ca_count']==2 and result['bad_bond_count']==0


def test_missing_or_nonfinite_ca_does_not_bridge_residues():
    xyz,rows=fixture();rows[1,1]=-1
    result=backbone_geometry(xyz,rows,np.ones(5,bool),[1,1,1,2,2],[1,2,3,1,2])
    assert result['ca_ca_count']==1 and result['ca_ca_missing_count']==2
    assert result['bad_bond_count']==0
    xyz[rows[4,1]]=np.nan
    result=backbone_geometry(xyz,rows,np.ones(5,bool),[1,1,1,2,2],[1,2,3,1,2])
    assert result['ca_ca_count']==0 and result['ca_ca_nonfinite_count']==1
    assert result['bad_bond_fraction'] is None


def test_native_observation_mask_and_design_ownership():
    xyz,rows=fixture();observed=np.ones(20,bool);observed[rows[0,1]]=False
    result=backbone_geometry(xyz,rows,[1,1,1,0,0],[1]*5,range(5),observed)
    assert result['ca_ca_count']==1 and result['bad_bond_count']==1
    assert result['ca_ca_missing_count']==1
