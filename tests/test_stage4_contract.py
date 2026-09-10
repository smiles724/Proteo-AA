"""State invariants, atom mapping, observed-target loss, and real head parity."""
import os
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from pxdesign_train.aa.atom_mapping import ATOM37, BB37, scatter_named_atoms
from pxdesign_train.aa.masking import assign_aa
from pxdesign_train.codesign import CoDesignState, CycleConfig, run_cycle
from pxdesign_train.sidechain.instantiate import instantiate_from_type_indices, sidechain_atoms, STD_AA_3
from pxdesign_train.sidechain.frames import frames_from_backbone_index, to_global
from pxdesign_train.sidechain.losses import sidechain_global_frame_aligned_loss


def state_fixture(batch=2, samples=2, length=6):
    xyz = torch.randn(batch, samples, length*4, 3)
    idx = torch.arange(length*4).reshape(length,4).expand(batch,samples,-1,-1)
    design = torch.ones(batch,samples,length,dtype=torch.bool)
    design[..., -2:] = False
    aa = torch.arange(length).expand(batch,samples,-1).clone()
    aa[..., 0] = 7
    aa[0,0,0] = 17  # different Trp/Gly inventory per item and sample
    ids, mask = instantiate_from_type_indices(aa)
    mask = mask & design[..., None]
    ids = torch.where(mask,ids,0)
    fixed_atoms = (~design).repeat_interleave(4,dim=-1)
    return CoDesignState(xyz, {}, idx, aa, torch.randn(batch,samples,length,10,3), ids, mask,
        torch.zeros(batch,samples,length,37,3), torch.zeros(batch,samples,length,37,dtype=torch.bool),
        xyz.clone(), fixed_atoms, torch.ones_like(design), design, torch.zeros_like(design),
        torch.ones_like(design), torch.ones_like(design),
        torch.arange(length).expand(batch,samples,-1), torch.zeros_like(aa))


def test_all_residue_mapping_and_two_by_two_axes():
    aa = torch.arange(20).reshape(2,2,5)
    ids, mask = instantiate_from_type_indices(aa)
    xyz = torch.randn(2,2,5,10,3, requires_grad=True)
    mapped, present = scatter_named_atoms(xyz,ids,mask)
    for i, name in enumerate(STD_AA_3):
        b,s,l = i//10,(i//5)%2,i%5
        for j, atom in enumerate(sidechain_atoms(name)):
            assert torch.equal(mapped[b,s,l,ATOM37.index(atom)], xyz[b,s,l,j])
        assert present[b,s,l].sum() == len(sidechain_atoms(name))
    mapped.sum().backward()
    assert torch.equal(xyz.grad[...,0] != 0, mask)
    state = state_fixture()
    R,t,valid = frames_from_backbone_index(state.backbone_xyz,state.bb_atom_idx)
    assert valid.shape == (2,2,6)
    assert torch.equal(t, state.backbone_xyz[...,1::4,:])
    assert state.generation_mask[0,0,0].sum() == 10
    assert state.generation_mask[1,1,0].sum() == 0


def test_query_input_hides_id_names_coordinates_and_masks():
    state = state_fixture()
    query = state.design_mask.clone(); query[...,1:] = False
    state = replace(state,query_mask=query)
    expected = state.aa_input()
    changed = replace(state,assigned_aa=torch.where(query, -500, state.assigned_aa),
        sc_xyz=torch.where(query[...,None,None], float("nan"),state.sc_xyz),
        sc_atom_name_ids=torch.where(query[...,None],99999,state.sc_atom_name_ids),
        generation_mask=state.generation_mask | query[...,None])
    actual = changed.aa_input()
    for key in expected:
        assert torch.equal(actual[key],expected[key]), key
    assert torch.all(actual["aatype_noised"][query] == 20)
    assert actual["atom_mask_noised"][query].sum() == 4 * query.sum()


def test_missing_lys_nz_generated_but_loss_invariant():
    aa = torch.tensor([11])
    ids,chem = instantiate_from_type_indices(aa)
    observed = chem.clone(); observed[..., -1] = False
    nz = sidechain_atoms("LYS").index("NZ")
    observed[...,nz] = False
    assert chem[0,nz]
    pred = torch.randn(1,1,10,3,requires_grad=True)
    target = torch.randn_like(pred)
    R = torch.eye(3).reshape(1,1,3,3); t = torch.zeros(1,1,3)
    loss = sidechain_global_frame_aligned_loss(pred,target,R,t,observed)
    changed = target.clone(); changed[...,nz,:] = 10000
    other = sidechain_global_frame_aligned_loss(pred,changed,R,t,observed)
    assert torch.equal(loss,other)
    changed[...,nz,:] = float("nan")
    masked_nan = sidechain_global_frame_aligned_loss(pred,changed,R,t,observed)
    assert torch.equal(loss,masked_nan)
    loss.backward()
    assert torch.equal(pred.grad[...,nz,:],torch.zeros_like(pred.grad[...,nz,:]))


def test_temperature_and_unknown_exclusion():
    logits = torch.zeros(1000,20); logits[:,3] = 3
    assert (assign_aa(logits,0) == 3).all()
    a = assign_aa(logits,1,torch.Generator().manual_seed(9))
    b = assign_aa(logits,1,torch.Generator().manual_seed(9))
    assert torch.equal(a,b) and (a != 3).any() and (a < 20).all()
    with pytest.raises(ValueError): assign_aa(logits,-1)


def mock_pack(state):
    ids, mask = instantiate_from_type_indices(state.assigned_aa)
    R,t,valid = frames_from_backbone_index(state.backbone_xyz,state.bb_atom_idx)
    mask = mask & (state.design_mask & valid)[...,None]
    local = torch.arange(30,dtype=state.backbone_xyz.dtype).reshape(10,3).expand(*mask.shape,3)/20
    xyz = to_global(local,R,t)
    return state.updated(sc_xyz=torch.where(mask[...,None],xyz,0),
        generation_mask=mask,sc_atom_name_ids=torch.where(mask,ids,0),
        sc_visible=state.seq_visible,feedback={})


class MockHead:
    def __call__(self,**inputs):
        x=inputs["denoised_coords"]
        logits=x.sum((-1,-2))[...,None].expand(*x.shape[:-2],20).clone()
        logits[...,11] += 10
        return logits,{}


def test_cycle_parity_fixed_receptor_and_final_state():
    state=state_fixture()
    def refine(st,enabled):
        return st.on_new_backbone(st.backbone_xyz+0.01,{})
    cfg=CycleConfig(rounds=3,decode_blocks=2)
    a,records=run_cycle(state,MockHead(),mock_pack,refine,cfg,torch.Generator().manual_seed(7))
    with torch.no_grad():
        b,_=run_cycle(state,MockHead(),mock_pack,refine,cfg,torch.Generator().manual_seed(7))
    assert torch.equal(a.sc_xyz,b.sc_xyz) and torch.equal(a.assigned_aa,b.assigned_aa)
    assert torch.equal(a.backbone_xyz[a.fixed_atom_mask],state.backbone_xyz[state.fixed_atom_mask])
    assert len(records["trace"]) == 3
    a.validate_final()


@pytest.fixture(scope="module")
def real_head():
    path=os.environ.get("FAMPNN_CHECKPOINT")
    if not path: pytest.skip("Set FAMPNN_CHECKPOINT to run released-weight parity/gradient gates")
    from pxdesign_train.aa.fampnn_head import FaMPNNHead
    return FaMPNNHead(path).eval()


def test_released_head_parity_query_invariance_and_coordinate_gradient(real_head):
    state=state_fixture(batch=1,samples=1,length=8)
    state=replace(state,query_mask=torch.zeros_like(state.design_mask))
    query=state.query_mask.clone();query[...,0]=True
    state=replace(state,query_mask=query,sc_xyz=state.sc_xyz.detach().requires_grad_())
    inputs=state.aa_input()
    rng=torch.get_rng_state()
    actual,_=real_head(**inputs)
    torch.set_rng_state(rng)
    official_inputs={k:v.flatten(0,1) for k,v in inputs.items()}
    expected,_=real_head.sequence_network(**official_inputs)
    torch.testing.assert_close(actual.flatten(0,1),expected[...,:20],rtol=0,atol=0)
    loss=actual[...,0,:].square().sum()
    grad,=torch.autograd.grad(loss,state.sc_xyz)
    assert torch.isfinite(grad).all()
    assert grad[...,1:6,:,:].abs().sum() > 0
    assert grad[...,0,:,:].abs().sum() == 0


def test_cluster_weights_and_partitions_use_eligible_rows():
    from pxdesign_train.data.clusters import inverse_cluster_weights,ClusterPartitionProvider
    ids=["a"]*5+["b"]*2+["c"]
    weights=inverse_cluster_weights(ids)
    assert sum(weights[:5]) == pytest.approx(1.)
    assert sum(weights[5:7]) == pytest.approx(1.)
    class Provider:
        def __len__(self): return 100
        def __getitem__(self,i): return i
    ids=[str(i//4) for i in range(100)]
    train=ClusterPartitionProvider(Provider(),ids,validation=False,fraction=.3,seed=4)
    val=ClusterPartitionProvider(Provider(),ids,validation=True,fraction=.3,seed=4)
    assert not set(train.cluster_ids) & set(val.cluster_ids)
    assert sorted(train.indices+val.indices) == list(range(100))


def test_released_head_updates_real_packer_through_visible_context(real_head):
    from types import SimpleNamespace
    from pxdesign_train.sidechain.module import SideChainModule
    from pxdesign_train.sidechain.frames import gather_backbone
    from pxdesign_train.stage4 import apply_phase, optimizer_groups
    torch.manual_seed(13)
    state = state_fixture(batch=1, samples=1, length=8)
    assigned = torch.full_like(state.assigned_aa, 11)
    ids, chemistry = instantiate_from_type_indices(assigned)
    mask = chemistry & state.design_mask[..., None]
    query = torch.zeros_like(state.design_mask); query[...,0] = True
    state = replace(state, assigned_aa=assigned, sc_atom_name_ids=torch.where(mask, ids, 0),
                    generation_mask=mask, query_mask=query)
    model = torch.nn.Module()
    model.aa_head = real_head
    model.sidechain_module = SideChainModule(c_res=16, c_atom=32, n_blocks=1, n_heads=4,
        n_cross_blocks=1, a_bs_concat=True, centre_coord_input=True)
    model.frozen_reference = torch.nn.Linear(3,3)
    model.configs = SimpleNamespace(stage4=SimpleNamespace(phase="IV-B", bb_trainable_prefixes=[],
                                                         aa_lr=1e-5, sc_lr=1e-5, bb_lr=1e-6))
    apply_phase(model)
    bb,_ = gather_backbone(state.backbone_xyz, state.bb_atom_idx)
    R,t,valid = frames_from_backbone_index(state.backbone_xyz, state.bb_atom_idx)
    xyz,_,_ = model.sidechain_module(h_res=torch.randn(1,8,16),
        restype_logits=torch.nn.functional.one_hot(assigned[0],20).float(),
        atom_name_ids=state.sc_atom_name_ids[0], atom_mask=mask[0], noisy_coords=state.sc_xyz[0],
        t=torch.ones(1), ca_coords=t[0], frame_R=R[0], frame_t=t[0],
        bb_coords=bb[0], res_mask=valid[0], ctx_mask=~state.design_mask[0])
    state = replace(state, sc_xyz=xyz[None])
    logits,_ = real_head(**state.aa_input())
    loss = -logits[...,0,:].log_softmax(-1)[...,11].mean()
    optimizer = torch.optim.Adam(optimizer_groups(model))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    packer_grads = [p.grad for p in model.sidechain_module.parameters() if p.grad is not None]
    assert packer_grads and all(torch.isfinite(g).all() for g in packer_grads)
    assert sum(float(g.abs().sum()) for g in packer_grads) > 0
    before = model.sidechain_module.out.weight.detach().clone()
    frozen = model.frozen_reference.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.sidechain_module.out.weight)
    assert torch.equal(frozen, model.frozen_reference.weight)
    model.configs.stage4.phase = "IV-A"
    model.train(); apply_phase(model)
    assert not model.sidechain_module.training
    assert not any(p.requires_grad for p in model.sidechain_module.parameters())
    assert all(p.requires_grad for p in model.aa_head.parameters())
    real_head.eval()


def test_pinder_filters_heldout_clusters_and_crop_sizes_before_weights(tmp_path):
    import pandas as pd
    from pxdesign_train.runner.pinder_provider import PinderPdbProvider
    from pxdesign_train.data.clusters import inverse_cluster_weights
    rows = [
        ("train_a1", "train", "a", 30), ("train_a2", "train", "a", 40),
        ("train_b", "train", "b", 20), ("oversize", "train", "b", 100),
        ("leak", "train", "heldout", 20), ("val", "val", "heldout", 20),
    ]
    frame = pd.DataFrame(rows, columns=["pinder_id", "source_split", "cluster_id", "binder_tokens"])
    frame["pdb_path"] = "unused.pdb"; frame["converted_binder_chain"] = "B"; frame["num_tokens"] = 120
    path = tmp_path / "manifest.parquet"; frame.to_parquet(path)
    provider = PinderPdbProvider(path, tmp_path, tmp_path / "cache", cluster_disjoint=True, max_binder_tokens=48)
    assert provider._pinder_ids == ["train_a1", "train_a2", "train_b"]
    assert inverse_cluster_weights(provider.cluster_ids) == [.5, .5, 1.]


def test_final_named_atom_export_keeps_fixed_coordinates(tmp_path):
    from pxdesign_train.structure import assemble_atoms, write_mmcif
    import biotite.structure.io.pdbx as pdbx
    state = mock_pack(state_fixture(batch=1,samples=1,length=6))
    n = state.backbone_xyz.shape[-2]
    feat = dict(atom_to_token_idx=torch.arange(6).repeat_interleave(4),
        structure_atom_name=["N","CA","C","O"]*6, structure_res_name=["ALA"]*n,
        structure_chain_id=["B"]*16+["R"]*8, structure_res_id=torch.arange(6).repeat_interleave(4),
        structure_element=["N","C","C","O"]*6, structure_ins_code=[""]*n,
        structure_hetero=torch.zeros(n,dtype=torch.bool), fixed_atom_mask=state.fixed_atom_mask[0,0])
    atoms = assemble_atoms(state,feat)
    assert len(atoms["atom_name"]) == n + int(state.generation_mask.sum())
    fixed = torch.tensor([chain == "R" for chain in atoms["chain_id"]])
    torch.testing.assert_close(atoms["coordinate"][fixed],state.fixed_atom_xyz[state.fixed_atom_mask])
    path = tmp_path / "generated.cif"; write_mmcif(atoms,path)
    saved = pdbx.get_structure(pdbx.CIFFile.read(path),model=1)
    assert saved.array_length() == len(atoms["atom_name"])
    assert list(saved.atom_name) == atoms["atom_name"]
    import biotite.structure as struc
    assert struc.get_residue_count(saved) == 6
