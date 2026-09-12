"""Migration contracts independent of checkpoint size and GPU availability."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
import numpy as np
from torch import nn
from pxdesign_train.checkpoints import (component_state, compose_components, normalize_state,
    integrated_record, restore_model, config_from_checkpoint, BACKBONE_PREFIXES, SC_LAYOUT_KEYS)
from pxdesign_train.codesign import CycleConfig, run_cycle
from pxdesign_train.initial_sampling import RandomStream, native_backbone
from pxdesign_train.stage4 import apply_phase, optimizer_groups, make_state, masked_aa_objective
from pxdesign_train.loss import PXDesignLoss
from test_stage4_contract import state_fixture, MockHead, mock_pack


class Components(nn.Module):
    def __init__(self):
        super().__init__()
        self.design_condition_embedder = nn.Linear(2, 2)
        self.diffusion_module = nn.Linear(2, 2)
        self.sidechain_module = nn.Linear(2, 2)
        self.aa_head = nn.Module()
        self.aa_head.sequence_network = nn.Linear(2, 20)
        self.aa_head.register_buffer("canonical_indices", torch.arange(20))
        self.aa_head.identity = dict(backend="fampnn", upstream_revision="test", mapping_version="test", model_config={})
        self.a_token_fusion = nn.Linear(2,2)
        self.refinement_pass_embedding = nn.Parameter(torch.zeros(2))
        self.aa_backend = "fampnn"
        self.enable_sidechain = True
        self.configs = SimpleNamespace(sidechain=SimpleNamespace(**{k:False for k in SC_LAYOUT_KEYS}),
            residue_type=SimpleNamespace(fampnn_checkpoint="unused"),
            stage4=SimpleNamespace(phase="sc_adapt", train_sc=False, bb_trainable_prefixes=[],
                aa_lr=1e-4, sc_lr=1e-4, bb_lr=1e-5, feedback_lr=1e-4), training=SimpleNamespace())


def donors(tmp_path, model):
    official = {k:torch.ones_like(v) for k,v in model.state_dict().items() if k.startswith(BACKBONE_PREFIXES)}
    sc = {k:torch.full_like(v, 7) for k,v in model.state_dict().items()}
    paths = tmp_path/'bb.pt', tmp_path/'sc.pt'
    torch.save(dict(model={'module._orig_mod.'+k:v for k,v in official.items()}), paths[0])
    torch.save(dict(model=sc, sidechain_arch={k:False for k in SC_LAYOUT_KEYS}), paths[1])
    return paths


def test_component_composition_is_selective_and_resume_self_contained(tmp_path):
    model = Components()
    bb, sc = donors(tmp_path, model)
    aa_before = {k:v.clone() for k,v in model.aa_head.state_dict().items()}
    feedback_before = model.a_token_fusion.weight.detach().clone()
    compose_components(nn.DataParallel(model), backbone_checkpoint=bb, sidechain_checkpoint=sc)
    assert torch.equal(model.diffusion_module.weight, torch.ones(2,2))
    assert torch.equal(model.design_condition_embedder.weight, torch.ones(2,2))
    assert torch.equal(model.sidechain_module.weight, torch.full((2,2),7.))
    assert torch.equal(model.a_token_fusion.weight, feedback_before)
    assert all(torch.equal(v,aa_before[k]) for k,v in model.aa_head.state_dict().items())
    apply_phase(model)
    checkpoint = dict(model=model.state_dict(), integrated=integrated_record(model))
    assert config_from_checkpoint(checkpoint).stage4.phase == 'sc_adapt'
    restored = restore_model(Components(), checkpoint)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert len(model.component_origins['backbone']['sha256']) == 64


@pytest.mark.parametrize('damage', ['condition', 'backbone', 'shape', 'extra'])
def test_incomplete_official_donor_fails_before_any_write(tmp_path, damage):
    model = Components()
    bb, sc = donors(tmp_path, model)
    checkpoint = torch.load(bb,weights_only=False)
    state = normalize_state(checkpoint['model'])
    if damage == 'condition': del state['design_condition_embedder.bias']
    elif damage == 'backbone': del state['diffusion_module.weight']
    elif damage == 'shape': state['diffusion_module.weight'] = torch.ones(3,2)
    else: state['diffusion_module.extra'] = torch.ones(1)
    torch.save(dict(model=state),bb)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match='Incomplete/incompatible'):
        compose_components(model,backbone_checkpoint=bb,sidechain_checkpoint=sc)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize('phase,groups', [('baseline',set()), ('sc_adapt',{'sc'}), ('sc_warmup',{'sc'}), ('sc_complex_adapt',{'sc'}),
    ('feedback_adapt',{'feedback'}), ('aa_adapt',{'aa'}), ('joint_adapt',{'aa','sc','feedback'})])
def test_phase_freezes_pretrained_components(phase, groups):
    model = Components();model.configs.stage4.phase = phase
    model.train();apply_phase(model)
    assert {g['name'] for g in optimizer_groups(model)} == groups
    assert not model.diffusion_module.training and not model.design_condition_embedder.training
    assert all(not p.requires_grad for p in model.diffusion_module.parameters())
    if phase in ('sc_warmup','sc_complex_adapt','sc_adapt','feedback_adapt','baseline'):
        assert not model.aa_head.training
        x = torch.randn(1,2,requires_grad=True)
        model.aa_head.sequence_network(x).square().sum().backward()
        assert x.grad.abs().sum() > 0


def test_zero_round_sequence_only_and_no_refinement_are_separate():
    state = state_fixture(batch=1,samples=1)
    def forbidden(*args): raise AssertionError('Unexpected pack/refine')
    final,_ = run_cycle(state,MockHead(),forbidden,forbidden,
        CycleConfig(rounds=0,packing_enabled=False,backbone_refinement_enabled=False))
    assert torch.equal(final.backbone_xyz,state.backbone_xyz)
    assert not final.generation_mask.any()
    calls=[]
    def refine(state, enabled): calls.append(enabled);return state
    run_cycle(state,MockHead(),mock_pack,refine,CycleConfig(rounds=1,sc_to_bb=False))
    assert calls == [False]
    calls.clear()
    run_cycle(state,MockHead(),mock_pack,refine,CycleConfig(rounds=1,backbone_refinement_enabled=False))
    assert calls == []


def test_rng_streams_are_independent():
    torch.manual_seed(19);before=torch.get_rng_state().clone()
    backbone=RandomStream(8);sequence=RandomStream(9)
    with backbone.use(): first=torch.randn(5)
    with sequence.use(): torch.randn(800)
    with backbone.use(): second=torch.randn(5)
    with RandomStream(8).use(): expected=torch.randn(10)
    assert torch.equal(torch.cat((first,second)),expected)
    assert torch.equal(before,torch.get_rng_state())


def test_native_observer_preserves_exact_trajectory():
    class Native:
        def diffusion_module(self, **kw): return kw['x_noisy']*.5
        def sample_diffusion(self, denoise_net, **kw):
            xyz=torch.randn(1,4,3)
            for sigma in (3.,1.):
                xyz=denoise_net(x_noisy=xyz,t_hat_noise_level=torch.tensor([sigma]))+torch.randn_like(xyz)*.1+float(np.random.rand())
            return xyz
    model=Native();empty=torch.zeros(1)
    with RandomStream(17).use(): expected=model.sample_diffusion(model.diffusion_module)
    with RandomStream(17).use(): actual,captured=native_backbone(model,{},empty,empty,empty,empty)
    assert torch.equal(expected,actual)
    assert not torch.equal(actual,captured['denoised_xyz'])


def test_denoising_weights_and_lddt_gating_precede_sample_reduction():
    loss=PXDesignLoss(align_before_mse=False,edm_weighting=True,sigma_data=2.,weight_mse=1.,weight_lddt=0.,weight_disto=0.)
    gt=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]]).expand(2,-1,-1)
    pred=gt.clone();pred[0,1,1]=1.;pred[1,1,1]=10.
    sigma=torch.tensor([1.,10.]);mask=torch.ones(3)
    kwargs=dict(pred_coordinate=pred,gt_coordinate_aug=gt,coordinate_mask=mask,sigma=sigma,rep_atom_mask=mask)
    out=loss(**kwargs)
    mse=loss._mse_term(pred,gt,mask,per_sample=True)
    expected=(mse*(sigma.square()+4)/(4*sigma.square())).mean()
    torch.testing.assert_close(out['loss'],expected)
    individual=loss(pred_coordinate=pred[:1],gt_coordinate_aug=gt[:1],coordinate_mask=mask,sigma=sigma[:1],rep_atom_mask=mask)
    torch.testing.assert_close(out['lddt'],individual['lddt']/2)
    empty=loss(**dict(kwargs,coordinate_mask=torch.zeros(3)))
    assert empty['loss'] == 0 and empty['lddt'] == 0
    assert masked_aa_objective([],torch.tensor([1]))[0] == 0


@pytest.mark.parametrize('policy', ['joint','fixed_context'])
def test_state_target_reference_is_profile_specific(policy):
    base=state_fixture(batch=1,samples=1)
    xyz=base.backbone_xyz[0]
    supplied=xyz+100.
    design=base.design_mask[0,0]
    length=design.numel()
    feat=dict(design_token_mask=design, aa_residue_mask=torch.ones_like(design),
        aa_bb_atom_idx=base.bb_atom_idx[0,0], aa_fixed_atom37_idx=torch.full((length,37),-1),
        aa_fixed_aatype=base.assigned_aa[0,0], fixed_atom_mask=(~design).repeat_interleave(4),
        atom_to_token_idx=torch.arange(length).repeat_interleave(4),
        residue_index=torch.arange(length), asym_id=base.chain_index[0,0])
    model=SimpleNamespace(enable_sidechain=False,_q_skip_cache=None,
        configs=SimpleNamespace(stage4=SimpleNamespace(initial_target_policy=policy)))
    state=make_state(model,feat,xyz,torch.ones(1),supplied)
    reference=xyz if policy == 'joint' else supplied
    assert torch.equal(state.fixed_atom_xyz[0],reference)
    assert torch.equal(state.backbone_xyz[0][state.fixed_atom_mask[0]],reference[state.fixed_atom_mask[0]])
    moved=state.on_new_backbone(state.backbone_xyz+3.,{})
    assert torch.equal(moved.backbone_xyz[state.fixed_atom_mask],state.fixed_atom_xyz[state.fixed_atom_mask])
    with pytest.raises(ValueError,match='one diffusion sample'):
        make_state(model,feat,xyz.expand(2,-1,-1),torch.ones(2),supplied)


def test_sc_architecture_mismatch_is_atomic(tmp_path):
    model=Components();bb,sc=donors(tmp_path,model)
    checkpoint=torch.load(sc,weights_only=False)
    checkpoint['sidechain_arch']['edm']=True
    torch.save(checkpoint,sc)
    before=model.diffusion_module.weight.detach().clone()
    with pytest.raises(ValueError,match='edm=false'):
        compose_components(model,backbone_checkpoint=bb,sidechain_checkpoint=sc)
    assert torch.equal(before,model.diffusion_module.weight)


def test_phase_transition_preserves_architecture_and_clears_resume_sources(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    model=Components()
    model.component_origins={'backbone':{'sha256':'official'}}
    model.configs.stage4.train_rounds=0
    model.configs.stage4.inference_rounds=0
    model.configs.stage4.backbone_refinement_enabled=False
    model.configs.stage4.sc_to_bb=False
    model.configs.stage4.initial_target_policy='joint'
    model.configs.loss=SimpleNamespace(weight_bb_post=0.)
    model.configs.training.warm_start_checkpoint='earlier_phase.pt'
    checkpoint={'integrated':integrated_record(model)}
    monkeypatch.setattr('pxdesign_train.checkpoints.read_checkpoint',lambda _: checkpoint)
    entry=Path(__file__).resolve().parents[1]/'scripts/training/train_protenix_monomer.py'
    spec=importlib.util.spec_from_file_location('phase_driver',entry)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    monkeypatch.setattr(sys,'argv',['train','--warm-start-checkpoint','sc.pt',
        '--stage4-phase','feedback_adapt','--stage4-train-rounds','2',
        '--backbone-refinement-enabled','--stage4-sc-to-bb','--weight-refine','1'])
    args=driver.parse_args()
    config=driver.build_configs(args,torch.device('cpu'))
    assert config.stage4.phase=='feedback_adapt' and config.stage4.train_rounds==2
    assert config.stage4.initial_target_policy=='joint'
    assert config.training.warm_start_checkpoint=='sc.pt'
    assert not config.training.resume_checkpoint and not config.training.backbone_checkpoint
    assert config.loss.weight_bb_post==1.
    assert config.sidechain.to_dict()==vars(model.configs.sidechain)
    # Resuming a phase-transition checkpoint must not reload its warm-start donor.
    checkpoint['integrated']['effective_config']=config.to_dict()
    monkeypatch.setattr(sys,'argv',['train','--resume-checkpoint','feedback.pt'])
    resumed=driver.build_configs(driver.parse_args(),torch.device('cpu'))
    assert resumed.training.resume_checkpoint=='feedback.pt'
    assert resumed.training.warm_start_checkpoint==''
    assert resumed.stage4.phase=='feedback_adapt'


def test_feedback_transition_requires_explicit_runtime():
    from pxdesign_train.checkpoints import transition_config
    model=Components();model.component_origins={}
    model.configs.stage4.train_rounds=0
    model.configs.loss=SimpleNamespace(weight_bb_post=0.)
    with pytest.raises(ValueError,match='explicit revision rounds'):
        transition_config({'integrated':integrated_record(model)},phase='feedback_adapt')


def test_explicit_scratch_sc_preserves_constructor_weights_and_frozen_components(tmp_path):
    model=Components();bb,sc=donors(tmp_path,model)
    before={k:v.clone() for k,v in model.state_dict().items()}
    compose_components(model,backbone_checkpoint=bb,sidechain_init='scratch')
    assert torch.equal(model.diffusion_module.weight,torch.ones(2,2))
    for k,v in model.state_dict().items():
        if not k.startswith(BACKBONE_PREFIXES): assert torch.equal(v,before[k])
    assert model.component_origins['sidechain']['origin']=='scratch'
    apply_phase(model)
    assert all(n.startswith('sidechain_module.') for n,p in model.named_parameters() if p.requires_grad)
    saved=dict(model=model.state_dict(),integrated=integrated_record(model))
    restored=restore_model(Components(),saved)
    assert restored.component_origins['sidechain']['origin']=='scratch'
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match='cannot also load'):
        compose_components(model,backbone_checkpoint=bb,sidechain_checkpoint=sc,sidechain_init='scratch')
    with pytest.raises(ValueError,match='explicit --sidechain-init scratch'):
        compose_components(Components(),backbone_checkpoint=bb)


def test_scratch_cli_uses_explicit_layout_without_donor(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from pxdesign_train.checkpoints import SCRATCH_SC_LAYOUT
    entry=Path(__file__).resolve().parents[1]/'scripts/training/train_protenix_monomer.py'
    spec=importlib.util.spec_from_file_location('scratch_driver',entry)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    fampnn=tmp_path/'fampnn.pt';fampnn.touch()
    monkeypatch.setattr(sys,'argv',['train','--training-stage','stage4_fampnn',
        '--sidechain-init','scratch','--backbone-checkpoint','official.pt',
        '--fampnn-checkpoint',str(fampnn),'--diffusion-batch-size','1',
        '--stage4-phase','sc_warmup','--data-mode','monomer'])
    args=driver.parse_args();driver.apply_training_stage_args(args)
    config=driver.build_configs(args,torch.device('cpu'))
    assert not config.training.sidechain_checkpoint
    assert config.training.sidechain_init=='scratch'
    assert {k:getattr(config.sidechain,k) for k in SCRATCH_SC_LAYOUT}==SCRATCH_SC_LAYOUT
    assert config.stage4.phase=='sc_warmup' and not config.stage4.sc_to_bb
    assert not config.sidechain.predicted_frame and not config.sidechain.predicted_mask
    assert config.sidechain.force_gt_type_logits
    assert config.stage4.weight_physical == config.stage4.weight_aa_pre == 0
    args.data_mode='mixed_monomer_complex'
    with pytest.raises(ValueError,match='monomer'):
        driver.build_configs(args,torch.device('cpu'))
    args.data_mode='monomer'
    args.sidechain_checkpoint='donor.pt'
    with pytest.raises(ValueError,match='cannot be combined'):
        driver.build_configs(args,torch.device('cpu'))


def test_native_sc_warmup_uses_native_frames_and_masks_without_sequence_decoding(monkeypatch):
    import pxdesign_train.stage4 as runtime
    length,slots=2,2
    feat=dict(aa_clean=torch.tensor([1,2]),design_token_mask=torch.tensor([True,False]),
        sc_gt_local=torch.zeros(length,slots,3),sc_frame_R=torch.eye(3).repeat(length,1,1),
        sc_frame_t=torch.tensor([[10.,0.,0.],[30.,0.,0.]]),sc_bb_coords=torch.zeros(length,3,3),
        sc_atom_mask=torch.tensor([[True,False],[True,True]]),restype=torch.full((length,32),-1.))
    labels={'coordinate':torch.randn(8,3)}
    offset=nn.Parameter(torch.ones(3))
    seen=[]
    def capture(model,f,xyz,*args):
        assert not torch.is_grad_enabled()
        assert torch.equal(xyz[0],labels['coordinate'])
        seen.append(f['restype'].clone())
        return dict(h=torch.zeros(1,1,length,4),sigma=torch.ones(1,1),q=None,
            feature_xyz=xyz[None],convention='test')
    def pack(f,out):
        assert torch.equal(out['aa_logits'].argmax(-1)[0,0],feat['aa_clean'])
        out.update(sc_generation_mask=f['design_token_mask'][None,:,None].expand(1,length,slots),
            sc_frame_R=f['sc_frame_R'][None],sc_frame_t=f['sc_frame_t'][None],
            sc_pred_global=f['sc_frame_t'][None,:,None,:]+offset[None,None,None,:])
    cfg=SimpleNamespace(phase='sc_warmup',train_rounds=0,sc_to_aa=False,sc_to_bb=False,backbone_refinement_enabled=False)
    model=SimpleNamespace(sc_predicted_frame=False,sc_predicted_mask=False,configs=SimpleNamespace(stage4=cfg),pack_backbone_state=pack)
    monkeypatch.setattr(runtime,'capture_packing_features',capture)
    out=runtime.supervised_sc_forward(model,feat,labels,None,None,None)
    assert out['sc_observed_atoms']==1
    assert out['sc_gt_mse'].item()==pytest.approx(3.,abs=1e-5)
    out['sc_gt_mse'].backward();torch.testing.assert_close(offset.grad,torch.full((3,),2.),atol=3e-6,rtol=1e-5)
    # Unobserved labels cannot change the packing inputs or masked loss.
    feat['sc_gt_local'][~feat['sc_atom_mask']]=float('nan')
    again=runtime.supervised_sc_forward(model,feat,labels,None,None,None)
    torch.testing.assert_close(out['sc_gt_mse'],again['sc_gt_mse'])
    assert all(torch.equal(v,feat['restype']) for v in seen)
    feat['sc_atom_mask'].zero_()
    empty=runtime.supervised_sc_forward(model,feat,labels,None,None,None)
    assert empty['sc_gt_mse']==0 and torch.isfinite(empty['sc_gt_mse'])
    cfg.sc_to_bb=True
    with pytest.raises(ValueError,match='feedback'):
        runtime.supervised_sc_forward(model,feat,labels,None,None,None)


def test_gt_to_generated_transition_restores_input_routing_and_objectives():
    from pxdesign_train.checkpoints import transition_config
    model=Components();model.component_origins={}
    model.configs.stage4=SimpleNamespace(phase='sc_warmup',train_rounds=0,inference_rounds=0,
        sc_to_aa=False,sc_to_bb=False,backbone_refinement_enabled=False,
        weight_aa_pre=0.,weight_aa_revision=0.,weight_physical=0.)
    model.configs.sidechain.predicted_frame=False
    model.configs.sidechain.predicted_mask=False
    model.configs.sidechain.force_gt_type_logits=True
    model.configs.loss=SimpleNamespace(weight_bb_post=0.)
    ckpt={'integrated':integrated_record(model)}
    native=transition_config(ckpt,phase='sc_complex_adapt')
    assert not native.sidechain.predicted_frame and native.stage4.weight_aa_pre==0
    generated=transition_config(ckpt,phase='sc_adapt',stage4_overrides={'weight_physical':0.02})
    assert generated.sidechain.predicted_frame and generated.sidechain.predicted_mask
    assert not generated.sidechain.force_gt_type_logits
    assert generated.stage4.weight_aa_pre==generated.stage4.weight_aa_revision==1
    assert generated.stage4.weight_physical==0.02
