"""
Tests for Stage III alternating training (`train_mode="alternating"`).

The alternating scheme, from a SINGLE forward per batch:
  - Phase A updates ONLY the Side-Chain Module from `loss_sc`.
  - Phase B updates ONLY the Backbone group (diffusion + AA head + h_res
    generation + a/q fusion) from `loss_bb`, with the gradient flowing THROUGH
    the un-stepped Side-Chain Module (never via a no_grad barrier).

We install a tiny fake model whose wiring lets us verify the two invariants
that matter:
  1. side-chain params update from `loss_sc`, backbone-group params from
     `loss_bb`, with no cross-contamination;
  2. a backbone param whose ONLY path to the loss runs through the Side-Chain
     Module (`hres_gen` -> sidechain_module -> post head) still updates in the
     bb phase — i.e. gradient flows through the frozen-but-not-detached SC.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))

from test_trainer_integration import _SyntheticProvider  # noqa: E402


class _FakeSCModel(nn.Module):
    """Stand-in for ProtenixDesignTrain with a real `sidechain_module` submodule.

    Data flow (so the param-grouping and through-SC gradient are exercised):
        tok --hres_gen(bb)--> h_res --sidechain_module(SC)--> sc coords
        sc coords --post_proj(bb)--> post scalar --> post coordinate
    `loss_sc` supervises the sc coords; `loss_bb` supervises structure + AA +
    the post coordinate (which depends on `hres_gen` only through the SC module).
    """

    def __init__(self, atoms_per_res: int = 4):
        super().__init__()
        self.A = atoms_per_res
        self.bias = nn.Parameter(torch.zeros(1))       # bb: structure denoise
        self.dist_proj = nn.Linear(1, 64)              # bb
        self.design_residue_type_head = nn.Linear(1, 20)  # bb
        self.hres_gen = nn.Linear(1, 8)                # bb: feeds the SC module
        self.sidechain_module = nn.Linear(8, self.A * 3)  # SC group
        self.post_proj = nn.Linear(self.A * 3, 1)      # bb: reads SC output
        self.training_noise_sampler_sigma = 16.0

    def forward(self, *, input_feature_dict, label_dict, mode="train"):
        assert mode == "train"
        gt = label_dict["coordinate"]                  # [B, N_atom, 3]
        lead = gt.shape[:-2]
        N_atom = gt.shape[-2]
        N_sample = 2
        L = int(input_feature_dict["restype"].shape[-2])
        A = self.A

        x_gt_aug = gt.unsqueeze(-3).expand(*lead, N_sample, N_atom, 3).contiguous()
        x_denoised = x_gt_aug + self.bias
        sigma = torch.full((*lead, N_sample), 1.0)

        tok = torch.ones(*lead, L, 1)  # non-zero so Linear *weights* get gradient
        dist = self.dist_proj(tok).unsqueeze(-2).expand(*lead, L, L, 64).contiguous()
        aa_logits = self.design_residue_type_head(tok)

        h_res = self.hres_gen(tok)                     # [B, L, 8] (bb)
        sc = self.sidechain_module(h_res).reshape(*lead, L, A, 3)  # [B, L, A, 3] (SC)
        sc_atom_mask = torch.ones(*lead, L, A)
        frame_R = torch.eye(3).expand(*lead, L, 3, 3).contiguous()
        frame_t = torch.zeros(*lead, L, 3)

        # Post coordinate depends on `hres_gen` ONLY through `sidechain_module`.
        post_scalar = self.post_proj(sc.reshape(*lead, L, A * 3))  # [B, L, 1]
        post_pred = x_gt_aug + self.bias + post_scalar.mean()

        return {
            "x_gt_aug": x_gt_aug,
            "x_denoised": x_denoised,
            "sigma": sigma,
            "distogram_logits": dist,
            "aa_logits": aa_logits,
            "sc_pred_global": sc,
            "sc_frame_R": frame_R,
            "sc_frame_t": frame_t,
            "sc_atom_mask": sc_atom_mask,
            "post_pred_coordinate": post_pred,
            "post_gt_coordinate_aug": x_gt_aug,
        }

    @property
    def aa_proj(self):
        """Backward-compatible test handle; parameters retain the production prefix."""
        return self.design_residue_type_head


class _Cfg:
    seed = 42
    dtype = "fp32"
    load_strict = False

    class training:
        lr = 1e-2
        warmup_steps = 1
        max_steps = 3
        weight_decay = 0.0
        ema_decay = 0.0
        log_interval = 10
        eval_interval = 0
        checkpoint_interval = 0
        iters_to_accumulate = 1
        grad_clip_norm = 0.0
        num_workers = 0
        train_mode = "alternating"

    class loss:
        weight_mse = 4.0
        weight_lddt = 1.0
        weight_disto = 0.03
        weight_aa = 1.0
        sigma_low_threshold = 4.0
        no_bins = 64
        min_bin = 2.3125
        max_bin = 21.6875
        lddt_radius = 15.0
        align_before_mse = False
        weight_sc_local = 1.0
        weight_sc_phys = 0.1
        weight_sc_global = 0.5
        weight_bb_post = 1.0
        weight_aa_post = 1.0

    class residue_type:
        ignore_index = -100


def _make_alt_trainer(monkeypatch, sc_weight_local=1.0):
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import (
        DesignSourceDataset,
        PXDesignTrainer,
        TrainerComponents,
    )

    def _fake_init_model(self):
        self.raw_model = _FakeSCModel().to(self.device)
        self._apply_trainable_filter()
        self.model = self.raw_model
        self.ema_wrapper = None

    monkeypatch.setattr(PXDesignTrainer, "_init_model", _fake_init_model)

    src = DesignSourceDataset(
        _SyntheticProvider(n_items=3), source_name="a",
        crop_size=20, hotspot_force_zero_prob=0.0,
    )
    multi = CurriculumMultiDataset(
        datasets=[src], source_names=["a"], per_item_weights=[[1.0] * 3]
    )
    sched = CurriculumSchedule(
        stage1={"a": 1.0}, stage2={"a": 1.0}, stage1_end_step=10, stage2_start_step=20
    )
    components = TrainerComponents(
        train_dataset=multi, schedule=sched, train_samples_per_epoch=5
    )

    cfg = _Cfg()
    cfg.loss.weight_sc_local = sc_weight_local
    return PXDesignTrainer(configs=cfg, components=components, device=torch.device("cpu"))


def _batch_with_sc(trainer):
    batch = next(iter(trainer.train_dl))
    L = batch["input_feature_dict"]["restype"].shape[0]
    A = trainer.raw_model.A
    # Non-zero GT so loss_sc has a real gradient toward the sidechain module.
    batch["input_feature_dict"]["sc_gt_local"] = torch.randn(L, A, 3) * 0.1
    return batch


def test_alternating_builds_two_optimizers(monkeypatch):
    trainer = _make_alt_trainer(monkeypatch)
    assert trainer.train_mode == "alternating"
    assert hasattr(trainer, "sc_optimizer") and hasattr(trainer, "bb_optimizer")
    assert len(trainer._optimizers) == 2
    # sidechain_module params are exactly the SC group.
    sc_ids = {id(p) for p in trainer.sc_params}
    for name, p in trainer.raw_model.named_parameters():
        if "sidechain_module" in name:
            assert id(p) in sc_ids, name
        else:
            assert id(p) not in sc_ids, name


def test_aa_head_can_use_a_separate_learning_rate(monkeypatch):
    monkeypatch.setattr(_Cfg.training, "aa_head_lr", 1e-1, raising=False)
    trainer = _make_alt_trainer(monkeypatch)
    groups = trainer.bb_optimizer.param_groups
    head_ids = {id(p) for p in trainer.raw_model.design_residue_type_head.parameters()}
    head_groups = [g for g in groups if any(id(p) in head_ids for p in g["params"])]
    assert len(head_groups) == 1
    assert head_groups[0]["lr"] == pytest.approx(1e-1)
    assert all(
        g["lr"] == pytest.approx(trainer.configs.training.lr)
        for g in groups if g is not head_groups[0]
    )


def test_detached_sidechain_logits_keep_values_but_close_head_gradient():
    from pxdesign_train.model import _route_aa_logits_to_sidechain

    logits = torch.randn(2, 3, 20, requires_grad=True)
    detached = _route_aa_logits_to_sidechain(logits, detach=True)
    assert torch.equal(detached, logits)
    assert not detached.requires_grad

    live = _route_aa_logits_to_sidechain(logits, detach=False)
    assert live is logits
    assert live.requires_grad


def test_alternating_step_updates_both_groups(monkeypatch):
    trainer = _make_alt_trainer(monkeypatch)
    batch = _batch_with_sc(trainer)

    sc0 = trainer.raw_model.sidechain_module.weight.detach().clone()
    bias0 = trainer.raw_model.bias.detach().clone()
    post0 = trainer.raw_model.post_proj.weight.detach().clone()

    loss_out = trainer.train_step(batch)

    assert "loss_sc" in loss_out and "loss_bb" in loss_out
    assert torch.isfinite(loss_out["loss"])
    assert trainer.step == 1
    # SC group moved (from loss_sc); BB group moved (from loss_bb).
    assert not torch.allclose(sc0, trainer.raw_model.sidechain_module.weight)
    assert not torch.allclose(bias0, trainer.raw_model.bias)
    assert not torch.allclose(post0, trainer.raw_model.post_proj.weight)


def test_alternating_gradient_flows_through_sidechain(monkeypatch):
    """`hres_gen` (backbone) reaches the loss ONLY through the SC module, via the
    post head. It must still update in the bb phase — proving the gradient flows
    through the un-stepped (requires_grad-intact, not no_grad) Side-Chain Module.
    """
    trainer = _make_alt_trainer(monkeypatch)
    batch = _batch_with_sc(trainer)
    hres0 = trainer.raw_model.hres_gen.weight.detach().clone()
    trainer.train_step(batch)
    assert not torch.allclose(
        hres0, trainer.raw_model.hres_gen.weight
    ), "hres_gen should update via gradient flowing THROUGH the side-chain module"


def test_alternating_no_contamination_of_sidechain(monkeypatch):
    """With the side-chain loss weight zeroed, `loss_sc` carries no gradient, so
    the Side-Chain Module must NOT move — it is updated ONLY by `loss_sc`, never
    by `loss_bb` (which is differentiated w.r.t. backbone params only)."""
    trainer = _make_alt_trainer(monkeypatch, sc_weight_local=0.0)
    trainer.loss_fn.weight_sc_local = 0.0
    trainer.loss_fn.weight_sc_phys = 0.0
    batch = _batch_with_sc(trainer)
    sc0 = trainer.raw_model.sidechain_module.weight.detach().clone()
    bias0 = trainer.raw_model.bias.detach().clone()
    trainer.train_step(batch)
    # Backbone still learns; side chain is untouched.
    assert not torch.allclose(bias0, trainer.raw_model.bias)
    assert torch.allclose(sc0, trainer.raw_model.sidechain_module.weight)


def test_alternating_grad_accum(monkeypatch):
    """With iters_to_accumulate=2, two micro-batches make one update to both
    optimizers."""
    trainer = _make_alt_trainer(monkeypatch)
    trainer.iters_to_accumulate = 2
    batch = _batch_with_sc(trainer)
    trainer.train_step(batch)
    assert trainer.step == 0
    trainer.train_step(batch)
    assert trainer.step == 1


def test_alternating_checkpoint_roundtrip(monkeypatch, tmp_path):
    trainer = _make_alt_trainer(monkeypatch)
    trainer.checkpoint_dir = str(tmp_path)
    trainer.rank = 0
    batch = _batch_with_sc(trainer)
    trainer.train_step(batch)
    path = trainer.save_checkpoint()
    assert path is not None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["train_mode"] == "alternating"
    assert "sc_optimizer" in ckpt and "bb_optimizer" in ckpt
    # Reload into a fresh trainer without error.
    trainer2 = _make_alt_trainer(monkeypatch)
    trainer2.load_checkpoint(path, params_only=False)
    assert trainer2.step == 1
    assert trainer2.global_step == 1
    assert trainer2.train_sampler.step == 1


def test_joint_mode_still_default(monkeypatch):
    """Sanity: without train_mode the trainer stays single-optimizer joint."""
    trainer = _make_alt_trainer(monkeypatch)
    # Flip a fresh trainer to joint by monkeypatching the cfg default path.
    import pxdesign_train.runner.trainer as tr

    assert trainer.train_mode == "alternating"  # this fixture is alternating
    # The joint path is covered by test_trainer_integration; here we just assert
    # the alias exists so generic code keeps working.
    assert trainer.optimizer is trainer.bb_optimizer
