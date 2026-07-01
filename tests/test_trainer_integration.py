"""
Integration test for `PXDesignTrainer`.

This test exercises the entire wiring — provider → DesignSourceDataset →
CurriculumMultiDataset → CurriculumSampler → DataLoader → trainer.train_step
— while *avoiding* the real `ProtenixDesign` forward (which requires ~139M
params and CUDA-only kernels we can't compile in this env).

We monkey-patch `PXDesignTrainer._init_model` to install a small fake denoise
network whose forward returns the right shapes from `x_noisy`. This lets us
verify:
  - the dataset chain produces a single batch with the keys the trainer
    expects (`input_feature_dict`, `label_dict`, `binder_token_mask`)
  - the trainer adds a batch dim, runs forward through the fake model, runs
    `PXDesignLoss`, and backprops
  - `train_step` increments `self.step` after `iters_to_accumulate` updates
  - the curriculum sampler's `set_step()` is called as the trainer advances
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


BACKBONE = ("N", "CA", "C", "O")
ATOMS_PER_RES = len(BACKBONE)


def _make_complex(binder_chain="C"):
    """Build a 3-chain backbone-only protein, same pattern as the cropper test."""
    biotite = pytest.importorskip("biotite.structure")
    pytest.importorskip("protenix")
    from protenix.data.tokenizer import Token, TokenArray

    n_a, n_b, n_c = 12, 12, 6
    n_res = n_a + n_b + n_c
    n_atom = n_res * ATOMS_PER_RES

    aa = biotite.AtomArray(length=n_atom)
    aa.coord = np.zeros((n_atom, 3), dtype=np.float32)

    def fill(start, chain, n, y):
        for r in range(n):
            for ai, name in enumerate(BACKBONE):
                i = start + r * ATOMS_PER_RES + ai
                aa.chain_id[i] = chain
                aa.res_id[i] = r + 1
                aa.res_name[i] = "GLY"
                aa.atom_name[i] = name
                aa.element[i] = "N" if name == "N" else ("O" if name == "O" else "C")
                cax = r * 3.8
                if name == "N":   aa.coord[i] = (cax - 1.0, y, 0.0)
                elif name == "CA": aa.coord[i] = (cax, y, 0.0)
                elif name == "C":  aa.coord[i] = (cax + 1.0, y, 0.0)
                else:              aa.coord[i] = (cax + 1.2, y + 1.0, 0.0)

    fill(0, "A", n_a, 0.0)
    fill(n_a * ATOMS_PER_RES, "B", n_b, 30.0)
    fill((n_a + n_b) * ATOMS_PER_RES, "C", n_c, 6.0)

    is_ca = aa.atom_name == "CA"
    aa.set_annotation("distogram_rep_atom_mask", is_ca.astype(int))
    aa.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    aa.set_annotation(
        "asym_id_int",
        np.array({"A": 0, "B": 1, "C": 2}[c] for c in aa.chain_id).astype(np.int64)
        if False else np.array([{"A": 0, "B": 1, "C": 2}[c] for c in aa.chain_id], dtype=np.int64),
    )

    tokens = []
    for r in range(n_res):
        atom_indices = list(range(r * ATOMS_PER_RES, (r + 1) * ATOMS_PER_RES))
        centre = atom_indices[BACKBONE.index("CA")]
        tokens.append(Token(r, atom_indices=atom_indices, centre_atom_index=centre))
    token_array = TokenArray(tokens)

    feat = {
        "distogram_rep_atom_mask": torch.from_numpy(is_ca.astype(np.int64)).long(),
        "restype": torch.zeros(n_res, 32),
        "deletion_mean": torch.ones(n_res),
        "profile": torch.ones(n_res, 32),
        "msa": torch.ones(1, n_res),
    }
    label = {
        "coordinate": torch.from_numpy(aa.coord),
        "coordinate_mask": torch.ones(n_atom, dtype=torch.long),
    }
    selector = lambda _aa: binder_chain
    return aa, token_array, feat, label, selector


class _SyntheticProvider:
    """A tiny provider that returns N copies of a synthetic complex.

    Real providers parse mmCIF; here we just want shape-checked data flow.
    """

    def __init__(self, n_items: int, binder_chain: str = "C"):
        self._items = [_make_complex(binder_chain=binder_chain) for _ in range(n_items)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class _FakeModel(nn.Module):
    """Stand-in for ProtenixDesignTrain.

    Returns the shapes PXDesignLoss + the trainer expect, with at least one
    learnable parameter so backward propagates non-trivial gradients.
    """

    def __init__(self, n_bins: int = 64, c_z: int = 128):
        super().__init__()
        # A scalar bias on the denoised coords so backward updates something.
        self.bias = nn.Parameter(torch.zeros(1))
        # A small head to produce distogram logits from a per-token learnable.
        self.dist_proj = nn.Linear(1, n_bins)
        self.aa_proj = nn.Linear(1, 20)
        self.n_bins = n_bins
        # Match the sigma_data the real model uses.
        self.training_noise_sampler_sigma = 16.0

    def forward(self, *, input_feature_dict, label_dict, mode="train"):
        assert mode == "train"
        gt = label_dict["coordinate"]  # [N_atom, 3] (unbatched)
        N_atom = gt.shape[-2]
        N_sample = 4
        x_gt_aug = gt.unsqueeze(-3).expand(*gt.shape[:-2], N_sample, N_atom, 3).contiguous()
        x_denoised = x_gt_aug + 0.01 * torch.randn_like(x_gt_aug) + self.bias
        sigma = torch.full((*gt.shape[:-2], N_sample), 1.0)
        N_token = int(input_feature_dict["restype"].shape[-2])  # works batched or unbatched
        tok_feat = torch.zeros(*gt.shape[:-2], N_token, 1)
        logits = self.dist_proj(tok_feat)
        logits = logits.unsqueeze(-2).expand(*gt.shape[:-2], N_token, N_token, self.n_bins).contiguous()
        aa_logits = self.aa_proj(tok_feat)
        return {
            "x_gt_aug": x_gt_aug,
            "x_denoised": x_denoised,
            "sigma": sigma,
            "distogram_logits": logits,
            "aa_logits": aa_logits,
            "h_res_candidate": tok_feat,
        }


def _make_trainer(monkeypatch, n_items=3):
    from pxdesign_train.data import (
        CurriculumMultiDataset,
        CurriculumSchedule,
    )
    from pxdesign_train.runner import (
        DesignSourceDataset,
        PXDesignTrainer,
        TrainerComponents,
    )

    # Stub model init to install our fake model — avoids the 139M-param real one.
    def _fake_init_model(self):
        self.raw_model = _FakeModel().to(self.device)
        self.model = self.raw_model
        self.ema_wrapper = None
        self._log(f"Using fake model ({sum(p.numel() for p in self.model.parameters())} params)")

    monkeypatch.setattr(PXDesignTrainer, "_init_model", _fake_init_model)

    # Tiny crop so even our 30-residue synthetic complex fits comfortably.
    src_a = DesignSourceDataset(
        _SyntheticProvider(n_items=n_items), source_name="a",
        crop_size=20, hotspot_force_zero_prob=0.0,
    )
    src_b = DesignSourceDataset(
        _SyntheticProvider(n_items=n_items), source_name="b",
        crop_size=20, hotspot_force_zero_prob=0.0,
    )

    multi = CurriculumMultiDataset(
        datasets=[src_a, src_b],
        source_names=["a", "b"],
        per_item_weights=[[1.0] * n_items, [1.0] * n_items],
    )
    sched = CurriculumSchedule(
        stage1={"a": 0.8, "b": 0.2},
        stage2={"a": 0.2, "b": 0.8},
        stage1_end_step=10,
        stage2_start_step=20,
    )
    components = TrainerComponents(
        train_dataset=multi,
        schedule=sched,
        train_samples_per_epoch=5,
    )

    # Minimal configs object. We only need the attrs the trainer reads.
    class _Cfg:
        seed = 42
        dtype = "fp32"
        load_strict = False
        class training:
            lr = 1e-3
            warmup_steps = 2
            max_steps = 3
            weight_decay = 0.0
            ema_decay = 0.0
            log_interval = 10
            eval_interval = 0
            checkpoint_interval = 0
            iters_to_accumulate = 1
            grad_clip_norm = 0.0
            num_workers = 0
        class loss:
            weight_mse = 4.0
            weight_lddt = 1.0
            weight_disto = 0.03
            weight_aa = 0.0
            sigma_low_threshold = 4.0
            no_bins = 64
            min_bin = 2.3125
            max_bin = 21.6875
            lddt_radius = 15.0
            align_before_mse = False
        class residue_type:
            ignore_index = -100

    return PXDesignTrainer(
        configs=_Cfg(),
        components=components,
        device=torch.device("cpu"),
    )


def test_trainer_single_step_runs(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    batch = next(iter(trainer.train_dl))
    # Verify the batch shape the dataset produces.
    feat = batch["input_feature_dict"]
    assert "restype" in feat and feat["restype"].shape[-1] == 36
    assert "conditional_templ" in feat and "conditional_templ_mask" in feat
    assert "hotspot" in feat
    assert "plddt" in feat
    assert "design_token_mask" in feat
    assert "aa_clean" in feat
    assert "aa_loss_mask" in feat
    assert "aa_corruption_mask" in feat
    assert "aa_t" in feat
    assert batch["source_name"] in ("a", "b")
    # The cropped complex should be <= crop_size in tokens.
    assert feat["restype"].shape[0] <= 20

    loss_out = trainer.train_step(batch)
    assert torch.isfinite(loss_out["loss"]), loss_out
    assert loss_out["loss"].item() >= 0


def test_trainer_gradients_flow(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    # Snapshot a parameter pre-step.
    pre = trainer.model.bias.detach().clone()
    batch = next(iter(trainer.train_dl))
    trainer.train_step(batch)
    # After one step (warmup_step=2, lr non-zero from step 1), bias should
    # have moved away from zero.
    post = trainer.model.bias.detach().clone()
    assert not torch.allclose(pre, post), "expected bias to update after train_step"


def test_trainer_step_counter_and_sampler_step(monkeypatch):
    """After iters_to_accumulate update steps the sampler should see the
    advanced step."""
    trainer = _make_trainer(monkeypatch)
    assert trainer.step == 0
    assert trainer.train_sampler.step == 0
    batch = next(iter(trainer.train_dl))
    trainer.train_step(batch)
    assert trainer.step == 1
    assert trainer.train_sampler.step == 1
    trainer.train_step(batch)
    assert trainer.step == 2
    assert trainer.train_sampler.step == 2


def test_trainer_run_loop_terminates(monkeypatch):
    """`run(max_steps=3)` should stop at step 3, even if epoch_size > 3."""
    trainer = _make_trainer(monkeypatch)
    trainer.run(max_steps=3)
    assert trainer.step == 3


def test_trainer_grad_accum_does_not_advance_step(monkeypatch):
    """With iters_to_accumulate=4, four micro-batches make one update."""
    trainer = _make_trainer(monkeypatch)
    trainer.iters_to_accumulate = 4
    batch = next(iter(trainer.train_dl))
    for _ in range(3):
        trainer.train_step(batch)
        assert trainer.step == 0  # no optimizer step yet
    trainer.train_step(batch)
    assert trainer.step == 1     # update on the 4th micro-batch


def test_trainer_masked_aa_head_gets_gradients(monkeypatch):
    trainer = _make_trainer(monkeypatch)
    trainer.loss_fn.weight_aa = 1.0
    batch = next(iter(trainer.train_dl))
    pre = trainer.model.aa_proj.bias.detach().clone()
    loss_out = trainer.train_step(batch)
    post = trainer.model.aa_proj.bias.detach().clone()

    assert "aa_ce" in loss_out and loss_out["aa_ce"].item() > 0
    assert "aa_acc" in loss_out
    assert loss_out["aa_mask_frac"].item() > 0
    assert not torch.allclose(pre, post), "expected AA head to update"
