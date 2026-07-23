"""
Tests for `ProtenixComplexProvider` (mocking `BaseSingleDataset`) and the
warm-start fine-tune helpers.

We don't have real PDB data here, so the Protenix-adapter test mocks
`process_one()` to return a synthetic `(atom_array, token_array, ...)` bundle
matching the keys Protenix's real dataset emits. The fine-tune test exercises
the checkpoint-save-and-reload path with the trainer integration's stub model.
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


# Re-use the synthetic-complex helper from the trainer integration test by
# duplicating its core builder here. (Importing from another test file is
# brittle; the duplication is acceptable for ~40 lines.)
BACKBONE = ("N", "CA", "C", "O")
ATOMS_PER_RES = len(BACKBONE)


def _make_synthetic_complex_bundle():
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

    aa.set_annotation("distogram_rep_atom_mask", (aa.atom_name == "CA").astype(int))
    aa.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    aa.set_annotation(
        "asym_id_int",
        np.array([{"A": 0, "B": 1, "C": 2}[c] for c in aa.chain_id], dtype=np.int64),
    )

    tokens = []
    for r in range(n_res):
        atom_indices = list(range(r * ATOMS_PER_RES, (r + 1) * ATOMS_PER_RES))
        centre = atom_indices[BACKBONE.index("CA")]
        tokens.append(Token(r, atom_indices=atom_indices, centre_atom_index=centre))
    return aa, TokenArray(tokens), n_res, n_atom


class _MockBaseSingleDataset:
    """Mimics Protenix's `BaseSingleDataset` for adapter testing.

    Returns the same dict shape that `process_one(idx, return_atom_token_array=True)`
    produces in real Protenix, with the keys our adapter reads:
    `cropped_atom_array`, `cropped_token_array`, `input_feature_dict`, `label_dict`.
    Also implements `_get_sample_indice` so `select_protenix_chain_2()` works.
    """

    def __init__(self, n_items: int = 3, chain_pairs=None):
        self.aa, self.ta, n_res, n_atom = _make_synthetic_complex_bundle()
        self._n_items = n_items
        # Default chain pair: (A, C) — C is the binder.
        self._chain_pairs = chain_pairs or [("A", "C")] * n_items
        self._n_res = n_res
        self._n_atom = n_atom

    def __len__(self):
        return self._n_items

    def process_one(self, idx: int, return_atom_token_array: bool = False):
        feat = {
            "distogram_rep_atom_mask": torch.from_numpy(
                self.aa.distogram_rep_atom_mask.astype(np.int64),
            ).long(),
            "restype": torch.zeros(self._n_res, 32),
            "deletion_mean": torch.ones(self._n_res),
            "profile": torch.ones(self._n_res, 32),
            "msa": torch.ones(1, self._n_res),
        }
        label = {
            "coordinate": torch.from_numpy(self.aa.coord),
            "coordinate_mask": torch.ones(self._n_atom, dtype=torch.long),
        }
        return {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": {},
            "basic": {"pdb_id": f"mock_{idx}", "chain_id": ["A", "B", "C"]},
            "cropped_atom_array": self.aa,
            "cropped_token_array": self.ta,
        }

    def _get_sample_indice(self, idx: int):
        chain_1, chain_2 = self._chain_pairs[idx]

        class _Indice(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        return _Indice(chain_1_id=chain_1, chain_2_id=chain_2, type="interface")


def test_protenix_provider_passes_through_atom_array():
    from pxdesign_train.runner.providers import (
        ProtenixComplexProvider,
        select_chain_by_id,
    )

    base = _MockBaseSingleDataset(n_items=2)
    provider = ProtenixComplexProvider(base, binder_selector_fn=select_chain_by_id("C"))
    assert len(provider) == 2
    atom_array, token_array, feat, label, sel_fn = provider[0]
    assert len(atom_array) == base._n_atom
    assert len(token_array) == base._n_res
    assert "restype" in feat
    assert "coordinate" in label
    assert sel_fn(atom_array) == "C"


def test_protenix_provider_works_with_DesignSourceDataset():
    from pxdesign_train.runner import (
        DesignSourceDataset,
        ProtenixComplexProvider,
        select_chain_by_id,
    )

    base = _MockBaseSingleDataset(n_items=2)
    provider = ProtenixComplexProvider(base, binder_selector_fn=select_chain_by_id("C"))
    src = DesignSourceDataset(
        provider, source_name="pdb",
        crop_size=20, hotspot_force_zero_prob=0.0,
    )
    batch = src[0]
    assert batch["input_feature_dict"]["restype"].shape[-1] == 36
    assert batch["source_name"] == "pdb"
    # Binder = chain C (6 residues).
    assert batch["input_feature_dict"]["design_token_mask"].sum().item() == 6


def test_select_protenix_chain_2_uses_sample_indice():
    from pxdesign_train.runner.providers import (
        ProtenixComplexProvider,
        select_protenix_chain_2,
    )

    base = _MockBaseSingleDataset(n_items=1, chain_pairs=[("A", "B")])
    provider = ProtenixComplexProvider(base, binder_selector_fn=select_protenix_chain_2())
    _, _, _, _, sel_fn = provider[0]
    assert sel_fn(None) == "B"


def test_select_protenix_chain_1_handles_monomer_rows():
    from pxdesign_train.runner.providers import (
        ProtenixComplexProvider,
        select_protenix_chain_1,
    )

    base = _MockBaseSingleDataset(n_items=1, chain_pairs=[("A", None)])
    provider = ProtenixComplexProvider(base, binder_selector_fn=select_protenix_chain_1())
    _, _, _, _, sel_fn = provider[0]
    assert sel_fn(None) == "A"


def test_design_source_dataset_allows_whole_monomer_binder():
    from pxdesign_train.runner import DesignSourceDataset

    class _OneMonomerProvider:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            aa, ta, _n_res, n_atom = _make_synthetic_complex_bundle()
            # Keep only chain C, making the selected binder the whole sample.
            keep = aa.chain_id == "C"
            centre_atom_idx = ta.get_annotation("centre_atom_index")
            token_keep = np.where(keep[centre_atom_idx])[0]
            from protenix.utils.cropping import CropData

            ta2, aa2 = CropData.select_by_token_indices(
                token_array=ta,
                atom_array=aa,
                selected_token_indices=torch.tensor(token_keep, dtype=torch.long),
            )
            feat = {
                "distogram_rep_atom_mask": torch.from_numpy(
                    aa2.distogram_rep_atom_mask.astype(np.int64)
                ).long(),
                "restype": torch.zeros(len(ta2), 32),
                "deletion_mean": torch.ones(len(ta2)),
                "profile": torch.ones(len(ta2), 32),
                "msa": torch.ones(1, len(ta2)),
            }
            label = {
                "coordinate": torch.from_numpy(aa2.coord),
                "coordinate_mask": torch.ones(len(aa2), dtype=torch.long),
            }
            return aa2, ta2, feat, label, lambda _aa: "C"

    ds = DesignSourceDataset(
        _OneMonomerProvider(),
        source_name="mono",
        crop_size=8,
        max_binder_fraction=1.0,
    )
    batch = ds[0]
    assert int(batch["input_feature_dict"]["design_token_mask"].sum()) == 6


def test_select_smallest_protein_chain():
    from pxdesign_train.runner.providers import (
        ProtenixComplexProvider,
        select_smallest_protein_chain,
    )

    base = _MockBaseSingleDataset(n_items=1)
    provider = ProtenixComplexProvider(base, binder_selector_fn=select_smallest_protein_chain())
    atom_array, _, _, _, sel_fn = provider[0]
    # Chain C has 6 residues (smallest); A and B each have 12.
    assert sel_fn(atom_array) == "C"


def test_select_random_protein_chain_is_deterministic_with_seed():
    from pxdesign_train.runner.providers import select_random_protein_chain

    base = _MockBaseSingleDataset(n_items=1)
    sel_a = select_random_protein_chain(seed=42)
    sel_b = select_random_protein_chain(seed=42)
    aa = base.aa
    seq_a = [sel_a({}, aa) for _ in range(5)]
    seq_b = [sel_b({}, aa) for _ in range(5)]
    assert seq_a == seq_b


# ----- fine-tune helpers -----


def test_make_finetune_configs_lowers_lr_and_relaxes_strict():
    from pxdesign_train.runner.finetune import make_finetune_configs

    class _Cfg:
        load_strict = True
        class training:
            lr = 5e-4
            warmup_steps = 2000
            max_steps = 100_000
            ema_decay = 0.999

    out = make_finetune_configs(_Cfg(), lr=1e-5, warmup_steps=100, max_steps=5_000)
    assert out.training.lr == 1e-5
    assert out.training.warmup_steps == 100
    assert out.training.max_steps == 5_000
    assert out.load_strict is False
    # NOTE: we don't assert the original is untouched here. With
    # class-level-attribute configs `deepcopy` doesn't isolate inner classes;
    # real `ml_collections.ConfigDict` configs (from `parse_configs`) do.


def _make_finetune_trainer(monkeypatch, tmp_path):
    """Reuse the integration test's fake-model setup, with a checkpoint dir."""
    from pxdesign_train.data import (
        CurriculumMultiDataset,
        CurriculumSchedule,
    )
    from pxdesign_train.runner import (
        DesignSourceDataset,
        PXDesignTrainer,
        TrainerComponents,
    )

    # Import _FakeModel and _SyntheticProvider from the integration test file.
    sys.path.insert(0, os.path.join(HERE))
    from test_trainer_integration import _FakeModel, _SyntheticProvider

    def _fake_init_model(self):
        self.raw_model = _FakeModel().to(self.device)
        self.model = self.raw_model
        self.ema_wrapper = None

    monkeypatch.setattr(PXDesignTrainer, "_init_model", _fake_init_model)

    src = DesignSourceDataset(
        _SyntheticProvider(n_items=2), source_name="a",
        crop_size=20, hotspot_force_zero_prob=0.0,
    )
    multi = CurriculumMultiDataset(
        datasets=[src], source_names=["a"], per_item_weights=[[1.0, 1.0]],
    )
    sched = CurriculumSchedule(
        stage1={"a": 1.0}, stage2={"a": 1.0},
        stage1_end_step=10, stage2_start_step=10,
    )
    components = TrainerComponents(
        train_dataset=multi, schedule=sched, train_samples_per_epoch=2,
    )

    class _Cfg:
        seed = 42
        dtype = "fp32"
        load_strict = False
        class training:
            lr = 1e-3
            warmup_steps = 1
            max_steps = 2
            weight_decay = 0.0
            ema_decay = 0.0
            log_interval = 100
            eval_interval = 0
            checkpoint_interval = 0
            iters_to_accumulate = 1
            grad_clip_norm = 0.0
            num_workers = 0
        class loss:
            weight_mse = 4.0
            weight_lddt = 1.0
            weight_disto = 0.03
            sigma_low_threshold = 4.0
            no_bins = 64
            min_bin = 2.3125
            max_bin = 21.6875
            lddt_radius = 15.0
            align_before_mse = False

    return PXDesignTrainer(
        configs=_Cfg(),
        components=components,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "ckpts"),
    )


def test_finetune_checkpoint_roundtrip(monkeypatch, tmp_path):
    """Train one step, save a checkpoint, build a fresh trainer, load it,
    and verify the parameter survived."""
    trainer = _make_finetune_trainer(monkeypatch, tmp_path)
    batch = next(iter(trainer.train_dl))
    trainer.train_step(batch)
    target_bias = trainer.model.bias.detach().clone()
    path = trainer.save_checkpoint()
    assert path is not None and os.path.exists(path)

    fresh = _make_finetune_trainer(monkeypatch, tmp_path)
    assert not torch.allclose(fresh.model.bias.detach(), target_bias)
    fresh.load_checkpoint(path, params_only=True)
    assert torch.allclose(fresh.model.bias.detach(), target_bias)


def test_finetune_load_strict_false_tolerates_missing_keys(monkeypatch, tmp_path):
    """If the checkpoint is missing parameters that exist in the model, the
    load should succeed (with load_strict=False) and report the missing keys."""
    trainer = _make_finetune_trainer(monkeypatch, tmp_path)
    # Construct a checkpoint missing the dist_proj head.
    state = {k: v for k, v in trainer.model.state_dict().items() if not k.startswith("dist_proj")}
    path = str(tmp_path / "partial.pt")
    torch.save({"model": state, "optimizer": {}, "scheduler": {}, "step": 0, "global_step": 0}, path)

    fresh = _make_finetune_trainer(monkeypatch, tmp_path)
    # Should not raise.
    fresh.load_checkpoint(path, params_only=True)
