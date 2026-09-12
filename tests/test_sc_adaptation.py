"""SC-only objective, source, partition and exact sampling contracts."""
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ml_collections import ConfigDict
from pxdesign_train.sc_adaptation import validate_phase, check_source, paired_quality
from pxdesign_train.runner.sc_stream import SCStream, MicrostepSampler, isolated_evaluation
from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule


def config(phase="sc_adapt", physical=.01):
    return ConfigDict(dict(stage4=dict(phase=phase, train_rounds=0, inference_rounds=0,
        sc_to_aa=False, sc_to_bb=False, backbone_refinement_enabled=False, packing_enabled=True,
        initial_target_policy="joint", weight_aa_pre=0., weight_aa_revision=0., weight_physical=physical,
        native_sc_augmentation=True), training=dict(diffusion_batch_size=1),
        sidechain=dict(edm=False), loss=dict(weight_mse=1., weight_lddt=1., weight_disto=1., weight_aa=1.)))


@pytest.mark.parametrize("phase", ["sc_complex_adapt", "sc_adapt"])
def test_complete_sc_only_phase_objective(phase):
    cfg = validate_phase(config(phase))
    assert all(value == 0 for value in cfg.loss.values())
    assert cfg.sidechain.pack_loss == .01
    assert cfg.sidechain.predicted_frame == (phase == "sc_adapt")
    assert cfg.sidechain.force_gt_type_logits == (phase != "sc_adapt")


@pytest.mark.parametrize("key,value", [("sc_to_aa", True), ("weight_aa_pre", 1.),
    ("weight_physical", float("nan")), ("native_sc_augmentation", False), ("inference_rounds", 1)])
def test_invalid_phase_settings_rejected(key, value):
    cfg = config(); cfg.stage4[key] = value
    with pytest.raises(ValueError): validate_phase(cfg)


def test_warmup_cannot_advertise_physical_objective():
    with pytest.raises(ValueError, match="forbids physical"):
        validate_phase(config("sc_warmup"))


def full_sample():
    return dict(backbone_source="full_sample", cached_backbone_xyz=torch.zeros(8,3),
        backbone_provenance=dict(seed=1, checkpoint_sha256="abc", steps=400,
            sampler="pxdesign_native", target_policy="joint"))


def test_unlabeled_full_sample_contract():
    assert check_source(full_sample(), {}) == "full_sample"
    for key in ("aa_clean", "sc_gt_local", "sc_observed_mask"):
        with pytest.raises(ValueError, match="must not carry"):
            check_source(dict(full_sample(), **{key: torch.zeros(1)}), {})
    with pytest.raises(ValueError): check_source(full_sample(), {"coordinate": torch.zeros(8,3)})
    with pytest.raises(ValueError): check_source({}, {})


def test_pseudo_target_quality_gate_and_missing_frames():
    cfg = validate_phase(config()).stage4
    xyz = torch.tensor([[-1.46,0,0], [0,0,0], [.5,1.446,0], [1.,2.,0.]]).repeat(3,1)
    feat = dict(aa_bb_atom_idx=torch.arange(12).reshape(3,4), sc_frame_valid=torch.tensor([True, True, False]))
    changed = xyz.clone(); changed[4:8] += 10
    eligible, ca, _ = paired_quality(feat, xyz, changed, cfg)
    assert eligible.tolist() == [True, False, False]
    assert ca[1] > 3


class Items:
    def __init__(self, source): self.source = source
    def __len__(self): return 7
    def __getitem__(self, index):
        return dict(source_name=self.source, sample_id=str(index), input_feature_dict={}, label_dict={},
                    draws=(random.random(), float(np.random.rand()), float(torch.rand(()))))


def test_microstep_replay_independent_of_prefetch_and_global_rng():
    dataset = CurriculumMultiDataset([Items("monomer"), Items("pinder")], ["monomer", "pinder"], [[1]*7, [1]*7])
    weights = dict(monomer=.75, pinder=.25)
    schedule = CurriculumSchedule(weights, weights, 0, 0)
    cfg = validate_phase(config()).stage4
    stream = SCStream(dataset, schedule, cfg, seed=11, microsteps=80)
    expected = [stream[i] for i in range(8, 16)]
    for i in range(16, 30): stream[i]  # discarded prefetch
    torch.rand(100); np.random.rand(100); random.random()
    sampler = MicrostepSampler(stream, 8); sampler.set_step(1)
    assert list(sampler)[:8] == list(range(8, 16))
    assert [stream[i] for i in list(sampler)[:8]] == expected


def test_evaluation_preserves_training_rng_even_on_error():
    class Eval:
        configs = SimpleNamespace(seed=7)
        @isolated_evaluation
        def evaluate(self):
            torch.rand(7); random.random(); np.random.rand()
            raise RuntimeError("test")
    torch.manual_seed(11)
    saved = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError): Eval().evaluate()
    assert torch.equal(saved, torch.get_rng_state())


def test_validation_data_cache_misses_do_not_consume_model_rng():
    from pxdesign_train.runner.sc_stream import CoordinatePanel
    class Loader:
        def __len__(self): return 1
        def __iter__(self):
            torch.rand(99); random.random(); np.random.rand()
            yield dict(input_feature_dict={}, label_dict={})
    torch.manual_seed(27)
    before = torch.get_rng_state()
    next(iter(CoordinatePanel(Loader(), "native")))
    assert torch.equal(before, torch.get_rng_state())


def test_pinder_cold_conversion_does_not_change_featurization_rng(monkeypatch):
    import pxdesign_train.runner.pinder_provider as module
    provider = object.__new__(module.PinderPdbProvider)
    provider._pinder_ids = ["1abc__A--B"]; provider._binder_chains = ["B"]
    cold = [True]
    def prepare(index):
        if cold[0]: torch.rand(91)
        cold[0] = False
        return "unused.cif"
    provider._ensure_cif = prepare
    class Featurizer:
        def __init__(self, *args, **kwargs): pass
        def __getitem__(self, index): return torch.rand(3)
    monkeypatch.setattr(module, "CifFileProvider", Featurizer)
    torch.manual_seed(91); first = provider[0]
    torch.manual_seed(91); second = provider[0]
    assert torch.equal(first, second)


def test_native_physical_coefficient_is_added_once_with_gradient():
    from pxdesign_train.runner.trainer import PXDesignTrainer
    parameter = torch.nn.Parameter(torch.tensor(2.))
    mse, physical = parameter.square(), parameter.pow(3)
    output = dict(supervised_sc=True, sc_gt_mse=mse, sc_physical=physical,
        sc_observed_atoms=torch.tensor(1), sc_skipped_noncanonical=torch.tensor(0), sc_invalid_native_frames=torch.tensor(0))
    trainer = object.__new__(PXDesignTrainer)
    trainer._to_device = lambda batch: batch
    trainer.model = lambda **kwargs: output
    trainer.configs = config("sc_complex_adapt")
    trainer.configs.stage4.weight_sc_aux = 1.
    loss = trainer.forward_loss(dict(input_feature_dict={}, label_dict={}))
    assert loss["loss"].item() == pytest.approx(4.08)
    loss["loss"].backward()
    assert parameter.grad.item() == pytest.approx(4.12)


def test_cross_source_pdb_partitions_precede_derivatives(tmp_path):
    import pandas as pd
    from pxdesign_train.runner.sc_partitions import prepare_partitions
    paths = [tmp_path/name for name in ("train.csv", "val.csv", "pinder.parquet")]
    pd.DataFrame(dict(pdb_id=["1abc", "2abc", "3abc"])).to_csv(paths[0], index=False)
    pd.DataFrame(dict(pdb_id=["2abc"])).to_csv(paths[1], index=False)
    pd.DataFrame(dict(pinder_id=["1abc__A--B", "2abc__A--B", "3abc__A--B"],
        source_split=["train", "train", "val"], cluster_id=[1,2,3])).to_parquet(paths[2])
    mono, pinder, audit = prepare_partitions(*paths, tmp_path)
    assert pd.read_csv(mono).pdb_id.tolist() == ["1abc"]
    assert pd.read_parquet(pinder).pinder_id.tolist() == ["1abc__A--B", "3abc__A--B"]
    assert not audit["homology_independence"]


def test_full_cache_preserves_source_mixture_and_rejects_missing_source():
    dataset = CurriculumMultiDataset([Items("monomer"), Items("pinder")], ["monomer", "pinder"], [[1]*7, [1]*7])
    weights = dict(monomer=.75, pinder=.25)
    schedule = CurriculumSchedule(weights, weights, 0, 0)
    cfg = validate_phase(config()).stage4
    cfg.native_fraction=.5; cfg.paired_fraction=.4; cfg.full_sample_fraction=.1
    class Cache:
        by_source = dict(monomer=[0], pinder=[1])
        def __len__(self): return 2
        def __getitem__(self, index):
            return dict(source_name=("monomer","pinder")[index],input_feature_dict={},label_dict={})
    cache=Cache()
    stream=SCStream(dataset,schedule,cfg,seed=7,microsteps=2000,full_samples=cache)
    samples=[stream[index] for index in range(2000)]
    full=[x for x in samples if x["input_feature_dict"]["backbone_source"]=="full_sample"]
    assert .65 < sum(x["source_name"]=="monomer" for x in full)/len(full) < .85
    cache.by_source=dict(pinder=[1])
    with pytest.raises(ValueError,match="lacks source"):
        SCStream(dataset,schedule,cfg,seed=7,microsteps=8,full_samples=cache)
