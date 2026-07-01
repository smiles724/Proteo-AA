"""
Training configs for PXDesign-d.

Numbers come straight from the technical report (Appendix C, p. 24). We extend
PXDesign's released `configs_base.py` rather than redefining the model — the
architecture is identical, only training-side knobs are added.
"""
from copy import deepcopy

from pxdesign.configs.configs_base import configs as base_configs


training_configs = deepcopy(base_configs)

# Override loss-relevant model knobs that the inference build set conservatively.
training_configs["load_strict"] = False  # we add heads not in the inference ckpt

# Two new model flags consumed by ProtenixDesignTrain.
training_configs["enable_distogram_head"] = True
training_configs["enable_diffusion_distogram_head"] = False  # head built but unused for now
training_configs["enable_residue_type_head"] = True

training_configs["residue_type"] = {
    "vocab_size": 20,
    "ignore_index": -100,
    "loss_on_design_only": True,
    "mask_mode": "time_dependent",
    "mask_prob": 1.0,
    "mask_min_prob": 0.0,
    "mask_max_prob": 1.0,
    # V3: feed the discrete masked-diffusion time aa_t into the AA head.
    "use_time_embedding": True,
}

# EDM training noise sampler.
training_configs["training_noise_sampler"] = {
    "p_mean": -1.2,
    "p_std": 1.5,
    "sigma_data": training_configs["sigma_data"],  # 16.0
}

# Training-side hyperparameters from the report.
training_configs["training"] = {
    # The macro-batch is 64 examples; "diffusion batch size 8" is N_sample per
    # example, i.e. how many (rotation, noise) draws we evaluate per item.
    "macro_batch_size": 64,
    "diffusion_batch_size": 8,
    "crop_size": 640,
    "max_steps": 100000,        # ballpark; not stated in report
    "warmup_steps": 2000,       # mirrors Protenix's demo
    "lr": 5e-4,
    "weight_decay": 0.0,
    "ema_decay": 0.999,
    "checkpoint_interval": 400,
    "log_interval": 50,
    "eval_interval": 400,
}

# Loss weights from eq. 4 of the report.
training_configs["loss"] = {
    "weight_mse": 4.0,
    "weight_lddt": 1.0,
    "weight_disto": 0.03,
    "weight_aa": 1.0,
    # V3: MDLM / absorbing-diffusion time weighting (1/t) for the AA CE. When
    # False the AA term is a plain masked-LM mean CE.
    "aa_time_weighting": True,
    "sigma_low_threshold": 4.0,  # σ below this gates LDDT and distogram terms
    "no_bins": training_configs["no_bins"],
    "min_bin": 2.3125,
    "max_bin": 21.6875,
    "lddt_radius": 15.0,
    "align_before_mse": True,
}

# Two-stage curriculum (data source weights — to be consumed by the dataloader).
training_configs["curriculum"] = {
    "stage1": {
        # Pre-training: monomer-only distillation data dominates.
        "weights": {"afdb_monomer": 0.5, "mgnify_monomer": 0.4, "pdb_complex": 0.1},
        "max_steps": 50000,
    },
    "stage2": {
        # Conditioned design: shift toward PDB complexes.
        "weights": {"afdb_monomer": 0.1, "mgnify_monomer": 0.1, "pdb_complex": 0.8},
        "max_steps": 50000,
    },
}
