from __future__ import annotations

import torch

from rotation_ablation.bundle import SidechainInputBundle
from rotation_ablation.experiment import ARMS, RotationAblation
from rotation_ablation.factory import random_modules, synthetic_bundle
from rotation_ablation.geometry import random_rotation


def _runner(seed: int = 7):
    module, feedback, _ = random_modules(
        c_res=16,
        c_atom=32,
        c_time=16,
        n_blocks=2,
        n_heads=4,
        n_cross_blocks=2,
        seed=seed,
    )
    bundle = synthetic_bundle(c_res=16, n_type=20, length=4, seed=seed)
    return RotationAblation(module, feedback, bundle)


def test_global_baseline_sensitive_but_fully_local_path_equivariant():
    runner = _runner()
    generator = torch.Generator().manual_seed(11)
    q = random_rotation(generator=generator)
    result = runner.one_transform(q, torch.zeros(3))

    baseline = result[ARMS[0].name]["C_coordinate_only"]
    full_local = result[ARMS[3].name]["C_coordinate_only"]
    assert baseline["invariant"]["xyz_projected"]["relative_frobenius"] > 1e-2
    assert baseline["prediction_global"]["eq_rmsd"] > 1e-2
    assert full_local["invariant"]["xyz_projected"]["relative_frobenius"] < 1e-6
    assert full_local["invariant"]["atom_feats"]["relative_frobenius"] < 1e-5
    assert full_local["prediction_global"]["eq_rmsd"] < 1e-5


def test_two_by_two_locates_input_and_output_failures():
    runner = _runner()
    generator = torch.Generator().manual_seed(13)
    q = random_rotation(generator=generator)
    result = runner.one_transform(q, torch.zeros(3))

    local_input_only = result[ARMS[1].name]["C_coordinate_only"]
    frame_head_only = result[ARMS[2].name]["C_coordinate_only"]
    assert local_input_only["invariant"]["atom_feats"]["relative_frobenius"] < 1e-5
    assert local_input_only["prediction_global"]["eq_rmsd"] > 1e-2
    assert frame_head_only["invariant"]["atom_feats"]["relative_frobenius"] > 1e-2


def test_translation_and_chi_positive_control_are_reported():
    runner = _runner()
    result = runner.run(n_rotations=2, seed=19, translation_scale=5.0)
    baseline = result["translation_only"][ARMS[0].name]["C_coordinate_only"]
    full_local = result["translation_only"][ARMS[3].name]["C_coordinate_only"]
    assert baseline["invariant"]["xyz_projected"]["relative_frobenius"] > 1e-2
    assert full_local["prediction_global"]["eq_rmsd"] < 1e-5
    chi = result["chi_positive_control"][ARMS[3].name]
    assert chi["available"]
    assert chi["input_change_rmsd"] > 0
    assert chi["xyz_feature_change"]["relative_frobenius"] > 0


def test_bundle_from_captured_global_call_recovers_local_coordinates():
    runner = _runner()
    b = runner.bundle
    args = (
        b.h_res,
        b.aa_logits,
        b.atom_name_ids,
        b.atom_mask,
        b.noisy_global,
        b.time,
    )
    captured = SidechainInputBundle.from_module_call(
        args,
        {
            "ca_coords": b.ca_coords,
            "frame_R": b.frame_R,
            "frame_t": b.frame_t,
        },
        local_coord_input=False,
    )
    assert torch.allclose(captured.noisy_local, b.noisy_local, atol=1e-5)
