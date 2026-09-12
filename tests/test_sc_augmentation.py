"""Coordinate/frame consistency and incomplete backbone context regressions."""
import torch
from test_design_featurizer import _ser_complex, _sc_targets
from pxdesign_train.sc_augmentation import augment_native_sc_inputs, transform_native_sc_inputs
from pxdesign_train.sidechain.losses import sidechain_global_frame_aligned_loss
from pxdesign_train.sidechain.physical import build_sidechain_context


def example():
    aa, feat, binder = _ser_complex(37., unresolved_slots={(0, 'O')})
    feat.update(_sc_targets(aa, feat, binder))
    xyz = torch.from_numpy(aa.coord.copy()).float()
    mask = torch.from_numpy(aa.is_resolved.copy()).bool()
    feat.update(fixed_atom_xyz=xyz.clone(), fixed_atom_mask=mask,
                ref_pos=torch.randn_like(xyz))
    return feat, dict(coordinate=xyz, coordinate_mask=mask)


def test_rigid_transform_targets_frames_context_and_masks_agree():
    feat, labels = example()
    old_xyz = labels['coordinate'].clone()
    torch.manual_seed(37)
    rng = torch.get_rng_state()
    aug, target = augment_native_sc_inputs(feat, labels)
    R, t = aug['_native_rigid_transform'].values()
    torch.testing.assert_close(R.T @ R, torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.det(R), torch.tensor(1.), atol=1e-6, rtol=1e-6)
    mask = labels['coordinate_mask']
    torch.testing.assert_close(target['coordinate'][mask], old_xyz[mask] @ R.T + t)
    torch.testing.assert_close(aug['fixed_atom_xyz'], target['coordinate'])
    torch.testing.assert_close(aug['sc_frame_R'], R @ feat['sc_frame_R'])
    torch.testing.assert_close(aug['sc_frame_t'], feat['sc_frame_t'] @ R.T + t)
    bbmask = feat['sc_bb_observed_mask']
    torch.testing.assert_close(aug['sc_bb_coords'][bbmask], feat['sc_bb_coords'][bbmask] @ R.T + t)
    assert aug['sc_bb_coords'][~bbmask].count_nonzero() == 0
    assert aug['sc_frame_valid'][0] and not bbmask[0, 3]
    local = feat['sc_gt_local']
    global_target = local @ feat['sc_frame_R'].transpose(-1, -2) + feat['sc_frame_t'][:, None]
    augmented_global = local @ aug['sc_frame_R'].transpose(-1, -2) + aug['sc_frame_t'][:, None]
    torch.testing.assert_close(augmented_global, global_target @ R.T + t, atol=1e-5, rtol=1e-5)
    pred = global_target + torch.randn_like(global_target)
    loss = sidechain_global_frame_aligned_loss(pred, local, feat['sc_frame_R'], feat['sc_frame_t'], feat['sc_loss_mask'])
    rotated_loss = sidechain_global_frame_aligned_loss(pred @ R.T + t, local, aug['sc_frame_R'], aug['sc_frame_t'], aug['sc_loss_mask'])
    torch.testing.assert_close(loss, rotated_loss, atol=1e-5, rtol=1e-5)
    for key in ('sc_gt_local', 'ref_pos', 'sc_chemical_mask', 'sc_loss_mask', 'sc_frame_valid'):
        torch.testing.assert_close(aug[key], feat[key])
    torch.testing.assert_close(old_xyz, labels['coordinate'])
    torch.set_rng_state(rng)
    again, repeated = augment_native_sc_inputs(feat, labels)
    torch.testing.assert_close(repeated['coordinate'], target['coordinate'], atol=0, rtol=0)


def test_unobserved_nan_coordinates_stay_inactive_under_augmentation():
    feat, labels = example()
    labels['coordinate'][~labels['coordinate_mask']] = float('nan')
    feat['sc_bb_coords'][~feat['sc_bb_observed_mask']] = float('nan')
    aug, target = augment_native_sc_inputs(feat, labels)
    assert torch.isfinite(target['coordinate']).all()
    assert torch.isfinite(aug['sc_bb_coords']).all()
    assert target['coordinate'][~labels['coordinate_mask']].count_nonzero() == 0


def test_physical_context_excludes_unobserved_backbone_atoms_and_centers():
    xyz = torch.randn(1, 8, 3)
    centers = torch.tensor([[1, 5]])
    bbidx = torch.tensor([[[0, 1, 2, 3], [-1, -1, -1, -1]]])
    tokens = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    present = torch.ones(1, 8, dtype=torch.bool)
    present[:, [3, 5]] = False
    xyz[:, [3, 5]] = float('nan')
    ca, context, atoms = build_sidechain_context(xyz=xyz, center_idx=centers,
        bb_atom_idx=bbidx, atom_to_token=tokens, atom_present=present,
        radius=100., max_atoms=8)
    assert not context.any()
    assert torch.isfinite(ca).all()
    assert atoms[1].sum() == 6
    assert torch.isfinite(atoms[0]).all()
