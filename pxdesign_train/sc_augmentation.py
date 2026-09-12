"""Rigid augmentation of native packing inputs in one shared complex frame."""
import torch


def random_rigid_transform(xyz, observed, translation_scale=1.):
    """Uniform SO(3), observed-atom centering and isotropic translation.

    Uses torch's checkpointed RNG. Return affine x' = R x + t, with column
    vectors; all structural coordinates in an example share this transform.
    """
    xyz = xyz.detach().float()
    observed = observed.bool() & torch.isfinite(xyz).all(-1)
    center = torch.where(observed[..., None], xyz, 0.).sum(-2) / observed.sum().clamp_min(1)
    q = torch.randn(4, device=xyz.device)
    w, x, y, z = (q / q.norm().clamp_min(1e-12)).unbind()
    R = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))).reshape(3,3)
    t = translation_scale * torch.randn(3, device=xyz.device) - R @ center
    return R, t


def transform_native_sc_inputs(feat, labels, rotation, translation):
    """Copy and transform global coordinates; preserve local geometry and masks.

    ref_pos is the per-reference-space conformer, not a complex-frame vector;
    like ordinary PXDesign augmentation it stays unchanged. sc_gt_local is
    already local and also stays unchanged. No transform touches caller tensors.
    """
    R, t = rotation.float(), translation.float()
    def point(xyz, mask):
        xyz = xyz.float()
        safe = torch.where(mask.bool()[..., None], xyz, 0.)
        return torch.where(mask.bool()[..., None], safe @ R.T + t, 0.)
    out, target = dict(feat), dict(labels)
    observed = labels['coordinate_mask'].bool() & torch.isfinite(labels['coordinate']).all(-1)
    target['coordinate'] = point(labels['coordinate'], observed)
    target['coordinate_mask'] = observed.to(labels['coordinate_mask'].dtype)
    valid = feat['sc_frame_valid'].bool()
    out['sc_frame_R'] = torch.where(valid[..., None, None], R @ feat['sc_frame_R'].float(),
                                    torch.eye(3,device=R.device))
    out['sc_frame_t'] = point(feat['sc_frame_t'], valid)
    out['sc_bb_coords'] = point(feat['sc_bb_coords'], feat['sc_bb_observed_mask'])
    if 'fixed_atom_xyz' in feat:
        out['fixed_atom_xyz'] = point(feat['fixed_atom_xyz'], feat['fixed_atom_mask'])
    out['_native_rigid_transform'] = dict(rotation=R, translation=t)
    return out, target


def augment_native_sc_inputs(feat, labels):
    with torch.autocast(device_type=labels['coordinate'].device.type, enabled=False):
        R, t = random_rigid_transform(labels['coordinate'], labels['coordinate_mask'])
        return transform_native_sc_inputs(feat, labels, R, t)
