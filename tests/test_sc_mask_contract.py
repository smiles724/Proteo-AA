"""Independent chemistry, model visibility, observation and frame contracts."""
import numpy as np
import pytest
import torch

from test_design_featurizer import _ser_complex, _sc_targets, _displace_atom
from test_sidechain_module import _toy_batch, _module
from pxdesign_train.sidechain.losses import sidechain_global_frame_aligned_loss


@pytest.mark.parametrize('damage', ['missing_N', 'unresolved_N', 'unresolved_CA', 'unresolved_C',
                                    'nan', 'inf', 'coincident', 'collinear', 'far_placeholder'])
def test_invalid_native_frame_never_supplies_a_local_target(damage):
    aa, feat, binder = _ser_complex(37.)
    if damage == 'missing_N':
        keep = ~((aa.res_id == 1) & (aa.atom_name == 'N'))
        aa = aa[keep];binder = binder[keep]
        feat['distogram_rep_atom_mask'] = torch.from_numpy(aa.distogram_rep_atom_mask.copy())
    elif damage.startswith('unresolved_'):
        _displace_atom(aa, 0, damage.split('_')[1], (0., 0., 0.), keep_resolved=False)
    else:
        ca = aa.coord[(aa.res_id == 1) & (aa.atom_name == 'CA')][0]
        xyz = {'nan': (float('nan'), 0., 0.), 'inf': (float('inf'), 0., 0.),
               'coincident': ca, 'collinear': ca + np.array([-1.2, 0., 0.]),
               'far_placeholder': (999., 999., 999.)}[damage]
        _displace_atom(aa, 0, 'N', xyz)
    out = _sc_targets(aa, feat, binder)
    assert out['sc_chemical_mask'][0, :2].all()  # SER still owns CB and OG.
    assert out['sc_observed_mask'][0, :2].all()  # Their coordinates were observed.
    assert not out['sc_frame_valid'][0]
    assert not out['sc_loss_mask'][0].any()
    assert out['sc_frame_valid'][1] and out['sc_loss_mask'][1, :2].all()
    assert torch.isfinite(out['sc_gt_local']).all()
    torch.testing.assert_close(out['sc_frame_R'][0], torch.eye(3))


def test_missing_sidechain_and_oxygen_preserve_chemical_inventory_and_frame():
    aa, feat, binder = _ser_complex(37., unresolved_slots={(0, 'OG'), (0, 'O')})
    out = _sc_targets(aa, feat, binder)
    assert out['sc_chemical_mask'][0, :2].all()
    assert out['sc_frame_valid'].all()  # O does not define the frame.
    assert not out['sc_bb_observed_mask'][0, 3]
    assert out['sc_loss_mask'][0, 0] and not out['sc_loss_mask'][0, 1]


def test_resolved_coordinate_origin_is_not_a_missing_atom():
    aa, feat, binder = _ser_complex(0.)
    out = _sc_targets(aa, feat, binder)
    assert out['sc_frame_valid'].all()  # Native CA can legitimately be at the origin.
    assert out['sc_loss_mask'][:, :2].all()


def test_bad_frame_and_unobserved_nan_targets_have_zero_loss_gradient():
    pred = torch.ones(3, 2, 3, requires_grad=True)
    target = torch.zeros_like(pred)
    target[2] = float('nan')
    R = torch.eye(3).repeat(3, 1, 1);R[0] = 0.;R[2] = float('nan')
    t = torch.zeros(3, 3)
    loss = sidechain_global_frame_aligned_loss(pred, target, R, t, torch.ones(3, 2, dtype=torch.bool))
    assert loss.item() == pytest.approx(3., abs=2e-6)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0].count_nonzero() == pred.grad[2].count_nonzero() == 0
    assert pred.grad[1].abs().sum() > 0


def test_masked_backbone_context_and_sc_padding_cannot_change_output_or_gradients():
    torch.manual_seed(7)
    _, mask, ids, h, logits, noisy, time = _toy_batch()
    mod = _module().eval()
    bb = torch.randn(1, 3, 4, 3)
    bb_mask = torch.ones(1, 3, 4, dtype=torch.bool);bb_mask[:, :, 3] = False
    ca = bb[:, :, 1].clone()
    reference = mod(h, logits, ids, mask, noisy, time, ca_coords=ca,
                    bb_coords=bb, bb_atom_mask=bb_mask)[0]
    noisy[~mask] = float('nan');bb[:, :, 3] = float('nan')
    result = mod(h, logits, ids, mask, noisy, time, ca_coords=ca,
                 bb_coords=bb, bb_atom_mask=bb_mask)[0]
    torch.testing.assert_close(reference, result)
    result.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None)


def test_empty_model_mask_has_finite_zero_output_and_backward():
    _, mask, ids, h, logits, noisy, time = _toy_batch()
    mod = _module()
    mask.zero_();noisy.fill_(float('nan'))
    result = mod(h, logits, ids, mask, noisy, time, ca_coords=torch.zeros(1, 3, 3),
                 bb_coords=torch.full((1, 3, 4, 3), float('nan')),
                 bb_atom_mask=torch.zeros(1, 3, 4,dtype=torch.bool))[0]
    assert torch.isfinite(result).all() and result.count_nonzero()==0
    result.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in mod.parameters() if p.grad is not None)
