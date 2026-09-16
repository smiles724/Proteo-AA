"""APM's packing-only objective, assembled from APM's own functions.

`flow_module.model_step` under `train_packing_only` sums exactly two terms,
both at weight 1.0 (`torsions_loss_weight` is applied in a different function
and never reaches this branch):

    supervised_chi_loss(chi_weight=1, angle_norm_weight=0.02)   # openfold
    cal_sidechain_fape_loss(...)                                # flow_module:461

The first is imported straight from openfold. The second is a method bound to
their LightningModule, so it is transcribed here -- but only the glue; every
tensor operation is still the openfold function APM calls, and the constants
are the same `openfold.np.residue_constants` tables APM loads.

One transcribed quirk, deliberately preserved. APM's dataset sets

    torsions_1    = torsion_angles[:, -4:]        # chi1..chi4
    bb_torsions_1 = torsions_1[:, :3]             # datasets.py:247

so `bb_torsions_1` holds chi1..chi3, not omega/phi/psi, and
`cal_sidechain_fape_loss` concatenates it into the first three slots that
`torsion_angles_to_frames` reads as omega/phi/psi. Those three slots drive only
rigid groups 1-3, and of the atom14 slots only O (group 3, psi) lives there --
so the effect is a wrong backbone-O frame contributing a roughly constant term
to FAPE, not a corrupted side chain. Reproducing it is the point: the released
checkpoint was trained against this objective, and "fixing" it here would mean
the new runs optimise something the reference never did.
"""
from __future__ import annotations

import torch

from openfold.np.residue_constants import (restype_atom14_mask,
                                           restype_atom14_rigid_group_positions,
                                           restype_atom14_to_rigid_group,
                                           restype_rigid_group_default_frame)
from openfold.utils.feats import (frames_and_literature_positions_to_atom14_pos,
                                  torsion_angles_to_frames)
from openfold.utils.loss import (compute_renamed_ground_truth, sidechain_loss,
                                 supervised_chi_loss)
from openfold.utils.rigid_utils import Rigid, Rotation

_GT_KEYS = ("atom14_gt_positions", "atom14_alt_gt_positions", "atom14_gt_exists",
            "atom14_atom_is_ambiguous", "atom14_alt_gt_exists")


class RigidGroupTables:
    """openfold's rigid-group constants, materialised once per device/dtype.

    APM's `default_tempalte` caches on first call and then ignores its dtype
    and device arguments; building explicitly avoids inheriting that.
    """

    def __init__(self, dtype, device):
        k = dict(device=device, requires_grad=False)
        self.default_frames = torch.tensor(restype_rigid_group_default_frame, dtype=dtype, **k)
        self.group_idx = torch.tensor(restype_atom14_to_rigid_group, **k)
        self.atom_mask = torch.tensor(restype_atom14_mask, dtype=torch.long, **k)
        self.lit_positions = torch.tensor(restype_atom14_rigid_group_positions, dtype=dtype, **k)


def predicted_atom14(pred_sin_cos, batch, tables):
    """Predicted chi (sin, cos) -> atom14 coordinates, via openfold's rigids."""
    backb_to_global = Rigid(Rotation(rot_mats=batch["rotmats_1"], quats=None),
                            batch["trans_1"])
    bb = torch.stack((batch["bb_torsions_1"].sin(), batch["bb_torsions_1"].cos()), dim=-1)
    alpha = torch.cat([bb.to(pred_sin_cos), pred_sin_cos], dim=2)
    frames = torsion_angles_to_frames(r=backb_to_global, alpha=alpha,
                                      aatype=batch["aatypes_1"],
                                      rrgdf=tables.default_frames)
    pos = frames_and_literature_positions_to_atom14_pos(
        frames, batch["aatypes_1"], tables.default_frames, tables.group_idx,
        tables.atom_mask, tables.lit_positions)
    return frames, pos


def sidechain_fape(pred_sin_cos, batch, tables):
    """flow_module.cal_sidechain_fape_loss, same calls, same order."""
    frames, pos = predicted_atom14(pred_sin_cos, batch, tables)
    renamed = compute_renamed_ground_truth(
        batch={k: batch[k] for k in _GT_KEYS}, atom14_pred_positions=pos)
    return sidechain_loss(
        sidechain_frames=frames.to_tensor_4x4()[None],
        sidechain_atom_pos=pos[None],
        rigidgroups_gt_frames=batch["rigidgroups_gt_frames"],
        rigidgroups_alt_gt_frames=batch["rigidgroups_alt_gt_frames"],
        rigidgroups_gt_exists=batch["rigidgroups_gt_exists"],
        renamed_atom14_gt_positions=renamed["renamed_atom14_gt_positions"],
        renamed_atom14_gt_exists=renamed["renamed_atom14_gt_exists"],
        alt_naming_is_better=renamed["alt_naming_is_better"])


def packing_loss(pred_sin_cos, pred_sin_cos_unnorm, batch, tables):
    """The full packing-only objective. Returns (total, per-term dict)."""
    loss_mask = batch["res_mask"] * batch["diffuse_mask"] * batch["plddt_mask"]
    gt_sin_cos = torch.stack((batch["torsions_1"].sin(), batch["torsions_1"].cos()), dim=-1)
    chi = supervised_chi_loss(
        angles_sin_cos=pred_sin_cos,
        unnormalized_angles_sin_cos=pred_sin_cos_unnorm,
        aatype=batch["aatypes_1"],
        seq_mask=loss_mask,
        chi_mask=batch["torsions_mask"],
        chi_angles_sin_cos=gt_sin_cos,
        chi_weight=1, angle_norm_weight=0.02, eps=1e-6)
    fape = sidechain_fape(pred_sin_cos, batch, tables)
    total = chi + fape
    return total, {"chi": chi.detach(), "fape": fape.detach(), "total": total.detach()}


__all__ = ["RigidGroupTables", "packing_loss", "sidechain_fape", "predicted_atom14"]
