"""Torsion objective for the one-step packer (AF2 Algorithm 27 / APM L_chi).

APM's packing objective is L_Packing = L_chi + L_FAPE, both "following the
AlphaFold2 implementations". This file supplies L_chi; the coordinate half is
already in `sidechain/losses.py` as `sidechain_global_frame_aligned_loss`, which
is the frame-aligned squared error on the built atoms (same role as FAPE here,
unclamped and squared -- stated in docs/sc_torsion_packer_apm_zh.md, not hidden).

Algorithm 27, verbatim in form:

    a_hat = a_raw / ||a_raw||
    L_torsion  = mean_valid  min( ||a_hat - a_true||^2 , ||a_hat - a_alt||^2 )
    L_anglenorm = mean_valid | ||a_raw|| - 1 |
    L_chi = L_torsion + 0.02 * L_anglenorm

WHY THE NORM TERM IS NOT OPTIONAL. The head emits an unnormalised 2-vector and
the angle is its direction. Without the norm penalty the length is a free
parameter that the gradient can shrink toward zero, where the direction -- the
thing we actually predict -- becomes numerically undefined. 0.02 is AF2's
coefficient.

PI-PERIODIC TORSIONS. For ASP chi2, GLU chi3, PHE chi2 and TYR chi2 the side
chain is indistinguishable under a 180 degree rotation of that bond, so chi and
chi+pi describe the SAME structure and penalising the difference teaches the
network to guess a coin flip. This is AF2's `chi_pi_periodic` table.

It is NOT the same set as `metrics.SWAPS`, which also lists ARG NH1/NH2, LEU
CD1/CD2 and VAL CG1/CG2. Those are atom-NAMING ambiguities that the coordinate
loss resolves by permuting atoms (`losses.symmetry_align_prediction`); they are
not torsion periodicities (rotating LEU chi2 by 180 degrees does not map the
side chain onto itself -- it swaps two methyls that are then renamed). Keeping
the two tables separate is deliberate: torsion periodicity belongs here, atom
naming belongs in the coordinate loss.
"""
import math
from typing import Optional

import torch

from .buildsc import chi_from_local
from .chi_constants import CHI_ATOM_IDX, CHI_MASK, CHI_ROTATABLE, MAX_CHI
from .instantiate import STD_AA_3

ANGLE_NORM_WEIGHT = 0.02

# AF2 chi_pi_periodic: [20, MAX_CHI] bool in STD_AA_3 order.
_PI_PERIODIC_SPEC = {"ASP": (1,), "GLU": (2,), "PHE": (1,), "TYR": (1,)}
CHI_PI_PERIODIC = torch.zeros(len(STD_AA_3), MAX_CHI, dtype=torch.bool)
for _name, _ks in _PI_PERIODIC_SPEC.items():
    for _k in _ks:
        CHI_PI_PERIODIC[STD_AA_3.index(_name), _k] = True


def native_chi_targets(
    type_idx: torch.Tensor,        # [..., L] long
    gt_sc_local: torch.Tensor,     # [..., L, MAX_SC, 3] native side chain, residue frame
    gt_bb_local: torch.Tensor,     # [..., L, 3, 3] native N, CA, C in the SAME frame
    observed: torch.Tensor,        # [..., L, MAX_SC] bool -- supervised atoms
    bb_observed: Optional[torch.Tensor] = None,   # [..., L, 3] bool
):
    """Ground-truth chi and the mask of torsions that are actually measurable.

    A torsion is a target only when ALL FOUR of its defining atoms are observed;
    an unresolved CG makes chi1 unknown, not zero. Same rule the evaluation
    metrics use, so train and eval count the same torsions.
    """
    dev = type_idx.device
    tix = type_idx.clamp(0, len(STD_AA_3) - 1).long()
    chi = chi_from_local(type_idx, gt_sc_local, gt_bb_local)            # [..., L, MAX_CHI]
    if bb_observed is None:
        bb_observed = torch.ones(*type_idx.shape, 3, dtype=torch.bool, device=dev)
    combined = torch.cat([bb_observed.bool(), observed.bool()], dim=-1)  # [..., L, 3+MAX_SC]
    idx = CHI_ATOM_IDX.to(dev)[tix]                                      # [..., L, MAX_CHI, 4]
    flat = idx.reshape(*idx.shape[:-2], MAX_CHI * 4)
    present = torch.gather(combined, -1, flat).reshape(*idx.shape[:-1], 4).all(-1)
    valid = present & CHI_MASK.to(dev)[tix] & torch.isfinite(chi)
    valid = valid & ((type_idx >= 0) & (type_idx < len(STD_AA_3)))[..., None]
    # A torsion the DECODER cannot realise is not a prediction target. PRO's ring
    # closes back onto the backbone N, so no rotation about CA-CB or CB-CG keeps
    # the pyrrolidine intact and BuildSC leaves those atoms at their CCD values
    # (see buildsc.py). Supervising chi there would train the head to emit a number
    # that moves no atom, and would report a chi error the model cannot fix. PRO's
    # ATOMS are still supervised by the coordinate loss -- they are simply fixed.
    valid = valid & CHI_ROTATABLE.to(dev)[tix]
    return torch.nan_to_num(chi), valid


def torsion_angle_loss(
    raw: torch.Tensor,             # [..., L, MAX_CHI, 2] unnormalised head output
    gt_chi: torch.Tensor,          # [..., L, MAX_CHI] radians
    valid: torch.Tensor,           # [..., L, MAX_CHI] bool
    type_idx: torch.Tensor,        # [..., L] long
    *,
    angle_norm_weight: float = ANGLE_NORM_WEIGHT,
    eps: float = 1e-8,
):
    """AF2 Algorithm 27. Returns (loss, metrics) with metrics detached."""
    with torch.autocast(device_type=raw.device.type, enabled=False):
        raw = raw.float()
        gt_chi = gt_chi.float()
        # openfold's form, `sqrt(sum(x^2) + eps)`, not `vector_norm` + clamp.
        # They agree everywhere except at raw == 0, where vector_norm's gradient
        # is NaN -- and raw IS exactly 0 for masked rows, because the head's
        # residual branch is zero-initialised (see packer.py). Masking the loss
        # does not remove that NaN; adding eps under the square root does.
        norm = (raw.square().sum(-1) + eps).sqrt()                       # [..., MAX_CHI]
        unit = raw / norm[..., None]
        # CHANNEL ORDER IS (sin, cos), matching APM's `gt_sin_cos` and openfold's
        # `supervised_chi_loss`. It is arbitrary mathematically and load-bearing
        # in practice: get it wrong and every angle is reflected about pi/4.
        true = torch.stack([gt_chi.sin(), gt_chi.cos()], dim=-1)
        periodic = CHI_PI_PERIODIC.to(raw.device)[
            type_idx.clamp(0, len(STD_AA_3) - 1).long()
        ]                                                                # [..., L, MAX_CHI]
        # chi + pi is (-sin, -cos). For non-periodic torsions the alternative IS
        # the truth, so the min() below is a no-op there rather than a special case.
        alt = torch.where(periodic[..., None], -true, true)
        d_true = (unit - true).square().sum(-1)
        d_alt = (unit - alt).square().sum(-1)
        per_chi = torch.minimum(d_true, d_alt)
        m = valid.to(per_chi.dtype)
        count = m.sum().clamp_min(1.0)
        torsion = (per_chi * m).sum() / count
        anglenorm = ((norm - 1.0).abs() * m).sum() / count
        loss = torsion + float(angle_norm_weight) * anglenorm

        with torch.no_grad():
            pred_chi = torch.atan2(unit[..., 0], unit[..., 1])
            delta = pred_chi - gt_chi
            # Periodic torsions are compared modulo pi, everything else modulo 2pi.
            wrapped = torch.atan2(delta.sin(), delta.cos()).abs()
            folded = torch.where(periodic, torch.minimum(wrapped, math.pi - wrapped), wrapped)
            mae = (folded * m).sum() / count
            within = {
                f"chi_within_{deg}deg": (
                    ((folded < math.radians(deg)) & valid).sum().float() / count
                )
                for deg in (20, 40)
            }
        metrics = dict(
            chi_torsion=torsion.detach(), chi_anglenorm=anglenorm.detach(),
            chi_mae_deg=(mae * 180.0 / math.pi).detach(),
            chi_supervised=count.detach(), **{k: v.detach() for k, v in within.items()},
        )
    return loss, metrics


def packing_chi_loss(feat, pack, native, loss_mask):
    """L_chi for the SC-only phases, from the native features the phase already has.

    Kept here rather than inline in stage4.py so the index gymnastics -- the
    per-token features are unbatched, the packer's output carries the flattened
    (item x sigma) row axis -- is testable without building a model.

    Args:
        feat: the phase's input feature dict (`sc_gt_local`, `sc_bb_coords`,
            `sc_frame_R`, `sc_frame_t`, `sc_bb_observed_mask`).
        pack: the packer's outputs, including `sc_pred_chi_raw` [B, L, MAX_CHI, 2].
        native: [L] long native residue types.
        loss_mask: [B, L, MAX_SC] or [L, MAX_SC] bool -- supervised side-chain atoms.
    Returns:
        (loss, metrics) with metric keys already prefixed `torsion/`.
    """
    from .frames import to_local

    # Measure the target chi on the NATIVE backbone in the native frame -- the same
    # convention `metrics.diagnose_packing` evaluates in, so the training target and
    # the reported chi recovery are the same quantity.
    native_bb_local = to_local(feat["sc_bb_coords"].float()[..., :3, :],
                               feat["sc_frame_R"].float(), feat["sc_frame_t"].float())
    lead = pack["sc_pred_chi_raw"].shape[0]
    expand = lambda x: x[None].expand(lead, *x.shape)
    gt_chi, valid = native_chi_targets(
        expand(native), expand(feat["sc_gt_local"].float()), expand(native_bb_local),
        loss_mask if loss_mask.dim() == 3 else expand(loss_mask),
        bb_observed=expand(feat["sc_bb_observed_mask"].bool()[..., :3]),
    )
    loss, metrics = torsion_angle_loss(pack["sc_pred_chi_raw"], gt_chi, valid, expand(native))
    return loss, {"torsion/" + key: value for key, value in metrics.items()}


__all__ = [
    "torsion_angle_loss", "native_chi_targets", "packing_chi_loss",
    "CHI_PI_PERIODIC", "ANGLE_NORM_WEIGHT",
]
