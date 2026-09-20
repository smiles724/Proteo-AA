"""The joint objective: the backbone anchor, the local term, and placement.

``L_total = L_BB + lambda_local L_local + lambda_place L_place``

``L_BB`` and ``L_local`` are the repository's existing losses, reused unchanged:
the same-frame EDM-weighted backbone loss from :mod:`pxf.couple.losses`, and
FaMPNN's own per-token side-chain diffusion loss from :mod:`pxf.train.losses`.
What is new here is the *placement* term and its control.

Why placement is a separate term from the local loss, rather than the same
error measured differently: the local term scores side chains in each residue's
own backbone frame, so it is blind to where that frame is. A backbone error
rotates every side chain in the residue and leaves the local error untouched.
Placement is the term that can see it, and it is the only one of the two whose
gradient reaches the backbone through the frames rather than only through the
encoder.

Three things it does differently from the local term, each deliberate:

**It scores physical atoms only.** Ghost slots sit at the frame origin by
construction; placing them and comparing against an atom that does not exist
measures nothing. (The local term *does* supervise them, to zero, because that
is the source objective. The two masks are different sets -- see
:mod:`pxf.joint.data`.)

**It is not multiplied by the EDM weight.** ``1/c_out(sigma_SC)^2`` diverges as
the side-chain noise goes to zero, and it is the right weight for a denoising
error that vanishes with it. A frame error does not vanish with it: at
near-clean side-chain noise the local prediction is nearly exact and the
placement error is almost entirely backbone. Weighting it by ``1/c_out^2`` would
multiply a backbone error by an arbitrarily large number for a reason that has
nothing to do with the backbone.

**It is robust.** ``rho(u) = sqrt(1 + u^2) - 1`` is quadratic near zero and
linear far from it, so one badly placed long side chain cannot dominate the
gradient the way a squared error would.
"""

from dataclasses import dataclass

import torch


# The atom-name swaps upstream declares chemically equivalent. Taken from
# FaMPNN's residue_constants rather than restated, so a change there is a
# change here. Note the table covers ASP/GLU/PHE/TYR only -- ARG's NH1/NH2 and
# the HIS/ASN/GLN density ambiguities are not in it, and inventing entries
# would mean scoring a swap the target's own convention does not license.
def _renaming_swaps():
    from fampnn.data import residue_constants as rc

    return rc.residue_atom_renaming_swaps


def robust(u):
    """``sqrt(1 + u^2) - 1``: quadratic near zero, linear in the tail.

    Written as ``u^2 / (1 + sqrt(1 + u^2))``, which is the same function without
    the cancellation. Taken literally in float32 the subtraction loses most of
    its significant digits for small ``u`` -- rho(1e-3) comes out 5% low -- and
    small is where a converging run spends its time.
    """
    squared = u.pow(2)
    return squared / (1.0 + torch.sqrt(1.0 + squared))


@dataclass
class JointLoss:
    """One step's objective and the parts it was made of."""

    total: torch.Tensor
    backbone: torch.Tensor
    local: torch.Tensor = None
    placement: torch.Tensor = None
    stats: dict = None

    def scalars(self):
        out = {"loss": float(self.total), "loss_bb": float(self.backbone)}
        if self.local is not None:
            out["loss_local"] = float(self.local)
        if self.placement is not None:
            out["loss_place"] = float(self.placement)
        for key, value in (self.stats or {}).items():
            out[key] = (
                float(value)
                if torch.is_tensor(value) or isinstance(value, (int, float))
                else value
            )
        return out


# ---- symmetry ---------------------------------------------------------------


def symmetry_pairs(device=None):
    """``[n_swap, 3]`` of ``(aatype, slot_a, slot_b)`` over the 33 side-chain slots."""
    from fampnn.data import residue_constants as rc
    from pxf import atom37

    side = list(rc.non_bb_idxs)
    position = {slot: index for index, slot in enumerate(side)}
    rows = []
    for three, swaps in _renaming_swaps().items():
        one = rc.restype_3to1[three]
        aa = rc.restype_order_with_x[one]
        for first, second in swaps.items():
            a, b = atom37.ATOM37.index(first), atom37.ATOM37.index(second)
            rows.append((aa, position[a], position[b]))
    if not rows:
        return torch.zeros(0, 3, dtype=torch.long, device=device)
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.no_grad()
def resolve_symmetry(predicted, target, mask, aatype):
    """Pick, per residue, whether to apply its chemically allowed swap.

    Returns ``(target, mask)`` with the swap applied where it lowers the masked
    squared error. The choice is discrete and detached -- a gradient through
    "which naming" is not defined -- and it is made once, so the same assignment
    scores every term that uses it.

    A swap is only considered when **both** atoms are observed. Swapping a
    present atom onto a missing one would score a real prediction against an
    absent target and drop a real one, which is a silent relabelling of the
    supervision rather than a symmetry.
    """
    pairs = symmetry_pairs(device=predicted.device)
    target = target.clone()
    mask = mask.clone()
    if pairs.numel() == 0:
        return target, mask
    for aa, first, second in pairs.tolist():
        rows = (aatype == aa) & (mask[..., first] > 0) & (mask[..., second] > 0)
        if not bool(rows.any()):
            continue
        straight = (
            (predicted[..., first, :] - target[..., first, :]).pow(2).sum(-1)
            + (predicted[..., second, :] - target[..., second, :]).pow(2).sum(-1)
        )
        swapped = (
            (predicted[..., first, :] - target[..., second, :]).pow(2).sum(-1)
            + (predicted[..., second, :] - target[..., first, :]).pow(2).sum(-1)
        )
        take = rows & (swapped < straight)
        if not bool(take.any()):
            continue
        a = target[..., first, :].clone()
        b = target[..., second, :].clone()
        target[..., first, :] = torch.where(take.unsqueeze(-1), b, a)
        target[..., second, :] = torch.where(take.unsqueeze(-1), a, b)
    return target, mask


# ---- the placement term -----------------------------------------------------


def placement_loss(placed, native, mask, *, aatype=None, symmetry=True):
    """``L_place``: robust error between placed and native side chains.

    ``placed`` and ``native`` are ``[..., L, 33, 3]`` in one shared global frame
    -- the preprocessing transform already put them there, so nothing is aligned
    here. Aligning per residue would erase exactly the placement error this term
    exists to measure; aligning to the prediction would credit the model for
    re-posing the structure.

    Reduced per example over supervised *components*, then averaged over the
    clone axis, matching the local term's normalization.
    """
    if aatype is not None and symmetry:
        native, mask = resolve_symmetry(placed.detach(), native, mask, aatype)
    mask = mask.to(placed.dtype).unsqueeze(-1).expand_as(placed)
    error = robust(placed.float() - native.float()) * mask
    dims = tuple(range(1, error.dim()))
    per_example = error.sum(dims) / mask.sum(dims).clamp(min=1e-6)
    with torch.no_grad():
        distance = (placed.float() - native.float()).norm(dim=-1)
        atoms = mask[..., 0]
        rmsd = ((distance.pow(2) * atoms).sum() / atoms.sum().clamp(min=1e-6)).sqrt()
    return per_example.mean(), dict(
        placement_atoms=mask[..., 0].sum().detach(),
        placement_rmsd=rmsd,
    )


def frame_only_placement(model_forward, batch, *, symmetry=True):
    """``L_frame``: native local conformations on predicted frames.

    The mechanistic control. It supervises the frames with geometry that needs
    no FaMPNN prediction at all -- ``C = T(B_hat) stopgrad(q*)`` -- so if it
    matches the learned-side-chain arm, the gain was extra frame supervision
    rather than anything the side-chain model knew. Native side chains are used
    here only as a training target; the trained backbone needs none at
    inference.
    """
    from fampnn.data import residue_constants as rc
    from pxf.joint.model import place_sidechains

    prediction = model_forward.prediction
    native_local = prediction.clone(batch.local_target).detach()
    physical = prediction.clone(batch.physical_mask)
    frames37 = prediction.clone(model_forward.coords37)
    supplied = (frames37.abs().sum(-1) > 0).to(frames37.dtype)
    placed, _exists = place_sidechains(
        native_local, frames37, physical, backbone_mask=supplied[..., rc.bb_idxs]
    )
    native_global = prediction.clone(batch.native_batch["x"][..., rc.non_bb_idxs, :])
    return placement_loss(
        placed,
        native_global,
        physical,
        aatype=prediction.aatype,
        symmetry=symmetry,
    )


# ---- the combined objective --------------------------------------------------


def combined_loss(
    backbone, *, local=None, placement=None, lambda_local=1.0, lambda_place=1.0,
    stats=None, sample_id=None,
):
    """Sum the enabled terms, refusing a non-finite one.

    Deliberately *not* the source-parity behaviour. ``pxf.train.losses.total_loss``
    replaces a NaN term with zero, which is right for a long unattended training
    run: one bad example should not end it. It is wrong here, where a silently
    dropped side-chain term turns a candidate arm into the backbone-only
    baseline and the two would then be compared as if they differed.
    """
    stats = dict(stats or {})
    terms = {"L_BB": backbone, "L_local": local, "L_place": placement}
    for name, value in terms.items():
        if value is not None and not torch.isfinite(value):
            raise ValueError(
                f"{name} is {float(value)} on "
                f"{sample_id if sample_id is not None else 'this example'}. "
                "Refusing to continue: dropping the term would quietly turn this "
                "arm into a different one"
            )
    total = backbone
    if local is not None:
        total = total + float(lambda_local) * local
    if placement is not None:
        total = total + float(lambda_place) * placement
    stats.update(lambda_local=float(lambda_local), lambda_place=float(lambda_place))
    return JointLoss(
        total=total, backbone=backbone, local=local, placement=placement, stats=stats
    )


# ---- coefficient calibration --------------------------------------------------


def gradient_norm(loss, target, *, retain=True):
    """RMS gradient of ``loss`` with respect to ``target``'s valid coordinates.

    The calibration quantity: coefficients are set so each auxiliary term's
    gradient at the backbone is a stated fraction of the anchor's, rather than
    by the raw sizes of losses that are normalized differently.
    """
    if not target.requires_grad:
        raise ValueError(
            "the backbone prediction does not require grad, so there is no "
            "gradient to calibrate against"
        )
    (grad,) = torch.autograd.grad(loss, target, retain_graph=retain, allow_unused=False)
    if grad is None:
        return 0.0
    return float(grad.pow(2).mean().sqrt())


def calibrate(anchor_norm, auxiliary_norm, *, ratio=0.1):
    """The coefficient making an auxiliary term ``ratio`` of the anchor's gradient.

    A zero or non-finite auxiliary gradient is a diagnostic failure -- the term
    is not reaching the backbone at all -- not something to divide around with
    an epsilon.
    """
    if not auxiliary_norm > 0 or not float("inf") > auxiliary_norm:
        raise ValueError(
            f"the auxiliary term's gradient at the backbone is {auxiliary_norm}, so "
            "it either does not reach the backbone or is not finite. Calibrating "
            "against it would hide that"
        )
    return float(ratio) * float(anchor_norm) / float(auxiliary_norm)
