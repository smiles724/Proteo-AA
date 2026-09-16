"""Losses for the staged coupling phases.

Deliberately thin. Phase 1 reuses FaMPNN's *native* side-chain objective rather
than inventing a packing loss: the plan is explicit that the adapter should be
trained against the schedule, weighting and denoising target the module was
trained with, so the only change is that ``h_V`` now carries the BB -> SC
residual. Phase 2 is the backbone denoising loss on the corrected backbone.

    phase 1   L_SC  = FaMPNN diffusion loss, h_V + A_BS(a_BB, sigma)   -> A_BS
    phase 2   L_BB  = EDM-weighted denoising loss on X_BB^1            -> A_SB
    phase 3   alternate the two, one per batch

``L_SC`` is the diffusion term **only** -- deliberately not
``L_MLM + L_diff + L_confidence``. The question phase 1 answers is whether
PXDesign's ``a_token`` carries information that improves side-chain packing:

    min_{theta_BS}  E[ L_SC( D_psi( h_V + A_BS(a_token, sigma_B) ) ) ]

and the sequence is held *fixed* throughout, so an MLM term would be scoring a
prediction of something already given. Adding it would move the loss without
bearing on the question, and a gain could no longer be read as better packing.
The confidence head is likewise excluded: it trains on stop-gradient inputs, so
it cannot reach ``A_BS`` at all and would only add noise to the curve.
:func:`sidechain_coupling_loss` therefore calls ``train_step.diffusion_loss``
directly rather than ``training_forward``, and ``test_couple_losses.py`` pins
that -- the objective is a design decision, not an accident of which helper was
convenient.

Phase 3 alternates rather than summing. Backpropagating L_BB through the
side-chain sampler into A_BS would make the two adapters compete through a
50-step rollout, which is expensive and gives poor credit assignment; alternating
keeps each adapter's gradient attributable to one objective.
"""

from dataclasses import dataclass

import torch

from pxf.couple import fampnn_iface as iface
from pxf.train import losses as train_losses
from pxf.train import step as train_step


@dataclass
class CoupledLoss:
    """One phase's loss and its diagnostics."""

    total: torch.Tensor
    kind: str
    stats: dict

    def scalars(self):
        out = {"loss": float(self.total), "loss_kind": self.kind}
        out.update(
            {
                k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float)) else v)
                for k, v in self.stats.items()
            }
        )
        return out


def sidechain_coupling_loss(
    model,
    batch,
    features,
    *,
    delta_h=None,
    multiplier=None,
    self_cond_p=None,
    generator=None,
    reduction=train_losses.DEFAULT_SIDECHAIN_REDUCTION,
):
    """``L_SC``: FaMPNN's diffusion objective alone, conditioned through ``A_BS``.

    ``features`` is the encoder's feature dict for the *generated* backbone;
    ``delta_h`` is the BB -> SC residual. Everything else -- the noise schedule,
    the 8-way clone, the EDM loss weighting, the teacher-forced sequence -- is
    reused from :mod:`pxf.train.step`, so the adapter is trained against the
    objective the module already knows.

    No MLM and no confidence term; see the module docstring for why. No
    ``scn_mlm_mask`` either: the coupling cycle packs with every side chain
    hidden, exactly as ``sidechain_pack`` does at inference, so every supervisable
    atom is a legitimate target.
    """
    conditioned = iface.with_residual(features, delta_h)
    loss, stats = train_step.diffusion_loss(
        model,
        batch,
        conditioned,
        multiplier=multiplier,
        self_cond_p=self_cond_p,
        generator=generator,
        scn_mlm_mask=None,
        reduction=reduction,
    )
    stats = dict(stats)
    stats["delta_h_norm"] = (
        torch.tensor(0.0) if delta_h is None else delta_h.detach().norm(dim=-1).mean()
    )
    return CoupledLoss(total=loss, kind="sidechain", stats=stats)


def backbone_denoising_loss(predicted, target, *, sigma, sigma_data=16.0, atom_mask=None):
    """``L_BB``: EDM-weighted denoising loss on the corrected backbone.

    ``predicted`` and ``target`` are coordinates on the same axis (PXDesign's
    flat atom axis, or a dense block) and are compared *without* superposition:
    a denoiser predicts in the frame it was given, so aligning first would hide
    exactly the error the loss is meant to penalize.

    The weight is EDM's ``1 / c_out(sigma)^2``, matching how the backbone module
    was trained. ``sigma_data`` defaults to Protenix's coordinate scale; pass the
    value from the backbone config to be exact.
    """
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=predicted.device)
    if sigma.dim() == 0:
        sigma = sigma.reshape(1)
    squared = (predicted.float() - target.float()).pow(2).sum(-1)  # [..., N_atom]
    # c_out(sigma) = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    weight = 1.0 / c_out.clamp_min(1e-8) ** 2
    while weight.dim() < squared.dim():
        weight = weight[..., None]
    mask = torch.ones_like(squared) if atom_mask is None else atom_mask.float()
    total = (squared * weight * mask).sum() / mask.sum().clamp_min(1.0)
    with torch.no_grad():
        rmsd = ((squared * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
    return CoupledLoss(
        total=total,
        kind="backbone",
        stats=dict(
            scored_atoms=mask.sum().detach(),
            backbone_rmsd_angstrom=rmsd,
            sigma=sigma.mean().detach(),
        ),
    )


def backbone_feedback_loss(
    cycle, target, *, sigma, sigma_data=16.0, atom_mask=None, require_feedback=True
):
    """``L_BB`` on a cycle's corrected backbone ``X_BB^1``.

    Refuses to fall back to the uncorrected proposal: scoring ``bb0`` would train
    nothing in ``A_SB`` while still producing a plausible loss curve.
    """
    if cycle.bb1_flat is None:
        if require_feedback:
            raise ValueError(
                "The cycle produced no corrected backbone, so this loss would "
                "score the uncorrected proposal and leave A_SB untrained. Enable "
                "the SC->BB adapter, or pass require_feedback=False deliberately."
            )
        predicted = cycle.bb0_flat
    else:
        predicted = cycle.bb1_flat
    loss = backbone_denoising_loss(
        predicted, target, sigma=sigma, sigma_data=sigma_data, atom_mask=atom_mask
    )
    loss.stats["delta_a_norm"] = (
        torch.tensor(0.0)
        if cycle.delta_a is None
        else cycle.delta_a.detach().norm(dim=-1).mean()
    )
    loss.stats["used_correction"] = torch.tensor(float(cycle.bb1_flat is not None))
    return loss


# ---- phase selection -------------------------------------------------------

SIDECHAIN_PHASES = ("bb_to_sc",)
BACKBONE_PHASES = ("sc_to_bb",)


def loss_kind_for(phase, step):
    """Which objective this step optimizes.

    Phase 3 alternates strictly by parity rather than sampling, so a run's
    objective sequence is reproducible from the step number alone.
    """
    if phase in SIDECHAIN_PHASES:
        return "sidechain"
    if phase in BACKBONE_PHASES:
        return "backbone"
    if phase == "joint":
        return "sidechain" if step % 2 == 0 else "backbone"
    if phase == "frozen":
        raise ValueError("Phase 'frozen' trains nothing; there is no loss to select")
    raise ValueError(f"Unknown phase {phase!r}")
