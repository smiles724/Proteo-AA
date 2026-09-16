"""FaMPNN training objectives, as specified in the preprint.

FaMPNN ships inference only (upstream issue #9 asks for training code and is
unanswered), so the objectives are written here from the paper. Everything that
*is* shipped -- the MAR and EDM interpolants, the denoiser MLP, the confidence
head, the local-frame transforms -- is reused rather than reimplemented.

Reference: Shuai et al., "Sidechain conditioning and modeling for full-atom
protein sequence design with FAMPNN", bioRxiv 2025.02.13.637498.

* Appendix C.1 -- the total objective is ``L_total = L_MLM + L_diff``, summed with
  no relative weighting ("We did not experiment with relative weightings of the
  losses on each objective").
* Appendix C -- masking rate ``t = sqrt(u)``, ``u ~ U(0,1)``, each residue *kept*
  with probability ``t``. So the MLM term is scored on the complement: the
  positions the interpolant masked.
* Section 4.3.1 -- the side-chain denoiser is trained on the L2 error to the clean
  coordinates under the variance-exploding EDM scheme, with EDM's own loss
  weighting ``1/c_out(sigma)^2`` (shipped as ``EDM.get_loss_weight``).
* Appendix D.4.1 -- the confidence head is a 33-way classifier over per-atom
  side-chain error binned evenly on [0, 4] Angstrom, trained with cross entropy.
"""

import torch
import torch.nn.functional as F

# Appendix D.4.1: 33 evenly spaced bins between 0 and 4 Angstrom. The shipped
# inference head builds LOWER EDGES as linspace(min_bin, max_bin, n_bins) and
# takes centres at edge + step/2, giving edges 0.125k and centres 0.0625 + 0.125k
# -- exactly the paper's Algorithm 7 linspace(0.0625, 4.0625, 33). Training targets
# must therefore use floor(error / step), not a round-to-nearest-centre, or every
# label lands half a bin low.
PSCE_MIN_BIN = 0.0
PSCE_MAX_BIN = 4.0
PSCE_NUM_BINS = 33
PSCE_BIN_WIDTH = (PSCE_MAX_BIN - PSCE_MIN_BIN) / (PSCE_NUM_BINS - 1)


def psce_bin_spec(module=None):
    """Bin spec from the model's own config when available, else the pinned default."""
    cfg = (
        getattr(getattr(module, "cfg", None), "sce_bins", None)
        if module is not None
        else None
    )
    if cfg is None:
        return PSCE_MIN_BIN, PSCE_MAX_BIN, PSCE_NUM_BINS
    return float(cfg.min_bin), float(cfg.max_bin), int(cfg.n_bins)


def _masked_mean(values, mask):
    """Mean of ``values`` over ``mask``, or an exact zero when nothing is scored.

    Returning a real zero that still carries grad keeps the loop differentiable
    on batches where a term has no supervision (e.g. an all-glycine crop).
    """
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    count = mask.sum()
    return total / count.clamp_min(1.0), count


def sequence_mlm_loss(seq_logits, aatype, seq_mlm_mask, seq_mask):
    """``L_MLM``: cross entropy on the residues the interpolant masked.

    ``seq_mlm_mask`` follows the upstream convention -- 1 where a residue was
    *kept*, 0 where it was masked -- so the objective scores ``1 - seq_mlm_mask``.
    Padding is excluded via ``seq_mask``.
    """
    scored = (1.0 - seq_mlm_mask) * seq_mask
    per_residue = F.cross_entropy(
        seq_logits.reshape(-1, seq_logits.shape[-1]).float(),
        aatype.reshape(-1).long(),
        reduction="none",
    ).reshape(aatype.shape)
    loss, count = _masked_mean(per_residue, scored)
    return loss, dict(masked_residues=count.detach())


def sidechain_diffusion_loss(x1_pred, x1_target, loss_weight, atom_mask):
    """``L_diff``: EDM-weighted L2 between predicted and clean side-chain atoms.

    Operates on local-frame side-chain coordinates ``[..., A, 3]``.
    ``loss_weight`` is EDM's ``1/c_out(sigma)^2`` per example, broadcast over
    atoms; ``atom_mask`` selects the atoms that actually exist and are supervised.

    The weighted squared error is summed over xyz and averaged over supervised
    atoms, which is the standard EDM denoising objective.
    """
    squared = (x1_pred.float() - x1_target.float()).pow(2).sum(-1)  # [..., A]
    weight = loss_weight.reshape(
        *loss_weight.shape, *([1] * (squared.dim() - loss_weight.dim()))
    )
    loss, count = _masked_mean(squared * weight, atom_mask)
    with torch.no_grad():
        unweighted, _ = _masked_mean(squared, atom_mask)
    return loss, dict(scored_atoms=count.detach(), sidechain_mse_local=unweighted.detach())


def psce_bin_targets(
    x_pred, x_target, *, min_bin=PSCE_MIN_BIN, max_bin=PSCE_MAX_BIN, num_bins=PSCE_NUM_BINS
):
    """Bin per-atom side-chain error onto the confidence head's classes.

    Bin ``k`` is the half-open interval ``[min + k*step, min + (k+1)*step)`` for
    ``step = (max - min) / (num_bins - 1)``, which is the binning implied by the
    shipped head's lower edges. Errors at or beyond the top edge saturate in the
    last bin.
    """
    error = (x_pred.float() - x_target.float()).norm(dim=-1)
    step = (max_bin - min_bin) / (num_bins - 1)
    index = torch.floor((error - min_bin) / step).long()
    return index.clamp_(0, num_bins - 1), error


def confidence_loss(psce_logits, x_pred, x_target, atom_mask, *, bin_spec=None):
    """Categorical cross entropy of the confidence head against binned error.

    ``x_pred`` must already be detached: per Appendix D.4 the confidence inputs
    carry a stop gradient so this term cannot influence the main model.
    """
    min_bin, max_bin, num_bins = bin_spec or (PSCE_MIN_BIN, PSCE_MAX_BIN, PSCE_NUM_BINS)
    if psce_logits.shape[-1] != num_bins:
        raise ValueError(
            f"confidence head emits {psce_logits.shape[-1]} bins, expected {num_bins}"
        )
    target, error = psce_bin_targets(
        x_pred, x_target, min_bin=min_bin, max_bin=max_bin, num_bins=num_bins
    )
    per_atom = F.cross_entropy(
        psce_logits.reshape(-1, psce_logits.shape[-1]).float(),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    loss, count = _masked_mean(per_atom, atom_mask)
    with torch.no_grad():
        mean_error, _ = _masked_mean(error, atom_mask)
    return loss, dict(
        confidence_atoms=count.detach(), true_sidechain_error=mean_error.detach()
    )


def total_loss(loss_mlm, loss_diff, loss_confidence=None):
    """``L_total = L_MLM + L_diff`` (Appendix C.1), confidence added separately.

    The confidence term is a separate head trained on stop-gradient inputs, so
    adding it changes no gradient reaching the main model; it is summed here only
    so one ``backward`` covers every parameter.
    """
    total = loss_mlm + loss_diff
    if loss_confidence is not None:
        total = total + loss_confidence
    return total
