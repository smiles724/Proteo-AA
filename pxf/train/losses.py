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


# The reduction used to turn per-atom squared error into one scalar. The authors
# did not release the training loop, so which one FaMPNN used cannot be
# established from the released code; both are implemented and the active choice
# is returned in the stats rather than left implicit.
#
#   per_residue  L_i = sum_a m_ia d_ia^2 / sum_a m_ia,  then L = mean_i w_i L_i
#                every residue counts once, so Trp does not outweigh Ala.
#   per_atom     L = sum_ia m_ia w_i d_ia^2 / sum_ia m_ia
#                atom-weighted, so large side chains dominate the gradient.
SIDECHAIN_REDUCTIONS = ("per_residue", "per_atom")
DEFAULT_SIDECHAIN_REDUCTION = "per_residue"


def _masked_mean(values, mask):
    """Mean of ``values`` over ``mask``, or an exact zero when nothing is scored.

    Returning a real zero that still carries grad keeps the loop differentiable
    on batches where a term has no supervision (e.g. an all-glycine crop).
    """
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    count = mask.sum()
    return total / count.clamp_min(1.0), count


def _broadcast_weight(loss_weight, ndim):
    """Right-pad a per-example weight with singleton axes up to ``ndim``."""
    pad = ndim - loss_weight.dim()
    if pad < 0:
        raise ValueError(
            f"loss weight has {loss_weight.dim()} dims, more than the {ndim} it must "
            "broadcast against"
        )
    return loss_weight.reshape(*loss_weight.shape, *([1] * pad))


def _per_residue_mean(squared, atom_mask):
    """Average each residue's squared error over *its own* supervised atoms.

    Returns ``(per_residue, residue_mask)``, both ``[..., L]``. A residue with no
    supervised atom -- glycine, an unresolved side chain, a target masked out by
    the interpolant -- gets an exact zero and is excluded by ``residue_mask``, so
    it neither contributes error nor dilutes the mean.
    """
    mask = atom_mask.to(squared.dtype)
    atoms_per_residue = mask.sum(-1)
    per_residue = (squared * mask).sum(-1) / atoms_per_residue.clamp_min(1.0)
    return per_residue, (atoms_per_residue > 0).to(squared.dtype)


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


def sidechain_diffusion_loss(
    x1_pred, x1_target, loss_weight, atom_mask, *, reduction=DEFAULT_SIDECHAIN_REDUCTION
):
    """``L_diff``: EDM-weighted L2 between predicted and clean side-chain atoms.

    Operates on local-frame side-chain coordinates ``[..., L, A, 3]``.
    ``loss_weight`` is EDM's ``1/c_out(sigma)^2`` per example, broadcast over
    residues and atoms; ``atom_mask`` selects the atoms that exist and are
    supervised.

    ``reduction`` picks how per-atom error becomes one scalar, and it changes what
    the objective actually optimizes:

    ``"per_residue"`` (default)
        Average within each residue first, then over residues. Every residue
        carries weight 1, so Trp (14 supervised slots) does not count seven times
        Ser (2). Prefer this when the quantity of interest is per-residue packing
        quality, which is what the downstream side-chain metrics all report.

    ``"per_atom"``
        One global average over supervised atoms, so large side chains dominate
        the gradient in proportion to their atom count.

    FaMPNN released inference only, so the original reduction is not recoverable
    from the code; both are kept and the active one is reported in the stats.
    """
    if reduction not in SIDECHAIN_REDUCTIONS:
        raise ValueError(
            f"Unknown reduction {reduction!r}; choose from {list(SIDECHAIN_REDUCTIONS)}"
        )
    squared = (x1_pred.float() - x1_target.float()).pow(2).sum(-1)  # [..., L, A]
    atom_mask = atom_mask.to(squared.dtype)

    if reduction == "per_residue":
        per_residue, residue_mask = _per_residue_mean(squared, atom_mask)
        weight = _broadcast_weight(loss_weight, per_residue.dim())
        loss, residues = _masked_mean(per_residue * weight, residue_mask)
        atoms = atom_mask.sum()
    else:
        weight = _broadcast_weight(loss_weight, squared.dim())
        loss, atoms = _masked_mean(squared * weight, atom_mask)
        residues = (atom_mask.sum(-1) > 0).to(squared.dtype).sum()

    with torch.no_grad():
        # Always per-atom and unweighted, so this diagnostic stays comparable
        # across reductions and across noise levels.
        unweighted, _ = _masked_mean(squared, atom_mask)
    return loss, dict(
        scored_atoms=atoms.detach(),
        scored_residues=residues.detach(),
        sidechain_mse_local=unweighted.detach(),
        reduction=reduction,
    )


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
