"""FaMPNN's training objectives, as the original training code computes them.

The released ``fampnn`` package is inference only -- the training branch was
stripped out of ``SidechainDiffusionModule.sidechain_diffusion`` and the loss
module was not shipped at all (upstream issue #9 asks for training code and is
unanswered). The objectives here are transcribed from the code the released
weights were trained by: ``allatom_design`` (commit ``51c9d53``), the research
tree ``fampnn`` was factored out of, specifically

    allatom_design/model/seq_denoiser/sd_loss.py          SDLoss
    allatom_design/configs/seq_denoiser/seq_denoiser.yaml loss:

so what follows is a transcription, not an inference from the preprint. Where
the two disagree the code wins, and the disagreements are called out below --
they are the places an implementation written from the paper alone goes wrong.

``L_total = L_seq + L_scn_mse + L_psce``, each weighted by ``loss_weights``
(all 1.0 in the released config, which is what "we did not experiment with
relative weightings" means in Appendix C.1).

Three details that a paper-only reading gets wrong, each with a test:

* **The sequence term is normalized by the crop length, not by the number of
  masked tokens** (``seq_loss.per_token_avg: false``). It is a *sum* over masked
  positions divided by the fixed example size, so a batch the interpolant barely
  masked contributes a correspondingly small loss. Dividing by the masked count
  instead rescales every step by a random factor and changes the balance against
  the diffusion term.
* **The sequence term carries label smoothing 0.1** and excludes positions whose
  true residue is unknown (``X``): the model must never be trained to emit the
  mask token as a prediction.
* **The diffusion term is averaged per coordinate component**, over every
  supervised slot of the example, and only then multiplied by EDM's per-example
  weight. It is not a per-residue average, and the division is by ``3 x atoms``,
  not by atoms.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Confidence bins: 33 lower edges evenly spaced on [0, 4] Angstrom, with the top
# bin running to infinity. The shipped inference head takes bin *centres* at
# ``edge + step/2``, so a training label must floor onto the edges; rounding to
# the nearest centre puts every label half a bin low.
PSCE_MIN_BIN = 0.0
PSCE_MAX_BIN = 4.0
PSCE_NUM_BINS = 33
PSCE_BIN_WIDTH = (PSCE_MAX_BIN - PSCE_MIN_BIN) / (PSCE_NUM_BINS - 1)
# model_cfg.inf: the open upper edge of the last bin, as a finite number.
PSCE_INF = 1.0e9

# How the side-chain squared error is reduced to one number per example.
#
#   per_token   sum over supervised coordinate components / their count
#               (``mse_loss.per_token_avg: true``, the released setting)
#   fixed_size  the same sum divided by the constant ``L * 33 * 3``, so a crop
#               with few resolved side chains scores lower rather than being
#               renormalized (``per_token_avg: false``)
#
# The EDM weight is applied *after* this reduction, per example, in both cases.
SIDECHAIN_REDUCTIONS = ("per_token", "fixed_size")
DEFAULT_SIDECHAIN_REDUCTION = "per_token"


@dataclass
class LossSettings:
    """The ``loss:`` block of the original config, with its own defaults.

    These are settings of the objective, not of the optimizer: two runs with
    different values here are minimizing different things, so every field is
    recorded in the checkpoint.
    """

    # seq_loss
    label_smoothing: float = 0.1
    n_aatype: int = 21
    seq_per_token_avg: bool = False
    # mse_loss
    sidechain_reduction: str = DEFAULT_SIDECHAIN_REDUCTION
    # psce_loss
    inf: float = PSCE_INF
    # loss_weights
    weight_seq: float = 1.0
    weight_sidechain: float = 1.0
    weight_confidence: float = 1.0

    def __post_init__(self):
        if self.sidechain_reduction not in SIDECHAIN_REDUCTIONS:
            raise ValueError(
                f"Unknown reduction {self.sidechain_reduction!r}; choose from "
                f"{list(SIDECHAIN_REDUCTIONS)}"
            )


DEFAULT_LOSS_SETTINGS = LossSettings()


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


# ---- the primitives, one per term ------------------------------------------


def masked_mse(x, y, mask, *, per_token_avg=True):
    """Per-example masked MSE over every trailing axis (``sd_loss.masked_mse``).

    ``mask`` is the same shape as ``x``, so with a coordinate-shaped mask the
    denominator counts *components* -- three per atom. Returns ``[b]``.
    """
    data_dims = tuple(range(1, x.dim()))
    mask = mask.to(x.dtype)
    squared = (x - y).pow(2) * mask
    if per_token_avg:
        return squared.sum(data_dims) / mask.sum(data_dims).clamp(min=1e-6)
    n = math.prod(squared.shape[1:])
    return squared.sum(data_dims) / n


def masked_cross_entropy(
    logits, target, mask, *, label_smoothing=0.1, n_aatype=21, per_token_avg=False
):
    """Per-example label-smoothed cross entropy (``sd_loss.masked_cross_entropy``).

    Smoothing is applied to the one-hot target and renormalized, which is a
    slightly different quantity from ``F.cross_entropy(label_smoothing=)``: the
    smoothing mass is ``label_smoothing / n_aatype`` *added* to every class
    before normalizing, not ``label_smoothing`` redistributed. Returns ``[b]``.
    """
    target_oh = F.one_hot(target.long(), num_classes=logits.shape[-1]).to(logits.dtype)
    target_oh = target_oh + label_smoothing / n_aatype
    target_oh = target_oh / target_oh.sum(dim=-1, keepdim=True)

    logprobs = F.log_softmax(logits.float(), dim=-1)
    cel = -(logprobs * target_oh).sum(dim=-1)

    mask = mask.to(cel.dtype)
    if per_token_avg:
        return (cel * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    return (cel * mask).sum(dim=-1) / mask.shape[1]


def masked_seq_accuracy(logits, target, mask):
    """Per-example accuracy over the scored positions. Returns ``[b]``."""
    correct = (logits.argmax(dim=-1) == target.long()).to(logits.dtype)
    mask = mask.to(correct.dtype)
    return (correct * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)


def psce_bin_targets(x_pred, x_target, *, min_bin=PSCE_MIN_BIN, max_bin=PSCE_MAX_BIN,
                     num_bins=PSCE_NUM_BINS, inf=PSCE_INF):
    """One-hot bin assignment for per-atom side-chain error (``sd_loss.psce_loss``).

    Bin ``k`` is ``[lower_k, lower_{k+1})`` over ``linspace(min, max, num_bins)``,
    with the last bin running to ``inf``. Returns ``(one_hot, error)``; an error
    at or beyond ``inf`` produces an all-zero row, exactly as upstream, which
    contributes nothing to the numerator while still counting in the denominator.
    """
    lower = torch.linspace(min_bin, max_bin, num_bins, device=x_pred.device)
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    error = torch.norm(x_pred.float() - x_target.float(), dim=-1)
    binned = ((error[..., None] >= lower) & (error[..., None] < upper)).to(torch.float32)
    return binned, error


# ---- the three terms -------------------------------------------------------


def sequence_mlm_loss(
    seq_logits,
    aatype,
    seq_mlm_mask,
    seq_mask,
    *,
    seq_unk_mask=None,
    settings=DEFAULT_LOSS_SETTINGS,
):
    """``L_seq``: cross entropy on the residues the interpolant masked.

    ``seq_mlm_mask`` follows the upstream convention -- 1 where a residue was
    *kept* -- so the scored set is ``(1 - seq_mlm_mask) * seq_mask``, minus the
    positions whose true identity is unknown (``seq_unk_mask``). Training on
    those would teach the model to predict the ``X`` token it uses as its own
    mask.
    """
    scored = (1.0 - seq_mlm_mask) * seq_mask
    if seq_unk_mask is not None:
        scored = scored * (1.0 - seq_unk_mask.to(scored.dtype))
    per_example = masked_cross_entropy(
        seq_logits,
        aatype,
        scored,
        label_smoothing=settings.label_smoothing,
        n_aatype=settings.n_aatype,
        per_token_avg=settings.seq_per_token_avg,
    )
    with torch.no_grad():
        accuracy = masked_seq_accuracy(seq_logits, aatype, scored).mean()
        # The same cross entropy, per masked token. The loss itself is normalized
        # by the crop length, so its value moves with however much the
        # interpolant happened to hide; this one is comparable across steps and
        # is what a training curve should be read off.
        per_token = masked_cross_entropy(
            seq_logits,
            aatype,
            scored,
            label_smoothing=settings.label_smoothing,
            n_aatype=settings.n_aatype,
            per_token_avg=True,
        ).mean()
    return per_example.mean(), dict(
        masked_residues=scored.sum().detach(),
        sequence_accuracy=accuracy.detach(),
        mlm_per_token=per_token.detach(),
    )


def sidechain_diffusion_loss(
    x1_pred, x1_target, loss_weight, atom_mask, *, reduction=DEFAULT_SIDECHAIN_REDUCTION
):
    """``L_scn_mse``: EDM-weighted L2 to the clean local-frame side chains.

    ``atom_mask`` may be ``[..., L, A]`` or the coordinate-shaped
    ``[..., L, A, 3]``; either way the reduction divides by the number of
    supervised *components*, which is what the original does (its mask is
    ``x_mask``, already expanded over xyz).

    ``loss_weight`` is EDM's ``1 / c_out(sigma)^2``, one value per example, and
    it multiplies the example's *already reduced* error. That ordering matters:
    weighting inside the sum and dividing by the pooled count would let examples
    with more resolved atoms carry more of the batch's weight.
    """
    if reduction not in SIDECHAIN_REDUCTIONS:
        raise ValueError(
            f"Unknown reduction {reduction!r}; choose from {list(SIDECHAIN_REDUCTIONS)}"
        )
    if atom_mask.dim() == x1_pred.dim() - 1:
        atom_mask = atom_mask.unsqueeze(-1).expand_as(x1_pred)
    mask = atom_mask.to(x1_pred.dtype)
    per_example = masked_mse(
        x1_pred.float(),
        x1_target.float(),
        mask,
        per_token_avg=(reduction == "per_token"),
    )
    with torch.no_grad():
        # Always per-component and unweighted, so the diagnostic stays comparable
        # across reductions and across noise levels.
        unweighted = per_example.detach().clone()
        if reduction != "per_token":
            unweighted = masked_mse(x1_pred.float(), x1_target.float(), mask)
    loss = per_example * loss_weight.to(per_example.dtype).reshape(-1)
    return loss.mean(), dict(
        scored_atoms=(mask.sum() / 3).detach(),
        scored_residues=(mask.sum(dim=(-1, -2)) > 0).to(mask.dtype).sum().detach(),
        sidechain_mse_local=unweighted.mean(),
        reduction=reduction,
    )


def confidence_loss(psce_logits, x_pred, x_target, atom_mask, *, bin_spec=None,
                    inf=PSCE_INF):
    """``L_psce``: cross entropy of the confidence head against binned error.

    ``x_pred`` is the diffusion rollout and must already be detached -- the head
    trains on stop-gradient inputs so it cannot reach the rest of the model.
    Reduced per example over ``[L, 33]``, then averaged over the batch.
    """
    min_bin, max_bin, num_bins = bin_spec or (PSCE_MIN_BIN, PSCE_MAX_BIN, PSCE_NUM_BINS)
    if psce_logits.shape[-1] != num_bins:
        raise ValueError(
            f"confidence head emits {psce_logits.shape[-1]} bins, expected {num_bins}"
        )
    binned, error = psce_bin_targets(
        x_pred, x_target, min_bin=min_bin, max_bin=max_bin, num_bins=num_bins, inf=inf
    )
    logprobs = F.log_softmax(psce_logits.float(), dim=-1)
    cel = -(logprobs * binned).sum(dim=-1)
    mask = atom_mask.to(cel.dtype)
    dims = tuple(range(1, cel.dim()))
    per_example = (cel * mask).sum(dim=dims) / mask.sum(dim=dims).clamp(min=1e-8)
    with torch.no_grad():
        mean_error = (error * mask).sum() / mask.sum().clamp(min=1e-8)
    return per_example.mean(), dict(
        confidence_atoms=mask.sum().detach(),
        true_sidechain_error=mean_error.detach(),
    )


def total_loss(loss_seq, loss_sidechain, loss_confidence=None,
               *, settings=DEFAULT_LOSS_SETTINGS):
    """The weighted sum, with upstream's non-finite guard.

    ``SDLoss`` drops a NaN or Inf term rather than letting it poison every
    parameter through the sum; a single bad example otherwise ends the run. The
    replacement zero still carries grad so ``backward`` has something to walk.
    """
    terms = (
        (loss_seq, settings.weight_seq),
        (loss_sidechain, settings.weight_sidechain),
        (loss_confidence, settings.weight_confidence),
    )
    total = None
    for loss, weight in terms:
        if loss is None:
            continue
        if not torch.isfinite(loss):
            loss = loss.new_tensor(0.0, requires_grad=loss.requires_grad)
        contribution = loss * weight
        total = contribution if total is None else total + contribution
    return total
