"""Side-chain coordinate losses.

The current training path supervises global-coordinate side-chain predictions
against a predicted-frame-aligned pseudo-target:

    x_target = stopgrad(F_hat) y_gt_local

where y_gt_local is the ground-truth side-chain geometry in the residue-local
frame. The legacy local-frame loss is kept for tests and older callers.
"""
import torch

from pxdesign_train.sidechain.frames import to_global


def sidechain_local_loss(
    pred_local: torch.Tensor,   # [..., A, 3]
    gt_local: torch.Tensor,     # [..., A, 3]
    mask: torch.Tensor,         # [..., A] bool/float
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked mean squared error over valid side-chain atoms. Returns scalar.

    Broadcasting-robust: when `pred_local` carries a leading per-sigma axis
    (`[N_sample, L, A, 3]`) but `gt_local`/`mask` are per-token (`[L, A, ...]`),
    the squared error broadcasts over the sample axis. We expand the mask to the
    broadcasted shape so the denominator counts the SAME atoms the numerator sums
    — otherwise the loss would scale with N_sample. This is a masked mean, so its
    scale is invariant to N_sample.
    """
    se = ((pred_local - gt_local) ** 2).sum(dim=-1)   # [..., A] (may broadcast)
    m = mask.to(se.dtype).expand_as(se)               # match numerator's coverage
    return (se * m).sum() / (m.sum() + eps)


def sidechain_global_frame_aligned_loss(
    pred_global: torch.Tensor,  # [..., L, A, 3]
    gt_local: torch.Tensor,     # [..., L, A, 3]
    frame_R: torch.Tensor,      # [..., L, 3, 3], local -> global
    frame_t: torch.Tensor,      # [..., L, 3]
    mask: torch.Tensor,         # [..., L, A] bool/float
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked MSE to GT local geometry attached to a stop-grad frame.

    `pred_global` is already in the global frame, so gradients flow to S_phi's
    coordinate output but not through `frame_R` / `frame_t`. This implements the
    predicted-frame-aligned supervision used when S_phi emits global coordinates.
    """
    target = to_global(gt_local, frame_R.detach(), frame_t.detach())
    se = ((pred_global - target) ** 2).sum(dim=-1)
    m = mask.to(se.dtype).expand_as(se)
    return (se * m).sum() / (m.sum() + eps)
