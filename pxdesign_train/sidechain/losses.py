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

    FORM vs THE PAPER -- read this before "fixing" either side.

    Overleaf (appendix, "Global Side-Chain Denoising with Local-Frame Supervision")
    writes the PRIMARY supervision in the LOCAL frame:

        (A)   y_hat = stopgrad(F_hat)^-1 . x_hat        loss = || y_hat - y_GT ||^2

    and calls the global pseudo-target an "auxiliary formulation ... useful for
    implementations that operate directly on global coordinates".

    This function implements that auxiliary form:

        (B)   x_target = stopgrad(F_hat) . y_GT         loss = || x_hat - x_target ||^2

    (A) and (B) are EXACTLY equivalent, not approximately. With F_hat = (R, t) rigid and
    R orthonormal:

        x_hat - (R.y_GT + t) = R . [ R^-1(x_hat - t) - y_GT ]
        ||R.v|| = ||v||                       =>  loss_A == loss_B
        d/dx_hat ||R^-1(x_hat - t) - y_GT||^2
            = 2R(R^-1(x_hat - t) - y_GT) = 2(x_hat - F_hat.y_GT)
            = d/dx_hat ||x_hat - F_hat.y_GT||^2
                                              =>  identical gradients too

    -- provided the frame is stop-grad on BOTH sides, which it is (.detach() below).

    So this is a presentation mismatch, not a defect: the paper's stated primary is (A),
    the code computes (B), and they are the same number and the same gradient. Pick ONE
    for consistency -- either switch the code to local-primary, or have the paper state
    that the implementation uses the equivalent global pseudo-target form. Do not "fix"
    it by changing the maths; there is nothing to fix.

    `pred_global` is already in the global frame, so gradients flow to S_phi's coordinate
    output but not through `frame_R` / `frame_t`.
    """
    target = to_global(gt_local, frame_R.detach(), frame_t.detach())
    se = ((pred_global - target) ** 2).sum(dim=-1)
    m = mask.to(se.dtype).expand_as(se)
    return (se * m).sum() / (m.sum() + eps)
