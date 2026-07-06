"""Local-frame side-chain coordinate loss.

Primary side-chain supervision: MSE between predicted and GT side-chain
coordinates expressed in the residue-local frame (SideCraft spec eq. L_sc^local).
Masked and averaged over valid side-chain atoms only.
"""
import torch


def sidechain_local_loss(
    pred_local: torch.Tensor,   # [..., A, 3]
    gt_local: torch.Tensor,     # [..., A, 3]
    mask: torch.Tensor,         # [..., A] bool/float
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked mean squared error over valid side-chain atoms. Returns scalar."""
    se = ((pred_local - gt_local) ** 2).sum(dim=-1)   # [..., A]
    m = mask.to(se.dtype)
    return (se * m).sum() / (m.sum() + eps)
