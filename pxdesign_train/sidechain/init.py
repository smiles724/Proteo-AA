"""Leakage-free side-chain initialization.

The hard leakage rule (SideCraft spec / Overleaf red text): we do NOT initialize
side-chain denoising by partially noising the ground-truth side-chain
coordinates, because a partially-noised side chain still encodes residue-specific
topology and thereby leaks amino-acid identity. Instead we start from isotropic
Gaussian noise in the residue-local frame, so the module receives only
inference-available information.

By construction these functions take NO ground-truth argument — that is enforced
by a test (`test_no_gt_argument`) so the leakage rule cannot be silently broken.

Two initializers live here:

* `gaussian_init_local` — isotropic N(0, sigma^2 I) in the residue-local frame.
  Legacy / A-B baseline. Because an isotropic Gaussian is rotation-invariant
  (R eps ~ eps in distribution), mapping it through the predicted frame F_hat
  carries **no** backbone-orientation information: S_phi is handed a global
  input cloud whose distribution is identical for every backbone orientation.
* `template_init_local` — Overleaf 0721 (0712) paragraph 221, the
  "template-anchored leakage-free initialization":

      y_{T,ij} = mu_ideal[a_i, j] + sigma_T * eps_ij,   eps ~ N(0, I)
      x_{T,ij} = F_hat_i y_{T,ij}

  mu_ideal is the *ideal* (type-conditioned, rotamer-free) side-chain geometry in
  the residue-local frame. It is ANISOTROPIC, so once rotated by F_hat the global
  initialization encodes the backbone orientation — which is the entire point of
  the paragraph. It still depends only on residue TYPE / atom mask, never on
  ground-truth side-chain coordinates, so the leakage rule holds.
"""
from typing import Optional, Tuple

import torch

from pxdesign_train.sidechain.instantiate import STD_AA_3

# Template perturbation scale (Angstrom, per coordinate). Small relative to
# side-chain bond lengths (~1.5 A) so the ideal template — and hence the
# orientation signal it carries through F_hat — survives the perturbation.
DEFAULT_SIGMA_T = 0.3


def gaussian_init_local(
    mask: torch.Tensor,
    sigma: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample local-frame side-chain coordinates from N(0, sigma^2 I).

    Args:
        mask: [..., A] bool valid-atom mask.
        sigma: noise scale.
        generator: optional torch.Generator for reproducibility.
    Returns:
        [..., A, 3] Gaussian, zeroed at invalid (padded) atoms.
    """
    shape = (*mask.shape, 3)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32) * sigma
    return noise * mask[..., None].to(noise.dtype)


def _ideal_template(
    type_idx: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    backbone: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indirection over the REGISTERED mu_ideal provider (lazy import; test-patchable).

    Returns (coords [..., MAX_SC, 3] float32, mask [..., MAX_SC] bool) in the
    residue-local frame, row/column order matching
    `instantiate.instantiate_from_type_indices` / `sidechain_atoms`.

    `generator` and `backbone` are forwarded so a STOCHASTIC provider (sampling a rotamer
    from a distribution) or a BACKBONE-DEPENDENT one (phi/psi-conditioned rotamer library)
    can be dropped in without touching this file, model.py or cogenerate.py. The shipped
    CCD provider ignores both.
    """
    from pxdesign_train.sidechain.templates import get_ideal_template_provider

    return get_ideal_template_provider()(type_idx, generator=generator, backbone=backbone)


def templates_available() -> bool:
    """True if the ideal-template table can be imported."""
    try:
        import pxdesign_train.sidechain.templates  # noqa: F401
    except Exception:
        return False
    return True


def template_init_local(
    type_idx: torch.Tensor,
    mask: torch.Tensor,
    sigma_T: float = DEFAULT_SIGMA_T,
    generator: Optional[torch.Generator] = None,
    backbone: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Overleaf 0721 (0712) paragraph 221: y_T = mu_ideal[a, j] + sigma_T * eps.

    Template-anchored, leakage-free side-chain initialization in the residue-local
    frame. Conditions ONLY on residue type + atom mask (both inference-available
    under the existing atom-mask teacher forcing); it never sees ground-truth
    side-chain coordinates.

    Args:
        type_idx: [...] long amino-acid indices (0..19, `instantiate.STD_AA_3`
            order). Out-of-range values (e.g. the -100 ignore index at
            non-design tokens) are clamped and rely on `mask` being False there.
        mask: [..., A] bool valid-atom mask, A == MAX_SC. Leading dims must match
            `type_idx`.
        sigma_T: template perturbation scale (Angstrom, per coordinate). Must stay
            small relative to side-chain bond lengths: a large sigma_T washes out
            the template's anisotropy, and with it the backbone-orientation signal
            that F_hat y_T is supposed to carry.
        generator: optional torch.Generator for reproducibility.

    Returns:
        [..., A, 3] float32 local-frame coordinates, zeroed at invalid atoms.
    """
    assert type_idx.shape == mask.shape[:-1], (
        f"type_idx {tuple(type_idx.shape)} must match mask leading dims "
        f"{tuple(mask.shape[:-1])}"
    )
    # Out-of-place: type_idx is often an expand()ed view (per-sigma tiling).
    safe_idx = type_idx.long().clamp(0, len(STD_AA_3) - 1)
    mu, tmask = _ideal_template(safe_idx, generator=generator, backbone=backbone)
    mu = mu.to(device=mask.device, dtype=torch.float32)
    tmask = tmask.to(device=mask.device, dtype=torch.bool)
    assert mu.shape == (*mask.shape, 3), (
        f"ideal template {tuple(mu.shape)} does not match mask {tuple(mask.shape)}"
    )

    noise = torch.randn(
        mu.shape, generator=generator, dtype=torch.float32, device=mu.device
    )
    y = mu + sigma_T * noise
    # An atom is generated only if it is both requested by the caller's mask and
    # present in the residue's ideal template.
    valid = mask.to(torch.bool) & tmask
    return y * valid[..., None].to(y.dtype)
