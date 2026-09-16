"""Inject the BB->SC residual into FaMPNN's *iterative* design path.

``pxf.couple.controller`` applies ``A_BS`` by handing a modified feature dict to
``pack_from_features``. That works because the packing task calls the side-chain
module once. FaMPNN's sequence-design path does not: ``SeqDenoiser.sample``
loops, and each step runs the MPNN encoder and then the side-chain module
internally, so there is no feature dict for a caller to substitute.

This module wraps ``scn_diffusion_module.sidechain_diffusion`` for the duration
of a ``with`` block and adds a fixed residual to ``h_V`` on every call. Nothing
in FaMPNN is edited -- the attribute is restored on exit -- so
``pxf.provenance`` still pins the submodule unpatched.

``delta_h`` is constant within a sample by construction: it is
``A_BS(a_token, sigma_B)``, and neither the backbone token features nor the
chosen ``sigma_B`` change while one structure is being decoded. So it is
computed once and reused, rather than recomputed per step.

    delta_h = adapters.delta_h(a_token, sigma_b)          # [B, L, c_h_V]
    with residual_on_sidechain_diffusion(model, delta_h):
        designed = designer.design(...)                   # arm 3
    designed_uncoupled = designer.design(...)              # arm 2, hook off
"""

from __future__ import annotations

import contextlib


def _add_residual(feature_dict, delta):
    """A shallow copy whose ``h_V`` carries ``delta``; shapes must already match."""
    h_v = feature_dict["h_V"]
    if delta.shape[-1] != h_v.shape[-1]:
        raise ValueError(
            f"residual width {delta.shape[-1]} does not match h_V "
            f"{h_v.shape[-1]}; wrong adapter for this FaMPNN"
        )
    if delta.shape[-2] != h_v.shape[-2]:
        raise ValueError(
            f"residual length {delta.shape[-2]} does not match h_V "
            f"{h_v.shape[-2]}; the adapter was built for a different structure"
        )
    if delta.shape[0] == 1 and h_v.shape[0] > 1:
        delta = delta.expand(h_v.shape[0], *delta.shape[1:])
    updated = dict(feature_dict)
    updated["h_V"] = h_v + delta.to(h_v.dtype).to(h_v.device)
    return updated


@contextlib.contextmanager
def residual_on_sidechain_diffusion(model, delta_h, *, counter=None):
    """Add ``delta_h`` to ``h_V`` on every side-chain diffusion call in the block.

    ``delta_h=None`` is a no-op that still installs the wrapper, so the coupled
    and uncoupled arms traverse identical code and any difference between them
    is the residual rather than the call path.

    ``counter`` optionally receives ``{"calls": n, "applied": n}`` so a caller
    can assert the hook actually fired -- a silently inert hook would make the
    coupled arm a duplicate of the uncoupled one and the comparison a null
    result that looks like a finding.
    """
    module = model.denoiser.scn_diffusion_module
    original = module.sidechain_diffusion
    stats = counter if counter is not None else {}
    stats.setdefault("calls", 0)
    stats.setdefault("applied", 0)

    def wrapped(mpnn_feature_dict, *args, **kwargs):
        stats["calls"] += 1
        if delta_h is None:
            return original(mpnn_feature_dict, *args, **kwargs)
        stats["applied"] += 1
        return original(_add_residual(mpnn_feature_dict, delta_h), *args, **kwargs)

    module.sidechain_diffusion = wrapped
    try:
        yield stats
    finally:
        # Restore by deleting the instance attribute so the bound method on the
        # class is visible again; assigning `original` back would leave a stale
        # bound method behind and defeat provenance's unpatched claim.
        try:
            del module.sidechain_diffusion
        except AttributeError:
            module.sidechain_diffusion = original
