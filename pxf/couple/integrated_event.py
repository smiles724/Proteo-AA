"""One coupling event, shared by integrated inference and feedback training.

The event is the whole of the integrated method: at one noise level, the
backbone proposal is used to design a sequence, the realized sequence and
packing are read back, and the backbone is re-evaluated AT THE SAME NOISY
STATE with that information injected. Everything else is the stock PXDesign
trajectory.

    bb0, a_token = D(x_noisy, sigma)                provisional, no feedback
    residual     = A_BS(a_token, sigma) * M_binder  the trained coupling
    seq, sc      = FaMPNN_iter(bb0, target ctx; residual)   100 steps
    h_packed     = E(bb0, seq, sc)                  re-encode, NO A_BS hook
    delta        = E1(h_packed, sigma) * M_binder   the trainable feedback
    bb1          = D(x_noisy, sigma, delta)         SAME state, corrected

This module produces everything up to and including ``h_packed``, plus the
provenance to prove it did. It exists as one module because inference and
feedback training must agree on it exactly: the feedback module is trained on
states this produces and is then asked to correct states this produces, and a
second implementation is how a train/inference skew gets in.

### Three corrections this module exists to get right

**The zero-payload shortcut is inference-only.** Collapsing an all-zero
residual to ``None`` is right at inference -- it takes the genuine no-feedback
path rather than an arithmetic imitation of it -- and WRONG in training: a
freshly zero-initialised output head produces exactly zero on step one, and
dropping the tensor drops the gradient path with it, so the head would never
receive a first update. :func:`mask_feedback` therefore takes an explicit
``zero_bypass`` flag rather than inferring intent from the values.

**The packing must be read off bb0, not off a native backbone.**
`pxf.train.bs_seq_sc.prepare_structure` caches native backbone inputs, because
its task is reconstruction from a deposited structure. Reusing it here would
train the feedback module on packing computed for a backbone the trajectory
never visited. :func:`prepare_event` takes the provisional estimate and packs
on that.

**A_BS applies exactly once, and not during the re-encode.** The residual
conditions the design decode (both heads, once per encoder call). The
re-encode that FEEDS the feedback module must see the donor's own
representation of the realized state, so the hook is uninstalled for it --
otherwise the feedback would read a representation already shifted by the
thing it is meant to correct. The legacy packing hook must not run at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from pxf import atom37


@dataclass
class EventProducts:
    """Everything one event produces. Tensors are on the caller's device."""

    bb0: torch.Tensor                 # [1, n_atom, 3] provisional clean estimate
    a_token: torch.Tensor             # [1, L, c_token] tapped features
    residual: Optional[torch.Tensor]  # [1, L, c_h_V] the A_BS residual, or None
    aatype: torch.Tensor              # [1, L] designed identities (target held)
    sequence: str                     # whole complex, in token order
    binder_sequence: str              # the designed rows only
    coords_af2: torch.Tensor          # [1, L, 37, 3] designed full-atom
    atom_mask_af2: torch.Tensor       # [1, L, 37] what the designer produced
    availability: torch.Tensor        # [1, L, 37] generated-atom availability
                                      #   (Visibility.available, not the input mask)
    h_base: Optional[torch.Tensor]    # encoder features of bb0, sequence masked
    h_packed: torch.Tensor            # encoder features of the REALIZED state
    psce: torch.Tensor                # [1, L, 33] predicted side-chain error
    binder_mask: torch.Tensor         # [1, L] 1 = a designed row
    sigma: float                      # the ACTUAL churned sigma
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.aatype.shape[-1])


def mask_feedback(
    delta,
    binder_mask: torch.Tensor,
    *,
    zero_bypass: bool,
    name: str = "feedback",
):
    """Restrict a feedback payload to binder rows. Differentiable.

    Masking happens AFTER the projection and its bias, which is the only place
    it can: a bias added post-mask would write a constant onto target rows, and
    a bias added pre-mask and then masked is exactly this.

    ``zero_bypass`` is the inference-only shortcut described in the module
    docstring. Pass ``False`` from training, always.

    Accepts a plain tensor or a ``ConditioningFeedback``; returns the same kind.
    """
    from pxf.couple.pxdesign_iface import ConditioningFeedback

    def one(tensor, trailing_pair: bool = False):
        if tensor is None:
            return None
        mask = binder_mask.reshape(1, -1).to(tensor.dtype).to(tensor.device)
        if trailing_pair:
            # E2 writes the binder-BINDER pair block: a row is kept only if
            # both of its tokens are binder rows. Cross-pair writes are a
            # separate policy and are not enabled here.
            m = mask.reshape(-1)
            keep = (m[:, None] * m[None, :]).unsqueeze(-1)
            return tensor * keep
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(-1)
        return tensor * mask

    if isinstance(delta, ConditioningFeedback):
        single = one(delta.delta_single)
        pair = one(delta.delta_pair, trailing_pair=True)
        if zero_bypass and _all_zero(single) and _all_zero(pair):
            return None
        return ConditioningFeedback(delta_single=single, delta_pair=pair)

    masked = one(delta)
    if zero_bypass and _all_zero(masked):
        return None
    return masked


def _all_zero(tensor) -> bool:
    return tensor is None or not bool(torch.any(tensor != 0))


def assert_target_rows_untouched(delta, binder_mask: torch.Tensor) -> None:
    """Bit-exact zero on every non-binder row, verified rather than trusted."""
    from pxf.couple.pxdesign_iface import ConditioningFeedback

    tensors = []
    if isinstance(delta, ConditioningFeedback):
        tensors = [t for t in (delta.delta_single,) if t is not None]
    elif delta is not None:
        tensors = [delta]
    target = ~binder_mask.reshape(-1).bool()
    for tensor in tensors:
        rows = tensor.reshape(tensor.shape[-2], -1)[target.to(tensor.device)]
        if rows.numel() and bool(torch.any(rows != 0)):
            raise AssertionError(
                f"{int((rows != 0).any(dim=-1).sum())} target row(s) carry a "
                "non-zero feedback value; the mask was applied before a bias "
                "or not at all"
            )


def prepare_event(
    *,
    denoise: Callable,
    x_noisy: torch.Tensor,
    sigma: float,
    structure,
    designer,
    adapters=None,
    context: str = "complex_sc",
    seed: int = 0,
    design_id: str = "event",
    target: str = "target",
    residual_source: str = "matched",
    want_h_base: bool = True,
    tap=None,
) -> EventProducts:
    """Run one event up to ``h_packed``. No solver step is taken here.

    ``denoise`` must accept ``(x_noisy, sigma, tap=...)`` and run WITHOUT
    feedback -- this is the provisional call, and the corrected call is the
    caller's job, at the same state.

    ``tap`` reuses a BackboneTap the caller already installed. The integrated
    sampler holds one for the whole trajectory, and installing a second here
    would hook ``layernorm_a`` twice: the capture would still work but the
    injection counters would double-count and a later residual would be added
    twice. Passing it in is how that is avoided rather than hoped for.
    """
    from pxf.bench.backbone_inputs import build_design_inputs, check_design_mask
    from pxf.bench.coupled_design import build_residual, conditioned
    from pxf.couple.fampnn_iface import encode
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.visibility import predicted_availability

    import numpy as np

    device = x_noisy.device
    topology = structure.topology
    n_tokens = int(structure.num_tokens)
    sigma_t = torch.full((1,), float(sigma), device=device, dtype=torch.float32)

    # ---- 1. the provisional estimate, and the features A_BS reads ---------
    if tap is None:
        with BackboneTap(designer_module(denoise)) as owned:
            calls_before = owned.calls
            bb0 = denoise(x_noisy, sigma_t, tap=owned)
            a_token, tap_calls = owned.a_token, owned.calls - calls_before
    else:
        calls_before = tap.calls
        bb0 = denoise(x_noisy, sigma_t, tap=tap)
        a_token, tap_calls = tap.a_token, tap.calls - calls_before
    if a_token is None:
        raise RuntimeError(
            "the tap captured no a_token; layernorm_a was never called, so the "
            "residual would be built from nothing"
        )
    if tap_calls != 1:
        raise AssertionError(
            f"the tap saw {tap_calls} denoiser call(s) for the provisional "
            "estimate, expected exactly 1"
        )

    # ---- 2. the complex, through the matrix's own mapping ------------------
    a2t = np.asarray(topology.atom_to_token_idx.cpu()).astype(int)
    res_names = np.asarray(topology.res_names)
    design = check_design_mask(
        np.asarray(structure.design_mask.cpu()),
        res_names=res_names,
        atom_to_token=a2t,
        n_tokens=n_tokens,
        what=design_id,
    )
    inputs = build_design_inputs(
        x0=bb0.reshape(-1, 3),
        a_token=a_token.reshape(n_tokens, -1),
        sigma=float(sigma),
        atom_names=np.asarray(topology.atom_names),
        res_names=res_names,
        atom_to_token=a2t,
        n_tokens=n_tokens,
        design=design,
        residue_index=topology.residue_index,
        asym_id=topology.chain_index,
        design_id=design_id,
        target=target,
        binder_length=int(design.sum()),
        context=context,
        device=device,
    )

    # ---- 3. the A_BS residual, binder rows only ---------------------------
    residual = None
    if adapters is not None:
        residual = build_residual(
            adapters,
            binder_mask=inputs.binder_mask,
            a_token=inputs.a_token,
            sigma=inputs.sigma,
            source=residual_source,
        )
        assert_target_rows_untouched(residual, inputs.binder_mask)

    # ---- 4. design sequence AND side chains on bb0 ------------------------
    with conditioned(designer.model, residual) as hook:
        result = designer.design(
            coords_af2=inputs.coords_af2,
            atom_mask=inputs.atom_mask,
            aatype=inputs.aatype,
            seq_mask=inputs.seq_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
            fixed_sequence_mask=inputs.fixed_sequence_mask,
            sidechain_context_mask=inputs.sidechain_context_mask,
            seed=seed,
        )["designs"][0]
    decode_hook_calls = hook.calls

    aatype = result.aatype.reshape(1, -1).long()
    coords = result.coords_af2.unsqueeze(0)
    produced = result.atom_mask_af2.unsqueeze(0)

    # ---- 5. what the packing actually realized ----------------------------
    # `pxf.couple.visibility` computes this rather than reusing the input's
    # missing_atom_mask, which marks every side-chain slot absent: correct for
    # the first encode, and wrong here, where it would keep the generated atoms
    # masked and leave h_packed == h_base -- a feedback module reading nothing.
    vis = predicted_availability(
        aatype,
        inputs.seq_mask,
        inputs.atom_mask,
        inputs.coords_af2,
        sidechains=coords,
    )

    # ---- 6. re-encode, WITHOUT the A_BS hook ------------------------------
    # The feedback module must read the donor's own view of the realized
    # state. Encoding it through the conditioned path would hand the feedback a
    # representation already shifted by the residual it exists to correct.
    base_features = None
    if want_h_base:
        with torch.no_grad():
            _l, _h, base_features = encode(
                designer.model, inputs.coords_af2, inputs.aatype,
                seq_mask=inputs.seq_mask,
                missing_atom_mask=1.0 - inputs.atom_mask,
                residue_index=inputs.residue_index,
                chain_index=inputs.chain_index,
                sidechain_visible=inputs.sidechain_context_mask.float(),
            )
    _logits, _hv, packed_features = encode(
        designer.model, coords, aatype,
        seq_mask=inputs.seq_mask,
        missing_atom_mask=vis.missing_atom_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        sidechain_visible=vis.sidechain_visible,
    )

    binder = inputs.binder_mask.reshape(-1).bool().cpu()
    binder_sequence = "".join(
        c for c, keep in zip(result.sequence, binder.tolist()) if keep
    )
    return EventProducts(
        bb0=bb0.detach(),
        a_token=a_token,
        residual=residual,
        aatype=aatype,
        sequence=result.sequence,
        binder_sequence=binder_sequence,
        coords_af2=coords,
        atom_mask_af2=produced,
        availability=vis.available,
        h_base=None if base_features is None else base_features["h_V"],
        h_packed=packed_features["h_V"],
        psce=result.psce.unsqueeze(0),
        binder_mask=inputs.binder_mask,
        sigma=float(sigma),
        provenance={
            "design_id": design_id,
            "target": target,
            "context": context,
            "binder_length": int(design.sum()),
            "n_tokens": n_tokens,
            "actual_sigma": float(sigma),
            "residual_source": None if adapters is None else residual_source,
            "decode_hook_calls": decode_hook_calls,
            "tap_calls": int(tap_calls),
            "seed": int(seed),
            "delta_h_norm": (0.0 if residual is None
                             else float(residual.detach().norm(dim=-1).mean())),
            "visibility": vis.record(),
        },
    )


def designer_module(denoise) -> Any:
    """The diffusion module a ``denoise`` callable belongs to.

    ``OfficialDenoiser.denoise`` is a bound method and its module is reachable;
    a test double may pass a plain function carrying ``diffusion_module``.
    Raising here rather than guessing keeps a stub from silently tapping
    nothing.
    """
    owner = getattr(denoise, "__self__", None)
    module = getattr(owner, "model", None)
    module = getattr(module, "diffusion_module", None) if module is not None else None
    if module is None:
        module = getattr(denoise, "diffusion_module", None)
    if module is None:
        raise TypeError(
            "cannot find the diffusion module behind this denoise callable; "
            "the tap has nothing to attach to"
        )
    return module
