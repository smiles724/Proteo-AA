"""Optional downstream sequence update after the single backbone correction.

This changes the output policy, not the upstream event E1 was trained to read.
Every arm, including the no-feedback controls, receives the same extra decode.
The original shared event is never overwritten. No second feedback is applied.
"""

from __future__ import annotations

import torch

SEQUENCE_POLICIES = ("event_fixed", "post_feedback_redesign")
REDECODE_SEED_OFFSET = 1_000_003


def check_sequence_policy(policy):
    if policy not in SEQUENCE_POLICIES:
        raise ValueError(f"unknown sequence policy {policy!r}; use {SEQUENCE_POLICIES}")
    return policy


@torch.no_grad()
def redecode_after_feedback(
    *, initial, corrected_bb, corrected_a_token, sigma, structure,
    designer, adapters, seed, context, design_id, target,
):
    """Design on bb1 using its fresh a_token at the original event sigma.

    The shared first-pass sequence and SC are not input to this decode. Only
    target identities/context are fixed. Caller must protect the backbone RNG.
    """
    from pxf.couple.integrated_event import design_from_estimate

    if corrected_a_token is None:
        raise RuntimeError("post-feedback re-decoding requires the corrected a_token")
    actual_sigma = float(sigma.reshape(-1)[0])
    if abs(actual_sigma - initial.sigma) > 1e-6:
        raise ValueError("re-decoding must use the same actual event sigma")
    products = design_from_estimate(
        bb0=corrected_bb.detach(), a_token=corrected_a_token.detach(),
        sigma=actual_sigma, structure=structure, designer=designer,
        adapters=adapters, context=context, seed=int(seed),
        design_id=design_id, target=target, want_h_base=False,
    )
    if not torch.equal(products.binder_mask, initial.binder_mask):
        raise ValueError("re-decoding changed the binder/target role mapping")
    targets = ~initial.binder_mask.bool()
    if not torch.equal(products.aatype[targets], initial.aatype[targets]):
        raise ValueError("re-decoding changed a fixed target identity")
    return products


def sequence_diagnostics(initial, output, *, policy, seed=None):
    """Scalar diagnostics; unchanged sequences are valid, not a failed decode."""
    check_sequence_policy(policy)
    binder = initial.binder_mask.bool()
    changed = int(((initial.aatype != output.aatype) & binder).sum())
    count = int(binder.sum())
    redesign = policy == "post_feedback_redesign"
    return {
        "sequence_policy": policy,
        "event_sequence": initial.binder_sequence,
        "output_sequence": output.binder_sequence,
        "redecode_calls": int(redesign),
        "sequence_decode_passes": 1 + int(redesign),
        "redecode_seed": int(seed) if redesign else None,
        "sequence_changed_positions": changed,
        "sequence_changed_fraction": changed / count if count else 0.0,
        "redecode_hook_calls": output.provenance.get("decode_hook_calls") if redesign else 0,
        "redecode_abs_source": "corrected_a_token" if redesign else None,
        "final_pack_residual_source": "corrected_event" if redesign else "provisional_event",
        "immediate_event_coordinate_rmsd": float(
            ((initial.bb0 - output.bb0).square().sum(-1).mean()).sqrt()
        ) if redesign else None,
    }
