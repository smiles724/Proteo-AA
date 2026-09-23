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


def sequence_diagnostics(initial, output, *, policy, seed=None,
                         terminal=None, decode_passes=None):
    """Scalar diagnostics; unchanged sequences are valid, not a failed decode.

    ``initial`` is the FIRST event, ``output`` the sequence actually written.
    ``terminal`` is the last event's provisional products when they differ
    from ``initial`` -- with several events those are different noisy states
    at different sigmas, and the immediate-correction geometry has to come
    from the terminal pair or it is not a correction measurement at all.
    """
    check_sequence_policy(policy)
    reference = terminal if terminal is not None else initial
    binder = initial.binder_mask.bool()
    changed = int(((initial.aatype != output.aatype) & binder).sum())
    count = int(binder.sum())
    redesign = policy == "post_feedback_redesign"
    return {
        "sequence_policy": policy,
        "event_sequence": initial.binder_sequence,
        "output_sequence": output.binder_sequence,
        "redecode_calls": int(redesign),
        # Decodes this arm actually RAN. The old 1-or-2 was the shared
        # first-decode provenance, which under-reports a four-event feedback
        # arm by three.
        "sequence_decode_passes": (
            int(decode_passes) if decode_passes is not None else 1 + int(redesign)
        ),
        "redecode_seed": int(seed) if redesign else None,
        "sequence_changed_positions": changed,
        "sequence_changed_fraction": changed / count if count else 0.0,
        "redecode_hook_calls": output.provenance.get("decode_hook_calls") if redesign else 0,
        "redecode_abs_source": "corrected_a_token" if redesign else None,
        "final_pack_residual_source": "corrected_event" if redesign else "provisional_event",
        # Provisional vs corrected AT THE SAME EVENT, so the same noisy
        # state and the same augmented frame. Taking this between the first
        # event and the last instead folds in the whole intervening
        # trajectory plus its random rotations and translations, which
        # cannot tell you whether any corrective call moved the backbone --
        # the question the experiment exists to answer.
        "immediate_event_coordinate_rmsd": float(
            ((reference.bb0 - output.bb0).square().sum(-1).mean()).sqrt()
        ) if redesign else None,
        "immediate_event_key": (
            list(reference.provenance.get("event_key", ()))
            if redesign else None
        ),
        # First event to final output. A DIFFERENT quantity, named
        # differently: frames differ, so it is a displacement, not a
        # correction.
        "first_to_final_coordinate_displacement": float(
            ((initial.bb0 - output.bb0).square().sum(-1).mean()).sqrt()
        ) if redesign else None,
    }
