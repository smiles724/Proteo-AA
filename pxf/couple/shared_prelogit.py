"""Condition BOTH outputs on one residual: the `shared_prelogit` application mode.

The legacy mode (`pxf.couple.pack_hook`, `application_mode: packing_only`)
substitutes ``h_V`` inside side-chain diffusion, which upstream calls *after*
the sequence logits for that pass are already computed. That is fine for
packing and it is why the existing A_BS trains at all -- but it means the
same-pass sequence loss has no gradient path into the adapter. Sequence can
only be affected indirectly, through side-chain context feeding a later
iterative step.

This module is the other mode. One residual is added to ``h_V`` once, and both
heads read the conditioned representation:

    delta_h = M_B * A_BS(a_token, sigma_B)
    h'      = h_V + delta_h
    logits  = W_out(h')                       <- sequence, now differentiable
    q0_hat  = D_SC(q_t, t_SC, h', s_GT)       <- side chains, as before

`W_out` is the pretrained head (`fampnn/model/fampnn.py:65`, applied at line
154 as ``logits = self.W_out(h_V)``); no new AA head is introduced and nothing
in the submodule is edited.

### Three invariants, all enforced rather than documented

**No double injection.** Using this mode together with the packing hook would
add the residual twice on the side-chain side -- once here, once inside
``sidechain_diffusion`` -- while the sequence side saw it only once. The two
modes are mutually exclusive and `application_mode` records which one a
checkpoint was trained under.

**No accumulation across decode steps.** The residual is constant within a
design, and a FaMPNN decode calls the encoder ~101 times. Adding ``delta`` to
a feature dict that is itself reused would compound it linearly, so every call
derives ``h'`` from the *original* ``h_V`` and writes into a shallow copy.
:func:`condition` never mutates its input, and
:func:`assert_not_accumulating` makes that checkable from a test.

**Target rows take exactly zero.** Inherited from
:func:`pxf.couple.binder_residual.binder_masked_residual`, which verifies it
bit-exactly rather than trusting the multiply.

### On detaching a_token

The schematic detaches: gradients train the adapter, not PXDesign, which is
frozen. Detaching is the default here and `detach_a_token=False` is available
for a later experiment that wants the backbone to feel the sequence loss --
but that is a different study and it must be declared, because leaving it on
silently would make the "frozen donor" claim false.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from pxf.couple.binder_residual import ChainRoles, binder_masked_residual

APPLICATION_MODES = ("packing_only", "shared_prelogit")


def sequence_head(seq_module):
    """The pretrained logits head, checked to be usable.

    `no_aatype_pred` turns the head off upstream; a model configured that way
    returns ``logits=None`` and conditioning it would silently produce a
    sequence objective with no output to train.
    """
    if getattr(seq_module, "no_aatype_pred", False):
        raise ValueError(
            "this FaMPNN is configured with no_aatype_pred=True: it has no "
            "sequence head, so the shared pre-logit path has nothing to "
            "condition"
        )
    head = getattr(seq_module, "W_out", None)
    if head is None:
        raise AttributeError(
            "the sequence-design module exposes no W_out; the shared pre-logit "
            "path reads the pretrained head and will not invent one"
        )
    return head


def condition(
    seq_module,
    features: dict[str, Any],
    delta: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """``(logits, conditioned_features)`` from one residual.

    Returns the ORIGINAL feature dict unchanged when ``delta`` is None, so the
    uncoupled arm takes the same path rather than an equivalent one, and
    recomputes logits only when there is something to recompute.
    """
    if "h_V" not in features:
        raise KeyError(f"no h_V in the feature dict; keys are {sorted(features)}")
    if delta is None:
        return None, features

    h_v = features["h_V"]
    if delta.shape[-1] != h_v.shape[-1]:
        raise ValueError(
            f"residual width {delta.shape[-1]} != h_V width {h_v.shape[-1]}"
        )
    if delta.shape[-2] != h_v.shape[-2]:
        raise ValueError(
            f"residual length {delta.shape[-2]} != h_V length {h_v.shape[-2]}"
        )

    head = sequence_head(seq_module)
    # Shallow copy, and h' derived from the original h_V every time. An
    # in-place add here would compound the residual across the ~101 encoder
    # calls of one decode.
    conditioned = dict(features)
    conditioned["h_V"] = h_v + delta.to(h_v.dtype).to(h_v.device)
    return head(conditioned["h_V"]), conditioned


def conditioned_forward(
    seq_module,
    features: dict[str, Any],
    *,
    adapters,
    a_token: torch.Tensor,
    sigma_b,
    roles: ChainRoles,
    source: str = "matched",
    gate=None,
    mean=None,
    detach_a_token: bool = True,
):
    """Build the binder-masked residual and condition both heads with it.

    The one entry point training and inference share. They must share it: a
    second implementation of "where the residual goes" is how a train/inference
    skew gets in, and this mode exists precisely because the injection site
    determines which objective can reach the adapter.
    """
    token = a_token.detach() if detach_a_token else a_token
    delta = binder_masked_residual(
        adapters, source, roles=roles, a_token=token, sigma=sigma_b,
        mean=mean, gate=gate,
    )
    logits, conditioned = condition(seq_module, features, delta)
    return logits, conditioned, delta


def assert_not_accumulating(features: dict[str, Any], original_h_v: torch.Tensor) -> None:
    """The caller's feature dict must still hold the unconditioned ``h_V``.

    Cheap to call once per decode in a test or a paranoid run. It catches the
    failure this module is most likely to grow later: someone "optimises" the
    shallow copy away, the residual starts compounding, and the symptom is a
    coupled arm that drifts further from the uncoupled one the longer the
    decode runs -- which looks exactly like a real effect.
    """
    if features["h_V"] is not original_h_v and not torch.equal(
        features["h_V"], original_h_v
    ):
        raise AssertionError(
            "the feature dict handed to condition() was mutated; the residual "
            "will compound across decode steps"
        )


def check_exclusive(application_mode: str, packing_hook_active: bool) -> None:
    """Refuse to run both application modes at once."""
    if application_mode not in APPLICATION_MODES:
        raise ValueError(
            f"unknown application_mode {application_mode!r}; "
            f"choose from {APPLICATION_MODES}"
        )
    if application_mode == "shared_prelogit" and packing_hook_active:
        raise AssertionError(
            "shared_prelogit is active AND the legacy packing hook is "
            "installed. The residual would be added twice on the side-chain "
            "side and once on the sequence side, which is neither mode."
        )
