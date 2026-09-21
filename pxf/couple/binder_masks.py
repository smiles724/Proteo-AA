"""Binder-role masks, and the leakage checks that make them trustworthy.

Four different masks get called "the mask" in this pipeline and conflating any
two of them produces a trained adapter that looks fine and learned from the
answer:

  ``seq_mask``          which rows are real residues at all. Target AND binder.
                        This is ENCODER CONTEXT and must never be zeroed to
                        exclude the target from the loss -- doing that removes
                        the target from the model's receptive field, which is
                        a different experiment.
  ``seq_mlm_mask``      which identities the encoder can SEE. Upstream's
                        convention, and it is the opposite of what the name
                        suggests: 1 means *kept*, and
                        ``sequence_mlm_loss`` scores ``(1 - mask) * seq_mask``.
  ``sidechain_visible`` which rows contribute side-chain COORDINATES.
  supervision masks     which rows a loss is computed over.

Encoder availability and supervision are not the same set and are not derived
from each other here. The target is visible to the encoder and excluded from
both losses; a masked binder residue is hidden from the encoder and supervised.

### The leak this is built to prevent

The sequence head predicts binder identities. If any input reveals them the
adapter learns to read that channel instead of the structure, and the result
is a designability number that cannot be reproduced at inference. There are
three routes and all three are checked:

1. **FaMPNN aatype** -- the encoder's residue types at supervised positions
   must be UNKNOWN, not the native identity.
2. **FaMPNN side chains** -- a masked residue's side-chain coordinates imply
   its identity almost exactly. Hiding the letter while showing the atoms
   leaks it.
3. **PXDesign conditioning** -- ``a_token`` is computed upstream. If the
   featurizer put native binder residue types into the backbone conditioning,
   the residual carries the answer even though FaMPNN never saw it. This is
   the subtle one, because the sequence model's own inputs look clean.

:func:`audit_leakage` returns a finding per route rather than a boolean, so a
failure says which channel leaked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from pxf import atom37
from pxf.couple.binder_residual import ChainRoles


@dataclass(frozen=True)
class BinderMaskSet:
    """Every mask one training example needs, kept distinct on purpose."""

    roles: ChainRoles
    seq_mask: torch.Tensor           # [B, L] 1 = real residue (target + binder)
    seq_mlm_mask: torch.Tensor       # [B, L] 1 = identity VISIBLE to the encoder
    sidechain_visible: torch.Tensor  # [B, L] 1 = side-chain coords visible
    aatype_encoder: torch.Tensor     # [B, L] what the encoder is told
    aatype_true: torch.Tensor        # [B, L] labels; never an encoder input

    @property
    def seq_supervision(self) -> torch.Tensor:
        """Rows scored by ``L_seq``: hidden, real, and (asserted) binder-only."""
        return (1.0 - self.seq_mlm_mask) * self.seq_mask

    @property
    def sc_supervision(self) -> torch.Tensor:
        """Rows eligible for ``L_SC``.

        All binder rows, not only the sequence-masked ones. The side-chain
        objective is defined on the design region and restricting it to hidden
        identities would quietly drop supervision the existing loss already
        provides.
        """
        binder = self.roles.binder.to(self.seq_mask.device).reshape(1, -1)
        return binder.to(self.seq_mask.dtype) * self.seq_mask

    def identity(self) -> dict[str, Any]:
        return {
            **self.roles.identity(),
            "n_seq_supervised": int(self.seq_supervision.sum()),
            "n_sc_supervised": int(self.sc_supervision.sum()),
            "n_sidechain_visible": int(self.sidechain_visible.sum()),
            "mask_fraction_of_binder": (
                float(self.seq_supervision.sum()) / max(self.roles.n_binder, 1)
            ),
        }


def build_masks(
    roles: ChainRoles,
    aatype_true: torch.Tensor,
    *,
    mask_fraction: float = 0.5,
    target_sidechain_dropout: float = 0.0,
    binder_sidechain_dropout: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> BinderMaskSet:
    """Mask a fraction of BINDER identities; leave the target intact and visible.

    ``binder_sidechain_dropout`` defaults to 1.0 -- every binder side chain
    hidden -- because a binder side chain is a near-perfect readout of the
    residue the sequence head is being asked to predict. Lowering it is a
    deliberate experiment, not a tuning knob.

    ``target_sidechain_dropout`` is the §5 context dropout on the target's
    resolved side chains, and it defaults to 0.0 so the default training
    context matches the `complex_sc` evaluation arm.
    """
    if aatype_true.dim() == 1:
        aatype_true = aatype_true[None]
    batch, length = aatype_true.shape
    if length != roles.length:
        raise ValueError(
            f"aatype has {length} rows, roles describe {roles.length}"
        )
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError(f"mask_fraction must be in [0, 1], got {mask_fraction}")

    device = aatype_true.device
    binder = roles.binder.to(device).reshape(1, -1).expand(batch, length)
    target = ~binder

    seq_mask = torch.ones(batch, length, device=device)

    # 1 = kept. The target is ALWAYS kept: it is fixed context, and marking it
    # kept is what excludes it from `sequence_mlm_loss` without touching
    # seq_mask. Zeroing seq_mask instead would remove it from the encoder.
    draw = torch.rand(batch, length, device=device, generator=generator)
    hide_binder = binder & (draw < mask_fraction)
    seq_mlm_mask = torch.ones(batch, length, device=device)
    seq_mlm_mask[hide_binder] = 0.0

    # Identities the encoder is told. Hidden binder rows become X.
    aatype_encoder = aatype_true.clone()
    aatype_encoder[hide_binder] = atom37.UNKNOWN_AA_INDEX

    sc_draw = torch.rand(batch, length, device=device, generator=generator)
    sidechain_visible = torch.ones(batch, length, device=device)
    sidechain_visible[binder & (sc_draw < binder_sidechain_dropout)] = 0.0
    sidechain_visible[target & (sc_draw < target_sidechain_dropout)] = 0.0
    # A residue whose identity is hidden must never show its side chain,
    # whatever the dropout draw did.
    sidechain_visible[hide_binder] = 0.0

    return BinderMaskSet(
        roles=roles,
        seq_mask=seq_mask,
        seq_mlm_mask=seq_mlm_mask,
        sidechain_visible=sidechain_visible,
        aatype_encoder=aatype_encoder,
        aatype_true=aatype_true,
    )


# ----------------------------------------------------------------- the audit


def audit_leakage(
    masks: BinderMaskSet,
    *,
    pxdesign_aatype: Optional[torch.Tensor] = None,
    pxdesign_design_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """One finding per leakage route. Empty ``problems`` is the only pass.

    ``pxdesign_aatype`` is the residue-type channel the BACKBONE conditioning
    saw. Supply it and route 3 is checked; omit it and route 3 is reported as
    unchecked rather than silently passed.
    """
    problems: list[str] = []
    findings: dict[str, Any] = {}
    device = masks.seq_mask.device
    binder = masks.roles.binder.to(device).reshape(1, -1)
    target = masks.roles.target.to(device).reshape(1, -1)
    supervised = masks.seq_supervision.bool()

    # --- structural: the masks mean what they are supposed to mean ----------
    if bool((supervised & target).any()):
        problems.append(
            f"{int((supervised & target).sum())} TARGET row(s) are scored by "
            "L_seq; the target is fixed context and must be excluded by the "
            "supervision mask"
        )
    if bool((masks.seq_mask == 0).any()):
        problems.append(
            "seq_mask has zeros: the target must stay in the encoder context "
            "and be excluded from the loss by seq_mlm_mask, not by seq_mask"
        )
    findings["n_supervised"] = int(supervised.sum())
    if int(supervised.sum()) == 0:
        problems.append(
            "no residue is supervised by L_seq; the sequence objective would "
            "be identically zero and the arm would silently reduce to SC-only"
        )

    # --- route 1: identities in the encoder's aatype ------------------------
    revealed = supervised & (masks.aatype_encoder != atom37.UNKNOWN_AA_INDEX)
    findings["identity_revealed"] = int(revealed.sum())
    if bool(revealed.any()):
        problems.append(
            f"ROUTE 1 (FaMPNN aatype): {int(revealed.sum())} supervised "
            "position(s) carry their native identity in the encoder input"
        )

    # --- route 2: side chains of supervised residues ------------------------
    sc_leak = supervised & masks.sidechain_visible.bool()
    findings["sidechain_revealed"] = int(sc_leak.sum())
    if bool(sc_leak.any()):
        problems.append(
            f"ROUTE 2 (FaMPNN side chains): {int(sc_leak.sum())} supervised "
            "position(s) show side-chain coordinates, which identify the "
            "residue almost exactly"
        )

    # --- route 3: PXDesign conditioning -------------------------------------
    if pxdesign_aatype is None:
        findings["pxdesign_checked"] = False
        findings["pxdesign_note"] = (
            "not checked: pass the residue-type channel the backbone "
            "conditioning saw. a_token can carry the answer even when FaMPNN's "
            "own inputs are clean."
        )
    else:
        findings["pxdesign_checked"] = True
        px = pxdesign_aatype.reshape(1, -1).to(device)
        # The featurizer scrubs the design region; anything that survives as a
        # native identity on a supervised row is a leak into a_token.
        px_leak = supervised & (px == masks.aatype_true.to(device)) & (
            px != atom37.UNKNOWN_AA_INDEX
        )
        findings["pxdesign_revealed"] = int(px_leak.sum())
        if bool(px_leak.any()):
            problems.append(
                f"ROUTE 3 (PXDesign conditioning): {int(px_leak.sum())} "
                "supervised position(s) keep their native residue type in the "
                "backbone conditioning, so a_token carries the answer"
            )
        if pxdesign_design_mask is not None:
            dm = pxdesign_design_mask.reshape(1, -1).to(device).bool()
            mismatch = int((dm != binder).sum())
            findings["design_mask_vs_binder_mismatch"] = mismatch
            if mismatch:
                problems.append(
                    f"PXDesign's design region and the binder role disagree on "
                    f"{mismatch} row(s); the residual and the supervision would "
                    "be computed over different residues"
                )

    findings["problems"] = problems
    findings["pass"] = not problems
    return findings


def pxdesign_restype_leak(
    restype: torch.Tensor,
    design_mask: torch.Tensor,
    aatype_true: torch.Tensor,
) -> dict[str, Any]:
    """Does PXDesign's `restype` channel carry the binder's identities?

    Vocabulary-independent by construction, which is why it is done this way
    rather than by comparing indices: `restype` is a 36-way encoding and
    `aa_clean` is a 21-way one, so an index comparison needs a mapping nobody
    has written down and would be wrong quietly.

    The featurizer's contract is that the design region is scrubbed to a single
    placeholder identity. If it holds, every design-region row of `restype` is
    the SAME vector and the channel carries no information about the sequence
    the head is being asked to predict. If more than one distinct row appears,
    the rows are varying with something -- and the first thing to check is
    whether they vary with the native identity.
    """
    design = design_mask.reshape(-1).bool()
    rows = restype.reshape(restype.shape[0], -1)[design]
    if rows.numel() == 0:
        return {"checked": False, "reason": "no design rows"}

    distinct = torch.unique(rows, dim=0)
    n_distinct = int(distinct.shape[0])
    out: dict[str, Any] = {
        "checked": True,
        "n_design_rows": int(rows.shape[0]),
        "n_distinct_restype_rows": n_distinct,
        "scrubbed": n_distinct == 1,
    }
    if n_distinct == 1:
        out["leak"] = False
        return out

    # More than one placeholder. Quantify whether the variation tracks the
    # native sequence: if distinct restype rows map one-to-one onto distinct
    # native identities, the channel is the answer in another encoding.
    native = aatype_true.reshape(-1)[design]
    n_native = int(torch.unique(native).numel())
    pairs = {
        (tuple(r.tolist()), int(a)) for r, a in zip(rows, native)
    }
    rows_per_identity = len({p[0] for p in pairs})
    out.update({
        "n_distinct_native_identities": n_native,
        "n_distinct_rows_seen": rows_per_identity,
        # A perfect correspondence is the damning case.
        "leak": bool(n_distinct >= n_native > 1 and len(pairs) == n_native),
    })
    return out
