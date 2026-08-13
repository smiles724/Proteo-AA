"""Side-chain lDDT (Mariani et al. 2013), hard-threshold, for reporting.

WHY, given we already report RMSD. RMSD is the wrong instrument for packing and
for the same reason the coordinate loss is: it scores every atom independently
against its own target, so two side chains can each look good and still occupy
the same space. It is also dominated by lever-arm error -- a few degrees of chi1
moves ARG's NH1 a long way, and that single atom then sets the number.

lDDT is superposition-free and local: it asks whether the DISTANCES from each
side-chain atom to its neighbourhood are reproduced. Clashes and mis-packing show
up directly, and a terminal atom that swings out is penalised in proportion to
how many of its contacts it breaks rather than by its displacement.

NOT the same object as `SmoothLDDTLoss`. That is a differentiable surrogate used
as a training term (and in the side-chain stage its weight is zero and it is
computed on the frozen BACKBONE coordinates, so the `val_lddt` in those logs says
nothing about side chains). This is the hard-threshold score you report.

DEFINITION, following the paper:
  * pairs (i, j) with i a SIDE-CHAIN atom, taken from the REFERENCE structure
    with d_ref < inclusion_radius (15 A);
  * pairs within the same residue are excluded -- they are fixed by the residue's
    own chemistry and would inflate the score;
  * a pair is preserved at threshold t if |d_model - d_ref| < t;
  * the score is the fraction preserved, averaged over t in {0.5, 1, 2, 4} A.

Two conventions are computed because they answer different questions and the
difference is not small:
  * `sc_env`  -- j ranges over the side-chain atoms AND the backbone, i.e. the
    side chain scored against its environment. Closest to how CASP reports it,
    and the one that reflects packing.
  * `sc_sc`   -- j ranges over side-chain atoms only. Measures side-chain-to-
    side-chain agreement without the backbone, which is identical between
    prediction and reference here and therefore makes the score look better.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_INCLUSION_RADIUS = 15.0


def lddt_score(
    pred: torch.Tensor,              # [N, 3] model coordinates of the SUBJECT atoms
    ref: torch.Tensor,               # [N, 3] reference coordinates of the same atoms
    partner_pred: torch.Tensor,      # [M, 3] model coordinates of the partner set
    partner_ref: torch.Tensor,       # [M, 3] reference coordinates of the partner set
    subject_residue: torch.Tensor,   # [N] long, residue index of each subject atom
    partner_residue: torch.Tensor,   # [M] long, residue index of each partner atom
    inclusion_radius: float = DEFAULT_INCLUSION_RADIUS,
    thresholds: tuple = DEFAULT_THRESHOLDS,
) -> tuple[float, int]:
    """Return (lDDT in [0, 1], number of pairs scored).

    The subject and partner sets may overlap; self-pairs and same-residue pairs
    are removed by the residue test, so passing the side-chain atoms as both is
    the `sc_sc` convention and costs nothing extra.

    Returns (nan, 0) when no pair falls inside the inclusion radius -- a single
    isolated residue, say. The caller must not average a nan in silently.
    """
    if pred.numel() == 0 or partner_pred.numel() == 0:
        return float("nan"), 0

    d_ref = torch.cdist(ref, partner_ref)              # [N, M]
    d_mod = torch.cdist(pred, partner_pred)

    # Same-residue pairs are excluded: their distances are set by the residue's
    # own covalent geometry, which the model is not being tested on, and they
    # would dominate the count for long side chains.
    diff_res = subject_residue[:, None] != partner_residue[None, :]
    valid = (d_ref < inclusion_radius) & diff_res
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        return float("nan"), 0

    delta = (d_mod - d_ref).abs()
    preserved = sum(((delta < t) & valid).sum() for t in thresholds)
    # No epsilon in the denominator: n_pairs == 0 already returned above, and a
    # guard that cannot fire would only stop a perfect prediction from scoring
    # exactly 1.0.
    return float(preserved) / (len(thresholds) * n_pairs), n_pairs


def sidechain_lddt(
    sc_pred: torch.Tensor,           # [L, A, 3] predicted side-chain atoms, GLOBAL
    sc_ref: torch.Tensor,            # [L, A, 3] reference side-chain atoms, GLOBAL
    sc_mask: torch.Tensor,           # [L, A] bool, which slots are real and scored
    bb_coords: Optional[torch.Tensor] = None,   # [L, K, 3] backbone, GLOBAL
    bb_mask: Optional[torch.Tensor] = None,     # [L, K] bool
    inclusion_radius: float = DEFAULT_INCLUSION_RADIUS,
    thresholds: tuple = DEFAULT_THRESHOLDS,
) -> dict:
    """Both conventions for one structure. See the module docstring.

    The backbone is IDENTICAL in prediction and reference here -- S_phi does not
    move it -- so backbone-to-backbone pairs would be preserved by construction
    and are never included; the backbone enters only as a partner for side-chain
    atoms, which is exactly the environment term that makes `sc_env` meaningful.
    """
    L, A = sc_mask.shape
    res_idx = torch.arange(L, device=sc_pred.device)[:, None].expand(L, A)

    m = sc_mask.reshape(-1).bool()
    p = sc_pred.reshape(-1, 3)[m]
    r = sc_ref.reshape(-1, 3)[m]
    ri = res_idx.reshape(-1)[m]

    out = {}
    out["lddt_sc_sc"], out["n_pairs_sc_sc"] = lddt_score(
        p, r, p, r, ri, ri, inclusion_radius, thresholds
    )

    if bb_coords is not None:
        K = bb_coords.shape[1]
        bm = (torch.ones(L, K, dtype=torch.bool, device=sc_pred.device)
              if bb_mask is None else bb_mask.bool())
        b_res = torch.arange(L, device=sc_pred.device)[:, None].expand(L, K)
        bflat = bb_coords.reshape(-1, 3)[bm.reshape(-1)]
        bri = b_res.reshape(-1)[bm.reshape(-1)]
        partner_p = torch.cat([p, bflat], 0)
        partner_r = torch.cat([r, bflat], 0)     # backbone is shared, by construction
        partner_i = torch.cat([ri, bri], 0)
        out["lddt_sc_env"], out["n_pairs_sc_env"] = lddt_score(
            p, r, partner_p, partner_r, ri, partner_i, inclusion_radius, thresholds
        )
    else:
        out["lddt_sc_env"], out["n_pairs_sc_env"] = float("nan"), 0
    return out
