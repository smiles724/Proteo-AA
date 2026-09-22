"""The declared chemistry guardrail metrics, in one place.

`configs/integrated_feedback/selection.yaml` names two guardrails, and a
checkpoint cannot be selected while either is unavailable -- the selector
refuses rather than treating a missing metric as a pass. So both live here and
are used by the acceptance gate and the evaluator alike: two implementations
of "what counts as a chemistry failure" would eventually disagree, and the one
that mattered would be whichever the selector happened to call.

These are SCREENS with declared thresholds, not a chemistry validation. They
exist to stop a correction buying backbone RMSD by breaking geometry, which is
the failure mode the guardrail was written for.
"""

from __future__ import annotations

import torch

from pxf import atom37

#: Consecutive CA-CA distance for a cis/trans peptide bond. Outside this a
#: residue is not bonded to its neighbour.
CA_CA_RANGE = (3.4, 4.4)
#: Non-adjacent backbone atoms this close are interpenetrating.
BACKBONE_CLASH = 2.0
#: Side-chain atoms in different residues this close are clashing.
SIDECHAIN_CLASH = 2.0


def backbone_chemistry(coords37, binder_tokens) -> dict:
    """CA-CA spacing and non-adjacent backbone clashes, on the binder.

    ``coords37`` is ``[L, 37, 3]`` or ``[1, L, 37, 3]``.
    """
    dense = coords37.reshape(-1, atom37.NUM_ATOM37, 3)
    rows = binder_tokens.reshape(-1).bool()
    if int(rows.sum()) < 3:
        return {"failed": None, "reason": "fewer than 3 binder residues"}

    ca = dense[rows, atom37.ATOM37.index("CA")]
    spacing = (ca[1:] - ca[:-1]).norm(dim=-1)
    bad_spacing = int(
        ((spacing < CA_CA_RANGE[0]) | (spacing > CA_CA_RANGE[1])).sum()
    )

    backbone = dense[rows][:, list(atom37.BACKBONE_SLOTS)].reshape(-1, 3)
    distance = torch.cdist(backbone.float(), backbone.float())
    index = torch.arange(backbone.shape[0], device=distance.device)
    # Four slots per residue, so //4 recovers the residue. Same or adjacent
    # residues are bonded and must not count as clashes.
    adjacent = (index[:, None] // 4 - index[None, :] // 4).abs() <= 1
    clashes = int((distance.masked_fill(adjacent, float("inf"))
                   < BACKBONE_CLASH).sum() // 2)
    return {
        "bad_ca_spacing": bad_spacing,
        "backbone_clashes": clashes,
        "failed": bool(bad_spacing or clashes),
        "ca_ca_range": list(CA_CA_RANGE),
        "clash_threshold": BACKBONE_CLASH,
    }


def sidechain_chemistry(coords37, occupancy, binder_tokens) -> dict:
    """Non-bonded side-chain clashes on the binder, from the REALIZED packing.

    ``occupancy`` is the packer's own output mask, not the input backbone's: an
    all-ones mask would count atoms the packer never built.
    """
    dense = coords37.reshape(-1, atom37.NUM_ATOM37, 3)
    present = occupancy.reshape(-1, atom37.NUM_ATOM37)
    rows = binder_tokens.reshape(-1).bool()

    atoms, residue_of = [], []
    for index in torch.nonzero(rows).reshape(-1).tolist():
        for slot in atom37.SIDECHAIN_SLOTS:
            if present[index, slot] > 0:
                atoms.append(dense[index, slot])
                residue_of.append(index)
    if not atoms:
        return {"failed": None, "reason": "no side-chain atoms were built",
                "atoms": 0}
    stacked = torch.stack(atoms).float()
    residue = torch.tensor(residue_of, device=stacked.device)
    distance = torch.cdist(stacked, stacked)
    distance = distance.masked_fill(
        residue[:, None] == residue[None, :], float("inf")
    )
    clashes = int((distance < SIDECHAIN_CLASH).sum() // 2)
    return {
        "atoms": len(atoms),
        "nonbonded_clashes": clashes,
        "failed": bool(clashes),
        "clash_threshold": SIDECHAIN_CLASH,
    }
