"""One cached PXDesign backbone -> the tensors FaMPNN's designer wants.

`scripts/cache_binder_backbones.py` ran in the official runtime and cached the
topology it derived there, because the consumer of the collection does not have
the official featurizer. This reads that payload. It is the ONLY place the
payload schema is interpreted, so an arm cannot quietly disagree with another
arm about which rows are the binder.

### What the payload actually contains

``x0`` is all-atom ``[1, n_atom, 3]`` in the GENERATED pose -- both chains
placed by the model, which is why the target's coordinates here are not the
native ones (audit section 9: 0.107 A superposed, 26 A away). Annotations are
**per atom**; ``design_mask``, ``asym_id`` and ``residue_index`` are per token.
``atom_to_token_idx`` is the bridge.

The design region is identified by PXDesign's own ``res_name == 'xpb'`` marker,
not by chain id or by a length arithmetic, because that predicate *is* the
featurizer's definition of the generated region. It is cross-checked against
``design_mask`` and against ``conditional_label``; those three agreeing is the
evidence that this reading of the payload is right, and they are checked on
every load rather than once in a test.

### Why the design region has only four atom types

PXDesign generates a backbone. Its tokens carry N, CA, C, O (plus a terminal
OXT), and nothing else -- there is no side chain to hide, and no identity: the
residue name is a marker, not an amino acid. So the binder enters as X with
backbone-only occupancy, and that is the input, not a masking choice this
module makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from pxf import atom37

#: PXDesign's marker for a generated (design-region) residue.
DESIGN_RES_NAME = "xpb"

#: The three context levels of `configs/binder_benchmark/arms.yaml`.
CONTEXTS = ("binder_only", "complex", "complex_sc")

CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ATOM37_SLOT = {name: i for i, name in enumerate(atom37.ATOM37)}


@dataclass
class DesignInputs:
    """Everything one arm needs for one backbone. Batch dim included ([1, L, ...])."""

    coords_af2: torch.Tensor          # [1, L, 37, 3]
    atom_mask: torch.Tensor           # [1, L, 37]
    aatype: torch.Tensor              # [1, L]  target identities; X on the binder
    seq_mask: torch.Tensor            # [1, L]
    residue_index: torch.Tensor       # [1, L]
    chain_index: torch.Tensor         # [1, L]
    fixed_sequence_mask: torch.Tensor # [1, L]  1 = identity held (the target)
    sidechain_context_mask: torch.Tensor  # [1, L]  1 = side chains shown
    binder_mask: torch.Tensor         # [1, L]  1 = a designed row
    a_token: torch.Tensor             # [1, L, 768] the tapped representation
    sigma: float                      # the ACTUAL sigma, churn included
    design_id: str
    target: str
    binder_length: int
    context: str

    @property
    def length(self) -> int:
        return int(self.aatype.shape[-1])


def _three_to_index() -> dict[str, int]:
    from fampnn.data import residue_constants as rc

    return {three: atom37.AA_ORDER.index(one)
            for three, one in rc.restype_3to1.items()
            if one in atom37.AA_ORDER}


def load_payload(path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _checked_design_mask(payload: dict[str, Any]) -> np.ndarray:
    """Per-token design mask, with the three independent markers cross-checked.

    ``design_mask`` alone would be enough to *run*. It is not enough to trust:
    if the payload were ever regenerated with a different token ordering, a
    stale mask would silently designate target rows as binder and every arm
    would agree with every other arm about the wrong thing. ``res_name`` and
    ``conditional_label`` come from different parts of the featurizer, so
    requiring all three to coincide is a real check.
    """
    topology = payload["topology"]
    ann = topology["annotations"]
    a2t = np.asarray(topology["atom_to_token_idx"])
    n_tokens = int(topology["n_tokens"])

    mask = np.asarray(topology["design_mask"]).astype(bool)

    by_resname = np.zeros(n_tokens, dtype=bool)
    is_xpb = np.asarray(ann["res_name"]) == DESIGN_RES_NAME
    by_resname[a2t[is_xpb]] = True

    by_label = np.zeros(n_tokens, dtype=bool)
    not_cond = ~np.asarray(ann["conditional_label"]).astype(bool)
    by_label[a2t[not_cond]] = True

    if not (np.array_equal(mask, by_resname) and np.array_equal(mask, by_label)):
        raise ValueError(
            f"{payload.get('design_id')}: the payload's three markers of the "
            f"design region disagree -- design_mask sums to {int(mask.sum())}, "
            f"res_name=='{DESIGN_RES_NAME}' to {int(by_resname.sum())}, "
            f"conditional_label==False to {int(by_label.sum())}. Refusing to "
            "guess which one names the binder."
        )
    expected = int(payload["binder_length"])
    if int(mask.sum()) != expected:
        raise ValueError(
            f"{payload.get('design_id')}: design mask selects {int(mask.sum())} "
            f"tokens but binder_length is {expected}"
        )
    return mask


def to_design_inputs(
    payload: dict[str, Any],
    *,
    context: str,
    device=None,
) -> DesignInputs:
    """Build one arm's inputs at one context level.

    ``context`` is the arm's, and it changes what FaMPNN *sees*, never what is
    designed: the binder rows are designed in all three.
    """
    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}; choose from {CONTEXTS}")

    topology = payload["topology"]
    ann = topology["annotations"]
    n_tokens = int(topology["n_tokens"])
    a2t = np.asarray(topology["atom_to_token_idx"]).astype(int)
    x0 = payload["x0"].reshape(-1, 3).float()
    if x0.shape[0] != a2t.shape[0]:
        raise ValueError(
            f"x0 has {x0.shape[0]} atoms but atom_to_token_idx has {a2t.shape[0]}"
        )

    design = _checked_design_mask(payload)

    # ---- scatter atoms into atom37 ----------------------------------------
    coords = torch.zeros(n_tokens, atom37.NUM_ATOM37, 3, dtype=torch.float32)
    mask = torch.zeros(n_tokens, atom37.NUM_ATOM37, dtype=torch.float32)
    names = np.asarray(ann["atom_name"])
    slots = np.array([_ATOM37_SLOT.get(str(n), -1) for n in names])
    unknown = names[slots < 0]
    if unknown.size:
        raise ValueError(
            f"atom name(s) outside the atom37 table: {sorted(set(map(str, unknown)))}"
        )
    coords[a2t, slots] = x0
    mask[a2t, slots] = 1.0

    # ---- identities --------------------------------------------------------
    three = _three_to_index()
    resname_per_token = np.empty(n_tokens, dtype=object)
    resname_per_token[a2t] = np.asarray(ann["res_name"])
    aatype = torch.full((n_tokens,), atom37.UNKNOWN_AA_INDEX, dtype=torch.long)
    for i, rn in enumerate(resname_per_token):
        if design[i]:
            continue  # generated: no identity exists to read
        idx = three.get(str(rn))
        if idx is not None:
            aatype[i] = idx
        # anything else (UNK, a modified residue) stays X, and because X is
        # never `fixed` below it is designed rather than teacher-forced as a
        # mask token.

    binder = torch.from_numpy(design)
    # A target row can only be held fixed if it actually has an identity.
    fixed = (~binder) & (aatype != atom37.UNKNOWN_AA_INDEX)

    residue_index = topology["residue_index"].long().clone()
    chain_index = topology["asym_id"].long().clone()
    a_token = payload["a_token"].reshape(n_tokens, -1).float()

    # ---- context level -----------------------------------------------------
    if context == "binder_only":
        keep = binder
    else:
        keep = torch.ones(n_tokens, dtype=torch.bool)

    coords, mask = coords[keep], mask[keep]
    aatype, binder, fixed = aatype[keep], binder[keep], fixed[keep]
    residue_index, chain_index = residue_index[keep], chain_index[keep]
    a_token = a_token[keep]

    if context == "complex_sc":
        # Target side chains held as context. Subset of `fixed`, which
        # design() enforces -- a known side chain implies a known identity.
        sc_context = fixed.clone()
    else:
        sc_context = torch.zeros_like(fixed)

    if context == "binder_only":
        # Nothing is held: the target is not in the receptive field at all.
        fixed = torch.zeros_like(fixed)
        sc_context = torch.zeros_like(fixed)
        # Side-chain slots of a binder row are empty anyway, but say so.
        mask[:, list(atom37.SIDECHAIN_SLOTS)] = 0.0
    else:
        # Binder rows carry no side chain in any context.
        sidechain = torch.tensor(atom37.SIDECHAIN_SLOTS, dtype=torch.long)
        mask[binder.nonzero(as_tuple=True)[0][:, None], sidechain[None, :]] = 0.0

    def batched(t):
        t = t[None]
        return t.to(device) if device is not None else t

    return DesignInputs(
        coords_af2=batched(coords),
        atom_mask=batched(mask),
        aatype=batched(aatype),
        seq_mask=batched(torch.ones_like(aatype, dtype=torch.float32)),
        residue_index=batched(residue_index),
        chain_index=batched(chain_index),
        fixed_sequence_mask=batched(fixed.long()),
        sidechain_context_mask=batched(sc_context.long()),
        binder_mask=batched(binder.long()),
        a_token=batched(a_token),
        # The tapped event's sigma INCLUDING churn. The scheduled value is
        # exactly half of it (churn factor 2.0) and conditioning A_BS on that
        # would query the adapter at half the noise the denoiser saw.
        sigma=float(payload["actual_sigma"]),
        design_id=str(payload["design_id"]),
        target=str(payload["target"]),
        binder_length=int(payload["binder_length"]),
        context=context,
    )


# ------------------------------------------------------- chain identity
#
# Four of the ten AlphaProteo targets are MULTI-CHAIN: H1, IL17A and VEGFA
# have two target chains, TNFa has three. So the binder is chain C or D on
# those, not B. An earlier writer mapped "asym_id 0 -> A, everything else ->
# B" and merged the binder into the target's second chain; ProteinMPNN then
# returned 283 residues for a 383-row complex, which is how it surfaced.
# These live here, beside the mask they must agree with, rather than in a
# script -- the AF2-IG sidecars and the R0 PDBs both need them and they must
# not be derived twice.


def chain_letter(asym_id: int) -> str:
    """``asym_id`` -> PDB chain letter."""
    if asym_id >= len(CHAIN_LETTERS) or asym_id < 0:
        raise ValueError(
            f"asym_id {asym_id} is outside the single-letter chain space"
        )
    return CHAIN_LETTERS[asym_id]


def chains_of(inputs: DesignInputs) -> list[str]:
    ids = sorted({int(v) for v in inputs.chain_index[0].tolist()})
    return [chain_letter(i) for i in ids]


def binder_chain_of(inputs: DesignInputs) -> str:
    binder = inputs.binder_mask[0].bool()
    ids = sorted({int(v) for v in inputs.chain_index[0][binder].tolist()})
    if len(ids) != 1:
        raise ValueError(
            f"the binder spans {len(ids)} chains ({ids}); expected exactly one"
        )
    return chain_letter(ids[0])


def target_chains_of(inputs: DesignInputs) -> list[str]:
    binder = binder_chain_of(inputs)
    return [c for c in chains_of(inputs) if c != binder]
