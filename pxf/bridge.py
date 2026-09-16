"""The backbone -> side-chain boundary.

PXDesign emits a flat, ragged atom list -- ``[..., N_atom, 3]`` plus an
``atom_to_token_idx`` map and per-atom names -- while FaMPNN consumes a dense
``[B, L, 37, 3]`` block per residue. Both use the same AF2 atom37 order (see
:mod:`pxf.atom37`), so this densification is the *entire* conversion: no
reordering, and no structure serialized to text on the way across.

Residues PXDesign was asked to design carry its ``xpb`` residue name and only the
``N/CA/C/O/OXT`` slots. Those positions have no native identity, so
:func:`native_sequence` reports them as unknown rather than guessing -- this
pipeline never lets the side-chain module invent a sequence.
"""

import torch

from pxf import atom37

# PXDesign's residue name for a token the generator was asked to design.
DESIGN_RESNAME = "xpb"
_TRASH = atom37.NUM_ATOM37

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _atom37_slots(atom_names):
    """Map per-atom names to atom37 slots; -1 for anything outside the vocabulary."""
    lookup = {name: i for i, name in enumerate(atom37.ATOM37)}
    return torch.tensor(
        [lookup.get(str(name), -1) for name in atom_names], dtype=torch.long
    )


def atoms_to_atom37(
    coords, atom_names, atom_to_token_idx, num_tokens, *, keep=None, strict=False
):
    """Scatter ``[..., N_atom, 3]`` atom coordinates into ``[..., L, 37, 3]``.

    Returns ``(coords37, mask37, dropped)`` where ``mask37`` marks the slots
    actually filled and ``dropped`` lists atom names outside the 37-atom
    vocabulary (rejected instead when ``strict``). Autograd flows through
    ``coords``.
    """
    if coords.shape[-1] != 3:
        raise ValueError(
            f"Expected trailing coordinate axis of 3, got {tuple(coords.shape)}"
        )
    num_atoms = coords.shape[-2]
    if len(atom_names) != num_atoms:
        raise ValueError(f"{len(atom_names)} atom names for {num_atoms} atoms")
    token = (
        torch.as_tensor(atom_to_token_idx, dtype=torch.long).reshape(-1).to(coords.device)
    )
    if token.numel() != num_atoms:
        raise ValueError(
            f"atom_to_token_idx has {token.numel()} entries for {num_atoms} atoms"
        )
    if num_tokens <= 0 or int(token.max()) >= num_tokens:
        raise ValueError(f"atom_to_token_idx exceeds num_tokens={num_tokens}")

    slot = _atom37_slots(atom_names).to(coords.device)
    valid = slot >= 0
    if keep is not None:
        valid = valid & torch.as_tensor(keep, dtype=torch.bool).reshape(-1).to(
            coords.device
        )
    dropped = sorted({str(n) for n, ok in zip(atom_names, (slot >= 0).tolist()) if not ok})
    if strict and dropped:
        raise ValueError(f"Atoms outside the atom37 vocabulary: {dropped[:12]}")

    flat = torch.where(
        valid, token * (atom37.NUM_ATOM37 + 1) + slot, torch.full_like(slot, _TRASH)
    )
    width = num_tokens * (atom37.NUM_ATOM37 + 1)
    lead = coords.shape[:-2]

    counts = torch.zeros(width, device=coords.device, dtype=torch.long)
    counts.scatter_add_(0, flat, valid.long())
    grid = counts.reshape(num_tokens, atom37.NUM_ATOM37 + 1)[:, : atom37.NUM_ATOM37]
    if (grid > 1).any():
        token_idx, slot_idx = (grid > 1).nonzero(as_tuple=True)
        raise ValueError(
            "Duplicate atom for a residue slot: "
            + ", ".join(
                f"token {int(t)} / {atom37.ATOM37[int(s)]}"
                for t, s in list(zip(token_idx, slot_idx))[:6]
            )
        )

    clean = torch.where(valid[..., None], coords, coords.new_zeros(()))
    out = coords.new_zeros(*lead, width, 3)
    out.scatter_add_(-2, flat[..., None].expand(*lead, num_atoms, 3), clean)
    coords37 = out.reshape(*lead, num_tokens, atom37.NUM_ATOM37 + 1, 3)[
        ..., : atom37.NUM_ATOM37, :
    ]
    mask37 = grid.bool().reshape(num_tokens, atom37.NUM_ATOM37)
    mask37 = mask37.expand(*lead, num_tokens, atom37.NUM_ATOM37) if lead else mask37
    return coords37, mask37, dropped


def token_reduce(per_atom, atom_to_token_idx, num_tokens, *, how="first"):
    """Collapse a per-atom sequence to one value per token.

    ``how="first"`` takes each token's first atom, which is what per-residue
    annotations need; ``how="any"`` ORs booleans, which is what design flags need.

    ``per_atom`` is a Python sequence, so the tensors built from it follow
    ``atom_to_token_idx``'s device rather than defaulting to the CPU: once the
    topology travels with the batch, a CPU scatter index against a CUDA topology
    is a device mismatch.
    """
    token = torch.as_tensor(atom_to_token_idx, dtype=torch.long).reshape(-1)
    device = token.device
    if how == "any":
        # Counting and then testing > 0 is the same as an OR, and scatter_add_
        # on int64 is implemented on every backend. scatter_reduce_(amax) on a
        # bool tensor is not: it works on the CPU and raises
        # `"cuda_scatter_gather_base_kernel_func" not implemented for 'Bool'`
        # on CUDA, which no CPU-only test run can reach.
        flags = torch.as_tensor(
            [bool(x) for x in per_atom], dtype=torch.bool, device=device
        )
        counts = torch.zeros(num_tokens, dtype=torch.long, device=device)
        counts.scatter_add_(0, token, flags.long())
        return counts > 0
    if how != "first":
        raise ValueError(f"Unknown reduction {how!r}")
    seen, values = {}, [None] * num_tokens
    for atom, tok in enumerate(token.tolist()):
        if tok not in seen:
            seen[tok] = atom
            values[tok] = per_atom[atom]
    if any(v is None for v in values):
        missing = [i for i, v in enumerate(values) if v is None]
        raise ValueError(f"Tokens with no atoms: {missing[:12]}")
    return values


def design_mask_from_res_names(res_names, atom_to_token_idx, num_tokens):
    """Boolean ``[L]`` marking tokens PXDesign was asked to design."""
    flags = [str(name) == DESIGN_RESNAME for name in res_names]
    return token_reduce(flags, atom_to_token_idx, num_tokens, how="any")


def native_sequence(res_names, atom_to_token_idx, num_tokens):
    """Native one-letter sequence per token, plus a mask of where it is known.

    Returns ``(sequence, known)``. Positions without a canonical native identity
    -- PXDesign design tokens, and any non-standard residue -- appear as ``"X"``
    with ``known=False``. The caller must supply real identities there before the
    side-chain module will run; nothing here fills them in.
    """
    per_token = token_reduce(list(res_names), atom_to_token_idx, num_tokens, how="first")
    letters, known = [], []
    for name in per_token:
        letter = THREE_TO_ONE.get(str(name).upper())
        letters.append(letter if letter is not None else "X")
        known.append(letter is not None)
    device = getattr(atom_to_token_idx, "device", None)
    return "".join(letters), torch.tensor(known, dtype=torch.bool, device=device)


def apply_sequence_overrides(sequence, known, overrides):
    """Fill unknown positions from ``overrides``; returns the completed sequence.

    ``overrides`` maps a zero-based token index to a one-letter residue, or is a
    string of the full length whose letters are used wherever ``known`` is False.
    """
    letters = list(sequence)
    if isinstance(overrides, str):
        if len(overrides) != len(letters):
            raise ValueError(
                f"Override sequence has length {len(overrides)}, expected {len(letters)}"
            )
        supplied = {i: overrides[i] for i in range(len(letters)) if not bool(known[i])}
    else:
        supplied = {int(k): v for k, v in dict(overrides).items()}
    for index, letter in supplied.items():
        if not 0 <= index < len(letters):
            raise ValueError(f"Override position {index} outside 0..{len(letters) - 1}")
        if letter not in atom37.AA_ORDER:
            raise ValueError(
                f"Override residue {letter!r} at position {index} is not canonical"
            )
        letters[index] = letter
    completed = "".join(letters)
    missing = [i for i, letter in enumerate(completed) if letter not in atom37.AA_ORDER]
    if missing:
        raise ValueError(
            f"{len(missing)} position(s) still have no canonical residue identity "
            f"(first few: {missing[:12]}). FaMPNN packs a *given* sequence and will not "
            "design one; supply these via --sequence/--sequence-fasta."
        )
    return completed
