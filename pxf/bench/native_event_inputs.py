"""Prepare a deposited complex so its binder looks the way inference does.

At inference the binder is a GENERATED backbone: four atoms per residue and no
identity (PXDesign marks it ``xpb``). A deposited complex is not like that --
its binder chain carries full side chains and real residue names. Featurizing
one directly and calling the result a reconstruction event gives the denoiser
the binder's native side-chain coordinates at the event's noise level, which at
sigma 0.429 is its deposited geometry with 0.43 A of jitter on it.

That is a leak, and the acceptance gate caught it: perturbing native binder
side chains moved the provisional backbone. The fix is not to relabel the
perturbation as legitimate -- it is to stop the coordinates being an input.

So the binder chain is reduced to N, CA, C, O BEFORE featurization, by
explicit atom-name selection on the structure file. Everything downstream then
sees the same binder representation inference does, and the native side chains
are neither an input nor a target (the loss is on backbone slots).

The target chain is untouched: its sequence and resolved side chains are
legitimate context, which is what ``complex_sc`` means.

### Why a temporary file rather than a tensor mask

Masking after featurization would leave the coordinates in the feature dict
and in ``backbone_target``, so every later consumer would have to remember to
mask them the same way. Selecting atoms at the file level makes the absence
structural: there is nothing to forget to mask.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: The atoms a generated binder residue has. OXT is terminal-only and absent
#: from PXDesign's generated region, so it is dropped too.
BINDER_BACKBONE = ("N", "CA", "C", "O")

PERTURBATIONS = ("none", "aatype", "sidechain", "both")


def prepare(
    cif_path,
    binder_chain: str,
    out_dir,
    *,
    perturb: str = "none",
    identity_shift: int = 7,
    sidechain_displacement: float = 17.0,
) -> dict:
    """Write a prepared CIF and report what was done to it.

    ``perturb`` corrupts NATIVE labels before any featurization, which is what
    makes the leakage test a measurement: the labels still exist at this point,
    so a path that reads them will show it.

    ``aatype`` rotates every binder residue NAME. ``sidechain`` displaces every
    binder side-chain atom before they are stripped -- if the stripping works,
    that perturbation must be invisible downstream, and if it does not, the
    gate fails.
    """
    import gemmi

    if perturb not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturb!r}")

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()

    from fampnn.data import residue_constants as rc

    canonical = [rc.restype_1to3[c] for c in "ARNDCQEGHILKMFPSTWYV"]
    stats = {
        "binder_chain": binder_chain,
        "perturb": perturb,
        "binder_residues": 0,
        "sidechain_atoms_removed": 0,
        "residues_renamed": 0,
        "sidechain_atoms_displaced": 0,
        "target_chains_untouched": [],
    }

    for model in structure:
        for chain in model:
            if chain.name != binder_chain:
                stats["target_chains_untouched"].append(chain.name)
                continue
            for residue in chain:
                stats["binder_residues"] += 1
                if perturb in ("sidechain", "both"):
                    for atom in residue:
                        if atom.name not in BINDER_BACKBONE:
                            atom.pos = gemmi.Position(
                                atom.pos.x + sidechain_displacement,
                                atom.pos.y + sidechain_displacement,
                                atom.pos.z + sidechain_displacement,
                            )
                            stats["sidechain_atoms_displaced"] += 1
                if perturb in ("aatype", "both") and residue.name in canonical:
                    index = canonical.index(residue.name)
                    residue.name = canonical[
                        (index + identity_shift) % len(canonical)
                    ]
                    stats["residues_renamed"] += 1
                # EXPLICIT binder backbone selection. This is the fix: the
                # binder contributes four atoms per residue, exactly as a
                # generated one does.
                #
                # Deleted by INDEX in reverse. Collecting the keepers into a
                # list and re-adding them aliases into the residue being
                # emptied, which segfaults the allocator (std::bad_alloc).
                for index in range(len(residue) - 1, -1, -1):
                    if residue[index].name not in BINDER_BACKBONE:
                        del residue[index]
                        stats["sidechain_atoms_removed"] += 1

    stats["target_chains_untouched"] = sorted(
        set(stats["target_chains_untouched"])
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cif_path).stem
    out_path = out_dir / f"{stem}__{perturb}.cif"
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(out_path))
    with open(out_path, "rb") as handle:
        stats["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    stats["path"] = str(out_path)
    if stats["binder_residues"] == 0:
        raise ValueError(
            f"{cif_path}: chain {binder_chain!r} has no residues; the binder "
            "selection would be empty"
        )
    return stats
