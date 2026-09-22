"""Turn a benchmark target into a Proteo-AA design input.

The benchmark asks for de-novo binder design: target structure in, a binder of a
given length out. Every training provider in this repo instead starts from a real
complex and *masks* one chain, so there is nothing here that can build "target +
L design residues" on its own. This module fills that gap without inventing a
second featurization path, because a second path is exactly how the leakage bug
described in `tests/test_data_contract_parity.py` happened.

The route is therefore: write a prepared mmCIF holding the cropped target plus a
placeholder poly-glycine binder chain, then hand it to the repo's own
`CifFileProvider` -> `DesignSourceDataset(inference_safe_binder=True)`. The
placeholder gets marked `[xpb]`, rebuilt to exactly N/CA/C/O with
residue-type-independent reference metadata, and excluded from
`conditional_templ` — i.e. it goes through the same contract the model is trained
under, and `cogenerate` replaces its coordinates with noise (`x = sigma_0 *
randn`, so no ground-truth coordinate is read at all).

Three things about the prepared file are worth knowing before reading the code:

* **`dataset="Distillation"`, not `"WeightedPDB"`.** The WeightedPDB parser runs
  the PDB-curation filters, and two of them destroy a prepared input: the
  assembly builder needs `pdbx_struct_assembly` records that describe a real
  deposition, and `remove_dissociation` deletes the placeholder chain outright
  (it is not in contact with anything, because it is not a real binder). The
  distillation parser skips assembly expansion and those filters while still
  tokenising polymers per residue, which is what a prepared single-model input
  needs.

* **Polymer entities must be declared.** A structure file with only `atom_site`
  is parsed as a bag of ligands and tokenised *per atom*, which silently turns a
  116-residue target into 930 tokens. `write_prepared_cif` therefore emits
  `entity` / `entity_poly` / `entity_poly_seq` / `struct_asym` alongside
  `atom_site`.

* **Residue numbering is sequential in the prepared file, author-based in the
  manifest.** The parser reads `label_seq_id` into `res_id`, so author numbers
  (which is how published hotspots are quoted) do not survive it. Prep resolves
  author -> sequential once, while it still has both, and records the mapping on
  the `PreparedInput`. Nothing downstream re-derives it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Ideal alpha-helix parameters for the placeholder backbone. These are geometry
# for a chain that is never scored and never seen by the model: its only jobs are
# to exist as a polymer of the right length, to survive parsing (sane bond
# lengths, consecutive CA ~3.8 A), and to sit next to the binding site so that a
# spatial crop keeps the interface. Do not read anything biophysical into them.
_HELIX_RADIUS = 2.27
_HELIX_RISE = 1.50
_HELIX_TWIST_DEG = 100.0
# How far outside the hotspot centroid the placeholder axis is placed. Far enough
# not to clash into the target, close enough that `DesignCropper`'s
# proximity ranking keeps the binding site when a target does need cropping.
_PLACEHOLDER_STANDOFF = 12.0

BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class PreparedInput:
    """A prepared benchmark input plus everything needed to interpret it."""

    task_id: str
    target_name: str
    cif_path: Path
    binder_chain_id: str
    binder_length: int
    # Number of tokens the featurized input must have: one per target residue
    # plus one per binder residue. The loader asserts against this, because a
    # silent crop would change the task.
    n_expected_tokens: int
    n_target_residues: int
    # Hotspots in the PREPARED file's numbering, i.e. (chain_id, label_seq_id).
    hotspots_prepared: tuple[tuple[str, int], ...]
    # Hotspots as published, i.e. (chain_id, author residue number).
    hotspots_author: tuple[tuple[str, int], ...]
    # prepared chain -> {sequential resid: author resid}, so designs can be
    # written back out in the numbering the target is normally discussed in.
    author_numbering: dict[str, dict[int, int]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_name": self.target_name,
            "cif_path": str(self.cif_path),
            "binder_chain_id": self.binder_chain_id,
            "binder_length": self.binder_length,
            "n_expected_tokens": self.n_expected_tokens,
            "n_target_residues": self.n_target_residues,
            "hotspots_prepared": [list(h) for h in self.hotspots_prepared],
            "hotspots_author": [list(h) for h in self.hotspots_author],
            "author_numbering": {
                chain: {str(k): v for k, v in mapping.items()}
                for chain, mapping in self.author_numbering.items()
            },
        }

    def write_sidecar(self) -> Path:
        path = self.cif_path.with_suffix(".prep.json")
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        return path


# ----------------------------------------------------------------- target crop


def load_cropped_target(cif_path: Path | str, crop_ranges: Sequence[tuple[str, int, int]]):
    """Read a deposited mmCIF and keep only the benchmark's target residues.

    `crop_ranges` are `(author chain id, first, last)` triples, inclusive, in the
    numbering the manifest quotes. Hydrogens are dropped (Protenix drops them too;
    keeping them only inflates the file) and so is everything that is not an
    amino acid — waters, ions and the glycans some of these entries carry. Note
    `filter_amino_acids` keeps modified residues such as MSE, so a crop does not
    silently acquire a gap where the deposition has a selenomethionine.
    """
    import biotite.structure as struc
    from biotite.structure.io.pdbx import CIFFile, get_structure

    path = Path(cif_path)
    if not path.is_file():
        raise FileNotFoundError(f"target structure not found: {path}")
    array = get_structure(
        CIFFile.read(str(path)), model=1, use_author_fields=True
    )
    array = array[struc.filter_amino_acids(array) & (array.element != "H")]

    keep = np.zeros(array.array_length(), dtype=bool)
    for chain_id, first, last in crop_ranges:
        selected = (
            (array.chain_id == chain_id)
            & (array.res_id >= int(first))
            & (array.res_id <= int(last))
        )
        if not selected.any():
            raise ValueError(
                f"{path.name}: crop {chain_id}{first}-{last} selected no amino "
                f"acids; author chains present: {sorted(set(array.chain_id))}"
            )
        keep |= selected
    return array[keep]


def residue_keys(array) -> list[tuple[str, int]]:
    """Ordered unique (chain_id, res_id) pairs, in file order."""
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chain_id, res_id in zip(array.chain_id, array.res_id):
        key = (str(chain_id), int(res_id))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def resolve_hotspots(array, hotspots: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    """Check every published hotspot survived the crop; return it unchanged.

    A hotspot that is missing here would be dropped from the conditioning and the
    run would quietly be a different, easier task, so this raises rather than
    warns.
    """
    present = set(residue_keys(array))
    resolved, missing = [], []
    for chain_id, resid in hotspots:
        key = (str(chain_id), int(resid))
        (resolved if key in present else missing).append(key)
    if missing:
        raise ValueError(
            f"hotspot residues absent from the cropped target: {missing}. Either "
            "the crop range or the hotspot numbering in the manifest is wrong "
            "(both are author numbering)."
        )
    return tuple(resolved)


# ---------------------------------------------------------- placeholder binder


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `normal`."""
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    # Cross with the basis vector least aligned to `normal`, so the cross product
    # is never near-degenerate.
    least = np.zeros(3)
    least[int(np.argmin(np.abs(normal)))] = 1.0
    axis = np.cross(normal, least)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return axis, np.cross(normal, axis)


def make_placeholder_binder(
    target,
    length: int,
    chain_id: str,
    anchor: Optional[np.ndarray] = None,
):
    """Build a poly-glycine placeholder chain of `length` residues.

    The chain is an ideal alpha-helix laid out tangentially to the target surface
    at `anchor` (the hotspot centroid) and standing `_PLACEHOLDER_STANDOFF` off
    it. Coordinates are inert — see the module docstring — so the only properties
    that matter are that parsing accepts it and that it sits by the binding site.
    """
    import biotite.structure as struc

    if length < 1:
        raise ValueError(f"binder_length must be >= 1, got {length}")

    target_centroid = target.coord.mean(axis=0)
    anchor = target_centroid if anchor is None else np.asarray(anchor, dtype=float)
    outward = anchor - target_centroid
    if float(np.linalg.norm(outward)) < 1e-6:
        outward = np.array([1.0, 0.0, 0.0])
    outward = outward / float(np.linalg.norm(outward))
    axis, radial = _orthonormal_basis(outward)

    origin = anchor + _PLACEHOLDER_STANDOFF * outward - (
        0.5 * length * _HELIX_RISE
    ) * axis

    twist = np.deg2rad(_HELIX_TWIST_DEG)
    ca = np.stack(
        [
            origin
            + i * _HELIX_RISE * axis
            + _HELIX_RADIUS * (np.cos(i * twist) * outward + np.sin(i * twist) * radial)
            for i in range(length)
        ]
    )

    array = struc.AtomArray(length * len(BACKBONE_ATOMS))
    for i in range(length):
        # Backbone atoms are placed by stepping towards the neighbouring CAs,
        # which gives ~1.5 A N-CA / CA-C without needing real internal
        # coordinates. The terminal residues borrow the one neighbour they have.
        prev_dir = ca[i - 1] - ca[i] if i > 0 else ca[i] - ca[i + 1] if length > 1 else axis
        next_dir = ca[i + 1] - ca[i] if i + 1 < length else ca[i] - ca[i - 1] if length > 1 else -axis
        prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), 1e-8)
        next_dir = next_dir / max(float(np.linalg.norm(next_dir)), 1e-8)
        carbonyl = np.cross(next_dir, outward)
        carbonyl /= max(float(np.linalg.norm(carbonyl)), 1e-8)
        coords = {
            "N": ca[i] + 1.46 * prev_dir,
            "CA": ca[i],
            "C": ca[i] + 1.52 * next_dir,
            "O": ca[i] + 1.52 * next_dir + 1.23 * carbonyl,
        }
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            k = i * len(BACKBONE_ATOMS) + j
            array.coord[k] = coords[atom_name]
            array.chain_id[k] = chain_id
            array.res_id[k] = i + 1
            array.res_name[k] = "GLY"
            array.atom_name[k] = atom_name
            array.element[k] = atom_name[0]
            array.hetero[k] = False
    return array


def choose_binder_chain_id(taken: Iterable[str]) -> str:
    """First unused single-letter chain id, preferring the repo's usual 'B'."""
    used = {str(c) for c in taken}
    for candidate in ["B", "Z", "Y", "X", "W"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]:
        if candidate not in used:
            return candidate
    raise ValueError(f"no single-letter chain id left; target uses {sorted(used)}")


# --------------------------------------------------------------- mmCIF writing


def write_prepared_cif(array, path: Path | str, entry_id: str = "PREP") -> Path:
    """Write `array` as an mmCIF Protenix's distillation parser can tokenise.

    Emits the polymer-entity categories (without them everything is parsed as a
    ligand and tokenised per atom) and renumbers each chain's residues to a
    contiguous `label_seq_id` starting at 1, so the file agrees with its own
    `entity_poly_seq`. `auth_seq_id` is set to the same sequential value on
    purpose: the parser reads `label_seq_id`, and leaving a second, different
    numbering in the file would just be a trap for whoever opens it next. The
    author numbers live in the `PreparedInput` sidecar.
    """
    from biotite.structure.info import one_letter_code
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    chains = list(dict.fromkeys(str(c) for c in array.chain_id))
    entities: list[dict[str, str]] = []
    poly_seq: list[tuple[str, str, str]] = []
    atom_site: dict[str, list[str]] = {
        key: []
        for key in (
            "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
            "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
            "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", "auth_comp_id",
            "auth_atom_id", "pdbx_PDB_model_num",
        )
    }

    atom_id = 0
    for entity_index, chain_id in enumerate(chains, start=1):
        sub = array[array.chain_id == chain_id]
        keys = residue_keys(sub)
        sequential = {key: i + 1 for i, key in enumerate(keys)}
        names = []
        for key in keys:
            match = sub[(sub.chain_id == key[0]) & (sub.res_id == key[1])]
            names.append(str(match.res_name[0]))
        entities.append(
            {
                "id": str(entity_index),
                "chain": chain_id,
                "seq": "".join(one_letter_code(n) or "X" for n in names),
            }
        )
        for num, res_name in enumerate(names, start=1):
            poly_seq.append((str(entity_index), str(num), res_name))

        for i in range(sub.array_length()):
            atom_id += 1
            seq_id = sequential[(str(sub.chain_id[i]), int(sub.res_id[i]))]
            atom_site["group_PDB"].append("ATOM")
            atom_site["id"].append(str(atom_id))
            atom_site["type_symbol"].append(str(sub.element[i]) or "C")
            atom_site["label_atom_id"].append(str(sub.atom_name[i]))
            atom_site["label_alt_id"].append(".")
            atom_site["label_comp_id"].append(str(sub.res_name[i]))
            atom_site["label_asym_id"].append(chain_id)
            atom_site["label_entity_id"].append(str(entity_index))
            atom_site["label_seq_id"].append(str(seq_id))
            atom_site["pdbx_PDB_ins_code"].append("?")
            for axis, value in zip("xyz", sub.coord[i]):
                atom_site[f"Cartn_{axis}"].append(f"{float(value):.3f}")
            atom_site["occupancy"].append("1.00")
            atom_site["B_iso_or_equiv"].append("0.00")
            atom_site["auth_seq_id"].append(str(seq_id))
            atom_site["auth_asym_id"].append(chain_id)
            atom_site["auth_comp_id"].append(str(sub.res_name[i]))
            atom_site["auth_atom_id"].append(str(sub.atom_name[i]))
            atom_site["pdbx_PDB_model_num"].append("1")

    n = len(entities)
    block = CIFBlock()
    block["entry"] = CIFCategory({"id": entry_id})
    block["entity"] = CIFCategory(
        {
            "id": [e["id"] for e in entities],
            "type": ["polymer"] * n,
            "pdbx_description": [f"chain {e['chain']}" for e in entities],
        }
    )
    block["entity_poly"] = CIFCategory(
        {
            "entity_id": [e["id"] for e in entities],
            "type": ["polypeptide(L)"] * n,
            "nstd_linkage": ["no"] * n,
            "nstd_monomer": ["no"] * n,
            "pdbx_seq_one_letter_code": [e["seq"] for e in entities],
            "pdbx_seq_one_letter_code_can": [e["seq"] for e in entities],
            "pdbx_strand_id": [e["chain"] for e in entities],
            "pdbx_target_identifier": ["?"] * n,
        }
    )
    block["entity_poly_seq"] = CIFCategory(
        {
            "entity_id": [row[0] for row in poly_seq],
            "num": [row[1] for row in poly_seq],
            "mon_id": [row[2] for row in poly_seq],
            "hetero": ["n"] * len(poly_seq),
        }
    )
    block["struct_asym"] = CIFCategory(
        {
            "id": [e["chain"] for e in entities],
            "entity_id": [e["id"] for e in entities],
            "pdbx_blank_PDB_chainid_flag": ["N"] * n,
            "pdbx_modified": ["N"] * n,
            "details": ["?"] * n,
        }
    )
    block["atom_site"] = CIFCategory(atom_site)

    cif = CIFFile()
    cif[entry_id] = block
    cif.write(str(path))
    return path


# ------------------------------------------------------------------- top level


def prepare_task(
    task,
    mmcif_dir: Path | str,
    output_dir: Path | str,
    overwrite: bool = False,
) -> PreparedInput:
    """Materialise one `BinderDesignTask` as a prepared mmCIF + sidecar."""
    target = task.target
    target.require_runnable()

    mmcif_dir = Path(mmcif_dir)
    candidates = [
        mmcif_dir / f"{target.pdb_id}.cif",
        mmcif_dir / f"{target.pdb_id}.cif.gz",
        mmcif_dir / target.pdb_id[1:3] / f"{target.pdb_id}.cif.gz",
    ]
    source = next((c for c in candidates if c.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            f"{target.name}: no structure for PDB {target.pdb_id.upper()} under "
            f"{mmcif_dir} (tried {[str(c) for c in candidates]})"
        )

    cropped = load_cropped_target(source, target.crop_ranges())
    hotspots_author = resolve_hotspots(cropped, target.hotspots or ())

    keys = residue_keys(cropped)
    per_chain_author: dict[str, dict[int, int]] = {}
    per_chain_counter: dict[str, int] = {}
    for chain_id, author_resid in keys:
        index = per_chain_counter.get(chain_id, 0) + 1
        per_chain_counter[chain_id] = index
        per_chain_author.setdefault(chain_id, {})[index] = author_resid

    # Hotspots must be expressed in the prepared file's per-chain sequential
    # numbering, since that is what the parser will report as `res_id`.
    author_to_sequential = {
        (chain_id, author): index
        for chain_id, mapping in per_chain_author.items()
        for index, author in mapping.items()
    }
    hotspots_prepared = tuple(
        (chain_id, author_to_sequential[(chain_id, resid)])
        for chain_id, resid in hotspots_author
    )

    anchor = (
        np.stack(
            [
                cropped[(cropped.chain_id == c) & (cropped.res_id == r)].coord.mean(axis=0)
                for c, r in hotspots_author
            ]
        ).mean(axis=0)
        if hotspots_author
        else None
    )
    binder_chain_id = choose_binder_chain_id(set(cropped.chain_id.tolist()))
    placeholder = make_placeholder_binder(
        cropped, task.binder_length, binder_chain_id, anchor=anchor
    )

    output_dir = Path(output_dir)
    cif_path = output_dir / f"{task.task_id}.cif"
    if overwrite or not cif_path.is_file():
        write_prepared_cif(cropped + placeholder, cif_path, entry_id=task.task_id)

    prepared = PreparedInput(
        task_id=task.task_id,
        target_name=target.name,
        cif_path=cif_path,
        binder_chain_id=binder_chain_id,
        binder_length=task.binder_length,
        n_expected_tokens=len(keys) + task.binder_length,
        n_target_residues=len(keys),
        hotspots_prepared=hotspots_prepared,
        hotspots_author=hotspots_author,
        author_numbering=per_chain_author,
    )
    prepared.write_sidecar()
    return prepared


# ----------------------------------------------------------- featurized inputs


def build_token_index(atom_array) -> dict[tuple[str, int], int]:
    """(chain_id, res_id) -> token index, from the distogram representative atoms.

    Token order is file order, so this is the map the `hotspot` channel is indexed
    by. It is only valid for an *uncropped* featurization, which is why
    `featurize_prepared_input` refuses to proceed when the crop fired.
    """
    representative = atom_array.distogram_rep_atom_mask.astype(bool)
    keys = zip(
        atom_array.chain_id[representative].tolist(),
        atom_array.res_id[representative].tolist(),
    )
    return {(str(c), int(r)): i for i, (c, r) in enumerate(keys)}


def apply_hotspots(
    feature_dict: dict[str, Any],
    token_index: dict[tuple[str, int], int],
    hotspots: Sequence[tuple[str, int]],
) -> int:
    """Overwrite the `hotspot` channel with the benchmark's published hotspots.

    Training *samples* hotspots from binder contacts; the benchmark conditions on
    a fixed published set, and the placeholder binder has no meaningful contacts
    to sample from anyway. Callers must build the dataset with
    `hotspot_force_zero_prob=1.0` so the sampled channel is deterministically
    empty before this replaces it — otherwise the two sets would union and the
    run would condition on hotspots nobody chose.
    """
    import torch

    hotspot = feature_dict["hotspot"]
    if float(hotspot.sum()) != 0.0:
        raise ValueError(
            "the hotspot channel is non-empty before the override; build the "
            "dataset with hotspot_force_zero_prob=1.0 so sampled hotspots cannot "
            "leak into the benchmark's fixed set"
        )
    updated = torch.zeros_like(hotspot)
    for chain_id, resid in hotspots:
        key = (str(chain_id), int(resid))
        if key not in token_index:
            raise KeyError(
                f"hotspot {chain_id}{resid} has no token in the featurized input"
            )
        updated[token_index[key]] = 1.0
    feature_dict["hotspot"] = updated
    return len(hotspots)


def make_design_dataset(
    prepared: PreparedInput,
    crop_size: int = 640,
    compute_sidechain: bool = False,
    seed: int = 0,
):
    """Prepared mmCIF -> `(DesignSourceDataset, token_index)`.

    Split out from `featurize_prepared_input` because `PXDesignTrainer` needs a
    dataset instance at construction time, and building a second one with
    different settings is how train/eval inputs drift apart.
    """
    from pxdesign_train.runner import DesignSourceDataset
    from pxdesign_train.runner.cif_provider import CifFileProvider

    provider = CifFileProvider(
        cif_paths=[str(prepared.cif_path)],
        binder_chain_ids=[prepared.binder_chain_id],
        # See the module docstring: WeightedPDB's curation filters delete the
        # placeholder chain and demand deposition-style assembly records.
        dataset="Distillation",
    )
    atom_array = provider[0][0]
    token_index = build_token_index(atom_array)

    dataset = DesignSourceDataset(
        provider=provider,
        source_name=f"cbdb_{prepared.target_name}",
        crop_size=int(crop_size),
        # `apply_hotspots` installs the published set; force the sampled channel
        # to empty here so the two cannot mix.
        hotspot_force_zero_prob=1.0,
        aa_mask_mode="all",
        aa_mask_prob=1.0,
        aa_mask_min_prob=1.0,
        aa_mask_max_prob=1.0,
        compute_sidechain=compute_sidechain,
        backbone_only_binder=True,
        inference_safe_binder=True,
        # Evaluation wants a deterministic input, not a re-randomised reference
        # frame on every pass.
        ref_pos_augment=False,
        max_crop_retries=1,
        seed=int(seed),
    )
    return dataset, token_index


def featurize_prepared_input(
    prepared: PreparedInput,
    crop_size: int = 640,
    compute_sidechain: bool = False,
    seed: int = 0,
    dataset=None,
    token_index: Optional[dict[tuple[str, int], int]] = None,
) -> dict[str, Any]:
    """Prepared mmCIF -> the featurized dict `cogenerate` consumes.

    Returns the `DesignSourceDataset` item with the `hotspot` channel replaced by
    the benchmark's published hotspots, plus the `PreparedInput` under
    `"prepared"` for downstream bookkeeping. Pass a `dataset`/`token_index` pair
    from `make_design_dataset` to avoid re-parsing the mmCIF per sample.
    """
    if dataset is None or token_index is None:
        dataset, token_index = make_design_dataset(
            prepared,
            crop_size=crop_size,
            compute_sidechain=compute_sidechain,
            seed=seed,
        )
    item = dataset[0]
    feature_dict = item["input_feature_dict"]

    n_token = int(feature_dict["design_token_mask"].numel())
    if n_token != prepared.n_expected_tokens:
        raise ValueError(
            f"{prepared.task_id}: featurized input has {n_token} tokens but the "
            f"prepared target+binder has {prepared.n_expected_tokens}. The crop "
            f"fired (crop_size={crop_size}), which would change the benchmark "
            "task; raise --crop-size instead."
        )
    n_design = int(feature_dict["design_token_mask"].sum())
    if n_design != prepared.binder_length:
        raise ValueError(
            f"{prepared.task_id}: {n_design} design tokens but binder length is "
            f"{prepared.binder_length}"
        )

    apply_hotspots(feature_dict, token_index, prepared.hotspots_prepared)
    item["prepared"] = prepared
    return item
