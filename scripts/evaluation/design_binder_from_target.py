#!/usr/bin/env python3
"""Design a binder de novo against a target structure.

Every existing inference path in this repo starts from a complex that already
contains the binder -- training scrubs a real chain and asks the model to
rebuild it. De novo design has no such chain: the input is a target, an epitope,
and a length. This builds the missing half.

    python scripts/evaluation/design_binder_from_target.py \
        --target benchmarks/alphaproteo10/targets/pdl1.yaml \
        --checkpoint <ckpt.pt> --out designs/

NUMBERING. Target configs carry AUTHOR numbering, because that is what
AlphaProteo's Table S1 and PXDesign's technical report publish and staying in it
keeps a config checkable against the paper it came from. Protenix renumbers on
parse: 5o45 chain A is auth 17-145 and becomes res_id 1-129. The two conventions
differ by a per-chain offset, and PXDesign's own PDL1 example is written in the
*parsed* one -- its `crop: ["1-116"]` and `hotspots: [40, 99, 107]` are exactly
this file's auth 17-132 / [56, 115, 123] after conversion, which is the check
that the mapping here is right. Reading a config in the wrong convention selects
the wrong residues and reports no error.
"""
from __future__ import annotations

import argparse
import logging
import os
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[2]
for p in (HERE, HERE / "Protenix", HERE / "PXDesign"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logger = logging.getLogger(__name__)

# The design-token residue name. The featurizer rewrites binder residues to this
# and reduces them to four backbone atoms before featurisation, so a fabricated
# binder does not need real identities -- only a valid backbone shape.
XPB = "xpb"
BACKBONE = ("N", "CA", "C", "O")
# Keep output IDs compatible with PDB-based evaluators such as PXDesignBench.
# Z is reserved for the (single) binder throughout this evaluation pipeline.
TARGET_CHAIN_IDS = tuple(string.ascii_uppercase.replace("Z", ""))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_target_config(path: str | Path) -> dict:
    """Read a target yaml into {file, chains: {id: (crops, hotspots)}, length}."""
    cfg = yaml.safe_load(Path(path).read_text())
    target = cfg["target"]
    chains = {}
    for chain_id, props in (target["chains"] or {}).items():
        props = props or {}
        crops = []
        for span in props.get("crop", []) or []:
            lo, hi = (int(x) for x in str(span).split("-"))
            crops.append((lo, hi))
        chains[str(chain_id)] = (crops, [int(h) for h in props.get("hotspots", []) or []])
    # Paths in the configs are written relative to the repo root, which is where
    # they are meant to be read from and where the download script puts the
    # structures. Resolving against the config's own directory instead would
    # work only when run from one particular place.
    raw = target["file"]
    structure = raw if os.path.isabs(raw) else str(HERE / raw.lstrip("./"))
    return {
        "file": structure,
        "chains": chains,
        "binder_length": int(cfg["binder_length"]),
        "name": Path(path).stem,
    }


# --------------------------------------------------------------------------
# target
# --------------------------------------------------------------------------

def _set_string_annotation(atom_array: Any, name: str, value: str) -> None:
    """Replace a string annotation without inheriting a too-short dtype."""
    atom_array.set_annotation(
        name,
        np.full(atom_array.array_length(), value, dtype=f"<U{max(1, len(value))}"),
    )


def _select_and_normalize_target_chains(
    atoms: Any, chains: dict,
) -> tuple[Any, np.ndarray]:
    """Select one source chain per config entry and assign canonical IDs.

    ``DataPipeline.get_data_from_mmcif()`` returns a biological assembly.  An
    author chain can consequently appear more than once: for example, 5vli's
    author chain A becomes structural chains A, A.1 and A.2.  Selecting only by
    ``auth_asym_id`` silently includes every assembly copy.  The source
    ``label_asym_id`` identifies the unsuffixed/source copy: after assembly
    expansion its first structural ``chain_id`` is still exactly that label,
    while additional copies receive suffixes.  For these ten inputs the first
    assembly operation is the identity operation.

    Configs remain in published author numbering.  Selected output chains are
    canonicalised to A, B, C, ... in YAML order.  This preserves PXDesign's
    deterministic chain ordering while deliberately using one-character IDs
    for PXDesignBench.  Z is reserved for the binder.
    """
    required = {"auth_asym_id", "label_asym_id", "mol_type", "auth_seq_id"}
    missing = required - set(atoms.get_annotation_categories())
    if missing:
        raise ValueError(f"structure is missing required annotations: {sorted(missing)}")
    if len(chains) > len(TARGET_CHAIN_IDS):
        raise ValueError(
            f"at most {len(TARGET_CHAIN_IDS)} target chains are supported; "
            "chain Z is reserved for the binder"
        )

    raw = np.asarray(atoms.auth_seq_id, dtype=str)
    auth = np.full(raw.shape, -1, dtype=np.int64)
    numeric = np.char.isdigit(np.char.lstrip(raw, "-"))
    auth[numeric] = raw[numeric].astype(np.int64)

    index_pieces = []
    hotspot_pieces = []
    output_id_pieces = []
    for output_chain, (author_chain, (crops, hotspots)) in zip(
        TARGET_CHAIN_IDS, chains.items()
    ):
        author_match = (
            (np.asarray(atoms.auth_asym_id) == author_chain)
            & (np.asarray(atoms.mol_type) == "protein")
        )
        if not author_match.any():
            available = sorted({
                str(value)
                for value in np.asarray(atoms.auth_asym_id)[
                    np.asarray(atoms.mol_type) == "protein"
                ]
            })
            raise ValueError(
                f"chain {author_chain} has no protein atoms "
                f"(structure has author chains {available})"
            )

        # The unsuffixed/source copy has chain_id == source label_asym_id.
        # Copies created by assembly operators retain label_asym_id but are
        # renamed A.1, A.2, ... by Protenix.
        primary = author_match & (
            np.asarray(atoms.chain_id) == np.asarray(atoms.label_asym_id)
        )
        primary_ids = sorted({
            str(value) for value in np.asarray(atoms.chain_id)[primary]
        })
        if len(primary_ids) != 1:
            candidates = sorted({
                str(value) for value in np.asarray(atoms.chain_id)[author_match]
            })
            raise ValueError(
                f"author chain {author_chain} does not resolve to exactly one "
                f"unsuffixed structural chain (primary={primary_ids}, all={candidates})"
            )
        source_chain = primary_ids[0]
        on_chain = primary & (np.asarray(atoms.chain_id) == source_chain)

        selected = on_chain.copy() if not crops else np.zeros(len(atoms), dtype=bool)
        for lo, hi in crops:
            span = on_chain & (auth >= lo) & (auth <= hi)
            if not span.any():
                raise ValueError(
                    f"chain {author_chain} crop {lo}-{hi} selected nothing; the "
                    f"source chain spans {auth[on_chain].min()}-{auth[on_chain].max()} "
                    "in author numbering"
                )
            selected |= span

        chain_hotspot = np.zeros(len(atoms), dtype=bool)
        for residue in hotspots:
            at = on_chain & (auth == residue)
            if not at.any():
                raise ValueError(
                    f"chain {author_chain} hotspot {residue} not present on "
                    f"source structural chain {source_chain}"
                )
            chain_hotspot |= at

        selected_indices = np.flatnonzero(selected)
        index_pieces.append(selected_indices)
        hotspot_pieces.append(chain_hotspot[selected_indices])
        output_id_pieces.append(
            np.full(len(selected_indices), output_chain, dtype="<U1")
        )

        copies = sorted({
            str(value) for value in np.asarray(atoms.chain_id)[author_match]
        })
        ignored = [chain_id for chain_id in copies if chain_id != source_chain]
        suffix = f"; ignored assembly copies {ignored}" if ignored else ""
        logger.info(
            "  target author chain %s (label %s) -> %s%s",
            author_chain,
            source_chain,
            output_chain,
            suffix,
        )

    if not index_pieces:
        raise ValueError("target config contains no chains")
    # Slice once so bonds between selected target chains survive. Integer-array
    # indexing also puts chains into YAML order before IDs are canonicalised.
    target = atoms[np.concatenate(index_pieces)]
    output_ids = np.concatenate(output_id_pieces)
    for annotation in ("chain_id", "auth_asym_id", "label_asym_id"):
        target.set_annotation(annotation, output_ids.copy())
    return target, np.concatenate(hotspot_pieces)


def parse_and_crop(structure_path: str, chains: dict) -> tuple[Any, np.ndarray]:
    """Parse, crop and canonically relabel configured author-numbered chains.

    Returns the cropped AtomArray and, per kept atom, whether it is a hotspot.
    """
    from protenix.data.pipeline.data_pipeline import DataPipeline

    _, bio = DataPipeline.get_data_from_mmcif(
        mmcif=structure_path, pdb_cluster_file=None, dataset="WeightedPDB"
    )
    if "atom_array" not in bio:
        raise RuntimeError(f"could not parse {structure_path}")
    return _select_and_normalize_target_chains(bio["atom_array"], chains)


# --------------------------------------------------------------------------
# binder
# --------------------------------------------------------------------------

def fabricate_binder(template: Any, length: int, chain_id: str = "Z") -> Any:
    """Build `length` placeholder residues to stand in for the binder.

    There is no binder yet, so its atoms have to be invented -- but every
    Protenix annotation (ref_pos, tokatom_idx, centre_atom_mask, ...) has to
    stay internally consistent or tokenisation and featurisation fail in ways
    that do not name this as the cause. Rather than construct them, take real
    backbone atoms from the target and relabel: the annotations then come along
    already valid, and the featurizer rewrites the identities to `xpb` and keeps
    only N/CA/C/O anyway, so the borrowed residue types never reach the model.

    Coordinates are placeholders too. Sampling initialises the design region
    from noise (`cogenerate` seeds x from the schedule), so these are consumed
    only by featurisation, never as a target or a starting point.
    """
    import biotite.structure as struc

    bb = np.isin(template.atom_name, BACKBONE) & (template.mol_type == "protein")
    if not bb.any():
        raise ValueError("target has no protein backbone atoms to borrow from")

    # Whole residues only, and only residues with all four backbone atoms: a
    # partial residue would give the frame builder an incomplete N/CA/C.
    donors = []
    for key in dict.fromkeys(zip(template.chain_id[bb], template.res_id[bb])):
        sel = bb & (template.chain_id == key[0]) & (template.res_id == key[1])
        if set(template.atom_name[sel]) >= set(BACKBONE):
            donors.append(sel)
    if not donors:
        raise ValueError("no complete N/CA/C/O residue in the target to borrow")

    picks = [donors[i % len(donors)] for i in range(length)]
    binder = template[picks[0]]
    for sel in picks[1:]:
        binder += template[sel]

    n_res = length
    per_res = binder.array_length() // n_res
    if per_res * n_res != binder.array_length():
        raise RuntimeError("borrowed residues are not uniform in atom count")

    binder.chain_id = np.array([chain_id] * binder.array_length())
    binder.auth_asym_id = np.array([chain_id] * binder.array_length())
    if "label_asym_id" in binder.get_annotation_categories():
        _set_string_annotation(binder, "label_asym_id", chain_id)
    if "label_entity_id" in binder.get_annotation_categories():
        # The binder is a new sequence/entity, not another symmetry copy of the
        # target residue that happened to donate its backbone annotations.
        existing = set(np.asarray(template.label_entity_id, dtype=str))
        numeric = [int(value) for value in existing if value.isdigit()]
        entity_id = str(max(numeric, default=0) + 1)
        while entity_id in existing:
            entity_id = str(int(entity_id) + 1)
        _set_string_annotation(binder, "label_entity_id", entity_id)
    ids = np.repeat(np.arange(1, n_res + 1), per_res)
    binder.res_id = ids
    binder.auth_seq_id = ids.astype(template.auth_seq_id.dtype)
    if "label_seq_id" in binder.get_annotation_categories():
        binder.label_seq_id = ids.astype(template.label_seq_id.dtype)

    # Representative-atom masks have to be re-pointed at CA. They are copied
    # along with the borrowed atoms, and for most residues they mark CB -- which
    # a backbone-only binder does not have, so the residue silently loses its
    # representative and disappears from every per-token computation. The
    # symptom is not an error but a token count that is too small: 105 fabricated
    # residues came through as 6, the glycines, whose representative is CA
    # already.
    is_ca = binder.atom_name == "CA"
    for mask_name in ("distogram_rep_atom_mask", "centre_atom_mask",
                      "plddt_m_rep_atom_mask"):
        if mask_name in binder.get_annotation_categories():
            current = getattr(binder, mask_name)
            binder.set_annotation(mask_name, is_ca.astype(current.dtype))

    # Displace so the placeholder does not sit inside the target. The sampler
    # overwrites these, but featurisation computes neighbour-dependent features
    # and coincident atoms would make those meaningless.
    span = template.coord.max(axis=0) - template.coord.min(axis=0)
    binder.coord = binder.coord + np.array([span[0] + 30.0, 0.0, 0.0], dtype=np.float32)
    return binder


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def build_features(target: Any, binder: Any, hotspot_on_target: np.ndarray) -> dict:
    """Assemble target + fabricated binder into what the model consumes.

    Re-tokenises rather than patching the target's token array: the binder adds
    tokens, and every downstream index (atom_to_token_idx, centre atoms, the
    side-chain backbone gather) is positional.
    """
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.core.parser import AddAtomArrayAnnot
    from protenix.data.tokenizer import AtomArrayTokenizer
    from protenix.data.utils import data_type_transform, make_dummy_feature

    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    combined = target + binder
    # Sliced target chains and the fabricated binder carry integer identifiers
    # from their source assembly/donor. Recompute them from the canonical chain
    # and entity annotations so the model sees the binder as its own asymmetry
    # unit rather than as part of a target chain.
    combined = AddAtomArrayAnnot.add_int_id(combined)
    combined = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(combined)
    combined = AddAtomArrayAnnot.add_ref_space_uid(combined)
    is_binder = np.concatenate(
        [np.zeros(target.array_length(), bool), np.ones(binder.array_length(), bool)]
    )
    hotspot_atoms = np.concatenate(
        [hotspot_on_target, np.zeros(binder.array_length(), bool)]
    )

    token_array = AtomArrayTokenizer(combined).get_token_array()
    feat = Featurizer(
        cropped_token_array=token_array,
        cropped_atom_array=combined,
        ref_pos_augment=False,   # inference: no augmentation
        lig_atom_rename=False,
    )
    feature_dict = feat.get_all_input_features()
    label_dict = feat.get_labels()
    feature_dict = make_dummy_feature(
        features_dict=feature_dict, dummy_feats=["msa", "template"]
    )
    feature_dict = data_type_transform(feat_or_label_dict=feature_dict)
    label_dict = data_type_transform(feat_or_label_dict=label_dict)
    feature_dict["is_distillation"] = torch.tensor([False])

    selection = DesignSelection(
        binder_atom_mask=is_binder,
        # The training featurizer *samples* hotspots from Ca-Ca contacts, which
        # is right for training -- the model has to work with zero, few or many.
        # At inference they are given, so sampling is switched off here and the
        # channel is written directly below.
        hotspot_force_zero_prob=1.0,
        compute_sidechain=True,
        backbone_only_binder=True,
    )
    feature_dict, label_dict, _ = DesignFeaturizer(selection).transform(
        combined, feature_dict, label_dict
    )

    # Hotspots, per token. `atom_to_token_idx` maps atoms to tokens, so an atom
    # mask becomes a token mask by scatter.
    a2t = feature_dict["atom_to_token_idx"].long()
    n_token = int(feature_dict["residue_index"].shape[0])
    hotspot = torch.zeros(n_token, dtype=torch.float32)
    hotspot[a2t[torch.from_numpy(hotspot_atoms)]] = 1.0
    feature_dict["hotspot"] = hotspot

    return feature_dict, label_dict, combined, is_binder


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def design(model, feature_dict: dict, *, n_step: int = 20, device: str = "cpu",
           **cogen_kwargs) -> dict:
    """Sample one binder. Returns cogenerate's output dict."""
    from pxdesign_train.cogenerate import cogenerate

    feat = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in feature_dict.items()
    }
    model = model.to(device).eval()
    with torch.no_grad():
        return cogenerate(model, feat, N_step=n_step, **cogen_kwargs)


def write_cif(path: str | Path, atoms: Any, coords: np.ndarray,
              is_binder: np.ndarray, sequence: np.ndarray | None) -> None:
    """Write the designed complex.

    ``sequence=None`` is the honest representation of a PXDesign-d backbone-only
    run: write a poly-glycine placeholder for the binder and leave sequence
    design to ProteinMPNN.  In particular, do not report the randomly
    initialised Proteo-AA head that accompanies an official PXDesign checkpoint
    as though PXDesign had generated those identities.
    """
    import biotite.structure.io.pdbx as pdbx

    out = atoms.copy()
    out.coord = np.asarray(coords, dtype=np.float32)
    res_names = out.res_name.copy()

    if sequence is None:
        res_names[is_binder] = "GLY"
        out.set_annotation("res_name", res_names)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        f = pdbx.CIFFile()
        pdbx.set_structure(f, out)
        f.write(str(path))
        return

    # `sequence` is indexed by TOKEN over the whole complex, -1 where nothing was
    # designed -- not by binder residue. Taking the first N entries would read
    # the target's tokens and leave the binder as xpb, which looks like the model
    # produced no sequence at all.
    from pxdesign_train.cogenerate import _AA3

    seq = np.asarray(sequence)
    designed = seq[seq >= 0]
    binder_res = list(dict.fromkeys(out.res_id[is_binder]))
    if len(designed) != len(binder_res):
        logger.warning("%d designed identities for %d binder residues",
                       len(designed), len(binder_res))
    for res, aa in zip(binder_res, designed):
        res_names[is_binder & (out.res_id == res)] = _AA3[int(aa)]
    out.set_annotation("res_name", res_names)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = pdbx.CIFFile()
    pdbx.set_structure(f, out)
    f.write(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="target yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="model weights; omitted runs an untrained model, which "
                         "checks the plumbing and nothing else")
    ap.add_argument("--out", default="designs")
    ap.add_argument("--n-step", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--binder-length",
        type=int,
        default=None,
        help="Override the YAML binder_length; use this to sweep A-CODE's "
             "published 80-130 range.",
    )
    ap.add_argument(
        "--sampler-mode",
        choices=("pxdesign_native", "minimal_euler"),
        default="pxdesign_native",
        help="Reverse-diffusion implementation. Use pxdesign_native for the "
             "official PXDesign backbone control.",
    )
    ap.add_argument(
        "--backbone-only",
        action="store_true",
        help="Ignore AA-head output and write a poly-Gly binder backbone for a "
             "subsequent ProteinMPNN stage.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)

    cfg = load_target_config(args.target)
    if args.binder_length is not None:
        if args.binder_length <= 0:
            ap.error("--binder-length must be positive")
        cfg["binder_length"] = args.binder_length
    logger.info("target %s: %s, binder length %d",
                cfg["name"], Path(cfg["file"]).name, cfg["binder_length"])

    target, hotspot = parse_and_crop(cfg["file"], cfg["chains"])
    binder = fabricate_binder(target, cfg["binder_length"])
    feature_dict, _, combined, is_binder = build_features(target, binder, hotspot)
    logger.info("  %d tokens (%d design), %d hotspot",
                int(feature_dict["residue_index"].shape[0]),
                int(feature_dict["design_token_mask"].sum()),
                int(feature_dict["hotspot"].sum()))

    model = build_model(args.checkpoint, args.device)
    out = design(
        model,
        feature_dict,
        n_step=args.n_step,
        device=args.device,
        sampler_mode=args.sampler_mode,
        seq_mode="complete_unmask",
        sidechain_cycle=False,
    )

    dest = Path(args.out) / f"{cfg['name']}_design.cif"
    sequence = None if args.backbone_only else out["sequence"].cpu().numpy()
    write_cif(dest, combined, out["coordinate"].squeeze(0).cpu().numpy(),
              is_binder, sequence)
    if args.backbone_only:
        logger.info("  backbone-only output: binder identities are poly-Gly placeholders")
    logger.info("wrote %s", dest)
    return 0


def build_model(checkpoint: str | None, device: str):
    """Instantiate the model, optionally loading weights."""
    from protenix.config.config import parse_configs

    from pxdesign_train.configs.configs_train import training_configs
    from pxdesign_train.model import ProtenixDesignTrain

    if checkpoint:
        record = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if "integrated" in record:
            from pxdesign_train.checkpoints import evaluation_model
            return evaluation_model(checkpoint, device=device)
    configs = parse_configs(training_configs, arg_str="")
    configs.enable_sidechain = True
    configs.enable_coevolution = True
    model = ProtenixDesignTrain(configs)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state)

        # Strip the DDP prefix. Checkpoints saved from a multi-GPU run carry
        # `module.` on every key -- the trainer removes it on load
        # (runner/trainer.py) and this has to as well. Without it, and with
        # strict=False below, NOTHING matches and the model runs on random
        # weights while reporting success.
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)

        # strict=False is needed -- a Stage III checkpoint legitimately lacks
        # some buffers -- but it also turns a total mismatch into a silent
        # no-op, so check that the load actually did something.
        loaded = len(model.state_dict()) - len(missing)
        if loaded < 0.5 * len(model.state_dict()):
            raise RuntimeError(
                f"{checkpoint} matched only {loaded} of "
                f"{len(model.state_dict())} parameters -- wrong checkpoint, or a "
                f"key prefix this does not handle. First unmatched keys: "
                f"{sorted(unexpected)[:3]}"
            )
        logger.info("  loaded %s (%d/%d params, missing=%d, unexpected=%d)",
                    checkpoint, loaded, len(model.state_dict()),
                    len(missing), len(unexpected))
    else:
        logger.warning("  NO CHECKPOINT -- untrained weights, output is noise")
    return model.to(device)


if __name__ == "__main__":
    raise SystemExit(main())
