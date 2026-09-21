#!/usr/bin/env python3
"""Crop the ten targets to amino acids and emit PXDesign input configs.

    python scripts/prepare_binder_targets.py --out runs/binder_targets
    python scripts/prepare_binder_targets.py --out runs/binder_targets --verify

This is the step that stands between `configs/binder_benchmark/targets.yaml`
and any backbone generation, and it exists because of a measured failure
rather than tidiness. Eight of the ten depositions carry glycans, ions or
ligands -- NAG, MAN, FUC, BMA, ZN, CL, BR, GOL, PEG, SO4, 9KK/CCS/NH2 -- and
the featurization path this repo drives the backbone through cannot consume
them:

    IndexError: index -100 is out of bounds for dimension 0 with size 21
    InferenceSafeBinder: strict token count 257 != native token count 287

`aa_clean` marks a non-amino-acid token -100 and a 21-way lookup indexes with
it. Measured on the benchmark itself: 5o45 fails as deposited, 1www succeeds.
So each target is cropped to *amino acids inside the published ranges* and
rewritten through Protenix's own `pdb_to_cif`, which is the same normalisation
`scripts/prepare_dimer_targets.py` applies for the same reason.

**`pdb_to_cif` renumbers, and the emitted config is in ITS numbering.**
Measured, not assumed: TrkA chain X residues 282-382 come back as chain A
residues 1-101, and TNFa's A/B/C each restart at 1. Every chain is renamed
A, B, C... in order and renumbered from 1, preserving the gaps where residues
were dropped or were never observed -- so H1's chain A comes out as five runs
(1-44, 46-50, 76-80, 107-111, 258-322), not one span. An earlier version of
this script asserted author numbering survived; the verify step caught it on
eight of ten targets, which is the only reason that assumption is not in the
emitted configs now.

So crops become runs in the converted numbering and hotspots become their
converted positions. That is also exactly PXDesign's own convention, which gives a free
end-to-end check: PDL1's published crop `A 17-132` with hotspots 56/115/123
must come out as `1-116` with hotspots 40/99/107, which is character-for-
character PXDesign's shipped `examples/PDL1_quick_start.yaml`. The script
asserts it.

The mapping is *derived* from the two files rather than assumed from the rule:
the pre-conversion crop and the post-conversion CIF are matched chain by chain
in order, so if `pdb_to_cif` ever changes its convention this breaks loudly
instead of shifting an epitope quietly.

It does not choose a binder length. `targets.yaml` carries A-CODE's 80-130
grid, and a single `binder_length` is not a reproduction of a range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "binder_benchmark" / "targets.yaml"
DEFAULT_MMCIF = Path("/scratch/m000137-pm06/Proteo-AA/protenix_data/mmcif")


def _parse_range(text: str) -> tuple[int, int]:
    first, _, last = str(text).partition("-")
    return int(first), int(last)


def crop_to_amino_acids(source: Path, chains: dict[str, Any], out_pdb: Path):
    """Keep only amino acids inside the published ranges. Author numbering intact.

    Insertion codes survive: 5vli chain A carries 49A inside the first H1 crop
    and dropping it would silently shorten the crop by one residue.
    """
    import gemmi

    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    model = structure[0]

    wanted = {c: [_parse_range(r) for r in spec["crop"]] for c, spec in chains.items()}
    kept_counts: dict[str, int] = {}
    dropped_non_aa: dict[str, int] = {}
    insertion_coded: dict[str, int] = {}

    new_model = gemmi.Model(model.name)
    for chain in model:
        if chain.name not in wanted:
            continue
        new_chain = gemmi.Chain(chain.name)
        for residue in chain:
            number = residue.seqid.num
            if not any(lo <= number <= hi for lo, hi in wanted[chain.name]):
                continue
            info = gemmi.find_tabulated_residue(residue.name)
            if not (info and info.is_amino_acid()):
                dropped_non_aa[residue.name] = dropped_non_aa.get(residue.name, 0) + 1
                continue
            if residue.seqid.icode.strip():
                # Recorded because the PDB round-trip loses it: 5vli chain A
                # carries 49A inside H1's first crop and the converted file
                # comes back one residue short.
                insertion_coded[chain.name] = insertion_coded.get(chain.name, 0) + 1
            new_chain.add_residue(residue)
            kept_counts[chain.name] = kept_counts.get(chain.name, 0) + 1
        if len(new_chain):
            new_model.add_chain(new_chain)

    cropped = gemmi.Structure()
    cropped.name = structure.name
    cropped.spacegroup_hm = "P 1"
    cropped.add_model(new_model)
    cropped.setup_entities()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    cropped.write_pdb(str(out_pdb))
    return kept_counts, dropped_non_aa, insertion_coded


def to_cif(pdb_path: Path, cif_path: Path, entry_id: str) -> Path:
    from protenix.data.utils import pdb_to_cif

    # entry_id has to look like a PDB code: the converter writes it into the
    # block header and Protenix reads it back out.
    pdb_to_cif(str(pdb_path), str(cif_path), entry_id=str(entry_id)[:4].lower())
    return cif_path


def build_mapping(pdb_path: Path, cif_path: Path, chains: dict[str, Any]):
    """(author chain, author resnum) -> (converted chain, converted resnum).

    Derived by walking the two files in parallel rather than by applying the
    rule `pdb_to_cif` currently happens to use. If the convention changes, the
    residue counts stop matching and this raises, instead of emitting a config
    whose hotspots point somewhere else.
    """
    import gemmi

    before = gemmi.read_structure(str(pdb_path))
    before.setup_entities()
    after = gemmi.read_structure(str(cif_path))
    after.setup_entities()

    before_chains = [c for c in before[0] if len(c)]
    after_chains = [c for c in after[0] if len(c)]
    if len(before_chains) != len(after_chains):
        raise SystemExit(
            f"{cif_path.name}: {len(before_chains)} chain(s) before conversion, "
            f"{len(after_chains)} after; cannot map the crop"
        )

    mapping: dict[tuple[str, int], tuple[str, int]] = {}
    chain_map: dict[str, str] = {}
    for old_chain, new_chain in zip(before_chains, after_chains):
        if len(old_chain) != len(new_chain):
            # The one tolerated cause is an insertion code, which the PDB
            # round-trip drops. Anything else means the two files are not the
            # same structure and the mapping below would be nonsense.
            lost = [r for r in old_chain if r.seqid.icode.strip()]
            if len(old_chain) - len(new_chain) != len(lost):
                raise SystemExit(
                    f"{cif_path.name}: chain {old_chain.name} has "
                    f"{len(old_chain)} residues before conversion and "
                    f"{len(new_chain)} after, and only {len(lost)} carry an "
                    "insertion code; the conversion changed the structure"
                )
            old_residues = [r for r in old_chain if not r.seqid.icode.strip()]
        else:
            old_residues = list(old_chain)
        chain_map[old_chain.name] = new_chain.name
        for old_res, new_res in zip(old_residues, new_chain):
            if old_res.name != new_res.name:
                raise SystemExit(
                    f"{cif_path.name}: residue identity changed during "
                    f"conversion ({old_res.name} -> {new_res.name})"
                )
            mapping[(old_chain.name, old_res.seqid.num)] = (
                new_chain.name, new_res.seqid.num
            )
    return chain_map, mapping


def convert_spec(chains: dict[str, Any], chain_map, mapping) -> dict[str, Any]:
    """Rewrite crops and hotspots into the converted file's numbering."""
    out: dict[str, Any] = {}
    problems: list[str] = []
    for chain_id, spec in chains.items():
        new_chain = chain_map.get(chain_id)
        if new_chain is None:
            problems.append(f"chain {chain_id} vanished during conversion")
            continue
        numbers = sorted(
            new for (old_c, _old_n), (new_c, new) in mapping.items()
            if old_c == chain_id and new_c == new_chain
        )
        if not numbers:
            problems.append(f"chain {chain_id} kept no residues")
            continue
        # Emit the actual contiguous runs, not min-max. H1's chain A crop is
        # four separate segments and collapsing it to "1-322" would describe a
        # span the file does not contain -- harmless for selection, wrong as a
        # record of what the target is.
        runs: list[tuple[int, int]] = []
        start = previous = numbers[0]
        for number in numbers[1:]:
            if number != previous + 1:
                runs.append((start, previous))
                start = number
            previous = number
        runs.append((start, previous))
        entry: dict[str, Any] = {"crop": [f"{a}-{b}" for a, b in runs]}
        entry["n_residues"] = len(numbers)
        hotspots = []
        for hotspot in spec.get("hotspots") or []:
            moved = mapping.get((chain_id, int(hotspot)))
            if moved is None:
                problems.append(
                    f"hotspot {chain_id}{hotspot} is not in the crop and has no "
                    "converted position"
                )
                continue
            hotspots.append(moved[1])
        if hotspots:
            entry["hotspots"] = sorted(hotspots)
        out[new_chain] = entry
    return out, problems


# PXDesign ships this example, and it is the same target through the same
# convention, so it is a free end-to-end check on the whole mapping.
PDL1_PUBLISHED = {"crop": ["1-116"], "hotspots": [40, 99, 107]}


def featurizes(cif_path: Path, binder_chain: str, crop_size: int) -> dict[str, Any]:
    """Prove the written file survives the path the backbone driver uses.

    The whole point of cropping is that the featurizer accepts the result, so
    this is the only check that actually closes the loop. It is optional
    because it imports the training stack.
    """
    try:
        from pxf.backbone.driver import featurize_structures, to_featurized

        items = featurize_structures(
            [str(cif_path)], crop_size=crop_size,
            binder_chain_ids=[binder_chain], parser_dataset="Distillation",
        )
        sample_id, dataset = items[0]
        structure = to_featurized(sample_id, dataset[0])
        return {"ok": True, "tokens": int(structure.topology.num_tokens)}
    except Exception as exc:  # noqa: BLE001 - failing to featurize is the finding
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF))
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", nargs="*", default=None)
    parser.add_argument("--verify", action="store_true",
                        help="also featurize each prepared target (imports the "
                             "training stack)")
    parser.add_argument("--crop-size", type=int, default=768)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out).expanduser().resolve()
    (out / "structures").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    entries = config["targets"]
    if args.target:
        wanted = {t.lower() for t in args.target}
        entries = [e for e in entries if e["name"].lower() in wanted]

    lengths = config["sampling"]["lengths"]
    records, failures = [], 0
    for entry in entries:
        name, pdb_id = entry["name"], entry["pdb_id"]
        source = Path(args.mmcif_dir) / f"{pdb_id}.cif"
        if not source.is_file():
            print(f"FAIL {name}: no structure at {source}")
            failures += 1
            continue

        pdb_path = out / "structures" / f"{name}.pdb"
        cif_path = out / "structures" / f"{name}.cif"
        kept, dropped, insertions = crop_to_amino_acids(
            source, entry["chains"], pdb_path
        )
        to_cif(pdb_path, cif_path, pdb_id)

        chain_map, mapping = build_mapping(pdb_path, cif_path, entry["chains"])
        converted, problems = convert_spec(entry["chains"], chain_map, mapping)

        # PDL1 is the end-to-end check on the whole mapping: PXDesign ships
        # this exact target through this exact convention.
        if name == "PDL1":
            got = {
                "crop": converted.get("A", {}).get("crop"),
                "hotspots": converted.get("A", {}).get("hotspots"),
            }
            if got != PDL1_PUBLISHED:
                problems.append(
                    f"PDL1 converts to {got} but PXDesign's shipped "
                    f"examples/PDL1_quick_start.yaml is {PDL1_PUBLISHED}. The "
                    "crop or the mapping is wrong."
                )

        record: dict[str, Any] = {
            "name": name, "pdb_id": pdb_id, "source": str(source),
            "cif": str(cif_path), "kept_residues": kept,
            "dropped_non_amino_acid": dropped,
            "insertion_coded_residues": insertions,
            "n_target_residues": sum(kept.values()),
            "chain_map": chain_map,
            "converted_chains": converted,
            "author_to_converted": {
                f"{c}{n}": f"{nc}{nn}" for (c, n), (nc, nn) in sorted(mapping.items())
            },
            "problems": problems,
        }

        # n_residues is bookkeeping for prepared.json, not part of the
        # PXDesign input schema.
        design_config = {
            "target": {
                "file": str(cif_path),
                "chains": {
                    c: {k: v for k, v in spec.items() if k != "n_residues"}
                    for c, spec in converted.items()
                },
            },
            "binder_lengths": list(lengths),
        }
        config_path = out / "configs" / f"{name}.yaml"
        config_path.write_text(
            "# Generated by scripts/prepare_binder_targets.py -- do not edit.\n"
            f"# Source: {source}\n"
            "# Cropped to amino acids inside the published ranges, then run\n"
            "# through protenix pdb_to_cif, which renames chains A,B,C... in\n"
            "# order and renumbers each chain from 1, PRESERVING the gaps where\n"
            "# residues were dropped or unobserved -- so a crop is a list of\n"
            "# runs, not one span. Crops and hotspots below are in THAT\n"
            "# numbering; prepared.json carries the full author -> converted map.\n"
            + yaml.safe_dump(design_config, sort_keys=False)
        )
        record["config"] = str(config_path)

        if args.verify:
            binder_chain = sorted(entry["chains"])[0]
            record["featurizes"] = featurizes(cif_path, binder_chain, args.crop_size)

        status = "ok"
        if problems:
            status = "FAIL"
            failures += 1
        elif args.verify and not record["featurizes"]["ok"]:
            status = "FAIL"
            failures += 1

        crop_text = " ".join(
            f"{c}:{v['n_residues']}aa/{len(v['crop'])}seg"
            for c, v in sorted(converted.items())
        )
        dropped_text = (
            " dropped " + ",".join(f"{k}x{v}" for k, v in sorted(dropped.items()))
            if dropped else ""
        )
        feat = ""
        if args.verify:
            feat = ("  featurizes=" + ("yes" if record["featurizes"]["ok"] else "NO"))
        print(f"{status:<4} {name:<8} {record['n_target_residues']:>4} aa  "
              f"{crop_text}{dropped_text}{feat}")
        for problem in problems:
            print(f"       PROBLEM: {problem}")
        if args.verify and not record["featurizes"]["ok"]:
            print(f"       {record['featurizes']['error']}")
        records.append(record)

    (out / "prepared.json").write_text(
        json.dumps({"targets": records}, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nwrote {out / 'prepared.json'}")
    if failures:
        raise SystemExit(f"{failures} target(s) failed preparation")
    print(f"{len(records)} target(s) prepared under {out}")


if __name__ == "__main__":
    main()
