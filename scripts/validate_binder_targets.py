#!/usr/bin/env python3
"""Check every crop and hotspot in the target config against the structures.

    python scripts/validate_binder_targets.py
    python scripts/validate_binder_targets.py --target H1 --show-alternatives

No GPU, no model, no network. This is the check that runs before a generation
job is submitted, because the failure it catches is silent: a crop range that
selects nothing, or a hotspot that does not resolve, produces a perfectly
well-formed design against an epitope nobody chose.

Four things are verified per target:

  * every configured chain is resolved to the id the **featurizer** uses.
    `CifFileProvider(binder_chain_ids=...)` matches `label_asym_id`; this
    config, Table S1 and gemmi all speak author ids. They differ for three of
    the ten -- SC2RBD auth E is label B, TrkA auth X is label C, VEGFA auth
    V/W are labels C/D -- and 6m0j is the dangerous one, because label E
    exists there as a glycan on ACE2, so the unconverted id selects something
    instead of failing. See `pxf/backbone/chain_ids.py`;
  * every crop range selects at least one residue, and the per-chain residue
    count is reported so a range that quietly clipped against unobserved
    density is visible rather than assumed;
  * every hotspot resolves to an actual residue, and its identity is printed --
    a hotspot that resolves to the wrong amino acid is the signature of an
    off-by-one or a label/author mix-up, and the identity is the cheapest way
    to see it;
  * for the seven depositions with a partner bound at the site, the minimum
    heavy-atom distance from each hotspot to a chain outside the crop. Real
    interface residues land in single-digit Angstroms. TNFa (1tnf, apo) and IR
    (nothing within 13 A of the L1 patch in 4zxb) have no partner and are
    reported as such instead of being silently passed.

`--show-alternatives` additionally evaluates any `numbering_alternatives` an
entry records. H1 has three surviving readings of its chain A crop and this is
how they are compared side by side; see the note in the config.

It also reports **non-amino-acid content**, which is not cosmetic. Eight of the
ten depositions carry glycans, ions or ligands (NAG, MAN, FUC, BMA, ZN, CL, BR,
GOL, PEG, SO4, 9KK/CCS/NH2); only 1tnf and 1www are clean. The featurization
path this repo drives the backbone through (`CifFileProvider` ->
`DesignSourceDataset`) does not tolerate them: `aa_clean` marks a non-amino-acid
token -100 and a downstream lookup raises `IndexError: index -100 is out of
bounds for dimension 0 with size 21`, or the crop fails an
`InferenceSafeBinder` token-count check first. Measured: 5o45 fails, 1www
succeeds. Any target with a non-empty count below must be cropped to amino
acids before featurizing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import yaml  # noqa: E402

from pxf.backbone.chain_ids import ChainIdError, featurizer_chain_id  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "binder_benchmark" / "targets.yaml"
DEFAULT_MMCIF = Path("/scratch/m000137-pm06/Proteo-AA/protenix_data/mmcif")

# Beyond this there is no partner at the site and the distance check is not
# evidence of anything; 1tnf is apo and 4zxb has nothing near the L1 patch.
NO_PARTNER_ANGSTROMS = 13.0


def _parse_range(text: str) -> tuple[int, int]:
    first, _, last = str(text).partition("-")
    return int(first), int(last)


def _non_amino_acid_tokens(model) -> dict[str, int]:
    """Residue names in the deposition that are not amino acids, waters aside.

    Reported because the featurizer cannot consume them, not because they are
    wrong: a glycan on an ectodomain is real structure. They have to be cropped
    out before the backbone driver sees the file.
    """
    import gemmi

    counts: dict[str, int] = {}
    for chain in model:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if (info and info.is_amino_acid()) or residue.name == "HOH":
                continue
            counts[residue.name] = counts.get(residue.name, 0) + 1
    return counts


def _load(mmcif_dir: Path, pdb_id: str):
    import gemmi

    for name in (f"{pdb_id.lower()}.cif", f"{pdb_id.upper()}.cif"):
        path = mmcif_dir / name
        if path.is_file():
            structure = gemmi.read_structure(str(path))
            structure.setup_entities()
            return structure[0], path
    raise SystemExit(f"no structure for {pdb_id} under {mmcif_dir}")


def _chain_residues(model, chain_id: str) -> list:
    for chain in model:
        if chain.name == chain_id:
            return list(chain)
    return []


def _selected(model, chain_id: str, ranges: list[str]) -> list:
    """Residues of `chain_id` inside any range, author numbering.

    Insertion codes are kept: 5vli chain A carries 49A inside the first H1
    crop, and dropping it would shorten the crop by one residue without
    anything saying so.
    """
    residues = _chain_residues(model, chain_id)
    bounds = [_parse_range(r) for r in ranges]
    return [r for r in residues
            if any(lo <= r.seqid.num <= hi for lo, hi in bounds)]


def _coords(residues) -> np.ndarray:
    points = [[a.pos.x, a.pos.y, a.pos.z] for r in residues for a in r]
    return np.asarray(points, dtype=float) if points else np.zeros((0, 3))


def _outside_crop_coords(model, cropped_chains: dict[str, list[str]]) -> np.ndarray:
    """Every heavy atom that the crop does not claim: the candidate partner."""
    kept = {
        (chain_id, r.seqid.num, r.seqid.icode)
        for chain_id, ranges in cropped_chains.items()
        for r in _selected(model, chain_id, ranges)
    }
    points = []
    for chain in model:
        for residue in chain:
            key = (chain.name, residue.seqid.num, residue.seqid.icode)
            if key in kept:
                continue
            points.extend([[a.pos.x, a.pos.y, a.pos.z] for a in residue])
    return np.asarray(points, dtype=float) if points else np.zeros((0, 3))


def _min_distance(residue, partner: np.ndarray) -> Optional[float]:
    if partner.size == 0:
        return None
    here = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in residue], dtype=float)
    if here.size == 0:
        return None
    d = np.sqrt(((here[:, None, :] - partner[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def check_target(entry: dict[str, Any], mmcif_dir: Path) -> dict[str, Any]:
    model, path = _load(mmcif_dir, entry["pdb_id"])
    chains = entry["chains"]
    crop_spec = {c: v["crop"] for c, v in chains.items()}
    partner = _outside_crop_coords(model, crop_spec)

    result: dict[str, Any] = {
        "name": entry["name"], "pdb_id": entry["pdb_id"], "structure": str(path),
        "chains": {}, "hotspots": [], "problems": [],
        "non_amino_acid_tokens": _non_amino_acid_tokens(model),
    }
    total = 0
    for chain_id, spec in chains.items():
        selected = _selected(model, chain_id, spec["crop"])
        total += len(selected)
        result["chains"][chain_id] = {"crop": spec["crop"], "n_residues": len(selected)}
        try:
            label = featurizer_chain_id(path, chain_id)
        except ChainIdError as error:
            label = None
            result["problems"].append(f"chain {chain_id}: {error}")
        result["chains"][chain_id]["featurizer_chain"] = label
        if not selected:
            result["problems"].append(
                f"chain {chain_id} crop {spec['crop']} selected NO residues"
            )
        for text in spec["crop"]:
            lo, hi = _parse_range(text)
            if not any(lo <= r.seqid.num <= hi for r in selected):
                result["problems"].append(
                    f"chain {chain_id} range {text} selected no residues"
                )

        for hotspot in spec.get("hotspots") or []:
            match = [r for r in selected if r.seqid.num == int(hotspot)]
            if not match:
                result["problems"].append(
                    f"hotspot {chain_id}{hotspot} does not resolve inside the crop"
                )
                result["hotspots"].append(
                    {"chain": chain_id, "resid": hotspot, "resolved": False}
                )
                continue
            residue = match[0]
            distance = _min_distance(residue, partner)
            result["hotspots"].append({
                "chain": chain_id, "resid": hotspot, "resolved": True,
                "res_name": residue.name,
                "min_distance_to_non_crop": (
                    None if distance is None else round(distance, 2)
                ),
            })
    result["n_target_residues"] = total

    distances = [h["min_distance_to_non_crop"] for h in result["hotspots"]
                 if h.get("min_distance_to_non_crop") is not None]
    if distances and min(distances) > NO_PARTNER_ANGSTROMS:
        result["partner_bound"] = False
        result["note"] = (
            f"nothing outside the crop comes within {NO_PARTNER_ANGSTROMS} A of "
            "any hotspot; this deposition has no partner at the site, so the "
            "distance check does not apply and resolution is the only evidence"
        )
    elif distances:
        result["partner_bound"] = True
    return result


def check_alternatives(entry: dict[str, Any], mmcif_dir: Path) -> list[dict[str, Any]]:
    """Score each recorded alternative numbering the same way as the primary."""
    alternatives = entry.get("numbering_alternatives") or {}
    out = []
    for label, override in alternatives.items():
        merged = {c: dict(v) for c, v in entry["chains"].items()}
        for chain_id, crop in override.items():
            if chain_id in merged:
                merged[chain_id]["crop"] = list(crop)
        probe = dict(entry)
        probe["chains"] = merged
        probe["name"] = f"{entry['name']} [{label}]"
        probe.pop("numbering_alternatives", None)
        out.append(check_target(probe, mmcif_dir))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF))
    parser.add_argument("--target", nargs="*", default=None)
    parser.add_argument("--show-alternatives", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    mmcif_dir = Path(args.mmcif_dir)
    entries = config["targets"]
    if args.target:
        wanted = {t.lower() for t in args.target}
        entries = [e for e in entries if e["name"].lower() in wanted]
        if not entries:
            raise SystemExit(f"no target named {args.target}")

    failures = 0
    for entry in entries:
        results = [check_target(entry, mmcif_dir)]
        if args.show_alternatives:
            results += check_alternatives(entry, mmcif_dir)
        for result in results:
            chains = " ".join(
                f"{c}:{d['n_residues']}" for c, d in result["chains"].items()
            )
            status = "FAIL" if result["problems"] else "ok"
            partner = result.get("partner_bound")
            tag = "" if partner is None else ("" if partner else "  [apo/no partner]")
            print(f"{status:<4} {result['name']:<22} {result['pdb_id']}  "
                  f"{result['n_target_residues']:>4} res  ({chains}){tag}")
            for hotspot in result["hotspots"]:
                if not hotspot["resolved"]:
                    print(f"       hotspot {hotspot['chain']}{hotspot['resid']}: "
                          "DOES NOT RESOLVE")
                    continue
                distance = hotspot["min_distance_to_non_crop"]
                shown = "n/a" if distance is None else f"{distance:.2f} A"
                print(f"       {hotspot['chain']}{hotspot['resid']:<4} "
                      f"{hotspot['res_name']:<4} nearest non-crop atom {shown}")
            relabelled = {
                c: d["featurizer_chain"] for c, d in result["chains"].items()
                if d.get("featurizer_chain") and d["featurizer_chain"] != c
            }
            if relabelled:
                shown = ", ".join(f"author {c} -> label {l}"
                                  for c, l in relabelled.items())
                print(f"       featurizer chain id differs: {shown}"
                      f"  -- pass the label id to binder_chain_ids")
            for problem in result["problems"]:
                print(f"       PROBLEM: {problem}")
            hetero = result.get("non_amino_acid_tokens") or {}
            if hetero:
                shown = ", ".join(f"{k}x{v}" for k, v in sorted(hetero.items())[:6])
                print(f"       non-amino-acid tokens: {shown}"
                      f"  -- must be cropped before featurizing")
            if result.get("note"):
                print(f"       note: {result['note']}")
            failures += len(result["problems"])

    print()
    if failures:
        raise SystemExit(f"{failures} problem(s) across {len(entries)} target(s)")
    print(f"all {len(entries)} target(s) validate against {mmcif_dir}")


if __name__ == "__main__":
    main()
