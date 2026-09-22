#!/usr/bin/env python3
"""Audit the complexes, then write the manifests feedback training may use.

    python scripts/prepare_integrated_feedback_data.py \
        --train configs/bs_seq_sc_train.marlowe.parquet \
        --calibration configs/bs_seq_sc_calib.marlowe.parquet \
        --validation configs/gen_stress_prepared.marlowe.parquet \
        --train-pool pdb --out runs/integrated_feedback_v1/data

Writes ``train_<pool>.parquet``, ``calibration_<pool>.parquet``,
``validation.parquet``, an ``exclusions.csv`` naming every dropped row and why,
and ``audit.json``. Nothing is filtered silently.

### What the existing manifests are, and are not

``pool`` separates two very different sources. The 256 ``pdb`` rows are
deposited structures; the 256 ``ted`` rows are AlphaFold-model-derived domain
dimers. **They are not all experimental binder complexes**, and "target" and
"binder" are task assignments this pipeline made, not evidence that anyone
designed or validated a binder. The first feedback pilot trains on PDB only,
which is why ``--train-pool`` exists and defaults to ``pdb``.

``cluster_id`` is NOT a homology cluster on the PDB rows: it is usually just
the PDB ID, and there are 512 distinct values for 512 rows. Treating it as a
cluster would make the homology audit vacuous, so sequences are compared
directly here and ``cluster_id`` is carried through untouched as provenance.

### The homology check, and its limits

Every candidate training chain is aligned against every validation chain,
every calibration chain, and every AlphaProteo benchmark TARGET chain. A row
is excluded when it reaches ``--identity-threshold`` identity over
``--coverage-threshold`` coverage against any of them, on either chain. Those
thresholds are a conservative pilot rule, not an established standard, and the
settings are recorded next to the counts.

The benchmark target sequences are read out of the cached backbone collection
(each payload's target-chain ``res_name``), so the check needs no download and
no separate target preparation. If the collection is absent the benchmark
families go UNCHECKED and that is recorded as such rather than passed.

Identity here is global pairwise alignment identity over the shorter sequence.
It is a screen, not a substitute for structural clustering: two chains under
threshold can still share a fold.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BACKBONES = "/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench/backbones"
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def three_to_one() -> dict[str, str]:
    from fampnn.data import residue_constants as rc

    return dict(rc.restype_3to1)


def chain_sequences(path, wanted) -> dict[str, str]:
    """``chain -> one-letter sequence`` for the requested chains, via gemmi."""
    import gemmi

    table = three_to_one()
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    out: dict[str, str] = {}
    for chain in structure[0]:
        if wanted and chain.name not in wanted:
            continue
        # CANONICAL RESIDUES ONLY. An earlier version admitted everything and
        # wrote 'X' for waters, ions and ligands; two chains with long solvent
        # runs then aligned X against X and scored as homologous, which
        # excluded 8 of 8 candidates in a smoke run -- five of them against
        # the same validation entry, which is what gave it away.
        out[chain.name] = "".join(
            table[res.name] for res in chain
            if res.name in table and table[res.name] in CANONICAL
        )
    return out


def audit_one(row, *, table, min_sc_completeness, max_resolution):
    """``(problems, facts)`` for one manifest row. Empty problems = accepted."""
    import gemmi
    import numpy as np

    problems: list[str] = []
    facts: dict = {}
    path = Path(str(row["cif_path"]))
    if not path.is_file():
        return [f"cif missing: {path}"], facts

    try:
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()
    except Exception as exc:  # noqa: BLE001 - a bad file is a finding
        return [f"unreadable: {type(exc).__name__}: {exc}"], facts

    model = structure[0]
    present = {c.name for c in model}
    binder = str(row["converted_binder_chain"])
    targets = [c for c in str(row["converted_target_chains"]).split(",") if c]
    facts["chains_present"] = sorted(present)
    missing_chains = [c for c in [binder, *targets] if c not in present]
    if missing_chains:
        problems.append(
            f"manifest names chain(s) {missing_chains} that the file does not "
            f"have (has {sorted(present)}); the author-to-label mapping is wrong"
        )
        return problems, facts

    # Resolution, where the file records one. 0.0 means absent in these caches,
    # not a perfect structure, so it is reported rather than treated as passing.
    resolution = float(structure.resolution or 0.0)
    facts["resolution"] = resolution or None
    if resolution <= 0.0:
        facts["resolution_missing"] = True
    elif resolution > max_resolution:
        problems.append(f"resolution {resolution:.2f} A > {max_resolution} A")

    # Canonical protein tokens, valid frames, side-chain completeness.
    n_res = n_noncanonical = n_bad_frame = 0
    sc_present = sc_expected = 0
    for name in [binder, *targets]:
        chain = next(c for c in model if c.name == name)
        for res in chain:
            one = table.get(res.name)
            if one is None or one not in CANONICAL:
                n_noncanonical += 1
                continue
            n_res += 1
            atoms = {a.name for a in res}
            if not {"N", "CA", "C"} <= atoms:
                n_bad_frame += 1
            expected = _heavy_sidechain_count(res.name)
            if expected:
                sc_expected += expected
                sc_present += sum(
                    1 for a in res
                    if a.name not in ("N", "CA", "C", "O", "OXT")
                    and a.element.name != "H"
                )
    facts.update(
        residues=n_res, noncanonical=n_noncanonical, bad_frames=n_bad_frame,
        sc_completeness=(sc_present / sc_expected if sc_expected else None),
    )
    if n_res == 0:
        problems.append("no canonical protein residues in the named chains")
        return problems, facts
    if n_noncanonical:
        problems.append(
            f"{n_noncanonical} non-canonical token(s); the side-chain module "
            "accepts only the canonical twenty"
        )
    if n_bad_frame:
        problems.append(f"{n_bad_frame} residue(s) without a full N/CA/C frame")
    if facts["sc_completeness"] is not None and (
        facts["sc_completeness"] < min_sc_completeness
    ):
        problems.append(
            f"side-chain heavy-atom completeness "
            f"{facts['sc_completeness']:.3f} < {min_sc_completeness}"
        )
    if int(row.get("interface_res") or 0) <= 0:
        problems.append("no interface residues recorded; the partners do not touch")
    return problems, facts


_SIDECHAIN_HEAVY = {
    "ALA": 1, "ARG": 7, "ASN": 4, "ASP": 4, "CYS": 2, "GLN": 5, "GLU": 5,
    "GLY": 0, "HIS": 6, "ILE": 4, "LEU": 4, "LYS": 5, "MET": 4, "PHE": 7,
    "PRO": 3, "SER": 2, "THR": 3, "TRP": 10, "TYR": 8, "VAL": 3,
}


def _heavy_sidechain_count(res_name: str) -> int:
    return _SIDECHAIN_HEAVY.get(res_name, 0)


def benchmark_target_sequences(backbones: str) -> tuple[dict[str, str], str]:
    """One target-chain sequence per AlphaProteo target, from the cached payloads.

    Reading them here avoids needing the targets prepared or downloaded: the
    payload already carries per-atom ``res_name`` and a per-token design mask,
    so the target chain's sequence is recoverable exactly.
    """
    manifest = Path(backbones) / "backbones.json"
    if not manifest.is_file():
        return {}, f"UNCHECKED: no backbone collection at {backbones}"

    import numpy as np
    import torch

    table = three_to_one()
    records = json.loads(manifest.read_text())["designs"]
    first: dict[str, str] = {}
    for record in records:
        if record["target"] in first:
            continue
        first[record["target"]] = record["design_id"]
    out: dict[str, str] = {}
    for target, design_id in sorted(first.items()):
        payload = torch.load(
            Path(backbones) / "designs" / f"{design_id}.pt",
            map_location="cpu", weights_only=False,
        )
        topology = payload["topology"]
        a2t = np.asarray(topology["atom_to_token_idx"]).astype(int)
        design = np.asarray(topology["design_mask"]).astype(bool)
        names = np.empty(int(topology["n_tokens"]), dtype=object)
        names[a2t] = np.asarray(topology["annotations"]["res_name"])
        out[target] = "".join(
            table.get(str(n), "X") for n, is_design in zip(names, design)
            if not is_design
        )
    return out, f"read from {len(out)} target(s) in the cached collection"


def identity_over_coverage(a: str, b: str) -> tuple[float, float]:
    """``(identity, coverage)`` from a LOCAL alignment.

        identity = matches / aligned columns
        coverage = aligned columns / len(shorter sequence)

    Both parts matter and an earlier version got both wrong. It aligned
    GLOBALLY and divided matches by the shorter length, so a 208-residue chain
    against an unrelated 76-residue one scored 0.316 "identity" purely from
    chance matches spread across the long sequence -- and coverage came out at
    1.0 by construction, because a global alignment always consumes the shorter
    sequence. The screen excluded 16 of 16 candidates, including glutathione
    S-transferase against a histone-fold protein.

    Local alignment is the right tool for "X% identity over Y% coverage": for
    unrelated sequences it finds a short high-scoring patch, which gives high
    identity but LOW coverage, and the conjunction of the two thresholds
    rejects it. For real homologues the matching region spans most of the
    shorter chain and both thresholds are met.
    """
    from Bio import Align

    if not a or not b:
        return 0.0, 0.0
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    try:
        alignment = aligner.align(a, b)[0]
    except Exception:  # noqa: BLE001 - non-standard letters
        return 0.0, 0.0
    top, bottom = alignment[0], alignment[1]
    aligned = sum(1 for x, y in zip(top, bottom) if x != "-" and y != "-")
    if not aligned:
        return 0.0, 0.0
    # 'X' never counts as an identity: an unknown matching an unknown is not
    # evidence of relatedness.
    matches = sum(
        1 for x, y in zip(top, bottom)
        if x == y and x != "-" and x != "X"
    )
    shorter = min(len(a), len(b))
    return matches / aligned, aligned / shorter


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--train-pool", default="pdb",
                        choices=("pdb", "ted", "both"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--min-sc-completeness", type=float, default=0.95)
    parser.add_argument("--identity-threshold", type=float, default=0.30)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--backbones", default=BACKBONES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import pandas as pd

    table = three_to_one()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train)
    calib = pd.read_parquet(args.calibration)
    val = pd.read_parquet(args.validation)

    held_out_ids = set(calib.example_id) | set(val.example_id)
    overlap = set(train.example_id) & held_out_ids
    print(f"train {len(train)} | calibration {len(calib)} | validation {len(val)}")
    print(f"example_id overlap train vs held-out: {len(overlap)}")

    # ---- sequences of everything we must stay away from -------------------
    print("collecting held-out and benchmark sequences ...")
    forbidden: list[tuple[str, str, str]] = []  # (source, id, sequence)
    for name, frame in (("validation", val), ("calibration", calib)):
        for row in frame.itertuples():
            wanted = {str(row.converted_binder_chain)} | {
                c for c in str(row.converted_target_chains).split(",") if c
            }
            try:
                for chain, seq in chain_sequences(row.cif_path, wanted).items():
                    forbidden.append((name, f"{row.example_id}:{chain}", seq))
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING {name}/{row.example_id}: {type(exc).__name__}")
    bench, bench_note = benchmark_target_sequences(args.backbones)
    for target, seq in bench.items():
        forbidden.append(("benchmark_target", target, seq))
    print(f"  {len(forbidden)} forbidden chain(s); benchmark targets: {bench_note}")

    # ---- audit the candidate training rows --------------------------------
    pools = ("pdb", "ted") if args.train_pool == "both" else (args.train_pool,)
    candidates = train[train.pool.isin(pools)].reset_index(drop=True)
    if args.limit:
        candidates = candidates.head(args.limit)
    print(f"auditing {len(candidates)} candidate row(s) from pool(s) {pools} ...")

    accepted, exclusions, facts_by_id = [], [], {}
    reasons = Counter()
    for n, row in enumerate(candidates.itertuples()):
        record = {c: getattr(row, c) for c in candidates.columns}
        problems, facts = audit_one(
            record, table=table,
            min_sc_completeness=args.min_sc_completeness,
            max_resolution=args.max_resolution,
        )
        if record["example_id"] in held_out_ids:
            problems.append("example_id also appears in calibration/validation")

        # homology, only if it survived the structural checks
        homology = None
        if not problems:
            wanted = {str(record["converted_binder_chain"])} | {
                c for c in str(record["converted_target_chains"]).split(",") if c
            }
            try:
                mine = chain_sequences(record["cif_path"], wanted)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"sequence read failed: {type(exc).__name__}")
                mine = {}
            for chain, seq in mine.items():
                for source, other_id, other in forbidden:
                    ident, cov = identity_over_coverage(seq, other)
                    if (ident >= args.identity_threshold
                            and cov >= args.coverage_threshold):
                        homology = {
                            "chain": chain, "against": other_id,
                            "source": source, "identity": round(ident, 4),
                            "coverage": round(cov, 4),
                        }
                        problems.append(
                            f"homologous to {source} {other_id}: identity "
                            f"{ident:.2f} over coverage {cov:.2f} (chain {chain})"
                        )
                        break
                if homology:
                    break

        facts["homology"] = homology
        facts_by_id[record["example_id"]] = facts
        if problems:
            for p in problems:
                reasons[p.split(":")[0].split(";")[0][:60]] += 1
            exclusions.append({
                "example_id": record["example_id"], "pool": record["pool"],
                "cif_path": record["cif_path"],
                "reasons": " | ".join(problems),
                **{f"fact_{k}": json.dumps(v, default=str) if isinstance(v, (dict, list))
                   else v for k, v in facts.items()},
            })
        else:
            accepted.append({**record, **{
                "audit_residues": facts.get("residues"),
                "audit_sc_completeness": facts.get("sc_completeness"),
                "audit_resolution": facts.get("resolution"),
                "audit_resolution_missing": bool(facts.get("resolution_missing")),
                "source_pool": record["pool"],
            }})
        if (n + 1) % 50 == 0 or n + 1 == len(candidates):
            print(f"  {n + 1}/{len(candidates)}  accepted {len(accepted)}",
                  flush=True)

    # ---- write ------------------------------------------------------------
    suffix = args.train_pool
    train_out = out / f"train_{suffix}.parquet"
    pd.DataFrame(accepted).to_parquet(train_out)

    calib_pool = calib[calib.pool.isin(pools)].reset_index(drop=True)
    calib_out = out / f"calibration_{suffix}.parquet"
    calib_pool.to_parquet(calib_out)
    val.to_parquet(out / "validation.parquet")

    if exclusions:
        with (out / "exclusions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({k for e in exclusions for k in e})
            )
            writer.writeheader()
            writer.writerows(exclusions)

    audit = {
        "settings": {
            "train_pool": args.train_pool,
            "max_resolution": args.max_resolution,
            "min_sc_completeness": args.min_sc_completeness,
            "identity_threshold": args.identity_threshold,
            "coverage_threshold": args.coverage_threshold,
            "identity_definition":
                "LOCAL pairwise (Smith-Waterman, BLOSUM62, gap -11/-1); "
                "identity = matches / aligned columns, where aligned columns "
                "are the non-gap-paired positions; coverage = aligned columns "
                "/ len(shorter sequence). 'X' never counts as a match. A "
                "SCREEN, not structural clustering.",
            "benchmark_targets": bench_note,
            "cluster_id_note":
                "cluster_id is the PDB ID on these rows, not a homology "
                "cluster; it is carried through as provenance only",
        },
        "counts": {
            "candidates": len(candidates),
            "accepted": len(accepted),
            "excluded": len(exclusions),
            "calibration_rows": len(calib_pool),
            "validation_rows": len(val),
            "validation_by_pool": val.pool.value_counts().to_dict(),
            "example_id_overlap_train_heldout": len(overlap),
            "forbidden_chains": len(forbidden),
        },
        "exclusion_reasons": dict(reasons.most_common()),
        "outputs": {
            "train": str(train_out), "calibration": str(calib_out),
            "validation": str(out / "validation.parquet"),
            "exclusions": str(out / "exclusions.csv") if exclusions else None,
        },
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))

    print()
    print(f"accepted {len(accepted)}/{len(candidates)}; excluded {len(exclusions)}")
    for reason, count in reasons.most_common(10):
        print(f"  {count:4d}  {reason}")
    print(f"\nvalidation panel by pool: {val.pool.value_counts().to_dict()} "
          "(report PDB and TED separately)")
    print(f"wrote {train_out}, {calib_out}, {out / 'validation.parquet'}, "
          f"{out / 'audit.json'}")


if __name__ == "__main__":
    main()
