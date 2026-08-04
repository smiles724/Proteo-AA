#!/usr/bin/env python3
"""Convert the official PINDER index into Proteo-AA training manifests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INDEX_COLUMNS = [
    "split",
    "id",
    "pdb_id",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
    "uniprot_R",
    "uniprot_L",
    "holo_R",
    "holo_L",
    "chain_R",
    "chain_L",
    "contains_antibody",
    "contains_antigen",
    "contains_enzyme",
]
METADATA_COLUMNS = [
    "id",
    "release_date",
    "resolution",
    "assembly",
    "complex_type",
    "length1",
    "length2",
    "length_resolved_1",
    "length_resolved_2",
    "entity_id_R",
    "entity_id_L",
    "buried_sasa",
    "intermolecular_contacts",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def string_series(frame: pd.DataFrame, name: str, missing: object = "") -> pd.Series:
    """Return a plain-object string series, including for Arrow dictionary data."""
    series = frame[name].astype(object)
    return series.where(series.notna(), missing).astype(str)


def output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    total = frame["length1"].astype(int) + frame["length2"].astype(int)
    out = pd.DataFrame(
        {
            "entity_1_id": string_series(frame, "entity_id_R", 1),
            "chain_1_id": "R",
            "mol_1_type": "prot",
            "cluster_1_id": string_series(frame, "cluster_id_R"),
            "entity_2_id": string_series(frame, "entity_id_L", 2),
            "chain_2_id": "L",
            "mol_2_type": "prot",
            "cluster_2_id": string_series(frame, "cluster_id_L"),
            "cluster_id": string_series(frame, "cluster_id"),
            "pdb_id": string_series(frame, "pdb_id"),
            "assembly_id": string_series(frame, "assembly"),
            "release_date": string_series(frame, "release_date"),
            "num_tokens": total,
            "num_prot_chains": 2,
            "resolution": frame["resolution"].astype(float),
            "type": "interface",
            "mol_type_group": "prot_prot",
            "sub_mol_1_type": "prot",
            "sub_mol_2_type": "prot",
            "eval_type": "prot_prot",
            "source_dataset": "PINDER",
            "source_release": "2024-02",
            "source_split": string_series(frame, "split"),
            "pinder_id": string_series(frame, "id"),
            "pdb_path": "pdbs/" + string_series(frame, "id") + ".pdb",
            "source_chain_R": string_series(frame, "chain_R"),
            "source_chain_L": string_series(frame, "chain_L"),
            "converted_binder_chain": "B",
            "target_tokens": frame["length1"].astype(int),
            "binder_tokens": frame["length2"].astype(int),
            "target_resolved_tokens": frame["length_resolved_1"].astype(int),
            "binder_resolved_tokens": frame["length_resolved_2"].astype(int),
            "uniprot_R": string_series(frame, "uniprot_R"),
            "uniprot_L": string_series(frame, "uniprot_L"),
            "complex_type": string_series(frame, "complex_type"),
            "contains_antibody": frame["contains_antibody"].astype(bool),
            "contains_antigen": frame["contains_antigen"].astype(bool),
            "contains_enzyme": frame["contains_enzyme"].astype(bool),
            "buried_sasa": frame["buried_sasa"].astype(float),
            "intermolecular_contacts": frame["intermolecular_contacts"].astype(int),
        }
    )
    return out


def prepare(args: argparse.Namespace) -> None:
    index_file = pq.ParquetFile(args.index)
    metadata_file = pq.ParquetFile(args.metadata)
    if index_file.metadata.num_rows != metadata_file.metadata.num_rows:
        raise ValueError("PINDER index and metadata row counts differ")
    if index_file.metadata.num_row_groups != metadata_file.metadata.num_row_groups:
        raise ValueError("PINDER index and metadata row-group counts differ")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    csv_handle = gzip.open(args.output_csv, "wt", newline="")
    wrote_csv_header = False
    rejection = Counter()
    source_by_split = Counter()
    selected_by_split = Counter()
    selected_pdbs: dict[str, set[str]] = defaultdict(set)
    selected_clusters: dict[str, set[str]] = defaultdict(set)
    selected_homodimers = Counter()
    selected_antibodies = Counter()
    selected_complex_types = Counter()
    selected_token_sum = Counter()
    selected_binder_sum = Counter()
    token_min: dict[str, int] = {}
    token_max: dict[str, int] = {}
    total_rows = 0
    selected_rows = 0
    allowed_splits = {"train", "val", "test"}
    binder_limit = int(args.crop_size * args.max_binder_fraction)

    try:
        for row_group in range(index_file.metadata.num_row_groups):
            idx = index_file.read_row_group(
                row_group, columns=INDEX_COLUMNS
            ).to_pandas()
            meta = metadata_file.read_row_group(
                row_group, columns=METADATA_COLUMNS
            ).to_pandas()
            if not idx["id"].equals(meta["id"]):
                raise ValueError(f"PINDER row group {row_group} is not ID-aligned")
            meta = meta.drop(columns=["id"])
            frame = pd.concat([idx.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
            total_rows += len(frame)
            source_by_split.update(
                frame["split"].astype(str).value_counts().to_dict()
            )

            keep = pd.Series(True, index=frame.index)
            checks = [
                (frame["split"].isin(allowed_splits), "invalid_split"),
                (frame["holo_R"].fillna(False) & frame["holo_L"].fillna(False), "missing_holo_chain"),
                (frame[["length1", "length2", "length_resolved_1", "length_resolved_2"]].notna().all(axis=1), "missing_length"),
            ]
            for condition, name in checks:
                failed = keep & ~condition
                rejection[name] += int(failed.sum())
                keep &= condition

            total_tokens = frame["length1"].fillna(-1) + frame["length2"].fillna(-1)
            for condition, name in [
                (total_tokens >= args.min_n_token, "below_min_n_token"),
                ((total_tokens <= args.max_n_token) if args.max_n_token > 0 else pd.Series(True, index=frame.index), "above_max_n_token"),
                (frame["length2"].fillna(binder_limit + 1) <= binder_limit, "binder_exceeds_crop_fraction"),
            ]:
                failed = keep & ~condition
                rejection[name] += int(failed.sum())
                keep &= condition

            selected = frame.loc[keep].copy()
            out = output_frame(selected)
            selected_rows += len(out)
            if out.empty:
                continue
            for split, group in out.groupby("source_split", sort=False):
                n = len(group)
                selected_by_split[split] += n
                selected_pdbs[split].update(group["pdb_id"].astype(str))
                selected_clusters[split].update(group["cluster_id"].astype(str))
                selected_homodimers[split] += int(
                    (group["uniprot_R"] == group["uniprot_L"]).sum()
                )
                selected_antibodies[split] += int(group["contains_antibody"].sum())
                selected_token_sum[split] += int(group["num_tokens"].sum())
                selected_binder_sum[split] += int(group["binder_tokens"].sum())
                selected_complex_types.update(group["complex_type"].value_counts().to_dict())
                lo, hi = int(group["num_tokens"].min()), int(group["num_tokens"].max())
                token_min[split] = min(token_min.get(split, lo), lo)
                token_max[split] = max(token_max.get(split, hi), hi)

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output_parquet, table.schema, compression="zstd")
            writer.write_table(table)
            out.to_csv(csv_handle, index=False, header=not wrote_csv_header)
            wrote_csv_header = True
    finally:
        if writer is not None:
            writer.close()
        csv_handle.close()

    if total_rows != selected_rows + sum(rejection.values()):
        raise AssertionError("Selection accounting does not sum to the source row count")
    summary = {
        "dataset": "PINDER",
        "release": "2024-02",
        "source_index": str(args.index.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "source_index_sha256": sha256(args.index),
        "source_metadata_sha256": sha256(args.metadata),
        "source_rows": total_rows,
        "source_splits": dict(sorted(source_by_split.items())),
        "selection": {
            "allowed_splits": sorted(allowed_splits),
            "requires_holo_R_and_L": True,
            "min_n_token": args.min_n_token,
            "max_n_token": args.max_n_token,
            "crop_size": args.crop_size,
            "max_binder_fraction": args.max_binder_fraction,
            "max_binder_tokens": binder_limit,
            "binder_chain": "L",
        },
        "rejected_rows": dict(sorted(rejection.items())),
        "selected_rows": selected_rows,
        "selected_unique_pdb_ids": len(set().union(*selected_pdbs.values())),
        "selected_unique_clusters": len(set().union(*selected_clusters.values())),
        "selected_complex_types": dict(sorted(selected_complex_types.items())),
        "splits": {
            split: {
                "rows": selected_by_split[split],
                "unique_pdb_ids": len(selected_pdbs[split]),
                "unique_clusters": len(selected_clusters[split]),
                "homodimer_rows": selected_homodimers[split],
                "antibody_rows": selected_antibodies[split],
                "num_tokens_min": token_min.get(split),
                "num_tokens_max": token_max.get(split),
                "num_tokens_mean": (
                    selected_token_sum[split] / selected_by_split[split]
                    if selected_by_split[split]
                    else None
                ),
                "binder_tokens_mean": (
                    selected_binder_sum[split] / selected_by_split[split]
                    if selected_by_split[split]
                    else None
                ),
            }
            for split in ("train", "val", "test")
        },
        "output_parquet": str(args.output_parquet.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "structure_files_expected": selected_rows,
        "structure_format": "PDB full-heavy-atom holo dimer; chains R and L",
        "conversion": "lazy Protenix pdb_to_cif; converted chains A and B",
    }
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-n-token", type=int, default=16)
    parser.add_argument("--max-n-token", type=int, default=1536)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--max-binder-fraction", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
