#!/usr/bin/env python3
"""Build a Protenix-style protein-protein index for the PPDiff PPBench release.

The official PPBench JSON is roughly 11 GiB and stores its columns as large
parallel arrays.  This script streams the sequence and target-mask columns and
skips the coordinate payload, so peak memory is proportional to one split's
sequence-length vector rather than the full JSON document.

PPBench contains C-alpha traces and does not publish the originating PDB IDs or
full-atom mmCIF assemblies.  The output therefore mirrors the useful filtering
columns in a Protenix index, while leaving PDB-specific fields empty and adding
an exact JSON record locator.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import BinaryIO


SPLITS = ("train", "valid", "test")
INDEX_COLUMNS = (
    "entity_1_id",
    "chain_1_id",
    "mol_1_type",
    "cluster_1_id",
    "entity_2_id",
    "chain_2_id",
    "mol_2_type",
    "cluster_2_id",
    "cluster_id",
    "pdb_id",
    "assembly_id",
    "release_date",
    "num_tokens",
    "num_prot_chains",
    "resolution",
    "type",
    "mol_type_group",
    "sub_mol_1_type",
    "sub_mol_2_type",
    "eval_type",
    "source_dataset",
    "source_split",
    "source_record_index",
    "source_record_id",
    "source_json_pointer",
    "target_tokens",
    "binder_tokens",
)


class BufferedScanner:
    """Small delimiter scanner optimized for very large, regular JSON files."""

    def __init__(self, handle: BinaryIO, chunk_size: int = 64 * 1024 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = b""
        self.pos = 0
        self.eof = False

    def _fill(self) -> None:
        if self.pos:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _ensure(self, count: int = 1) -> None:
        while len(self.buffer) - self.pos < count and not self.eof:
            self._fill()
        if len(self.buffer) - self.pos < count:
            raise EOFError("Unexpected end of PPBench JSON")

    def peek(self) -> int:
        self._ensure()
        return self.buffer[self.pos]

    def read_byte(self) -> int:
        value = self.peek()
        self.pos += 1
        return value

    def skip_ws(self) -> None:
        while True:
            value = self.peek()
            if value not in b" \t\r\n":
                return
            self.pos += 1

    def expect(self, value: int) -> None:
        observed = self.read_byte()
        if observed != value:
            raise ValueError(
                f"Expected byte {chr(value)!r}, observed {chr(observed)!r}"
            )

    def read_until(self, delimiter: bytes, *, collect: bool = True) -> bytes:
        parts: list[bytes] = []
        keep = max(len(delimiter) - 1, 0)
        while True:
            index = self.buffer.find(delimiter, self.pos)
            if index >= 0:
                if collect:
                    parts.append(self.buffer[self.pos : index])
                self.pos = index + len(delimiter)
                return b"".join(parts) if collect else b""
            if self.eof:
                raise EOFError(f"Delimiter not found: {delimiter!r}")
            end = max(self.pos, len(self.buffer) - keep)
            if collect:
                parts.append(self.buffer[self.pos : end])
            self.pos = end
            self._fill()

    def skip_until(self, delimiter: bytes) -> None:
        self.read_until(delimiter, collect=False)


def read_sequence_lengths(scanner: BufferedScanner, split: str) -> list[int]:
    scanner.skip_until(f'"{split}": {{"seqs": ['.encode())
    lengths: list[int] = []
    scanner.skip_ws()
    if scanner.peek() == ord("]"):
        scanner.read_byte()
        return lengths
    while True:
        scanner.expect(ord('"'))
        sequence = scanner.read_until(b'"')
        lengths.append(len(sequence))
        scanner.skip_ws()
        separator = scanner.read_byte()
        if separator == ord("]"):
            return lengths
        if separator != ord(","):
            raise ValueError(f"Malformed sequence array in split {split!r}")
        scanner.skip_ws()


def iter_target_counts(scanner: BufferedScanner, split: str):
    scanner.skip_until(b'"target": [')
    scanner.skip_ws()
    if scanner.peek() == ord("]"):
        scanner.read_byte()
        return
    record_index = 0
    while True:
        scanner.expect(ord("["))
        target_mask = scanner.read_until(b"]")
        yield record_index, target_mask.count(b"1"), target_mask.count(b"0")
        record_index += 1
        scanner.skip_ws()
        separator = scanner.read_byte()
        if separator == ord("]"):
            return
        if separator != ord(","):
            raise ValueError(f"Malformed target array in split {split!r}")
        scanner.skip_ws()


def index_row(
    split: str,
    record_index: int,
    num_tokens: int,
    target_tokens: int,
    binder_tokens: int,
) -> dict[str, object]:
    record_id = f"ppdiff_{split}_{record_index:06d}"
    return {
        "entity_1_id": 1,
        "chain_1_id": "target",
        "mol_1_type": "prot",
        "cluster_1_id": "",
        "entity_2_id": 2,
        "chain_2_id": "binder",
        "mol_2_type": "prot",
        "cluster_2_id": "",
        "cluster_id": "",
        "pdb_id": "",
        "assembly_id": "",
        "release_date": "",
        "num_tokens": num_tokens,
        "num_prot_chains": 2,
        "resolution": "",
        "type": "interface",
        "mol_type_group": "prot_prot",
        "sub_mol_1_type": "prot",
        "sub_mol_2_type": "prot",
        "eval_type": "prot_prot",
        "source_dataset": "PPDiff_PPBench",
        "source_split": split,
        "source_record_index": record_index,
        "source_record_id": record_id,
        "source_json_pointer": f"/PDB/{split}/{record_index}",
        "target_tokens": target_tokens,
        "binder_tokens": binder_tokens,
    }


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    input_path: Path,
    output_index: Path,
    summary_path: Path,
    min_n_token: int,
    max_n_token: int,
) -> None:
    output_index.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, dict[str, object]] = {}
    total_source = 0
    total_selected = 0

    with (
        input_path.open("rb") as source,
        gzip.open(output_index, "wt", newline="") as output,
    ):
        scanner = BufferedScanner(source)
        writer = csv.DictWriter(output, fieldnames=INDEX_COLUMNS)
        writer.writeheader()

        for split in SPLITS:
            sequence_lengths = read_sequence_lengths(scanner, split)
            source_rows = len(sequence_lengths)
            selected_rows = 0
            rejected = Counter()
            length_min = None
            length_max = None
            target_rows = 0

            for record_index, target_tokens, binder_tokens in iter_target_counts(
                scanner, split
            ):
                target_rows += 1
                if record_index >= source_rows:
                    raise ValueError(
                        f"{split}: target array has more rows than sequence array"
                    )
                num_tokens = sequence_lengths[record_index]
                if target_tokens + binder_tokens != num_tokens:
                    rejected["mask_sequence_length_mismatch"] += 1
                    continue
                if target_tokens == 0:
                    rejected["missing_target_protein"] += 1
                    continue
                if binder_tokens == 0:
                    rejected["missing_binder_protein"] += 1
                    continue
                if num_tokens < min_n_token:
                    rejected["below_min_n_token"] += 1
                    continue
                if max_n_token > 0 and num_tokens > max_n_token:
                    rejected["above_max_n_token"] += 1
                    continue
                writer.writerow(
                    index_row(
                        split,
                        record_index,
                        num_tokens,
                        target_tokens,
                        binder_tokens,
                    )
                )
                selected_rows += 1
                length_min = (
                    num_tokens if length_min is None else min(length_min, num_tokens)
                )
                length_max = (
                    num_tokens if length_max is None else max(length_max, num_tokens)
                )

            if target_rows != source_rows:
                raise ValueError(
                    f"{split}: {source_rows} sequence rows but {target_rows} target rows"
                )
            split_summary[split] = {
                "source_rows": source_rows,
                "selected_protein_protein_rows": selected_rows,
                "rejected_rows": dict(sorted(rejected.items())),
                "selected_num_tokens_min": length_min,
                "selected_num_tokens_max": length_max,
            }
            total_source += source_rows
            total_selected += selected_rows

    summary = {
        "dataset": "PPDiff PPBench",
        "input_json": str(input_path.resolve()),
        "input_json_bytes": input_path.stat().st_size,
        "input_json_sha256": sha256(input_path),
        "selection": {
            "type": "interface",
            "mol_1_type": "prot",
            "mol_2_type": "prot",
            "requires_nonempty_target_and_binder": True,
            "min_n_token": min_n_token,
            "max_n_token": max_n_token,
        },
        "splits": split_summary,
        "source_rows_total": total_source,
        "selected_protein_protein_rows_total": total_selected,
        "output_index": str(output_index.resolve()),
        "notes": [
            "PPBench stores one C-alpha coordinate per residue, not full-atom mmCIF.",
            "The official release does not include source PDB IDs or cluster IDs.",
            "Each chain-pair interface is oriented twice (target/binder and binder/target).",
        ],
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-n-token", type=int, default=0)
    parser.add_argument(
        "--max-n-token",
        type=int,
        default=0,
        help="Maximum total tokens; 0 keeps all lengths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(
        args.input,
        args.output_index,
        args.summary,
        args.min_n_token,
        args.max_n_token,
    )


if __name__ == "__main__":
    main()
