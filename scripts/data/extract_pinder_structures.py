#!/usr/bin/env python3
"""Extract only manifest-selected PINDER dimers from the official archive."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(
        pd.read_parquet(args.manifest, columns=["pinder_id"])["pinder_id"].astype(str)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    existing = 0
    archive_matches = 0

    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".pdb"):
                continue
            pinder_id = Path(member.filename).stem
            if pinder_id not in selected:
                continue
            archive_matches += 1
            pdb_id = pinder_id[:4].lower()
            destination = args.output_root / pdb_id[1:3] / f"{pinder_id}.pdb"
            if destination.is_file() and destination.stat().st_size == member.file_size:
                existing += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".pdb.tmp.{os.getpid()}")
            with archive.open(member) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            os.replace(temporary, destination)
            extracted += 1
            if extracted % args.progress_every == 0:
                print(f"extracted={extracted:,} existing={existing:,}", flush=True)

    missing = len(selected) - archive_matches
    summary = {
        "archive": str(args.archive.resolve()),
        "manifest": str(args.manifest.resolve()),
        "selected_ids": len(selected),
        "archive_matches": archive_matches,
        "newly_extracted": extracted,
        "already_present": existing,
        "missing_from_archive": missing,
        "output_root": str(args.output_root.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    if missing:
        raise RuntimeError(f"{missing:,} selected PINDER IDs were absent from the archive")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
