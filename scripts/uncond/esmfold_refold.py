#!/usr/bin/env python3
"""ESMFold a table of sequences. Runs in the esmfold venv, nothing else.

    python scripts/uncond/esmfold_refold.py --sequences seqs.csv --out refolds/

``--sequences`` is a CSV with ``fold_id`` and ``sequence``. One prediction per
row, written to ``<out>/<fold_id>.pdb``, plus ``refolds.csv`` carrying pLDDT.

Folds are cached by ``fold_id``: a rerun skips anything already on disk, which
matters because the unconditional benchmark folds 9 sequences per sample
(1 co-designed + 8 ProteinMPNN) and a length sweep is thousands of calls.

Uses transformers' ``EsmForProteinFolding`` rather than ``esm.pretrained``:
the latter needs openfold and its CUDA extensions, the former ships a pure
PyTorch port. Same weights.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

#: ESMFold has no token for unknown residues. The benchmark spec says to
#: substitute glycine, so do it here, once, and report how often.
UNKNOWN = set("XBZJOU")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", required=True,
                        help="CSV with fold_id,sequence")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=os.environ.get(
        "ESMFOLD_WEIGHTS", "facebook/esmfold_v1"))
    parser.add_argument("--num-recycles", type=int, default=None,
                        help="None uses the model default")
    parser.add_argument("--max-length", type=int, default=600,
                        help="skip longer; ESMFold memory is quadratic")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="attention chunking; lower if OOM")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    out = Path(args.out)
    (out / "pdb").mkdir(parents=True, exist_ok=True)

    with open(args.sequences, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("sequence")]
    rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
    todo = [r for r in rows
            if not (out / "pdb" / f"{r['fold_id']}.pdb").is_file()]
    print(f"{len(rows)} row(s) in shard, {len(todo)} to fold "
          f"({len(rows) - len(todo)} cached)")
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    model = EsmForProteinFolding.from_pretrained(args.weights)
    model = model.to(device).eval()
    if device == "cuda":
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(args.chunk_size)

    written, substituted, skipped = 0, 0, 0
    started = time.time()
    records = []
    for index, row in enumerate(todo, 1):
        sequence = row["sequence"].strip().upper()
        if len(sequence) > args.max_length:
            skipped += 1
            continue
        clean = "".join("G" if c in UNKNOWN else c for c in sequence)
        substituted += sum(1 for c in sequence if c in UNKNOWN)

        with torch.no_grad():
            tokens = tokenizer([clean], return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(device)
            kwargs = ({} if args.num_recycles is None
                      else {"num_recycles": args.num_recycles})
            output = model(tokens, **kwargs)
        pdb = model.output_to_pdb(output)[0]
        (out / "pdb" / f"{row['fold_id']}.pdb").write_text(pdb)
        plddt = float(output["plddt"][0, :, 1].mean())
        records.append({"fold_id": row["fold_id"], "plddt": round(plddt, 4),
                        "length": len(clean)})
        written += 1
        if index % 25 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}  {time.time() - started:.0f}s")

    path = out / f"refolds.{args.shard_index}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_id", "plddt", "length"])
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {written} structure(s) -> {out/'pdb'}; {path}")
    if substituted:
        print(f"NOTE: {substituted} unknown residue(s) substituted with glycine")
    if skipped:
        print(f"NOTE: {skipped} sequence(s) skipped over --max-length {args.max_length}")


if __name__ == "__main__":
    main()
