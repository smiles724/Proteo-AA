#!/usr/bin/env python3
"""Check a backbone collection against its own manifest.

    python scripts/verify_binder_backbones.py runs/binder_bench/backbones
    python scripts/verify_binder_backbones.py <dir> --quick   # digests only

Needs nothing but torch: no PXDesign, no Protenix, no GPU. That is the point --
it runs on the cluster that *consumes* the collection, which is the one that
cannot regenerate it.

This exists because the collection is not reproducible. Two identical
invocations of `cache_binder_backbones.py` at the same seed on the same GPU
produce backbones up to 0.55 A apart, so there is no way to re-derive a design
and compare. A transfer that silently truncated one file, or a directory that
accidentally mixes designs from two generation runs, would show up downstream
as an arm difference rather than as a broken file. The digests are the only
thing standing between those two readings.

Checked per design: the file exists, its sha256 matches the manifest, it
loads, and its topology is internally consistent -- `atom_to_token_idx` spans
the atom axis of `x0`, `a_token` is one row per token, and the design mask
selects exactly the requested binder length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check_one(record: dict, base: Path, *, quick: bool) -> list[str]:
    """Problems with one design, empty when it is sound."""
    import torch

    problems: list[str] = []
    # Resolve against the manifest's directory, not the recorded path: the
    # recorded one is the cluster the collection was generated on.
    path = base / "designs" / f"{record['design_id']}.pt"
    if not path.is_file():
        return [f"missing: {path}"]

    expected = record.get("sha256")
    if expected is None:
        problems.append("no sha256 in the manifest; generated before digests")
    else:
        actual = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if actual != expected:
            return [f"sha256 {actual[:12]} != manifest {expected[:12]}"]
    if quick:
        return problems

    payload = torch.load(path, map_location="cpu", weights_only=False)
    topology = payload.get("topology")
    if topology is None:
        return problems + ["no topology; the design stage cannot read x0"]

    n_atom = payload["x0"].reshape(-1, 3).shape[0]
    n_tokens = int(topology["n_tokens"])
    a2t = topology["atom_to_token_idx"]
    if a2t.numel() != n_atom:
        problems.append(f"atom_to_token_idx has {a2t.numel()} entries for {n_atom} atoms")
    if int(a2t.max()) >= n_tokens:
        problems.append(f"atom_to_token_idx reaches token {int(a2t.max())} of {n_tokens}")
    if payload["a_token"].reshape(-1, payload["a_token"].shape[-1]).shape[0] != n_tokens:
        problems.append(f"a_token is not one row per token ({n_tokens})")
    for key in ("residue_index", "asym_id", "design_mask"):
        if topology[key].numel() != n_tokens:
            problems.append(f"{key} has {topology[key].numel()} entries for {n_tokens} tokens")
    designed = int(topology["design_mask"].sum())
    if designed != int(record["binder_length"]):
        problems.append(
            f"design mask selects {designed} tokens, binder_length is "
            f"{record['binder_length']}"
        )
    for name in ("atom_name", "res_name"):
        got = len(topology["annotations"].get(name, ()))
        if got != n_atom:
            problems.append(f"annotation {name} has {got} entries for {n_atom} atoms")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("collection", help="directory holding backbones.json")
    parser.add_argument("--quick", action="store_true",
                        help="digests only; skip loading every design")
    args = parser.parse_args()

    base = Path(args.collection).expanduser().resolve()
    manifest = json.loads((base / "backbones.json").read_text())
    designs = manifest["designs"]

    failed: dict[str, list[str]] = {}
    for record in designs:
        problems = check_one(record, base, quick=args.quick)
        if problems:
            failed[record["design_id"]] = problems

    per_target: dict[str, int] = {}
    for record in designs:
        per_target[record["target"]] = per_target.get(record["target"], 0) + 1
    for target, count in sorted(per_target.items()):
        print(f"{target:<8} {count:>4}")

    sources = manifest.get("sources", {})
    print(f"\nruntime: protenix {sources.get('protenix', {}).get('version')}, "
          f"pxdesign {sources.get('pxdesign', {}).get('version')}, "
          f"weights {sources.get('pxdesign_weights', {}).get('sha256', '?')[:12]}")
    print(f"{len(designs) - len(failed)}/{len(designs)} design(s) verified"
          + (" (digests only)" if args.quick else ""))

    if failed:
        for design_id, problems in sorted(failed.items())[:20]:
            print(f"  FAIL {design_id}: {'; '.join(problems)}")
        raise SystemExit(f"{len(failed)} design(s) failed verification")


if __name__ == "__main__":
    main()
