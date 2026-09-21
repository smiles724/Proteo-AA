#!/usr/bin/env python3
"""Validate the binder-role masks and the §5 leakage controls on real complexes.

    python scripts/audit_binder_masks.py \
        --manifest configs/gen_stress_prepared.marlowe.parquet \
        --out runs/mask_audit

Runs BEFORE any objective is wired, and on the 31 prepared dimers rather than
on synthetic tensors, because the leak this is looking for lives in the
featurizer's output and not in the mask arithmetic.

Three routes by which the binder's native identity can reach a model that is
being asked to predict it:

  1. FaMPNN's ``aatype`` at supervised positions.
  2. FaMPNN's side-chain coordinates at those positions -- a side chain
     identifies its residue almost exactly.
  3. PXDesign's ``restype`` conditioning, which feeds ``a_token`` and
     therefore the residual. This is the one the sequence model's own inputs
     cannot reveal, and the only one that needs a real featurization to test.

Route 3 is checked vocabulary-independently. ``restype`` is a 36-way encoding
and ``aa_clean`` a 21-way one, so comparing indices needs a mapping nobody has
written down. Instead: the featurizer's contract is that the design region is
scrubbed to a single placeholder, so every design-region row of ``restype``
should be identical. One distinct row means the channel carries no sequence
information at all. More than one means it varies with something, and the
report says whether it varies with the native identity.

These 31 are the VAL split. They are for building and validating the mask
plumbing. They are NOT the training set -- training on them spends the
held-out set that §7 and §9 need.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "gen_stress_prepared.marlowe.parquet"


def audit_one(row, *, mask_fraction: float, crop_size: int, seed: int) -> dict[str, Any]:
    from pxf.backbone.chain_ids import featurizer_chain_id
    from pxf.backbone.driver import featurize_structures, to_featurized
    from pxf.couple.binder_masks import (
        audit_leakage, build_masks, pxdesign_restype_leak,
    )
    from pxf.couple.binder_residual import ChainRoles

    out: dict[str, Any] = {"example_id": row.example_id, "pool": row.pool,
                           "split": row.split}

    # The author id from the manifest is NOT what binder_chain_ids matches.
    label = featurizer_chain_id(row.cif_path, row.converted_binder_chain)
    out["binder_chain_author"] = str(row.converted_binder_chain)
    out["binder_chain_label"] = str(label)

    sample_id, dataset = featurize_structures(
        [row.cif_path], crop_size=crop_size,
        binder_chain_ids=[label], parser_dataset="Distillation",
    )[0]
    structure = to_featurized(sample_id, dataset[0])

    roles = ChainRoles(binder=structure.design_mask.reshape(-1).bool().cpu())
    generator = torch.Generator().manual_seed(seed)
    masks = build_masks(
        roles, structure.aatype.reshape(1, -1).cpu(),
        mask_fraction=mask_fraction, generator=generator,
    )
    out["masks"] = masks.identity()

    report = audit_leakage(masks)
    restype = structure.feature_dict.get("restype")
    if restype is None:
        report["problems"].append("no `restype` in the feature dict")
        out["route3"] = {"checked": False, "reason": "restype absent"}
    else:
        route3 = pxdesign_restype_leak(
            restype.cpu(), roles.binder, masks.aatype_true.reshape(-1)
        )
        out["route3"] = route3
        if route3.get("leak"):
            report["problems"].append(
                "ROUTE 3 (PXDesign restype): the design region's conditioning "
                f"varies with the native identity "
                f"({route3['n_distinct_restype_rows']} distinct rows over "
                f"{route3.get('n_distinct_native_identities')} identities)"
            )
        elif route3.get("checked") and not route3.get("scrubbed"):
            report["problems"].append(
                "ROUTE 3 (PXDesign restype): the design region is not scrubbed "
                f"to one placeholder ({route3['n_distinct_restype_rows']} "
                "distinct rows); it may still be safe, but the featurizer's "
                "contract does not hold and this needs explaining"
            )
        report["pass"] = not report["problems"]

    out["leakage"] = report
    out["tokens"] = int(structure.topology.num_tokens)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=None)
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.manifest)
    if args.limit:
        frame = frame.head(args.limit)

    results, failures, errors = [], 0, 0
    for row in frame.itertuples():
        try:
            record = audit_one(
                row, mask_fraction=args.mask_fraction,
                crop_size=args.crop_size, seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - a failure to featurize is a finding
            print(f"ERROR {row.example_id}: {type(exc).__name__}: {str(exc)[:120]}")
            results.append({"example_id": row.example_id,
                            "error": f"{type(exc).__name__}: {exc}"})
            errors += 1
            continue

        ok = record["leakage"]["pass"]
        failures += 0 if ok else 1
        r3 = record["route3"]
        print(
            f"{'ok  ' if ok else 'LEAK'} {record['example_id']:<7} "
            f"chain {record['binder_chain_author']}->{record['binder_chain_label']}  "
            f"tokens {record['tokens']:>4}  "
            f"binder {record['masks']['n_binder']:>3}  "
            f"supervised {record['masks']['n_seq_supervised']:>3}  "
            f"restype_rows {r3.get('n_distinct_restype_rows', '?')}"
        )
        for problem in record["leakage"]["problems"]:
            print(f"       {problem}")
        results.append(record)

    clean = [r for r in results if not r.get("error") and r["leakage"]["pass"]]
    summary = {
        "manifest": args.manifest,
        "mask_fraction": args.mask_fraction,
        "n_examples": len(results),
        "n_clean": len(clean),
        "n_leaking": failures,
        "n_errors": errors,
        "split": sorted({r.get("split") for r in results if r.get("split")}),
        "note": (
            "VAL split. Validation of the mask plumbing only -- training on "
            "these spends the held-out set."
        ),
        "examples": results,
    }
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mask_audit.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"\nwrote {out / 'mask_audit.json'}")

    print(f"\n{len(clean)}/{len(results)} clean, {failures} leaking, {errors} errors")
    if failures or errors:
        raise SystemExit(f"{failures} leak(s), {errors} error(s)")
    print("no leakage on any route; the mask plumbing is safe to wire an objective to")


if __name__ == "__main__":
    main()
