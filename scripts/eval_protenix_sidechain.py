#!/usr/bin/env python3
"""Side-chain packing on Protenix's recentPDB eval split, before and after a tune.

THE TASK is the packing task: native backbone in, native sequence in, side
chains out, scored against the deposited coordinates. Same definition as
``eval_monomer_sidechain.py``; the difference is the dataset and the masking.

**Why this split.** ``indices/recentPDB_low_homology_maxtoken1536.csv`` holds
1,818 entries released 2022-05-04 to 2023-01-11, and the training index is cut at
2021-09-30. The intersection is empty, so a fine-tune measured here cannot be
reading back its own training data, and the split is already low-homology. 1,642
of the 1,818 survive parsing and the completeness cut, which is enough to see a
regression rather than guess at one.

**Why the mask.** Metrics are reported on the residues whose deposited side
chains are actually trustworthy -- ``supervise_mask`` from the eval mask set --
because scoring against zero-occupancy, altloc-tie or B>80 side chains measures
crystallographic noise, and noise moves in both directions. The unmasked figures
are computed in the same pass and reported alongside, so it is visible whether
the filter changed the conclusion rather than assumed.

    # baseline: the released weights
    python scripts/eval_protenix_sidechain.py --weights 0.0 --out runs/eval_before

    # after: the same command with a checkpoint
    python scripts/eval_protenix_sidechain.py --checkpoint runs/ft/checkpoints/final.pt \
        --out runs/eval_after

    # and the comparison
    python scripts/eval_protenix_sidechain.py --compare runs/eval_before runs/eval_after

Metrics come from Proteo-AA's own ``pxdesign_train.sidechain`` implementations, so
they are comparable to every earlier run. Dataset figures sum counts across
targets and divide once, making them atom-weighted rather than an average of
per-target averages.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.eval_protenix")

# The headline numbers the fine-tune has to not break, plus the diagnostics that
# say why if it does.
REPORT = {
    "rmsd": ("symmetry_rmsd",),
    "rotamer recovery": (
        "rotamer_recovery",
        "chi_recovery_20deg",
        "chi_recovery_40deg",
        "chi1_accuracy_20deg",
        "chi1_chi2_accuracy_20deg",
    ),
    "lddt": ("lddt_sc_sc", "lddt_sc_env"),
    "covalent failures": (
        "bad_bond_fraction",
        "bond_mae",
        "rotamer_outlier_fraction_40deg",
        "completeness",
    ),
}
# Regression is judged on these: lower is better for the first, higher for the rest.
LOWER_IS_BETTER = (
    "symmetry_rmsd",
    "bad_bond_fraction",
    "bond_mae",
    "rotamer_outlier_fraction_40deg",
)
HEADLINE = ("symmetry_rmsd", "rotamer_recovery", "chi_recovery_20deg", "lddt_sc_sc")
# A metric that did not move is not a regression, and neither is float noise on
# an atom-weighted ratio over ~10^6 atoms. Only a move this large is adjudicated.
REGRESSION_TOLERANCE = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        default=None,
        help="two output dirs from earlier runs; print the delta table and exit",
    )
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument(
        "--eval-index",
        default=None,
        help="Protenix eval index CSV (default: recentPDB_low_homology_maxtoken1536)",
    )
    parser.add_argument(
        "--mask-root",
        default="/hai/scratch/yfsun/protenix_sidechain/out_eval_fampnn_strictB",
        help="eval-split mask set; built by process.py --ids-file + make_fampnn.py",
    )
    parser.add_argument("--mmcif-dir", default=None)
    parser.add_argument("--label", default="recentPDB_low_homology")
    parser.add_argument("--weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0, help="skip longer entries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--metrics-root", default=None)
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    return parser.parse_args(argv)


def compare(before_dir, after_dir):
    """Delta table between two runs. The failure mode is a regression, so say so."""
    before = json.loads((Path(before_dir) / "sidechain_metrics.json").read_text())
    after = json.loads((Path(after_dir) / "sidechain_metrics.json").read_text())
    if before["n_scored"] != after["n_scored"]:
        logger.warning(
            "the two runs scored different target counts (%d vs %d); the deltas are "
            "not on the same residues",
            before["n_scored"],
            after["n_scored"],
        )
    print(f"\n=== {before['label']}: before -> after ===")
    print(f"  before: {before_dir}   ({before['n_scored']} packings)")
    print(f"  after : {after_dir}   ({after['n_scored']} packings)\n")
    regressions = []
    for scope in ("supervised", "all_canonical"):
        b, a = before["summary"].get(scope), after["summary"].get(scope)
        if not b or not a:
            continue
        print(f"  [{scope}]")
        print(f"    {'metric':34s} {'before':>10s} {'after':>10s} {'delta':>10s}")
        for keys in REPORT.values():
            for key in keys:
                lo, hi = float(b[key]), float(a[key])
                delta = hi - lo
                signed = -delta if key in LOWER_IS_BETTER else delta
                if abs(delta) < REGRESSION_TOLERANCE:
                    flag = ""  # unchanged: neither an improvement nor a loss
                elif signed > 0:
                    flag = "  ok"
                else:
                    flag = "  WORSE"
                    if key in HEADLINE and scope == "supervised":
                        regressions.append((key, lo, hi))
                print(f"    {key:34s} {lo:10.4f} {hi:10.4f} {delta:+10.4f}{flag}")
        print()
    if regressions:
        print("  REGRESSION on the headline metrics (supervised scope):")
        for key, lo, hi in regressions:
            print(f"    {key}: {lo:.4f} -> {hi:.4f}")
        return 1
    print("  no regression on the headline metrics.")
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        raise SystemExit("--out is required unless --compare is given")

    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics
    from pxf.eval.sidechain_metrics import aggregate, score
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker
    from pxf.train import protenix as P

    mmcif_dir = Path(args.mmcif_dir or P.DEFAULT_MMCIF_DIR)
    eval_index = Path(
        args.eval_index
        or mmcif_dir.parent / "indices" / "recentPDB_low_homology_maxtoken1536.csv"
    )
    masks = P.SideChainMaskSet(args.mask_root)
    eval_ids = P.ids_from_index(eval_index)
    ids = [p for p in eval_ids if p in masks]
    logger.info("%d of %d eval ids have a mask in %s", len(ids), len(eval_ids), masks.root)
    if not ids:
        raise SystemExit(
            f"no eval id has a mask in {masks.root}. The training mask set covers the "
            "before-2021-09-30 index only; build the eval split's own with "
            "`process.py --ids-file <eval index>` then `make_fampnn.py --drop-extreme-b`."
        )
    if args.max_targets:
        ids = ids[: args.max_targets]

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    canonical = load_metrics(args.metrics_root)
    packer = FaMPNNSideChainPacker(
        args.checkpoint,
        variant=args.weights,
        num_steps=args.num_steps,
        strict_sources=not args.allow_unpinned_sources,
    ).to(select_device(args.device))
    logger.info(
        "FaMPNN %s%s on %s; %d target(s)",
        packer.variant,
        f" from {args.checkpoint}" if args.checkpoint else " (released weights)",
        packer.device,
        len(ids),
    )

    from pxf import atom37

    backbone = list(atom37.BACKBONE_SLOTS)
    # Two scopes in one pass: the trustworthy residues, and every canonical one.
    counts = {"supervised": [], "all_canonical": []}
    rows, skipped = [], []
    started = time.time()

    for index, pdb_id in enumerate(ids):
        try:
            entry = P.read_entry(pdb_id, mmcif_dir=mmcif_dir, masks=masks)
        except (ValueError, KeyError, FileNotFoundError) as error:
            skipped.append(dict(target=pdb_id, reason=str(error)[:200]))
            continue
        length = len(entry)
        if args.max_length and length > args.max_length:
            skipped.append(dict(target=pdb_id, reason=f"length {length} > max"))
            continue
        # The metrics cover the canonical twenty; read_entry only emits those, but
        # an X would poison the frame maths, so this stays an assertion.
        canonical_res = entry.aatype < 20
        scopes = {
            "supervised": canonical_res & entry.supervise.bool(),
            "all_canonical": canonical_res,
        }
        if not bool(scopes["supervised"].any()):
            skipped.append(dict(target=pdb_id, reason="no supervised residues"))
            continue

        # Backbone-only input with the native sequence: the packing task.
        given = torch.zeros_like(entry.atom_mask)
        given[:, backbone] = entry.atom_mask[:, backbone]
        coords = entry.x * given[..., None]

        for sample in range(args.samples):
            packed = packer(
                coords_af2=coords[None],
                aatype=entry.aatype[None],
                atom_mask=given[None],
                residue_index=entry.residue_index[None],
                chain_index=entry.chain_index[None],
                seed=args.seed + index * 1000 + sample,
            )
            summaries = {}
            for scope, mask in scopes.items():
                target_counts, summary = score(
                    packed["coords_af2"][0].cpu(),
                    packed["atom_mask_af2"][0].cpu(),
                    entry.x,
                    entry.atom_mask,
                    entry.aatype,
                    canonical=canonical,
                    residue_mask=mask,
                )
                counts[scope].append(target_counts)
                summaries[scope] = summary
            primary = summaries["supervised"]
            rows.append(
                dict(
                    target=pdb_id,
                    sample=sample,
                    length=length,
                    supervised_residues=int(scopes["supervised"].sum()),
                    canonical_residues=int(canonical_res.sum()),
                    backbone_shift_angstrom=packed["backbone_shift"],
                    **{k: primary[k] for group in REPORT.values() for k in group},
                    **{
                        f"all_{k}": summaries["all_canonical"][k]
                        for k in ("symmetry_rmsd", "rotamer_recovery")
                    },
                )
            )
        if index % 50 == 0 or index == len(ids) - 1:
            last = rows[-1]
            logger.info(
                "[%4d/%4d] %-6s L=%-5d sup=%-5d rmsd=%.3f rot=%.3f chi20=%.3f",
                index + 1,
                len(ids),
                pdb_id,
                length,
                last["supervised_residues"],
                last["symmetry_rmsd"],
                last["rotamer_recovery"],
                last["chi_recovery_20deg"],
            )

    if not rows:
        raise SystemExit("every target was skipped")
    summary = {
        scope: aggregate(scope_counts, canonical=canonical)
        for scope, scope_counts in counts.items()
    }
    elapsed = time.time() - started

    record = dict(
        label=args.label,
        dataset=str(eval_index),
        mask_set=masks.identity(),
        n_targets=len(ids),
        n_scored=len(rows),
        samples_per_target=args.samples,
        seed=args.seed,
        skipped=skipped,
        elapsed_seconds=round(elapsed, 1),
        summary=summary,
        per_target=rows,
        provenance=dict(
            sidechain=packer.identity,
            metrics=canonical.record(),
            task="native backbone + native sequence -> side chains, no design",
            split=(
                "recentPDB low-homology, released after 2022-05-04; the training "
                "index is cut at 2021-09-30 and the intersection is empty"
            ),
        ),
        arguments=vars(args),
    )
    (out / "sidechain_metrics.json").write_text(json.dumps(record, indent=2, default=str))
    with (out / "per_target.csv").open("w") as stream:
        columns = list(rows[0])
        stream.write(",".join(columns) + "\n")
        for row in rows:
            stream.write(",".join(str(row[c]) for c in columns) + "\n")

    print(f"\n=== {args.label}: {len(rows)} packings over {len(ids)} entries ===")
    for scope in ("supervised", "all_canonical"):
        print(
            f"\n  [{scope}]  scored side-chain atoms "
            f"{float(summary[scope]['observed_atoms']):.0f}"
        )
        for group, keys in REPORT.items():
            print(f"    {group}")
            for key in keys:
                print(f"      {key:34s} {summary[scope][key]:.4f}")
    if skipped:
        print(f"\n  skipped {len(skipped)}: e.g. {skipped[:3]}")
    print(f"\n  wrote {out}/sidechain_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
