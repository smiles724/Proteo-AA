#!/usr/bin/env python3
"""Materialize AFDB records as mmCIF for the coupling phases.

The coupling path reads mmCIF through ``pxdesign_train``'s featurizer, and the
AFDB builder discards the files it downloaded (``raw`` is transient). So the
records are re-emitted from the packed shards; see :func:`pxf.train.afdb.write_cif`
for the format requirements, several of which fail silently if unmet.

    python scripts/export_afdb_cifs.py --count 2000 \\
        --out-dir /hai/scratch/yfsun/afdb_laproteina/cif_phase1 \\
        --manifest configs/phase1_structures_afdb.txt

**Verification is sampled, not exhaustive, and that is a deliberate difference
from ``survey_coupling_structures.py``.** For PDB entries the usable set had to
be discovered one file at a time, because whether the featurizer agrees with a
protein-only parse depends on the entry (assembly vs asymmetric unit, hetero
groups, crop validity). AFDB is homogeneous by construction -- every record is a
single chain of canonical residues with a complete atom set and a contiguous
residue index -- so a random sample measures the whole set. Featurizing all of
them up front would cost about 1.7 s each and tell us nothing new; the sampled
rate is reported in the manifest header so the assumption is on the record and
falsifiable.

Records are chosen by striding the length-sorted id list, so the exported set
spans the length distribution rather than clustering at the short end (the
shortest few thousand AFDB entries are all near the 33-residue floor).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401

logger = logging.getLogger("pxf.export_afdb")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--count", type=int, default=2000, help="how many to export")
    p.add_argument("--out-dir", required=True, help="where the .cif files go")
    p.add_argument("--manifest", required=True, help="path list the launcher reads")
    p.add_argument("--data-root", default=None, help="the AFDB build's run/ directory")
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--max-length", type=int, default=512, help="skip longer records")
    p.add_argument(
        "--verify",
        type=int,
        default=40,
        help="featurize this many at random to measure the pass rate (0 to skip)",
    )
    p.add_argument("--crop-size", type=int, default=512, help="crop used when verifying")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def verify(paths, *, crop_size):
    """Featurize a sample and check it against a direct parse of the same file."""
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf import atom37
    from pxf.backbone.driver import featurize_structures, to_featurized

    passed, failures = 0, []
    for path in paths:
        try:
            sample_id, source = featurize_structures([str(path)], crop_size=crop_size)[0]
            structure = to_featurized(sample_id, source[0])
            native = process_single_pdb(load_feats_from_pdb(str(path)))
            if int(native["aatype"].shape[0]) != structure.num_tokens:
                failures.append(dict(path=str(path), reason="token count"))
                continue
            if atom37.sequence_from_aatype(structure.aatype) != atom37.sequence_from_aatype(
                native["aatype"].long()
            ):
                failures.append(dict(path=str(path), reason="sequence mismatch"))
                continue
            passed += 1
        except Exception as error:  # noqa: BLE001 - the reason is the output
            failures.append(
                dict(path=str(path), reason=f"{type(error).__name__}: {str(error)[:140]}")
            )
    return passed, failures


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from pxf.train import afdb as A

    shards = A.AFDBShards(args.data_root, split=args.split)
    eligible = [a for a in shards.afids if shards.lengths[a] <= args.max_length]
    if not eligible:
        raise SystemExit(f"no record is <= {args.max_length} residues")
    # Stride the length-sorted list so the export spans the distribution; the
    # shortest few thousand AFDB entries all sit near the 33-residue floor.
    by_length = sorted(eligible, key=lambda a: (shards.lengths[a], a))
    count = min(args.count, len(by_length))
    step = len(by_length) / count
    chosen = [by_length[min(int(i * step), len(by_length) - 1)] for i in range(count)]
    logger.info(
        "%d of %d eligible records, strided across lengths %d-%d",
        len(chosen),
        len(eligible),
        shards.lengths[chosen[0]],
        shards.lengths[chosen[-1]],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written, longest = [], 0
    for index, afid in enumerate(chosen):
        record = shards.read(afid)
        path = A.write_cif(record, out_dir / f"{afid}.cif")
        written.append(path)
        longest = max(longest, len(record))
        if index and index % 500 == 0:
            logger.info(
                "%d/%d written (%.1f/s)",
                index,
                len(chosen),
                index / (time.time() - started),
            )
    logger.info(
        "%d written in %.0fs; longest %d residues",
        len(written),
        time.time() - started,
        longest,
    )

    passed, failures, sampled = len(written), [], 0
    if args.verify:
        rng = random.Random(args.seed)
        sample = rng.sample(written, min(args.verify, len(written)))
        sampled = len(sample)
        logger.info("verifying %d at random", sampled)
        passed, failures = verify(sample, crop_size=args.crop_size)
        logger.info("%d/%d passed", passed, sampled)
        if passed < sampled:
            for failure in failures[:5]:
                logger.warning("  %s: %s", Path(failure["path"]).stem, failure["reason"])

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# AFDB structures for the coupling phases, produced by",
        "#   scripts/export_afdb_cifs.py \\",
        f"#       --count {args.count} --out-dir {args.out_dir} \\",
        f"#       --manifest {args.manifest}",
        f"# {len(written)} records from the La-Proteina AFDB subset ({args.split} split),",
        f"# strided across the length distribution; longest {longest} residues.",
        f"# Run with --crop-size >= {longest}, or the side-chain targets misalign.",
        f"# Verification is sampled, not exhaustive: {passed}/{sampled} of a random"
        if sampled
        else "# Verification skipped.",
    ]
    if sampled:
        header += [
            "# sample featurized and matched a direct parse of the same file. AFDB is",
            "# homogeneous by construction -- one chain, canonical residues, complete",
            "# atoms -- so a sample measures the set, unlike the PDB case in",
            "# scripts/survey_coupling_structures.py.",
        ]
    manifest.write_text("\n".join(header + [str(p) for p in written]) + "\n")
    report = Path(f"{args.manifest}.report.json")
    report.write_text(
        json.dumps(
            dict(
                source=shards.identity(),
                exported=len(written),
                eligible=len(eligible),
                max_length=args.max_length,
                longest_exported=longest,
                min_crop_size_required=longest,
                verified_sample=sampled,
                verified_passed=passed,
                failures=failures,
            ),
            indent=2,
        )
    )
    print(f"\nexported {len(written)} -> {out_dir}")
    print(f"  sampled verification: {passed}/{sampled}")
    print(f"  --crop-size must be >= {longest}")
    print(f"  manifest -> {manifest}")
    return 0 if passed == sampled else 1


if __name__ == "__main__":
    raise SystemExit(main())
