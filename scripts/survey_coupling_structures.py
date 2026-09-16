#!/usr/bin/env python3
"""Which structures the coupling path can actually train on, checked not assumed.

The phase 1-3 pipeline is fussy about its inputs, and every constraint is a real
one rather than a convenience:

* ``pxdesign_train``'s ``DesignSourceDataset`` must find a crop-valid example,
  which some targets never satisfy;
* the featurized token count must equal the count FaMPNN parses from the same
  file, because ``_native_atom37`` aligns the side-chain targets positionally.
  These disagree more often than one would guess: for PDB entry ``104l`` the
  featurizer emits 166 tokens (assembly 1, one chain) while gemmi and FaMPNN both
  see 328 (the two-chain asymmetric unit). The index's ``num_prot_chains``
  describes the assembly, so it cannot predict this -- only featurizing can;
* the sequences must match, not just the lengths;
* ``--crop-size`` must be at least the longest structure, since a crop breaks the
  same correspondence.

So the usable set is discovered by trying, and the rejections are reported with
reasons rather than silently dropped. Writing the result to a file that the
launcher consumes directly is what makes a coupling run reproducible: the run's
data source is then a committed manifest, not a glob evaluated at submit time.

    python scripts/survey_coupling_structures.py \
        --glob '/hai/scratch/yfsun/casp14/cif/*.cif' \
        --glob '/hai/scratch/yfsun/casp15/cif/*.cif' \
        --crop-size 512 --out configs/phase1_structures_casp14_15.txt
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

logger = logging.getLogger("pxf.survey")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--glob",
        action="append",
        required=True,
        metavar="PATTERN",
        help="shell glob for candidate .cif files; repeatable",
    )
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--out", required=True, help="manifest the launcher will read")
    p.add_argument(
        "--report",
        default=None,
        help="where to write the rejection report (default: <out>.report.json)",
    )
    p.add_argument("--proteoaa-root", default=None)
    return p.parse_args(argv)


def survey(paths, *, crop_size, proteoaa_root=None):
    """Return ``(usable, rejected)``; ``usable`` is ``[(stem, path, n_tokens)]``."""
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf import atom37
    from pxf.backbone.driver import featurize_structures, to_featurized

    usable, rejected = [], []
    for path in paths:
        stem = Path(path).stem
        try:
            sample_id, source = featurize_structures(
                [path], crop_size=crop_size, proteoaa_root=proteoaa_root
            )[0]
            structure = to_featurized(sample_id, source[0])
            native = process_single_pdb(load_feats_from_pdb(str(path)))
            n_file = int(native["aatype"].shape[0])
            if n_file != structure.num_tokens:
                rejected.append(
                    dict(
                        target=stem,
                        reason="token count",
                        featurized=structure.num_tokens,
                        in_file=n_file,
                    )
                )
                continue
            if atom37.sequence_from_aatype(structure.aatype) != atom37.sequence_from_aatype(
                native["aatype"].long()
            ):
                rejected.append(dict(target=stem, reason="sequence mismatch"))
                continue
            usable.append((stem, str(path), int(structure.num_tokens)))
        except Exception as error:  # noqa: BLE001 - the reason is the output
            rejected.append(
                dict(target=stem, reason=f"{type(error).__name__}: {str(error)[:160]}")
            )
        logger.info(
            "%-12s %s", stem, "usable" if usable and usable[-1][0] == stem else "rejected"
        )
    return usable, rejected


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = sorted({p for pattern in args.glob for p in globlib.glob(pattern)})
    if not paths:
        raise SystemExit(f"no files matched {args.glob}")
    logger.info("%d candidate(s) at crop size %d", len(paths), args.crop_size)

    usable, rejected = survey(
        paths, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )
    if not usable:
        raise SystemExit("no structure survived the survey")
    lengths = sorted(n for _, _, n in usable)
    longest = lengths[-1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A leading comment block so the manifest records its own provenance; the
    # launcher's reader skips '#' lines.
    header = [
        "# Structures the coupling path can train on, produced by",
        "#   scripts/survey_coupling_structures.py \\",
        *[f"#       --glob '{pattern}' \\" for pattern in args.glob],
        f"#       --crop-size {args.crop_size} --out {args.out}",
        f"# {len(usable)} usable of {len(paths)} candidates; "
        f"lengths {lengths[0]}-{longest}.",
        f"# Run with --crop-size >= {longest}, or the side-chain targets misalign.",
    ]
    out.write_text("\n".join(header + [path for _, path, _ in usable]) + "\n")

    report = Path(args.report or f"{args.out}.report.json")
    report.write_text(
        json.dumps(
            dict(
                candidates=len(paths),
                usable=len(usable),
                crop_size=args.crop_size,
                min_crop_size_required=longest,
                lengths=dict(
                    min=lengths[0], median=lengths[len(lengths) // 2], max=longest
                ),
                globs=args.glob,
                rejected=rejected,
            ),
            indent=2,
        )
    )
    print(f"\nusable {len(usable)}/{len(paths)}  lengths {lengths[0]}-{longest}")
    print(f"  --crop-size must be >= {longest}")
    print(f"  manifest -> {out}")
    print(f"  rejections -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
