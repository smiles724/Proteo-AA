#!/usr/bin/env python3
"""Find two-chain complexes the featurizer will actually accept.

    python scripts/screen_dev_complexes.py --limit 6
    python scripts/screen_dev_complexes.py --ids 7f7p 7f91 --verbose

Phase 0 needs held-out complexes to run the BB->SC hook against. Choosing them
by reading the deposition does not work, and this script exists because that
was tried: of the first four picked on chain composition, three failed.

Two distinct traps, both silent in different ways.

**Non-amino-acid content.** `aa_clean` marks any non-amino-acid token -100 and
a downstream 21-way lookup indexes with it, so a glycan, an ion or a nucleic
acid chain raises `IndexError: index -100 is out of bounds` from two layers
down. 7bca looked like a clean dimer under a chain filter that counted only
protein chains; it is a protein-DNA complex.

**Author vs label chain ids.** `binder_chain_ids` is matched against
`label_asym_id`, not the author id that gemmi reports and a human reads. They
diverge whenever the deposition's author ids are not A, B, C... in file order.
7p0s has author chains A, U, B, C and label ids A, B, C, D, so asking for
author "B" -- a 132-residue chain -- selected label B, a 24-residue peptide,
and returned a perfectly well-formed 26-token design region. Nothing failed.
See `pxf/backbone/chain_ids.py`.

So a candidate is only accepted if it survives being featurized *and* the
design region that comes back is the size the chain it names should produce.
That last check is what catches the second trap, and it is the reason this is
a script rather than a one-off.

The chain-size filter counts **every** amino-acid chain, with no minimum. An
earlier version ignored chains under 50 residues, which is how a four-chain
deposition was recorded as a dimer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pxf.backbone.chain_ids import featurizer_chain_id  # noqa: E402

DEFAULT_MMCIF = Path("/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/mmcif")

# The design region may exceed the resolved residue count -- the featurizer
# works from entity_poly, so unobserved residues are tokens too -- but it
# cannot be much smaller. 7p0s failed this at 24/132.
MIN_DESIGN_FRACTION = 0.8


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mmcif-dir", default=str(DEFAULT_MMCIF))
    p.add_argument("--ids", nargs="*", help="screen only these, in this order")
    p.add_argument("--limit", type=int, default=4,
                   help="stop after this many survive; 0 screens everything")
    p.add_argument("--min-total", type=int, default=150)
    p.add_argument("--max-total", type=int, default=300)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--out", default=None, help="write the survivors as JSON")
    p.add_argument("--verbose", action="store_true",
                   help="report rejects and why, not just survivors")
    return p.parse_args(argv)


def chain_census(cif_path):
    """Amino-acid residues per author chain, and whether anything else is present.

    `dirty` covers ligands, ions, glycans and nucleic acids -- everything the
    featurizer's -100 path chokes on. Waters are excluded: the parser drops
    them and they never reach a token.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    counts, dirty = {}, False
    for chain in structure[0]:
        n = 0
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info and info.is_amino_acid():
                n += 1
            elif residue.name != "HOH":
                dirty = True
        if n:
            counts[chain.name] = n
    return counts, dirty


def propose(cif_path, min_total, max_total):
    """The (binder, target) split for a candidate, or a reason it is not one."""
    counts, dirty = chain_census(cif_path)
    if dirty:
        return None, "carries non-amino-acid residues"
    if len(counts) != 2:
        return None, f"{len(counts)} amino-acid chain(s), want 2: {counts}"
    total = sum(counts.values())
    if not (min_total <= total <= max_total):
        return None, f"{total} residues outside [{min_total}, {max_total}]"
    binder = min(counts, key=lambda c: (counts[c], c))
    target = next(c for c in counts if c != binder)
    return {
        "id": Path(cif_path).stem,
        "binder_chain_author": binder,
        "binder_res": counts[binder],
        "target_chain_author": target,
        "target_res": counts[target],
        "total": total,
    }, None


def screen_one(cif_path, entry, crop_size):
    """Featurize with the *label* id and check the design region is the binder.

    Returns the entry augmented with the featurizer's own numbers, or raises
    with a reason. The size check is the point: featurizing successfully is
    necessary and not sufficient, because selecting the wrong chain also
    succeeds.
    """
    from pxf.backbone.driver import featurize_structures, to_featurized

    label = featurizer_chain_id(cif_path, entry["binder_chain_author"])
    entry["binder_chain"] = label
    entry["target_chain"] = featurizer_chain_id(
        cif_path, entry["target_chain_author"]
    )
    sample_id, dataset = featurize_structures(
        [str(cif_path)], crop_size=crop_size,
        binder_chain_ids=[label], parser_dataset="WeightedPDB",
    )[0]
    item = to_featurized(sample_id, dataset[0])

    tokens = int(item.topology.num_tokens)
    design = int(item.design_mask.sum())
    entry["featurizer_tokens"] = tokens
    entry["featurizer_design_tokens"] = design
    if design == 0:
        raise ValueError("no design tokens: nothing would be generated")
    if design == tokens:
        raise ValueError("every token is a design token: this is not conditioned")
    floor = MIN_DESIGN_FRACTION * entry["binder_res"]
    if design < floor:
        raise ValueError(
            f"design region is {design} tokens for a {entry['binder_res']}-residue "
            f"chain (< {MIN_DESIGN_FRACTION:g}x): the featurizer selected "
            f"something other than author chain {entry['binder_chain_author']}"
        )
    return entry


def main(argv=None):
    args = parse_args(argv)
    mmcif = Path(args.mmcif_dir)
    if args.ids:
        paths = [mmcif / f"{i}.cif" for i in args.ids]
    else:
        paths = sorted(mmcif.glob("*.cif"))
    print(f"screening {len(paths)} structure(s) under {mmcif}", flush=True)

    survivors, rejected = [], 0
    for path in paths:
        try:
            entry, reason = propose(path, args.min_total, args.max_total)
        except Exception as error:  # noqa: BLE001 - gemmi raises broadly
            entry, reason = None, f"unreadable: {type(error).__name__}"
        if entry is None:
            rejected += 1
            if args.verbose and reason and "outside" not in reason:
                print(f"  skip {path.stem}: {reason}", flush=True)
            continue
        try:
            entry = screen_one(path, entry, args.crop_size)
        except Exception as error:  # noqa: BLE001 - upstream raises broadly
            rejected += 1
            print(f"  fail {path.stem}: {type(error).__name__}: "
                  f"{str(error)[:110]}", flush=True)
            continue
        survivors.append(entry)
        print(f"  OK   {entry['id']}  author {entry['binder_chain_author']}"
              f"->label {entry['binder_chain']}  "
              f"{entry['featurizer_tokens']} tokens, "
              f"{entry['featurizer_design_tokens']} design", flush=True)
        if args.limit and len(survivors) >= args.limit:
            break

    print(f"\n{len(survivors)} survivor(s), {rejected} rejected")
    if args.out:
        Path(args.out).write_text(json.dumps(survivors, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
