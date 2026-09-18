#!/usr/bin/env python
"""Normalise selected dimers into something Protenix will actually parse.

Neither pool is directly consumable, and they fail differently:

``ted``  the structures are ``.pdb``, so the mmCIF parser reports
         ``ValueError: There are no blocks in the file``.
``pdb``  the structures are ``.cif`` but stripped: no ``pdbx_struct_assembly``
         (``KeyError`` under the WeightedPDB parser) and no usable entity_poly
         (``KeyError: '1'`` in ``build_ref_chain_with_atom_array``). They are
         mirror-processed files, not full PDB entries.

Both failures surface two layers from the cause, as
``DesignSourceDataset: failed to find a crop-valid example ... parsed <path>
without atom_array/token_array``, which names a symptom rather than the
problem. So the normalisation is explicit here instead of guessed at the call
site: rewrite to PDB with gemmi, convert with Protenix's own
``pdb_to_cif``, and read with ``parser_dataset="Distillation"`` -- the
already-assembled path, which is what the PINDER provider uses for the same
reason.

**The binder chain is decided after conversion, not before.** The converter can
relabel chains, so choosing on the source file can designate a chain that no
longer exists under that name. The rule is the same either way -- smaller chain
generated, larger chain the fixed target, ties broken on chain id -- but it is
applied to the converted file and the result is recorded, mirroring the PINDER
prep utility's ``converted_binder_chain``.
"""

import argparse
import json
from pathlib import Path

CACHE = "/hai/scratch/yfsun/proteo_aa_runs/pxf_gen_stress/cif_cache"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--targets", required=True, help="from select_dimer_targets.py")
    p.add_argument("--out", required=True, help="augmented manifest parquet")
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument(
        "--verify",
        action="store_true",
        help="featurize each converted file and record the token split. Slower, "
        "and the only way to know the whole path works before a GPU job does",
    )
    return p.parse_args(argv)


def normalise(source, example_id, cache):
    """``.pdb``/``.cif`` -> a Protenix-parseable CIF. Cached, idempotent."""
    import gemmi
    from protenix.data.utils import pdb_to_cif

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cif_path = cache / f"{example_id}.cif"
    if cif_path.is_file() and cif_path.stat().st_size > 0:
        return cif_path
    pdb_path = cache / f"{example_id}.pdb"
    if not pdb_path.is_file():
        structure = gemmi.read_structure(str(source))
        structure.setup_entities()
        structure.write_pdb(str(pdb_path))
    # entry_id has to look like a PDB code; the converter writes it into the
    # block header and Protenix reads it back.
    pdb_to_cif(str(pdb_path), str(cif_path), entry_id=str(example_id)[:4].lower())
    return cif_path


def chain_sizes(path):
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    return {chain.name: len(chain) for chain in structure[0]}


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    frame = pd.read_parquet(args.targets)
    rows, failures = [], []
    for row in frame.itertuples():
        try:
            cif = normalise(row.path, row.example_id, args.cache)
            sizes = chain_sizes(cif)
            if len(sizes) < 2:
                raise ValueError(f"only {len(sizes)} chain(s) after conversion: {sizes}")
            binder = min(sizes, key=lambda c: (sizes[c], c))
            target = [c for c in sizes if c != binder]
            entry = dict(
                example_id=row.example_id,
                pool=row.pool,
                source_path=row.path,
                cif_path=str(cif),
                converted_binder_chain=binder,
                converted_target_chains=",".join(sorted(target)),
                chain_sizes=json.dumps(sizes),
                binder_residues=int(sizes[binder]),
                target_residues=int(sum(sizes[c] for c in target)),
                cluster_id=row.cluster_id,
                split=row.split,
                interface_res=int(row.interface_res),
            )
            if args.verify:
                from pxf.backbone.driver import featurize_structures, to_featurized

                sid, dataset = featurize_structures(
                    [str(cif)],
                    crop_size=args.crop_size,
                    binder_chain_ids=[binder],
                    parser_dataset="Distillation",
                )[0]
                item = to_featurized(sid, dataset[0])
                design = item.design_mask
                entry.update(
                    tokens=int(item.num_tokens),
                    generated_tokens=int(design.sum()),
                    target_tokens=int((~design).sum()),
                    flat_atoms=len(item.topology.atom_names),
                )
                if entry["generated_tokens"] == 0:
                    raise ValueError("no design tokens: nothing would be generated")
                if entry["target_tokens"] == 0:
                    raise ValueError("no target tokens: this is not conditioned")
            rows.append(entry)
        except Exception as error:  # noqa: BLE001 - upstream raises broadly
            failures.append(dict(example_id=row.example_id, reason=str(error)[:300]))
            print(f"  FAILED {row.example_id}: {str(error)[:160]}")

    if not rows:
        raise SystemExit(f"nothing prepared; {len(failures)} failure(s)")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared = pd.DataFrame(rows)
    prepared.to_parquet(out, index=False)
    (out.with_suffix(".report.json")).write_text(
        json.dumps(
            dict(
                targets=str(args.targets),
                cache=str(args.cache),
                prepared=len(prepared),
                failed=len(failures),
                failures=failures,
                verified=bool(args.verify),
                parser="Distillation",
                normalisation="gemmi -> PDB -> protenix pdb_to_cif -> CIF",
                binder_rule="smaller chain after conversion; ties on chain id",
            ),
            indent=2,
            default=str,
        )
    )
    print(f"\nprepared {len(prepared)}/{len(frame)} target(s) -> {out}")
    by_pool = prepared.groupby("pool").size().to_dict()
    print(f"  by pool: {by_pool}")
    cols = [
        "example_id",
        "pool",
        "converted_binder_chain",
        "binder_residues",
        "target_residues",
    ]
    if args.verify:
        cols += ["tokens", "generated_tokens", "target_tokens"]
    print(prepared[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
