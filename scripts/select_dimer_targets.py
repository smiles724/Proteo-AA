#!/usr/bin/env python
"""Pick held-out dimer targets for the generation stress test.

Two pools, because they fail differently and agreement across them is worth
more than either alone:

``pdb``   experimental multimers from the PDB mirror. Real structures, real
          interfaces, but their deposition dates predate PXDesign's training,
          so a target may be something the backbone model has seen.
``ted``   Teddymer AFDB dimers. Predicted rather than experimental, so the
          "native" interface is itself a model's opinion -- but the pool carries
          per-dimer quality gates and one cluster per row.

**What "held out" means here, precisely.** Both manifests carry a ``split`` and
only ``split == "val"`` rows are eligible. The PDB pool additionally carries
``eval_target_leak``, and the five flagged rows are quarantined by the manifest
into their own ``excluded_eval_target`` split rather than sitting in ``val`` --
this script re-checks the flag anyway rather than trusting that. Held out from
the *adapter* is separately guaranteed by construction: A_SB trained on AFDB
monomers, so no dimer in either pool appears in its training data.

What neither pool can establish is held-out-ness against **PXDesign's own
pretraining**, which is ByteDance's and not recorded here. ``--max-deposition``
filters the PDB pool by deposition date for anyone who wants to bound that.

**The target/partner rule is fixed, not per-target.** For a two-chain dimer the
*larger* chain is the target (conditioning, coordinates held fixed) and the
smaller is the partner PXDesign must generate. Deterministic, so no target is
chosen to flatter a result, and ties break on chain id.
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA_PATH = "/hai/scratch/yfsun/proteina_complexa_data/DATA_PATH"
POOLS = {
    "pdb": "pdb_mirror_processed/pdb_multimers.parquet",
    "ted": "latent_train_parquet_files/ted_dimers_partial_qc.parquet",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-path", default=DATA_PATH)
    p.add_argument("--out", required=True, help="manifest parquet to write")
    p.add_argument(
        "--per-pool",
        type=int,
        default=16,
        help="targets from each pool; 16 + 16 matches the 32-target budget",
    )
    p.add_argument("--split", default="val")
    p.add_argument("--min-res", type=int, default=80)
    p.add_argument("--max-res", type=int, default=480, help="must fit --crop-size")
    p.add_argument(
        "--min-interface-res",
        type=int,
        default=10,
        help="a dimer with almost no interface is not a binder-design task",
    )
    p.add_argument("--max-resolution", type=float, default=3.5, help="pdb pool only")
    p.add_argument("--max-deposition", default=None, help="e.g. 2021-01-01, pdb only")
    p.add_argument("--min-plddt", type=float, default=80.0, help="ted pool only")
    p.add_argument("--max-pae", type=float, default=5.0, help="ted pool only")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def stable_order(values, seed):
    """Deterministic shuffle keyed by content, not by row position.

    blake2b rather than a seeded RNG over the frame: the order then survives a
    manifest that gains or loses rows, so re-running after a data refresh keeps
    the targets it can rather than reshuffling everything.
    """
    return sorted(
        values,
        key=lambda v: hashlib.blake2b(f"{seed}|{v}".encode(), digest_size=8).hexdigest(),
    )


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    root = Path(args.data_path)
    chosen, report = [], {}

    # --- experimental multimers ---
    pdb = pd.read_parquet(root / POOLS["pdb"])
    before = len(pdb)
    pdb = pdb[
        pdb.split.eq(args.split)
        & pdb.ok.astype(bool)
        & ~pdb.eval_target_leak.astype(bool)
        & pdb.n_chains_kept.eq(2)
        & pdb.total_res.between(args.min_res, args.max_res)
        & pdb.iface_res_total.ge(args.min_interface_res)
        & pdb.resolution_val.le(args.max_resolution)
    ]
    if args.max_deposition:
        pdb = pdb[
            pd.to_datetime(pdb.deposition_date_first) <= pd.Timestamp(args.max_deposition)
        ]
    # One dimer per PDB entry: two assemblies of one entry are not two targets.
    pdb = pdb.drop_duplicates(subset=["pdb_id"])
    order = stable_order(pdb.example_id.tolist(), args.seed)
    picked = order[: args.per_pool]
    report["pdb"] = dict(
        rows=before,
        eligible=len(pdb),
        picked=len(picked),
        filters=dict(
            split=args.split,
            resolution_max=args.max_resolution,
            total_res=[args.min_res, args.max_res],
            min_interface_res=args.min_interface_res,
            n_chains_kept=2,
            leak_excluded=True,
            deposition_max=args.max_deposition,
        ),
    )
    for row in pdb[pdb.example_id.isin(picked)].itertuples():
        chosen.append(
            dict(
                example_id=row.example_id,
                pool="pdb",
                path=row.path,
                chains=row.chains_str,
                total_res=int(row.total_res),
                interface_res=int(row.iface_res_total),
                resolution=float(row.resolution_val),
                deposition=str(row.deposition_date_first)[:10],
                cluster_id=str(row.pdb_id),
                split=row.split,
            )
        )

    # --- Teddymer AFDB dimers ---
    ted = pd.read_parquet(root / POOLS["ted"])
    before = len(ted)
    ted = ted[
        ted.split.eq(args.split)
        & ted.passes_quality_gates.astype(bool)
        & ted.interface_length.ge(args.min_interface_res)
        & ted.avg_int_plddt.ge(args.min_plddt)
        & ted.avg_int_pae.le(args.max_pae)
    ]
    # One per cluster and one per UniProt: the pool is already one cluster per
    # row, so this only bites if that ever stops being true.
    ted = ted.drop_duplicates(subset=["cluster_id"]).drop_duplicates(subset=["uniprot"])
    order = stable_order(ted.example_id.tolist(), args.seed)
    picked = order[: args.per_pool]
    report["ted"] = dict(
        rows=before,
        eligible=len(ted),
        picked=len(picked),
        filters=dict(
            split=args.split,
            quality_gates=True,
            min_interface_length=args.min_interface_res,
            min_plddt=args.min_plddt,
            max_pae=args.max_pae,
            deduped_on=["cluster_id", "uniprot"],
        ),
    )
    for row in ted[ted.example_id.isin(picked)].itertuples():
        chosen.append(
            dict(
                example_id=row.example_id,
                pool="ted",
                path=row.path,
                chains="A,B",
                total_res=None,
                interface_res=int(row.interface_length),
                resolution=None,
                deposition=None,
                cluster_id=str(row.cluster_id),
                split=row.split,
                plddt=float(row.avg_int_plddt),
                pae=float(row.avg_int_pae),
                uniprot=str(row.uniprot),
            )
        )

    frame = pd.DataFrame(chosen)
    missing = [p for p in frame.path if not Path(p).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} selected structure(s) are absent, e.g. {missing[:3]}. "
            "This dataset has shipped LFS pointer stubs before; check the files."
        )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    (out.with_suffix(".report.json")).write_text(
        json.dumps(
            dict(
                data_path=str(root),
                seed=args.seed,
                per_pool=args.per_pool,
                selected=len(frame),
                pools=report,
                target_rule="larger chain is the target; smaller is generated",
                holdout_note=(
                    "split=='val' in both manifests; PDB leak flag re-checked. "
                    "Held out from A_SB by construction (it trained on AFDB "
                    "monomers). NOT established against PXDesign pretraining."
                ),
            ),
            indent=2,
            default=str,
        )
    )
    print(f"selected {len(frame)} target(s) -> {out}")
    for pool, entry in report.items():
        print(
            f"  {pool}: {entry['eligible']:,} eligible of {entry['rows']:,} "
            f"-> {entry['picked']} picked"
        )
    print(
        frame[["example_id", "pool", "interface_res", "cluster_id"]].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
