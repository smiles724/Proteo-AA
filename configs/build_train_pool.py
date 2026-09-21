#!/usr/bin/env python
"""Build the train-split complex pool for the joint (bs_seq_sc) run.

Same filters, same conversion and the same role rule as the val gen_stress set:
``select_dimer_targets.py`` picks, ``prepare_dimer_targets.py``'s
``normalise()``/``chain_sizes()`` convert, and the binder is the smaller chain
*after* conversion (ties on chain id). Two screens are added, both applied
before an example is accepted rather than discovered during training:

``generated_tokens <= crop_size``
    3d4u failed the val run at featurization with a 1739-token binder against
    crop 512. Manifest ``total_res`` does not bound the converted binder, so
    the bound is applied to the featurized count.

``non_aa_tokens == 0``
    Ligands, ions and waters make the side-chain targets unalignable --
    ``pxf.backbone.driver`` warns "the coupling path needs a protein-only
    entry; its side-chain targets will not align". An objective with a
    side-chain term cannot use those entries. Counted exactly as the driver
    does: ``aa_clean < 0 or >= 20``.

Candidates are walked in the selector's deterministic order and the walk stops
as soon as each pool fills its quota, so the accepted set is a prefix of that
order: re-running with a larger quota extends it instead of reshuffling.

Train and calibration are disjoint by construction -- calibration takes the
entries immediately after the train quota, and both pools are already
deduplicated (``pdb_id`` for pdb; ``cluster_id`` and ``uniprot`` for ted).
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

REPO = Path("/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-pxdesign-fampnn-pack")
MARLOWE_CACHE = "/scratch/m000137-pm06/Proteo-AA/pxf/train_pool/cif_cache"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", required=True, help="from select_dimer_targets.py --split train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache", required=True, help="normalised CIF/PDB cache")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--train-per-pool", type=int, default=256)
    p.add_argument("--calib-per-pool", type=int, default=8)
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--allow-non-protein", action="store_true",
                   help="disable the protein-only screen (not for joint training)")
    return p.parse_args(argv)


def load_preparer(repo):
    """Reuse the committed normalise()/chain_sizes() rather than restating them."""
    path = Path(repo) / "scripts" / "prepare_dimer_targets.py"
    spec = importlib.util.spec_from_file_location("_prep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen(row, prep, cache, crop_size, allow_non_protein):
    """Convert, featurize and decide. Returns (entry, reject_reason)."""
    from pxf.backbone.driver import featurize_structures, to_featurized

    cif = prep.normalise(row.path, row.example_id, cache)
    sizes = prep.chain_sizes(cif)
    if len(sizes) < 2:
        return None, f"only {len(sizes)} chain(s) after conversion: {sizes}"
    binder = min(sizes, key=lambda c: (sizes[c], c))
    target = [c for c in sizes if c != binder]

    sid, dataset = featurize_structures([str(cif)], crop_size=crop_size,
                                        binder_chain_ids=[binder],
                                        parser_dataset="Distillation")[0]
    raw = dataset[0]
    item = to_featurized(sid, raw)
    design = item.design_mask

    # The driver's own definition, so the count matches its warning exactly.
    aa = raw["input_feature_dict"]["aa_clean"].reshape(-1).long()[: item.topology.num_tokens]
    non_aa = int(((aa < 0) | (aa >= 20)).sum())

    entry = dict(
        example_id=row.example_id, pool=row.pool, source_path=row.path,
        cif_path=str(cif), converted_binder_chain=binder,
        converted_target_chains=",".join(sorted(target)),
        chain_sizes=json.dumps(sizes), binder_residues=int(sizes[binder]),
        target_residues=int(sum(sizes[c] for c in target)),
        cluster_id=row.cluster_id, split=row.split,
        interface_res=int(row.interface_res), tokens=int(item.num_tokens),
        generated_tokens=int(design.sum()), target_tokens=int((~design).sum()),
        flat_atoms=len(item.topology.atom_names), non_aa_tokens=non_aa,
    )
    if entry["generated_tokens"] == 0:
        return entry, "no design tokens: nothing would be generated"
    if entry["target_tokens"] == 0:
        return entry, "no target tokens: this is not conditioned"
    if entry["generated_tokens"] > crop_size:
        return entry, f"binder {entry['generated_tokens']} tokens > crop {crop_size}"
    if non_aa and not allow_non_protein:
        return entry, f"{non_aa} of {entry['tokens']} tokens are not amino acids"
    return entry, None


def remap(frame, pd):
    """A Marlowe-side copy: cif_path rewritten, source_path blanked not guessed."""
    out = frame.copy()
    out["source_path_hai"] = out["source_path"]
    out["source_path"] = ""  # the 152 GB source tree stays on HAI; fail loudly
    out["cif_path"] = out["cif_path"].map(lambda p: f"{MARLOWE_CACHE}/{Path(p).name}")
    return out


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    prep = load_preparer(args.repo)
    targets = pd.read_parquet(args.targets)
    quota = args.train_per_pool + args.calib_per_pool
    out_dir = Path(args.out_dir)
    (out_dir / "marlowe").mkdir(parents=True, exist_ok=True)

    accepted, screened, started = {}, [], time.time()
    for pool, group in targets.groupby("pool", sort=True):
        accepted[pool] = []
        for n, row in enumerate(group.itertuples(), 1):
            if len(accepted[pool]) >= quota:
                break
            try:
                entry, reason = screen(row, prep, args.cache, args.crop_size,
                                       args.allow_non_protein)
            except Exception as error:  # noqa: BLE001 - upstream raises broadly
                entry, reason = dict(example_id=row.example_id, pool=row.pool), str(error)[:300]
            record = dict(entry or {}, accepted=reason is None, reject_reason=reason or "")
            screened.append(record)
            if reason is None:
                accepted[pool].append(entry)
            else:
                print(f"  reject {row.example_id}: {reason[:120]}", flush=True)
            if n % 25 == 0:
                print(f"[{pool}] seen {n}, accepted {len(accepted[pool])}/{quota}, "
                      f"{time.time() - started:.0f}s", flush=True)
        print(f"[{pool}] done: accepted {len(accepted[pool])} of {quota} "
              f"from {n} candidate(s)", flush=True)

    train_rows, calib_rows, short = [], [], {}
    for pool, entries in accepted.items():
        train_rows += entries[: args.train_per_pool]
        calib_rows += entries[args.train_per_pool: quota]
        if len(entries) < quota:
            short[pool] = dict(accepted=len(entries), wanted=quota)

    train, calib = pd.DataFrame(train_rows), pd.DataFrame(calib_rows)
    overlap = set(train.example_id) & set(calib.example_id)
    clusters = set(train.cluster_id) & set(calib.cluster_id)
    assert not overlap, f"train/calib share {len(overlap)} example(s)"
    assert not clusters, f"train/calib share {len(clusters)} cluster(s)"
    assert (train.split == "train").all() and (calib.split == "train").all()
    assert train.generated_tokens.max() <= args.crop_size
    if not args.allow_non_protein:
        assert train.non_aa_tokens.max() == 0 and calib.non_aa_tokens.max() == 0

    train.to_parquet(out_dir / "bs_seq_sc_train.parquet", index=False)
    calib.to_parquet(out_dir / "bs_seq_sc_calib.parquet", index=False)
    pd.DataFrame(screened).to_parquet(out_dir / "screened_all.parquet", index=False)
    remap(train, pd).to_parquet(out_dir / "marlowe" / "bs_seq_sc_train.marlowe.parquet", index=False)
    remap(calib, pd).to_parquet(out_dir / "marlowe" / "bs_seq_sc_calib.marlowe.parquet", index=False)

    shipped = pd.concat([train, calib]).cif_path.map(lambda p: Path(p).stem).tolist()
    (out_dir / "marlowe" / "transfer_files.txt").write_text(
        "\n".join(f"{s}{ext}" for s in shipped for ext in (".cif", ".pdb")) + "\n")

    report = dict(
        targets=str(args.targets), cache=str(args.cache), crop_size=args.crop_size,
        train=len(train), calibration=len(calib), candidates_screened=len(screened),
        short_of_quota=short, protein_only=not args.allow_non_protein,
        by_pool_train=train.groupby("pool").size().to_dict(),
        by_pool_calibration=calib.groupby("pool").size().to_dict(),
        rejects=pd.DataFrame(screened).query("not accepted").reject_reason
                  .str.replace(r"\d+", "N", regex=True).value_counts().head(12).to_dict(),
        generated_tokens=dict(min=int(train.generated_tokens.min()),
                              median=int(train.generated_tokens.median()),
                              max=int(train.generated_tokens.max())),
        target_tokens=dict(min=int(train.target_tokens.min()),
                           median=int(train.target_tokens.median()),
                           max=int(train.target_tokens.max())),
        binder_rule="smaller chain after conversion; ties on chain id",
        normalisation="gemmi -> PDB -> protenix pdb_to_cif -> CIF",
        parser="Distillation", seconds=round(time.time() - started, 1),
    )
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if not short else 3


if __name__ == "__main__":
    raise SystemExit(main())
