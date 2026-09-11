#!/usr/bin/env python3
"""Unconditional monomer backbone geometry: can this checkpoint make a chain at all?

The AlphaProteo-10 benchmark measures binder designability, but every arm there
generates the WHOLE complex from noise, so a broken binder chain is confounded
with the model also having to rebuild the target. This probe removes the target:
real Protenix monomer rows, every residue a design token, coordinates started
from Gaussian noise, full reverse diffusion. What comes back either has
protein bond geometry or it does not.

Measured, not predicted-structure metrics: consecutive CA-CA distance should be
3.80 +/- 0.03 A in any real protein. A model that has not learned backbone
geometry fails this before any folding metric can say anything, and reporting
designability in that regime measures nothing (see
docs/alphaproteo10_designability_result_zh.md, 0/3280).

    python scripts/evaluation/probe_monomer_backbone_geometry.py \
        --checkpoint <ckpt> --output-dir <dir> --limit-index 8 --n-step 400

`--sampler-mode` defaults to `pxdesign_native` on purpose: that is what the
benchmark used, and `cogenerate`'s own default (`minimal_euler`) is a different
trajectory, so leaving it implicit would compare two unrelated things.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

CA_IDEAL = 3.80
CA_TOL = 0.30


def ca_rows_for_design(feat: dict, design: np.ndarray) -> np.ndarray | None:
    """Atom row of each design token's CA.

    `sc_bb_atom_idx` is filled in (N, CA, C, O) order by the featurizer, so
    column 1 is CA. It is present on every branch the side-chain module runs
    on. `aa_bb_atom_idx` is NOT -- that one arrives with the Stage IV work --
    so the fallback reconstructs CA from atom-to-token ownership: an [xpb]
    design token carries exactly those four atoms in that order (report p23),
    making its second atom the CA.
    """
    bb = feat.get("sc_bb_atom_idx")
    if bb is not None:
        rows = bb.detach().cpu().numpy()[:, 1]
        if (rows[design] >= 0).sum() >= 4:
            return rows
    a2t = feat.get("atom_to_token_idx")
    if a2t is None:
        return None
    owner = a2t.detach().cpu().numpy().astype(int)
    rows = np.full(design.shape[0], -1, dtype=int)
    for token in np.nonzero(design)[0]:
        atoms = np.nonzero(owner == token)[0]
        if atoms.size >= 2:
            rows[token] = atoms[1]          # N, CA, C, O -> CA
    return rows


def geometry(coords: np.ndarray, design: np.ndarray, ca_rows: np.ndarray | None) -> dict:
    """CA-CA statistics over consecutive design residues."""
    if ca_rows is None:
        return {}
    keep = design & (ca_rows >= 0)
    if int(keep.sum()) < 4:
        return {}
    ca = coords[ca_rows[keep]]
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    centred = ca - ca.mean(0)
    return {
        "n_residues": int(keep.sum()),
        "ca_ca_median": float(np.median(d)),
        "ca_ca_min": float(d.min()),
        "ca_ca_max": float(d.max()),
        "bad_bond_fraction": float((np.abs(d - CA_IDEAL) > CA_TOL).mean()),
        "radius_of_gyration": float(np.sqrt((centred ** 2).sum(1).mean())),
        "end_to_end": float(np.linalg.norm(ca[0] - ca[-1])),
    }


def main() -> None:
    evaluation_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(evaluation_dir.parent / "training"))

    from infer_aa_head_protenix_monomer import parse_args, _seed_everything, _write_csv
    # Defined in export_protenix_backbones_for_mpnn, not in the monomer script
    # that uses it -- it retries the crop, which a bad row needs.
    from export_protenix_backbones_for_mpnn import _source_item_with_crop

    # Reuse the free-backbone monomer harness's own CLI so the data path,
    # index filtering and crop policy are identical to it.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sampler-mode", default="pxdesign_native",
                   choices=["pxdesign_native", "minimal_euler"])
    known, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = parse_args()

    from train_protenix_monomer import (
        _bootstrap_paths, _recent_index_path, apply_training_stage_args,
        build_components, build_configs, build_monomer_index, fill_missing_args,
    )

    args.training_stage = "aa_head_warmup"
    args.data_mode = "monomer"
    # build_configs / build_components read flags this probe's parser does not
    # define. `fill_missing_args` exists for exactly that -- its docstring
    # records five eval scripts dying one attribute at a time -- so back-fill
    # from the training parser's defaults rather than chasing AttributeErrors.
    args = fill_missing_args(args)
    apply_training_stage_args(args)
    _bootstrap_paths(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    source_index = (Path(args.source_index).resolve() if args.source_index
                    else _recent_index_path(data_root))
    filtered_index = (Path(args.filtered_index).resolve() if args.filtered_index
                      else output_dir / "monomer_index.csv.gz")
    build_monomer_index(
        source_index=source_index, output_index=filtered_index,
        min_n_token=int(args.min_n_token), max_n_token=int(args.max_n_token),
        limit=int(args.limit_index), rebuild=bool(args.rebuild_index),
    )
    components, n_items = build_components(args, filtered_index)
    # Two different objects, as in infer_aa_head_protenix_monomer: the trainer
    # wants the curriculum WRAPPER (it reads `source_names` off it), while item
    # access must go through `.datasets[0]` -- indexing the wrapper samples by
    # training weight instead of walking the held-out rows in order, which
    # would not raise, only quietly return the wrong samples.
    source_dataset = components.train_dataset.datasets[0]

    # Everything above is cheap and is exactly where the wiring breaks:
    # imports, back-filled args, index build, dataset shape, first item.
    # Validate it on a login node before spending a GPU slot -- five
    # submissions died in this setup one error at a time, each costing a queue
    # round-trip to learn a single line.
    if args.dry_run:
        batch, _crop = _source_item_with_crop(source_dataset, int(args.start_index))
        feat = batch["input_feature_dict"]
        design = feat["design_token_mask"].detach().cpu().numpy().astype(bool)
        rows = ca_rows_for_design(feat, design)
        resolved = int((rows[design] >= 0).sum()) if rows is not None else 0
        print(f"rows={n_items} tokens={design.size} design={int(design.sum())}")
        print(f"ca_rows_resolved={resolved}")
        print(f"sampler_mode={known.sampler_mode} n_step={args.n_step}")
        if resolved < 4:
            raise SystemExit("dry run: could not resolve design-token CA rows")
        print("DRY_RUN_OK")
        return

    device = torch.device(args.device)
    from pxdesign_train.cogenerate import cogenerate
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents

    configs = build_configs(args, device)
    trainer = PXDesignTrainer(
        configs=configs,
        components=TrainerComponents(train_dataset=components.train_dataset,
                                     schedule=components.schedule,
                                     train_samples_per_epoch=1, eval_dataloader=None),
        device=device, checkpoint_dir=None,
        load_checkpoint_path=None, checkpoint_params_only=True,
    )
    trainer.load_checkpoint(str(Path(args.checkpoint).expanduser().resolve()),
                            params_only=True)
    trainer.model.eval()
    precision = trainer._train_precision()

    stop = min(n_items, int(args.start_index) + int(args.limit_index or 8))
    rows: list[dict] = []
    for idx in range(int(args.start_index), stop):
        _seed_everything(int(args.seed) + idx)
        try:
            batch, _ = _source_item_with_crop(source_dataset, idx)
            feat = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch["input_feature_dict"].items()}
            ctx = (torch.autocast("cuda", dtype=precision, cache_enabled=False)
                   if device.type == "cuda" else nullcontext())
            with torch.no_grad(), ctx:
                generated = cogenerate(
                    trainer.model, input_feature_dict=feat, N_step=int(args.n_step),
                    sidechain_cycle=False, seq_mode="complete_unmask",
                    sampler_mode=known.sampler_mode,
                )
            coords = generated["coordinate"].detach().float().cpu().numpy()
            if coords.ndim == 3:
                coords = coords[0]
            cpu = batch["input_feature_dict"]
            design = cpu["design_token_mask"].detach().cpu().numpy().astype(bool)
            stats = geometry(coords, design, ca_rows_for_design(cpu, design))
            if not stats:
                raise ValueError("no usable design-token CA rows")
            sequence = generated["sequence"].detach().cpu().numpy()
            sel = sequence[design]
            stats.update(
                index=idx, sample_id=batch.get("sample_id"),
                gly_ala_fraction=float(np.isin(sel, [5, 0]).mean()) if sel.size else float("nan"),
            )
            rows.append(stats)
            print(f"[{idx}] {stats['sample_id']}  L={stats['n_residues']}  "
                  f"CA-CA={stats['ca_ca_median']:.2f}  bad={stats['bad_bond_fraction']:.1%}  "
                  f"Rg={stats['radius_of_gyration']:.1f}", flush=True)
        except Exception as exc:                      # a bad crop must not end the probe
            print(f"[{idx}] skipped: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        raise SystemExit("probe produced no rows")
    _write_csv(output_dir / "monomer_geometry.csv", rows)
    arr = lambda k: np.array([r[k] for r in rows], dtype=float)
    summary = {
        "checkpoint": str(args.checkpoint), "sampler_mode": known.sampler_mode,
        "n_step": int(args.n_step), "n_samples": len(rows),
        "ca_ca_median": float(np.median(arr("ca_ca_median"))),
        "ca_ca_min": float(arr("ca_ca_min").min()),
        "bad_bond_fraction_mean": float(arr("bad_bond_fraction").mean()),
        "radius_of_gyration_median": float(np.median(arr("radius_of_gyration"))),
        "gly_ala_fraction_median": float(np.nanmedian(arr("gly_ala_fraction"))),
        "ideal_ca_ca": CA_IDEAL, "tolerance": CA_TOL,
    }
    (output_dir / "monomer_geometry_summary.json").write_text(json.dumps(summary, indent=2))
    print("MONOMER_GEOMETRY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
