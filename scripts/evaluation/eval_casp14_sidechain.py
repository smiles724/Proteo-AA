#!/usr/bin/env python3
"""Side-chain packing benchmark on CASP14 targets.

THE TASK, as side-chain-packing papers define it: give the model the
experimentally determined backbone and the true amino-acid sequence, ask it to
reconstruct the side-chain atoms, and score side-chain RMSD. That is exactly the
Stage II-A configuration (`--training-stage sidechain_warmup`): GT backbone
frames (`predicted_frame=False`), GT residue types (`force_gt_type_logits=True`),
and S_phi as the only side-chain generator.

WHY IT REUSES `PXDesignTrainer.evaluate()` RATHER THAN A FRESH FORWARD PASS.
`sc_local` already is the masked mean squared deviation, in A^2, between the
predicted side-chain atoms and the GT geometry mapped through the residue frame
(`sidechain_global_frame_aligned_loss`). So the atom-weighted side-chain RMSD is
just `sqrt(sc_local)`, and the per-protein breakdown gives it per target. Writing
a second scoring path would risk disagreeing with the number training optimised.

TARGET SOURCE. The CASP14 public natives (`casp14.targets.T.public_*.tar.gz`)
carry no PDB identifier and leave the chain-ID column blank, and a coordinate-only
CIF converted from them lacks the mmCIF categories Protenix's parser requires
(`pdbx_struct_assembly`, populated `label_seq_id`, ...). This script therefore
scores the deposited PDB entry for each target, which is the same experimental
structure and parses natively. `--mapping` supplies target -> (pdb_id, chain);
build it with `map_casp14_targets.py`.

CAVEAT THE CALLER MUST WEIGH. Our training index is
`weightedPDB_indices_before_2021-09-30`, and CASP14 targets were released in 2020,
so these structures are very likely INSIDE the training pool. Treat the result as
a sanity check against published packing numbers, NOT as held-out generalisation.
`--report-train-overlap` marks which targets appear in the training index.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path,
                    help="JSON from casp_natives_to_cif.py: {targets: [{target, cif, chain, "
                         "native_len}]}. PREFERRED — scores the CASP natives directly, so no "
                         "target->PDB mapping is involved and it works for CASP14/15/16 alike.")
    src.add_argument("--mapping", type=Path,
                    help="JSON: {target: {pdb_ids: [...], seqres_chains: ['6y4f_A', ...]}}. "
                         "Scores the DEPOSITED entry instead; only viable where our local "
                         "seqres/mmCIF snapshots cover the edition (i.e. CASP14).")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--mmcif-dir", type=Path,
                   default=Path("/hai/scratch/yfsun/protenix_data/mmcif"))
    p.add_argument("--data-root", type=Path, default=Path("/hai/scratch/yfsun/protenix_data"))
    p.add_argument("--protenix-code-dir", default="/hai/users/y/f/yfsun/Protein Project/Protenix")
    p.add_argument("--pxdesign-code-dir", default="/hai/users/y/f/yfsun/Protein Project/11/PXDesign")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32", help="fp32 on CPU; bf16 is fine on GPU")
    p.add_argument("--crop-size", type=int, default=1536,
                   help="Must exceed the largest target or it stops being a whole-target score")
    p.add_argument("--targets", default=None, help="comma-separated subset, e.g. T1049,T1026")
    p.add_argument("--limit", type=int, default=0, help="score at most N targets (0 = all)")
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--report-train-overlap", action="store_true",
                   help="flag targets present in the pre-2021-09-30 training index")
    return p.parse_args()


def build_target_list(args) -> tuple[list[dict], list[dict]]:
    """Split the mapping into (scorable targets, explicitly excluded targets).

    Oversized targets are dropped HERE, up front, rather than left to fail inside
    the dataloader. `max_crop_retries=1` deliberately makes a crop failure raise
    instead of silently substituting a different protein, so an over-long target
    would otherwise abort the whole benchmark (T1044/6vr4 is 2194 tokens against
    crop_size 1536 — it killed job 98153 at index 8). Excluding it explicitly and
    reporting it keeps both properties: no silent substitution, no lost run.
    """
    wanted = set(args.targets.split(",")) if args.targets else None
    max_len = int(args.crop_size) - 64

    if args.manifest is not None:
        # Direct-native path: the manifest already names the CIF and the asym id
        # that the parsed AtomArray will use as chain_id.
        man = json.loads(args.manifest.read_text())
        out, excluded = [], list(man.get("failed", []))
        for rec in man.get("targets", []):
            name = rec["target"]
            if wanted is not None and name not in wanted:
                continue
            if not Path(rec["cif"]).exists():
                excluded.append({"target": name, "reason": f"missing CIF {rec['cif']}"})
                continue
            if rec.get("native_len", 0) > max_len:
                excluded.append({"target": name, "native_len": rec["native_len"],
                                 "reason": f"{rec['native_len']} residues exceeds crop_size "
                                           f"{args.crop_size} (limit {max_len})"})
                continue
            out.append({"target": name, "pdb_id": name, "chain": rec.get("chain"),
                        "cif": rec["cif"], "native_len": rec.get("native_len")})
            if args.limit and len(out) >= args.limit:
                break
        return out, excluded

    mapping = json.loads(args.mapping.read_text())
    # `max_len` (set above) leaves headroom because the deposited entry's chain is
    # usually a little longer than the CASP native, e.g. 2166 -> 2194.
    out: list[dict] = []
    excluded: list[dict] = []
    for target in sorted(mapping):
        if wanted is not None and target not in wanted:
            continue
        rec = mapping[target]
        if not rec.get("pdb_ids"):
            excluded.append({"target": target, "reason": "no PDB entry matched"})
            continue
        native_len = rec.get("native_len") or 0
        if native_len > max_len:
            excluded.append({
                "target": target, "native_len": native_len,
                "reason": f"{native_len} residues exceeds crop_size {args.crop_size} "
                          f"(limit {max_len}); scoring it would crop the target and "
                          "stop being a whole-target number",
            })
            continue
        # Prefer a pdb_id whose matched seqres chain we know, so the binder chain
        # is the chain that actually carries the target sequence — scoring the
        # wrong chain of a multi-chain entry would silently measure another protein.
        chain = None
        pdb_id = rec["pdb_ids"][0]
        for tag in rec.get("seqres_chains", []):
            if "_" in tag:
                cand_id, cand_chain = tag.split("_", 1)
                if cand_id in rec["pdb_ids"]:
                    pdb_id, chain = cand_id, cand_chain
                    break
        cif = args.mmcif_dir / f"{pdb_id}.cif"
        if not cif.exists():
            excluded.append({"target": target, "pdb_id": pdb_id,
                             "reason": f"{cif.name} not present in {args.mmcif_dir}"})
            continue
        out.append({"target": target, "pdb_id": pdb_id, "chain": chain,
                    "cif": str(cif), "native_len": rec.get("native_len")})
        if args.limit and len(out) >= args.limit:
            break
    return out, excluded


def train_index_overlap(args, targets: list[dict]) -> dict:
    """Which scored PDB entries appear in the training index (contamination)."""
    import pandas as pd
    idx = (args.data_root / "indices" /
           "weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz")
    if not idx.exists():
        return {}
    d = pd.read_csv(idx, usecols=["pdb_id", "release_date"], low_memory=False)
    have = dict(zip(d.pdb_id, d.release_date))
    return {t["target"]: {"pdb_id": t["pdb_id"], "in_training_index": t["pdb_id"] in have,
                          "release_date": have.get(t["pdb_id"])} for t in targets}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for extra in (args.protenix_code_dir, args.pxdesign_code_dir):
        if extra and extra not in sys.path:
            sys.path.insert(0, extra)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(args.data_root))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    import torch
    from torch.utils.data import DataLoader

    from pxdesign_train.runner import (
        DesignSourceDataset, PXDesignTrainer, TrainerComponents,
    )
    from pxdesign_train.runner.cif_provider import CifFileProvider
    from pxdesign_train.runner.trainer import _identity_collate
    from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule

    targets, excluded = build_target_list(args)
    if not targets:
        raise SystemExit("no scorable targets — check --mapping / --mmcif-dir")
    logging.info("scoring %d CASP14 targets (%d excluded)", len(targets), len(excluded))
    for t in targets:
        logging.info("  %-9s -> %s chain %s (native %s res)",
                     t["target"], t["pdb_id"], t["chain"], t["native_len"])
    # Never let a dropped target look like a covered one.
    for e in excluded:
        logging.warning("  EXCLUDED %-9s %s", e["target"], e["reason"])

    def make_loader(entry: dict) -> DataLoader:
        """A one-item loader for a single target.

        Each target gets its OWN dataset so a failure is isolated to that target.
        A shared multi-target loader made the benchmark all-or-nothing: with
        `max_crop_retries=1` (deliberate — never silently substitute a different
        protein) a single bad entry aborted the whole run inside the dataloader.
        Job 98153 died that way on T1044 (2194 tokens > crop), and 98155 on
        T1046s2, whose matched seqres chain "T" is not a protein chain of 6psk.

        `compute_sidechain` builds the GT side-chain targets; `ref_pos_augment=False`
        keeps the score deterministic rather than re-randomising the reference frame.
        """
        prov = CifFileProvider(
            cif_paths=[entry["cif"]],
            binder_chain_ids=[entry["chain"]] if entry.get("chain") else None,
        )
        ds = DesignSourceDataset(
            prov, source_name="casp14",
            crop_size=int(args.crop_size),
            compute_sidechain=True, backbone_only_binder=True,
            inference_safe_binder=True, ref_pos_augment=False,
            hotspot_force_zero_prob=0.0,
            aa_mask_mode="all", aa_mask_prob=1.0,
            max_crop_retries=1,      # never silently substitute another target
        )
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=_identity_collate)

    dataset = make_loader(targets[0]).dataset   # for the trainer's components
    loader = make_loader(targets[0])

    # Stage II-A configs, straight from the training script so the scored
    # configuration cannot drift from the trained one.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tpm", str(REPO_ROOT / "scripts" / "training" / "train_protenix_monomer.py"))
    tpm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpm)
    # The architecture comes from the checkpoint's own record, never from this
    # script's defaults. Hardcoding `--training-stage sidechain_warmup` and nothing
    # else takes that stage's DEFAULTS -- frame-aware head, template residual --
    # which describe only one of the arms. Loading a checkpoint trained under a
    # different one is shape-compatible and silent.
    ckpt_head = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    arch = ckpt_head.get("sidechain_arch") or {}
    if not arch:
        raise SystemExit(
            f"{args.checkpoint} records no sidechain_arch, so the configuration its "
            "S_phi was trained under is unknown and cannot be reconstructed here."
        )
    print(f"checkpoint_arch={json.dumps(arch, sort_keys=True)}")

    def _flag(name, on):
        return f"--sc-{name}" if on else f"--no-sc-{name}"

    saved_argv = sys.argv
    sys.argv = [
        "x", "--training-stage", "sidechain_warmup",
        "--data-root", str(args.data_root),
        "--protenix-code-dir", args.protenix_code_dir,
        "--pxdesign-code-dir", args.pxdesign_code_dir,
        "--output-dir", str(args.output_dir),
        "--crop-size", str(args.crop_size), "--max-n-token", str(args.crop_size),
        "--eval-interval", "0", "--eval-samples", "0", "--num-workers", "0",
        "--device", args.device, "--dtype", args.dtype,
        "--template-provider", args.template_provider,
        _flag("frame-aware-head", arch.get("frame_aware_head", False)),
        _flag("centre-coord-input", arch.get("centre_coord_input", False)),
        _flag("template-residual", arch.get("template_residual", False)),
        _flag("edm", arch.get("edm", False)),
    ]
    cfg_args = tpm.parse_args()
    device = torch.device(args.device)
    configs = tpm.build_configs(cfg_args, device)
    sys.argv = saved_argv
    configs.sidechain.bb_context = bool(arch.get("bb_context", True))
    configs.sidechain.a_bs_concat = bool(arch.get("a_bs_concat", True))
    configs.sidechain.q_bs = bool(arch.get("q_bs", False))
    configs.sidechain.type_logits_input = bool(arch.get("type_logits_input", True))
    # `checkpoint_include_prefixes` exists so Stage II can warm-start the BACKBONE
    # and train S_phi from scratch. Applied to an EVALUATION it drops the very
    # module being scored: the previous CASP runs logged "kept 732/1151 tensors,
    # missing=419" and were scoring a randomly-initialised S_phi.
    configs.training.checkpoint_include_prefixes = []

    schedule = CurriculumSchedule(stage1={"casp14": 1.0}, stage2={"casp14": 1.0},
                                  stage1_end_step=1, stage2_start_step=2)
    multi = CurriculumMultiDataset(datasets=[dataset], source_names=["casp14"],
                                   per_item_weights=[[1.0] * len(dataset)])
    components = TrainerComponents(train_dataset=multi, schedule=schedule,
                                   train_samples_per_epoch=1, eval_dataloader=loader)

    trainer = PXDesignTrainer(
        configs=configs, components=components, device=device,
        checkpoint_dir=None, load_checkpoint_path=str(args.checkpoint.resolve()),
        checkpoint_params_only=True,
    )

    # Verify the thing being benchmarked actually arrived. Comparing tensors, not
    # reading a log line: this failed silently once and the numbers looked
    # plausible enough to be published from.
    live = trainer.model.state_dict()
    sc_keys = [k for k in ckpt_head["model"] if k.startswith("sidechain_module.")]
    if not sc_keys:
        raise SystemExit("checkpoint contains no sidechain_module weights")
    bad = [k for k in sc_keys
           if k not in live or not torch.equal(live[k].cpu(), ckpt_head["model"][k].cpu())]
    if bad:
        raise SystemExit(
            f"{len(bad)}/{len(sc_keys)} sidechain_module tensors did not survive the "
            f"load (e.g. {bad[:3]}). Scoring would measure an untrained S_phi."
        )
    print(f"verified {len(sc_keys)} sidechain_module tensors loaded")
    del ckpt_head

    # Score one target at a time so a single unusable entry costs one row, not
    # the whole benchmark. sc_local is mean squared deviation in A^2, so the
    # side-chain RMSD is its square root.
    per_target = []
    for entry in targets:
        name = entry["target"]
        try:
            trainer.eval_dl = make_loader(entry)
            m = trainer.evaluate()
            rows = trainer.last_eval_per_protein
            if not rows:
                raise RuntimeError("evaluate() returned no per-protein row")
            sc = float(rows[0].get("sc_local", float("nan")))
            rmsd = math.sqrt(sc) if sc == sc and sc >= 0 else None
            per_target.append({
                "target": name, "pdb_id": entry["pdb_id"], "chain": entry.get("chain"),
                "native_len": entry.get("native_len"),
                "sc_local_A2": sc, "sidechain_rmsd_A": rmsd,
                "set_metrics": {k: float(v) for k, v in m.items()},
            })
            logging.info("  scored %-9s %s  sc_local=%.4f  RMSD=%.3f A",
                         name, entry["pdb_id"], sc, rmsd if rmsd else float("nan"))
        except Exception as exc:                      # noqa: BLE001 — isolate the target
            excluded.append({"target": name, "pdb_id": entry["pdb_id"],
                             "chain": entry.get("chain"),
                             "reason": f"scoring failed: {type(exc).__name__}: {str(exc)[:200]}"})
            logging.warning("  FAILED  %-9s %s: %s", name, entry["pdb_id"], str(exc)[:160])

    per_target.sort(key=lambda x: (x["sidechain_rmsd_A"] is None, x["sidechain_rmsd_A"]))
    good = [p["sidechain_rmsd_A"] for p in per_target if p["sidechain_rmsd_A"] is not None]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "n_targets_scored": len(per_target),
        "n_targets_excluded": len(excluded),
        "excluded_targets": excluded,
        # Macro average: each target counts once regardless of size, which is the
        # convention packing papers report. An atom-weighted pooled number is NOT
        # reported here: each target's sc_local is already a mean over its own
        # atoms, so pooling correctly needs per-target atom counts, and faking it
        # by averaging the means would silently under-weight the large targets.
        "sidechain_rmsd_A_mean_over_targets": (sum(good) / len(good)) if good else None,
        "sidechain_rmsd_A_median_over_targets": (sorted(good)[len(good) // 2]) if good else None,
    }
    if args.report_train_overlap:
        summary["train_index_overlap"] = train_index_overlap(args, targets)

    out = {"summary": summary, "per_target": per_target}
    (args.output_dir / "casp14_sidechain.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== CASP14 side-chain packing (GT backbone + GT sequence) ===")
    print(f"{'target':<10}{'pdb':<7}{'len':>5}  {'sc_local(A^2)':>13}  {'RMSD(A)':>8}")
    for p in per_target:
        rmsd = "n/a" if p["sidechain_rmsd_A"] is None else f"{p['sidechain_rmsd_A']:.3f}"
        print(f"{p['target']:<10}{p['pdb_id']:<7}{str(p['native_len']):>5}  "
              f"{p['sc_local_A2']:>13.4f}  {rmsd:>8}")
    if excluded:
        print(f"\nEXCLUDED {len(excluded)} target(s) — not covered by this number:")
        for e in excluded:
            print(f"  {e['target']:<10}{e['reason']}")
    print(f"\ntargets scored              : {summary['n_targets_scored']}")
    if summary["sidechain_rmsd_A_mean_over_targets"]:
        print(f"side-chain RMSD (mean/target): {summary['sidechain_rmsd_A_mean_over_targets']:.3f} A")
        print(f"side-chain RMSD (median)     : {summary['sidechain_rmsd_A_median_over_targets']:.3f} A")
    print(f"wrote {args.output_dir / 'casp14_sidechain.json'}")


if __name__ == "__main__":
    main()
