#!/usr/bin/env python3
"""Run the A-CODE ConditionalBinderDesignBenchmark on a Stage III checkpoint.

This is the generation half of the benchmark. For every (target, binder length)
cell it co-generates `n_samples` binders with `pxdesign_train.cogenerate`, writes
each design out as a target+binder PDB plus a FASTA, and records one row per
design in `designs.csv`. Designability then comes from folding those designs with
AF2 initial-guess and running `score_af2ig_designability.py` over the metrics —
this script deliberately does not shell out to AF2, because the filter is a
published, fixed threshold set and keeping it in a separate step means a rerun of
the filter never needs a GPU.

Two details that matter for the numbers:

* **The target is re-anchored, not trusted from the sample.** PXDesign-d
  soft-conditions the target through binned pair distances and denoises *every*
  atom from noise, target included, so a sample's target coordinates are a
  prediction. Writing that out would hand AF2-IG a binder posed against a
  predicted target. Each design is therefore superimposed back onto the
  deposited target (target CA atoms) before it is written, so every output shares
  one frame: the real one.

* **Variants.** A-CODE reports the co-designed sequence and a single
  ProteinMPNN redesign separately. This script produces the co-design arm and
  stamps every row `variant=co_design`. The PMPNN arm reuses the same backbones:
  run `scripts/evaluation/run_proteinmpnn_recovery.py`-style redesign over
  `designs/*.pdb` at temperature 0.0001 (PXDesign's setting), one sequence per
  structure, and record the result as `variant=pmpnn` in the metrics CSV.

Example::

    python scripts/evaluation/eval_conditional_binder_benchmark.py \
        --checkpoint /path/to/stage3/checkpoints/step30000.pt \
        --output-dir /path/to/runs/cbdb \
        --targets PDL1 SC2RBD --lengths 80 100 --samples-per-length 8
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("cbdb")

# 20-AA index -> 3-letter, in the order `cogenerate` returns.
AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA1 = list("ARNDCQEGHILKMFPSTWYV")


# ------------------------------------------------------------------- the model


def _stage3_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Build the `build_configs` namespace for Stage III (co-evolution).

    The namespace is taken from the training script's own `parse_args()` defaults
    rather than hand-listed here. Hand-listing was tried first and is a
    maintenance trap: `build_configs` reads ~45 attributes, the set grows
    whenever a config flag is added, and every omission surfaces as an
    `AttributeError` deep inside config construction (this cost two rounds of
    exactly that). Reading the defaults from the parser means a new training flag
    arrives with its training default already in place.

    Only eval-side facts are then overridden — one step, no optimiser, no EMA —
    plus the checkpoint path, which `adopt_sidechain_arch_from_checkpoint` needs
    to take S_phi's layout from the checkpoint's own record. Note that no
    side-chain architecture key is pinned here on purpose: pinning them is the
    failure that function's docstring describes, where a Stage III run either
    cannot load its Stage II ancestor or silently arms an untrained channel.
    """
    from train_protenix_monomer import apply_training_stage_args
    from train_protenix_monomer import parse_args as train_parse_args

    saved_argv = sys.argv
    try:
        # No flag of the training CLI is required, so an empty argv yields a
        # namespace of pure defaults. Attributes are then set by name, which
        # keeps this independent of the training CLI's flag *spelling*.
        sys.argv = ["train_protenix_monomer.py"]
        namespace = train_parse_args()
    finally:
        sys.argv = saved_argv

    namespace.training_stage = args.training_stage
    namespace.seed = int(args.seed)
    namespace.dtype = args.dtype
    namespace.crop_size = int(args.crop_size)
    namespace.template_provider = args.template_provider
    namespace.sc_ablation_arm = args.sc_ablation_arm
    # Read by adopt_sidechain_arch_from_checkpoint.
    namespace.load_checkpoint = str(args.checkpoint)
    # Sampling only: one nominal step, no optimiser schedule, no EMA shadow.
    namespace.max_steps = 1
    namespace.warmup_steps = 0
    namespace.ema_decay = 0.0
    namespace.num_workers = 0
    namespace.iters_to_accumulate = 1
    namespace.log_interval = 0
    namespace.eval_interval = 0
    namespace.checkpoint_interval = 0

    apply_training_stage_args(namespace)
    return namespace


def _build_trainer(args: argparse.Namespace, device: torch.device, dataset):
    from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents
    from train_protenix_monomer import build_configs

    namespace = _stage3_namespace(args)
    configs = build_configs(namespace, device)

    source = "cbdb_bootstrap"
    multi = CurriculumMultiDataset(
        datasets=[dataset],
        source_names=[source],
        per_item_weights=[[1.0] * len(dataset)],
    )
    schedule = CurriculumSchedule(
        stage1={source: 1.0},
        stage2={source: 1.0},
        stage1_end_step=0,
        stage2_start_step=0,
    )
    trainer = PXDesignTrainer(
        configs=configs,
        components=TrainerComponents(
            train_dataset=multi, schedule=schedule, train_samples_per_epoch=1
        ),
        device=device,
        checkpoint_dir=None,
    )
    trainer.load_checkpoint(str(args.checkpoint), params_only=True)
    trainer.model.eval()
    return trainer


# ------------------------------------------------------------------ the output


def _superimpose_onto_deposited(
    generated: np.ndarray,
    feature_dict: dict[str, torch.Tensor],
    label_coord: np.ndarray,
) -> np.ndarray:
    """Move a sample into the deposited target's frame.

    Fits on the target CA atoms (the condition, whose true pose we know) and
    applies the resulting rigid transform to every atom, binder included. Returns
    the transformed coordinates.
    """
    import biotite.structure as struc

    design_token = feature_dict["design_token_mask"].bool().cpu()
    atom_to_token = feature_dict["atom_to_token_idx"].long().cpu()
    is_target_atom = ~design_token[atom_to_token]
    is_ca = feature_dict["eval_ca_atom_mask"].bool().cpu()
    fit = (is_target_atom & is_ca).numpy()
    if fit.sum() < 3:
        raise ValueError("fewer than 3 target CA atoms to superimpose on")

    _, transform = struc.superimpose(label_coord[fit], generated[fit])
    return transform.apply(generated)


def _write_design_pdb(
    path: Path,
    prepared,
    feature_dict: dict[str, torch.Tensor],
    coords: np.ndarray,
    sequence: np.ndarray,
    target_atom_array,
    sidechain: dict[int, dict[str, Any]],
) -> None:
    """Write the deposited target plus the designed binder as one PDB.

    PDB rather than mmCIF because every AF2 initial-guess wrapper in circulation
    (dl_binder_design, BindCraft's filter path) takes PDBs. Target atoms are the
    deposited coordinates in author numbering, so the file opens cleanly next to
    the original entry; binder atoms come from the sample — backbone from the
    diffusion output, side chains from S_phi when the co-evolution cycle ran.
    The target is written first and the binder last, which is the chain order
    those wrappers assume when they pick out "the design".
    """
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    from pxdesign_train.benchmarks.target_prep import BACKBONE_ATOMS

    design_token = feature_dict["design_token_mask"].bool().cpu().numpy()
    atom_to_token = feature_dict["atom_to_token_idx"].long().cpu().numpy()

    design_token_ids = np.where(design_token)[0]
    token_to_binder_index = {int(t): i for i, t in enumerate(design_token_ids)}

    records: list[tuple[str, int, str, str, np.ndarray]] = []
    # Binder backbone, in the four-atom order the inference-safe rebuild uses.
    per_token_atoms: dict[int, list[int]] = {}
    for atom_index, token_index in enumerate(atom_to_token):
        if design_token[token_index]:
            per_token_atoms.setdefault(int(token_index), []).append(atom_index)
    for token_index, atom_indices in sorted(per_token_atoms.items()):
        binder_index = token_to_binder_index[token_index]
        aa_index = int(sequence[token_index])
        res_name = AA3[aa_index] if 0 <= aa_index < len(AA3) else "GLY"
        for slot, atom_index in enumerate(atom_indices[: len(BACKBONE_ATOMS)]):
            records.append(
                (
                    prepared.binder_chain_id,
                    binder_index + 1,
                    res_name,
                    BACKBONE_ATOMS[slot],
                    coords[atom_index],
                )
            )
        entry = sidechain.get(token_index) or sidechain.get(str(token_index))
        if entry:
            for name, xyz in zip(entry["atom_names"], np.asarray(entry["coords"])):
                if name in BACKBONE_ATOMS:
                    continue
                records.append(
                    (
                        prepared.binder_chain_id,
                        binder_index + 1,
                        str(entry.get("restype3", res_name)),
                        str(name),
                        np.asarray(xyz, dtype=float),
                    )
                )

    n_target_atoms = target_atom_array.array_length()
    array = struc.AtomArray(n_target_atoms + len(records))
    array.coord[:n_target_atoms] = target_atom_array.coord
    array.chain_id[:n_target_atoms] = target_atom_array.chain_id
    array.res_id[:n_target_atoms] = target_atom_array.res_id
    array.res_name[:n_target_atoms] = target_atom_array.res_name
    array.atom_name[:n_target_atoms] = target_atom_array.atom_name
    array.element[:n_target_atoms] = target_atom_array.element
    array.hetero[:n_target_atoms] = False
    for offset, (chain_id, res_id, res_name, atom_name, xyz) in enumerate(records):
        i = n_target_atoms + offset
        array.coord[i] = xyz
        array.chain_id[i] = chain_id
        array.res_id[i] = res_id
        array.res_name[i] = res_name
        array.atom_name[i] = atom_name
        array.element[i] = atom_name[0]
        array.hetero[i] = False

    pdb = PDBFile()
    pdb.set_structure(array)
    pdb.write(str(path))


def _binder_sequence_string(sequence: np.ndarray, design_token: np.ndarray) -> str:
    out = []
    for token_index in np.where(design_token)[0]:
        aa_index = int(sequence[token_index])
        out.append(AA1[aa_index] if 0 <= aa_index < len(AA1) else "X")
    return "".join(out)


# -------------------------------------------------------------------- the loop


def _iter_pending(rows_done: set[str], task) -> list[str]:
    return [sid for sid in task.sample_ids() if sid not in rows_done]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    training_dir = REPO_ROOT / "scripts" / "training"
    sys.path.insert(0, str(training_dir))
    from train_protenix_monomer import _bootstrap_paths

    _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    from pxdesign_train.benchmarks import ConditionalBinderDesignBenchmark
    from pxdesign_train.benchmarks.target_prep import (
        featurize_prepared_input,
        load_cropped_target,
        make_design_dataset,
        prepare_task,
    )

    benchmark = (
        ConditionalBinderDesignBenchmark.load(args.manifest)
        if args.manifest
        else ConditionalBinderDesignBenchmark.load()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    prep_dir = output_dir / "inputs"
    design_dir = output_dir / "designs"
    for directory in (prep_dir, design_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = benchmark.tasks(
        only=args.targets or None,
        lengths=args.lengths or None,
        samples_per_length=args.samples_per_length,
    )
    pending = benchmark.pending_targets()
    if pending:
        logger.warning(
            "skipping %d target(s) with no published definition in the manifest: %s. "
            "Table 4 has 10 columns; this run can fill %d of them.",
            len(pending),
            ", ".join(t.name for t in pending),
            len(benchmark.runnable_targets()),
        )
    if not tasks:
        raise SystemExit("no runnable tasks selected")

    logger.info(
        "benchmark=%s %s tasks=%d designs=%d filter: %s",
        "ConditionalBinderDesignBenchmark",
        benchmark.version,
        len(tasks),
        sum(t.n_samples for t in tasks),
        benchmark.af2ig.describe(),
    )

    # Prepare every input up front: a missing structure or a hotspot that does not
    # resolve should stop the run before a GPU is touched.
    prepared_by_task = {}
    for task in tasks:
        prepared_by_task[task.task_id] = prepare_task(
            task, args.mmcif_dir, prep_dir, overwrite=args.rebuild_inputs
        )
        logger.info(
            "prepared %-16s target_res=%-4d binder=%-4d hotspots=%s",
            task.task_id,
            prepared_by_task[task.task_id].n_target_residues,
            task.binder_length,
            ",".join(
                f"{c}{r}" for c, r in prepared_by_task[task.task_id].hotspots_author
            ),
        )
    if args.validate_only:
        print(json.dumps({"validated_tasks": [t.task_id for t in tasks]}, indent=2))
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but no GPU is available")
    device = torch.device(args.device)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    bootstrap_dataset, _ = make_design_dataset(
        prepared_by_task[tasks[0].task_id],
        crop_size=args.crop_size,
        compute_sidechain=args.compute_sidechain,
    )
    trainer = _build_trainer(args, device, bootstrap_dataset)

    from pxdesign_train.cogenerate import cogenerate

    designs_csv = output_dir / "designs.csv"
    done: set[str] = set()
    if designs_csv.is_file() and not args.overwrite:
        with designs_csv.open(newline="") as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle)}
        logger.info("resuming: %d designs already recorded", len(done))

    fieldnames = [
        "sample_id", "target", "pdb_id", "binder_length", "seed", "variant",
        "sequence", "design_pdb", "n_hotspots", "has_full_atom_sidechain",
        "n_sidechain_atoms", "seconds",
    ]
    handle = designs_csv.open("a" if done else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not done:
        writer.writeheader()
        handle.flush()

    for task in tasks:
        prepared = prepared_by_task[task.task_id]
        remaining = _iter_pending(done, task)
        if not remaining:
            logger.info("%s: complete, skipping", task.task_id)
            continue

        dataset, token_index = make_design_dataset(
            prepared, crop_size=args.crop_size, compute_sidechain=args.compute_sidechain
        )
        target_atom_array = load_cropped_target(
            _structure_path(args.mmcif_dir, task.target.pdb_id),
            task.target.crop_ranges(),
        )
        item = featurize_prepared_input(
            prepared,
            crop_size=args.crop_size,
            compute_sidechain=args.compute_sidechain,
            dataset=dataset,
            token_index=token_index,
        )
        batch = trainer._to_device({k: v for k, v in item.items() if k != "prepared"})
        feature_dict = batch["input_feature_dict"]
        label_coord = item["label_dict"]["coordinate"].cpu().numpy()
        design_token = item["input_feature_dict"]["design_token_mask"].bool().cpu().numpy()

        dtype = trainer._train_precision()
        autocast = (
            torch.autocast("cuda", dtype=dtype, cache_enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )

        for sample_id in remaining:
            seed = _sample_seed(args.seed, sample_id)
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            started = time.time()
            with torch.no_grad(), autocast:
                result = cogenerate(
                    trainer.raw_model,
                    feature_dict,
                    N_step=args.diffusion_steps,
                    sidechain_cycle=args.sidechain_cycle,
                    seq_mode=args.seq_mode,
                )
            coords = result["coordinate"].float().cpu().numpy()
            sequence = result["sequence"].cpu().numpy()
            while sequence.ndim > 1:
                sequence = sequence[0]
            coords = _superimpose_onto_deposited(coords, item["input_feature_dict"], label_coord)

            sidechain = result.get("sidechain", {}) or {}
            design_pdb = design_dir / f"{sample_id}.pdb"
            _write_design_pdb(
                design_pdb,
                prepared,
                item["input_feature_dict"],
                coords,
                sequence,
                target_atom_array,
                sidechain,
            )
            sequence_string = _binder_sequence_string(sequence, design_token)
            (design_dir / f"{sample_id}.fasta").write_text(
                f">{sample_id} target={task.target.name} length={task.binder_length}\n"
                f"{sequence_string}\n"
            )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "target": task.target.name,
                    "pdb_id": (task.target.pdb_id or "").upper(),
                    "binder_length": task.binder_length,
                    "seed": seed,
                    "variant": "co_design",
                    "sequence": sequence_string,
                    "design_pdb": str(design_pdb),
                    "n_hotspots": len(prepared.hotspots_author),
                    "has_full_atom_sidechain": bool(result.get("has_full_atom_sidechain")),
                    "n_sidechain_atoms": sum(
                        int(np.asarray(v["coords"]).shape[0]) for v in sidechain.values()
                    ),
                    "seconds": round(time.time() - started, 2),
                }
            )
            handle.flush()
            done.add(sample_id)
            logger.info(
                "%s seq[:20]=%s sidechain_atoms=%d %.1fs",
                sample_id,
                sequence_string[:20],
                sum(int(np.asarray(v["coords"]).shape[0]) for v in sidechain.values()),
                time.time() - started,
            )

    handle.close()

    summary = {
        "benchmark": "ConditionalBinderDesignBenchmark",
        "manifest_version": benchmark.version,
        "checkpoint": str(checkpoint),
        "training_stage": args.training_stage,
        "diffusion_steps": args.diffusion_steps,
        "sidechain_cycle": bool(args.sidechain_cycle),
        "seq_mode": args.seq_mode,
        "filter": benchmark.af2ig.describe(),
        "chain_break_offset": benchmark.af2ig.chain_break_offset,
        "targets_run": sorted({t.target.name for t in tasks}),
        "targets_pending": [t.name for t in benchmark.pending_targets()],
        "n_designs": len(done),
        "designs_csv": str(designs_csv),
        "next_step": (
            "fold designs/ with AF2 initial-guess (binder chain offset "
            f"{benchmark.af2ig.chain_break_offset}), then run "
            "scripts/evaluation/score_af2ig_designability.py"
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _structure_path(mmcif_dir: str, pdb_id: Optional[str]) -> Path:
    directory = Path(mmcif_dir)
    for candidate in (
        directory / f"{pdb_id}.cif",
        directory / f"{pdb_id}.cif.gz",
        directory / str(pdb_id)[1:3] / f"{pdb_id}.cif.gz",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no structure for {pdb_id} under {directory}")


def _sample_seed(base: int, sample_id: str) -> int:
    """Deterministic per-design seed, stable across resumes and task ordering."""
    import hashlib

    digest = hashlib.sha256(f"{base}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Stage III (co-evolution) checkpoint")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--manifest", default=None, help="override the test-set manifest JSON")
    p.add_argument(
        "--mmcif-dir",
        default="/hai/scratch/yfsun/protenix_data/mmcif",
        help="local mmCIF mirror the target PDB entries are read from",
    )
    p.add_argument("--targets", nargs="*", default=None, help="subset of target names")
    p.add_argument("--lengths", nargs="*", type=int, default=None, help="override the length grid")
    p.add_argument(
        "--samples-per-length",
        type=int,
        default=None,
        help="override the manifest's samples-per-length (default keeps the paper-scale grid)",
    )
    p.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="A-CODE follows PXDesign with 1000 Euler steps; PXDesign's own report used 400 for binders",
    )
    p.add_argument("--seq-mode", default="complete_unmask", choices=["complete_unmask", "sequential"])
    p.add_argument(
        "--sidechain-cycle",
        action="store_true",
        help="run S_phi inside the reverse loop so designs carry full-atom side chains",
    )
    p.add_argument(
        "--compute-sidechain",
        action="store_true",
        help="also compute side-chain targets during featurization (not needed for sampling)",
    )
    p.add_argument("--training-stage", default="coevolution", choices=["coevolution", "predicted_mask"])
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--sc-ablation-arm", default="default")
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--validate-only", action="store_true", help="prepare inputs and exit")
    p.add_argument("--rebuild-inputs", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="ignore an existing designs.csv")
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    return p.parse_args()


if __name__ == "__main__":
    main()
