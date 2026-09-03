#!/usr/bin/env python3
"""Pretrain Proteo-AA on monomer-only Protenix wwPDB data.

This driver wires the existing Proteo-AA trainer to Protenix preprocessed data.
It filters the Protenix weighted-PDB index to strict protein monomer rows:

    type == "chain"
    mol_1_type == "prot"
    num_prot_chains == 1
    num_tokens within [min_n_token, max_n_token]

Protenix then returns the full reference protein chain (`crop_size=0`,
`use_reference_chains_only=True`), and Proteo-AA treats that whole monomer as
the design/binder region.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional


def _has_package(path: Path, package: str) -> bool:
    return (path / package / "__init__.py").exists()


def _resolve_code_dir(
    package: str,
    explicit: Optional[str],
    repo_root: Path,
) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not _has_package(p, package):
            raise FileNotFoundError(f"{p} does not contain package {package!r}")
        return p

    repo_names = {
        "protenix": ["Protenix", "protenix"],
        "pxdesign": ["PXDesign", "pxdesign"],
    }.get(package, [package])
    candidates = []
    for name in repo_names:
        candidates.extend(
            [
                repo_root / name,
                repo_root.parent / name,
                repo_root.parent / "Protein Project" / name,
                repo_root.parent / "Protein Project" / "11" / name,
                Path.home() / "Protein Project" / name,
                Path.home() / "Protein Project" / "11" / name,
            ]
        )
    for p in candidates:
        if _has_package(p, package):
            return p.resolve()
    return None


def _bootstrap_paths(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    protenix_dir = _resolve_code_dir("protenix", args.protenix_code_dir, repo_root)
    pxdesign_dir = _resolve_code_dir("pxdesign", args.pxdesign_code_dir, repo_root)

    for p in [repo_root, pxdesign_dir, protenix_dir]:
        if p is not None:
            sys.path.insert(0, str(p))

    if protenix_dir is None:
        raise FileNotFoundError(
            "Could not find Protenix source. Pass --protenix-code-dir."
        )
    if pxdesign_dir is None:
        raise FileNotFoundError(
            "Could not find PXDesign source. Pass --pxdesign-code-dir."
        )
    return repo_root


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"runs/protenix_monomer_{stamp}"


def _source_index_path(data_root: Path) -> Path:
    return (
        data_root
        / "indices"
        / "weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz"
    )


def _recent_index_path(data_root: Path) -> Path:
    return data_root / "indices" / "recentPDB_low_homology_maxtoken1536.csv"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def build_monomer_index(
    *,
    source_index: Path,
    output_index: Path,
    min_n_token: int,
    max_n_token: int,
    limit: int,
    rebuild: bool,
    sample_seed: Optional[int] = None,
) -> Path:
    if output_index.exists() and not rebuild:
        return output_index

    import pandas as pd

    output_index.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_index)
    n0 = len(df)
    num_tokens = df["num_tokens"].astype(int)
    num_prot_chains = df["num_prot_chains"].astype(int)
    mask = (
        df["type"].astype(str).eq("chain")
        & df["mol_1_type"].astype(str).eq("prot")
        & num_prot_chains.eq(1)
        & num_tokens.ge(int(min_n_token))
        & num_tokens.le(int(max_n_token))
    )
    out = df.loc[mask].copy()
    if limit > 0:
        if sample_seed is None:
            out = (
                out.sort_values(["num_tokens", "pdb_id", "chain_1_id"])
                .iloc[:limit]
                .copy()
            )
        else:
            out = out.sample(
                n=min(int(limit), len(out)),
                random_state=int(sample_seed),
            ).copy()
    out = out.sort_values(["num_tokens", "pdb_id", "chain_1_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "Monomer index is empty after filtering. "
            f"source rows={n0}, min_n_token={min_n_token}, max_n_token={max_n_token}"
        )
    out.to_csv(output_index, index=False)
    logging.info(
        "Wrote monomer index %s: %d/%d rows",
        output_index,
        len(out),
        n0,
    )
    return output_index


def build_ppi_complex_index(
    *,
    source_index: Path,
    output_index: Path,
    min_n_token: int,
    max_n_token: int,
    limit: int,
    rebuild: bool,
    sample_seed: Optional[int] = None,
) -> Path:
    if output_index.exists() and not rebuild:
        return output_index

    import pandas as pd

    output_index.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_index)
    n0 = len(df)
    num_tokens = df["num_tokens"].astype(int)
    num_prot_chains = df["num_prot_chains"].astype(int)
    mask = (
        df["type"].astype(str).eq("interface")
        & df["mol_1_type"].astype(str).eq("prot")
        & df["mol_2_type"].astype(str).eq("prot")
        & num_prot_chains.ge(2)
        & num_tokens.ge(int(min_n_token))
    )
    if int(max_n_token) > 0:
        mask = mask & num_tokens.le(int(max_n_token))
    out = df.loc[mask].copy()
    if limit > 0:
        if sample_seed is None:
            out = (
                out.sort_values(["num_tokens", "pdb_id", "chain_1_id", "chain_2_id"])
                .iloc[:limit]
                .copy()
            )
        else:
            out = out.sample(
                n=min(int(limit), len(out)),
                random_state=int(sample_seed),
            ).copy()
    out = out.sort_values(
        ["num_tokens", "pdb_id", "chain_1_id", "chain_2_id"]
    ).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "Protein-protein complex index is empty after filtering. "
            f"source rows={n0}, min_n_token={min_n_token}, max_n_token={max_n_token}"
        )
    out.to_csv(output_index, index=False)
    logging.info(
        "Wrote protein-protein complex index %s: %d/%d rows",
        output_index,
        len(out),
        n0,
    )
    return output_index


def build_source_components(
    args: argparse.Namespace,
    filtered_index: Path,
    *,
    source_name: str,
    binder_selector_fn,
    max_binder_fraction: float,
):
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import (
        DesignSourceDataset,
        ProtenixComplexProvider,
        TrainerComponents,
    )
    from protenix.data.pipeline.dataset import BaseSingleDataset

    # Import from Protenix config helpers to get MSA/template featurizers with
    # the same settings as Protenix training, but with paths rooted at --data-root.
    from configs.configs_base import configs as protenix_base_configs
    from configs.configs_data import data_configs
    from protenix.config.config import parse_configs
    from protenix.config.extend_types import ListValue
    from protenix.data.pipeline.dataset import (
        get_msa_featurizer,
        get_template_featurizer,
    )

    protenix_cfg_dict = deepcopy(protenix_base_configs)
    protenix_cfg_dict["data"] = deepcopy(data_configs)
    dcfg = protenix_cfg_dict["data"]
    data_root = str(Path(args.data_root).resolve())

    dcfg["monomer_pretrain"] = {
        "base_info": {
            "mmcif_dir": os.path.join(data_root, "mmcif"),
            "bioassembly_dict_dir": os.path.join(data_root, "mmcif_bioassembly"),
            "indices_fpath": str(filtered_index),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "min_n_token": -1,
            "use_reference_chains_only": True,
            "exclusion": {},
        },
        "cropping_configs": {
            "method_weights": ListValue([1.0, 0.0, 0.0]),
            "crop_size": 0,
        },
        "sampler_configs": {"sampler_type": "uniform"},
        "lig_atom_rename": False,
        "shuffle_mols": False,
        "shuffle_sym_ids": False,
        "constraint": {"enable": False},
    }
    dcfg["train_sets"] = ListValue(["monomer_pretrain"])
    # Protenix ListValue cannot represent an empty list because it infers dtype
    # from value[0]. The trainer below never consumes Protenix test sets, so keep
    # this non-empty placeholder to satisfy config parsing.
    dcfg["test_sets"] = ListValue(["monomer_pretrain"])
    dcfg["msa"]["enable_prot_msa"] = bool(args.use_msa)
    dcfg["msa"]["enable_rna_msa"] = False
    dcfg["template"]["enable_prot_template"] = bool(args.use_template)

    protenix_configs = parse_configs(
        protenix_cfg_dict,
        arg_str="",
        fill_required_with_null=True,
    )
    dataset_name = "monomer_pretrain"
    dataset_param = {
        "name": dataset_name,
        **protenix_configs.data[dataset_name].base_info,
        "cropping_configs": protenix_configs.data[dataset_name].cropping_configs,
        "msa_featurizer": get_msa_featurizer(protenix_configs, dataset_name, "train"),
        "template_featurizer": get_template_featurizer(
            protenix_configs, dataset_name, "train"
        ),
        "lig_atom_rename": False,
        "shuffle_mols": False,
        "shuffle_sym_ids": False,
        "constraint": {},
        "ref_pos_augment": bool(args.ref_pos_augment),
        "limits": int(args.dataset_limit),
    }
    base_dataset = BaseSingleDataset(**dataset_param)
    provider = ProtenixComplexProvider(
        base_dataset=base_dataset,
        binder_selector_fn=binder_selector_fn,
        expose_sample_indice=True,
    )

    src = DesignSourceDataset(
        provider=provider,
        source_name=source_name,
        crop_size=int(args.crop_size),
        max_binder_fraction=float(max_binder_fraction),
        hotspot_force_zero_prob=1.0,
        aa_mask_mode=args.aa_mask_mode,
        aa_mask_prob=float(args.aa_mask_prob),
        aa_mask_min_prob=float(args.aa_mask_min_prob),
        aa_mask_max_prob=float(args.aa_mask_max_prob),
        compute_sidechain=not args.disable_sidechain,
        # Keep the design/binder region backbone-only for every backbone or AA
        # objective. This is independent of whether the side-chain module is
        # enabled: otherwise diffusion_internal a_token can see noisy GT
        # side-chain coordinates when the AA head is trained.
        backbone_only_binder=True,
        # Passed EXPLICITLY (it also defaults True) so the train/inference
        # contract is visible at the training entry rather than buried in a
        # dataclass default. Without it the binder keeps its native side-chain
        # atom rows, whose names/elements/count decode the residue identity
        # straight into q -> a_token -> AA head.
        inference_safe_binder=not args.allow_binder_sidechain_leakage,
        # The strict rebuild discards the provider's featurization, so this has to
        # be forwarded here or --no-ref-pos-augment (which the eval loader sets)
        # would silently stop reaching the model.
        ref_pos_augment=bool(args.ref_pos_augment),
        max_crop_retries=int(args.max_crop_retries),
        seed=int(args.seed),
    )
    multi = CurriculumMultiDataset(
        datasets=[src],
        source_names=[source_name],
        per_item_weights=[[1.0] * len(src)],
    )
    schedule = CurriculumSchedule(
        stage1={source_name: 1.0},
        stage2={source_name: 1.0},
        stage1_end_step=0,
        stage2_start_step=0,
    )
    components = TrainerComponents(
        train_dataset=multi,
        schedule=schedule,
        train_samples_per_epoch=int(args.train_samples_per_epoch),
    )
    return components, len(src)


def build_components(args: argparse.Namespace, filtered_index: Path):
    from pxdesign_train.runner import select_protenix_chain_1

    return build_source_components(
        args,
        filtered_index,
        source_name="protenix_monomer",
        binder_selector_fn=select_protenix_chain_1(),
        max_binder_fraction=1.0,
    )


def build_pinder_source_components(args: argparse.Namespace, manifest: Path):
    """Build a design source from selected PINDER holo dimers."""
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import (
        DesignSourceDataset,
        PinderPdbProvider,
        TrainerComponents,
    )

    provider = PinderPdbProvider(
        manifest_path=manifest,
        pinder_root=Path(args.pinder_root),
        cif_cache_dir=Path(args.pinder_cif_cache),
        archive_path=Path(args.pinder_archive),
        split="train",
        limit=int(args.complex_limit_index),
    )
    src = DesignSourceDataset(
        provider=provider,
        source_name="pinder_ppi_complex",
        crop_size=int(args.crop_size),
        max_binder_fraction=float(args.complex_max_binder_fraction),
        hotspot_force_zero_prob=1.0,
        aa_mask_mode=args.aa_mask_mode,
        aa_mask_prob=float(args.aa_mask_prob),
        aa_mask_min_prob=float(args.aa_mask_min_prob),
        aa_mask_max_prob=float(args.aa_mask_max_prob),
        compute_sidechain=not args.disable_sidechain,
        backbone_only_binder=True,
        # See the monomer builder above: explicit train/inference contract.
        inference_safe_binder=not args.allow_binder_sidechain_leakage,
        ref_pos_augment=bool(args.ref_pos_augment),
        max_crop_retries=int(args.max_crop_retries),
        seed=int(args.seed),
    )
    multi = CurriculumMultiDataset(
        datasets=[src],
        source_names=["pinder_ppi_complex"],
        per_item_weights=[[1.0] * len(src)],
    )
    schedule = CurriculumSchedule(
        stage1={"pinder_ppi_complex": 1.0},
        stage2={"pinder_ppi_complex": 1.0},
        stage1_end_step=0,
        stage2_start_step=0,
    )
    return TrainerComponents(
        train_dataset=multi,
        schedule=schedule,
        train_samples_per_epoch=int(args.train_samples_per_epoch),
    ), len(src)


def build_mixed_components(
    args: argparse.Namespace,
    monomer_index: Path,
    protenix_complex_index: Optional[Path],
    pinder_manifest: Optional[Path],
):
    from pxdesign_train.data import CurriculumMultiDataset, CurriculumSchedule
    from pxdesign_train.runner import (
        TrainerComponents,
        select_protenix_chain_1,
        select_protenix_chain_2,
    )

    mono_components, n_mono = build_source_components(
        args,
        monomer_index,
        source_name="protenix_monomer",
        binder_selector_fn=select_protenix_chain_1(),
        max_binder_fraction=1.0,
    )
    datasets = [mono_components.train_dataset.datasets[0]]
    source_names = ["protenix_monomer"]
    per_item_weights = [[1.0] * len(datasets[0])]
    source_counts = {
        "monomer": n_mono,
        "protenix_complex": 0,
        "pinder_complex": 0,
    }

    if args.complex_provider in {"protenix", "both"}:
        if protenix_complex_index is None:
            raise ValueError("Protenix complex index is required for this provider")
        complex_components, n_complex = build_source_components(
            args,
            protenix_complex_index,
            source_name="protenix_ppi_complex",
            binder_selector_fn=select_protenix_chain_2(),
            max_binder_fraction=float(args.complex_max_binder_fraction),
        )
        ds = complex_components.train_dataset.datasets[0]
        datasets.append(ds)
        source_names.append("protenix_ppi_complex")
        per_item_weights.append([1.0] * len(ds))
        source_counts["protenix_complex"] = n_complex

    if args.complex_provider in {"pinder", "both"}:
        if pinder_manifest is None:
            raise ValueError("PINDER manifest is required for this provider")
        complex_components, n_complex = build_pinder_source_components(
            args, pinder_manifest
        )
        ds = complex_components.train_dataset.datasets[0]
        datasets.append(ds)
        source_names.append("pinder_ppi_complex")
        per_item_weights.append([1.0] * len(ds))
        source_counts["pinder_complex"] = n_complex

    start_monomer = float(args.stage2_start_monomer_frac)
    end_monomer = float(args.stage2_end_monomer_frac)
    start_complex = 1.0 - start_monomer
    end_complex = 1.0 - end_monomer
    if args.complex_provider == "both":
        pinder_share = float(args.pinder_complex_frac)
        complex_stage1 = {
            "protenix_ppi_complex": start_complex * (1.0 - pinder_share),
            "pinder_ppi_complex": start_complex * pinder_share,
        }
        complex_stage2 = {
            "protenix_ppi_complex": end_complex * (1.0 - pinder_share),
            "pinder_ppi_complex": end_complex * pinder_share,
        }
    else:
        complex_source_name = (
            "pinder_ppi_complex"
            if args.complex_provider == "pinder"
            else "protenix_ppi_complex"
        )
        complex_stage1 = {complex_source_name: start_complex}
        complex_stage2 = {complex_source_name: end_complex}

    multi = CurriculumMultiDataset(
        datasets=datasets,
        source_names=source_names,
        per_item_weights=per_item_weights,
    )
    schedule = CurriculumSchedule(
        stage1={"protenix_monomer": start_monomer, **complex_stage1},
        stage2={"protenix_monomer": end_monomer, **complex_stage2},
        stage1_end_step=int(args.curriculum_stage1_end_step),
        stage2_start_step=int(args.curriculum_stage2_start_step),
        sources=source_names,
    )
    components = TrainerComponents(
        train_dataset=multi,
        schedule=schedule,
        train_samples_per_epoch=int(args.train_samples_per_epoch),
    )
    source_counts["complex"] = (
        source_counts["protenix_complex"] + source_counts["pinder_complex"]
    )
    return components, source_counts


def build_eval_dataloader(args: argparse.Namespace, output_dir: Path):
    if int(args.eval_interval) <= 0 or int(args.eval_samples) <= 0:
        return None, 0, None

    from torch.utils.data import DataLoader, Subset

    from pxdesign_train.runner.trainer import _identity_collate

    data_root = Path(args.data_root).resolve()
    source_index = (
        Path(args.eval_source_index).resolve()
        if args.eval_source_index
        else _recent_index_path(data_root)
    )
    filtered_index = (
        Path(args.eval_filtered_index).resolve()
        if args.eval_filtered_index
        else output_dir / "cache" / "recentPDB_monomer_validation_index.csv.gz"
    )
    build_monomer_index(
        source_index=source_index,
        output_index=filtered_index,
        min_n_token=int(args.min_n_token),
        max_n_token=int(args.max_n_token),
        limit=int(args.eval_samples),
        rebuild=bool(args.rebuild_eval_index),
        sample_seed=int(args.eval_seed),
    )

    eval_args = argparse.Namespace(**vars(args))
    eval_args.dataset_limit = int(args.eval_samples)
    eval_args.train_samples_per_epoch = max(1, int(args.eval_samples))
    eval_args.ref_pos_augment = False
    eval_args.data_mode = "monomer"
    # Pin validation to FULL masking regardless of what training uses. eval_args
    # is a whole-namespace copy, so a training stage that switches to partial
    # masking would otherwise switch validation too -- and partial masking is an
    # easier task (neighbouring identities are visible), so val_aa_acc would rise
    # for a reason that has nothing to do with the model. Holding this fixed keeps
    # the metric comparable to every historical run, including the 0.1310 that
    # every stage has reported so far.
    eval_args.aa_mask_mode = "all"
    eval_args.aa_mask_prob = 1.0
    eval_args.aa_mask_min_prob = 0.0
    eval_args.aa_mask_max_prob = 1.0
    eval_components, n_items = build_components(eval_args, filtered_index)
    eval_dataset = eval_components.train_dataset
    if int(args.eval_samples) > 0:
        eval_dataset = Subset(
            eval_dataset, range(min(int(args.eval_samples), len(eval_dataset)))
        )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.eval_num_workers),
        collate_fn=_identity_collate,
    )
    return eval_loader, len(eval_dataset), filtered_index


def build_configs(args: argparse.Namespace, device):
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import (
        apply_sidechain_ablation_arm,
        training_configs,
    )

    configs = parse_configs(training_configs, arg_str="")
    configs.seed = int(args.seed)
    configs.dtype = args.dtype
    configs.load_strict = False

    configs.training.crop_size = int(args.crop_size)
    configs.training.max_steps = int(args.max_steps)
    configs.training.lr = float(args.lr)
    configs.training.warmup_steps = int(args.warmup_steps)
    configs.training.log_interval = int(args.log_interval)
    configs.training.eval_interval = int(args.eval_interval)
    configs.training.checkpoint_interval = int(args.checkpoint_interval)
    configs.training.ema_decay = float(args.ema_decay)
    configs.training.num_workers = int(args.num_workers)
    configs.training.iters_to_accumulate = int(args.iters_to_accumulate)
    configs.training.grad_clip_norm = float(args.grad_clip_norm)
    configs.training.resume_lr = args.resume_lr
    configs.training.trainable_param_keywords = []

    configs.residue_type.mask_mode = args.aa_mask_mode
    configs.residue_type.mask_prob = float(args.aa_mask_prob)
    configs.residue_type.mask_min_prob = float(args.aa_mask_min_prob)
    configs.residue_type.mask_max_prob = float(args.aa_mask_max_prob)
    configs.residue_type.input_source = args.aa_input_source
    configs.residue_type.trunk_grad_scale = float(args.trunk_grad_scale)

    configs.loss.align_before_mse = bool(device.type == "cuda")
    if args.disable_aa_loss:
        configs.loss.weight_aa = 0.0
    configs.enable_sidechain = not args.disable_sidechain
    configs.enable_coevolution = bool(args.enable_coevolution)
    if args.disable_sidechain:
        configs.loss.weight_sc_local = 0.0
        configs.loss.weight_sc_phys = 0.0
        configs.loss.weight_sc_global = 0.0
    else:
        configs.sidechain.predicted_frame = bool(args.predicted_frame)
        configs.sidechain.per_sigma = bool(args.per_sigma)
        configs.sidechain.refinement_sigma = float(args.refinement_sigma)
        configs.sidechain.template_provider = args.template_provider
        configs.sidechain.template_init = not args.disable_template_init
        configs.sidechain.trunk_grad_scale = float(args.sc_trunk_grad_scale)
        if args.sc_frame_aware_head is not None:
            configs.sidechain.frame_aware_head = bool(args.sc_frame_aware_head)
        if args.sc_centre_coord_input is not None:
            configs.sidechain.centre_coord_input = bool(args.sc_centre_coord_input)
        if args.sc_template_residual is not None:
            configs.sidechain.template_residual = bool(args.sc_template_residual)
        if args.sc_mismatch_loss is not None:
            configs.sidechain.mismatch_loss = str(args.sc_mismatch_loss)
        if args.sc_pack_loss is not None:
            configs.sidechain.pack_loss = float(args.sc_pack_loss)
        if args.sc_edm is not None:
            configs.sidechain.edm = bool(args.sc_edm)
        if args.sc_edm_sigma_data is not None:
            configs.sidechain.edm_sigma_data = float(args.sc_edm_sigma_data)
        if args.sc_edm_p_mean is not None:
            configs.sidechain.edm_p_mean = float(args.sc_edm_p_mean)
        if args.sc_edm_p_std is not None:
            configs.sidechain.edm_p_std = float(args.sc_edm_p_std)
        if args.sc_edm_sigma_min is not None:
            configs.sidechain.edm_sigma_min = float(args.sc_edm_sigma_min)
        if args.sc_edm_sigma_max is not None:
            configs.sidechain.edm_sigma_max = float(args.sc_edm_sigma_max)
        if args.sc_edm_infer_steps is not None:
            configs.sidechain.edm_infer_steps = int(args.sc_edm_infer_steps)
    apply_sidechain_ablation_arm(configs, args.sc_ablation_arm)

    if args.training_stage == "backbone_only":
        configs.loss.weight_aa = 0.0
        configs.enable_sidechain = False
        configs.enable_coevolution = False
        configs.loss.weight_sc_local = 0.0
        configs.loss.weight_sc_phys = 0.0
        configs.loss.weight_sc_global = 0.0
        configs.loss.weight_bb_post = 0.0
        configs.loss.weight_aa_post = 0.0
    elif args.training_stage == "aa_head_warmup":
        # Train a structure-conditioned AA predictor on top of a frozen
        # stage-1 backbone.  The coordinate forward pass is still required to
        # produce diffusion_internal token features, but only the residue-type
        # head receives optimizer updates.
        configs.loss.weight_mse = 0.0
        configs.loss.weight_lddt = 0.0
        configs.loss.weight_disto = 0.0
        configs.loss.weight_aa = 1.0
        configs.loss.weight_bb_post = 0.0
        configs.loss.weight_aa_post = 0.0
        configs.enable_sidechain = False
        configs.enable_coevolution = False
        configs.loss.weight_sc_local = 0.0
        configs.loss.weight_sc_phys = 0.0
        configs.loss.weight_sc_global = 0.0
        configs.residue_type.trunk_grad_scale = 0.0
        configs.training.trainable_param_keywords = [
            "design_residue_type_head.",
        ]
        configs.training.ema_decay = 0.0
    elif args.training_stage == "sidechain_warmup":
        configs.loss.weight_mse = 0.0
        configs.loss.weight_lddt = 0.0
        configs.loss.weight_disto = 0.0
        configs.loss.weight_aa = 0.0
        configs.loss.weight_bb_post = 0.0
        configs.loss.weight_aa_post = 0.0
        configs.enable_sidechain = True
        configs.enable_coevolution = False
        configs.sidechain.predicted_frame = False
        configs.sidechain.per_sigma = False
        configs.sidechain.template_init = True
        configs.sidechain.template_provider = args.template_provider
        configs.sidechain.trunk_grad_scale = 0.0
        configs.sidechain.force_gt_type_logits = True
        # 14-slot atom axis, stated explicitly rather than inherited. This is the
        # ONE Stage II decision that cannot be deferred: it fixes the input layout
        # every Stage III arm will warm-start from. Adds no parameters, costs a
        # slightly wider intra-residue attention (14x14 instead of 10x10), and the
        # four backbone slots are keys only -- never decoded, never in the loss.
        # Leaving it off here would force a second Stage II run for every q arm.
        configs.sidechain.bb_context = True
        # B->S wiring, asymmetric on purpose:
        #   a_bs_concat=True  -- B's a_token summarises a residue from its four
        #       BACKBONE atoms; S_phi's pooled feature summarises the same residue
        #       from all fourteen slots. They are not the same quantity, so the
        #       network should learn how to combine them rather than have them
        #       summed. (The asymmetry only became real once the binder's native
        #       side chains stopped reaching a_token.) Same interface shape as
        #       Stage III, which is what makes the warmed-up weights transferable.
        #   q_bs=False -- it hands S_phi the Backbone Module's own per-atom
        #       features, and here the backbone is FROZEN. S_phi would fit one
        #       frozen state's q, which moves as soon as Stage III updates the
        #       backbone. The geometry it needs is already in the frames and the
        #       ideal template.
        configs.sidechain.a_bs_concat = True
        configs.sidechain.q_bs = False
        # No S->B feedback: there is no refinement pass to inject into, and
        # trunk_grad_scale=0 already makes h_res read-only.
        configs.sidechain.a_direct = False
        configs.sidechain.a_direct_pre = False
        configs.sidechain.q_direct = False
        # `design_condition_embedder` is a TOP-LEVEL module of ProtenixDesign, so
        # filtering the warm-start to "diffusion_module." alone dropped it: it
        # would stay randomly initialised AND frozen (it is not in
        # trainable_param_keywords). It produces s_inputs and z, which feed the
        # diffusion conditioning, the atom pair bias and the token transformer —
        # so h_res in this stage would be a different quantity than the h_res of
        # Stage I / III, and any w_res learnt here could not transfer.
        # `design_residue_type_head` is carried for the same reason, and dropping it
        # is what made Stage III start from a chance-level AA head. Stage II does not
        # train it (weight_aa=0, and it is not in trainable_param_keywords) and does
        # not depend on it (force_gt_type_logits feeds S_phi the true types), so
        # loading it only preserves what the donor had -- but NOT loading it meant
        # every Stage II checkpoint saved a fresh random AA head, and Stage III
        # inherits the whole file. Measured on a Stage III smoke off such a
        # checkpoint: aa_ce 2.991 vs ln(20)=2.996, aa_acc ~5%, i.e. chance.
        #
        # Grafting a trained AA head in afterwards is NOT equivalent: the head is
        # coupled to the trunk it was trained against. Overlaying the joint run's
        # head onto this lineage's backbone (which differ by ~8% median relative
        # weight difference) drove aa_ce to ~50 at unchanged chance accuracy --
        # confidently wrong. Carrying the head along the lineage keeps it consistent
        # with the backbone it was trained with; `--load-aa-head-from` exists for
        # deliberate grafts, not as a substitute for this.
        configs.training.checkpoint_include_prefixes = [
            "diffusion_module.",
            "design_condition_embedder.",
            "design_residue_type_head.",
        ]
        # Stable Stage II-A parameterization: S_phi reads residue-local template
        # coordinates and predicts a zero-initialized local correction. The known
        # GT frame performs the local->global map used by the coordinate loss.
        configs.sidechain.frame_aware_head = (
            True if args.sc_frame_aware_head is None else bool(args.sc_frame_aware_head)
        )
        configs.sidechain.centre_coord_input = (
            True if args.sc_centre_coord_input is None else bool(args.sc_centre_coord_input)
        )
        configs.sidechain.template_residual = (
            True if args.sc_template_residual is None else bool(args.sc_template_residual)
        )
        if configs.sidechain.template_residual and not configs.sidechain.frame_aware_head:
            raise ValueError("--sc-template-residual requires --sc-frame-aware-head")
        configs.training.trainable_param_keywords = ["sidechain_module."]
        configs.training.ema_decay = 0.0
    elif args.training_stage == "joint":
        # Jointly optimize backbone denoising and residue-type prediction. Keep
        # side-chain generation off; this stage is BB + AA head only.
        configs.loss.weight_aa = 1.0
        configs.enable_sidechain = False
        configs.enable_coevolution = False
        configs.loss.weight_sc_local = 0.0
        configs.loss.weight_sc_phys = 0.0
        configs.loss.weight_sc_global = 0.0
        configs.loss.weight_bb_post = 0.0
        configs.loss.weight_aa_post = 0.0
    elif args.training_stage == "aa_head_on_stage2":
        # Train ONLY the AA head, on a frozen Stage II model, IN THE STAGE III
        # CONFIGURATION. This exists because an AA head is not portable across
        # configurations: `aa_head_warmup` trains one with enable_sidechain and
        # enable_coevolution OFF, and grafting that head into Stage III (both ON)
        # produced confidently-wrong predictions -- aa_ce 54-76 at chance accuracy,
        # measured with BOTH a joint donor and an aa_head_warmup donor whose
        # backbone was byte-identical (636/636) to the Stage II trunk. Identical
        # weights were not enough; the head has to be fitted to the feature path it
        # will be used in.
        #
        # So: same machinery as Stage III (side chain on, refinement pass on,
        # predicted frames, per-sigma), but the backbone and S_phi are frozen and
        # only design_residue_type_head learns. The checkpoint this writes therefore
        # carries all three components -- trained backbone, trained S_phi, and an AA
        # head fitted to them -- which is the single-file Stage III start that no
        # existing checkpoint provided.
        configs.loss.weight_mse = 0.0
        configs.loss.weight_lddt = 0.0
        configs.loss.weight_disto = 0.0
        configs.loss.weight_aa = 1.0
        # The refinement pass still runs (it is part of the feature path we are
        # fitting to) but its losses are zeroed: the modules they would train are
        # frozen, so they contribute no gradient and only noise to the logs.
        configs.loss.weight_bb_post = 0.0
        configs.loss.weight_aa_post = 0.0
        configs.loss.weight_sc_local = 0.0
        configs.loss.weight_sc_phys = 0.0
        configs.loss.weight_sc_global = 0.0
        configs.enable_sidechain = True
        configs.enable_coevolution = True
        # Fit the head on Stage III's teacher-forced geometry path: S_phi reads
        # GT backbone frames while its learned h/a/q and type-logit features still
        # come from B_pre. Stage IV switches the geometry to predicted frames.
        configs.sidechain.predicted_frame = False
        configs.sidechain.per_sigma = True
        configs.sidechain.template_init = True
        configs.sidechain.template_provider = args.template_provider
        configs.sidechain.force_gt_type_logits = False
        # Frozen trunk: no gradient path from the AA loss into the backbone.
        configs.residue_type.trunk_grad_scale = 0.0
        # Load the Stage II checkpoint whole -- backbone, conditioning and S_phi.
        configs.training.checkpoint_include_prefixes = []
        configs.training.trainable_param_keywords = ["design_residue_type_head."]
        configs.training.ema_decay = 0.0
        adopt_sidechain_arch_from_checkpoint(configs, args)
    elif args.training_stage in ("coevolution", "predicted_mask"):
        # Paper Stage III (coevolution) and Stage IV (predicted_mask). Neither had
        # an entry here: `joint` above is BB + AA head with the side chain OFF, so
        # the co-evolution machinery that Stage III is *about* -- S_phi, the
        # feedback channel, the refinement pass, L_bb^post / L_aa^post -- was
        # reachable only from scripts/examples/finetune_mini.py.
        configs.loss.weight_aa = 1.0
        configs.enable_sidechain = True
        configs.enable_coevolution = True
        # Curriculum boundary confirmed with Fang:
        #   Stage III/co-evolution: S_phi gets GT backbone coordinates/frames,
        #     but B_pre still supplies its learned h/a/q + type-logit features.
        #   Stage IV/predicted_mask: geometry and atom set are both predicted,
        #     matching inference-time inputs.
        # This switch changes only data routing, not parameter shapes/layout, and
        # B_post still refines B_pre's predicted x_hat_0 in both stages.
        configs.sidechain.predicted_frame = (
            args.training_stage == "predicted_mask"
        )
        configs.sidechain.per_sigma = True
        configs.sidechain.template_init = True
        configs.sidechain.template_provider = args.template_provider
        configs.sidechain.trunk_grad_scale = float(args.sc_trunk_grad_scale)
        configs.sidechain.force_gt_type_logits = False
        # Warm-start from the Stage II checkpoint: backbone, conditioning and the
        # side-chain module all carry over, so no prefix filter.
        configs.training.checkpoint_include_prefixes = []
        configs.training.trainable_param_keywords = []
        if args.training_stage == "predicted_mask":
            # Stage IV: instantiate the atom set from the PREDICTED identity, and
            # supervise coordinates only where that identity is right. Turning
            # predicted_mask on is also what makes L_aa^post safe to supervise:
            # under GT-type teacher forcing the atom composition carries the
            # answer, so post_aa is deliberately not produced (model.py gates it).
            configs.sidechain.predicted_mask = True
            configs.sidechain.route_by_type = True
        else:
            configs.sidechain.predicted_mask = False
            configs.sidechain.route_by_type = False
        # Stage III/IV continue a Stage II side-chain module, so S_phi's input
        # LAYOUT has to be whatever that checkpoint was trained with -- not this
        # stage's defaults. Adopt it from the checkpoint's own record.
        adopt_sidechain_arch_from_checkpoint(configs, args)
    return configs


# CLI flags that pin an arch key directly. If the user passed one, it wins over
# whatever the checkpoint recorded -- adopting silently over an explicit flag
# would be worse than the mismatch we are fixing.
_ARCH_KEY_TO_CLI_ARG = {
    "centre_coord_input": "sc_centre_coord_input",
    "frame_aware_head": "sc_frame_aware_head",
    "template_residual": "sc_template_residual",
    "edm": "sc_edm",
}


def adopt_sidechain_arch_from_checkpoint(configs, args: argparse.Namespace) -> dict:
    """Make S_phi's layout match the warm-start checkpoint's recorded `sidechain_arch`.

    WHY THIS IS NEEDED. `sidechain_warmup` pins its own layout (bb_context,
    a_bs_concat, q_bs) and threads centre_coord_input/frame_aware_head from the
    CLI. The coevolution bundle pinned NONE of them, so it fell back to the
    config defaults -- which do not match what Stage II trains with. Measured
    against a real Stage II checkpoint (fixed_global_decay_from_50k/step65000):

        centre_coord_input  checkpoint=True  stage-III default=False  -> FATAL
        q_bs                checkpoint=False stage-III default=True   -> silent

    The first is a layout key, so `_check_sidechain_arch` aborts the run: Stage III
    could not warm-start from its own Stage II checkpoint at all (job 101060 died
    in 38s on exactly this). The second is an additive key, so it only logs -- it
    would have switched on a fusion channel that has no trained weights, a
    curriculum change nobody asked for.

    Hardcoding Stage II's values here would fix today's pair and rot the moment an
    arm trains a different layout, and this project runs many arms. The checkpoint
    already records the layout for precisely this purpose, so honour it.

    Returns the keys actually changed, for logging by the caller.
    """
    ckpt_path = getattr(args, "load_checkpoint", None)
    if not ckpt_path:
        return {}
    try:
        import torch

        # map_location="meta" so no tensor storage is materialised; we only want
        # the small metadata dict that sits alongside the weights.
        saved = torch.load(str(ckpt_path), map_location="meta", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — a bad path is the loader's error to raise
        logging.warning(
            "could not read %s to adopt its side-chain layout (%s: %s); leaving the "
            "configured layout as-is. If it disagrees with the checkpoint, "
            "_check_sidechain_arch will refuse the load.",
            ckpt_path, type(exc).__name__, exc,
        )
        return {}

    arch = (saved or {}).get("sidechain_arch") or {}
    if not arch:
        logging.warning(
            "%s records no sidechain_arch, so its S_phi layout is unknown and cannot "
            "be adopted. Pass the --sc-* flags that produced it, or expect "
            "_check_sidechain_arch to refuse the load.", ckpt_path,
        )
        return {}

    adopted, kept = {}, {}
    for key, value in arch.items():
        cli_arg = _ARCH_KEY_TO_CLI_ARG.get(key)
        if cli_arg is not None and getattr(args, cli_arg, None) is not None:
            if bool(getattr(configs.sidechain, key, False)) != bool(value):
                kept[key] = (bool(value), bool(getattr(configs.sidechain, key, False)))
            continue
        if bool(getattr(configs.sidechain, key, False)) != bool(value):
            adopted[key] = bool(value)
            setattr(configs.sidechain, key, bool(value))

    if adopted:
        logging.info(
            "Adopted S_phi layout from %s: %s (Stage III continues that module, so "
            "its input layout is not this stage's to choose)",
            ckpt_path, ", ".join(f"{k}={v}" for k, v in sorted(adopted.items())),
        )
    if kept:
        logging.warning(
            "Explicit CLI flag(s) disagree with %s and were KEPT: %s. The load will "
            "be refused if any of these is a layout key.",
            ckpt_path,
            ", ".join(f"{k}: checkpoint={a} using={b}" for k, (a, b) in sorted(kept.items())),
        )
    return adopted


# The S->B feedback channels, and the top-level module each one instantiates.
# Presence of the module in a checkpoint's state dict is the evidence that the
# channel was on when those weights were trained.
_FEEDBACK_CHANNEL_MODULES = {
    "a_direct_pre": "a_token_fusion_pre",
    "a_direct": "a_token_fusion",
    "q_direct": "q_atom_fusion",
}


def adopt_feedback_channels_from_checkpoint(configs, ckpt_path) -> dict:
    """Switch the S->B feedback channels to match what the checkpoint carries.

    `sidechain_arch` records S_phi's own INPUT layout, but not the S->B feedback
    channels -- those come from `--sc-ablation-arm`, and the arm is not written
    into the checkpoint. So a consumer that picks a different arm than the run
    that produced the weights builds a different model and never learns it did:
    the extra tensors land in `unexpected` (which nothing checks) and the missing
    channels are simply absent from the forward pass.

    Measured: `eval_aa_head_strict_backbone.py` defaults to arm "no" -- correct
    for the `joint` stage, where the side chain is off entirely -- and scoring an
    `aa_head_on_stage2` checkpoint under it dropped `a_token_fusion_pre` and
    `q_atom_fusion` (16 tensors, "unexpected=16"). That is precisely the
    fitted-to-a-different-feature-path error `aa_head_on_stage2` exists to avoid.

    The state dict answers the question directly, so ask it rather than the arm.

    CALL THIS ONLY WHEN REPRODUCING A FITTED MODEL, i.e. from an evaluator scoring
    weights that already exist. It is deliberately NOT called from `build_configs`,
    because on the TRAINING side the checkpoint is a donor, not a specification:
    Stage III/IV -- and `aa_head_on_stage2` itself -- warm-start from a Stage II
    checkpoint that has no fusion modules at all and are *supposed* to switch
    these channels on. They are zero-initialised, so enabling one is an exact
    no-op at step 0 and a normal curriculum move. Wiring this into the training
    path would silently delete the co-evolution channels those stages exist to
    train, which is a worse bug than the one it fixes.
    """
    try:
        import torch

        # map_location="meta": only the key names are needed, no storage.
        saved = torch.load(str(ckpt_path), map_location="meta", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "could not read %s to reconcile its S->B feedback channels (%s: %s); "
            "the configured channels are left as-is and may not match the weights.",
            ckpt_path, type(exc).__name__, exc,
        )
        return {}
    state = (saved or {}).get("model") or {}
    if not state:
        return {}
    prefixes = {k.removeprefix("module.").split(".", 1)[0] for k in state}
    adopted = {}
    for flag, module_name in _FEEDBACK_CHANNEL_MODULES.items():
        present = module_name in prefixes
        if bool(getattr(configs.sidechain, flag, False)) != present:
            setattr(configs.sidechain, flag, present)
            adopted[flag] = present
    if adopted:
        logging.info(
            "Adopted S->B feedback channels from %s's weights: %s (the ablation arm "
            "is not recorded in the checkpoint, so the module set is the evidence)",
            ckpt_path, ", ".join(f"{k}={v}" for k, v in sorted(adopted.items())),
        )
    return adopted


# Masking schedule for every stage that trains `design_residue_type_head`.
#
# WHY NOT "all". Until now every such stage used aa_mask_mode="all" -- every
# design position masked at once -- which pins aa_t at 1.0 and quietly collapses
# the masked-diffusion objective into its simplest special case: the MDLM
# importance weight 1/t becomes 1/1, and the head's sinusoidal time embedding
# adds the same constant vector on every example, i.e. a bias term. The
# "time_dependent" schedule has been registered in `DesignSelection` since the
# pipeline was written and had never been selected by any stage.
#
# The cost is not the dead machinery, it is what the head never sees. Under full
# masking it is only ever asked to predict every identity from backbone geometry
# alone, so it cannot learn to condition on already-decided neighbours -- which
# is where most of ProteinMPNN's ~33% recovery on predicted backbones comes
# from, against our ~13%. Inference has the same gap from the other side:
# `sequential` mode reveals progressively and feeds commitments back through
# `restype`, but a head trained only on fully-masked inputs has never seen a
# partial sequence.
#
# "time_dependent" draws t ~ U(0, 1) per example and masks each design position
# independently with probability t (aa_mask_min/max_prob default to 0.0/1.0, so
# prob == t). That is the standard MDLM/LLaDA schedule, and it answers the
# original objection to it: there is no hand-designed masking ratio to choose.
#
# Stages that do NOT train the head (backbone_only, sidechain_warmup) keep
# "none" -- they disable the AA loss, so the schedule is not theirs to set.
AA_MASK_MODE_TRAINING = "time_dependent"


def apply_training_stage_args(args: argparse.Namespace) -> None:
    if args.training_stage == "backbone_only":
        args.disable_sidechain = True
        args.disable_aa_loss = True
        args.aa_mask_mode = "none"
        args.enable_coevolution = False
    elif args.training_stage == "aa_head_warmup":
        args.disable_sidechain = True
        args.disable_aa_loss = False
        args.aa_mask_mode = AA_MASK_MODE_TRAINING
        args.enable_coevolution = False
        args.trunk_grad_scale = 0.0
    elif args.training_stage == "sidechain_warmup":
        args.disable_sidechain = False
        args.disable_aa_loss = True
        args.aa_mask_mode = "none"
        args.enable_coevolution = False
        args.predicted_frame = False
        args.per_sigma = False
    elif args.training_stage == "joint":
        args.disable_sidechain = True
        args.disable_aa_loss = False
        args.aa_mask_mode = AA_MASK_MODE_TRAINING
        args.enable_coevolution = False
    elif args.training_stage == "aa_head_on_stage2":
        args.disable_sidechain = False
        args.disable_aa_loss = False
        # Scoped comment lives on AA_MASK_MODE_TRAINING above. This stage is the
        # cheapest place to measure the change: it trains only the head against a
        # frozen backbone and S_phi (1.30M of 262.05M) and reports at step ~9000.
        args.aa_mask_mode = AA_MASK_MODE_TRAINING
        args.enable_coevolution = True
        # This head is fitted to the Stage III feature path, whose S_phi geometry
        # is teacher-forced from GT backbone frames.
        args.predicted_frame = False
        args.per_sigma = True
        args.trunk_grad_scale = 0.0
    elif args.training_stage in ("coevolution", "predicted_mask"):
        args.disable_sidechain = False
        args.disable_aa_loss = False
        args.aa_mask_mode = AA_MASK_MODE_TRAINING
        args.enable_coevolution = True
        # Stage III teacher-forces only S_phi's backbone geometry. Stage IV opens
        # that input to the predicted frame; learned B_pre features remain live in
        # both stages.
        args.predicted_frame = args.training_stage == "predicted_mask"
        args.per_sigma = True


# Dataset locations, resolved once so no default is written as a literal path.
#
# WHY. These defaults were absolute paths under /hai/scratch/yfsun, which made
# the repo unrunnable by anyone else without editing argparse defaults in place --
# and the same literals appear in ~40 places across scripts/, so "just edit it"
# meant editing it many times and missing some. One root variable now drives all
# of them, and every individual path stays independently overridable for layouts
# that do not nest the way this one does.
#
#   PROTEOAA_DATA_ROOT   parent of every dataset      (default /hai/scratch/yfsun)
#   PROTENIX_DATA_ROOT   Protenix training data       (default $PROTEOAA_DATA_ROOT/protenix_data)
#   PINDER_ROOT          PINDER release root          (default $PROTEOAA_DATA_ROOT/pinder/2024-02)
#
# See docs/datasets.md for what each one has to contain and where to get it.
_DATA_ROOT = os.environ.get("PROTEOAA_DATA_ROOT", "/hai/scratch/yfsun")
DEFAULT_PROTENIX_DATA_ROOT = os.environ.get(
    "PROTENIX_DATA_ROOT", os.path.join(_DATA_ROOT, "protenix_data")
)
DEFAULT_PINDER_ROOT = os.environ.get(
    "PINDER_ROOT", os.path.join(_DATA_ROOT, "pinder", "2024-02")
)
DEFAULT_PINDER_CIF_CACHE = os.environ.get(
    "PINDER_CIF_CACHE", os.path.join(DEFAULT_PINDER_ROOT, "cif_cache")
)
DEFAULT_PINDER_ARCHIVE = os.environ.get(
    "PINDER_ARCHIVE", os.path.join(DEFAULT_PINDER_ROOT, "raw", "pdbs.zip")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--training-stage",
        default="backbone_only",
        choices=[
            "backbone_only", "aa_head_warmup", "sidechain_warmup", "joint",
            "aa_head_on_stage2", "coevolution", "predicted_mask",
        ],
        help="Training objective bundle. backbone_only is the default pretraining "
             "stage; 'coevolution' is paper Stage III (both modules + the "
             "refinement pass) and 'predicted_mask' is Stage IV (atom sets "
             "instantiated from the predicted identity).",
    )
    p.add_argument(
        "--data-mode",
        default="monomer",
        choices=["monomer", "mixed_monomer_complex"],
        help="monomer keeps the original stage-1 data. mixed_monomer_complex adds protein-protein interfaces with a curriculum.",
    )
    p.add_argument("--data-root", default=DEFAULT_PROTENIX_DATA_ROOT)
    p.add_argument("--source-index", default="")
    p.add_argument("--output-dir", default=_default_output_dir())
    p.add_argument(
        "--load-checkpoint",
        default="",
        help="Resume from a saved training checkpoint. By default this restores model, optimizer, scheduler, and step.",
    )
    p.add_argument(
        "--warm-start-params-only",
        action="store_true",
        help="When used with --load-checkpoint, load only model parameters instead of exact training state.",
    )
    p.add_argument(
        "--load-aa-head-from",
        default="",
        help="Overlay the residue-type (AA) head from a SECOND checkpoint, applied "
             "after --load-checkpoint. No single checkpoint has all three trained "
             "components: a Stage II side-chain run carries the backbone and S_phi "
             "but re-initialised the AA head and froze it (chance-level: aa_ce ~2.99 "
             "vs ln(20)=2.996), while a `joint` run has a trained AA head but never "
             "builds S_phi. Stage III needs all three, so compose: --load-checkpoint "
             "<stage II> --load-aa-head-from <joint>. Only valid with "
             "--warm-start-params-only; overlaying onto a full resume would roll back "
             "the AA head this run already trained.",
    )
    p.add_argument("--filtered-index", default="")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--limit-index", type=int, default=0)
    p.add_argument("--complex-source-index", default="")
    p.add_argument("--complex-filtered-index", default="")
    p.add_argument(
        "--complex-provider",
        default="protenix",
        choices=["protenix", "pinder", "both"],
        help="Use Protenix complexes, a prepared PINDER manifest, or both complex sources.",
    )
    p.add_argument(
        "--pinder-manifest",
        default="",
        help="Prepared PINDER PPI manifest. Defaults to <pinder-root>/indices/pinder_ppi_complex.parquet.",
    )
    p.add_argument(
        "--pinder-root",
        default=DEFAULT_PINDER_ROOT,
        help="PINDER release root containing pdbs/.",
    )
    p.add_argument(
        "--pinder-cif-cache",
        default=DEFAULT_PINDER_CIF_CACHE,
        help="Persistent cache for lazy PINDER PDB-to-mmCIF conversion.",
    )
    p.add_argument(
        "--pinder-archive",
        default=DEFAULT_PINDER_ARCHIVE,
        help="Official structure archive used when a selected PDB has not been extracted.",
    )
    p.add_argument("--complex-limit-index", type=int, default=0)
    p.add_argument("--complex-max-n-token", type=int, default=1536)
    p.add_argument("--complex-max-binder-fraction", type=float, default=0.75)
    p.add_argument("--stage2-start-monomer-frac", type=float, default=0.90)
    p.add_argument("--stage2-end-monomer-frac", type=float, default=0.65)
    p.add_argument(
        "--pinder-complex-frac",
        type=float,
        default=0.5,
        help="When --complex-provider both, fraction of the complex curriculum mass assigned to PINDER.",
    )
    p.add_argument("--curriculum-stage1-end-step", type=int, default=0)
    p.add_argument("--curriculum-stage2-start-step", type=int, default=10000)
    p.add_argument("--dataset-limit", type=int, default=-1)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)

    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--train-samples-per-epoch", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--checkpoint-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Run validation every N optimizer steps. Use 0 to disable.",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=64,
        help="Number of strict monomer rows sampled from recent PDB for validation.",
    )
    p.add_argument("--eval-source-index", default="")
    p.add_argument("--eval-filtered-index", default="")
    p.add_argument("--rebuild-eval-index", action="store_true")
    p.add_argument("--eval-seed", type=int, default=1000003)
    p.add_argument("--eval-num-workers", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--iters-to-accumulate", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the monomer dataset and featurize one item, but do not construct/train the model.",
    )

    p.add_argument(
        "--aa-mask-mode",
        default="all",
        # Must match DesignSelection's valid_modes (data/featurizer.py):
        # {"all", "none", "fixed", "time_dependent"}. "partial" was never one of
        # them, so --aa-mask-mode partial passed argparse and then raised inside
        # the featurizer, after the job had been scheduled.
        choices=["all", "fixed", "none", "time_dependent"],
    )
    p.add_argument("--aa-mask-prob", type=float, default=1.0)
    p.add_argument("--aa-mask-min-prob", type=float, default=0.0)
    p.add_argument("--aa-mask-max-prob", type=float, default=1.0)
    p.add_argument(
        "--aa-input-source",
        default="diffusion_internal",
        choices=["s_inputs", "diffusion_internal"],
    )
    p.add_argument("--trunk-grad-scale", type=float, default=1.0)
    p.add_argument(
        "--disable-aa-loss",
        action="store_true",
        help="Disable the amino-acid prediction loss. Use with --aa-mask-mode none for backbone-only training.",
    )

    p.add_argument("--disable-sidechain", action="store_true")
    p.add_argument(
        "--allow-binder-sidechain-leakage",
        action="store_true",
        help="LEAKAGE ABLATION ONLY. Keep the binder's native side-chain atom rows "
             "in the model input. Their atom names/elements/count decode the residue "
             "identity into a_token, so the AA head can read the answer instead of "
             "predicting it (measured ~96%% vs ~6%% accuracy). Training runs must "
             "leave this off; it exists so the leaky arm stays reproducible.",
    )
    p.add_argument("--enable-coevolution", action="store_true")
    p.add_argument(
        "--predicted-frame", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--refinement-sigma", type=float, default=2.0,
        help=(
            "Fixed EDM conditioning/residual scale for B_post in both training "
            "and inference; refinement adds no noise (default: 2.0 A)."
        ),
    )
    p.add_argument("--per-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--disable-template-init", action="store_true")
    p.add_argument(
        "--resume-lr", type=float, default=None,
        help="On a FULL resume, override the restored base learning rate. "
             "Without this the scheduler restores the old rate and --lr is "
             "silently ignored.",
    )
    p.add_argument("--sc-trunk-grad-scale", type=float, default=1.0)
    p.add_argument(
        "--sc-edm", action=argparse.BooleanOptionalAction, default=None,
        help="EDM side-chain diffusion: sample sigma, precondition, and feed "
             "log(sigma) to the time embedding. Requires --sc-centre-coord-input "
             "and is incompatible with --sc-template-residual.",
    )
    p.add_argument("--sc-edm-sigma-data", type=float, default=None)
    p.add_argument("--sc-edm-p-mean", type=float, default=None)
    p.add_argument("--sc-edm-p-std", type=float, default=None)
    p.add_argument("--sc-edm-sigma-min", type=float, default=None)
    p.add_argument("--sc-edm-sigma-max", type=float, default=None)
    p.add_argument("--sc-edm-infer-steps", type=int, default=None)
    p.add_argument(
        "--sc-mismatch-loss", default=None,
        choices=["none", "clash", "legacy", "compat"],
        help="0722 L_compat arm. 'none' disables the physical term outright.",
    )
    p.add_argument(
        "--sc-pack-loss", type=float, default=None,
        help="Weight of the GENERAL steric term over every supervised side-chain "
             "atom (0 = off, the default). Separate from --sc-mismatch-loss, which "
             "is 0722's L_compat and is empty under teacher forcing.",
    )
    p.add_argument(
        "--sc-frame-aware-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Predict local offsets and map them through the active backbone frame.",
    )
    p.add_argument(
        "--sc-centre-coord-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recentre S_phi's per-atom coordinate embedding on the residue CA. "
             "Coordinates themselves are always global (single-frame contract); this "
             "only removes the absolute-position offset from the embedding.",
    )
    p.add_argument(
        "--sc-template-residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Predict a zero-initialized correction to the template/noisy local input.",
    )
    p.add_argument(
        "--sc-ablation-arm",
        default="default",
        # No `choices=` on purpose. SC_ABLATION_ARMS is the registry (21 arms) and
        # `apply_sidechain_ablation_arm` already rejects an unknown name with the
        # valid list in the message. The hardcoded list here had drifted to 7 and
        # was the ONLY thing blocking the other 15 -- `a-bs`, the arm that differs
        # from the Stage III default in nothing but the two S->B feedback channels
        # (a_direct_pre, q_direct), and therefore the only clean control for "does
        # the side chain help the backbone", failed at argparse with
        # "invalid choice: 'a-bs'". Importing the registry here is not an option:
        # parse_args runs before `_bootstrap_paths`, so `pxdesign_train` (and its
        # protenix import) is not yet on sys.path. Deferring to the one validator
        # that can see the registry keeps the CLI from drifting again.
        help="Side-chain feedback ablation arm; see SC_ABLATION_ARMS in "
             "pxdesign_train/configs/configs_train.py for the full list.",
    )

    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--use-template", action="store_true")
    p.add_argument(
        "--ref-pos-augment", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    return p.parse_args()


def fill_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    """Back-fill every option this parser defines that `args` does not carry.

    WHY THIS EXISTS. `build_configs` and `build_components` are shared with the
    evaluation scripts, but each of those defines its OWN argparse parser holding
    the subset of flags it needs. So every time a new option is added here and
    read in one of those two functions, every eval script breaks with
    `AttributeError: 'Namespace' object has no attribute '<new_flag>'` -- at
    import-adjacent setup time, i.e. after the job has been queued, scheduled and
    has loaded its dataset index. Measured: five eval scripts died this way on
    `allow_binder_sidechain_leakage`, and the next run died on `resume_lr`,
    one attribute at a time.

    Chasing them one at a time is the wrong fix; the eval parsers cannot be
    expected to track this one. Filling the gap from THIS parser's own defaults
    makes the contract "the training parser's defaults apply unless the caller
    overrides them", which is what the eval scripts already assume.

    Only ABSENT attributes are filled -- anything the caller set, including a
    deliberate `False`, is left exactly as it is.
    """
    saved = sys.argv
    try:
        sys.argv = [saved[0] if saved else "train_protenix_monomer.py"]
        defaults = parse_args()
    finally:
        sys.argv = saved
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def dry_run_components(components, n_items: int) -> None:
    batch = components.train_dataset[0]
    feat = batch["input_feature_dict"]
    label = batch["label_dict"]
    print(f"dry_run_ok dataset_rows={n_items}")
    print(f"source_name={batch['source_name']}")
    print(f"n_tokens={feat['restype'].shape[-2]}")
    print(f"n_atoms={label['coordinate'].shape[-2]}")
    print(f"design_tokens={int(feat['design_token_mask'].sum().item())}")
    print(f"has_sidechain_targets={'sc_gt_local' in feat}")
    if hasattr(components.train_dataset, "source_names"):
        print(f"sources={','.join(components.train_dataset.source_names)}")
        print(f"schedule_step0={components.schedule.weights_at(0)}")
        print(
            f"schedule_final={components.schedule.weights_at(components.schedule.stage2_start_step)}"
        )


def main() -> None:
    args = parse_args()
    apply_training_stage_args(args)
    repo_root = _bootstrap_paths(args)
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch

    data_root = Path(args.data_root).resolve()
    source_index = (
        Path(args.source_index).resolve()
        if args.source_index
        else _source_index_path(data_root)
    )
    output_dir = Path(args.output_dir).resolve()
    filtered_index = (
        Path(args.filtered_index).resolve()
        if args.filtered_index
        else output_dir / "cache" / "protenix_monomer_index.csv.gz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.max_n_token) > int(args.crop_size):
        raise ValueError(
            "--max-n-token must be <= --crop-size for whole-monomer binder training"
        )
    for name, value in [
        ("stage2_start_monomer_frac", args.stage2_start_monomer_frac),
        ("stage2_end_monomer_frac", args.stage2_end_monomer_frac),
        ("complex_max_binder_fraction", args.complex_max_binder_fraction),
        ("pinder_complex_frac", args.pinder_complex_frac),
    ]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [0, 1], got {value}"
            )

    build_monomer_index(
        source_index=source_index,
        output_index=filtered_index,
        min_n_token=int(args.min_n_token),
        max_n_token=int(args.max_n_token),
        limit=int(args.limit_index),
        rebuild=bool(args.rebuild_index),
    )
    protenix_complex_index = None
    pinder_manifest = None
    if args.data_mode == "mixed_monomer_complex":
        if args.complex_provider in {"protenix", "both"}:
            complex_source_index = (
                Path(args.complex_source_index).resolve()
                if args.complex_source_index
                else source_index
            )
            protenix_complex_index = (
                Path(args.complex_filtered_index).resolve()
                if args.complex_filtered_index
                else output_dir / "cache" / "protenix_ppi_complex_index.csv.gz"
            )
            build_ppi_complex_index(
                source_index=complex_source_index,
                output_index=protenix_complex_index,
                min_n_token=int(args.min_n_token),
                max_n_token=int(args.complex_max_n_token),
                limit=int(args.complex_limit_index),
                rebuild=bool(args.rebuild_index),
            )
        if args.complex_provider in {"pinder", "both"}:
            pinder_manifest = (
                Path(args.pinder_manifest).resolve()
                if args.pinder_manifest
                else (
                    Path(args.complex_filtered_index).resolve()
                    if args.complex_provider == "pinder"
                    and args.complex_filtered_index
                    else Path(args.pinder_root).resolve()
                    / "indices"
                    / "pinder_ppi_complex.parquet"
                )
            )
            if not pinder_manifest.is_file():
                raise FileNotFoundError(
                    "Prepared PINDER manifest not found: "
                    f"{pinder_manifest}. Run scripts/data/prepare_pinder_dataset.py."
                )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)
    if args.data_mode == "mixed_monomer_complex":
        components, n_items = build_mixed_components(
            args, filtered_index, protenix_complex_index, pinder_manifest
        )
    else:
        components, n_items = build_components(args, filtered_index)
    eval_loader, n_eval, eval_filtered_index = build_eval_dataloader(args, output_dir)
    components.eval_dataloader = eval_loader
    configs = build_configs(args, device)
    if args.dry_run:
        dry_run_components(components, n_items)
        if eval_loader is not None:
            print(f"validation_rows={n_eval}")
            print(f"validation_index={eval_filtered_index}")
        if args.training_stage == "sidechain_warmup":
            print(
                "sidechain_parameterization="
                f"centre_input={bool(configs.sidechain.centre_coord_input)},"
                f"frame_aware_head={bool(configs.sidechain.frame_aware_head)},"
                f"template_residual={bool(configs.sidechain.template_residual)}"
            )
            print(
                "optimizer="
                f"lr={float(configs.training.lr):g},"
                f"accumulate={int(configs.training.iters_to_accumulate)},"
                f"grad_clip={float(configs.training.grad_clip_norm):g}"
            )
        return

    from pxdesign_train.runner.train import train_from_components

    logging.info("Repo root: %s", repo_root)
    logging.info("Data root: %s", data_root)
    if isinstance(n_items, dict):
        logging.info("Monomer dataset rows: %d", n_items["monomer"])
        logging.info(
            "Protein-protein complex dataset rows: %d total, %d Protenix, %d PINDER",
            n_items["complex"],
            n_items.get("protenix_complex", 0),
            n_items.get("pinder_complex", 0),
        )
        logging.info(
            "Stage-2 source mixture: %.2f/%.2f monomer/complex -> %.2f/%.2f monomer/complex, ramp %d-%d",
            float(args.stage2_start_monomer_frac),
            1.0 - float(args.stage2_start_monomer_frac),
            float(args.stage2_end_monomer_frac),
            1.0 - float(args.stage2_end_monomer_frac),
            int(args.curriculum_stage1_end_step),
            int(args.curriculum_stage2_start_step),
        )
        logging.info("Stage-2 source weights at step 0: %s", components.schedule.weights_at(0))
        logging.info(
            "Stage-2 source weights at final ramp step: %s",
            components.schedule.weights_at(components.schedule.stage2_start_step),
        )
    else:
        logging.info("Monomer dataset rows: %d", n_items)
    if eval_loader is not None:
        logging.info("Recent-PDB validation rows: %d", n_eval)
        logging.info("Recent-PDB validation index: %s", eval_filtered_index)
    if args.training_stage == "sidechain_warmup":
        logging.info(
            "Side-chain parameterization: centre_input=%s, frame_aware_head=%s, "
            "template_residual=%s",
            bool(configs.sidechain.centre_coord_input),
            bool(configs.sidechain.frame_aware_head),
            bool(configs.sidechain.template_residual),
        )
        logging.info(
            "Side-chain optimizer: lr=%g, accumulate=%d, grad_clip=%g",
            float(configs.training.lr),
            int(configs.training.iters_to_accumulate),
            float(configs.training.grad_clip_norm),
        )
    logging.info("Output dir: %s", output_dir)
    train_from_components(
        configs=configs,
        components=components,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        load_checkpoint_path=args.load_checkpoint or None,
        checkpoint_params_only=bool(args.warm_start_params_only),
        overlay_aa_head_path=args.load_aa_head_from or None,
        max_steps=int(args.max_steps),
    )


if __name__ == "__main__":
    main()
