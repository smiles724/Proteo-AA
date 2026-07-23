"""
CLI entrypoint for PXDesign-d training.

This module **does not** know how to read your PDB / AFDB / MGnify data on
disk — that's deliberately your job, since data layout varies wildly. To use
it, write a small Python driver that:

  1. Builds one `ComplexProvider` per source (see `runner/data.py` for the
     protocol).
  2. Wraps each provider in a `DesignSourceDataset(provider, source_name=...)`.
  3. Combines them with `CurriculumMultiDataset` + per-item weights
     (uniform = `[1.0] * len(source)` if you don't have a smarter prior).
  4. Defines a `CurriculumSchedule` over the source names.
  5. Calls `train_from_components(configs, components)`.

The CLI here is a thin wrapper that just parses configs and dispatches to
your driver — see `scripts/train_demo.sh` for an example invocation against
the synthetic test data.

Test commit
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from pxdesign_train.runner.trainer import PXDesignTrainer, TrainerComponents


def train_from_components(
    configs: Any,
    components: TrainerComponents,
    *,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    checkpoint_dir: str | None = None,
    load_checkpoint_path: str | None = None,
    checkpoint_params_only: bool = True,
    max_steps: int | None = None,
) -> PXDesignTrainer:
    """Construct a trainer and run the loop. Returns the trainer for callers
    that want to inspect state after training.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    trainer = PXDesignTrainer(
        configs=configs,
        components=components,
        device=device,
        rank=rank,
        world_size=world_size,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint_path=load_checkpoint_path,
        checkpoint_params_only=checkpoint_params_only,
    )
    trainer.run(max_steps=max_steps)
    return trainer


def main() -> None:
    """Placeholder main: expects a driver to be wired in.

    A typical user-side driver looks like:

        from pxdesign_train.runner.train import train_from_components
        from pxdesign_train.runner.trainer import TrainerComponents
        from pxdesign_train.data import (
            CurriculumMultiDataset, CurriculumSchedule,
        )
        from pxdesign_train.runner.data import DesignSourceDataset
        from pxdesign_train.configs.configs_train import training_configs
        from protenix.config.config import parse_configs

        configs = parse_configs(training_configs, arg_str='')
        # IMPORTANT: forward the residue-type masking config, otherwise
        # DesignSourceDataset defaults to aa_mask_mode='all' and the
        # time_dependent masked-diffusion schedule in configs is silently ignored.
        rt = configs.residue_type
        aa_kw = dict(aa_mask_mode=rt.mask_mode, aa_mask_prob=rt.mask_prob,
                     aa_mask_min_prob=rt.mask_min_prob, aa_mask_max_prob=rt.mask_max_prob)
        afdb_ds = DesignSourceDataset(my_afdb_provider, source_name='afdb', **aa_kw)
        pdb_ds  = DesignSourceDataset(my_pdb_provider,  source_name='pdb', **aa_kw)
        multi = CurriculumMultiDataset(
            datasets=[afdb_ds, pdb_ds],
            source_names=['afdb', 'pdb'],
            per_item_weights=[[1.0]*len(afdb_ds), pdb_weights],
        )
        sched = CurriculumSchedule(
            stage1={'afdb': 0.9, 'pdb': 0.1},
            stage2={'afdb': 0.1, 'pdb': 0.9},
            stage1_end_step=40_000,
            stage2_start_step=60_000,
        )
        components = TrainerComponents(
            train_dataset=multi,
            schedule=sched,
            train_samples_per_epoch=1_000,
        )
        train_from_components(configs, components, checkpoint_dir='./ckpts')
    """
    raise SystemExit(
        "pxdesign_train.runner.train: this module is a library. Write a driver "
        "that builds your ComplexProviders and calls `train_from_components()`. "
        "See `scripts/train_demo.sh` for an example."
    )


if __name__ == "__main__":
    main()
