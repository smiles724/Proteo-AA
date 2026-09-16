"""APM's own featurisation of APM's pickles, called rather than reimplemented.

The earlier evaluation path re-derived frames and side-chain targets from the
raw atom37 in each pickle. That was fine for scoring, but for *training* a
reimplementation is a liability: APM's packing loss reads eleven tensors
(`rigidgroups_*`, `atom14_*`, `bb_torsions_1`, ...) that come out of openfold's
`data_transforms`, and any disagreement in those would show up as a quietly
different objective rather than as an error.

So this module imports `apm.data.datasets._process_csv_row_FAESM` and APM's
filter helpers directly from the vendored checkout, and adds only what APM's
`PdbDataset` wrapper would have added around them. Nothing here recomputes a
quantity APM already computes.

Requires on PYTHONPATH:  $APM_REFERENCE (apm + openfold)  and  $PYEXTRA
(dm-tree, lightning-utilities, and the `torch_scatter` shim), because
`apm/data/utils.py` imports all three at module scope.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from apm.data.datasets import _length_filter, _max_coil_filter, _rog_filter
from apm.data.datasets import _process_csv_row_FAESM

# apm/configs/datasets.yaml :: pdb_dataset.filter, the settings the released
# sidechain checkpoint was trained under.
PDB_FILTER = dict(max_num_res=384, min_num_res=60, max_coil_percent=0.5,
                  rog_quantile=0.96)

# Tensors the packing loss consumes; everything else in the feature dict is for
# APM's backbone/co-design path and is dropped, so a stray key cannot silently
# become a model input.
LOSS_KEYS = ("aatypes_1", "rotmats_1", "trans_1", "torsions_1", "torsions_mask",
             "res_mask", "diffuse_mask", "chain_idx", "res_idx",
             "bb_torsions_1", "bb_torsions_mask",
             "rigidgroups_gt_frames", "rigidgroups_alt_gt_frames",
             "rigidgroups_gt_exists", "atom14_gt_positions",
             "atom14_alt_gt_positions", "atom14_gt_exists",
             "atom14_atom_is_ambiguous", "atom14_alt_gt_exists")


def featurise(path, crop_size=None):
    """One pickle -> APM's feature dict, trimmed to the packing-loss tensors.

    `crop_size=None` is APM's monomer path: no crop, because the length filter
    already caps monomers at 384. Multimers pass their crop size and take
    APM's spatial crop.
    """
    feats = _process_csv_row_FAESM(str(path), crop_size=crop_size, anno="ESM",
                                   conditional_multimer_prop=0.0,
                                   conditional_multimer_ratio=0.0,
                                   train_packing_only=True)
    out = {k: feats[k] for k in LOSS_KEYS}
    # add_plddt_mask is False for pdb_dataset, so PdbDataset fills ones here
    # (datasets.py:414) and `mask_plddt: True` in the experiment config is a
    # no-op on PDB. Kept explicit so the loss mask reads the same as APM's.
    out["plddt_mask"] = torch.ones_like(out["res_mask"])
    out["name"] = Path(path).stem
    return out


class APMPackingDataset(Dataset):
    """Chains as items. Lengths vary, so batching is length-bucketed elsewhere."""

    def __init__(self, files, crop_size=None, seed: int = 0):
        self.files = [Path(f) for f in files]
        self.crop_size = crop_size
        self.seed = int(seed)
        if not self.files:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # APM's featuriser draws from the global `random` (chain shuffling, and
        # the conditional-chain coin we pinned to 0.0). Seed per item so a run
        # is reproducible under multi-worker loading.
        random.seed(self.seed * 1_000_003 + i)
        return featurise(self.files[i], crop_size=self.crop_size)


def pdb_monomer_files(root, metadata, apply_filters=True):
    """APM's filtered PDB-monomer training split, as absolute pickle paths.

    yfsun's extraction is already length/coil/oligomeric filtered; running
    APM's own filters over it again is cheap and makes the selection provable
    rather than assumed. Any row the filters drop is reported by the caller.
    """
    root = Path(root)
    meta = pd.read_csv(metadata, low_memory=False)
    meta = meta[meta["processed_path"].astype(str).str.contains("train_set")]
    n_raw = len(meta)
    if apply_filters:
        meta = _length_filter(meta, PDB_FILTER["min_num_res"], PDB_FILTER["max_num_res"])
        meta = _max_coil_filter(meta, PDB_FILTER["max_coil_percent"])
        meta = _rog_filter(meta, PDB_FILTER["rog_quantile"])
    files = [root / f"{n}.pkl" for n in meta["pdb_name"].astype(str)]
    files = [f for f in files if f.is_file()]
    return files, n_raw


def post2021_val_files(pdb_test_dir, test_ids_csv):
    """APM's 449 post-2021 held-out monomers."""
    ids = pd.read_csv(test_ids_csv)["pdb_name"].astype(str)
    out = [Path(pdb_test_dir) / f"{i}.pkl" for i in ids]
    return [f for f in out if f.is_file()]


__all__ = ["APMPackingDataset", "featurise", "pdb_monomer_files",
           "post2021_val_files", "LOSS_KEYS", "PDB_FILTER"]
