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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from apm.data.datasets import (_length_filter, _max_coil_filter,
                               _mean_plddt_filter, _rog_filter)
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
    """Chains as items. Lengths vary, so batching is length-bucketed elsewhere.

    `a_token_dir` attaches the precomputed trunk feature for the a_token arms.
    It is read rather than recomputed because at zero coordinate noise with the
    orientation pinned the trunk is a deterministic function of the structure
    (`scripts/data/build_a_token_cache.py`), so a forward per step would buy
    nothing but cost one 259M-parameter model per batch.
    """

    def __init__(self, files, crop_size=None, seed: int = 0, a_token_dir=None):
        self.files = [Path(f) for f in files]
        self.crop_size = crop_size
        self.seed = int(seed)
        self.a_token_dir = Path(a_token_dir) if a_token_dir else None
        if not self.files:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # APM's featuriser draws from the global `random` (chain shuffling, and
        # the conditional-chain coin we pinned to 0.0). Seed per item so a run
        # is reproducible under multi-worker loading.
        random.seed(self.seed * 1_000_003 + i)
        out = featurise(self.files[i], crop_size=self.crop_size)
        if self.a_token_dir is not None:
            name = self.files[i].stem
            path = self.a_token_dir / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"no cached a_token for {name} at {path}. Training the "
                    "a_token arm without it would silently feed zeros, which is "
                    "the `none` arm wearing another name."
                )
            a = torch.from_numpy(np.load(path)).float()
            if a.shape[0] != out["aatypes_1"].shape[0]:
                raise ValueError(
                    f"{name}: cached a_token has {a.shape[0]} rows but the "
                    f"chain has {out['aatypes_1'].shape[0]} residues"
                )
            out["a_token"] = a
        return out


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


# ---------------------------------------------------------------------------
# Multi-source training index (APM's use_AFDB / use_SWISSPROT)
# ---------------------------------------------------------------------------
# APM's pdb_dataset sets `use_AFDB: True` and `use_SWISSPROT: True`, and the
# released checkpoint reaches 1.347 symmetry_rmsd where our monomer-only run
# reaches 1.611. Each source has its OWN filter chain and its own cluster file,
# and the order within a chain matters because `_rog_filter` fits a quantile,
# so the filters below are APM's calls in APM's order rather than a shared
# pipeline:
#
#   PDB        length -> coil -> rog                       (datasets.py:645)
#   AFDB       length -> coil -> rog -> mean_plddt >= 95    (datasets.py:663)
#   SWISSPROT  length -> avg_plddt >= 85 -> coil -> rog     (datasets.py:586)
#
# Cluster ids are offset per source exactly as APM offsets them, so one epoch
# still draws one sample per cluster and a cluster cannot span two sources.
METADATA_ALL = Path("/hai/scratch/shenjm/apm_weights/metadata_all")
EXTRACTED = Path("/hai/scratch/yfsun/apm/extracted")

SOURCE_DIRS = {
    "PDB": EXTRACTED / "data_APM/pdb_monomer",
    "AFDB": EXTRACTED / "data_APM/afdb",
    "SWISSPROT": EXTRACTED / "swissprot_data",
}
SWISSPROT_PLDDT = 85.0          # datasets.py:592 -- NOT the AFDB threshold
AFDB_PLDDT = 95.0               # datasets.yaml :: filter.AFDB_plddt_threshold


def _read_clusters(cluster_path, synthetic=False):
    """apm/data/datasets.py:270, copied because it is six lines and pure."""
    out = {}
    with open(cluster_path) as fh:
        for i, line in enumerate(fh):
            for chain in line.split(" "):
                pdb = chain.strip() if synthetic else chain.split("_")[0].strip()
                out[pdb.upper()] = i
    return out


def _swissprot_clusters(path):
    """`swissprot_cluster50_cluster.tsv` is `cluster_rep <TAB> member`."""
    groups = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            groups.setdefault(parts[0], []).append(parts[1])
    return {member: i for i, members in enumerate(groups.values()) for member in members}


def multi_source_index(sources=("PDB", "AFDB", "SWISSPROT"), data_root=None,
                       filters=None):
    """Combined training index over APM's sources.

    Returns (DataFrame with pdb_name/modeled_seq_len/cluster/src/path, per-source
    counts). Rows whose pickle is absent are dropped and counted, because the
    extraction is a subset of the metadata for SwissProt.
    """
    F = dict(PDB_FILTER) if filters is None else dict(filters)
    dirs = dict(SOURCE_DIRS)
    if data_root:
        # `data_root` already points AT data_APM (the trainer's default is
        # .../extracted/data_APM), so this is one level, not two.
        dirs["PDB"] = Path(data_root) / "pdb_monomer"
    frames, counts, offset = [], {}, 0

    for src in sources:
        if src == "PDB":
            csv = (Path(data_root) / "meta_data.csv" if data_root
                   else EXTRACTED / "data_APM/meta_data.csv")
            meta = pd.read_csv(csv, low_memory=False)
            meta = meta[meta["processed_path"].astype(str).str.contains("train_set")]
            raw = len(meta)
            meta = _length_filter(meta, F["min_num_res"], F["max_num_res"])
            meta = _max_coil_filter(meta, F["max_coil_percent"])
            meta = _rog_filter(meta, F["rog_quantile"])
            # Keep the cluster column the completed monomer-only runs used, so
            # "more data" is the only thing that changes between the two runs.
            clusters = meta["cluster"].astype(int)
        elif src == "AFDB":
            meta = pd.read_csv(METADATA_ALL / "metadata_90.csv", low_memory=False)
            raw = len(meta)
            meta = _length_filter(meta, F["min_num_res"], F["max_num_res"])
            meta = _max_coil_filter(meta, F["max_coil_percent"])
            meta = _rog_filter(meta, F["rog_quantile"])
            meta = _mean_plddt_filter(meta, AFDB_PLDDT)
            lut = _read_clusters(METADATA_ALL / "AFDB.clusters", synthetic=True)
            clusters = meta["pdb_name"].astype(str).str.upper().map(lut)
        elif src == "SWISSPROT":
            meta = pd.read_csv(METADATA_ALL / "swissprot_metadata.csv", low_memory=False)
            raw = len(meta)
            meta = meta[(meta.modeled_seq_len >= F["min_num_res"]) &
                        (meta.modeled_seq_len <= F["max_num_res"])]
            meta = meta[meta.avg_plddt >= SWISSPROT_PLDDT]
            meta = _max_coil_filter(meta, F["max_coil_percent"])
            meta = _rog_filter(meta, F["rog_quantile"])
            lut = _swissprot_clusters(METADATA_ALL / "swissprot_cluster50_cluster.tsv")
            clusters = meta["pdb_name"].astype(str).map(lut)
        else:
            raise ValueError(f"unknown source {src!r}")

        meta = meta.copy()
        # A name with no cluster entry becomes its own cluster, which is what
        # APM's `cluster_lookup` does for a missing pdb.
        miss = clusters.isna()
        if miss.any():
            clusters = clusters.copy()
            clusters[miss] = range(int(clusters.max() or 0) + 1,
                                   int(clusters.max() or 0) + 1 + int(miss.sum()))
        meta["cluster"] = clusters.astype(int) + offset
        offset = int(meta["cluster"].max()) + 1
        meta["src"] = src
        meta["path"] = [str(dirs[src] / f"{n}.pkl") for n in meta["pdb_name"].astype(str)]
        on_disk = [Path(x).is_file() for x in meta["path"]]
        counts[src] = dict(raw=raw, filtered=len(meta), on_disk=int(sum(on_disk)))
        frames.append(meta[on_disk][["pdb_name", "modeled_seq_len", "cluster", "src", "path"]])

    combined = pd.concat(frames, ignore_index=True)
    return combined, counts
