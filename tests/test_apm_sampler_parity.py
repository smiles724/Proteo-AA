"""Our one-replica batcher must emit APM's batches, index for index.

Almost everything in the APM-data training path is APM's own function, called
rather than copied. The sampler is the exception: `LengthBatcher_nonRep` is
written against a distributed rig, so it was reimplemented for one replica.
That makes it the piece most able to diverge without anything failing --
a different batch composition is not an error, just a different experiment.

Skipped when the vendored APM checkout or the real metadata is not reachable,
because this is a parity test against an external reference, not a unit test
of our own logic.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
META = Path("/hai/scratch/yfsun/apm/extracted/data_APM/meta_data.csv")

apm_available = importlib.util.find_spec("apm") is not None


@pytest.mark.skipif(not apm_available, reason="apm_reference not on PYTHONPATH")
@pytest.mark.skipif(not META.is_file(), reason="APM metadata not reachable")
@pytest.mark.parametrize("epoch", [0, 1, 5])
def test_batches_match_apms(epoch):
    import numpy as np
    import pandas as pd
    from apm.data.datasets import _length_filter, _max_coil_filter, _rog_filter
    from apm.data.protein_dataloader import LengthBatcher_nonRep

    from pxdesign_train.sidechain.apm_dataset import PDB_FILTER

    spec = importlib.util.spec_from_file_location(
        "tr", str(REPO / "scripts" / "training" / "train_packer_apm_data.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    meta = pd.read_csv(META, low_memory=False)
    meta = meta[meta.processed_path.astype(str).str.contains("train_set")]
    meta = _length_filter(meta, PDB_FILTER["min_num_res"], PDB_FILTER["max_num_res"])
    meta = _max_coil_filter(meta, PDB_FILTER["max_coil_percent"])
    meta = _rog_filter(meta, PDB_FILTER["rog_quantile"]).reset_index(drop=True)

    theirs_csv = meta.copy()
    theirs_csv["index"] = np.arange(len(theirs_csv))
    theirs = LengthBatcher_nonRep(
        sampler_cfg=SimpleNamespace(max_batch_size=64, max_num_res_squared=400_000),
        metadata_csv=theirs_csv, seed=123, shuffle=True, num_replicas=1, rank=0)
    ours = tr.LengthBatcher(meta, max_batch_size=64, max_num_res_squared=400_000,
                            seed=123)

    theirs.epoch = epoch
    ours.set_epoch(epoch)
    assert theirs._epoch_batches() == list(iter(ours))
