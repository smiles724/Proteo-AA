"""Deterministic, microstep-indexed sampling for constant SC adaptation recipes."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


@contextmanager
def item_rng(seed):
    # Workers must not initialize CUDA. Restore caller state for num_workers=0.
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)


def isolated_evaluation(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from pxdesign_train.checkpoints import rng_state, restore_rng
        saved = rng_state()
        try:
            seed_all(int(getattr(self.configs, "seed", 0)) + 1000003)
            return fn(self, *args, **kwargs)
        finally:
            restore_rng(saved)
    return wrapper


def fingerprint(paths, settings):
    import gzip
    def content_hash(path):
        if str(path).endswith(".csv.gz"):
            with gzip.open(path, "rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()
        return sha256_file(path)
    record = dict(files={str(Path(p).resolve()): content_hash(p) for p in paths}, settings=settings)
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class MicrostepSampler(Sampler):
    def __init__(self, dataset, accumulation):
        self.dataset, self.accumulation, self.cursor = dataset, int(accumulation), 0

    def set_step(self, step):
        self.cursor = int(step) * self.accumulation

    def __iter__(self):
        return iter(range(self.cursor, len(self.dataset)))

    def __len__(self):
        return len(self.dataset) - self.cursor


class SCStream(Dataset):
    """Index determines source, provider item, crop RNG and reconstruction seed.

    Prefetch never advances the saved cursor; only consumed trainer microsteps do.
    Source and coordinate mixtures use independent draws. Full samples are a
    separate unlabeled cache and cannot borrow a paired item's labels.
    """
    def __init__(self, dataset, schedule, cfg, *, seed, microsteps, full_samples=None):
        if schedule.stage1 != schedule.stage2:
            raise ValueError("SC adaptation requires constant source mixtures")
        weights = dataset.merged_weights(schedule.stage1).numpy()
        self.cdf = np.cumsum(weights / weights.sum())
        self.dataset, self.cfg, self.seed = dataset, cfg, int(seed)
        self.microsteps, self.full_samples = int(microsteps), full_samples
        if float(cfg.full_sample_fraction) and not full_samples:
            raise ValueError("Full-sample training requires an explicit cache")
        if float(cfg.full_sample_fraction):
            for source, weight in schedule.stage1.items():
                if weight > 0 and not full_samples.by_source.get(source):
                    raise ValueError(f"Full-sample cache lacks source {source}; source and coordinate mixtures must stay independent")

    def __len__(self):
        return self.microsteps

    def __getitem__(self, index):
        from pxdesign_train.sc_adaptation import choose_source
        seed = int(np.random.SeedSequence([self.seed, int(index)]).generate_state(1)[0])
        source = "native" if self.cfg.phase in ("sc_geometry_repair", "sc_complex_adapt") else choose_source(self.cfg, seed=seed+19)
        rng = np.random.default_rng(seed+31)
        idx = min(int(np.searchsorted(self.cdf, rng.random())), len(self.dataset)-1)
        with item_rng(seed):
            if source == "full_sample":
                protein_source = self.dataset.source_names[self.dataset._source_idx_per_item[idx]]
                candidates = self.full_samples.by_source[protein_source]
                batch = self.full_samples[candidates[int(rng.integers(len(candidates)))]]
            else:
                batch = self.dataset[idx]
        batch = dict(batch)
        batch["input_feature_dict"] = dict(batch["input_feature_dict"], backbone_source=source, input_seed=seed)
        batch["microstep"] = int(index)
        return batch


class FullSampleCache(Dataset):
    def __init__(self, manifest, *, checkpoint_sha256, partition):
        self.path = Path(manifest).resolve()
        record = json.loads(self.path.read_text())
        if record.get("schema") != "sc_full_samples_v1" or record.get("partition") != partition:
            raise ValueError("Full-sample cache schema/partition mismatch")
        self.items = record["items"]
        self.by_source = {}
        for index,item in enumerate(self.items):
            self.by_source.setdefault(item["source_name"], []).append(index)
        self.checkpoint_sha256 = checkpoint_sha256
        if not self.items:
            raise ValueError("Full-sample cache is empty")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from pxdesign_train.sc_adaptation import check_source
        item = self.items[index]
        path = self.path.parent / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Full-sample cache checksum changed: {path}")
        batch = torch.load(path, map_location="cpu", weights_only=False)
        feat = batch["input_feature_dict"]
        if batch.get("source_name") != item["source_name"] or batch.get("sample_id") != item["sample_id"]:
            raise ValueError("Full-sample cache source/sample identity mismatch")
        check_source(feat, batch["label_dict"])
        if feat["backbone_provenance"]["checkpoint_sha256"] != self.checkpoint_sha256:
            raise ValueError("Cached backbone came from a different official checkpoint")
        if int(feat["backbone_provenance"]["steps"]) != 400:
            raise ValueError("This full-sample validation panel requires 400 native sampler steps")
        if feat["backbone_provenance"].get("test_fixture"):
            raise ValueError("Synthetic test fixtures cannot enter a training/validation cache")
        return batch


class CoordinatePanel:
    """Fixed data/seed/sigma view of an existing validation loader."""
    def __init__(self, loader, source, sigma=None, seed=1000003):
        self.loader, self.source, self.sigma, self.seed = loader, source, sigma, seed

    def __iter__(self):
        # CIF conversion/reference featurization may consume RNG only on a cache
        # miss. Keep that consumption out of the model's SC initialization stream.
        with item_rng(self.seed):
            iterator = iter(self.loader)
        for index in range(len(self.loader)):
            with item_rng(self.seed+index):
                batch = next(iterator)
            feat = dict(batch["input_feature_dict"], backbone_source=self.source, input_seed=self.seed+index)
            if self.sigma is not None:
                feat["reconstruction_sigma"] = self.sigma
            yield dict(batch, input_feature_dict=feat)

    def __len__(self):
        return len(self.loader)
