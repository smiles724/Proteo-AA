"""Cluster partitioning and within-source weights, after eligibility filtering."""
from collections import Counter
import hashlib


def inverse_cluster_weights(cluster_ids):
    if not cluster_ids or any(c is None or str(c).lower() in ("", "nan", "none") for c in cluster_ids):
        raise ValueError("Cluster sampling requires a cluster ID for every eligible row")
    counts = Counter(map(str, cluster_ids))
    return [1.0 / counts[str(c)] for c in cluster_ids]


def is_validation_cluster(cluster_id, fraction=0.1, seed=0):
    if not 0 < fraction < 1:
        raise ValueError("Cluster validation fraction must lie in (0,1)")
    digest = hashlib.sha256(f"{seed}:{cluster_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


class ClusterPartitionProvider:
    """Keep the provider's eligible rows in a deterministic cluster partition."""
    def __init__(self, provider, cluster_ids, validation=False, fraction=0.1, seed=0, limit=0):
        inverse_cluster_weights(cluster_ids)  # validate metadata before partition
        if len(provider) != len(cluster_ids):
            raise ValueError("Provider rows and cluster IDs are misaligned")
        self.provider = provider
        self.indices = [i for i,c in enumerate(cluster_ids) if is_validation_cluster(c,fraction,seed) == validation]
        if limit > 0:
            self.indices = self.indices[:limit]
        self.cluster_ids = [str(cluster_ids[i]) for i in self.indices]
        if not self.indices:
            raise ValueError("No eligible rows in the selected cluster partition")

    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.provider[self.indices[index]]
    def sample_id(self, index): return self.provider.sample_id(self.indices[index])


def source_weights(dataset, required=False):
    clusters = getattr(dataset.provider, "cluster_ids", None)
    if clusters is None:
        if required:
            raise ValueError("Complex source lost cluster IDs during provider construction")
        return [1.] * len(dataset)
    if len(clusters) != len(dataset):
        raise ValueError("Dataset rows and cluster weights are misaligned")
    return inverse_cluster_weights(clusters)
