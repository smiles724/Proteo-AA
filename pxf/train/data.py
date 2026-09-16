"""Training data pipeline: fixed-size crops with optional structural noise.

FaMPNN ships no training dataset, so this implements the preprint's scheme:

* Appendix B.3.1 -- every example is brought to a fixed size, by cropping a single
  randomly placed contiguous span when longer and padding when shorter.
* Appendix B.3.2 / Algorithm 1 -- multichain examples use a two-chain contiguous
  crop; the paper also alternates with AlphaFold-Multimer spatial cropping at even
  odds (:func:`spatial_crop`).
* Appendix B.1 -- structural noise is independent ``N(0, sigma^2)`` on every x, y, z
  coordinate, with ``sigma = 0.3`` and ``0.0`` being the two released models.
* Appendix B.3.2 -- one example is drawn per training cluster per epoch.

The per-example featurization itself is upstream's (``load_feats_from_pdb`` ->
``process_single_pdb``), so features match what inference produces.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

# What the training step consumes; everything else upstream computes is dropped.
BATCH_KEYS = (
    "x",
    "aatype",
    "seq_mask",
    "missing_atom_mask",
    "residue_index",
    "chain_index",
)


def contiguous_crop(length, size, *, generator=None):
    """Indices of one randomly placed contiguous span of ``size`` residues.

    Appendix B.3.1: the start is uniform on ``[0, length - size]``. Returns all
    indices when the example is already at or below the target size.
    """
    if length <= size:
        return torch.arange(length)
    start = int(torch.randint(0, length - size + 1, (1,), generator=generator))
    return torch.arange(start, start + size)


def multimer_contiguous_crop(chain_lengths, size, *, generator=None):
    """Paper Algorithm 1: split a crop budget across two chains.

    ``c1`` is drawn uniformly from the feasible range so that ``c2 = size - c1``
    also fits, then each chain contributes one contiguous span.
    """
    if len(chain_lengths) != 2:
        raise ValueError(f"Algorithm 1 covers two chains, got {len(chain_lengths)}")
    l1, l2 = int(chain_lengths[0]), int(chain_lengths[1])
    total = l1 + l2
    if total <= size:
        return torch.arange(total)
    low = min(l1, max(0, size - l2))
    high = min(l1, size)
    c1 = int(torch.randint(low, high + 1, (1,), generator=generator))
    c2 = size - c1
    s1 = int(torch.randint(0, max(1, total - size + 1), (1,), generator=generator))
    keep = torch.zeros(total, dtype=torch.bool)
    keep[s1 : s1 + c1] = True
    if c2 > 0:
        lo2, hi2 = s1 + c1, max(s1 + c1, total - c2)
        s2 = (
            int(torch.randint(lo2, hi2 + 1, (1,), generator=generator))
            if hi2 >= lo2
            else lo2
        )
        keep[s2 : s2 + c2] = True
    return keep.nonzero(as_tuple=True)[0]


def spatial_crop(ca_coords, chain_index, size, *, generator=None):
    """AlphaFold-Multimer style interface crop: nearest residues to a seed.

    Seeds on a residue near the interface when one exists, then keeps the
    ``size`` residues closest to it by CA distance, which is what maximizes
    interface coverage for two-chain examples.
    """
    length = ca_coords.shape[0]
    if length <= size:
        return torch.arange(length)
    chains = torch.unique(chain_index)
    seed = None
    if len(chains) == 2:
        a = chain_index == chains[0]
        b = chain_index == chains[1]
        distances = torch.cdist(ca_coords[a], ca_coords[b])
        if distances.numel():
            flat = int(distances.argmin())
            seed = int(a.nonzero(as_tuple=True)[0][flat // distances.shape[1]])
    if seed is None:
        seed = int(torch.randint(0, length, (1,), generator=generator))
    order = torch.cdist(ca_coords[seed][None], ca_coords)[0].argsort()
    return order[:size].sort().values


def add_structural_noise(x, sigma, *, generator=None):
    """Independent ``N(0, sigma^2)`` on every coordinate (Appendix B.1)."""
    if not sigma:
        return x
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return x + noise * float(sigma)


def pad_or_crop(example, indices, size):
    """Select ``indices`` from a per-residue example and pad out to ``size``."""
    from fampnn.data.data import pad_to_max_len

    selected = {key: example[key][indices] for key in BATCH_KEYS}
    return pad_to_max_len(
        {key: value.unsqueeze(0) for key, value in selected.items()}, size
    )


@dataclass
class ClusterIndex:
    """Training members grouped by cluster, for one-sample-per-cluster epochs."""

    clusters: dict

    @classmethod
    def from_csv(cls, path):
        """Read ``cluster_id,path`` rows (header optional)."""
        groups = {}
        with Path(path).open() as stream:
            for row in csv.reader(stream):
                if len(row) < 2 or row[0].strip().lower() in ("cluster", "cluster_id"):
                    continue
                groups.setdefault(row[0].strip(), []).append(row[1].strip())
        if not groups:
            raise ValueError(f"No cluster,path rows in {path}")
        return cls(groups)

    @classmethod
    def from_paths(cls, paths):
        """One cluster per structure -- the degenerate case, for small sets."""
        return cls({str(i): [str(p)] for i, p in enumerate(paths)})

    def __len__(self):
        return len(self.clusters)

    def sample(self, generator=None):
        """One member per cluster, as Appendix B.3.2 specifies per epoch."""
        keys = sorted(self.clusters)
        picks = []
        for key in keys:
            members = self.clusters[key]
            index = int(torch.randint(0, len(members), (1,), generator=generator))
            picks.append(members[index])
        return picks


class StructureCropDataset(Dataset):
    """Fixed-size crops of PDB structures, ready for :func:`pxf.train.step`.

    ``crop_size`` is the paper's "fixed example size" (256 for CATH, 1024 for PDB).
    ``noise`` is the structural noise sigma; the two released models used 0.0 and 0.3.
    """

    def __init__(
        self,
        paths,
        *,
        crop_size=256,
        noise=0.0,
        spatial_crop_p=0.5,
        seed=0,
        noise_targets=True,
    ):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("StructureCropDataset needs at least one structure")
        self.crop_size = int(crop_size)
        self.noise = float(noise)
        self.spatial_crop_p = float(spatial_crop_p)
        self.seed = int(seed)
        # Appendix B.1 says noise is added to "protein structure examples", which
        # reads as the whole example -- so the diffusion target is noised too.
        # That is consistent with the 0.3 A model packing worse than the 0.0 A one.
        # Set False to noise only the encoder's view and keep clean targets.
        self.noise_targets = bool(noise_targets)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.paths)

    def _generator(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * self.epoch + index)
        return generator

    def __getitem__(self, index):
        from fampnn.data.data import load_feats_from_pdb, process_single_pdb

        from fampnn.data import residue_constants as rc

        generator = self._generator(index)
        path = self.paths[index]
        example = process_single_pdb(load_feats_from_pdb(str(path)))
        length = example["x"].shape[0]

        chain_index = example["chain_index"]
        chains = torch.unique(chain_index)
        if (
            len(chains) == 2
            and float(torch.rand((), generator=generator)) < self.spatial_crop_p
        ):
            indices = spatial_crop(
                example["x"][:, rc.atom_order["CA"]],
                chain_index,
                self.crop_size,
                generator=generator,
            )
        elif len(chains) == 2:
            counts = [int((chain_index == c).sum()) for c in chains]
            indices = multimer_contiguous_crop(counts, self.crop_size, generator=generator)
        else:
            indices = contiguous_crop(length, self.crop_size, generator=generator)

        item = pad_or_crop(example, indices, self.crop_size)
        item = {key: value.squeeze(0) for key, value in item.items()}
        if self.noise:
            noised = add_structural_noise(item["x"], self.noise, generator=generator)
            item["x"] = noised if self.noise_targets else item["x"]
            item["x_input"] = noised
        item["name"] = path.stem
        return item


def collate(items):
    """Stack fixed-size examples; non-tensor fields are gathered into lists."""
    batch = {key: torch.stack([item[key] for item in items]) for key in BATCH_KEYS}
    if "x_input" in items[0]:
        batch["x_input"] = torch.stack([item["x_input"] for item in items])
    batch["name"] = [item.get("name") for item in items]
    return batch


def build_loader(
    paths,
    *,
    batch_size=1,
    crop_size=256,
    noise=0.0,
    seed=0,
    num_workers=0,
    shuffle=True,
    **kwargs,
):
    """A DataLoader over fixed-size crops."""
    from torch.utils.data import DataLoader

    dataset = StructureCropDataset(
        paths, crop_size=crop_size, noise=noise, seed=seed, **kwargs
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    return dataset, loader
