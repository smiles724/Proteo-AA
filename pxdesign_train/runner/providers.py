"""
ComplexProvider adapters for real datasets.

A `ComplexProvider` is anything indexable that yields one uncropped, real
protein complex per `__getitem__`. The trainer's `DesignSourceDataset` then
crops it (via `DesignCropper`) and design-featurizes it (via
`DesignFeaturizer`) before handing it to the model.

This module provides one production adapter and a small library of binder
selectors:

  - `ProtenixComplexProvider`: wraps a `protenix.data.pipeline.dataset.
    BaseSingleDataset`. **Important**: construct the underlying dataset with
    `cropping_configs={"crop_size": 0, ...}` so Protenix returns the full
    bioassembly. Our `DesignCropper` does the design-aware crop.

  - Binder selectors:
      * `select_chain_by_id("B")` — always pick this chain.
      * `select_protenix_chain_2()` — for PPI interface samples, pick the
        second chain from `sample_indice` (typically the smaller binder).
      * `select_smallest_protein_chain()` — pick whichever protein chain has
        the fewest residues. Reasonable monomer default.
      * `select_random_protein_chain(seed=...)` — uniform random pick over
        all protein chains.

Tests for this module use a tiny mock dataset rather than real PDB data, so
this file ships without any heavy I/O dependencies at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np


# Each binder selector is a callable: (raw_data_dict, atom_array) -> chain_id|mask.
BinderSelectorFn = Callable[[dict[str, Any], Any], Union[str, np.ndarray]]


# ----- pre-built binder selectors -----


def select_chain_by_id(chain_id: str) -> BinderSelectorFn:
    """Always pick a fixed chain. Useful when your indices_list pre-assigns
    a binder per sample (e.g. a column in the dataframe)."""

    def _sel(_data, _atom_array):
        return chain_id

    return _sel


def _clean_chain_id(chain_id: Any) -> Optional[str]:
    """Normalize pandas/CSV missing values from Protenix index rows."""
    if chain_id is None:
        return None
    try:
        if bool(np.isnan(chain_id)):
            return None
    except TypeError:
        pass
    s = str(chain_id)
    if s == "" or s.lower() == "nan":
        return None
    return s


def select_protenix_chain_1() -> BinderSelectorFn:
    """Pick the first reference chain from a Protenix index row.

    This is the correct selector for monomer `type == "chain"` rows, where
    `chain_1_id` is the protein chain and `chain_2_id` is missing.
    """

    def _sel(data, _atom_array):
        chain_pair = data.get("__binder_chain_pair__")
        if chain_pair is None:
            raise RuntimeError(
                "select_protenix_chain_1 needs `__binder_chain_pair__` in data — "
                "make sure ProtenixComplexProvider was constructed with "
                "expose_sample_indice=True."
            )
        chain_1, _chain_2 = chain_pair
        chain_1 = _clean_chain_id(chain_1)
        if chain_1 is None:
            raise ValueError("Protenix sample has no chain_1_id")
        return chain_1

    return _sel


def select_protenix_chain_2() -> BinderSelectorFn:
    """For PPI interface samples in Protenix's `BaseSingleDataset`, pick the
    second reference chain (`sample_indice.chain_2_id`).

    Falls back to `chain_1_id` if `chain_2_id` is missing (e.g. a chain-type
    sample, where there's no second chain).
    """

    def _sel(data, _atom_array):
        # process_one stashes the sample_indice in `data["basic"]` only as
        # `chain_id` (the *cropped* chain list). We piggyback on a custom
        # field that ProtenixComplexProvider stuffs into the data dict.
        chain_pair = data.get("__binder_chain_pair__")
        if chain_pair is None:
            raise RuntimeError(
                "select_protenix_chain_2 needs `__binder_chain_pair__` in data — "
                "make sure you constructed ProtenixComplexProvider with the "
                "default `expose_sample_indice=True`."
            )
        chain_1, chain_2 = chain_pair
        chain_1 = _clean_chain_id(chain_1)
        chain_2 = _clean_chain_id(chain_2)
        return chain_2 if chain_2 is not None else chain_1

    return _sel


def select_smallest_protein_chain() -> BinderSelectorFn:
    """Pick the protein chain with the fewest residues."""

    def _sel(_data, atom_array):
        is_protein = (atom_array.mol_type == "protein")
        chains, counts = np.unique(atom_array.chain_id[is_protein], return_counts=True)
        if len(chains) == 0:
            raise ValueError("No protein chains found in atom_array")
        return str(chains[np.argmin(counts)])

    return _sel


def select_random_protein_chain(seed: Optional[int] = None) -> BinderSelectorFn:
    """Pick a uniformly-random protein chain. Useful for monomer training
    where we just need *some* contiguous region to designate as the binder."""

    rng = np.random.default_rng(seed)

    def _sel(_data, atom_array):
        is_protein = (atom_array.mol_type == "protein")
        chains = np.unique(atom_array.chain_id[is_protein])
        if len(chains) == 0:
            raise ValueError("No protein chains found in atom_array")
        return str(rng.choice(chains))

    return _sel


# ----- adapters -----


@dataclass
class ProtenixComplexProvider:
    """Adapter from Protenix's `BaseSingleDataset` to our `ComplexProvider`.

    Args:
        base_dataset: an already-constructed `BaseSingleDataset`. Pass
            `cropping_configs.crop_size=0` at construction so Protenix returns
            the full bioassembly arrays — our `DesignCropper` does the
            design-aware crop instead.
        binder_selector_fn: a callable from `select_*()` above (or your own).
        expose_sample_indice: if True (default), inject the sample_indice's
            chain pair into the returned data dict under
            `__binder_chain_pair__` so selectors like `select_protenix_chain_2()`
            can read it. Set False if you don't need it.
    """

    base_dataset: Any
    binder_selector_fn: BinderSelectorFn
    expose_sample_indice: bool = True

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        data = self.base_dataset.process_one(idx, return_atom_token_array=True)
        atom_array = data["cropped_atom_array"]
        token_array = data["cropped_token_array"]
        feature_dict = data["input_feature_dict"]
        label_dict = data["label_dict"]

        if self.expose_sample_indice:
            # `_get_bioassembly_data` returned the (possibly-resolved) sample_indice
            # only inside `process_one`'s scope; reach back into the dataset to
            # extract the chain pair for this idx.
            indice = self.base_dataset._get_sample_indice(idx)
            chain_1 = indice.get("chain_1_id") if hasattr(indice, "get") else None
            chain_2 = indice.get("chain_2_id") if hasattr(indice, "get") else None
            data["__binder_chain_pair__"] = (
                _clean_chain_id(chain_1),
                _clean_chain_id(chain_2),
            )

        # Capture data + selector_fn in a closure that has the signature
        # DesignSourceDataset expects from a BinderSelector.
        fn = self.binder_selector_fn
        binder_selector = lambda aa, _d=data: fn(_d, aa)
        return atom_array, token_array, feature_dict, label_dict, binder_selector
