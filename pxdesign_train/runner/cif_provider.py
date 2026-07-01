"""
CIF-file-based ComplexProvider for fine-tuning on curated structures.

When you have a handful of CIF files (e.g. your own wet-lab targets), this
provider lets you skip the full Protenix `BaseSingleDataset` setup (bioassembly
pickles, PDB cluster files, MSA databases). It parses each CIF directly via
`DataPipeline.get_data_from_mmcif`, featurizes with Protenix's base
`Featurizer`, and returns the (atom_array, token_array, feature_dict,
label_dict, binder_selector) 5-tuple our `DesignSourceDataset` expects.

Limitations vs. the full pipeline:
  - No MSA features (auto-filled with dummy zeros). The report says MSA is only
    used for the *filter stage*, not the diffusion generator, so this is fine.
  - No template features (dummy zeros).
  - Requires the CCD components file (Protenix reads `$PROTENIX_DATA_ROOT_DIR/
    components.cif`). Run `download_tool_weights.sh` or set the env var.

Usage:
    from pxdesign_train.runner.cif_provider import CifFileProvider
    from pxdesign_train.runner import DesignSourceDataset, select_chain_by_id

    provider = CifFileProvider(
        cif_paths=["target1.cif", "target2.cif"],
        binder_chain_ids=["B", "C"],         # one per CIF
    )
    # Forward the masking config so the AA head trains under the intended
    # schedule; the default is aa_mask_mode='all'.
    rt = configs.residue_type
    src = DesignSourceDataset(
        provider, source_name="my_targets", crop_size=640,
        aa_mask_mode=rt.mask_mode, aa_mask_prob=rt.mask_prob,
        aa_mask_min_prob=rt.mask_min_prob, aa_mask_max_prob=rt.mask_max_prob,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from protenix.data.core.featurizer import Featurizer
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.utils import data_type_transform, make_dummy_feature

logger = logging.getLogger(__name__)

BinderSelector = Any  # str | np.ndarray


class CifFileProvider:
    """Parse CIF files and serve them as `ComplexProvider` items.

    Args:
        cif_paths: list of paths to CIF (or `.cif.gz`) files.
        binder_chain_ids: parallel list — the chain_id to treat as binder for
            each CIF. If None, defaults to `select_smallest_protein_chain`.
        cache: if True (default), parsed structures are cached in memory after
            the first access. Set False for very large sets to save RAM.
    """

    def __init__(
        self,
        cif_paths: list[Union[str, Path]],
        binder_chain_ids: Optional[list[str]] = None,
        cache: bool = True,
    ) -> None:
        self.cif_paths = [str(p) for p in cif_paths]
        self.binder_chain_ids = binder_chain_ids
        if self.binder_chain_ids is not None and len(self.binder_chain_ids) != len(self.cif_paths):
            raise ValueError(
                f"binder_chain_ids length {len(self.binder_chain_ids)} != "
                f"cif_paths length {len(self.cif_paths)}"
            )
        self._cache_enabled = cache
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.cif_paths)

    def __getitem__(self, idx: int):
        if self._cache_enabled and idx in self._cache:
            return self._cache[idx]

        cif_path = self.cif_paths[idx]
        logger.info(f"CifFileProvider: parsing {cif_path}")

        # 1. Parse CIF → bioassembly_dict with atom_array + token_array.
        indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif=cif_path, pdb_cluster_file=None,
        )
        if "atom_array" not in bioassembly_dict or "token_array" not in bioassembly_dict:
            raise RuntimeError(f"Failed to parse CIF: {cif_path}")

        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict["token_array"]

        # 2. Featurize with Protenix's base Featurizer (restype 32-channel, ref,
        #    bonds, masks, etc.). Dummy MSA + template.
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=True,
            lig_atom_rename=False,
        )
        feature_dict = feat.get_all_input_features()
        label_dict = feat.get_labels()

        feature_dict = make_dummy_feature(
            features_dict=feature_dict, dummy_feats=["msa", "template"],
        )
        feature_dict = data_type_transform(feat_or_label_dict=feature_dict)
        label_dict = data_type_transform(feat_or_label_dict=label_dict)
        feature_dict["is_distillation"] = torch.tensor([False])

        # 3. Build the binder selector.
        if self.binder_chain_ids is not None:
            chain_id = self.binder_chain_ids[idx]
            binder_selector = lambda _aa, _c=chain_id: _c
        else:
            def _smallest(aa):
                is_prot = aa.mol_type == "protein"
                chains, counts = np.unique(aa.chain_id[is_prot], return_counts=True)
                if len(chains) == 0:
                    raise ValueError(f"No protein chains in {self.cif_paths[idx]}")
                return str(chains[np.argmin(counts)])
            binder_selector = _smallest

        result = (atom_array, token_array, feature_dict, label_dict, binder_selector)
        if self._cache_enabled:
            self._cache[idx] = result
        return result
