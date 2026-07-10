"""
Binder-aware spatial crop for PXDesign-d training.

The PXDesign technical report (p. 24) trains at crop size **640 residues**.
Protenix ships three crop strategies in `protenix/utils/cropping.py`
(`ContiguousCropping`, `SpatialCropping`, `SpatialInterfaceCropping`), but none
of them are quite right for design:

  - `ContiguousCropping` can split the binder mid-chain — fatal for design,
    since the model must see the whole binder it's supposed to denoise.
  - `SpatialCropping` and `SpatialInterfaceCropping` pick a *random* reference
    token from the reference chain and grow outward. For design we know
    which residues are the binder, so a random pick on the binder chain is
    wasteful: we'd often miss part of the binder.

This module provides `DesignCropper`, a variant that:

  1. Identifies all binder tokens (via chain_id or per-atom mask).
  2. Keeps them all (asserts `binder_size <= crop_size`; binders in PXDesign
     are < 200 residues, well under the 640 budget).
  3. Fills the remaining budget with **target tokens nearest any binder
     centre atom**, ranked ascending by Euclidean distance.
  4. Returns cropped `AtomArray`, `TokenArray`, and the new binder atom mask
     (atom indices change after cropping; the mask must be rebuilt).

We reuse Protenix's `CropData.select_by_token_indices` for the actual array
slicing — it knows how to renumber `token.atom_indices` and `centre_atom_index`
after a crop, which is fiddly to redo correctly.
"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from scipy.spatial.distance import cdist

from protenix.utils.cropping import CropData


@dataclass
class CropResult:
    """Output of `DesignCropper.crop`. All fields are post-crop."""

    atom_array: object  # biotite AtomArray
    token_array: object  # protenix TokenArray
    binder_atom_mask: np.ndarray  # [N_atom_post] boolean
    binder_token_mask: np.ndarray  # [N_token_post] boolean
    n_binder_tokens: int
    n_target_tokens: int
    original_atom_indices: np.ndarray  # [N_atom_post], indices into pre-crop AtomArray
    original_token_indices: np.ndarray  # [N_token_post], indices into pre-crop TokenArray


class DesignCropper:
    """Binder-aware crop to a fixed total token budget.

    Args:
        crop_size: total token budget (640 per PXDesign report).
        max_binder_fraction: safety cap. If the binder takes up more than this
            fraction of the crop, raise — usually means the caller mis-selected
            a huge chain as the binder.
    """

    def __init__(
        self,
        crop_size: int = 640,
        max_binder_fraction: float = 0.5,
    ) -> None:
        if crop_size < 2:
            raise ValueError(f"crop_size must be >= 2, got {crop_size}")
        self.crop_size = crop_size
        self.max_binder_fraction = max_binder_fraction

    def crop(
        self,
        atom_array,
        token_array,
        binder_chain_id: Optional[str] = None,
        binder_atom_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> CropResult:
        """Apply the binder-anchored crop.

        Exactly one of `binder_chain_id` or `binder_atom_mask` should be given,
        mirroring `DesignSelection`. The mask refers to the *uncropped* atom
        array; the result carries the rebuilt mask for the cropped array.

        Returns: `CropResult` with the cropped arrays and the new binder masks.
        """
        binder_atoms = self._resolve_binder_atoms(
            atom_array, binder_chain_id, binder_atom_mask
        )
        if not binder_atoms.any():
            raise ValueError("DesignCropper: binder selection is empty")

        binder_token_indices = self._binder_token_indices(token_array, atom_array, binder_atoms)
        n_binder = len(binder_token_indices)
        total_tokens = len(token_array)

        if n_binder > self.crop_size:
            raise ValueError(
                f"DesignCropper: binder has {n_binder} tokens but crop_size={self.crop_size}. "
                "Increase crop_size or shrink the binder selection."
            )
        if n_binder > int(self.crop_size * self.max_binder_fraction):
            raise ValueError(
                f"DesignCropper: binder is {n_binder} tokens — more than "
                f"{self.max_binder_fraction:.0%} of crop_size={self.crop_size}. "
                "If this is intentional, raise `max_binder_fraction`."
            )

        if total_tokens <= self.crop_size:
            # Whole complex fits; no need to drop anything.
            selected_token_indices = torch.arange(total_tokens, dtype=torch.long)
        else:
            target_indices = self._rank_target_tokens_by_proximity(
                atom_array=atom_array,
                token_array=token_array,
                binder_token_indices=binder_token_indices,
            )
            n_target_to_keep = self.crop_size - n_binder
            kept_target = target_indices[:n_target_to_keep]
            # Sort so cropped token order follows original sequence order.
            selected = np.concatenate([binder_token_indices, kept_target])
            selected.sort()
            selected_token_indices = torch.from_numpy(selected).long()

        original_token_indices = selected_token_indices.detach().cpu().numpy()
        original_atom_indices = []
        for token_idx in original_token_indices:
            original_atom_indices.extend(token_array[int(token_idx)].atom_indices)
        original_atom_indices = np.asarray(original_atom_indices, dtype=np.int64)

        cropped_token_array, cropped_atom_array = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=selected_token_indices,
        )
        if len(cropped_atom_array) != len(original_atom_indices):
            raise AssertionError(
                "DesignCropper internal error: original atom index count does not "
                f"match cropped AtomArray length ({len(original_atom_indices)} vs "
                f"{len(cropped_atom_array)})"
            )

        # Rebuild masks on the cropped array. Cropping preserves
        # `cropped_atom_array.chain_id` annotations, so we can recompute the
        # binder atom mask the same way we got the original one.
        new_binder_atom_mask = self._resolve_binder_atoms(
            cropped_atom_array, binder_chain_id, None,
        ) if binder_chain_id is not None else self._rebuild_atom_mask_from_residues(
            atom_array, cropped_atom_array, binder_atoms,
        )

        rep_atom_mask = cropped_atom_array.distogram_rep_atom_mask.astype(bool)
        new_binder_token_mask = new_binder_atom_mask[rep_atom_mask]

        return CropResult(
            atom_array=cropped_atom_array,
            token_array=cropped_token_array,
            binder_atom_mask=new_binder_atom_mask,
            binder_token_mask=new_binder_token_mask,
            n_binder_tokens=int(new_binder_token_mask.sum()),
            n_target_tokens=int((~new_binder_token_mask).sum()),
            original_atom_indices=original_atom_indices,
            original_token_indices=original_token_indices,
        )

    # ----- internals -----

    @staticmethod
    def _resolve_binder_atoms(
        atom_array,
        binder_chain_id: Optional[str],
        binder_atom_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> np.ndarray:
        n_set = int(binder_chain_id is not None) + int(binder_atom_mask is not None)
        if n_set != 1:
            raise ValueError(
                "DesignCropper: pass exactly one of binder_chain_id or binder_atom_mask"
            )
        if binder_chain_id is not None:
            return np.asarray(atom_array.chain_id == binder_chain_id, dtype=bool)
        mask = (
            binder_atom_mask.cpu().numpy()
            if isinstance(binder_atom_mask, torch.Tensor)
            else np.asarray(binder_atom_mask, dtype=bool)
        )
        if mask.shape != (len(atom_array),):
            raise ValueError(
                f"binder_atom_mask shape {mask.shape} != atom_array length {len(atom_array)}"
            )
        return mask

    @staticmethod
    def _binder_token_indices(token_array, atom_array, binder_atoms: np.ndarray) -> np.ndarray:
        """A token is part of the binder iff its centre atom is a binder atom."""
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        return np.where(binder_atoms[centre_atom_indices])[0]

    @staticmethod
    def _rank_target_tokens_by_proximity(
        atom_array,
        token_array,
        binder_token_indices: np.ndarray,
    ) -> np.ndarray:
        """Return target token indices sorted ascending by min-distance to any binder.

        Distance is centre-atom to centre-atom; matches Protenix's spatial
        cropper. We tie-break by token index to keep results deterministic
        across runs given the same input.
        """
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_coords = atom_array.coord[centre_atom_indices]  # [N_token, 3]

        n_token = len(token_array)
        is_binder = np.zeros(n_token, dtype=bool)
        is_binder[binder_token_indices] = True

        binder_coords = centre_coords[is_binder]
        target_coords = centre_coords[~is_binder]
        target_indices = np.where(~is_binder)[0]

        if len(binder_coords) == 0 or len(target_coords) == 0:
            return target_indices  # nothing to rank

        # Cdist returns [N_target, N_binder]; take min over the binder axis.
        d = cdist(target_coords, binder_coords, metric="euclidean")
        min_d = d.min(axis=1)

        # Stable sort with a tiny index-based tiebreaker (same as Protenix's
        # `noise_break_tie` in `get_spatial_crop_index`).
        tiebreak = np.arange(len(target_indices)) * 1e-6
        order = np.argsort(min_d + tiebreak, kind="stable")
        return target_indices[order]

    @staticmethod
    def _rebuild_atom_mask_from_residues(
        original_atom_array,
        cropped_atom_array,
        original_binder_atoms: np.ndarray,
    ) -> np.ndarray:
        """If the binder was specified by atom mask (not chain_id), rebuild the
        equivalent mask on the cropped array.

        We use (chain_id, res_id) pairs as the residue identity. This is robust
        to atom reordering but assumes the binder selection respects whole
        residues — which is the only sensible binder definition anyway.
        """
        binder_residue_keys = set(
            (cid, rid) for cid, rid in zip(
                original_atom_array.chain_id[original_binder_atoms],
                original_atom_array.res_id[original_binder_atoms],
            )
        )
        return np.array(
            [
                (cid, rid) in binder_residue_keys
                for cid, rid in zip(cropped_atom_array.chain_id, cropped_atom_array.res_id)
            ],
            dtype=bool,
        )
