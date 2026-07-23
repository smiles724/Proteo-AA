"""
Per-item design featurization layer for training datasets.

`DesignSourceDataset` wraps an underlying "complex provider" (anything indexable
that returns a real protein complex) with the PXDesign-specific transformations:

    provider[i] -> (atom_array, token_array, feature_dict, label_dict)
              |
              v
    crop to crop_size (binder kept whole)
              |
              v
    design-featurize (mark binder as xpb, build conditional_templ, hotspot, etc.)
              |
              v
    return {input_feature_dict, label_dict, binder_token_mask, source_name}

Providers are *up to the caller*. For Protenix's `BaseSingleDataset` you'd
write a small `ProtenixProviderAdapter` that extracts the four objects from
its returned dict. For an AFDB monomer reader you'd write a different
adapter. The trainer doesn't care which.

Each provider also needs to tell the cropper which residues are the binder:
  - For PPI complexes: usually a designated chain (`binder_chain_id`).
  - For monomers in the distillation pool: the *whole monomer* is the binder.
The provider returns this via `binder_selector(atom_array)` — a callable that
takes the AtomArray and returns either a chain_id string or a per-atom mask.

This keeps each source's binder-selection policy at the source layer, not
the featurizer layer.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from pxdesign_train.data.cropper import DesignCropper
from pxdesign_train.data.featurizer import DesignFeaturizer, DesignSelection


BinderSelector = Callable[[Any], Union[str, np.ndarray]]


class ComplexProvider(Protocol):
    """Minimum interface the trainer needs from each data source.

    Implementers return one *uncropped, unfeaturized* protein complex per
    index, plus a hint about which residues should be treated as the binder.

    Real-data providers (Protenix `BaseSingleDataset`, an AFDB shard reader,
    etc.) wrap their native data format with this interface. Tests can use
    trivial in-memory providers.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[
        Any,                       # biotite AtomArray (Protenix-annotated)
        Any,                       # protenix TokenArray
        dict[str, torch.Tensor],   # base feature_dict (Protenix-style)
        dict[str, torch.Tensor],   # base label_dict (coordinate + coordinate_mask)
        BinderSelector,            # how to pick the binder on this complex
    ]: ...


@dataclass
class DesignSourceDataset(Dataset):
    """Adapter from a `ComplexProvider` to a featurized training-ready dataset.

    Args:
        provider: the underlying complex provider.
        source_name: tag used by the trainer for logging / curriculum sampling.
        crop_size: token budget for `DesignCropper`. Per the report this is 640.
        max_binder_fraction: maximum fraction of the crop that the binder may
            occupy before the cropper rejects the example.
        hotspot_radius / hotspot_max_frac / hotspot_force_zero_prob: forwarded
            to `DesignSelection`. See `featurizer.py` for semantics.
        aa_mask_mode / aa_mask_prob / aa_mask_min_prob / aa_mask_max_prob:
            forwarded to `DesignSelection` for all-mask or partial
            residue identity corruption.
        max_crop_retries: if a sampled item cannot be design-cropped, try this
            many following provider items before raising.
        seed: base seed; combined with index per call to keep the RNG
            deterministic for a given (epoch, index) without per-epoch state.
    """

    provider: ComplexProvider
    source_name: str
    crop_size: int = 640
    max_binder_fraction: float = 0.5
    hotspot_radius: float = 8.0
    hotspot_max_frac: float = 0.5
    hotspot_force_zero_prob: float = 0.2
    aa_mask_mode: str = "all"
    aa_mask_prob: float = 1.0
    aa_mask_min_prob: float = 0.0
    aa_mask_max_prob: float = 1.0
    compute_sidechain: bool = False
    backbone_only_binder: bool = False
    # NOTE: no prior default existed anywhere for this; 8 is a chosen value.
    # Override at construction if a different retry budget is wanted.
    max_crop_retries: int = 8
    seed: int = 0
    _cropper: DesignCropper = field(init=False)

    def __post_init__(self) -> None:
        self._cropper = DesignCropper(
            crop_size=self.crop_size,
            max_binder_fraction=self.max_binder_fraction,
        )
        # P3 leakage coupling (SAFETY, not a nicety): `backbone_only_binder` is
        # what makes the featurizer run `_scrub_design_sidechain_coords` — the P1
        # leakage fix that collapses design-region side-chain coords onto CA so the
        # backbone denoiser never sees ground-truth side-chain geometry. Whenever we
        # compute side-chain targets we MUST also scrub, otherwise the scrub silently
        # does not fire and GT side-chain geometry leaks into the backbone input.
        if self.compute_sidechain and not self.backbone_only_binder:
            self.backbone_only_binder = True

    def __len__(self) -> int:
        return len(self.provider)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self.provider)
        retries = max(1, min(int(self.max_crop_retries), n))
        last_crop_error: Optional[ValueError] = None

        for attempt in range(retries):
            local_idx = (idx + attempt) % n
            try:
                return self._get_one(local_idx)
            except ValueError as exc:
                if "DesignCropper:" not in str(exc):
                    raise
                last_crop_error = exc

        raise ValueError(
            f"DesignSourceDataset: failed to find a crop-valid example after "
            f"{retries} attempts starting at index {idx}. Last crop error: "
            f"{last_crop_error}"
        )

    def _get_one(self, idx: int) -> dict[str, Any]:
        atom_array, token_array, feat, label, binder_selector = self.provider[idx]

        # Resolve the binder selector into something the cropper accepts.
        sel = binder_selector(atom_array)
        if isinstance(sel, str):
            crop = self._cropper.crop(atom_array, token_array, binder_chain_id=sel)
        else:
            crop = self._cropper.crop(atom_array, token_array, binder_atom_mask=sel)

        # Slice base feature_dict and label_dict to the kept tokens / atoms.
        feat = _slice_feature_dict(feat, atom_array, token_array, crop)
        label = _slice_label_dict(label, atom_array, token_array, crop)

        # Apply design featurization on the cropped complex.
        rng = np.random.default_rng((self.seed + idx) % (2**32))
        selection = DesignSelection(
            binder_atom_mask=crop.binder_atom_mask,
            hotspot_radius=self.hotspot_radius,
            hotspot_max_frac=self.hotspot_max_frac,
            hotspot_force_zero_prob=self.hotspot_force_zero_prob,
            aa_mask_mode=self.aa_mask_mode,
            aa_mask_prob=self.aa_mask_prob,
            aa_mask_min_prob=self.aa_mask_min_prob,
            aa_mask_max_prob=self.aa_mask_max_prob,
            compute_sidechain=self.compute_sidechain,
            backbone_only_binder=self.backbone_only_binder,
            rng=rng,
        )
        new_feat, new_label, new_aa = DesignFeaturizer(selection).transform(
            crop.atom_array, feat, label,
        )

        return {
            "input_feature_dict": new_feat,
            "label_dict": new_label,
            "binder_token_mask": torch.from_numpy(crop.binder_token_mask),
            "source_name": self.source_name,
        }


def _slice_feature_dict(
    feat: dict[str, torch.Tensor],
    orig_atom_array,
    orig_token_array,
    crop,
) -> dict[str, torch.Tensor]:
    """Cut Protenix-style per-token / per-atom feature tensors to the crop.

    We need this because the provider's feature_dict was computed on the
    UNCROPPED arrays; after cropping the tensor sizes don't match. We compare
    by length to decide which axis to slice.

    A provider that featurizes *after* cropping (e.g. a future Protenix
    pipeline that runs the featurizer on the cropped arrays) can bypass this
    and pass `feature_dict={}` — the design featurizer will re-derive what it
    needs from the cropped AtomArray.
    """
    n_token_orig = len(orig_token_array)
    n_atom_orig = len(orig_atom_array)
    n_token_new = len(crop.token_array)
    n_atom_new = len(crop.atom_array)

    # Build token-index and atom-index masks from the cropped data.
    # The cropper kept tokens by global token index; for atoms it kept those
    # tokens' atoms. We rederive selection masks by matching residue (chain_id,
    # res_id) keys — robust to any reordering inside `select_by_token_indices`.
    kept_token_keys = set(
        (cid, rid) for cid, rid in zip(
            crop.atom_array.chain_id[crop.atom_array.distogram_rep_atom_mask.astype(bool)],
            crop.atom_array.res_id[crop.atom_array.distogram_rep_atom_mask.astype(bool)],
        )
    )
    orig_rep_mask = orig_atom_array.distogram_rep_atom_mask.astype(bool)
    orig_token_keys = list(zip(
        orig_atom_array.chain_id[orig_rep_mask],
        orig_atom_array.res_id[orig_rep_mask],
    ))
    token_keep_mask = np.array([k in kept_token_keys for k in orig_token_keys])

    atom_keep_mask = np.array([
        (cid, rid) in kept_token_keys
        for cid, rid in zip(orig_atom_array.chain_id, orig_atom_array.res_id)
    ])

    sliced: dict[str, torch.Tensor] = {}
    for k, v in feat.items():
        if not isinstance(v, torch.Tensor):
            sliced[k] = v
            continue
        if v.dim() == 0:
            sliced[k] = v
            continue
        # Crop EVERY axis whose length matches an original axis — not just the first.
        # The if/elif form stopped after axis 0, so a pair tensor [N_token, N_token, C]
        # kept its second token axis uncropped (that is the bug this test pins).
        tok_idx = torch.from_numpy(np.nonzero(token_keep_mask)[0])
        atm_idx = torch.from_numpy(np.nonzero(atom_keep_mask)[0])
        for dim in range(v.dim()):
            if v.shape[dim] == n_token_orig:
                v = torch.index_select(v, dim, tok_idx.to(v.device))
            elif n_atom_orig != n_token_orig and v.shape[dim] == n_atom_orig:
                v = torch.index_select(v, dim, atm_idx.to(v.device))
        # Else leave unchanged (could be a scalar / pair / extra dim that
        # the design featurizer either doesn't use or recomputes).
        sliced[k] = v

    # Ensure crop-index-valued features are consistent with the post-crop arrays.
    if "atom_to_token_idx" in feat:
        sliced["atom_to_token_idx"] = _tensor_like_index(
            _atom_to_token_idx_from_token_array(crop.token_array, n_atom_new),
            feat["atom_to_token_idx"],
        )
    if "token_index" in feat:
        sliced["token_index"] = torch.arange(
            n_token_new,
            dtype=feat["token_index"].dtype,
            device=feat["token_index"].device,
        )

    # ATOM-INDEX-VALUED features must have their VALUES remapped, not just their
    # rows sliced: after a crop the atom numbering changes, so an index into the
    # pre-crop atom axis silently points at the wrong atom (or out of range).
    # `sc_bb_atom_idx` [N_token, 4] holds atom indices of (N, CA, C, O);
    # `sc_token_center_idx` [N_token] holds each token's representative (CA) atom.
    #
    # In the training path (`DesignSourceDataset._get_one`) this key is NOT present
    # here: cropping happens BEFORE `DesignFeaturizer.transform`, so the featurizer
    # already resolves the indices against the CROPPED AtomArray and they need no
    # remap. This branch exists for any caller that featurizes first and crops after
    # (e.g. a cached/pre-featurized feature dict) — without it that path is a silent
    # wrong-atom gather. Rows whose atom was dropped by the crop become -1, which is
    # exactly what every downstream consumer already treats as "invalid".
    for _key in ("sc_bb_atom_idx", "sc_token_center_idx"):
        if _key in feat and isinstance(feat[_key], torch.Tensor):
            v = feat[_key]
            if v.shape[0] == n_token_orig:
                tok_idx = torch.from_numpy(np.nonzero(token_keep_mask)[0]).to(v.device)
                v = torch.index_select(v, 0, tok_idx)
                atom_old_to_new = _old_to_new_indexer(
                    np.nonzero(atom_keep_mask)[0], n_atom_orig,
                )
                sliced[_key] = _remap_index_values(v, atom_old_to_new)

    return sliced


def _slice_label_dict(label, orig_atom_array, orig_token_array, crop) -> dict[str, torch.Tensor]:
    n_atom_orig = len(orig_atom_array)
    n_token_orig = len(orig_token_array)
    if hasattr(crop, "original_atom_indices"):
        atom_indexer = np.asarray(crop.original_atom_indices, dtype=np.int64)
    else:
        kept_token_keys = set(
            (cid, rid) for cid, rid in zip(
                crop.atom_array.chain_id[crop.atom_array.distogram_rep_atom_mask.astype(bool)],
                crop.atom_array.res_id[crop.atom_array.distogram_rep_atom_mask.astype(bool)],
            )
        )
        atom_indexer = np.array([
            (cid, rid) in kept_token_keys
            for cid, rid in zip(orig_atom_array.chain_id, orig_atom_array.res_id)
        ])

    if hasattr(crop, "original_token_indices"):
        token_indexer = np.asarray(crop.original_token_indices, dtype=np.int64)
    else:
        rep_mask = orig_atom_array.distogram_rep_atom_mask.astype(bool)
        token_indexer = np.array([
            (cid, rid) in kept_token_keys
            for cid, rid in zip(
                orig_atom_array.chain_id[rep_mask],
                orig_atom_array.res_id[rep_mask],
            )
        ])

    atom_old_to_new = _old_to_new_indexer(atom_indexer, n_atom_orig)

    sliced = {}
    for k, v in label.items():
        if not isinstance(v, torch.Tensor):
            sliced[k] = v
            continue
        if v.dim() == 0:
            sliced[k] = v
            continue

        v = _slice_tensor_by_crop(
            v=v,
            token_indexer=token_indexer,
            atom_indexer=atom_indexer,
            n_token_orig=n_token_orig,
            n_atom_orig=n_atom_orig,
        )
        if k == "frame_atom_index":
            v = _remap_index_values(v, atom_old_to_new)
        sliced[k] = v
    return sliced


def _torch_index(indexer: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indexer, dtype=torch.long, device=device)


def _index_select(v: torch.Tensor, dim: int, indexer: np.ndarray) -> torch.Tensor:
    return v.index_select(dim, _torch_index(indexer, device=v.device))


def _slice_tensor_by_crop(
    *,
    v: torch.Tensor,
    token_indexer: np.ndarray,
    atom_indexer: np.ndarray,
    n_token_orig: int,
    n_atom_orig: int,
) -> torch.Tensor:
    """Slice common Protenix tensor layouts after a design crop."""
    if v.dim() >= 2 and v.shape[0] == n_token_orig and v.shape[1] == n_token_orig:
        return _index_select(_index_select(v, 0, token_indexer), 1, token_indexer)
    if v.dim() >= 2 and v.shape[0] == n_atom_orig and v.shape[1] == n_atom_orig:
        return _index_select(_index_select(v, 0, atom_indexer), 1, atom_indexer)
    if v.dim() >= 4 and v.shape[1] == n_token_orig and v.shape[2] == n_token_orig:
        return _index_select(_index_select(v, 1, token_indexer), 2, token_indexer)
    if v.shape[0] == n_token_orig:
        return _index_select(v, 0, token_indexer)
    if v.shape[0] == n_atom_orig:
        return _index_select(v, 0, atom_indexer)
    if v.dim() >= 2 and v.shape[1] == n_token_orig:
        return _index_select(v, 1, token_indexer)
    if v.dim() >= 2 and v.shape[1] == n_atom_orig:
        return _index_select(v, 1, atom_indexer)
    return v


def _old_to_new_indexer(indexer: np.ndarray, n_orig: int) -> np.ndarray:
    old_to_new = np.full(n_orig, -1, dtype=np.int64)
    if indexer.dtype == bool:
        old_indices = np.flatnonzero(indexer)
    else:
        old_indices = np.asarray(indexer, dtype=np.int64)
    old_to_new[old_indices] = np.arange(len(old_indices), dtype=np.int64)
    return old_to_new


def _remap_index_values(v: torch.Tensor, old_to_new: np.ndarray) -> torch.Tensor:
    out = v.clone()
    valid = out >= 0
    if not bool(valid.any()):
        return out
    old_to_new_t = torch.as_tensor(old_to_new, dtype=torch.long, device=out.device)
    old_values = out[valid].long()
    remapped = old_to_new_t[old_values]
    out[valid] = remapped.to(dtype=out.dtype)
    return out


def _atom_to_token_idx_from_token_array(token_array, n_atom: int) -> np.ndarray:
    atom_to_token = np.full(n_atom, -1, dtype=np.int64)
    for token_idx, token in enumerate(token_array):
        for atom_idx in token.atom_indices:
            atom_to_token[int(atom_idx)] = token_idx
    if np.any(atom_to_token < 0):
        raise ValueError("Cropped token_array does not cover every cropped atom")
    return atom_to_token


def _tensor_like_index(values: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)
