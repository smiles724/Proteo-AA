"""Regression tests for the 4 bugs that made `main` untrainable.

Three came in with commit 18b33c0 ("merge data pipeline"), which dropped
dataclass fields / a safety coupling from `DesignSourceDataset` and mis-pasted a
line into `_slice_feature_dict`. The fourth came in with 069645a ("New framing
losses"), which changed the model to emit global side-chain coords + the
predicted frame but never taught `trainer.py` to forward them to the loss.

Each test below fails on the unfixed code:
  BUG 1  TypeError (unexpected kwarg) / AttributeError (max_crop_retries)
  BUG 2  compute_sidechain=True leaves backbone_only_binder False -> P1 scrub
         never fires -> GT side-chain coords leak into the backbone denoiser
  BUG 3  NameError: name 'atom_indexer' is not defined
  BUG 4  loss sees sc_pred_global=None -> has_global_sc False -> the entire
         side-chain coordinate loss block is skipped, sc_local == 0.0 forever
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from pxdesign_train.runner import data as data_mod
from pxdesign_train.runner.data import DesignSourceDataset
from pxdesign_train.runner.trainer import PXDesignTrainer


class _StubProvider:
    """Minimal ComplexProvider stand-in.

    `__getitem__` always raises the cropper's ValueError so we can observe the
    retry loop (and hence `max_crop_retries`) without building real structures.
    """

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        self.calls.append(idx)
        raise ValueError("DesignCropper: stub always rejects")


# --------------------------------------------------------------------------
# BUG 1 — restored dataclass fields
# --------------------------------------------------------------------------

def test_construct_with_sidechain_kwargs():
    """BUG 1: these kwargs are used in `_get_one` but had no field declaration."""
    ds = DesignSourceDataset(
        provider=_StubProvider(),
        source_name="stub",
        compute_sidechain=True,
        backbone_only_binder=True,
    )
    assert ds.compute_sidechain is True
    assert ds.backbone_only_binder is True


def test_max_crop_retries_has_usable_default():
    """BUG 1: `__getitem__` reads self.max_crop_retries -> AttributeError before."""
    ds = DesignSourceDataset(provider=_StubProvider(), source_name="stub")
    assert isinstance(ds.max_crop_retries, int)
    assert ds.max_crop_retries >= 1


def test_max_crop_retries_is_actually_used_by_getitem():
    """The retry budget must reach the real `__getitem__` loop, not just exist.

    Unfixed: AttributeError. Fixed: the loop retries min(max_crop_retries, n)
    times and then raises the crop-exhausted ValueError.
    """
    provider = _StubProvider(n=4)
    ds = DesignSourceDataset(provider=provider, source_name="stub", max_crop_retries=3)

    with pytest.raises(ValueError, match="failed to find a crop-valid example"):
        ds[0]

    # Retried exactly `max_crop_retries` provider items (capped at len(provider)).
    assert provider.calls == [0, 1, 2]


def test_max_crop_retries_is_capped_by_provider_length():
    provider = _StubProvider(n=2)
    ds = DesignSourceDataset(provider=provider, source_name="stub", max_crop_retries=8)
    with pytest.raises(ValueError, match="failed to find a crop-valid example"):
        ds[0]
    assert provider.calls == [0, 1]


# --------------------------------------------------------------------------
# BUG 2 — P3 leakage auto-coupling
# --------------------------------------------------------------------------

def test_compute_sidechain_auto_sets_backbone_only_binder():
    """BUG 2 (SAFETY): backbone_only_binder is what triggers the featurizer's
    `_scrub_design_sidechain_coords` (the P1 leakage fix). If it stays False the
    scrub silently does not fire and GT side-chain geometry reaches the backbone
    denoiser. compute_sidechain must therefore force it on.
    """
    ds = DesignSourceDataset(
        provider=_StubProvider(),
        source_name="stub",
        compute_sidechain=True,
        # deliberately NOT passing backbone_only_binder
    )
    assert ds.backbone_only_binder is True, (
        "P3 coupling missing: compute_sidechain=True without backbone_only_binder "
        "means the P1 side-chain-coord scrub never runs -> GT side-chain leakage"
    )


def test_no_coupling_when_sidechain_off():
    """The coupling must not fire spuriously when we are not computing targets."""
    ds = DesignSourceDataset(provider=_StubProvider(), source_name="stub")
    assert ds.compute_sidechain is False
    assert ds.backbone_only_binder is False


def test_explicit_backbone_only_binder_is_preserved():
    ds = DesignSourceDataset(
        provider=_StubProvider(),
        source_name="stub",
        compute_sidechain=False,
        backbone_only_binder=True,
    )
    assert ds.backbone_only_binder is True


# --------------------------------------------------------------------------
# BUG 3 — orphan line in _slice_feature_dict
# --------------------------------------------------------------------------

class _FakeAtomArray:
    def __init__(self, chain_id, res_id, rep_mask):
        self.chain_id = np.asarray(chain_id)
        self.res_id = np.asarray(res_id)
        self.distogram_rep_atom_mask = np.asarray(rep_mask)

    def __len__(self) -> int:
        return len(self.chain_id)


class _FakeCrop:
    def __init__(self, atom_array, token_array):
        self.atom_array = atom_array
        self.token_array = token_array


def test_slice_feature_dict_has_no_orphan_indexer():
    """BUG 3: `atom_old_to_new = _old_to_new_indexer(atom_indexer, ...)` was
    mis-pasted from `_slice_label_dict`; `atom_indexer` is undefined in this
    scope, so any call raised NameError. (The identical line in
    `_slice_label_dict` is legitimate and must stay.)
    """
    # 3 residues x 2 atoms; the crop keeps residues 1 and 3.
    orig_atoms = _FakeAtomArray(
        chain_id=["A", "A", "A", "A", "A", "A"],
        res_id=[1, 1, 2, 2, 3, 3],
        rep_mask=[1, 0, 1, 0, 1, 0],
    )
    kept_atoms = _FakeAtomArray(
        chain_id=["A", "A", "A", "A"],
        res_id=[1, 1, 3, 3],
        rep_mask=[1, 0, 1, 0],
    )
    orig_tokens = [0, 1, 2]
    crop = _FakeCrop(kept_atoms, token_array=[0, 1])

    feat = {
        "per_token": torch.arange(3, dtype=torch.float32),          # [N_token]
        "per_atom": torch.arange(6, dtype=torch.float32),           # [N_atom]
        "scalar": torch.tensor(7.0),
        "not_a_tensor": "keepme",
    }

    out = data_mod._slice_feature_dict(feat, orig_atoms, orig_tokens, crop)

    assert torch.equal(out["per_token"], torch.tensor([0.0, 2.0]))
    assert torch.equal(out["per_atom"], torch.tensor([0.0, 1.0, 4.0, 5.0]))
    assert out["not_a_tensor"] == "keepme"


def test_slice_label_dict_keeps_its_legitimate_indexer():
    """Guard: the fix must not have deleted the *real* use in _slice_label_dict."""
    src = inspect.getsource(data_mod._slice_label_dict)
    assert "atom_old_to_new = _old_to_new_indexer(atom_indexer, n_atom_orig)" in src
    assert "_remap_index_values(v, atom_old_to_new)" in src


# --------------------------------------------------------------------------
# BUG 4 — trainer must forward the global side-chain kwargs to the loss
# --------------------------------------------------------------------------

class _SpyLoss:
    """Captures the kwargs `forward_loss` hands to the loss."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"loss": torch.tensor(0.0)}


class _FakeModel:
    """Emits exactly what the post-069645a model emits: GLOBAL side-chain coords
    plus the predicted frame (R, t). Notably it does NOT emit sc_pred_local.
    """

    def __init__(self) -> None:
        self.n_atom, self.n_res, self.n_sc = 4, 2, 3

    def __call__(self, input_feature_dict, label_dict, mode):
        return {
            "x_denoised": torch.zeros(1, 1, self.n_atom, 3),
            "x_gt_aug": torch.zeros(1, 1, self.n_atom, 3),
            "sigma": torch.ones(1, 1),
            "sc_pred_global": torch.zeros(1, self.n_res, self.n_sc, 3),
            "sc_frame_R": torch.eye(3).expand(1, self.n_res, 3, 3).clone(),
            "sc_frame_t": torch.zeros(1, self.n_res, 3),
            "sc_atom_mask": torch.ones(1, self.n_res, self.n_sc, dtype=torch.bool),
        }


def _make_trainer_with_spy() -> tuple[PXDesignTrainer, _SpyLoss]:
    """Real `PXDesignTrainer.forward_loss` on a stub model + spy loss.

    We bypass `__init__` (it would build the full Protenix model from configs);
    `forward_loss` only reads self.device / self.model / self.loss_fn, so this
    exercises the genuine code path — a real spy, not a source assertion.
    """
    trainer = object.__new__(PXDesignTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = _FakeModel()
    spy = _SpyLoss()
    trainer.loss_fn = spy
    return trainer, spy


def _make_batch(n_atom: int = 4, n_res: int = 2, n_sc: int = 3) -> dict:
    return {
        "input_feature_dict": {
            "distogram_rep_atom_mask": torch.ones(1, n_atom),
            "sc_gt_local": torch.zeros(1, n_res, n_sc, 3),
        },
        "label_dict": {"coordinate_mask": torch.ones(1, n_atom)},
    }


def test_trainer_forwards_global_sidechain_kwargs():
    """BUG 4: trainer passed only sc_pred_local (which the model no longer
    emits -> None), so the loss saw no side-chain prediction at all.
    """
    trainer, spy = _make_trainer_with_spy()
    trainer.forward_loss(_make_batch())

    for name in ("sc_pred_global", "sc_frame_R", "sc_frame_t"):
        assert name in spy.kwargs, f"trainer never passes {name} to the loss"
        assert spy.kwargs[name] is not None, f"trainer passed {name}=None"

    # backward compat: the legacy local kwarg is still forwarded
    assert "sc_pred_local" in spy.kwargs


def test_forwarded_kwargs_satisfy_loss_has_global_sc():
    """The forwarded kwargs must actually flip loss.py's `has_global_sc` gate.

    Mirrors the real condition in PXDesignLoss.forward:
        has_global_sc = (sc_pred_global is not None and sc_gt_local is not None
                         and sc_frame_R is not None and sc_frame_t is not None
                         and sc_atom_mask is not None)
    If any is None the whole side-chain coordinate block is skipped and
    sc_local is exactly 0.0 for the entire run.
    """
    trainer, spy = _make_trainer_with_spy()
    trainer.forward_loss(_make_batch())
    k = spy.kwargs

    has_global_sc = (
        k.get("sc_pred_global") is not None
        and k.get("sc_gt_local") is not None
        and k.get("sc_frame_R") is not None
        and k.get("sc_frame_t") is not None
        and k.get("sc_atom_mask") is not None
    )
    assert has_global_sc, (
        "loss.py would skip the side-chain coordinate loss entirely -> "
        "S_phi gets zero coordinate supervision"
    )


def test_loss_signature_accepts_the_forwarded_names():
    """Cheap guard that trainer and loss keep agreeing on the kwarg names."""
    from pxdesign_train.loss import PXDesignLoss

    params = inspect.signature(PXDesignLoss.forward).parameters
    for name in ("sc_pred_global", "sc_frame_R", "sc_frame_t", "sc_pred_local"):
        assert name in params, f"PXDesignLoss.forward lost the {name} kwarg"
