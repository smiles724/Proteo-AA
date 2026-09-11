"""An unparseable PDB entry must be skipped, not kill a DataLoader worker.

Upstream parsing and tokenisation fail on individual entries for reasons that
belong to the file. Measured: PINDER `2atc__A3_P0A786--2atc__B3_P0A7F3` has an
atom with element "X", and Protenix's tokenizer raises
`ValueError: Unknown atom element: X` out of `get_data_from_mmcif`. That bare
ValueError, and the sibling `RuntimeError: Failed to parse CIF`, carried no
retryable prefix, so `DesignSourceDataset.__getitem__` re-raised them out of
worker 0 and ended Stage IV job 113955 at step 300.

These tests pin the contract that makes such an entry a skip: the provider
re-tags the failure with the `CifProvider:` prefix and chains the cause, and
the retry loop absorbs that prefix.
"""
from __future__ import annotations

import sys
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))


def _provider(monkeypatch, raiser):
    pytest.importorskip("protenix")
    from pxdesign_train.runner import cif_provider as module

    monkeypatch.setattr(
        module.DataPipeline, "get_data_from_mmcif", staticmethod(raiser)
    )
    return module.CifFileProvider(cif_paths=["/nonexistent/2atc.cif"], cache=False)


def test_upstream_tokenizer_error_becomes_a_retryable_rejection(monkeypatch):
    def _raise(**_kwargs):
        raise ValueError("Unknown atom element: X")

    provider = _provider(monkeypatch, _raise)

    with pytest.raises(ValueError) as excinfo:
        provider[0]

    message = str(excinfo.value)
    assert message.startswith("CifProvider:"), message
    assert "Unknown atom element: X" in message
    assert "2atc.cif" in message
    # The original stays reachable; re-tagging must not discard the diagnosis.
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_missing_atom_array_is_retryable_not_a_runtime_error(monkeypatch):
    provider = _provider(monkeypatch, lambda **_kwargs: ([], {}))

    with pytest.raises(ValueError) as excinfo:
        provider[0]

    assert str(excinfo.value).startswith("CifProvider:")
    assert "atom_array/token_array" in str(excinfo.value)


def test_retry_loop_absorbs_the_cif_provider_prefix(monkeypatch):
    """The prefix is only useful if `__getitem__` actually catches it."""
    from pxdesign_train.runner.data import DesignSourceDataset

    class _Provider:
        def __len__(self):
            return 50

        def __getitem__(self, idx):
            return ("aa", "ta", {}, {}, lambda _a: "B")

    ds = DesignSourceDataset(
        provider=_Provider(), source_name="s", max_crop_retries=4
    )

    calls = {"n": 0}

    def _fail_once_then_work(idx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(
                "CifProvider: ValueError parsing /x/2atc.cif: Unknown atom element: X"
            )
        return {"ok": True}

    monkeypatch.setattr(ds, "_get_one", _fail_once_then_work)

    assert ds[0] == {"ok": True}
    assert calls["n"] == 2, "the rejection should have been retried, not raised"
