"""Device selection must not trust ``torch.cuda.is_available()`` alone.

This cluster mixes GPU generations. A torch build compiled through ``sm_90``
reports a newer card as "available" and then dies on the first kernel launch, so
the default path probes before committing.
"""
import pytest
import torch

from pxf import device as dev


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    dev._PROBE_CACHE.clear()
    yield
    dev._PROBE_CACHE.clear()


def test_explicit_request_is_honoured_verbatim():
    # Asking for a device means you want its real error, not a silent CPU run.
    assert dev.select_device("cpu") == torch.device("cpu")
    assert dev.select_device("cuda:3") == torch.device("cuda:3")


def test_environment_variable_sets_the_default(monkeypatch):
    monkeypatch.setenv("PXF_DEVICE", "cpu")
    assert dev.select_device() == torch.device("cpu")


def test_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("PXF_DEVICE", "cuda:7")
    assert dev.select_device("cpu") == torch.device("cpu")


def test_probe_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (calls.append(1), False)[1])
    assert dev.cuda_is_usable(0) is False
    assert dev.cuda_is_usable(0) is False
    assert len(calls) == 1


def test_no_cuda_means_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.select_device() == torch.device("cpu")
    assert dev.unsupported_capability(0) is None


def test_an_unusable_gpu_falls_back_with_an_explanation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA B200")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    monkeypatch.setattr(dev, "cuda_is_usable", lambda index=0: False)
    mismatch = dev.unsupported_capability(0)
    assert mismatch == ("NVIDIA B200", "sm_100", ["sm_80", "sm_90"])
    with pytest.warns(RuntimeWarning, match="no kernels for NVIDIA B200"):
        assert dev.select_device() == torch.device("cpu")


def test_a_supported_gpu_reports_no_mismatch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    assert dev.unsupported_capability(0) is None


def test_a_working_gpu_is_selected(monkeypatch):
    monkeypatch.setattr(dev, "cuda_is_usable", lambda index=0: True)
    assert dev.select_device() == torch.device("cuda:0")


def test_probe_swallows_a_failing_kernel_launch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution")

    monkeypatch.setattr(torch, "zeros", explode)
    assert dev.cuda_is_usable(0) is False
