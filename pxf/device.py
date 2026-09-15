"""Device selection that survives a GPU the local torch build cannot drive.

``torch.cuda.is_available()`` only reports that a driver and device exist, not
that the installed torch has kernels for that device's compute capability. On a
mixed-GPU cluster the two diverge: a build compiled through ``sm_90`` sees an
``sm_100`` card, answers "available", and then fails on the first kernel launch
with "no kernel image is available for execution on the device" -- after the
model has already been constructed and moved.

:func:`select_device` settles the question by actually launching a kernel.
"""
import os
import warnings
import torch

_PROBE_CACHE = {}


def cuda_is_usable(index=0):
    """True only if a real kernel launch on this device succeeds."""
    if index in _PROBE_CACHE:
        return _PROBE_CACHE[index]
    usable = False
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe = torch.zeros(8, device=f"cuda:{index}")
                torch.cuda.synchronize(index)
                usable = bool((probe + 1).sum().item() == 8)
        except Exception:
            usable = False
    _PROBE_CACHE[index] = usable
    return usable


def unsupported_capability(index=0):
    """``(device_name, capability, supported)`` when torch lacks kernels for the GPU."""
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        return None
    supported = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    if f"sm_{major}{minor}" in supported:
        return None
    return (torch.cuda.get_device_name(index), f"sm_{major}{minor}", supported)


def select_device(requested=None):
    """Resolve a device, falling back to CPU when CUDA is present but unusable.

    An explicit request is honoured as given -- if you ask for ``cuda`` you get
    the real error rather than a silent CPU run. ``PXF_DEVICE`` overrides the
    default. Only the default path probes and falls back.
    """
    requested = requested or os.environ.get("PXF_DEVICE")
    if requested:
        return torch.device(requested)
    if cuda_is_usable(0):
        return torch.device("cuda:0")
    mismatch = unsupported_capability(0)
    if mismatch:
        name, capability, supported = mismatch
        warnings.warn(
            f"Falling back to CPU: this torch build ({torch.__version__}) has no kernels "
            f"for {name} ({capability}); it supports {', '.join(supported)}. Install a "
            f"matching torch, move to a supported GPU, or set PXF_DEVICE=cuda to insist.",
            RuntimeWarning, stacklevel=2)
    return torch.device("cpu")
