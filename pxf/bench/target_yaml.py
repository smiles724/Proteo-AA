"""One binder length per PXDesign run, from a prepared multi-length config.

A tiny module with NO import side effects, and that is the point. This lived
in `scripts/cache_binder_backbones.py`, and the integrated matrix reached it
by exec'ing that file -- which at import inserts `scripts/` on `sys.path` and
pulls in `_bootstrap`, whose whole job is to front the repo's vendored
PXDesign and Protenix. Doing that inside a process running under the OFFICIAL
runtime shadows the pristine checkout with the patched v2.0.0 one, and the
next PXDesign call dies with `KeyError: 'd_lm'` -- exactly the pairing
`pxf/official/require.py` exists to refuse, reintroduced through the back
door by a helper import.

So the helper lives here, importable from either runtime, and both callers
use it.
"""

from __future__ import annotations

from pathlib import Path


def write_single_length_yaml(target_config, length: int, out) -> Path:
    """PXDesign takes one binder length per run; prepared configs hold a grid."""
    import yaml

    target_config, out = Path(target_config), Path(out)
    payload = dict(yaml.safe_load(target_config.read_text()))
    payload.pop("binder_lengths", None)
    payload["binder_length"] = int(length)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out
