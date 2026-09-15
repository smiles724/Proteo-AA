"""Load the canonical Proteo-AA side-chain metric implementations.

The side-chain metrics reported for this pipeline are deliberately *not*
reimplemented here. They are the same functions used for the earlier Proteo-AA
side-chain work -- ``packing_metrics``/``summarize_metrics`` (symmetry-aware
RMSD, chi recovery, covalent bond errors) and ``sidechain_lddt`` -- so numbers
are directly comparable to those runs instead of merely similar to them.

Importing them is slightly awkward: ``pxdesign_train/__init__.py`` pulls in the
whole PXDesign/Protenix training model stack, which this pipeline does not need
and cannot construct. So the parent packages are stubbed with their real
``__path__`` and the leaf modules imported directly. The metric modules
themselves only need ``torch`` plus ``protenix.data.constants``, which this repo
already vendors as a submodule.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib
import os
import subprocess
import sys
import types

# The Proteo-AA checkout holding the metric implementations.
DEFAULT_ROOT = Path("/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases")
ENV_VAR = "PROTEOAA_METRICS_ROOT"
MODULES = ("metrics", "lddt", "frames", "instantiate")


@dataclass
class CanonicalMetrics:
    """The upstream metric functions, plus provenance for the report."""
    metrics: object
    lddt: object
    frames: object
    instantiate: object
    root: Path
    revision: str

    @property
    def packing_metrics(self):
        return self.metrics.packing_metrics

    @property
    def summarize_metrics(self):
        return self.metrics.summarize_metrics

    @property
    def sidechain_lddt(self):
        return self.lddt.sidechain_lddt

    def record(self):
        return dict(source="proteo-aa/pxdesign_train.sidechain", root=str(self.root),
                    revision=self.revision, modules=list(MODULES),
                    reimplemented=False)


def resolve_root(root=None):
    return Path(root or os.environ.get(ENV_VAR) or DEFAULT_ROOT).resolve()


def load(root=None):
    """Import the metric modules from ``root`` without running the heavy parents."""
    root = resolve_root(root)
    package = root / "pxdesign_train"
    if not (package / "sidechain" / "metrics.py").is_file():
        raise ValueError(
            f"No Proteo-AA side-chain metrics under {package}. Point {ENV_VAR} at a "
            "Proteo-AA checkout containing pxdesign_train/sidechain/metrics.py.")
    # protenix.data.constants is the metrics' only external dependency.
    for path in (root, Path(__file__).resolve().parents[2] / "Protenix"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    for name, directory in (("pxdesign_train", package),
                            ("pxdesign_train.sidechain", package / "sidechain")):
        existing = sys.modules.get(name)
        if existing is None or not getattr(existing, "__path__", None):
            stub = types.ModuleType(name)
            stub.__path__ = [str(directory)]
            sys.modules[name] = stub

    loaded = {name: importlib.import_module(f"pxdesign_train.sidechain.{name}")
              for name in MODULES}
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True).strip()
    except Exception:
        revision = "unknown"
    return CanonicalMetrics(root=root, revision=revision, **loaded)
