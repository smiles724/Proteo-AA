"""Load Proteo-AA's ``pxdesign_train`` modules without running its heavy parents.

PXDesign's official inference runner designs a binder against a target and
cannot generate or denoise a monomer, so the backbone is driven through
Proteo-AA's ``pxdesign_train`` instead: its featurizer accepts an arbitrary
structure, scrubs the design region to backbone, and hands back the feature dict
PXDesign's diffusion module consumes.

Importing it needs the same care as the metrics in :mod:`pxf.eval.canonical`:
``pxdesign_train/__init__.py`` builds the whole training model, which is neither
needed nor constructible here, so the parent packages are stubbed with their real
``__path__`` and the leaves imported directly.
"""

import importlib
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "PROTEOAA_ROOT"
# Searched in order; these worktrees get reorganized (they have already moved
# once, from ~/ into ~/"Proteo-AA old"/), so a single hardcoded path breaks.
CANDIDATE_ROOTS = (
    Path("/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-sc-adaptation-phases"),
    Path("/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases"),
    Path("/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"),
    Path("/hai/users/y/f/yfsun/Proteo-AA"),
)
MARKER = Path("pxdesign_train/runner/cif_provider.py")
SUBPACKAGES = ("configs", "runner", "data", "sidechain", "eval")


def default_root():
    for candidate in CANDIDATE_ROOTS:
        if (candidate / MARKER).is_file():
            return candidate
    return CANDIDATE_ROOTS[0]


def resolve_root(root=None):
    return Path(root or os.environ.get(ENV_VAR) or default_root()).resolve()


@dataclass
class ProteoAA:
    """The pxdesign_train pieces the backbone driver needs, plus provenance."""

    root: Path
    revision: str
    cif_provider: object
    data: object
    configs: object

    @property
    def CifFileProvider(self):
        return self.cif_provider.CifFileProvider

    @property
    def DesignSourceDataset(self):
        return self.data.DesignSourceDataset

    def record(self):
        return dict(
            source="proteo-aa/pxdesign_train",
            root=str(self.root),
            revision=self.revision,
            reimplemented=False,
            reason="PXDesign's inference runner cannot denoise a monomer",
        )


def load(root=None):
    """Import ``pxdesign_train``'s leaves from ``root``."""
    root = resolve_root(root)
    package = root / "pxdesign_train"
    if not (root / MARKER).is_file():
        raise ValueError(
            f"No pxdesign_train under {package}. Point {ENV_VAR} at a Proteo-AA "
            f"checkout containing {MARKER}."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for name, directory in [("pxdesign_train", package)] + [
        (f"pxdesign_train.{sub}", package / sub) for sub in SUBPACKAGES
    ]:
        existing = sys.modules.get(name)
        if existing is None or not getattr(existing, "__path__", None):
            stub = types.ModuleType(name)
            stub.__path__ = [str(directory)]
            sys.modules[name] = stub
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        revision = "unknown"
    return ProteoAA(
        root=root,
        revision=revision,
        cif_provider=importlib.import_module("pxdesign_train.runner.cif_provider"),
        data=importlib.import_module("pxdesign_train.runner.data"),
        configs=importlib.import_module("pxdesign_train.configs.configs_train"),
    )
