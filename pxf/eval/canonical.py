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

import importlib
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

# Proteo-AA checkouts that may hold the metric implementations. Searched in
# order, because these worktrees get reorganized -- they have already moved once,
# from ~/ into ~/"Proteo-AA old"/ -- and a single hardcoded path silently breaks.
ENV_VAR = "PROTEOAA_METRICS_ROOT"
CANDIDATE_ROOTS = (
    Path("/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-sc-adaptation-phases"),
    Path("/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases"),
    Path("/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-sc-repair-arm-f"),
    Path("/hai/users/y/f/yfsun/Proteo-AA"),
)
METRICS_RELATIVE = Path("pxdesign_train/sidechain/metrics.py")


def default_root():
    """First candidate that actually carries the metric implementations."""
    for candidate in CANDIDATE_ROOTS:
        if (candidate / METRICS_RELATIVE).is_file():
            return candidate
    return CANDIDATE_ROOTS[0]


MODULES = ("metrics", "lddt", "frames", "instantiate")
# pxdesign_train.eval.uncond_metrics: the unconditional self-consistency harness
# (Ca and all-atom scRMSD with residue pairing, sample_id conventions, fasta IO).
UNCOND_MODULE = "eval.uncond_metrics"


@dataclass
class CanonicalMetrics:
    """The upstream metric functions, plus provenance for the report."""

    metrics: object
    lddt: object
    frames: object
    instantiate: object
    uncond: object
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
        return dict(
            source="proteo-aa/pxdesign_train.sidechain",
            root=str(self.root),
            revision=self.revision,
            modules=list(MODULES) + [UNCOND_MODULE],
            reimplemented=False,
        )


def resolve_root(root=None):
    return Path(root or os.environ.get(ENV_VAR) or default_root()).resolve()


def load(root=None):
    """Import the metric modules from ``root`` without running the heavy parents."""
    root = resolve_root(root)
    package = root / "pxdesign_train"
    if not (package / "sidechain" / "metrics.py").is_file():
        raise ValueError(
            f"No Proteo-AA side-chain metrics under {package}. Point {ENV_VAR} at a "
            "Proteo-AA checkout containing pxdesign_train/sidechain/metrics.py."
        )
    # The metrics need protenix.data.constants.ATOM14, which the Protenix this
    # repo pins (c3bfc36) provides -- the same revision Proteo-AA uses -- so the
    # vendored submodule serves them directly and no version juggling is needed.
    # Verify rather than assume: ATOM14 postdates v0.5.0+pxd, the revision
    # PXDesign's own installer names, so a repo that re-pinned there would have to
    # supply Proteo-AA's Protenix instead.
    repo_protenix = Path(__file__).resolve().parents[2] / "Protenix"
    for path in (root, repo_protenix, root / "Protenix"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from protenix.data.constants import ATOM14  # noqa: F401
    except ImportError as error:
        raise ValueError(
            "The side-chain metrics need protenix.data.constants.ATOM14, which the "
            f"Protenix on this path does not provide ({error}). ATOM14 postdates "
            "v0.5.0+pxd; use a Protenix at or after c3bfc36."
        ) from error

    for name, directory in (
        ("pxdesign_train", package),
        ("pxdesign_train.sidechain", package / "sidechain"),
    ):
        existing = sys.modules.get(name)
        if existing is None or not getattr(existing, "__path__", None):
            stub = types.ModuleType(name)
            stub.__path__ = [str(directory)]
            sys.modules[name] = stub

    stub = types.ModuleType("pxdesign_train.eval")
    stub.__path__ = [str(package / "eval")]
    sys.modules.setdefault("pxdesign_train.eval", stub)
    loaded = {
        name: importlib.import_module(f"pxdesign_train.sidechain.{name}")
        for name in MODULES
    }
    loaded["uncond"] = importlib.import_module("pxdesign_train.eval.uncond_metrics")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        revision = "unknown"
    return CanonicalMetrics(root=root, revision=revision, **loaded)
