"""Source and weight pinning for both components.

Nothing is re-hosted: PXDesign, Protenix and FaMPNN are submodule commit
pointers, and FaMPNN's weights ship inside its own repository. This module is the
single place that answers "exactly which code and which weights produced this
output", and it refuses to proceed when the answer would be wrong.

The only tolerated source modification is the checked-in PXDesign embedders
patch, which adapts PXDesign's ``InputFeatureEmbedder`` to the Protenix 2.0
``atom_attention_encoder`` signature. FaMPNN is required to be pristine.
"""
from pathlib import Path
import hashlib
import subprocess

PINNED = {
    "pxdesign": "f788441313c84c3074fe9596ac2433f96b15c763",
    # Protenix c3bfc36 (v2.0.0-11). NOTE: PXDesign's own install.sh pins
    # v0.5.0+pxd (d18aa1d), and PXDesign's *official inference runner* only works
    # against that one -- later revisions reorganize protenix.data and add
    # required cache args to DiffusionModule.forward. This repo pins c3bfc36
    # instead because it is the revision Proteo-AA's pxdesign_train and the
    # side-chain metrics require, and because PXDesign's inference API cannot
    # express monomer backbone generation anyway (it designs a binder against a
    # target). The consequence is that the backbone is driven through
    # pxdesign_train, not through pxdesign.runner.inference. See docs/backbone.md.
    "protenix": "c3bfc365b3e1341a11935eddfe7bfdc308092147",
    "fampnn": "aaf788b1502ad95d5c5a84455cfc53f2544f3b45",
}
LAYOUT = {
    "pxdesign": ("PXDesign", "pxdesign"),
    "protenix": ("Protenix", "protenix"),
    "fampnn": ("fampnn", "fampnn"),
}
# One recorded patch: PXDesign @ f788441 calls AtomAttentionEncoder with a
# feature dict, which Protenix c3bfc36 changed to positional arguments. Pure
# call-site adaptation, no behaviour change. (It is unnecessary against
# Protenix v0.5.0+pxd, which is why it disappears if that pin is ever adopted.)
ALLOWED_PATCHES = {"pxdesign": ("patches/pxdesign-embedders-protenix-2.0.patch",)}

# FaMPNN's published checkpoints, which ship inside the submodule under weights/.
# 0.0 Angstrom noise is what upstream's own packing example uses; the 0.3 variants
# are tuned for sequence design and mutation scoring, which this pipeline does not do.
FAMPNN_WEIGHTS = {
    "0.0": "fampnn_0_0.pt",
    "0.3": "fampnn_0_3.pt",
    "0.3-cath": "fampnn_0_3_cath.pt",
}
DEFAULT_FAMPNN_WEIGHTS = "0.0"
PXDESIGN_MODEL_NAME = "pxdesign_v0.1.0"


def repo_root():
    return Path(__file__).resolve().parents[1]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _is_exactly_patched(root, patches):
    """True when reverse-applying every recorded patch leaves the subtree clean.

    Stronger and less brittle than comparing diff text: it proves the working
    tree is HEAD plus exactly this patch set, regardless of how git happens to
    order or format a combined diff.
    """
    paths = [str(repo_root() / patch) for patch in patches]
    if any(not Path(path).is_file() for path in paths):
        return False
    result = subprocess.run(["git", "-C", str(root), "apply", "--reverse", "--check",
                             *paths], capture_output=True)
    return result.returncode == 0


def component_record(name, *, strict=True):
    """Revision, patch digest and path for one upstream; raises when drifted."""
    if name not in LAYOUT:
        raise ValueError(f"Unknown component {name!r}")
    directory, subtree = LAYOUT[name]
    root = repo_root() / directory
    if not (root / ".git").exists():
        raise ValueError(f"Submodule {directory} is not initialized; run scripts/setup.sh")
    revision = _git(root, "rev-parse", "HEAD")
    if strict and revision != PINNED[name]:
        raise ValueError(f"{name} is at {revision}, not the validated {PINNED[name]}")
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", subtree])
    if diff.strip():
        allowed = ALLOWED_PATCHES.get(name, ())
        if not allowed or not _is_exactly_patched(root, allowed):
            raise ValueError(
                f"Unrecorded modifications to {name} source. The working tree must be "
                f"HEAD plus exactly {list(allowed) or '<no patches>'}; keep upstream "
                "otherwise pristine and put compatibility fixes in pxf.compat-style "
                "runtime shims where possible.")
    return dict(component=name, revision=revision, path=str(root),
                patch_sha256=hashlib.sha256(diff).hexdigest(), patched=bool(diff.strip()))


def runtime_sources(*, strict=True, components=("pxdesign", "protenix", "fampnn")):
    """Verified record for every requested upstream, keyed by component name."""
    return {name: component_record(name, strict=strict) for name in components}


def fampnn_checkpoint(variant=DEFAULT_FAMPNN_WEIGHTS, *, root=None):
    """Resolve a FaMPNN weight variant to its path inside the submodule."""
    if variant not in FAMPNN_WEIGHTS:
        raise ValueError(f"Unknown FaMPNN weights {variant!r}; choose from {sorted(FAMPNN_WEIGHTS)}")
    base = Path(root) if root is not None else repo_root() / "fampnn" / "weights"
    path = base / FAMPNN_WEIGHTS[variant]
    if not path.is_file():
        raise ValueError(
            f"FaMPNN checkpoint not found: {path}. The weights ship inside the fampnn "
            "submodule; run scripts/setup.sh to initialize it.")
    return path


def weight_record(path, *, variant=None):
    """Identify a weight file by resolved path, size and digest."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Weight file not found: {resolved}")
    record = dict(path=str(resolved), sha256=file_sha256(resolved), bytes=resolved.stat().st_size)
    if variant is not None:
        record.update(variant=variant, filename=FAMPNN_WEIGHTS[variant])
    return record
