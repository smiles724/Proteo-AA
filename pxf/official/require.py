"""Refuse, loudly and early, when the official runtime's Protenix is absent.

PXDesign's official inference path imports ``protenix.data.parser``, which
exists only in Protenix ``v0.5.0+pxd`` (``d18aa1da``). This repo vendors
``c3bfc36`` (v2.0.0), where that module moved to ``protenix.data.core.parser``.

**Never alias one to the other.** It is a two-line change that makes the
import succeed, and it silently pairs PXDesign with a parser it was not
written against -- a loud ImportError converted into a quiet version skew
whose output still looks like a structure.

CORRECTION (this docstring used to overstate the case): aliasing is NOT what
produced the interpenetrating backbones. §9 measured that cause and it was
the `FixedTarget` coordinate overwrite -- the local sampler pinned the target
to its native frame while the binder was generated in the model's own frame
(26.417 A raw vs 0.107 A superposed, centroid 18 A apart) -- together with
`step_scale_eta` 1.5 vs 2.5 (§8). The interpenetration is a sampler-contract
bug, not a parser bug. The refusal below still stands, on the narrower and
true grounds of version skew.

That overstatement had a cost: it was read as meaning the official runtime
could not be obtained here at all, and the backbone stage was reported
blocked on HAI for longer than it needed to be. It was not blocked.
Protenix v0.5.0+pxd is a public git tag and installs on Marlowe; see
`scripts/utilities/install_pxdesign_official.sh`, validated against §7's
positive control (12/12 clash-free). Installing the correct version is not
aliasing, and this guard passes once it is installed.

So the contract is: anything that needs the official runtime calls
:func:`require_official_protenix` at entry, gets a message naming the audit
section and the machine where the environment lives, and exits. It does not
discover the problem four frames deep inside PXDesign's data pipeline.

Two environments satisfy this guard:

    /hai/scratch/yfsun/envs/pxdesign_official        (HAI, original)
    /users/yfsun/.venvs/pxdesign_official            (Marlowe, built later)
        protenix 0.5.0+pxd @ d18aa1da, pxdbench 0.1.2, torch 2.3.1, deepspeed 0.15.4

The Marlowe one additionally needs a PRISTINE PXDesign checkout
(`/users/yfsun/pxdesign_pristine`, f788441): this repo's working copy patches
`pxdesign/model/embedders.py` to bridge to the vendored v2.0.0's newer
AtomAttentionEncoder signature, and the official encoder -- which takes the
whole feature dict -- fails on it with `KeyError: 'd_lm'`. The two runtimes
are kept in separate trees deliberately.
"""

from __future__ import annotations

import importlib.util

#: The revision PXDesign's own install.sh pins.
REQUIRED_REVISION = "d18aa1da"
REQUIRED_VERSION = "0.5.0+pxd"
OFFICIAL_ENV = "/hai/scratch/yfsun/envs/pxdesign_official"
OFFICIAL_RUNNER = "/hai/scratch/yfsun/pxdesign_official/run_target.sh"

_MESSAGE = f"""\
This step needs PXDesign's OFFICIAL runtime, which requires Protenix
{REQUIRED_VERSION} ({REQUIRED_REVISION}). This environment has {{found}}.

`protenix.data.parser` does not exist here; v2.0.0 moved it to
`protenix.data.core.parser`. DO NOT alias them: that pairs PXDesign with a
parser it was not written against and turns this loud failure into a quiet
version skew whose output still looks like a structure.

(Aliasing is NOT what caused the interpenetrating backbones in
docs/target_conditioning_audit.md. Section 9 measured that cause: the
FixedTarget coordinate overwrite, plus eta 1.5 vs 2.5. This refusal stands on
version skew alone.)

The fix is to INSTALL the right version, which works here and is not an alias:
    scripts/utilities/install_pxdesign_official.sh

An environment that satisfies this guard:
    {OFFICIAL_ENV}                    (HAI)
    /users/yfsun/.venvs/pxdesign_official   (Marlowe)

Build the Marlowe one with scripts/utilities/install_pxdesign_official.sh,
and drive it from a PRISTINE PXDesign checkout. See
docs/target_conditioning_audit.md sections 5-10."""


def official_protenix_available() -> tuple[bool, str]:
    """``(usable, what_was_found)``. Never imports PXDesign's pipeline."""
    try:
        import protenix  # noqa: F401
    except ImportError:
        return False, "no protenix at all"

    version = getattr(__import__("protenix"), "__version__", "unknown")
    # The import that actually distinguishes the two revisions. find_spec
    # rather than import, so a missing module costs nothing and a present one
    # is not executed as a side effect of a check.
    has_parser = importlib.util.find_spec("protenix.data.parser") is not None
    if has_parser:
        return True, f"protenix {version} (protenix.data.parser present)"
    return False, f"protenix {version} (protenix.data.parser absent)"


def require_official_protenix(step: str = "this step") -> None:
    """Exit with the audit's explanation, or return silently."""
    usable, found = official_protenix_available()
    if usable:
        return
    raise SystemExit(f"{step}: " + _MESSAGE.format(found=found))
