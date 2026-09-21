"""Refuse, loudly and early, when the official runtime's Protenix is absent.

PXDesign's official inference path imports ``protenix.data.parser``, which
exists only in Protenix ``v0.5.0+pxd`` (``d18aa1da``). This repo vendors
``c3bfc36`` (v2.0.0), where that module moved to ``protenix.data.core.parser``.

**Never alias one to the other.** It is a two-line change that makes the
import succeed, and it is exactly the pairing that produced interpenetrating
backbones -- 282 atom pairs under 2.6 A, 0.207 A minimum (docs/
target_conditioning_audit.md §5, §10). Aliasing converts a loud ImportError
into quiet wrong physics, and the output still looks like a structure.

So the contract is: anything that needs the official runtime calls
:func:`require_official_protenix` at entry, gets a message naming the audit
section and the machine where the environment lives, and exits. It does not
discover the problem four frames deep inside PXDesign's data pipeline.

The official environment lives on HAI, not here:

    /hai/scratch/yfsun/envs/pxdesign_official
        protenix 0.5.0+pxd @ d18aa1da, pxdbench 0.1.2 @ f6d0d72, torch 2.3.1
    /hai/scratch/yfsun/pxdesign_official/run_target.sh

De novo generation and the §9 evaluation run there and their artifacts ship
back. Rebuilding it on Marlowe is explicitly not the plan.
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
`protenix.data.core.parser`. DO NOT alias them. That pairing is what produced
the interpenetrating backbones recorded in docs/target_conditioning_audit.md
(282 atom pairs under 2.6 A, 0.207 A minimum) -- it turns this loud failure
into quiet wrong physics that still looks like a structure.

The official environment is on HAI, not Marlowe:
    {OFFICIAL_ENV}
    {OFFICIAL_RUNNER}

Run de novo generation and the section 9 evaluation there and ship the
artifacts back. See docs/target_conditioning_audit.md sections 5 and 10."""


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
