"""Every `--sc-*` training flag must actually reach the config.

This exists because it already happened. `--sc-edm` was added to the parser and
the block that copies it into `configs.sidechain` was never inserted, so the flag
parsed cleanly, the job started, the run directory was named `edm_global`, and
the model trained the ordinary one-step objective for 300 steps before anyone
noticed. argparse cannot catch this: an accepted-and-ignored flag looks exactly
like an accepted-and-honoured one from the command line.

The failure mode is the same family as `test_train_inference_parity`: a switch
that is configured in one place and read in another, with nothing tying the two
together. So tie them together.
"""
import inspect
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_protenix_monomer.py"

# Flags whose effect is not a `configs.*` assignment. Each needs a reason.
NOT_A_CONFIG_ASSIGNMENT = {
    "sc_ablation_arm": "consumed by apply_sidechain_ablation_arm(configs, args.sc_ablation_arm)",
}


def _declared_sc_flags(src: str) -> set[str]:
    """dest names for every --sc-* argument the parser declares."""
    return {
        "sc_" + m.group(1).replace("-", "_")
        for m in re.finditer(r'"--sc-([a-z0-9-]+)"', src)
    }


def test_every_sc_flag_is_read_somewhere():
    src = SCRIPT.read_text()
    declared = _declared_sc_flags(src)
    assert declared, "no --sc-* flags found; the scraper is broken"

    unused = sorted(
        d for d in declared
        if d not in NOT_A_CONFIG_ASSIGNMENT and f"args.{d}" not in src
    )
    assert not unused, (
        f"These --sc-* flags are declared but never read: {unused}. The job will "
        "accept them on the command line and silently ignore them, so a run can be "
        "launched, named and reported as something it is not."
    )


def test_every_sc_flag_reaches_the_sidechain_config():
    """Reading the flag is not enough -- it has to be written somewhere.

    A flag read only inside an f-string log line would pass the test above while
    still changing nothing about the run.
    """
    src = SCRIPT.read_text()
    declared = _declared_sc_flags(src)
    missing = []
    for dest in sorted(declared):
        if dest in NOT_A_CONFIG_ASSIGNMENT:
            continue
        # Look for an assignment whose right-hand side mentions the flag.
        if not re.search(
            r"config[a-z_.]*\s*=\s*[^\n]*\bargs\.%s\b" % re.escape(dest), src
        ):
            missing.append(dest)
    assert not missing, (
        f"These --sc-* flags are read but never assigned into a config: {missing}"
    )


def test_the_flag_that_was_silently_ignored_is_covered():
    """Pin the specific regression."""
    src = SCRIPT.read_text()
    assert "configs.sidechain.edm = bool(args.sc_edm)" in src
    assert "configs.sidechain.pack_loss = float(args.sc_pack_loss)" in src
    assert "configs.sidechain.mismatch_loss = str(args.sc_mismatch_loss)" in src
