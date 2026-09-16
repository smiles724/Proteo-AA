"""The structure manifest a coupling run reads.

A run's data source has to be reproducible after the fact. Before this, phase 1
was launched from a list built by a throwaway script in a session scratchpad --
the job trained fine and the list was unrecoverable. So the manifest carries its
own provenance as comments, and the reader skips them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

tc = pytest.importorskip("train_couple")

MANIFEST = Path(__file__).resolve().parents[1] / "configs/phase1_structures_casp14_15.txt"


def test_comments_and_blank_lines_are_skipped(tmp_path):
    real = tmp_path / "a.cif"
    real.write_text("x")
    manifest = tmp_path / "m.txt"
    manifest.write_text(
        "# produced by scripts/survey_coupling_structures.py\n"
        "#   --crop-size 512\n"
        "\n"
        f"{real}\n"
        "   # an indented comment\n"
    )
    assert tc.resolve_structures(str(manifest)) == [str(real)]


def test_a_missing_path_is_still_an_error(tmp_path):
    """Skipping comments must not turn into skipping anything unreadable."""
    manifest = tmp_path / "m.txt"
    manifest.write_text("# a comment\n/nonexistent/x.cif\n")
    with pytest.raises(SystemExit, match="do not exist"):
        tc.resolve_structures(str(manifest))


def test_a_comment_only_manifest_is_refused(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("# nothing but provenance\n\n")
    with pytest.raises(SystemExit, match="empty"):
        tc.resolve_structures(str(manifest))


@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not present")
def test_the_committed_phase1_manifest_is_readable_and_documented():
    """The manifest phase 1 actually ran on."""
    text = MANIFEST.read_text()
    assert text.startswith("# Structures the coupling path can train on")
    # It must say which crop size it needs; a smaller one misaligns the targets.
    assert "--crop-size >= 482" in text
    assert "survey_coupling_structures.py" in text

    paths = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert len(paths) == 56
    assert all(p.endswith(".cif") for p in paths)
    assert len(set(paths)) == len(paths), "a duplicate would be trained on twice"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not present")
def test_the_rejection_report_sits_next_to_the_manifest():
    """Why 22 of 78 candidates were dropped should not need rediscovering."""
    import json

    report = Path(f"{MANIFEST}.report.json")
    assert report.is_file()
    record = json.loads(report.read_text())
    assert record["usable"] == 56 and record["candidates"] == 78
    assert record["min_crop_size_required"] == 482
    assert len(record["rejected"]) == 22
    assert all("reason" in entry for entry in record["rejected"])
