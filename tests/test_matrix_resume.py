"""Cell-level resume for the integrated matrix.

Without this a 480-cell run loses everything to one time limit: designs.csv
was written once, at the end.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

MATRIX = Path(__file__).resolve().parents[1] / "scripts" / "run_integrated_binder_matrix.py"


def _mod():
    spec = importlib.util.spec_from_file_location("matrix_resume", MATRIX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(target, length, seed, arm):
    return {"target": target, "binder_length": str(length),
            "generation_seed": str(seed), "arm": arm, "sample_id": f"{arm}"}


def test_finished_cells_are_skipped_and_their_rows_kept(tmp_path):
    m = _mod()
    rows = [_row("PDL1", 80, 101, a) for a in ("U03", "J03")]
    m._flush_cell(tmp_path, rows, m._cell_key("PDL1", 80, 101))

    done, kept = m._load_resume(tmp_path)
    assert done == {"PDL1_L80_s101"}
    assert len(kept) == 2


def test_rows_from_an_unfinished_cell_are_dropped(tmp_path):
    # A job killed mid-cell leaves fewer arms than the cell should have, or
    # arms computed against a prefix the rerun will not reproduce. Keeping
    # them would put a silently incomplete cell in the table.
    m = _mod()
    good = [_row("PDL1", 80, 101, "U03")]
    m._flush_cell(tmp_path, good, m._cell_key("PDL1", 80, 101))

    # simulate a crash: rows for a second cell land, the marker never does
    with (tmp_path / "designs.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(m.ROW_COLUMNS),
                                extrasaction="ignore")
        writer.writerow(_row("PDL1", 90, 101, "U03"))

    done, kept = m._load_resume(tmp_path)
    assert done == {"PDL1_L80_s101"}
    assert [r["binder_length"] for r in kept] == ["80"]


def test_no_marker_file_means_a_fresh_run(tmp_path):
    m = _mod()
    done, kept = m._load_resume(tmp_path)
    assert done == set() and kept == []


def test_the_marker_is_written_after_the_rows(tmp_path):
    # Ordering is the crash-safety property: rows first, then the marker.
    # Reversed, a crash between them would mark a cell done whose rows are
    # missing, and the rerun would skip it forever.
    m = _mod()
    m._flush_cell(tmp_path, [_row("PDL1", 80, 101, "U03")],
                  m._cell_key("PDL1", 80, 101))
    designs = (tmp_path / "designs.csv").stat().st_mtime_ns
    marker = (tmp_path / m.DONE_FILE).stat().st_mtime_ns
    assert marker >= designs
