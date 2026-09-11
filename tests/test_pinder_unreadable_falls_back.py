"""A PINDER structure we cannot open must route to the archive, not explode.

`Path.is_file()` only stats. A shared PINDER tree can hold files another user
extracted under a restrictive umask -- about a third of
/hai/scratch/yfsun/pinder/2024-02/pdbs is mode 600 -- and those pass
`is_file()`, so the archive fallback was skipped and the run died later in
`pdb_to_cif` with PermissionError, inside a DataLoader worker and far from the
cause. Both Stage IV LigandMPNN jobs (114339, 114340) died this way.
"""
import os

import pytest

from pxdesign_train.runner.pinder_provider import _is_readable_file


def test_unreadable_file_counts_as_absent(tmp_path):
    readable = tmp_path / "readable.pdb"
    readable.write_text("ATOM\n")
    assert _is_readable_file(readable)

    unreadable = tmp_path / "unreadable.pdb"
    unreadable.write_text("ATOM\n")
    unreadable.chmod(0o000)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip("running as a user that bypasses file modes")
        # The distinction the old code missed: present, but not openable.
        assert unreadable.is_file()
        assert not _is_readable_file(unreadable)
    finally:
        unreadable.chmod(0o600)


def test_missing_file_counts_as_absent(tmp_path):
    assert not _is_readable_file(tmp_path / "nope.pdb")
    # A directory is not a usable structure file either.
    assert not _is_readable_file(tmp_path)


def test_provider_uses_the_readability_check_at_every_decision():
    """Pin all three call sites, not just the two that caused the crash."""
    import inspect

    from pxdesign_train.runner.pinder_provider import PinderPdbProvider

    source = inspect.getsource(PinderPdbProvider._ensure_cif)
    # Manifest path, sharded fallback, and the post-extraction guard: all three
    # decide "can this be used", and a bare is_file() answers the wrong question.
    assert source.count("_is_readable_file(") == 3
    assert "if not pdb_path.is_file()" not in source
    assert "if sharded_path.is_file()" not in source
