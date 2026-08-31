import zipfile

import pandas as pd


def test_pinder_pdb_cache_is_independent_from_release_root(tmp_path):
    from pxdesign_train.runner.pinder_provider import PinderPdbProvider

    release_root = tmp_path / "shared_release"
    pdb_cache = tmp_path / "user_cache" / "pdbs"
    cif_cache = tmp_path / "user_cache" / "cifs"
    manifest = tmp_path / "manifest.csv"
    pinder_id = "2tnf__A1_P06804--2tnf__C1_P06804"
    pd.DataFrame(
        [
            {
                "pinder_id": pinder_id,
                "pdb_path": f"pdbs/{pinder_id}.pdb",
                "converted_binder_chain": "B",
                "source_split": "train",
            }
        ]
    ).to_csv(manifest, index=False)

    archive = tmp_path / "pdbs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(f"pdbs/{pinder_id}.pdb", "ATOM\n")

    provider = PinderPdbProvider(
        manifest_path=manifest,
        pinder_root=release_root,
        cif_cache_dir=cif_cache,
        pdb_cache_dir=pdb_cache,
        archive_path=archive,
    )

    expected = (pdb_cache / "tn" / f"{pinder_id}.pdb").resolve()
    assert provider._cached_pdb_path(0) == expected
    assert release_root.resolve() not in expected.parents
    assert provider._extract_from_archive(0, expected) == expected
    assert expected.read_text() == "ATOM\n"
    assert not list(expected.parent.glob(f".{expected.name}.tmp.*"))
