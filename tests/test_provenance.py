"""Provenance must pin both components and tolerate only the recorded patch."""
import pytest

from pxf import provenance


def test_all_three_upstreams_are_at_their_pinned_revisions():
    sources = provenance.runtime_sources()
    assert set(sources) == {"pxdesign", "protenix", "fampnn"}
    for name, record in sources.items():
        assert record["revision"] == provenance.PINNED[name], name


def test_fampnn_source_is_pristine():
    assert provenance.component_record("fampnn")["patched"] is False


def test_only_pxdesign_may_carry_patches():
    assert set(provenance.ALLOWED_PATCHES) == {"pxdesign"}
    for patch in provenance.ALLOWED_PATCHES["pxdesign"]:
        assert (provenance.repo_root() / patch).is_file(), patch


def test_patch_set_is_validated_by_reverse_apply():
    """Proves the tree is HEAD + exactly the recorded patches, not merely similar."""
    root = provenance.repo_root() / "PXDesign"
    assert provenance._is_exactly_patched(root, provenance.ALLOWED_PATCHES["pxdesign"])
    # A missing patch file must fail closed rather than pass.
    assert not provenance._is_exactly_patched(root, ("patches/does-not-exist.patch",))


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        provenance.component_record("alphafold")


def test_default_weights_are_the_packing_variant():
    # Upstream's own packing example uses the 0.0 Angstrom-noise checkpoint.
    assert provenance.DEFAULT_FAMPNN_WEIGHTS == "0.0"
    assert provenance.FAMPNN_WEIGHTS["0.0"] == "fampnn_0_0.pt"


def test_weights_ship_inside_the_submodule():
    path = provenance.fampnn_checkpoint()
    assert path.is_file() and path.name == "fampnn_0_0.pt"
    assert "fampnn/weights" in str(path)


def test_unknown_weight_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown FaMPNN weights"):
        provenance.fampnn_checkpoint("0.9")


def test_weight_record_digests_the_file(tmp_path):
    target = tmp_path / "w.pt"
    target.write_bytes(b"abc")
    record = provenance.weight_record(target)
    assert record["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert record["bytes"] == 3


def test_missing_weight_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="Weight file not found"):
        provenance.weight_record(tmp_path / "absent.pt")
