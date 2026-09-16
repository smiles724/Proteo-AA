"""The AFDB subset: reading its shards and re-emitting mmCIF.

The mmCIF export earned a test file of its own. Getting it wrong does not raise:
the file parses, every filter reports zero atoms dropped, and the atom array
comes back empty or annotated ``mol_type == "ligand"``. Four separate fields
have to be right at once, and three of them fail that quietly.
"""

import pytest
import torch

from pxf.train import afdb as A

pytest.importorskip("gemmi")

needs_shards = pytest.mark.skipif(
    not (A.DEFAULT_DATA_ROOT / "manifest.sqlite").is_file(),
    reason=f"no AFDB build at {A.DEFAULT_DATA_ROOT}",
)


@pytest.fixture(scope="module")
def shards():
    return A.AFDBShards(split="train")


@needs_shards
def test_the_manifest_and_shards_agree(shards):
    assert len(shards) > 300_000
    afid = shards.afids[0]
    record = shards.read(afid)
    # read() checks this itself; asserting it here says why it matters.
    assert record.afid == afid
    assert len(record) == shards.lengths[afid]


@needs_shards
def test_records_satisfy_the_atom37_contract(shards):
    record = shards.read(shards.afids[0])
    length = len(record)
    assert record.x.shape == (length, A.NUM_ATOM37, 3)
    assert record.atom_mask.shape == (length, A.NUM_ATOM37)
    assert record.aatype.dtype == torch.int64
    assert 0 <= int(record.aatype.min()) and int(record.aatype.max()) <= 20
    assert 0.0 <= float(record.plddt.min()) and float(record.plddt.max()) <= 100.0


@needs_shards
def test_afdb_needs_no_crystallographic_filtering(shards):
    """The contrast with the PDB, stated as a test.

    ``pxf.train.protenix`` exists because deposited side chains can be present
    but untrustworthy. AFDB predicts every atom of every residue, so the
    per-atom mask is empty and there is nothing for a per-residue veto to do.
    """
    for afid in shards.afids[:3]:
        batch = A.training_batch(shards.read(afid))
        assert float(batch["missing_atom_mask"].sum()) == 0.0
        assert float(batch["seq_mask"].sum()) == batch["aatype"].shape[0]


@needs_shards
def test_the_exported_cif_survives_the_featurizer(shards, tmp_path):
    """The whole round trip, which is the only check that means anything here.

    Length *and* sequence, because ``_native_atom37`` compares both and a
    disagreement misaligns every side-chain target.
    """
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf import atom37
    from pxf.backbone.driver import featurize_structures, to_featurized

    afid = min(shards.afids, key=lambda a: shards.lengths[a])
    record = shards.read(afid)
    path = A.write_cif(record, tmp_path / f"{afid}.cif")

    sample_id, source = featurize_structures([str(path)], crop_size=512)[0]
    structure = to_featurized(sample_id, source[0])
    native = process_single_pdb(load_feats_from_pdb(str(path)))

    assert structure.num_tokens == int(native["aatype"].shape[0]) == len(record)
    assert atom37.sequence_from_aatype(structure.aatype) == atom37.sequence_from_aatype(
        native["aatype"].long()
    )


@needs_shards
def test_the_exported_cif_carries_the_fields_that_fail_silently(shards, tmp_path):
    """Four fields, three of which produce no error when wrong.

    * ``label_seq_id`` unset -> Protenix's consecutive-CA filter casts '.' to
      int64 and raises, reported as "parsed without atom_array".
    * ``label_asym_id`` not matching ``pdbx_struct_assembly_gen.asym_id_list``
      -> expand_assembly keeps nothing; no error, empty atom array.
    * ``label_entity_id`` unset -> no mapping to entity_poly_type, so mol_type
      defaults to "ligand" and the selector reports no protein chains.
    * ``pdbx_struct_assembly`` absent -> KeyError on oligomeric_count, which is
      read unguarded.
    """
    import gemmi

    afid = min(shards.afids, key=lambda a: shards.lengths[a])
    path = A.write_cif(shards.read(afid), tmp_path / "x.cif")
    block = gemmi.cif.read(str(path)).sole_block()

    assembly = block.find("_pdbx_struct_assembly.", ["id", "oligomeric_count"])
    assert len(assembly) == 1 and assembly[0][1] == "1"

    generated = block.find(
        "_pdbx_struct_assembly_gen.", ["assembly_id", "oper_expression", "asym_id_list"]
    )
    assert len(generated) == 1
    asym_in_assembly = generated[0][2]

    operators = block.find("_pdbx_struct_oper_list.", ["id", "type"])
    assert len(operators) == 1, "the identity operator must parse as exactly one row"

    sites = block.find("_atom_site.", ["label_asym_id", "label_entity_id", "label_seq_id"])
    assert len(sites) > 0
    asym_ids = {row[0] for row in sites}
    assert asym_ids == {asym_in_assembly}, (
        "the assembly must name the asym id the atoms actually carry, or the "
        "assembly expands to nothing without an error"
    )
    assert all(row[1] not in (".", "?", "-") for row in sites), "label_entity_id unset"
    assert all(row[2].isdigit() for row in sites), "label_seq_id must be an integer"
    # Sequential from 1 along the polymer.
    seq_ids = sorted({int(row[2]) for row in sites})
    assert seq_ids == list(range(1, len(seq_ids) + 1))


@needs_shards
def test_the_entity_sequence_is_emitted(shards, tmp_path):
    """Without _entity_poly_seq, poly_res_names lookup is a KeyError."""
    import gemmi

    afid = min(shards.afids, key=lambda a: shards.lengths[a])
    record = shards.read(afid)
    path = A.write_cif(record, tmp_path / "x.cif")
    block = gemmi.cif.read(str(path)).sole_block()
    poly_seq = block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"])
    assert len(poly_seq) == len(record)


@needs_shards
def test_plddt_goes_to_b_iso(shards, tmp_path):
    """AFDB's own convention. Anything reading it as disorder has it backwards."""
    import gemmi

    afid = min(shards.afids, key=lambda a: shards.lengths[a])
    record = shards.read(afid)
    path = A.write_cif(record, tmp_path / "x.cif")
    block = gemmi.cif.read(str(path)).sole_block()
    rows = block.find("_atom_site.", ["label_seq_id", "B_iso_or_equiv"])
    first = {int(row[0]): float(row[1]) for row in rows}
    assert first[1] == pytest.approx(float(record.plddt[0]), abs=1e-2)


@needs_shards
def test_a_record_with_no_canonical_residues_is_refused(shards, tmp_path):
    record = shards.read(shards.afids[0])
    record.aatype = torch.full_like(record.aatype, 20)  # all UNK
    with pytest.raises(ValueError, match="no atoms to write"):
        A.write_cif(record, tmp_path / "x.cif")


def test_a_bad_magic_is_refused():
    with pytest.raises(ValueError, match="ALP1"):
        A._decode(b"XXXX" + b"\x00" * 64)
