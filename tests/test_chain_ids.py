"""Author chain ids must be converted before they reach the featurizer.

The interesting failure is not "chain not found" -- that one announces itself.
It is 6m0j, where the author id of the RBD is ``E`` and ``E`` *also* exists as
a label id belonging to a glycan on the other chain. Asking for ``E`` there
returns a well-formed design against the wrong molecule, so the test that
matters is the one asserting the resolver disagrees with the naive answer.

These use hand-written mmCIF fragments rather than the depositions so they run
without the data mirror; the real files are checked in
``scripts/validate_binder_targets.py``, which has them.
"""
import pytest

from pxf.backbone.chain_ids import (
    ChainIdError,
    featurizer_chain_id,
    protein_chain_map,
)

HEADER = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def _atom_site_rows(cif_path):
    """The written rows back out, whitespace-split, for assertions about the file."""
    return [line.split() for line in cif_path.read_text().splitlines()
            if line.startswith(("ATOM ", "HETATM "))]


def write_cif(tmp_path, rows, name="probe.cif"):
    """rows: (comp, label_asym, auth_asym, seq) -> a one-atom-per-residue CIF."""
    lines = [HEADER]
    for index, (comp, label, auth, seq) in enumerate(rows, start=1):
        group = "ATOM" if comp not in ("NAG", "HOH", "SO4") else "HETATM"
        lines.append(
            f"{group} {index} CA {comp} {label} {seq} {auth} "
            f"{index}.000 0.000 0.000\n"
        )
    path = tmp_path / name
    path.write_text("".join(lines))
    return path


def test_identical_ids_pass_through(tmp_path):
    cif = write_cif(tmp_path, [
        ("ALA", "A", "A", 1), ("GLY", "A", "A", 2),
        ("SER", "B", "B", 1), ("LEU", "B", "B", 2),
    ])
    assert protein_chain_map(cif) == {"A": ["A"], "B": ["B"]}
    assert featurizer_chain_id(cif, "B") == "B"


def test_6m0j_shape_resolves_away_from_the_glycan(tmp_path):
    """auth E is label B; label E is a NAG on auth A. The naive answer is wrong.

    This is the 6m0j layout: ACE2 as auth A, the RBD as auth E, and glycans
    on ACE2 taking their own label ids -- one of which collides with the
    author id of the chain we actually want.
    """
    cif = write_cif(tmp_path, [
        ("ALA", "A", "A", 1), ("GLY", "A", "A", 2),   # ACE2, auth A
        ("SER", "B", "E", 1), ("LEU", "B", "E", 2),   # RBD,  auth E -> label B
        ("NAG", "E", "A", "."),                        # glycan on ACE2
    ])
    assert featurizer_chain_id(cif, "E") == "B"
    # The trap, stated as an assertion: label "E" is present in the file, so a
    # selector handed the author id matches a glycan rather than failing. Only
    # the protein map knows that "E" is not a chain anyone asked for.
    labels_in_file = {row[4] for row in _atom_site_rows(cif)}
    assert "E" in labels_in_file
    assert protein_chain_map(cif) == {"A": ["A"], "E": ["B"]}


def test_1www_shape_where_no_author_id_is_a_label_id(tmp_path):
    cif = write_cif(tmp_path, [
        ("ALA", "A", "V", 1), ("GLY", "B", "W", 1),
        ("SER", "C", "X", 1), ("LEU", "D", "Y", 1),
    ])
    assert featurizer_chain_id(cif, "X") == "C"


def test_waters_and_ligands_do_not_claim_a_chain(tmp_path):
    cif = write_cif(tmp_path, [
        ("ALA", "A", "A", 1),
        ("HOH", "B", "A", "."), ("SO4", "C", "A", "."),
        ("GLY", "D", "B", 1),
    ])
    assert protein_chain_map(cif) == {"A": ["A"], "B": ["D"]}
    assert featurizer_chain_id(cif, "B") == "D"


def test_unknown_author_chain_names_the_alternatives(tmp_path):
    cif = write_cif(tmp_path, [("ALA", "A", "V", 1), ("GLY", "B", "W", 1)])
    with pytest.raises(ChainIdError) as excinfo:
        featurizer_chain_id(cif, "B")
    message = str(excinfo.value)
    assert "author id 'B'" in message
    assert "V->A" in message and "W->B" in message


def test_two_polymer_entities_in_one_author_chain_is_ambiguous(tmp_path):
    cif = write_cif(tmp_path, [
        ("ALA", "A", "A", 1), ("GLY", "B", "A", 1), ("SER", "C", "B", 1),
    ])
    assert protein_chain_map(cif)["A"] == ["A", "B"]
    with pytest.raises(ChainIdError, match="more than one polymer entity"):
        featurizer_chain_id(cif, "A")


def test_order_follows_the_file(tmp_path):
    """First appearance wins, so the report reads in deposition order."""
    cif = write_cif(tmp_path, [
        ("ALA", "C", "L", 1), ("GLY", "A", "H", 1), ("SER", "B", "V", 1),
    ])
    assert list(protein_chain_map(cif)) == ["L", "H", "V"]


def test_a_file_without_atom_site_rows_is_an_error(tmp_path):
    path = tmp_path / "empty.cif"
    path.write_text("data_empty\n_entry.id empty\n")
    with pytest.raises(ChainIdError, match="no _atom_site rows"):
        protein_chain_map(path)
