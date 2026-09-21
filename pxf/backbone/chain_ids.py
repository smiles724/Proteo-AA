"""Author chain ids are not the featurizer's chain ids.

Everything upstream of the featurizer names chains the way the depositor did:
AlphaProteo Table S1, `configs/binder_benchmark/targets.yaml`, a gemmi read of
the file, a PDB web page. That is the **author** id, `auth_asym_id`.

`CifFileProvider(binder_chain_ids=[...])` compares against
`atom_array.chain_id`, and Protenix populates that from **`label_asym_id`** --
the mmCIF-internal id, assigned one per entity instance and in file order. The
two agree for a plain two-chain deposition, which is why this goes unnoticed,
and they diverge exactly where the benchmark lives:

    6m0j   auth E (the SARS-CoV-2 RBD)  ->  label B
    1www   auth V W X Y                 ->  label A B C D
    1bj1   auth L H V W J K             ->  label A B C D E F
    7p0s   auth A U B C                 ->  label A B C D

The failure is not always loud. For 1www and 1bj1 the author id does not exist
as a label id, so the selector matches nothing. For **6m0j it does**: label E
exists -- it is a NAG on ACE2 -- so asking for "E" silently designs against a
glycan and returns a well-formed result. That is the case this module exists
to prevent.

Waters and glycans take label ids of their own, so the mapping is restricted
to amino-acid polymer chains: the resolved id is the one the design/target
split is about.

Measured, not assumed: `CifFileProvider` on 7p0s reports chains A(148 res),
B(26), C(148), D(26) with `auth_asym_id` A, U, B, C respectively -- the label
ids this module derives from the file, in that correspondence.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

__all__ = ["protein_chain_map", "featurizer_chain_id", "ChainIdError"]


class ChainIdError(ValueError):
    """An author chain id that does not resolve to exactly one label id."""


def protein_chain_map(cif_path) -> "OrderedDict[str, list[str]]":
    """Author chain id -> the label_asym_id(s) of its amino-acid chains.

    A list, not a scalar, because nothing in mmCIF forbids one author chain
    from holding two polymer entities. In the ten benchmark depositions and
    the dev set it is always length one; `featurizer_chain_id` is where that
    expectation is enforced rather than assumed here.
    """
    import gemmi

    block = gemmi.cif.read(str(cif_path)).sole_block()
    table = block.find("_atom_site.", ["label_asym_id", "auth_asym_id",
                                       "label_comp_id"])
    if not table:
        raise ChainIdError(f"{cif_path}: no _atom_site rows with label/auth ids")

    seen: "OrderedDict[tuple[str, str], bool]" = OrderedDict()
    amino: dict[str, bool] = {}
    for row in table:
        label, auth, comp = row[0], row[1], row[2]
        is_aa = amino.get(comp)
        if is_aa is None:
            info = gemmi.find_tabulated_residue(comp)
            is_aa = bool(info and info.is_amino_acid())
            amino[comp] = is_aa
        if is_aa:
            seen[(auth, label)] = True

    out: "OrderedDict[str, list[str]]" = OrderedDict()
    for auth, label in seen:
        out.setdefault(auth, []).append(label)
    return out


def featurizer_chain_id(cif_path, auth_chain: str) -> str:
    """The id to hand `binder_chain_ids` for the author chain `auth_chain`.

    Raises rather than guessing: a wrong chain here designs against the wrong
    molecule and the result still looks like a result.
    """
    mapping = protein_chain_map(cif_path)
    labels = mapping.get(str(auth_chain))
    if not labels:
        known = ", ".join(f"{a}->{'/'.join(l)}" for a, l in mapping.items())
        raise ChainIdError(
            f"{Path(cif_path).name}: no amino-acid chain with author id "
            f"{auth_chain!r}; author->label for this file is {known}"
        )
    if len(labels) > 1:
        raise ChainIdError(
            f"{Path(cif_path).name}: author chain {auth_chain!r} covers more "
            f"than one polymer entity ({', '.join(labels)}); the design/target "
            "split has to name one of them explicitly"
        )
    return labels[0]
