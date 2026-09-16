"""The La-Proteina AFDB subset: reading its shards and exporting mmCIF.

The subset is built by ``/hai/users/y/f/yfsun/afdb-laproteina`` into zstd shards
of packed atom37 records (``coords [L,37,3]``, ``atom_mask [L,37]``, ``aatype``,
``residue_index``, ``plddt``), indexed by a sqlite manifest. 344,507 records, no
failures.

Two consumers, and they need different things:

**The FaMPNN objective** takes atom37 directly, so
:class:`AFDBSideChainDataset` hands it the record with no intermediate format.
AFDB needs none of the crystallographic filtering the PDB does -- there is no
occupancy, no altloc, no unresolved side chain -- so ``missing_atom_mask`` comes
out all-zero and every side chain is supervised. See :mod:`pxf.train.protenix`
for the PDB case, where that is emphatically not true.

**The coupling phases** go through ``pxdesign_train``'s featurizer, which reads
mmCIF. The builder discards the downloaded CIFs (``raw`` is transient), so
:func:`write_cif` re-emits one from the packed record. That is a detour, but the
alternative is a second atom37 -> PXDesign feature path, and the featurizer is
the component whose output the driver was verified against.

Why AFDB suits the coupling path when the PDB did not: the three constraints that
reject most Protenix mmCIFs are all satisfied by construction. Every record is a
single chain, contains no ligands, ions or waters, and has a contiguous
``residue_index``, so the featurized token count matches what FaMPNN parses from
the same file instead of disagreeing by an assembly. See ``docs/data.md``.

**pLDDT lands in ``B_iso``**, which is AFDB's own convention and makes the export
faithful to the source. Anything that reads that column as crystallographic
disorder has the sign backwards on these files: high is good here.
"""

from __future__ import annotations

import pathlib
import sqlite3
import struct
import subprocess
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_DATA_ROOT = pathlib.Path("/hai/scratch/yfsun/afdb_laproteina/run")

# The record framing from afdb_laproteina/shards.py.
MAGIC = b"ALP1"
_HEAD = struct.Struct("<4sHIH")
NUM_ATOM37 = 37


@dataclass
class AFDBRecord:
    """One AFDB structure, in the atom37 contract :mod:`pxf.atom37` pins."""

    afid: str
    aatype: torch.Tensor  # [L] int64, 20 == UNK
    x: torch.Tensor  # [L, 37, 3] float32, Angstroms
    atom_mask: torch.Tensor  # [L, 37] float32
    residue_index: torch.Tensor  # [L] int64, 1-based label_seq_id
    plddt: torch.Tensor  # [L] float32, 0-100

    def __len__(self):
        return int(self.aatype.shape[0])


def _decode(body):
    """Unpack one decompressed record body."""
    magic, version, length, id_bytes = _HEAD.unpack_from(body, 0)
    if magic != MAGIC:
        raise ValueError(f"bad record magic {magic!r}; not an ALP1 shard")
    offset = _HEAD.size
    afid = body[offset : offset + id_bytes].decode()
    offset += id_bytes

    def take(dtype, count, shape):
        nonlocal offset
        array = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += array.nbytes
        return array

    coords = take("<f4", length * NUM_ATOM37 * 3, (length, NUM_ATOM37, 3))
    mask = take("u1", length * NUM_ATOM37, (length, NUM_ATOM37))
    aatype = take("u1", length, (length,))
    residue_index = take("<i4", length, (length,))
    plddt = take("<f4", length, (length,))
    if offset != len(body):
        raise ValueError(f"{afid}: {len(body) - offset} trailing byte(s) in the record")
    return AFDBRecord(
        afid=afid,
        aatype=torch.from_numpy(aatype.astype(np.int64)),
        x=torch.from_numpy(coords.copy()),
        atom_mask=torch.from_numpy(mask.astype(np.float32)),
        residue_index=torch.from_numpy(residue_index.astype(np.int64)),
        plddt=torch.from_numpy(plddt.copy()),
    )


class AFDBShards:
    """Random access to the shards, via the manifest's record offsets.

    Reads the manifest read-only and copies nothing. ``zstandard`` is used when
    importable and the ``zstd`` CLI otherwise, because the two have been
    alternately available in this environment and a dataset that cannot be read
    is worse than a subprocess.
    """

    def __init__(self, data_root=None, *, split="train"):
        self.root = pathlib.Path(data_root or DEFAULT_DATA_ROOT)
        manifest = self.root / "manifest.sqlite"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        query = (
            "SELECT afid, shard_id, rec_offset, rec_nbytes, length FROM ids "
            "WHERE shard_id IS NOT NULL"
        )
        parameters = ()
        if split is not None:
            query += " AND split = ?"
            parameters = (split,)
        # mode=ro still fails on a live WAL manifest; immutable=1 reads it anyway,
        # which is correct here because the build is finished.
        with sqlite3.connect(f"file:{manifest}?immutable=1", uri=True) as connection:
            rows = connection.execute(query + " ORDER BY afid", parameters).fetchall()
        if not rows:
            raise RuntimeError(f"no records for split={split!r} under {self.root}")
        self.split = split
        self.afids = [row[0] for row in rows]
        self._index = {row[0]: (row[1], row[2], row[3]) for row in rows}
        self.lengths = {row[0]: int(row[4]) for row in rows}
        self._decompressor = None

    def __len__(self):
        return len(self.afids)

    def __contains__(self, afid):
        return afid in self._index

    def _decompress(self, blob):
        try:
            import zstandard

            if self._decompressor is None:
                self._decompressor = zstandard.ZstdDecompressor()
            return self._decompressor.decompress(blob)
        except ImportError:
            return subprocess.run(
                ["zstd", "-d", "-c"], input=blob, capture_output=True, check=True
            ).stdout

    def read(self, afid):
        shard_id, offset, nbytes = self._index[afid]
        path = self.root / "shards" / f"shard_{int(shard_id):05d}.zst"
        with path.open("rb") as stream:
            stream.seek(int(offset))
            blob = stream.read(int(nbytes))
        record = _decode(self._decompress(blob))
        if record.afid != afid:
            raise ValueError(
                f"manifest points {afid} at a record holding {record.afid}; the "
                "shards and the manifest are out of step"
            )
        return record

    def identity(self):
        return dict(
            root=str(self.root), split=self.split, records=len(self), source="la-proteina"
        )


def write_cif(record, path, *, chain="A"):
    """Emit ``record`` as an mmCIF the featurizer can read.

    Built through ``gemmi`` rather than by string formatting, so the output is a
    real mmCIF with entities set up rather than something that merely parses
    today. Only atoms flagged present are written; AFDB predictions are complete,
    so in practice that is every atom of every canonical residue.
    """
    import gemmi

    from fampnn.data import residue_constants as rc
    from pxf import atom37 as atom37_module

    structure = gemmi.Structure()
    structure.name = record.afid
    structure.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    gemmi_chain = gemmi.Chain(chain)

    aatype = record.aatype.tolist()
    mask = record.atom_mask
    coords = record.x
    plddt = record.plddt.tolist()
    sequence = []
    written = 0
    for position, code in enumerate(aatype):
        if not 0 <= int(code) < 20:
            # UNK has no canonical atom set; skipping keeps the file's residue
            # count equal to what any downstream parser will agree on.
            continue
        name3 = rc.restype_1to3[atom37_module.AA_ORDER[int(code)]]
        residue = gemmi.Residue()
        residue.name = name3
        residue.seqid = gemmi.SeqId(int(record.residue_index[position]), " ")
        residue.het_flag = "A"
        # Both of these have to be set here, before setup_entities:
        #
        # label_seq  - gemmi leaves it unset for a structure assembled from bare
        #   residues (neither setup_entities nor assign_label_seq_id fills it
        #   in), and it writes as '.'. Protenix's consecutive-CA filter casts
        #   that column to int64 unguarded.
        # subchain   - setup_entities otherwise invents one ("Axp"), which gemmi
        #   writes as label_asym_id. pdbx_struct_assembly_gen has to name that
        #   same id or expand_assembly keeps nothing, and every filter reports
        #   zero drops while the atom_array comes back None. Pinning it *after*
        #   setup_entities instead orphans the subchain, so label_entity_id
        #   writes as '.', Protenix cannot map atoms to entity_poly_type, and
        #   mol_type silently becomes "ligand" -- which reads as "no protein
        #   chains in this file".
        residue.label_seq = len(sequence) + 1
        residue.subchain = chain
        for slot, atom_name in enumerate(atom37_module.ATOM37):
            if float(mask[position, slot]) <= 0:
                continue
            atom = gemmi.Atom()
            atom.name = atom_name
            # Every atom37 name begins with its element: C, N, O or S.
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(*(float(v) for v in coords[position, slot]))
            atom.occ = 1.0
            atom.b_iso = float(plddt[position])  # AFDB convention; high is good
            residue.add_atom(atom)
            written += 1
        if len(residue):
            gemmi_chain.add_residue(residue)
            sequence.append(name3)
    if not written:
        raise ValueError(f"{record.afid}: no atoms to write")

    model.add_chain(gemmi_chain)
    structure.add_model(model)
    structure.setup_entities()
    # The entity needs its sequence, or gemmi emits no _entity_poly_seq and
    # Protenix's build_ref_chain_with_atom_array raises KeyError on the entity
    # id while looking up poly_res_names.
    for entity in structure.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            entity.polymer_type = gemmi.PolymerType.PeptideL
            entity.full_sequence = sequence

    _add_identity_assembly_check(structure, chain)

    document = structure.make_mmcif_document()
    _add_identity_assembly(document.sole_block(), chain)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.write_file(str(path))
    return path


def _add_identity_assembly_check(structure, chain):
    """Fail loudly if the subchain did not take.

    A mismatch between the written ``label_asym_id`` and the asym id the
    assembly lists produces no error anywhere: the parse succeeds, every filter
    reports zero drops, and the atom array comes back empty. So it is checked
    here, where the cause is still visible.
    """
    subchains = {residue.subchain for model in structure for ch in model for residue in ch}
    if subchains != {chain}:
        raise AssertionError(
            f"expected every residue in subchain {chain!r}, got {sorted(subchains)}. "
            "pdbx_struct_assembly_gen would name an asym id that does not exist "
            "and the assembly would expand to nothing."
        )


def _add_identity_assembly(block, chain):  # noqa: D401
    """Add the assembly categories Protenix's parser requires.

    ``gemmi``'s ``make_mmcif_document`` emits no assembly information, and
    Protenix reads ``pdbx_struct_assembly.oligomeric_count`` unguarded -- a
    missing category is a ``KeyError``, not a fallback. ``assembly_gen`` and
    ``oper_list`` do degrade gracefully, but they are what say the deposited
    coordinates *are* the assembly; without them the featurizer would be
    guessing. Real AFDB mmCIFs carry all three, so this restores what the
    builder dropped rather than inventing anything.

    One chain, one identity operator: these are monomer predictions.
    """
    import gemmi

    assembly = block.init_loop(
        "_pdbx_struct_assembly.",
        ["id", "details", "method_details", "oligomeric_details", "oligomeric_count"],
    )
    assembly.add_row(
        ["1", gemmi.cif.quote("author_defined_assembly"), "?", "monomeric", "1"]
    )

    generated = block.init_loop(
        "_pdbx_struct_assembly_gen.", ["assembly_id", "oper_expression", "asym_id_list"]
    )
    generated.add_row(["1", "1", chain])

    # The identity transform, spelled out: biotite's _get_transformations reads
    # every matrix and vector component.
    operators = block.init_loop(
        "_pdbx_struct_oper_list.",
        ["id", "type", "name"]
        + [f"matrix[{i}][{j}]" for i in (1, 2, 3) for j in (1, 2, 3)]
        + [f"vector[{i}]" for i in (1, 2, 3)],
    )
    identity = ["1.0" if i == j else "0.0" for i in (1, 2, 3) for j in (1, 2, 3)]
    # quote(): "identity operation" contains a space, and an unquoted value
    # splits into two, giving the row one more token than the loop has columns.
    # biotite reports that as a DeserializationError on the whole category.
    operators.add_row(
        ["1", gemmi.cif.quote("identity operation"), "1_555"]
        + identity
        + ["0.0", "0.0", "0.0"]
    )


def training_batch(record):
    """One unbatched example with the keys ``pxf.train.step`` requires.

    ``missing_atom_mask`` is derived the same way as for the PDB, and comes out
    all-zero: AFDB predicts every atom of every residue. No per-residue veto is
    applied or needed -- there is no occupancy or altloc to be ambiguous about.
    """
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    aatype = record.aatype.clamp(max=20)
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)
    missing = (exists * (1.0 - record.atom_mask)).clamp(0.0, 1.0)
    return {
        "x": record.x,
        "aatype": aatype,
        "seq_mask": torch.ones(len(record), dtype=torch.float32),
        "missing_atom_mask": missing,
        "residue_index": record.residue_index,
        "chain_index": torch.zeros(len(record), dtype=torch.long),
        "name": record.afid,
    }
