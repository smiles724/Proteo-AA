"""Protenix mmCIF entries plus per-residue side-chain supervision masks.

Two independent masks decide what the side-chain objective may look at, and the
whole point of this module is that they **compose** rather than replace one
another:

``missing_atom_mask`` (FaMPNN's own, per **atom**)
    Built from atom presence in atom37: ``(1 - all_atom_mask) * (1 -
    ghost_atom_mask)``. It knows that Ser has no CG and that residue 41's OE1 was
    never resolved. It does not know anything about crystallography.

``supervise_mask`` (ours, per **residue**)
    From ``/hai/scratch/yfsun/protenix_sidechain``. Clears residues whose
    side-chain coordinates are *present but not trustworthy*: zero or partial
    occupancy, an alternate-conformer tie, a side-chain mean B above 80, a failed
    chirality or CA-CB bond check. FaMPNN derives ``all_atom_mask`` from presence
    alone and never reads occupancy or altloc, so this is genuinely additive.

:func:`veto_unsupervised_sidechains` multiplies the second into the first. It
never clears a bit the per-atom mask set, and it never touches a backbone slot --
a residue with an unreliable side chain still has a usable backbone, and taking
its N/CA/C away would destroy the local frame and drop the residue from ``L_MLM``
as well.

**Alignment is positional and unforgiving.** Mask element ``i`` is the ``i``-th
standard amino-acid residue of the entry under ``gemmi`` model 0, chains in file
order, skipping waters, ligands, nucleotides and modified residues. The
iteration is imported from the mask builder's own ``sc_masks`` module rather than
reimplemented here, and :func:`read_entry` asserts the resulting length against
the mask's recorded ``mask_len``, so a divergence is an error rather than a
silent off-by-one that shifts every label.

**Splits.** The masks were built from Protenix's
``weightedPDB_indices_before_2021-09-30`` index -- all 165,470 entries lie inside
it. The evaluation index ``recentPDB_low_homology_maxtoken1536.csv`` (1,818
entries, released 2022-05-04 to 2023-01-11) has an empty intersection with it, so
the temporal split is clean; it also means the eval entries have **no mask in
this set** and need their own pass of the builder.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_SC_MASKS_DIR = pathlib.Path("/hai/scratch/yfsun/protenix_sidechain")
DEFAULT_MASK_ROOT = DEFAULT_SC_MASKS_DIR / "out_fampnn_strictB"
DEFAULT_MMCIF_DIR = pathlib.Path("/hai/scratch/yfsun/protenix_data/mmcif")
# ``sc_masks.iter_standard_residues`` imports afdb_laproteina.constants.
DEFAULT_AFDB_SRC = pathlib.Path("/hai/users/y/f/yfsun/afdb-laproteina/src")

# The column to filter entries on in the FaMPNN variant. The builder writes both
# `keep` (inherited) and `keep_fampnn`; the shipped SideChainMasks loader
# auto-detects and lands on `keep`, which is not this variant's column. They
# happen to agree today (161,537 each) -- naming it explicitly is what keeps a
# regenerated variant from silently changing the training set.
KEEP_COLUMN = "keep_fampnn"


def _sc_masks_module(sc_masks_dir=None, afdb_src=None):
    """Import the builder's own loader, so alignment has a single definition."""
    for path in (sc_masks_dir or DEFAULT_SC_MASKS_DIR, afdb_src or DEFAULT_AFDB_SRC):
        path = str(pathlib.Path(path))
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import sc_masks
    except ImportError as error:  # pragma: no cover - environment problem
        raise ImportError(
            "could not import sc_masks; pass sc_masks_dir pointing at the "
            f"directory holding sc_masks.py (tried {sc_masks_dir or DEFAULT_SC_MASKS_DIR})"
        ) from error
    return sc_masks


class SideChainMaskSet:
    """The per-residue supervision masks, filtered on :data:`KEEP_COLUMN`.

    Wraps the builder's ``SideChainMasks`` for the memmap and offset arithmetic
    and applies the ``keep_fampnn`` filter on top, because that loader's own
    column auto-detection selects ``keep``.
    """

    def __init__(self, root=None, *, sc_masks_dir=None, afdb_src=None, keep_column=None):
        self.root = pathlib.Path(root or DEFAULT_MASK_ROOT)
        self.keep_column = keep_column or KEEP_COLUMN
        module = _sc_masks_module(sc_masks_dir, afdb_src)
        self._bits = module

        # Read the index first: wrong-variant is cheaper to detect than a
        # 288 MB memmap is to open, and the column is what identifies the variant.
        entries = self.root / "entries.csv"
        with entries.open() as stream:
            reader = csv.DictReader(stream)
            if self.keep_column not in (reader.fieldnames or ()):
                raise ValueError(
                    f"{entries} has no {self.keep_column!r} column (found "
                    f"{reader.fieldnames}). This variant may not be the FaMPNN one."
                )
            self.kept, self.lengths = set(), {}
            for row in reader:
                if row["error"] or int(row[self.keep_column]) != 1:
                    continue
                self.kept.add(row["pdb_id"])
                self.lengths[row["pdb_id"]] = int(row["mask_len"])

        self._masks = module.SideChainMasks(self.root, keep_only=False)
        self.meta = self._masks.meta
        # Drop anything the wrapped loader indexed but this column excludes.
        self.index = {k: v for k, v in self._masks.index.items() if k in self.kept}

    def __len__(self):
        return len(self.index)

    def __contains__(self, pdb_id):
        return pdb_id in self.index

    def pdb_ids(self):
        return sorted(self.index)

    def bits(self, pdb_id):
        """Raw ``uint16`` per residue: why each one was excluded."""
        if pdb_id not in self.index:
            raise KeyError(f"{pdb_id} is not in {self.root} under {self.keep_column}=1")
        offset, length = self.index[pdb_id]
        return np.asarray(self._masks._masks[offset : offset + length])

    def supervise_mask(self, pdb_id):
        """``[n_res]`` bool, True where the side chain may be supervised."""
        return torch.from_numpy((self.bits(pdb_id) & self._bits.SUPERVISE) != 0)

    def reasons(self, pdb_id):
        """Counts per exclusion bit, for reporting what a filter actually cost."""
        bits = self.bits(pdb_id)
        names = (
            "MISSING_SC",
            "ZERO_OCC",
            "PARTIAL_OCC",
            "ALTLOC",
            "ALTLOC_TIE",
            "EXTREME_B",
            "NO_SIDECHAIN",
            "CHIRALITY_BAD",
            "BOND_OUTLIER",
        )
        return {name: int((bits & getattr(self._bits, name) != 0).sum()) for name in names}

    def identity(self):
        """What a run should record about the mask set it trained against."""
        variant = dict(self.meta.get("fampnn_variant", {}))
        return dict(
            root=str(self.root),
            keep_column=self.keep_column,
            entries=len(self),
            blockers=variant.get("blockers"),
            dropped_extreme_b=variant.get("dropped_extreme_b"),
            supervised_residues=variant.get("supervised_fampnn"),
            derived_from=self.meta.get("derived_from"),
        )


@dataclass
class ParsedEntry:
    """One mmCIF entry in atom37, aligned with its supervision mask."""

    pdb_id: str
    aatype: torch.Tensor  # [L] int64, FaMPNN restype order
    x: torch.Tensor  # [L, 37, 3] float32, Angstroms
    atom_mask: torch.Tensor  # [L, 37] float32, 1 where an atom is present
    residue_index: torch.Tensor  # [L] int64
    chain_index: torch.Tensor  # [L] int64
    supervise: torch.Tensor | None = None  # [L] bool, or None when unmasked

    def __len__(self):
        return int(self.aatype.shape[0])


def read_entry(pdb_id, *, mmcif_dir=None, masks=None, sc_masks_dir=None, afdb_src=None):
    """Read ``<pdb_id>.cif`` into atom37, in the mask's own residue order.

    ``masks`` is an optional :class:`SideChainMaskSet`; when given, the parsed
    residue count is checked against its recorded ``mask_len`` and the resulting
    ``supervise`` vector is attached. That check is the alignment contract: the
    masks are positional, so a length mismatch means every label is shifted.
    """
    import gemmi

    from fampnn.data import residue_constants as rc

    module = _sc_masks_module(sc_masks_dir, afdb_src)
    # sc_masks imports this inside iter_standard_residues, so there is no
    # module-level alias to borrow; the same table is what defines "standard".
    from afdb_laproteina import constants as C

    path = pathlib.Path(mmcif_dir or DEFAULT_MMCIF_DIR) / f"{pdb_id}.cif"
    if not path.is_file():
        raise FileNotFoundError(path)

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_hydrogens()
    residues = [(chain, res) for chain, res in module.iter_standard_residues(structure)]
    length = len(residues)
    if length == 0:
        raise ValueError(f"{pdb_id}: no standard amino-acid residues")

    if masks is not None:
        expected = masks.lengths.get(pdb_id)
        if expected is None:
            raise KeyError(f"{pdb_id} has no mask in {masks.root}")
        if expected != length:
            raise ValueError(
                f"{pdb_id}: parsed {length} standard residues but the mask records "
                f"{expected}. The masks are positional, so this would shift every "
                "label; check the gemmi version and the setup_entities/"
                "remove_hydrogens order."
            )

    x = torch.zeros(length, 37, 3, dtype=torch.float32)
    atom_mask = torch.zeros(length, 37, dtype=torch.float32)
    aatype = torch.empty(length, dtype=torch.int64)
    residue_index = torch.empty(length, dtype=torch.int64)
    chain_index = torch.empty(length, dtype=torch.int64)
    chain_ids = {}

    for i, (chain, res) in enumerate(residues):
        one = C.RESTYPE_3TO1[res.name]
        aatype[i] = rc.restype_order_with_x[one]
        residue_index[i] = int(res.seqid.num)
        chain_index[i] = chain_ids.setdefault(chain.name, len(chain_ids))
        for atom in res:
            slot = rc.atom_order.get(atom.name)
            if slot is None:  # OXT and anything else outside atom37
                continue
            x[i, slot] = torch.tensor(
                [atom.pos.x, atom.pos.y, atom.pos.z], dtype=torch.float32
            )
            atom_mask[i, slot] = 1.0

    return ParsedEntry(
        pdb_id=pdb_id,
        aatype=aatype,
        x=x * atom_mask[..., None],  # zero absent slots, as process_single_pdb does
        atom_mask=atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        supervise=masks.supervise_mask(pdb_id) if masks is not None else None,
    )


def missing_atom_mask_from_presence(aatype, atom_mask):
    """FaMPNN's own per-atom mask: exists for this restype but is not present."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype.long())
    return (exists * (1.0 - atom_mask)).clamp(0.0, 1.0)


def veto_unsupervised_sidechains(missing_atom_mask, supervise, *, aatype=None):
    """Fold the per-residue veto into the side-chain slots of the per-atom mask.

    Returns a new mask in which every side-chain slot of an unsupervised residue
    reads as missing. Downstream this is exactly the intent: ``encoder_inputs``
    multiplies by ``1 - missing_atom_mask``, so the untrustworthy side chain stops
    being fed to the encoder as context, and ``sidechain_targets`` multiplies by
    the same, so it stops being a target. Both are wanted -- a side chain that is
    too ambiguous to score is also too ambiguous to condition on.

    Composition, never substitution. The result is ``>=`` the input everywhere, so
    nothing the per-atom mask marked missing is ever revived, and only the 33
    non-backbone slots are touched: a residue with an unreliable side chain keeps
    its N/CA/C/O, hence its local frame and its ``L_MLM`` label.
    """
    from fampnn.data import residue_constants as rc

    missing = missing_atom_mask.clone().float()
    veto = (~supervise.bool()).to(missing.dtype)
    while veto.dim() < missing.dim() - 1:
        veto = veto.unsqueeze(0)
    sidechain = list(rc.non_bb_idxs)
    missing[..., sidechain] = torch.maximum(
        missing[..., sidechain], veto.unsqueeze(-1).expand_as(missing[..., sidechain])
    )
    if aatype is not None:
        # Ghost slots stay ghosts: an atom that does not exist for this residue
        # type must not be reported as a missing one.
        from fampnn.data.data import get_rc_tensor

        exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype.long())
        missing = missing * exists
    if not bool((missing >= missing_atom_mask.float() - 1e-6).all()):
        raise AssertionError("the veto cleared a bit the per-atom mask had set")
    return missing


def training_batch(entry, *, apply_supervision=True):
    """One unbatched example with the keys ``pxf.train.step`` requires.

    With ``apply_supervision`` the per-residue veto is folded in; without it, the
    example carries FaMPNN's per-atom mask alone, which is the ablation that shows
    what the crystallographic filter is worth.
    """
    missing = missing_atom_mask_from_presence(entry.aatype, entry.atom_mask)
    if apply_supervision:
        if entry.supervise is None:
            raise ValueError(
                f"{entry.pdb_id} was read without a mask set, so supervision cannot "
                "be applied; pass masks= to read_entry or apply_supervision=False"
            )
        missing = veto_unsupervised_sidechains(
            missing, entry.supervise, aatype=entry.aatype
        )
    return {
        "x": entry.x,
        "aatype": entry.aatype,
        "seq_mask": torch.ones(len(entry), dtype=torch.float32),
        "missing_atom_mask": missing,
        "residue_index": entry.residue_index,
        "chain_index": entry.chain_index,
        "name": entry.pdb_id,
    }


class ProtenixSideChainDataset(torch.utils.data.Dataset):
    """Protenix entries as fixed-size training examples, masks folded in.

    Cropping, noise and padding are reused from :mod:`pxf.train.data`, so an
    example from here is interchangeable with one from ``StructureCropDataset``
    and ``collate`` accepts either.

    Order of operations matters in one place: the per-residue veto is applied
    **before** the crop, on the full-length entry, because the mask is indexed by
    full-length residue position. Cropping first and then masking would apply the
    mask at the wrong offsets.

    Args:
        pdb_ids: the entries to use. No default -- the split is the caller's.
        masks: a :class:`SideChainMaskSet`, or None to train on FaMPNN's per-atom
            mask alone (the ablation).
        crop_size: fixed example size, padded when the entry is shorter.
        noise: Angstroms of iid coordinate noise (Appendix B.1).
        spatial_crop_p: probability of an interface crop for two-chain entries.
    """

    def __init__(
        self,
        pdb_ids,
        *,
        masks=None,
        mmcif_dir=None,
        crop_size=256,
        noise=0.0,
        noise_targets=True,
        spatial_crop_p=0.5,
        seed=0,
        apply_supervision=True,
    ):
        pdb_ids = [str(p) for p in pdb_ids]
        if not pdb_ids:
            raise ValueError("ProtenixSideChainDataset needs at least one pdb_id")
        if apply_supervision and masks is None:
            raise ValueError(
                "apply_supervision=True needs a SideChainMaskSet; pass masks=, or "
                "apply_supervision=False to train on the per-atom mask alone"
            )
        if masks is not None:
            unknown = [p for p in pdb_ids if p not in masks]
            if unknown:
                raise ValueError(
                    f"{len(unknown)} id(s) have no mask under "
                    f"{masks.keep_column}=1, e.g. {unknown[:5]}. The mask set covers "
                    "the before-2021-09-30 training index only."
                )
        self.pdb_ids = pdb_ids
        self.masks = masks
        self.mmcif_dir = pathlib.Path(mmcif_dir or DEFAULT_MMCIF_DIR)
        self.crop_size = int(crop_size)
        self.noise = float(noise)
        self.noise_targets = bool(noise_targets)
        self.spatial_crop_p = float(spatial_crop_p)
        self.seed = int(seed)
        self.apply_supervision = bool(apply_supervision)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.pdb_ids)

    def _generator(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * self.epoch + index)
        return generator

    def __getitem__(self, index):
        from fampnn.data import residue_constants as rc
        from pxf.train import data as train_data

        generator = self._generator(index)
        pdb_id = self.pdb_ids[index]
        entry = read_entry(pdb_id, mmcif_dir=self.mmcif_dir, masks=self.masks)
        example = training_batch(entry, apply_supervision=self.apply_supervision)
        name = example.pop("name")
        length = int(example["aatype"].shape[0])

        chains = torch.unique(example["chain_index"])
        if (
            len(chains) == 2
            and float(torch.rand((), generator=generator)) < self.spatial_crop_p
        ):
            indices = train_data.spatial_crop(
                example["x"][:, rc.atom_order["CA"]],
                example["chain_index"],
                self.crop_size,
                generator=generator,
            )
        elif len(chains) == 2:
            counts = [int((example["chain_index"] == c).sum()) for c in chains]
            indices = train_data.multimer_contiguous_crop(
                counts, self.crop_size, generator=generator
            )
        else:
            indices = train_data.contiguous_crop(
                length, self.crop_size, generator=generator
            )

        item = train_data.pad_or_crop(example, indices, self.crop_size)
        item = {k: v[0] if torch.is_tensor(v) and v.dim() else v for k, v in item.items()}
        if self.noise:
            noised = train_data.add_structural_noise(
                item["x"], self.noise, generator=generator
            )
            if self.noise_targets:
                item["x"] = noised
            else:
                item["x_input"] = noised
        item["name"] = name
        return item


def ids_from_index(path, *, column="pdb_id", lower=True):
    """Unique ids from a Protenix index CSV, in sorted order.

    Reads ``.csv`` and ``.csv.gz``. Used for both splits: the training index
    ``weightedPDB_indices_before_2021-09-30_...csv.gz`` and the evaluation index
    ``recentPDB_low_homology_maxtoken1536.csv``.
    """
    import gzip

    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} has no {column!r} column")
        ids = {row[column].lower() if lower else row[column] for row in reader}
    return sorted(i for i in ids if i)
