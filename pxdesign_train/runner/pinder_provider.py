"""PINDER holo-dimer provider for the Proteo-AA training pipeline.

PINDER distributes ground-truth dimers as PDB files with receptor/ligand
chains ``R`` and ``L``.  Protenix's training featurizer consumes its enriched
mmCIF representation instead.  This provider performs the official Protenix
PDB-to-mmCIF conversion lazily, keeps on-disk PDB/mmCIF caches, and then
delegates to :class:`CifFileProvider` using the distillation parser because a
PINDER dimer is already an assembled structure. Missing selected PDBs can be
materialized directly from the official ``pdbs.zip`` archive.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Union

from pxdesign_train.runner.cif_provider import CifFileProvider


class PinderPdbProvider:
    """Serve selected PINDER holo dimers as Proteo-AA complex-provider items.

    The manifest must contain ``pinder_id``, ``pdb_path`` and
    ``converted_binder_chain``.  The preparation utility emits ``B`` as the
    converted binder chain: Protenix's converter maps PINDER's input chains
    ``R`` and ``L`` to ``A`` and ``B`` respectively.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        pinder_root: Union[str, Path],
        cif_cache_dir: Union[str, Path],
        archive_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        limit: int = -1,
    ) -> None:
        import pandas as pd

        self.manifest_path = Path(manifest_path).resolve()
        self.pinder_root = Path(pinder_root).resolve()
        self.cif_cache_dir = Path(cif_cache_dir).resolve()
        self.archive_path = Path(
            archive_path or self.pinder_root / "raw" / "pdbs.zip"
        ).resolve()
        self._archive: Optional[zipfile.ZipFile] = None
        self._archive_pid: Optional[int] = None
        columns = [
            "pinder_id",
            "pdb_path",
            "converted_binder_chain",
            "source_split",
        ]
        if self.manifest_path.suffix == ".parquet":
            frame = pd.read_parquet(self.manifest_path, columns=columns)
        else:
            frame = pd.read_csv(self.manifest_path, usecols=columns)
        frame = frame.loc[frame["source_split"].astype(str).eq(split)].reset_index(
            drop=True
        )
        if limit > 0:
            frame = frame.iloc[: int(limit)].copy()
        if frame.empty:
            raise ValueError(
                f"PINDER manifest {self.manifest_path} has no rows for split={split!r}"
            )
        self._pinder_ids = frame["pinder_id"].astype(str).tolist()
        self._pdb_paths = frame["pdb_path"].astype(str).tolist()
        self._binder_chains = frame["converted_binder_chain"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self._pinder_ids)

    def _cached_cif_path(self, idx: int) -> Path:
        pinder_id = self._pinder_ids[idx]
        # Shard by the source PDB ID to avoid a million-entry flat directory.
        pdb_id = pinder_id[:4].lower()
        return self.cif_cache_dir / pdb_id[1:3] / f"{pinder_id}.cif"

    def _archive_handle(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._archive is None or self._archive_pid != pid:
            if self._archive is not None:
                self._archive.close()
            self._archive = zipfile.ZipFile(self.archive_path)
            self._archive_pid = pid
        return self._archive

    def _extract_from_archive(self, idx: int, destination: Path) -> Path:
        if not self.archive_path.is_file():
            return destination
        archive = self._archive_handle()
        pinder_id = self._pinder_ids[idx]
        candidates = (f"pdbs/{pinder_id}.pdb", f"{pinder_id}.pdb")
        member = next((name for name in candidates if name in archive.NameToInfo), None)
        if member is None:
            raise FileNotFoundError(
                f"PINDER dimer {pinder_id!r} is absent from {self.archive_path}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".pdb.tmp.{os.getpid()}")
        try:
            with archive.open(member) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def _ensure_cif(self, idx: int) -> Path:
        cif_path = self._cached_cif_path(idx)
        if cif_path.is_file() and cif_path.stat().st_size > 0:
            return cif_path

        pdb_path = self.pinder_root / self._pdb_paths[idx]
        if not pdb_path.is_file():
            pinder_id = self._pinder_ids[idx]
            pdb_id = pinder_id[:4].lower()
            sharded_path = (
                self.pinder_root / "pdbs" / pdb_id[1:3] / f"{pinder_id}.pdb"
            )
            if sharded_path.is_file():
                pdb_path = sharded_path
            else:
                pdb_path = self._extract_from_archive(idx, sharded_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(
                f"Missing PINDER holo dimer: {pdb_path}. "
                "Extract the selected structures before training."
            )

        from protenix.data.utils import pdb_to_cif

        cif_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cif_path.with_suffix(f".cif.tmp.{os.getpid()}")
        try:
            pdb_to_cif(
                str(pdb_path),
                str(tmp_path),
                entry_id=self._pinder_ids[idx][:4].lower(),
            )
            os.replace(tmp_path, cif_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return cif_path

    def __getitem__(self, idx: int):
        cif_path = self._ensure_cif(idx)
        provider = CifFileProvider(
            [cif_path],
            binder_chain_ids=[self._binder_chains[idx]],
            cache=False,
            dataset="Distillation",
        )
        return provider[0]
