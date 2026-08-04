# Script layout

- `training/`: Protenix monomer and complex training drivers and Slurm launchers.
- `evaluation/`: checkpoint selection, monomer/AA-head evaluation, and ProteinMPNN recovery.
- `sidechain/`: side-chain table builders and template-quality evaluation.
- `plotting/`: training-log plotting utilities.
- `utilities/`: setup, checkpoint checks, smoke tests, and external-tool bootstrap scripts.
- `examples/`: minimal training and fine-tuning examples.
- `data/`: dataset preparation utilities, including PINDER manifest generation
  and selected-structure extraction.

Run scripts from the repository root so their documented relative paths and
default log locations resolve consistently.

For the prepared PINDER 2024-02 source, set `COMPLEX_PROVIDER=pinder` when
submitting either stage-2 launcher. The default paths are
`/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.parquet` and
`/hai/scratch/yfsun/pinder/2024-02`. Selected PDBs are read from `pdbs/` when
already extracted, or materialized lazily from `raw/pdbs.zip`; override these
locations with `PINDER_MANIFEST`, `PINDER_ROOT`, and `PINDER_ARCHIVE` if needed.
Set `EXTRACT_SELECTED=1` when running `data/download_pinder_2024_02.sh` to
pre-extract every selected dimer instead of relying on lazy materialization.
