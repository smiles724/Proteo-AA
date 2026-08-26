# Datasets — what binder-design training reads, and how to get it

Everything below is addressed through **two** environment variables. Set these
and no script needs editing:

```bash
export PROTEOAA_DATA_ROOT=/your/scratch          # parent of every dataset
export PROTEOAA_CODE_ROOT=/your/code             # parent of the Protenix + PXDesign checkouts
```

Expected layout under `$PROTEOAA_DATA_ROOT`:

```
$PROTEOAA_DATA_ROOT/
├── protenix_data/          # Protenix preprocessed wwPDB  — REQUIRED
├── pinder/2024-02/         # PINDER holo dimers           — required for binder training
└── proteo_aa_runs/         # outputs; created by the scripts
```

and under `$PROTEOAA_CODE_ROOT`: `Protenix/` and `11/PXDesign/`.

Every individual path stays overridable (`DATA_ROOT`, `PINDER_ROOT`,
`PINDER_MANIFEST`, `PINDER_CIF_CACHE`, `PINDER_ARCHIVE`, `RUNS_ROOT`,
`PROTENIX_CODE_DIR`, `PXDESIGN_CODE_DIR`, `PYTHON_BIN`) for layouts that don't
nest this way.

---

## Already on this cluster — skip the download

`/hai/scratch/yfsun` is world-readable, so anyone with a cluster account can use
the existing 1.4 TB copy instead of re-downloading it. **Do not copy it.** Point
at it and override only the two things training needs to WRITE:

```bash
export PROTEOAA_DATA_ROOT=/hai/scratch/yfsun      # shared, read-only to you
export MY_SCRATCH=/hai/scratch/$USER              # your own space

# writes go to your scratch, reads come from the shared copy
export RUNS_ROOT=$MY_SCRATCH/proteo_aa_runs
export PINDER_ROOT=$MY_SCRATCH/pinder/2024-02     # provider extracts PDBs here
export PINDER_CIF_CACHE=$MY_SCRATCH/pinder/cif_cache
export PINDER_ARCHIVE=/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip        # 157 GB, shared
export PINDER_MANIFEST=/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.parquet
mkdir -p "$PINDER_ROOT" "$PINDER_CIF_CACHE" "$RUNS_ROOT"
```

**Why the PINDER overrides are not optional.** `PinderPdbProvider` materialises
dimers lazily — it extracts each PDB from the archive into `<pinder_root>/pdbs/`
and writes the converted mmCIF into `<cif_cache>/` on first use. Left pointing at
the shared copy, training dies on the first cache miss with a permission error,
and since only 9,859 of 1.44M systems are pre-converted, that is almost
immediately. Everything else (`protenix_data/`, the manifest, `pdbs.zip`,
checkpoints) is read-only and can stay shared.

Optional head start: copy the 9,859 conversions already done rather than redoing
them at ~0.42 items/s.

```bash
cp -r /hai/scratch/yfsun/pinder/2024-02/cif_cache/. "$PINDER_CIF_CACHE"/   # 3.7 GB
cp -r /hai/scratch/yfsun/pinder/2024-02/pdbs/.      "$PINDER_ROOT"/pdbs/   # 3.6 GB
```

Checkpoints are readable at the same root and are **not** part of any public
download — they are this project's own output:

```
/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt
/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt
```

Code is not shared this way — clone the Proteo-AA repo and the public Protenix
and PXDesign repos yourself, then set `PROTEOAA_CODE_ROOT`.

The rest of this document covers building the datasets from scratch, for anyone
off this cluster.

---

## What each source is

| Source | Size | Rows used | Needed for | Origin |
|---|---|---|---|---|
| **Protenix preprocessed wwPDB** (`protenix_data/`) | ~1.2 TB | 64,040 monomers + 188,277 PPI complexes | everything | ByteDance Protenix release |
| **PINDER 2024-02** (`pinder/2024-02/`) | ~165 GB | 1,437,458 train dimers | binder / complex training only | PINDER (`storage.googleapis.com/pinder`) |
| Derived indices | ~1 MB | — | built automatically | this repo, into `<run>/cache/` |

Both monomer and PPI-complex training rows come out of the **same** Protenix
directory — they are two different filters over
`indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz`,
not two downloads. PINDER is the only additional download binder training needs.

---

## 1. Protenix preprocessed wwPDB — required

This is the base dataset for every stage, monomer or binder. Download it with
Protenix's own script, from the Protenix checkout:

```bash
cd $PROTEOAA_CODE_ROOT/Protenix
export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
mkdir -p "$PROTENIX_ROOT_DIR"
bash scripts/database/download_protenix_data.sh --full      # --full, NOT the default --inference_only
```

`--full` matters: the default `--inference_only` omits `mmcif_bioassembly` and
`mmcif_msa_template`, which is everything training reads. Protenix's own docs
budget ≥1.5 TB of disk for this step.

The directories the training path actually touches:

| Path | Used for |
|---|---|
| `indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz` | source index for both the monomer and PPI-complex filters |
| `indices/recentPDB_low_homology_maxtoken1536.csv` | the 491-protein validation set |
| `mmcif_bioassembly/` | structures |
| `mmcif_msa_template/` | MSA / template features |
| `common/` | CCD components, sequence→PDB map, release dates |
| `mmcif/` | raw mmCIF (CASP scoring path) |

`rna_msa/`, `posebusters_*`, `recentPDB_bioassembly/` and `search_database/`
arrive with the download but are not read by binder training.

**Note on the training cutoff.** The index is `before_2021-09-30`. Anything you
later evaluate that predates that date is potentially in-distribution — this is
why the CASP14 numbers in `step52500_eval.md` carry a contamination caveat.

## 2. PINDER 2024-02 — required for binder design

Curated holo protein–protein dimers with a defined receptor/ligand split, which
is the shape of the binder-design task. One command, already path-configurable:

```bash
bash scripts/data/download_pinder_2024_02.sh
```

It fetches `index.parquet`, `metadata.parquet` and `pdbs.zip` (~169 GB) from
`https://storage.googleapis.com/pinder/2024-02`, checks `pdbs.zip` against md5
`28e6e2724c1597514c7f6b6774ff32b4`, and then runs
`scripts/data/prepare_pinder_dataset.py` to build the training manifest:

```
pinder/2024-02/
├── raw/{index,metadata}.parquet, pdbs.zip      # official release
├── indices/pinder_ppi_complex.parquet          # the manifest training reads
├── manifests/dataset_statistics.json
├── pdbs/                                       # extracted PDBs (lazy)
└── cif_cache/                                  # PDB→mmCIF conversions (lazy)
```

The manifest is filtered to `min_n_token=16`, `max_n_token=1536`,
`crop_size=640`, `max_binder_fraction=0.75`, and carries `cluster_id`,
`source_split`, `release_date`, `resolution`, `binder_tokens`,
`contains_antibody/antigen/enzyme`, `buried_sasa` and
`intermolecular_contacts` — all available for per-item weighting.

Splits: **train 1,437,458 / val 1,810 / test 1,768.** Val and test are one row
per cluster, so they are non-redundant; train is not (see the warning below).

### Lazy materialisation — plan for it

`pdbs.zip` is *not* unpacked by default. `PinderPdbProvider` extracts and
converts each dimer to mmCIF on first use, into `pdbs/` and `cif_cache/`. On
this machine only **9,859 of 1.44M** are cached, converted inside the training
loop at an observed ~0.42 items/s.

To pre-materialise instead of paying it as training stalls:

```bash
EXTRACT_SELECTED=1 bash scripts/data/download_pinder_2024_02.sh
```

Budget disk for it: the 9,859 currently cached account for a large part of the
165 GB the `pinder/` tree already occupies, so a full extraction is
substantially more than `pdbs.zip` alone. Extract the subset you will actually
sample, not all 1.44M.

### ⚠ Train split is cluster-redundant

1,437,458 rows over **40,231 clusters**; the largest cluster is 82,272 rows.
Under the uniform row sampling the trainer currently does
(`train_protenix_monomer.py:472,484`), the effective number of interface
families is **78** (inverse Simpson) and the top 10 clusters take 30% of draws.
Protenix complexes are milder: 188,277 rows / 25,930 clusters → effective 595.
Weighting rows by `1/cluster_size` fixes it and is already supported by
`CurriculumMultiDataset(per_item_weights=...)`; it is simply not set yet.

---

## 3. Present on disk but NOT used by training

| Path | Size | Why it isn't in the mix |
|---|---|---|
| `ppdiff_data/` (PPDiff PPBench) | 12 GB | 701,709 PPI rows, but **Cα-only** — cannot supervise side chains in a full-atom pipeline — and it has an index prep script only, no provider in `pxdesign_train/runner/`. |
| `casp14/`, `casp15/`, `casp16/` | 82 MB | evaluation only (`eval_casp14_sidechain.py`) |
| `protenix_v1_20240522/`, `data/`, `unused_data/` | 1.9 TB | older snapshots, not referenced by current scripts |

---

## Disk budget

| Item | Size |
|---|---|
| `protenix_data/` | ~1.2 TB (Protenix says budget 1.5 TB) |
| `pinder/2024-02/` | ~165 GB, plus growth as `cif_cache/` fills |
| `proteo_aa_runs/` | grows fast — each checkpoint is 1.0–2.0 GB |

**~1.4 TB before any training output.**

---

## Verify a fresh install

```bash
test -f "$PROTEOAA_DATA_ROOT/protenix_data/indices/recentPDB_low_homology_maxtoken1536.csv" && echo "protenix indices OK"
test -d "$PROTEOAA_DATA_ROOT/protenix_data/mmcif_bioassembly"                              && echo "protenix structures OK"
test -f "$PROTEOAA_DATA_ROOT/pinder/2024-02/indices/pinder_ppi_complex.parquet"            && echo "pinder manifest OK"
test -f "$PROTEOAA_DATA_ROOT/pinder/2024-02/raw/pdbs.zip"                                  && echo "pinder archive OK"
```

Then prove the whole path end to end — assembly, both warm starts, the data mix
and the refinement pass — in about 90 seconds on one GPU:

```bash
R=$PROTEOAA_DATA_ROOT/proteo_aa_runs
SMOKE=1 \
LOAD_CHECKPOINT=$R/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt \
AA_HEAD_CHECKPOINT=$R/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt \
sbatch scripts/training/slurm_stage3_coevolution_binder.sh
```

Checkpoints are **not** part of either public download — they are this project's
own training output and have to be copied across with the data.
