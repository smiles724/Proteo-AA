# Running binder-design training

Stage III co-evolution on protein–protein interfaces. Both modules train
together with the refinement pass live (`S_φ → h_res' → a second backbone/AA
pass`), warm-started from a Stage II side-chain checkpoint plus a separately
trained AA head.

This is the same stage as `slurm_stage3_coevolution_monomer.sh`. The only
difference is the data: `--data-mode mixed_monomer_complex` adds two complex
sources under a curriculum instead of training on monomers alone.

---

## 1. Prerequisites

**Data.** See [`datasets.md`](datasets.md). On this cluster nothing needs
downloading — `/hai/scratch/yfsun` is world-readable. Set the roots, and point
the two things that get WRITTEN at your own scratch:

```bash
export PROTEOAA_DATA_ROOT=/hai/scratch/yfsun          # shared, read-only
export PROTEOAA_CODE_ROOT=/your/code                  # Protenix/ and 11/PXDesign/
export MY_SCRATCH=/hai/scratch/$USER

export RUNS_ROOT=$MY_SCRATCH/proteo_aa_runs
export PINDER_ROOT=$MY_SCRATCH/pinder/2024-02         # provider extracts PDBs here
export PINDER_CIF_CACHE=$MY_SCRATCH/pinder/cif_cache  # and writes mmCIFs here
export PINDER_ARCHIVE=/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip
export PINDER_MANIFEST=/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.parquet
mkdir -p "$PINDER_ROOT" "$PINDER_CIF_CACHE" "$RUNS_ROOT"
```

Skipping the last four is the most common failure: PINDER materialises dimers
lazily and writes as it goes, so a read-only cache dies on the first miss.

**Checkpoints.** Two, and both are required:

| Role | Path (readable at `/hai/scratch/yfsun`) |
|---|---|
| backbone + S_φ | `proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt` |
| AA head | `proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt` |

The script refuses to start without both, on purpose. A Stage II checkpoint
carries a **chance-level** AA head — Stage II excluded the head from its warm
start and then froze it at random init, measured at CE 3.00 (= ln 20) with
`strict_random` scoring the same as `strict_native`. Starting off that head
wastes the run, so it fails loudly instead. See
[`step52500_eval.md`](step52500_eval.md) §3.

---

## 2. Full run

Drop `SMOKE=1`:

```bash
R=/hai/scratch/yfsun/proteo_aa_runs
LOAD_CHECKPOINT=$R/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt \
AA_HEAD_CHECKPOINT=$R/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt \
sbatch scripts/training/slurm_stage3_coevolution_binder.sh
```

Defaults: 30,000 steps, crop 512, lr 1e-5, 500 warmup steps, grad accumulation
8, grad clip 1.0, checkpoint + eval every 2,000 steps, bf16. Logs land in
`logs/training/stage3_binder/`, checkpoints in
`$RUNS_ROOT/stage3_binder_coevolution/<jobid>/checkpoints/`.

### The data mix

| | monomer | protenix_ppi | pinder_ppi |
|---|---|---|---|
| step ≤ 2,000 | 0.50 | 0.15 | 0.35 |
| step ≥ 15,000 | 0.25 | 0.225 | 0.525 |

linearly interpolated in between. Controlled by three variables:

```bash
STAGE2_START_MONOMER_FRAC=0.50   # monomer share at the start
STAGE2_END_MONOMER_FRAC=0.25     # monomer share after the ramp
PINDER_COMPLEX_FRAC=0.70         # PINDER's share of whatever is left for complexes
CURRICULUM_STAGE1_END_STEP=2000
CURRICULUM_STAGE2_START_STEP=15000
```

PINDER carries the larger complex share because 85% of its dimers fit a 640 crop
whole (median 455 tokens, binder 198) against 41% for Protenix complexes (median
756) — `--complex-max-n-token` is only an index filter, the real crop is
`CROP_SIZE`, so an over-long complex is cropped and can lose the interface.
Protenix complexes stay non-trivial because they are the only complex source
carrying MSA features. The monomer share holds the fold prior and is the only
source tied to the 491-protein validation set.

**The ratio is a starting point, not an optimum.** There is no ablation over it.

---

## 3. Selecting a checkpoint

**The `val_*` lines in the training log measure monomers, not binders.**
`build_eval_dataloader` pins the validation set to monomers
(`train_protenix_monomer.py:566`) even in `mixed_monomer_complex`, which is why
`EVAL_SAMPLES` is cut to 128 here. Do not select on them.

Score the binder chain instead, on the PINDER val split (1,810 complexes, one
row per cluster, so non-redundant):

```bash
CHECKPOINT=<ckpt> sbatch scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh
```

Reference from the previous mixed **backbone** run (step50000, mix
0.65/0.175/0.175): binder Cα lDDT 0.698, binder Cα RMSD 2.35 Å, binder TM 0.637.
Measured before the 2026-08-06 leakage fix, so re-measure on current code before
treating it as the number to beat.

Side-chain and AA-head quality on the monomer benchmarks:
`scripts/evaluation/eval_sidechain_arms.py` and
`scripts/evaluation/eval_aa_head_strict_backbone.py` (use `--model-stage
aa_head_on_stage2` for a Stage III-config head).

---

## 4. Known limitations

**PINDER train is cluster-redundant and sampled uniformly over rows.**
1,437,458 rows over 40,231 clusters, largest cluster 82,272 rows. Effective
number of interface families under the current sampling: **78** (inverse
Simpson); the top 10 clusters take 30% of draws. Protenix complexes are milder —
188,277 rows / 25,930 clusters → effective 595. Read the mixture fractions as
shares of *draws*, not of distinct interfaces. The fix is per-item weights of
`1/cluster_size`, which `CurriculumMultiDataset(per_item_weights=...)` already
supports and `train_protenix_monomer.py:472,484` currently hardcodes to uniform.

**PINDER conversion cost.** Only 9,859 of 1.44M systems are pre-converted, at an
observed ~0.42 items/s inside the training loop. Copy the existing cache (3.7 GB
+ 3.6 GB, see [`datasets.md`](datasets.md)) or pre-extract offline.

**`HResInjector` is not zero-initialised**, unlike `ATokenFusion`/`QAtomFusion`.
Stage III warm-starting from Stage II therefore does not begin as a clean no-op:
the refinement pass adds a random projection of `h_res'` into `s_trunk` from step
one. Worth knowing before reading early curves.

**Side-chain quality drops on predicted frames.** The same S_φ scores 1.82 Å²
against GT backbone frames and 5.82 Å² against predicted ones. Stage III is
where that gap has to close; do not expect the Stage II packing number here.

---
