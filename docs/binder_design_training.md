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

## 2. Smoke first

Six steps at a small crop, ~90 seconds. Run it before every 24h slot, and
re-run it at your intended `CROP_SIZE` — the refinement pass runs the backbone
twice, so peak memory is ~2× Stage II at the same crop.

```bash
R=/hai/scratch/yfsun/proteo_aa_runs
SMOKE=1 CROP_SIZE=512 \
LOAD_CHECKPOINT=$R/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt \
AA_HEAD_CHECKPOINT=$R/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt \
sbatch scripts/training/slurm_stage3_coevolution_binder.sh
```

A healthy smoke log shows all four:

```
Model has 262.05M parameters (262.05M trainable)
Alternating training: 116.56M side-chain params, 145.49M backbone-group params
Loaded .../step52500.pt (missing=21, unexpected=0)
Overlaid 10 AA head tensor(s) from .../step9000.pt
step=1 ... sc_local=4.097 ... bb_post=41.64
```

`missing=21` is expected — the co-evolution modules (`a_token_fusion_pre`,
`q_atom_fusion`, `hres_injector`, `refinement_pass_embedding`) do not exist in a
Stage II checkpoint and are created fresh here. **`unexpected=0` is the one to
watch**: anything else means the model you built is not the one the checkpoint
was trained as. A non-zero `bb_post` confirms the refinement pass is actually
running.

---

## 3. Full run

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

### Common overrides

```bash
CROP_SIZE=384                    # first thing to drop if you OOM
MAX_STEPS=50000
LR=2e-5
COMPLEX_PROVIDER=pinder          # or protenix; default both
STAGE2_END_MONOMER_FRAC=0.10     # more aggressively binder-focused
RUN_ROOT=$RUNS_ROOT/my_named_run
```

---

## 4. Selecting a checkpoint

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

## 5. Known limitations

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

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `mkdir: cannot create directory '/var/lib/slurm/logs'` | `REPO_ROOT` resolved to the sbatch spool copy. Submit from the repo root or pass `REPO_ROOT=`. Current script probes for the checkout, so this should not recur. |
| `ERROR: could not locate the Proteo-AA checkout` | none of the script's dir, `SLURM_SUBMIT_DIR`, `$PWD` is a checkout. Pass `REPO_ROOT=`. |
| `Permission denied` writing a `.cif` under `cif_cache` | using the shared PINDER tree read-only. Set `PINDER_ROOT` and `PINDER_CIF_CACHE` to your own scratch. |
| `ERROR: missing directory .../protenix_data` | `PROTEOAA_DATA_ROOT` wrong. |
| `AttributeError: 'Namespace' object has no attribute '<flag>'` | an eval script's parser is missing a flag `build_configs`/`build_components` reads. Call `fill_missing_args(args)` after its `parse_args()`. |
| `unexpected=N` (N>0) on load | model built differs from the checkpoint. For an already-fitted model, reconcile with `adopt_feedback_channels_from_checkpoint`. |
| OOM early in training | lower `CROP_SIZE` (512 → 384). The refinement pass doubles the backbone's activation memory. |
