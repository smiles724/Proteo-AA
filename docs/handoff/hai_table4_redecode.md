# HAI: the AlphaProteo benchmark for redecode, 1 event and 4 events

Two full Table 4 rows. **This is the largest thing asked for so far --
roughly 165-270 GPU-h across both protocols.** Read the cost section and
the calibration step before launching all twenty jobs.

## Code

```bash
cd <Proteo-AA-pxdesign-fampnn-pack>
git fetch origin feat/multi-event-feedback
git checkout feat/multi-event-feedback      # 56d39a2 or later
```

`feat/multi-event-feedback` carries both protocols and, critically,
`--resume`. Earlier branches wrote `designs.csv` once at the end, so a job
that hit its time limit lost its entire shard -- unsurvivable at 480 cells.
Do not run this from `feat/post-feedback-redecode`.

Use the reporter from this same branch: it knows `schedule_id` and refuses
to pool protocols.

## Data

Unchanged. The bundle you already have is correct -- re-staging touched only
`repo/HEAD` and `repo/BRANCH`, verified by diffing `SHA256SUMS`. Same donor
`b075867bae942dc0`, FaMPNN 0.3 `8969b3f1f3c94117`, J03 s0
`c506c7e1c43104dd`, s1 `ee673b982c4f5dd7`. Nothing to re-pull.

## The grid

```
10 targets x 6 lengths {80,90,100,110,120,130} x 8 seeds {101..108}
  = 480 cells x 7 arms = 3,360 designs PER PROTOCOL
```

That gives n = 48 per arm per target, matching the published rows. Under
redecode all seven arms emit distinct sequences, so effective n equals the
design count.

## Step 1: calibrate on TNFa. Do not skip this.

PDL1 is the smallest target (116 residues); TNFa is ~438, and the pair stack
is roughly quadratic in tokens. Every cost below rests on a 3-5x scaling
factor that has never been measured. One TNFa cell settles it:

```bash
export REPO=$PWD
export OUT=/hai/scratch/yfsun/pxf_runs/table4_redecode
python scripts/run_integrated_binder_matrix.py \
  --targets-config "$BUNDLE/targets/configs_binder_benchmark/targets.yaml" \
  --prepared-dir "$PREPARED" \
  --checkpoint-dir "$BUNDLE/checkpoints/donors" \
  --checkpoint-selection "$BUNDLE/selection/selected_checkpoints.json" \
  --bs-checkpoint 0="$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt" \
  --bs-checkpoint 1="$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt" \
  --fampnn-checkpoint "$BUNDLE/checkpoints/donors/fampnn_0_3.pt" \
  --fampnn-variant 0.3 \
  --targets TNFa --lengths 130 --seeds 101 \
  --event-sigma 0.429 --sequence-policy post_feedback_redesign \
  --resume --out "$OUT/calibration_tnfa"
```

TNFa at L=130 is the most expensive cell in the grid. Report its arm time
before going further. On PDL1 L=80 the same cell is **110 s**; if TNFa L=130
comes in above ~15 minutes, the full run is a different proposition from
the one costed here and needs re-planning, not launching.

## Step 2: the two protocols, one job per target

Keep them in **separate output roots**. The reporter refuses a CSV mixing
sequence policies or event schedules, and pooling them would be wrong
anyway.

```bash
TARGETS="BHRF1 H1 IL17A IL7RA IR PDL1 SC2RBD TNFa TrkA VEGFA"
COMMON="--targets-config $BUNDLE/targets/configs_binder_benchmark/targets.yaml \
  --prepared-dir $PREPARED \
  --checkpoint-dir $BUNDLE/checkpoints/donors \
  --checkpoint-selection $BUNDLE/selection/selected_checkpoints.json \
  --bs-checkpoint 0=$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt \
  --bs-checkpoint 1=$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt \
  --fampnn-checkpoint $BUNDLE/checkpoints/donors/fampnn_0_3.pt \
  --fampnn-variant 0.3 \
  --lengths 80 90 100 110 120 130 --seeds 101 102 103 104 105 106 107 108 \
  --sequence-policy post_feedback_redesign --resume"

# --- 1 event, the trained sigma
for T in $TARGETS; do
  CMD="scripts/run_integrated_binder_matrix.py" \
  ARGS="$COMMON --targets $T --event-sigma 0.429 --out $OUT/ev1/$T" \
  sbatch --export=ALL,CMD,ARGS --time=12:00:00 \
         scripts/slurm/hai/integrated_official.sh
done

# --- 4 events
for T in $TARGETS; do
  CMD="scripts/run_integrated_binder_matrix.py" \
  ARGS="$COMMON --targets $T --event-sigmas 4.0 2.0 1.0 0.429 --out $OUT/ev4/$T" \
  sbatch --export=ALL,CMD,ARGS --time=24:00:00 \
         scripts/slurm/hai/integrated_official.sh
done
```

**Re-submitting the identical command resumes** -- finished cells are read
from `completed_cells.txt` and skipped, and rows from a cell that was
interrupted mid-way are dropped rather than left partial. Use this rather
than raising the time limit blindly.

The four sigmas resolve to steps **303 / 319 / 334 / 350** on the ladder you
recorded in `2ccefe7`. The schedule comes from the noise scheduler and
`n_step`, not from the target, so those steps are the same for all ten
targets. Check the printed steps on the first job anyway.

`--checkpoint-dir` must contain the Protenix base model, not only
`pxdesign_v0.1.0.pt`. A donors-only directory makes PXDesign attempt a CDN
download; on Marlowe that burned a 40-minute job on a stalled transfer and
left a truncated checkpoint behind. If the bundle's donors directory lacks
the base model, point this at your PXDesign `release_data/checkpoint`.

## Step 3: fold, then report

Per protocol, over the ten per-target directories:

```bash
for T in $TARGETS; do
  RUN_DIR=$OUT/ev1/$T sbatch <proteo-aa>/scripts/evaluation/slurm_fold_af2ig.sh
done
```

Folding is ~0.65 s/design at PDL1 size, so ~1.2 GPU-h per protocol there,
3-5 with target scaling. It is not the bottleneck; generation dominates
about 20:1.

Then the Table 4 row. `scripts/report_table4.py` reads the cached-backbone
collections; it will need its glob pointed at these directories, or use
`scripts/report_integrated_cells.py` per target and assemble. Either way,
**report the two protocols as separate rows** and keep them out of the
existing table's pairing.

## Cost

Measured per cell on Marlowe, PDL1 L=80, seven arms:

| protocol | arm time | 480 cells @ PDL1 | with 3-5x scaling |
|---|---|---|---|
| redecode, 1 event | 110 s | 14.7 GPU-h | **45-75** |
| redecode, 4 events | 267 s | 35.6 GPU-h | **110-180** |
| folding, both | -- | 2.4 GPU-h | 7-12 |
| | | | **~165-270 total** |

Fixed model-load overhead is ~60 s per job and amortises across the 48
cells in a shard, so marginal cost is arm time. (An earlier estimate of
165-245 for the 1-event row alone was wrong -- it paid that overhead 480
times.)

## Two caveats the rows must carry

**Not a sequence-stage swap against the existing rows.** The six published
rows share ONE pre-generated 480-backbone collection across all arms. The
integrated matrix generates its own backbone per cell. So a redecode row
differs from `PXDesign` / `J03` / `U03` in the generator as well as the
sequence stage, and across-row differences confound the two. The clean
internal comparison is redecode-1-event against redecode-4-event, which
share a protocol.

**The 4-event row is off-distribution.** A_BS and the feedback adapters
were trained at ONE event at sigma 0.429. Three of the four injections are
at sigmas they never saw. The run prints `OFF-DISTRIBUTION`. Label the row
as a checkpoint-transfer probe.

## What I would say before you spend this

Every measurement so far argues the result will be null:

- the 4-event residual **does not accumulate** -- at step 350 it is
  26.05-26.57 after three prior corrections against 26.15-26.56 after none,
  and not consistent in sign
- redecode left designability unchanged at n=1 (3/3 -> 7/7, intervals
  overlapping)
- **U03 rises with every other arm** across the three pilot cells
  (ipTM 0.797 -> 0.795 -> 0.832) despite receiving zero injections, which
  points at the entry point rather than the feedback
- seed-to-seed spread exceeds arm-to-arm spread in every cell measured
- J03 - U03 was undetected at n=480 on the cached-backbone path

Two much cheaper experiments would sharpen the decision:

1. **A 4-event cell with `--sequence-policy event_fixed`**, or a U03-only
   entry-sigma sweep, ~1 GPU-minute. If U03 alone improves with an earlier
   entry point, the four-event gain is the entry point and not the
   feedback -- which changes what the big run is even measuring.
2. **The 112-design paired pilot** (2 targets x 8 seeds x 1 length,
   16 independent prefixes), ~0.5 GPU-h at one event.

If you want the rows regardless, everything above is ready to run.
