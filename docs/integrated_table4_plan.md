# Getting redecode into the AlphaProteo Table 4

The six rows in `alphaproteo_table4_combined.md` all come from the
**cached-backbone** path: a finished PXDesign backbone in,
`design_binder_matrix.py` designs a sequence on it. Redecode does not exist
there -- there is no trajectory to re-enter. A redecode row means running
the **integrated** matrix over the same grid.

## The grid

n = 48 per arm per target, matching the existing rows:

```
10 targets x 6 lengths {80,90,100,110,120,130} x 8 generation seeds
  = 480 cells x 7 arms = 3,360 designs
```

Under redecode all seven arms emit distinct sequences, so effective n is
3,360 -- unlike `event_fixed`, where the three arms in an A_BS seed group
share one decode and effective n is 3/7 of that.

## Cost, from measured cells

Marlowe, PDL1 L=80, seven arms, per cell:

| protocol | arm time | job wall | fixed overhead |
|---|---|---|---|
| `event_fixed`, 1 event | 40 s | 1 m 34 s | ~54 s |
| **redecode, 1 event** | **110 s** | 2 m 52 s | ~62 s |
| redecode, 4 events | 267 s | 5 m 25 s | ~58 s |

The ~55-60 s fixed cost is model loading and amortises across cells in one
job, so the marginal cost of a cell is its arm time.

**480 cells at PDL1 size:** 480 x 110 s = **14.7 GPU-h** for redecode
(5.3 h for `event_fixed`, 35.6 h for four events).

**Then target scaling, which is the uncertain part.** PDL1 is the smallest
target at 116 residues; TNFa is ~438. PDL1 L=80 is 196 tokens, TNFa L=80 is
~518, and the pair stack is roughly quadratic -- about 7x on that pair
alone. Averaged over the ten targets a 3-5x multiplier is plausible, giving
**45-75 GPU-h** for a redecode row.

HAI's independent estimate for `event_fixed` was 60-90 GPU-h, derived from
wall-per-cell rather than marginal arm time -- i.e. assuming one job per
cell and paying the fixed overhead 480 times. Which number applies depends
entirely on how the run is sharded. **Do not plan against either without
measuring one large target first**; TNFa at one length and one seed costs
minutes and removes the guess.

Folding is unaffected: ~0.65 s/design at PDL1 size, ~1.1 GPU-h for 3,360,
3-5 h with scaling. Generation dominates roughly 20:1.

## What had to change first

`--resume`, added here. `designs.csv` was written once at the end, so a job
that hit its time limit lost its entire shard. At 480 cells that is not
survivable. Now every cell is flushed and fsynced as it finishes, then
recorded in `completed_cells.txt`; `--resume` skips recorded cells and
keeps their rows.

Rows whose cell is **not** marked finished are dropped on resume. A job
killed mid-cell leaves fewer arms than the cell should have, and keeping
them would put a silently incomplete cell in the table. The marker is
written after the rows for the same reason -- reversed, a crash between
them would mark a cell done whose rows never landed.

## Sharding

One job per target, ten jobs, each 48 cells:

```bash
for T in BHRF1 H1 IL17A IL7RA IR PDL1 SC2RBD TNFa TrkA VEGFA; do
  CMD="scripts/run_integrated_binder_matrix.py" \
  ARGS="--targets-config configs/binder_benchmark/targets.yaml \
        --prepared-dir $DATA/runs/binder_bench/targets/configs \
        --checkpoint-dir $PRISTINE/release_data/checkpoint \
        --checkpoint-selection $DATA/runs/integrated_feedback_v1/evaluation/selected_checkpoints.json \
        --bs-checkpoint 0=$DATA/runs/bs_seq_sc/J03_seed0/checkpoints/step00000500.pt \
        --bs-checkpoint 1=$DATA/runs/bs_seq_sc/J03_seed1/checkpoints/step00000500.pt \
        --fampnn-checkpoint $REPO/fampnn/weights/fampnn_0_3.pt --fampnn-variant 0.3 \
        --targets $T --lengths 80 90 100 110 120 130 \
        --seeds 101 102 103 104 105 106 107 108 \
        --event-sigma 0.429 --sequence-policy post_feedback_redesign \
        --resume --out $OUT/table4_redecode/$T" \
  sbatch --export=ALL,CMD,ARGS --time=08:00:00 \
         scripts/slurm/marlowe/integrated_official.sh
done
```

Re-submitting the same command resumes. Start with **TNFa** (the most
expensive) to calibrate the time limit before launching the rest.

`--checkpoint-dir` must hold the Protenix base model, not only
`pxdesign_v0.1.0.pt` -- a donors-only directory makes PXDesign attempt a CDN
download and burn the whole job on a stalled transfer.

## Two caveats the row must carry

**Different backbones.** The cached-backbone rows share one pre-generated
480-backbone collection across arms; the integrated matrix generates its own
backbone per cell. So a redecode row is not a sequence-stage swap against
the existing rows -- the generator differs too, and across-row differences
confound the two. The existing rows are comparable to each other; a
redecode row is comparable to a redecode `event_fixed` row, which does not
exist yet and would cost another 5-25 GPU-h.

**The evidence so far does not motivate this.** At n=1 cell each: redecode
left designability at 100% (3/3 -> 7/7, intervals overlapping), the
four-event residual does not accumulate, and every cell measured has
seed-to-seed spread exceeding arm-to-arm spread. The prior J03 - U03
comparison at n=480 showed no detected improvement. The 112-design paired
pilot (2 targets x 8 seeds x 1 length, 16 independent prefixes, ~0.5 GPU-h
at one event) is the cheap thing that could change that assessment, and it
should come first.
