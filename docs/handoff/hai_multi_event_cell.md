# HAI handoff: the multi-event feedback cell

Marlowe's queue put this ~2 days out (fairshare 0.0036). The cell itself is
~4.5 minutes of compute. Running it on HAI instead.

## Code: pull a NEW branch

```bash
cd <your Proteo-AA-pxdesign-fampnn-pack checkout>
git fetch origin feat/multi-event-feedback
git checkout feat/multi-event-feedback     # 84d6b6d or later
```

**Not `exp/binder-design-matrix`, not `feat/post-feedback-redecode`.** The
branches stack:

| branch | adds |
|---|---|
| `exp/binder-design-matrix` | single event, sequence fixed at the event |
| `feat/post-feedback-redecode` | re-decode after the corrective call |
| **`feat/multi-event-feedback`** | **N events, N corrective calls** + six review fixes |

`feat/multi-event-feedback` contains both earlier changes. Use the reporter
from **this same branch** -- it knows `schedule_id` and the new columns, and
an older one will mis-pair or drop rows.

Your `scripts/slurm/hai/integrated_official.sh` (from `cfa0fae`) is on this
branch unchanged; keep using it.

## Data: nothing to re-pull

The bundle at `$BUNDLE` is re-staged, but **only `repo/HEAD` and
`repo/BRANCH` changed** -- verified by diffing `SHA256SUMS` before and
after. Every checkpoint, target, donor and release-data byte is identical.

Do **not** re-pull the 1.1 GB. If you want the pin updated:

```bash
rsync -a <marlowe>:$REMOTE_BUNDLE/repo/ $BUNDLE/repo/
( cd $BUNDLE && sha256sum -c --quiet SHA256SUMS )   # should still pass
```

Same inputs as the cells you already ran: donor `b075867bae942dc0`, FaMPNN
0.3 `8969b3f1f3c94117`, J03 s0 `c506c7e1c43104dd`, s1 `ee673b982c4f5dd7`.

## The command

```bash
export OUTROOT=/hai/scratch/yfsun/pxf_runs/integrated_feedback_v2
python scripts/run_integrated_binder_matrix.py \
  --targets-config   "$BUNDLE/targets/configs_binder_benchmark/targets.yaml" \
  --prepared-dir     "$PREPARED" \
  --checkpoint-dir   "$BUNDLE/checkpoints/donors" \
  --checkpoint-selection "$BUNDLE/selection/selected_checkpoints.json" \
  --bs-checkpoint 0="$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt" \
  --bs-checkpoint 1="$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt" \
  --fampnn-checkpoint "$BUNDLE/checkpoints/donors/fampnn_0_3.pt" \
  --fampnn-variant 0.3 \
  --targets PDL1 --lengths 80 --seeds 101 \
  --event-sigmas 4.0 2.0 1.0 0.429 \
  --sequence-policy post_feedback_redesign \
  --out "$OUTROOT/generation_cell_multi_event"
```

**Fresh output directory.** Do not reuse a previous cell's.

**One trap Marlowe hit:** `--checkpoint-dir` must contain the Protenix base
model, not only `pxdesign_v0.1.0.pt`. Pointing it at a donors-only directory
made PXDesign try to download `protenix_base_default_v0.5.0.pt` from the CDN
and burn a 40-minute job on a stalled transfer -- and it left a *truncated*
mini checkpoint in the donor directory. If the bundle's `checkpoints/donors`
lacks the base model on your side, point `--checkpoint-dir` at your PXDesign
`release_data/checkpoint` instead.

## Why these four sigmas

Resolved against **your** recorded ladder
(`docs/results/event_sigma_ladder.csv`, from `2ccefe7`):

| requested | step | churned sigma | position |
|---|---|---|---|
| 4.0 | 303 | 4.1250 | 76 % |
| 2.0 | 319 | 2.0625 | 80 % |
| 1.0 | 334 | 1.0469 | 84 % |
| 0.429 | 350 | 0.4453 | 88 % — **the trained event** |

Distinct and spread, rather than clustered in the last few steps. The run
prints the resolved steps; check them before reading anything else.

## Expect

Runtime, from Marlowe measurements on the identical cell:

| protocol | arm time | wall |
|---|---|---|
| `event_fixed`, 1 event | 40 s | 1 m 34 s |
| redecode, 1 event | 110 s | 2 m 52 s |
| **redecode, 4 events** | **~230 s (projected)** | **~4 m 30 s** |

Check before folding:

- resolved steps are 303/319/334/350 (or whatever your ladder gives -- but
  four *distinct* ones)
- `events_scheduled=4` on all seven arms
- `event_injections`: **0** for U03/J03, **4** for each E1 arm
- `per_event_feedback_norms` has **four** entries per arm, positionally
  matching `event_steps`; controls read `-` for skipped events
- `sequence_decode_passes` is >= its control's, never less
- `schedule_id` appears in every `sample_id`

**Read `per_event_feedback_norms` first.** Whether the residual grows,
decays or stays flat across the four events says more about whether repeated
feedback can accumulate than the designability number from one cell ever
will.

## Then fold

Same AF2-IG path as before (`--data-dir /hai/scratch/yfsun/af2_params`), then

```bash
python scripts/report_integrated_binder_matrix.py \
  --designs "$OUTROOT/generation_cell_multi_event/designs.csv" \
  --out     "$OUTROOT/generation_cell_multi_event/report"
```

The reporter **refuses** a CSV mixing schedules or sequence policies. Keep
this cell's outputs separate from the single-event ones; do not pool.

## What this cannot show

**Off-distribution.** A_BS and the feedback adapters were trained against
ONE event at sigma 0.429. Three of these four injections are at sigmas they
never saw. This is a **checkpoint-transfer probe**. The run prints
`OFF-DISTRIBUTION`. Do not present it as the same arm as a single-event run.

**n = 1.** For reference, the single-event redecode cell on Marlowe came back
7/7 designable with 7 distinct sequences, ipTM 0.746–0.820 — and the *same
arm* on the two adapter seeds sat at both ends of that range. Seed effect
larger than arm effect, again. One cell cannot separate anything.

**No cross-schedule pairing.** A four-event run has taken three corrections
by the time it reaches step 350, so it is not decoding the state a
one-event run decodes there. The decode seeds deliberately differ; equal
seeds would imply a pairing that does not exist.

If the norms look like they accumulate, the next unit is the 112-design
paired pilot (2 targets x 8 generation seeds x 1 length x 7 arms = 16
independent prefixes), ~1 GPU-h at four events. That is the smallest thing
that could decide anything.
