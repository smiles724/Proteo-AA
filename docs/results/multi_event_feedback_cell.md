# Four feedback events: the residual does not accumulate

PDL1, L=80, generation seed 101, seven arms, `--event-sigmas 4.0 2.0 1.0
0.429`, `--sequence-policy post_feedback_redesign`. Marlowe job 499467,
5 m 25 s, branch `feat/multi-event-feedback`.

Events resolved to steps **303 / 319 / 334 / 350**, churned sigmas 4.125 /
2.063 / 1.047 / 0.445. The last is the event the adapters were trained at;
the other three are off-distribution, and the run says so.

## Plumbing

| | value |
|---|---|
| injections | 0 on U03/J03, **4** on each E1 arm |
| solver calls | 97 (vs 50 resuming at step 350 alone) |
| event to final | **1.10 Å** (vs 0.19 Å single late event) |
| backbone-only interface | 5.12-5.15 Å, **0 clashes**, 7/7 |
| distinct output sequences | **7** |
| arm time | 266.9 s, 6.6x the single-event `event_fixed` baseline |

The earlier entry point does what it was meant to: resuming at step 303
leaves 1.10 Å of trajectory instead of 0.19 Å, so a residual injected there
has room to express itself.

## The result: no accumulation

Residual norm across the four events, monotone in all four feedback arms:

```
24.92  ->  25.85  ->  26.38  ->  26.57
```

That rise is the norm tracking sigma, not feedback compounding. The
decisive comparison is the **same event under two schedules** -- step 350,
with three prior corrections versus none:

| arm | 1 event @ step 350 | 4 events @ step 350 | delta |
|---|---|---|---|
| bb_only s0 | 26.410 | 26.567 | +0.157 |
| full s0 | 26.153 | 26.051 | -0.102 |
| bb_only s1 | 26.329 | 26.400 | +0.071 |
| full s1 | 26.563 | 26.545 | -0.018 |

**After three corrections the residual at the trained event is
indistinguishable from having had none** -- ±0.16 on ~26.4, and not even
consistent in sign. Each event's residual is whatever that sigma produces,
independent of history. The mechanism does not build on itself.

Sequence Hamming to the matched J03 arm rose from 1-6 at one event to 5-11
at four. That is four injections perturbing more than one does, which is
expected and is not evidence of compounding -- the residual measurement
above is the direct test, and it is flat.

## What this does not establish

**n = 1.** One target, one length, one generation seed. The single-event
redecode cell on the same backbone came back 7/7 designable with ipTM
0.746-0.820, where the *same arm* on the two adapter seeds sat at both ends
of the range. Seed spread exceeds arm spread in every cell measured so far.

**Off-distribution.** Three of the four injections query adapters trained
at one event at sigma 0.429. This is a checkpoint-transfer probe. A flat
residual could be the mechanism not compounding, or the adapters producing
something uninformative at sigmas they never saw. These are not separated
here, and separating them needs retraining against the schedule.

**No cross-schedule pairing.** A four-event run has taken three corrections
before it reaches step 350, so it is not decoding the state a one-event run
decodes there. The comparison above is between two different states at the
same sigma -- which is the right comparison for "does the residual depend on
history", and the wrong one for anything paired.

## Cost, if anyone proposes scaling this

Measured per cell (PDL1 L=80, 7 arms), and extrapolated to the 480 cells /
3,360 designs a Table 4 row needs:

| protocol | arm time | wall | GPU-h at PDL1 size | with target scaling |
|---|---|---|---|---|
| `event_fixed`, 1 event | 40 s | 1 m 34 s | 20.5 | 60-90 |
| redecode, 1 event | 110 s | 2 m 52 s | ~56 | 165-245 |
| **redecode, 4 events** | **267 s** | **5 m 25 s** | **~135** | **400-600** |

400-600 GPU-h for a row, on a project whose fairshare is 0.0036, to measure
a mechanism that the residual data says does not compound. The 112-design
paired pilot (2 targets x 8 generation seeds x 1 length x 7 arms, 16
independent prefixes) is ~1.2 GPU-h at four events and is the only unit
worth spending before that decision.
