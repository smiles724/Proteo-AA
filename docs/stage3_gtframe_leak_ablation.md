# Is the co-evolution gain real, or is it GT-backbone leakage?

`5abd92f` teacher-forces Stage III's side-chain frames: `predicted_frame=False`,
so S_phi reads the **GT backbone** (`sc_frame_R` / `sc_frame_t` / `sc_bb_coords`)
instead of frames built from `x_denoised` (`model.py:1370-1375`).

That creates a path from the GT backbone into a loss that is scored against the
GT backbone:

```
GT backbone (N,CA,C) -> sc_frame_R/t -> S_phi -> a_sc / q_sc_bb
                     -> a_direct_pre / q_direct -> B_post -> bb_post (MSE vs GT)
```

The a/q fusion can therefore lower `bb_post` by decoding GT frame position and
orientation out of the feedback features, with no side-chain reasoning involved.
Identity is not at risk — the AA loss is on the first pass's `aa_logits`, computed
before S_phi runs, and `post_aa_logits` is not emitted while
`predicted_mask=False` — but the *coordinate* measurement is.

This matters because `bb_post` vs `mse` is exactly the measurement that showed the
co-evolution channel working (see `stage3_sidechain_helps_backbone.md`, +5.5% on
validation at step 3000). That result was obtained with `predicted_frame=True`,
where no such path exists. It cannot be re-quoted from runs on this commit.

## The ablation

A 2x2 over {frame source} x {S->B feedback}. This is possible without any code
change because `5abd92f` touched only `scripts/training/train_protenix_monomer.py`
plus docs and tests — **no `pxdesign_train/` files** — so the model is identical
across the two commits and `predicted_frame` is the only difference.

| cell | job | frames | `a_direct_pre` / `q_direct` | commit |
|---|---|---|---|---|
| **A1** | 103961 | predicted (`x_denoised`) | on | 287ebc2 |
| **A2** | 103966 | predicted (`x_denoised`) | off (`--sc-ablation-arm a-bs`) | 287ebc2 |
| **A3** | 104217 | **GT** | on | 5abd92f |
| **A4** | 104218 | **GT** | off (`--sc-ablation-arm a-bs`) | 5abd92f |

Everything else is held fixed: start = Stage II `step52500` + AA head `step9000`,
`lr=1e-5`, crop 384, `iters_to_accumulate=8`, `refinement_sigma=2.0`,
`EVAL_INTERVAL=1000`, `CHECKPOINT_INTERVAL=1000`, `MAX_STEPS=8000`, same seed.
A2/A4 differ from A1/A3 in nothing but the two S->B channels (`a_bs_concat=True`,
`q_bs=False`, `bb_context=True`, `hres_inject=False` in all four).

## The statistic

Per micro-batch, `gain = (mse - bb_post) / mse`; per eval, the same on `val_mse`
and `val_bb_post`. Then:

```
G_pred = gain(A1) - gain(A2)     # channel benefit with NO GT path available
G_gt   = gain(A3) - gain(A4)     # channel benefit WITH the GT path available
leak   = G_gt - G_pred
```

Each cell already has its own control, so `G_pred` and `G_gt` are each internally
valid; subtracting them isolates what the GT frames add.

## Reading the outcome

| observation | conclusion |
|---|---|
| `G_gt ≈ G_pred` | GT frames add nothing to `bb_post`; the gain is side-chain content and the teacher forcing is safe for this measurement. |
| `G_gt >> G_pred` | The extra is leakage: the channel is carrying GT geometry, and any Stage III improvement reported under `predicted_frame=False` is inflated. |
| `G_pred ≈ 0`, `G_gt > 0` | The "co-evolution works" result is *entirely* leakage. Would also contradict A1-vs-A2, so treat as a red flag on the setup rather than a finding. |
| `G_gt < G_pred` | Teacher forcing is *hurting* the channel — plausible if S_phi's GT-frame features are off-distribution for a fusion that was learning against predicted frames. |

## The outcome: `G_gt ≈ G_pred`. No leakage.

A1/A2 ran to the 24 h wall clock (step ~7 900 of 8 000, TIMEOUT with exit 0);
A3/A4 read at step ~7 900, still running.

Per-micro-batch over the whole run:

| cell | frames | feedback | median gain | 1st half | 2nd half | `bb_post` beats `mse` |
|---|---|---|---|---|---|---|
| **A1** | predicted | on | **+7.28 %** | +1.30 % | **+13.98 %** | 85 % |
| **A2** | predicted | off | +0.09 % | +0.00 % | +0.14 % | 59 % |
| **A3** | GT | on | +4.46 % | +0.87 % | +12.00 % | 83 % |
| **A4** | GT | off | +0.09 % | +0.00 % | +0.17 % | 59 % |

On the 308-protein validation set, at matched steps:

| step | `G_pred` = A1−A2 | `G_gt` = A3−A4 | **`leak`** |
|---|---|---|---|
| 1000 | +0.14 % | +0.10 % | −0.04 % |
| 2000 | +1.60 % | +0.80 % | −0.80 % |
| 3000 | +5.43 % | +3.07 % | −2.36 % |
| 4000 | +8.73 % | +5.31 % | −3.42 % |
| 5000 | +12.25 % | +9.06 % | −3.19 % |
| 6000 | +14.17 % | +12.58 % | −1.59 % |
| 7000 | +16.10 % | +16.25 % | **+0.15 %** |

`leak` is ≤ 0 for six steps and +0.15 % at the last, i.e. **the two arms converge
to the same place**. Feeding S_phi the GT backbone adds nothing to `bb_post`.
Three conclusions:

1. **The `bb_post` gain is side-chain content, not GT geometry.** The concern at
   the top of this note — that the a/q fusion could lower `bb_post` by decoding
   GT frame position and orientation out of the feedback features — does not
   materialise. Had it, `G_gt` would have run *above* `G_pred`, since only A3/A4
   have that path available. It never does.
2. **`stage3_sidechain_helps_backbone.md` stands**, and understates the effect:
   at step 7 000 the validation gain is **+16.24 %**, still climbing monotonically
   (0.09 → 1.65 → 5.53 → 8.87 → 12.30 → 14.28 → 16.24), against a control pinned
   at +0.1 % for all 7 000 steps.
3. **`predicted_frame=False` is safe for this measurement.** It reaches the same
   gain, so it neither inflates nor is required.

**A transient, not a deficit.** Read at step 6 000 the GT arm was 1.6–3.4 points
*behind*, which looked like teacher forcing actively hurting the channel. Step
7 000 closed the gap. The correct reading is that GT frames make the channel
**slower to develop, and no better once developed** — plausible if S_phi's
GT-frame features are initially off-distribution for a fusion whose weights are
being learned against `x_denoised` frames, with the fusion adapting over a few
thousand steps. Worth remembering as a general caution: on curves this steep, a
gap read mid-climb measures the phase difference, not the asymptote.

## What this ablation does not settle

* **Only one refinement step**, at `refinement_sigma=2.0`. It says nothing about
  sampled structures; that needs an inference-time comparison, where GT frames are
  unavailable by construction.
* **A stronger test exists and is not run here.** Feed GT frames but make the
  side-chain *content* uninformative (random side-chain coordinates, or shuffled
  type logits) while leaving the geometry path intact. Any remaining `bb_post`
  gain would then be attributable to geometry alone. That needs a small code
  change to corrupt S_phi's input, so it is deliberately left as a follow-up.
* **Cross-commit comparison.** A1/A2 run on 287ebc2 and A3/A4 on 5abd92f. Argued
  safe above (identical `pxdesign_train/`), but it is an argument, not a
  measurement. Re-running A1/A2 on 5abd92f with an explicit
  `predicted_frame=True` override would remove even that caveat — currently not
  possible from the CLI, because `apply_training_stage_args` sets
  `args.predicted_frame = args.training_stage == "predicted_mask"` unconditionally,
  overriding anything passed.
