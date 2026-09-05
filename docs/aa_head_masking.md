# AA head: masking schedule

What `aa_mask_mode` controls, why every stage that trains
`design_residue_type_head` now uses `time_dependent`, and what to watch in the
logs.

## What changed

Every stage that trains the AA head used `aa_mask_mode="all"` — every design
position masked at once. That pins `aa_t` at 1.0, which collapses the
masked-diffusion objective into its simplest special case:

| | `"all"` (before) | `"time_dependent"` (now) |
|---|---|---|
| `aa_t` | constant 1.0 | `~ U(0, 1)` per example |
| mask fraction | 1.0 | tracks `t` |
| MDLM weight `1/t` | constant 1.0 — **no weighting at all** | 1 … 100, median 2 |
| head's time embedding | same constant vector every step — a bias term | carries the real mask level |
| partially-masked examples | **0%** | ~97% |

The cost of `"all"` is not the dead machinery, it is what the head never sees.
Under full masking it is only ever asked to predict every identity from backbone
geometry alone, so it cannot learn to condition on **already-decided
neighbours** — which is where most of ProteinMPNN's ~33% recovery on predicted
backbones comes from, against our ~13%.

Inference has the same gap from the other side: `seq_mode="sequential"` reveals
positions progressively and feeds commitments back through `restype`, but a head
trained only on fully-masked inputs has never seen a partial sequence.

`time_dependent` draws `t ~ U(0,1)` per example and masks each design position
independently with probability `t` (`aa_mask_min/max_prob` default to 0.0/1.0,
so `prob == t`). That is the standard MDLM/LLaDA schedule — there is no
hand-designed masking ratio to pick, which was the original objection to it.

## Where it is set

One constant, `AA_MASK_MODE_TRAINING` in
`scripts/training/train_protenix_monomer.py`, read by all four stages that train
the head:

```
aa_head_warmup            AA_MASK_MODE_TRAINING
joint                     AA_MASK_MODE_TRAINING
aa_head_on_stage2         AA_MASK_MODE_TRAINING
coevolution / predicted_mask   AA_MASK_MODE_TRAINING

backbone_only             "none"   — disables the AA loss
sidechain_warmup          "none"   — disables the AA loss
```

A constant rather than four string literals, so the schedule is a property of
the objective and cannot drift apart per bundle. `--aa-mask-mode` still
overrides it per run.

**Validation is deliberately pinned to `"all"`** (`build_eval_dataloader`).
`eval_args` is a whole-namespace copy of the training args, so without the pin
the eval set would switch to partial masking along with training. Partial
masking is a strictly easier task — neighbouring identities are visible — so
`val_aa_acc` would rise for a reason unrelated to the model, and stop being
comparable to the 0.1310 every stage has reported so far.

## What to watch in the logs

### `aa_mask_frac`

The fraction of tokens carrying an AA loss. Under `"all"` it equals the design
region's share of the crop and never moves. Under `time_dependent` it is that
share times `t`, so **across steps it should vary**.

If it looks constant, check what is being read: `DesignSourceDataset` seeds its
RNG as `default_rng((seed + idx) % 2**32)` (`runner/data.py`), deliberately, so a
given index always yields the same `t`. A smoke test that re-reads index 0 every
step therefore sees a constant `aa_mask_frac` — that is the harness, not the
schedule. Real training walks thousands of indices and `t` spreads over (0, 1).

### The MDLM weight, if training destabilises

The per-token CE is importance-weighted by `1 / max(t, aa_time_eps)`. With
`aa_time_eps = 1e-2` that is a **1× … 100× range**, median 2×.

The loss normalises by the summed weights (a weighted *mean*, not a weighted
sum), so the scalar stays CE-comparable and the overall scale is bounded. But a
single low-`t` example can still dominate a batch's gradient.

**If `aa_ce` becomes spiky or the AA head's gradient norm swings, raise
`aa_time_eps` first** — `5e-2` caps the weight at 20×, `1e-1` at 10×. It is in
`training_configs["residue_type"]`. Prefer this to lowering the AA learning
rate: the instability is weight variance, not step size.

### `aa_head_grad_norm` / `aa_head_relative_update`

Already logged. Under `"all"` these were smooth because every example carried
the same weight. Expect more variance now; sustained growth is the signal to
touch `aa_time_eps`, a one-off spike is not.

## Reverting

Set `AA_MASK_MODE_TRAINING = "all"`, or pass `--aa-mask-mode all` for a single
run. Nothing else is coupled to it: the schedule only reaches
`DesignSelection`, and every structural loss is untouched.

## Note for runs in flight

This changes the objective for `coevolution`, so a Stage III run restarted on
this commit is **not directly comparable** to one started before it. The AA head
is the only module affected. Validation stays on `"all"`, so `val_aa_acc` itself
remains comparable across the boundary — it is the *training* signal that
changed.

## Checkpoints

An AA head trained under `"all"` has never seen a partial sequence, and this
head is known not to transfer across configurations: grafting one trained with
`enable_sidechain`/`enable_coevolution` off into a stage with both on produced
`aa_ce` 54–76 at chance accuracy, **even when the donor's backbone was
byte-identical (636/636)** to the target trunk. So measuring this change means
retraining a head, not re-scoring the existing one.

`aa_head_on_stage2` is the cheap place to do it: it trains only the head
(1.30M of 262.05M) against a frozen backbone and S_phi, and its previous run
reported at step ~9000 in 14.5h — directly comparable to the 0.1270 that run
recorded, since validation masking is pinned.

---

# The inference half

Training on partial masks only pays off if sampling actually produces them.
`seq_mode` decides that, and it now defaults to `sequential`.

## What the two modes do

**`complete_unmask`** (the previous default) keeps the design region fully
masked for the whole trajectory, re-predicts every position from scratch at
every step, and reads out a single argmax at the last one. No identity ever
informs another. `aa_t` is hard-coded to 1.0, which is the only value a
full-mask-trained head has seen — which is why this was the right default for
those checkpoints, and why it remains the mode to pass for them.

**`sequential`** decodes a few positions per round, commits them into `restype`,
and re-encodes the trunk so the positions still to be called can see them. It
also feeds the head a real `aa_t` — the fraction still masked — which is exactly
the quantity `time_dependent` training teaches it to read.

## Choosing what to commit each round: `commit_strategy`

| | how many | which |
|---|---|---|
| `topk` (default) | the schedule: the design region split evenly over the sampling steps | highest confidence |
| `threshold` | **the model**: everything above `commit_threshold` | highest confidence |
| `random` | same count as `topk` | uniformly — the **control arm** |

Confidence is the max softmax probability. Ordering by it rather than at random
is LLaDA's *low-confidence remasking*, which that paper measured as clearly
better on language. `random` exists so the same comparison can be made here
rather than cited: it is the only way to find out whether the ordering buys
anything on **proteins**.

### The two guards on `threshold`, and why both are needed

A bare threshold degenerates in both directions, and which one you get depends
on how the checkpoint happens to be calibrated — a property of the weights, not
of this code. Both failures are silent.

**Nothing clears the bar.** No commits, identical state next round, and the run
finishes having decided nothing. The guard commits the single most confident
position, so the trajectory always advances. Expect this to be the *common* case
early on: an AA head near 13% accuracy is not confident about much.

**Everything clears it.** The whole design region lands in round one, which is
`complete_unmask` wearing a schedule — no position is ever chosen with another's
identity visible, which is the entire point of the mode. `commit_max_frac`
(default 0.5) caps each round at that share of what is left.

### Picking `commit_threshold`

Do not guess it. The default of 0.9 is a placeholder, and it cannot be derived
from the numbers already in hand: cross-entropy constrains the probability on
the *true* class, while confidence is the probability on the *arg-max* class,
and those come apart when the model is usually wrong. `val_aa_ce = 2.809` with
`val_aa_acc = 0.127` is equally consistent with a typical confidence of 0.9 and
one of 0.15.

Measure it: dump `probs.max(dim=-1)` over the validation set and look at (a) the
percentiles, (b) accuracy bucketed by confidence — if the high-confidence
buckets are not more accurate, ordering by confidence is not buying anything and
neither strategy will help — and (c) what share of positions each candidate
threshold would admit per round.

## `temperature`

Was declared on `cogenerate` and never read; sampling was always greedy. Now
live: above 0, the identity is drawn from `softmax(logits / temperature)`.

**Confidence is still read off the untempered distribution.** Temperature is a
diversity knob for *which* residue gets chosen; confidence has to keep meaning
"how sure is the model", because the commit order ranks on it. Taking it from
the tempered distribution would make a high temperature look like high
certainty and commit the least reliable positions first.

ProteinMPNN-class methods run this low — 0.1, sometimes 1e-4 — and get diversity
from sampling many sequences and scoring them, not from loosening any one of
them. Pass `generator` for reproducibility.

## Order of operations

These knobs cannot be tuned against the current checkpoint. A head trained with
`aa_mask_mode="all"` has seen neither a falling `aa_t` nor a partially filled
`restype`, so its confidences under `sequential` are not trustworthy and every
strategy would be ranking on noise.

1. Retrain a head with `AA_MASK_MODE_TRAINING` (`aa_head_on_stage2`, ~14.5h,
   comparable to the 0.1270 that run recorded).
2. Measure the confidence distribution and calibration on it.
3. Then set `commit_threshold` and run the strategy ablation.

Until (1) lands, pass `seq_mode="complete_unmask"` explicitly for any run using
an older head.

One thing that needs no code and can be swept immediately: `N_step`. The
schedule is an even split, so at `N_step >= len(design region)` it is already
one position per round — full sequential decoding. LLaDA reports accuracy
improving consistently with more sampling steps.
