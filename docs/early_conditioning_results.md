# E1 and E2: result and exit decision

**Decision: do not proceed with SC → BB feedback. The injection site was a real
limitation and moving it produces a large, robust, chemistry-safe correction —
but that correction is reproduced by a BB-only conditioner at 2.6× less cost,
and the side-chain-specific component is indistinguishable from zero.**

Runs: `/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/{early_s_*,atom_*}`,
evaluations `pxf_sb_eval/{e1_exit_119883,e1_exit_noema_119884,e2_exit_119957}`,
geometry audit `pxf_early_cond/e1_geom_119956`. All arms: 2,000 steps, lr 1e-4,
clip 1.0, 50 packing steps, σ ∈ [0.1, 2.0] Å, bypass BB→SC policy, frozen
PXDesign/FaMPNN.

## The arms were comparable, provably

All six arms loaded the late pilot's own upstream cache with **2001 hits / 0
misses**, fingerprint `aef7838b5c77f925a68df1a4fcf31e39`, 512 states over 128
structures. So every arm — and the late pilot's three — trained on bit-identical
`bb0` and `sc0`. Same pool, seed, σ window, gate and budget.

## Held-out backbone accuracy

64 AFDB targets × 4 σ = 256 corrective events, val split, disjoint from train by
accession. Wiring: the zero-feedback arm at the conditioning site deviates from
`bb0` by 1.4e-5 Å (E1) and 1.6e-5 Å (E2) against a 1e-4 tolerance.

| arm | BB RMSD | lDDT-BB | TM | vs `bb0` (paired) | cost |
|---|---|---|---|---|---|
| `bb0` | 0.3341 | 0.9785 | 0.9830 | — | 1.00× |
| **E1** `early_s_full` | **0.3117** | 0.9822 | 0.9850 | +0.0224 [+0.0194, +0.0255] | 3.73× |
| **E1** `early_s_bb_only` | 0.3124 | 0.9822 | 0.9849 | +0.0217 | **1.43×** |
| **E1** `early_s_generic` | 0.3304 | 0.9788 | 0.9832 | +0.0037 | 1.43× |
| **E2** `atom_sz_full` | 0.3129 | 0.9821 | 0.9849 | +0.0212 [+0.0182, +0.0243] | 3.79× |
| **E2** `atom_sz_bb_only` | 0.3148 | 0.9817 | 0.9847 | +0.0193 | **1.48×** |
| **E2** `atom_s_full` | 0.3147 | 0.9818 | 0.9847 | +0.0194 | 3.72× |
| `refine` (sampler step) | 0.3341 | 0.9785 | 0.9830 | −0.0000 | 1.37× |

**The site was the limitation.** The late pilot's best arm improved `bb0` by
+0.0007 Å and moved neither lDDT nor TM. The same readout injected into
`s_single` improves it by **+0.0224 Å — 32× more** — and moves lDDT-BB
0.9785 → 0.9822 and TM 0.9830 → 0.9850. The late pilot's conclusion ("the
optimizer found no use for side-chain information") was correctly scoped to its
injection site; at this site the optimizer finds a great deal of use for
*something*.

**That something is not side chains.**

| comparison | pooled | verdict |
|---|---|---|
| E1 `full` vs `bb_only` | **+0.0007 [−0.0006, +0.0019]** | spans zero; need ≥0.05 |
| E2 `sz_full` vs `sz_bb_only` | **+0.0019 [+0.0011, +0.0027]** | real; 26× short of 0.05 |
| E2 pair branch (`sz_full` vs `s_full`) | +0.0018 [+0.0009, +0.0028] | real; 28× short |

## Per noise level

Pooling hides the structure, and here it inverts the late pilot's headline.

| σ | `bb0` | E1 `full` | E1 `bb_only` | E1 full−bb_only | E2 full−bb_only |
|---|---|---|---|---|---|
| 0.105 | 0.083 | +0.0025* | +0.0026* | −0.0001 | +0.0001 |
| 0.314 | 0.182 | +0.0084* | +0.0089* | −0.0005 | +0.0005* |
| 0.847 | 0.366 | +0.0205* | +0.0219* | **−0.0014** | +0.0015* |
| 1.939 | 0.706 | **+0.0581*** | +0.0533* | +0.0048 | +0.0054* |

`*` = paired interval excludes zero.

**The gain now lives where there is most to gain.** The late pilot found its
only significant effects at the three *lowest* σ and nothing at the highest;
this is the reverse, growing ~23× across the sweep. At σ = 1.939 the correction
clears the 0.05 Å absolute floor against `bb0` (+0.058) — but `bb_only` clears
it too (+0.053), which is the whole problem.

**E1's side-chain input measurably *hurts* at two σ** (−0.0005 and −0.0014, both
excluding zero), consistent with it spending fitting budget for nothing. E2's
helps at three, by 0.0001–0.0054 Å.

## The mechanism: the arms are blind to side-chain conformation

The matched-conformation control rotates the packed χ angles by 60°, holds the
backbone and sequence fixed, and **re-encodes** — so `h_packed`, the χ features
and the environment features all move together.

| | change when the packing it reads is scrambled |
|---|---|
| E1 `early_s_full` | −0.0007 Å [−0.0013, −0.0000] |
| E2 `atom_sz_full` | **+0.0000 Å [−0.0001, +0.0002]** |

E2 reads predicted side-chain atoms directly, and rotating every rotamer it sees
changes its output by nothing measurable. Whatever its +0.0019 Å over
`atom_sz_bb_only` comes from, it is not side-chain *conformation* — the
remaining candidates are the χ-validity, reliability and confidence channels
that `bb_only` zeroes, i.e. side-chain *presence* rather than geometry.

## What the correction actually does to the coordinates

Measured, after an earlier draft of this document inferred "largely
self-cancelling" from a 1.9 Å maximum displacement. That inference was wrong in
every respect. 64 × 4 events, supervised backbone atoms:

| arm | gain | RMS disp | RMS after superposition | rigid fraction | median | p95 | max |
|---|---|---|---|---|---|---|---|
| `early_s_full` | +0.0224 | 0.1102 | 0.1087 | **0.026** | 0.0750 | 0.2025 | 0.5041 |
| `early_s_bb_only` | +0.0217 | 0.1058 | 0.1041 | 0.031 | 0.0726 | 0.1937 | 0.4796 |
| `early_s_generic` | +0.0037 | 0.0544 | 0.0521 | 0.072 | 0.0399 | 0.0967 | 0.2260 |

* **The typical correction is 0.11 Å, not 1.9 Å.** The figure quoted earlier was
  a single worst atom in the 32-example in-loop validation; over the full panel
  the maximum is 0.50 Å.
* **Almost none of it is a global pose change** — rigid superposition removes
  2.6–7.2% of the squared movement.
* **Bigger corrections are the ones that help**: corr(displacement, gain) =
  **+0.81** for `full`, +0.77 for `bb_only`, +0.20 for `generic`. Nothing about
  this is self-cancelling.
* **`full` and `bb_only` move identically** — 0.110 vs 0.106 Å RMS, same rigid
  fraction, same median/p95/max profile. Per the diagnostic's own criterion,
  the movement is a property of the receiving interface, not of the side-chain
  information.

## Chemistry does not regress; it improves

| | `bb0` | `early_s_full` |
|---|---|---|
| backbone bond RMS deviation (Å) | 0.0352 | **0.0302** |
| backbone bond max deviation (Å) | 0.2694 | 0.2240 |
| backbone angle RMS deviation (°) | 2.458 | 2.554 |
| backbone clashes / residue | 0.0570 | 0.0498 |

After fresh repacking on each arm's own backbone (`sc1`, never `bb1 + sc0`):
symmetry RMSD 1.4438 → 1.4266, bad-bond fraction 0.0023 → 0.0018, rotamer
outliers 0.3392 → 0.3337. All improve.

## Robust to the weighting choice

Unlike the late pilot — where the only EMA-robust gain was the σ-only control's
— these results barely move. At σ = 1.939, `full` is +0.0581 (EMA) against
+0.0587 (raw); `full` − `bb_only` is +0.0048 (EMA) against +0.0015 (raw), both
spanning zero. The conclusion is the same under either.

## What this licenses

**Not sampler integration for SC feedback, and not the Gate B stage.** Both
candidates fail the criterion against their own matched BB-only control by
26–70×, and the side-chain component is zero (E2) or negative at two of four
noise levels (E1).

**It does license a separate, non-SC candidate.** `early_s_bb_only` is a
0.0217 Å paired improvement over `bb0` with an interval excluding zero, growing
to +0.053 Å at σ = 1.939, at **1.43× cost** — it needs no packing rollout, only
`h_base`, which `propose` already produces. It improves lDDT, TM and backbone
chemistry. That is a real result about *early conditioning of the backbone
denoiser on its own frozen encoder features*, and it should be pursued under
that description rather than as side-chain feedback.

`early_s_generic` (+0.0037) shows this is not merely a σ-conditioned bias: the
per-residue features matter. `refine` (−0.0000) shows a second sampler step buys
nothing here, so the gain is not "one more denoiser call".

## Caveats stated rather than buried

* **Protein-level, not cluster-level intervals.** The la-proteina AFDB manifest
  carries no cluster labels; train/val is split by accession, which bounds
  leakage but not within-panel homology.
* **2,000 steps.** Unlike the late pilot, the loss curves and the EMA/raw
  agreement give no sign that the feature-reading arms are unconverged — but a
  budget sweep was not run, and the E1-vs-E2 ordering (+0.0224 vs +0.0212) is
  within the range a longer run could move.
* **One σ window, one donor, one panel.** The σ = 1.939 behaviour is the edge of
  the trained window and the most extrapolation-prone point in the table.
* **E2's χ/reliability channels were not individually ablated.** The +0.0019 Å
  is attributed to "side-chain presence rather than geometry" by elimination
  (the conformation control is exactly zero), not by a direct ablation.
