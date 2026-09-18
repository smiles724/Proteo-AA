# SC → BB pilot: result and exit decision

**Decision: do not proceed to sampler integration.** The correction is real,
consistent and statistically solid, and it is not side-chain-specific and not
within an order of magnitude of the roadmap criterion.

Runs: `/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot/{full,bb_only,generic}_1182{15,16,17}`,
evaluation `/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/exit4_118228`,
probe `.../probe_118225`. All three arms: 2,000 steps, lr 1e-4, clip 1.0, 50
packing steps, σ ∈ [0.1, 2.0] Å, bypass BB→SC policy, frozen PXDesign/FaMPNN.

The evaluation was run four times while the cost attribution was corrected; the
accuracy numbers below are bit-stable across all four (same seeds, checkpoints
and panel), and only the cost column changed. Earlier run directories
`exit_118224`, `exit2_118226` and `exit3_118227` are superseded: their cost
tables are wrong. See the commit history for what each got wrong and in which
direction.

## The arms were comparable, provably

All three share the upstream cache fingerprint `79ea11f761f0059d311e01244b06af22`,
and the two controls ran **2001 cache hits / 0 misses** — every frozen half came
from the same file, so all three trained on bit-identical `bb0` and `sc0`. Same
pool (512 examples over 128 structures), same seed, same σ window (steps 305–363
of 400), same gate, same 262,560 parameters.

## Held-out backbone accuracy

64 targets × 4 σ = 256 corrective events, val split (disjoint from train by
accession).

| arm | Cα RMSD | BB RMSD | lDDT Cα | lDDT BB | TM-score |
|---|---|---|---|---|---|
| `bb0` (proposal) | 0.3334 | 0.3341 | 0.9784 | 0.9785 | 0.9830 |
| `zero` (wiring) | 0.3334 | 0.3341 | 0.9784 | 0.9785 | 0.9830 |
| `full` | 0.3326 | 0.3334 | 0.9785 | 0.9785 | 0.9831 |
| `bb_only` | 0.3325 | **0.3333** | 0.9784 | 0.9785 | 0.9830 |
| `generic` | 0.3329 | 0.3337 | 0.9784 | 0.9785 | 0.9830 |
| `refine` (sampler step) | 0.3333 | 0.3341 | 0.9785 | 0.9785 | 0.9830 |
| `perturbed` | 0.3326 | 0.3334 | 0.9785 | 0.9785 | 0.9831 |

Wiring: the zero-feedback arm deviates from `bb0` by 2.19e-5 Å against a 1e-4
tolerance — float noise, so the comparison is between `bb0` and a correction.

Paired, per target, on BB RMSD:

* `full` vs `bb0`: **+0.0007 Å [+0.0005, +0.0010]**, n = 64. Excludes zero, so
  the correction is real.
* `full` vs `bb_only`: **−0.0000 Å [−0.0002, +0.0002]**. Indistinguishable, and
  `bb_only` is nominally ahead.
* Criterion needs ≥ 0.05 Å against the strongest comparable-cost alternative.
  Short by ~70×.

Side chains did not regress — `sc1` on the corrected backbone is marginally
better than `sc0` on the proposal (symmetry RMSD 1.4438 → 1.4314, bad-bond
fraction 0.0023 → 0.0021, rotamer outliers 0.3392 → 0.3344), consistent with
`bb1` being a hair closer to native.

## The mechanism: the adapter learned to ignore the side chains

This is the informative part, and it is not "the representation carries nothing".

| stage | response to a 60° rotamer flip |
|---|---|
| `h_packed`, the representation `A_SB` reads | **29%** (0.291 relative, floor 1.7e-7) |
| `A_SB`'s output, after training | **1.1%** mean, 3.2% max over 256 events |
| resulting backbone RMSD | **0.0000 Å** (0.3334 vs 0.3334) |

The probe says the path is *informative* on all 48 real-panel probes. The trained
adapter nonetheless converts a 29% change in its input into a 1.1% change in its
output, and no change at all in the backbone. It converged to approximately the
σ-conditioned bias `generic` fits directly — which is why `generic` matches it,
and why `generic` reaches it with a **smaller** residual (‖Δa‖ 0.374 vs 0.567),
having a simpler function to fit.

So the optimizer found no use for side-chain information at this injection site,
rather than the information being unavailable.

## Per noise level

Pooling hid the structure. 64 targets at each σ; `*` marks a paired interval
that excludes zero.

| arm | σ=0.105 | σ=0.314 | σ=0.847 | σ=1.939 |
|---|---|---|---|---|
| `bb0` BB RMSD | 0.0832 | 0.1820 | 0.3657 | 0.7055 |
| `full` improvement | +0.0004* | +0.0013* | +0.0010* | +0.0003 |
| `bb_only` improvement | +0.0005* | +0.0015* | +0.0012* | −0.0000 |
| `generic` improvement | +0.0003* | +0.0008* | +0.0006* | +0.0000 |
| `refine` improvement | −0.0000 | −0.0001 | −0.0003 | **+0.0006** |

Three things only visible here:

**The gain lives where there is least to gain.** The three significant points
are the three lowest-σ ones, where the proposal is already within 0.37 Å. At
σ=1.939, with 0.71 Å of error available, every feedback arm is inert.

**`refine` is the mirror image and wins at the top.** Neutral-to-harmful at low
σ, and the only arm that helps at σ=1.939 (+0.0006, beating `full`'s +0.0003). A
sampler step pays when real noise remains; feedback pays when the estimate has
nearly converged. So "the second call buys nothing wherever it is spent" is too
flat — *where* it is spent matters, it is just that neither reaches the
criterion.

**SC-specificity fails at every σ, and inverts at one.** `full` vs `bb_only`:

| σ | delta | 95% CI | verdict |
|---|---|---|---|
| 0.105 | −0.0001 | [−0.0002, +0.0000] | indistinguishable |
| 0.314 | −0.0003 | [−0.0004, −0.0001] | **`bb_only` better** |
| 0.847 | −0.0001 | [−0.0005, +0.0002] | indistinguishable |
| 1.939 | +0.0003 | [−0.0004, +0.0010] | indistinguishable |

At σ=0.314 the side-chain input measurably *hurts*, consistent with it spending
fitting budget for nothing. `perturbed` ≈ `full` at all four levels (−0.0000 to
−0.0001), so rotamer-scrambling is inert everywhere rather than only on average.

The adapter also pushes a relatively *larger* residual as σ rises (rel. residual
0.0132 → 0.0219 while ‖Δa‖ falls 0.630 → 0.482) — it tries hardest exactly where
it achieves least.

## Cost

Equal denoiser-call counts are not equal cost. Deployed cost per corrective
event, measured on an H200:

| arm | calls | needs packing | s/event | vs `bb0` |
|---|---|---|---|---|
| `bb0` | 1 | no | 0.051 | 1.00× |
| `refine` | 2 | no | 0.070 | 1.38× |
| `zero` | 2 | no | 0.071 | 1.40× |
| `bb_only` | 2 | no | 0.073 | 1.44× |
| `generic` | 2 | no | 0.073 | 1.44× |
| `full` | 2 | **yes** | **0.190** | **3.75×** |
| `perturbed` | 2 | yes | 0.189 | 3.74× |

Only the arms reading `h_packed` need the 50-step rollout and the re-encode.
`bb_only` reads `h_base`, which `propose` produces alongside `bb0`, and `generic`
reads nothing — so both are BB-only in cost as well as in information. (All three
variants share one code path so their parameter counts match exactly, which means
the controls *as executed* do compute a packing and discard it; pricing that
would price the parameter-matching rather than the method.)

**So the candidate costs 2.6× `bb_only` (0.190 vs 0.073) for a result
statistically indistinguishable from it**, and `bb_only` is nominally ahead
(0.3333 vs 0.3334). `refine` — one deterministic Euler step of PXDesign's own
schedule, at 1.38× — performs the same as `bb0` (0.3341), so at this noise level
the second denoiser call buys almost nothing wherever it is spent, and buys it
most expensively through side-chain feedback.

## What this does and does not license

It does not license sampler integration, and therefore not the two-event
training stage either.

A negative result here concerns **this late injection site** (after
`layernorm_a`, before `atom_attention_decoder`) at **this σ window**, with this
budget. Three things would each be a different experiment:

1. ~~**Headroom.**~~ **Contradicted by the per-σ breakdown below — do not chase
   this.** The original argument was that `bb0` at 0.334 Å leaves little to
   correct, so a higher-σ window with more error would give the adapter more to
   work with. The sweep says the reverse: across σ ∈ {0.105, 0.314, 0.847,
   1.939} the proposal's error ranges 0.083 → 0.706 Å, and every trained arm
   helps *significantly* at the three low-σ points and does **nothing** at the
   highest (`full` +0.0003 Å, interval includes zero). More headroom produced
   less gain, not more.
2. **Injection site.** Feedback before the diffusion transformer is the plan's
   named follow-up and is a capacity experiment, not a repeat of this one.
3. **Budget.** `full` has the most input dimensions to fit in the same 2,000
   steps, so underfitting is not excluded. The plan's "extend to 5,000 only if
   promising" is not met: `generic` ≥ `full` at every evaluation point (500,
   1000, 2000), so more steps would be extending the arm that is behind.

## Was this pilot affected by the phase-chaining bug?

No, verified rather than assumed. The documented phase chain
(`PHASE=2 RESUME=<phase1 final.pt>`) performed **zero** optimizer updates while
reporting success — `--resume` restored the step counter, so a phase-1
checkpoint at step 20,000 loaded into a phase-2 run whose `max_steps` is also
20,000 broke on its first batch and still wrote a "final" checkpoint. It also
loaded phase 1's AdamW moments into phase 2's optimizer, silently, because the
two directions are identically shaped six-parameter adapters.

These runs did not go through that path. From the checkpoints:

| arm | step | `resume` | `init_from` | optimizer updates | A_SB ‖W₂‖₁ | A_BS ‖W₂‖₁ |
|---|---|---|---|---|---|---|
| `full` | 2000 | None | None | 2000 | 393.3 | 0.0 |
| `bb_only` | 2000 | None | None | 2000 | 470.4 | 0.0 |
| `generic` | 2000 | None | None | 2000 | 341.1 | 0.0 |

Each has 80 training log rows (`log_every` 25 × 80 = 2000) and 4 evaluation
rows, the AdamW step counter reads 2000, and A_SB's zero-initialized output
projection moved far off zero while A_BS stayed exactly zero (bypass, frozen).
The step-0 evaluation independently confirms the start point: `val_delta_a_norm`
0.0 and `bb1` equal to `bb0`.

The pilot never used `--resume` or `--init-from` because `bs_policy` is bypass,
which applies no BB→SC residual — so there was no Phase-1 policy to inherit.

## Caveats stated rather than buried

* The paired interval is **protein-level, not cluster-level**: the la-proteina
  AFDB manifest has no cluster labels, so within-panel homology would make it
  optimistic. Train/val is split by accession, which bounds leakage but not this.
* `L_BB` is an EDM-weighted coordinate objective. Its σ weight is exactly
  Protenix's `diffusion_per_sample_scale`, but the donor's objective also
  rigid-aligns the target and adds smooth-lDDT and bond terms, so absolute loss
  values are not comparable to a PXDesign training curve.
* The Phase-1 BB→SC policy is bypass, so `sc0` is pretrained FaMPNN's packing.
  A selected Phase-1 policy would change what the feedback reads; it would not
  change that the adapter currently ignores it.
