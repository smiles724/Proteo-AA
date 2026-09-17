# SC → BB pilot: result and exit decision

**Decision: do not proceed to sampler integration.** The correction is real,
consistent and statistically solid, and it is not side-chain-specific and not
within an order of magnitude of the roadmap criterion.

Runs: `/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot/{full,bb_only,generic}_1182{15,16,17}`,
evaluation `/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/exit_118224`,
probe `.../probe_118225`. All three arms: 2,000 steps, lr 1e-4, clip 1.0, 50
packing steps, σ ∈ [0.1, 2.0] Å, bypass BB→SC policy, frozen PXDesign/FaMPNN.

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

## Cost

Equal denoiser-call counts are not equal cost. The BB-only alternatives need
`bb0` alone; the feedback arms also pay for the 50-step packing rollout and the
re-encode. `refine` — one deterministic Euler step of PXDesign's own schedule —
is cheaper than feedback and performs the same as `bb0` (0.3341), so at this
noise level the second denoiser call buys nothing wherever it is spent.

## What this does and does not license

It does not license sampler integration, and therefore not the two-event
training stage either.

A negative result here concerns **this late injection site** (after
`layernorm_a`, before `atom_attention_decoder`) at **this σ window**, with this
budget. Three things would each be a different experiment:

1. **Headroom.** `bb0` is already at 0.334 Å here, so `max(0.05 Å, 3%)` demands a
   ~15% relative improvement. A higher-σ window has more error to remove — and
   less determined side-chain evidence to remove it with, which is the tension
   the window was chosen to balance.
2. **Injection site.** Feedback before the diffusion transformer is the plan's
   named follow-up and is a capacity experiment, not a repeat of this one.
3. **Budget.** `full` has the most input dimensions to fit in the same 2,000
   steps, so underfitting is not excluded. The plan's "extend to 5,000 only if
   promising" is not met: `generic` ≥ `full` at every evaluation point (500,
   1000, 2000), so more steps would be extending the arm that is behind.

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
