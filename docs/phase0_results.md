# Phase 0: the BB→SC hook on real weights, measured

**Result: clean on all six dev complexes, and the risk is the opposite of the
one the checks were written to catch.** The residual is not a whisper against
`h_V` — it is the same size or larger. Phase 1A's gate sweep is therefore the
load-bearing knob, not a formality.

Run: `scripts/phase0_pack_hook_check.py --seq-steps 100 --device cpu`, CPU,
~4 min/complex. Report and the screen that chose the set are archived at
`/scratch/m000137-pm06/Proteo-AA/pxf/runs/pxf_phase0/dev6_cpu/`.

| | |
|---|---|
| adapter | `couple_phase1/checkpoints/final.pt`, step 20,000, EMA weights |
| donor | `pxdesign_v0.1.0.pt` |
| FaMPNN | 0.0, matching the variant A_BS was trained against |
| settings | σ_B 0.429, gate `one`, 100 unmasking steps, temperature 0.1, seed 0 |
| dev set | `configs/binder_benchmark/dev_complexes.yaml`, six held-out recentPDB low-homology dimers, none an AlphaProteo target |

All seven checks pass on all six: `load`, `noop`, `residual`, `counters`,
`invariance`, `visibility`, `logits`.

## What was measured

| id | tokens | design | residual ‖·‖ | `h_V` ‖·‖ | ratio | target ΔÅ | binder ΔÅ | seq differs | SC atoms |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 7f7p | 192 | 96 | 4.997 | 5.045 | 0.99 | 0.423 | 5.39 | yes | 377 |
| 7f91 | 280 | 140 | 4.952 | 5.327 | 0.93 | 1.570 | 5.82 | yes | 490 |
| 7ozt | 307 | 132 | 4.344 | 5.051 | 0.86 | 0.014 | 4.72 | yes | 470 |
| 7ppb | 466 | 125 | 6.055 | 4.733 | **1.28** | 0.000 | 8.93 | **no** | 477 |
| 7tdq | 176 | 88 | 5.732 | 5.290 | 1.08 | 2.184 | 5.99 | yes | 303 |
| 7uxr | 354 | 177 | 5.254 | 4.932 | 1.07 | 0.001 | 7.46 | yes | 754 |

Norms are per-row means over binder rows. "target ΔÅ" and "binder ΔÅ" are max
coordinate deltas between the coupled and uncoupled arms.

## Three things this changes

**The residual is the size of the signal.** Ratio 0.86–1.28, and above 1.0 on
three of six. The `residual` check was written expecting the opposite failure
— a residual orders of magnitude below `h_V`, arithmetically applied and
practically inert, which would look exactly like "coupling does not help".
It is the reverse. This fits the two-sided behaviour `bs_policy` already
documents (helps when badly noised, hurts when nearly native) and it means
gate values well below 1.0 are the interesting region.

**The sequence pathway is live on five of six, not six of six.** Where it is
live, the coupled and uncoupled sequences differ at 100 steps, so the residual
reaches the sequence through generated side-chain context rather than only the
final repack — the mechanism Phase 1A depends on. 7ppb is the exception and is
not an inert case: it has the *largest* residual ratio (1.28) and the largest
binder coordinate change (8.93 Å), with an identical sequence. The residual
moved its packing substantially without crossing an argmax boundary at
temperature 0.1. Worth a second look under the gate sweep rather than treating
five of six as six of six.

**The inert-pathway hypothesis is ruled out.** `missing_atom_mask` is built
once from UNKNOWN identities and reused for the whole loop, which was the
leading candidate for explaining the pilot's null result. Zero binder
side-chain slots are marked missing on every complex, and 303–754 side-chain
atoms are produced. Generated side chains do reach later encoder calls.

## What the checks do and do not assert

The `invariance` check asserts the target's **identities** are unchanged; it
reports the target's coordinate delta without bounding it. The 0.000–2.184 Å
spread above is therefore a measurement, not a pass. It is expected rather
than a leak: target rows receive a bit-exactly zero residual (`target_row_norm_max`
is 0.0 on every complex), but in the `complex` arm the target's side chains are
generated too, so changing the binder's packing changes the messages the target
receives and hence its own repacked coordinates. Under `complex_sc`, where the
target's resolved side chains are held fixed, this should collapse — that is a
prediction worth checking rather than an assumption.

`noop` is the strongest of the seven: `delta_h=None` reproduces the uncoupled
sampler exactly, max coordinate delta 0.0 with an identical sequence, while the
wrapper is still traversed 101 times.

## What this does not establish

Nothing here measures whether A_BS **helps**. Every check is a wiring and
magnitude check on a held-out dev set chosen for being featurizable, not for
being representative. Designability is Phase 1A.

One conflict constrains that comparison: **A_BS was trained against FaMPNN 0.0,
but the design path defaults to 0.3** (upstream's recommendation for sequence
design). The primary adapter-on/off comparison must run 0.0 for both arms,
which is not the configuration the benchmark numbers were produced under.

## Getting the dev set right cost a round

The first four complexes were chosen by reading chain composition out of the
deposition. Three failed, and the fourth failure was the instructive one
because it did not fail — 7p0s's author chain `B` selected `label_asym_id` B, a
24-residue peptide, and returned a well-formed 26-token design region that
passed all seven checks while testing the wrong chain. See
`pxf/backbone/chain_ids.py` and `scripts/screen_dev_complexes.py`; the same
author-vs-label divergence hits SC2RBD, TrkA and VEGFA in the benchmark.
`phase0_pack_hook_check.py` now aborts if the featurized split is not the one
the config was screened on.
