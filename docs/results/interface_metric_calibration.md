# What `min_bb_bb` actually measures, and why 0/1 clash-free is the expected reading

Written 2026-09-22, after the first integrated generation cell came back with
every arm at 0/1 clash-free and a minimum of ~0.90 Å, and that was read as
interpenetration in the shared path.

## The two numbers are not the same number

`_geometry` in `scripts/run_integrated_binder_matrix.py` computes the
binder-target minimum over **all atoms** and counts pairs under 2.6 Å.
The section-10 positive control in `docs/target_conditioning_audit.md`
(1jfl, 2.44-4.61 Å, native 2.79 Å) measured **backbone only** --
`docs/gen_stress_stage1.md` says so explicitly: "NATIVE 1jfl, backbone only:
min 2.792 Å, pairs <2.6 Å = 0".

Comparing an all-atom minimum against a backbone-only baseline will always
look alarming, because side chains reach across an interface and backbones
do not.

## Calibration

96 PDL1 designs from the cached-backbone path, `runs/binder_bench/designs_v1`
-- designs that were already AF2-IG scored and reported, i.e. the closest
thing available to a known-good population:

| | min | p05 | median | max |
|---|---|---|---|---|
| all-atom interface min (Å) | 0.524 | 1.120 | **1.828** | 2.651 |
| backbone-only interface min (Å) | 1.989 | 2.731 | **4.099** | 5.873 |
| all-atom pairs < 2.6 Å | 0 | 1 | 5 | 16 |
| backbone-only pairs < 2.6 Å | 0 | 0 | 0 | 1 |

Clash-free under the 2.6 Å threshold: **2/96 all-atom, 92/96 backbone-only.**

Reproduce with the same metric on the same files:

```
python3 scripts/utilities/interface_minima.py \
    runs/binder_bench/designs_v1/PDL1/designs/*.pdb
```

## What follows

1. **A `clash_free` count near zero is the expected reading for good
   designs.** The all-atom 2.6 Å threshold rejects 94 of 96 designs that went
   on to be scored. It carries no information about the design.
2. **0.90 Å all-atom is low but inside the observed range.** Three of the 96
   known-good designs are below it. It is roughly a 3rd-percentile value, not
   an out-of-distribution one.
3. **The discriminating quantity is backbone-only, with the pair count.**
   Genuine interpenetration in section 9 was 0.207 Å with **282** pairs under
   2.6 Å. A tight but real interface in section 10 was 2.44 Å with **1**. The
   count separates these far more cleanly than the minimum does.

Both numbers are now emitted: `min_bb_bb_distance` / `interface_clashes` and
`min_bb_only_distance` / `interface_clashes_bb_only`.

## The threshold is left alone

2.6 Å is mis-specified for the all-atom quantity, in the same way the
side-chain guardrail in `configs/bs_seq_sc/selection.yaml` is mis-specified
and for the same reason: it was declared against one quantity and is being
applied to another. It is **not** retuned here. Retuning a declared threshold
after seeing the result it produced is how you select the answer you already
wanted, and this is now the third guardrail in this project with that defect
-- which is itself worth saying out loud rather than papering over one at a
time. Emitting both numbers is an information change; choosing the guardrail
is a separate decision that belongs to whoever owns the criterion.
