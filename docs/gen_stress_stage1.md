# Generation stress test, stage 1: machinery verified, generation is not

**Do not scale to 4×2 or the 32-target panel yet.** The harness passes every
mechanical acceptance check, and the generation it is measuring produces
backbones that interpenetrate the fixed target. Comparing arms on those would
be comparing three equally broken structures.

## What passes

One target (`1jfl`, 228+228 homodimer), one seed, one event, three arms.

| check | result |
|---|---|
| replay, feedback disabled, **through PXDesign's real decoder** | baseline drift **0.000 Å** |
| residue mapping | 228/456 tokens, `first_token 0`, `contiguous_tail false` |
| target coordinates fixed | **0.00e+00 Å** in all arms |
| exactly one injection | baseline 0, bb_only 1, full 1 |
| one shared sequence across arms | 1 unique sequence / 3 rows |
| matched denoiser calls | 31 per arm |
| event is a solver position | `(step 169, substage 0)`, σ = 0.8586 |

## What fails, and how it was localised

Interface geometry was physically impossible: 1092 clashes at 0.207 Å.

**First cause, fixed.** The target was pinned only after the Euler update, so
the next step's churn re-noised it — at `t_hat = 2σ` that is `σ√3`, ~1.5 Å at
σ = 0.86 — and the denoiser conditioned against a smeared target all the way
down. The reported "target drift 0.00 Å" was true and beside the point: the last
pin ran after the last update, so the *final* coordinates were exact while every
*input* had been noised. Pinning now happens after augmentation, after churn
immediately before the denoiser reads it, and on the denoised output.
Clashes 1525 → 1092.

**Second cause, open.** Splitting the remaining clashes by atom class:

| | min distance | pairs < 2.6 Å |
|---|---|---|
| generated BB × target BB | **0.207 Å** | 282 |
| generated BB × target SC | 0.403 Å | 248 |
| generated SC × target BB | 0.219 Å | 307 |
| generated SC × target SC | 0.377 Å | 255 |

Backbone-on-backbone interpenetration is not target-blind side-chain packing;
the generated *backbone* overlaps the target.

**The metric is not at fault.** The native complex under the identical metric:

```
NATIVE 1jfl, backbone only: min 2.792 Å, pairs <2.6 Å = 0
generated:                  min 0.207 Å, pairs <2.6 Å = 282
```

The chains are otherwise placed sensibly — centroids 31.6 Å apart, radii of
gyration 18.2 and 16.8 Å, CA–CA same-index distance 38.8 Å — so this is not a
collapsed or stacked solution. The backbone is simply not respecting the target.

## Why, as far as can be established

* PXDesign's model **does not consume** `fixed_atom_mask` / `fixed_atom_xyz`.
  They are produced by the featurizer and read nowhere in the model code; the
  only consumer found anywhere is `pxdesign_train/stage4.py`, which applies them
  as a coordinate overwrite. So the model's sole channel for the target's
  position is `x_noisy`.
* This repo drives PXDesign through `MONOMER_DATASET`, a preset whose stated
  purpose is making "a monomer's whole chain the design region", and
  `pxf/backbone/driver.py` records that PXDesign's own inference runner "takes a
  target plus a binder to design... cannot denoise a given monomer, and its
  sampler is incompatible with the Protenix revision this repo pins."

So this repo has never run PXDesign's genuine binder-design path, and stage 1
attempted target-conditioned generation through a monomer-denoising driver.
Pinning the target into `x_noisy` is the best available substitute and is
evidently not equivalent.

## What would resolve it

1. Establish whether the donor can do target-conditioned generation at all
   through any path available here — e.g. whether `inference_safe_binder` plus
   hotspots changes the outcome, or whether the trunk conditioning already
   encodes the target and the failure is elsewhere.
2. Failing that, PXDesign's real binder-design inference needs its own Protenix
   revision, which is a separate piece of work from this experiment.

Until one of those lands, the arm comparison is not interpretable: all three
arms interpenetrate equally (1092 / 1093 / 1094 clashes), which is what one
should expect when a one-event 0.02 Å correction is applied to a structure that
is wrong by 2.5 Å.
