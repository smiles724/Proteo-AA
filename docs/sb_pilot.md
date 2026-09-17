# The SC → BB pilot

One corrective event, at a fixed noisy state, with a fixed native sequence and
every pretrained component frozen. The question is narrow on purpose:

> does predicted side-chain information improve backbone denoising **beyond a
> trained BB-only control**?

Not "beyond the uncorrected proposal". That is the weaker comparison, and
passing it alone shows only that a learned correction at the decoder input helps
— which a control reading no side chains at all can also do.

## What runs

```
with no_grad:            frozen, cacheable
    bb0, a0 = D(x_σ, σ; 0)
    sc0     = P(bb0, s; selected BB→SC policy)
    packed  = E(bb0, sc0, s)
z        = R(packed)     trainable
Δa       = A_SB(z, σ)    trainable
bb1, _   = D(x_σ, σ; Δa) NOT under no_grad
loss     = L_BB(bb1, target, mask)
```

Both denoiser calls get the same `x_noisy`, the same `σ` and the same
conditioning. The corrective call must stay outside `no_grad`: PXDesign's
parameters are frozen, but its atom decoder has to differentiate with respect to
`Δa`. `CoupledDenoiser.corrective_event` raises if `Δa` comes back not requiring
grad while adapter parameters do — the alternative is a healthy-looking loss
curve that trains nothing.

The injection site is the existing hook: after `layernorm_a`, immediately before
`atom_attention_decoder`. Downstream of the diffusion transformer, upstream of
coordinate production, so it is a small first intervention. A negative result
concerns *this* site; moving feedback before the transformer is a separate
capacity experiment.

## The three arms

| arm | `z` carries | what it establishes |
|---|---|---|
| `full` | node (`h_packed`), chi torsions, packing environment, sequence, psCE | the candidate |
| `bb_only` | node (`h_base`, side-chain-masked), sequence | benefit beyond a learned BB correction |
| `generic` | nothing — `z ≡ 0` | benefit beyond generic σ conditioning |

Groups a variant does not read are fed **structural zeros** rather than removed,
so all three have an identical parameter count (262,560) and differ in
information, not capacity. `test_the_controls_cannot_see_a_rotamer_change`
rotates chis with the backbone and sequence fixed and asserts both controls' `z`
is bit-identical while the candidate's moves.

Measured, so "approximately parameter count" can be stated exactly: `full` and
`bb_only` have all 262,560 parameters receiving gradient; `generic` has 261,632
(99.65%), the missing 928 being the readout's LayerNorm and sequence embedding,
which `z ≡ 0` makes unreachable. That is the correct behaviour for a σ-only
control — it *should* be unable to use the node features or the sequence — and
0.35% is not a capacity difference that could explain a result.

Same receiving hook, same fixed pool of 512 examples, same optimizer budget,
same seeded frozen half. The upstream cache is shared, so the packing each arm
reads is the same file, not merely the same distribution.

## Measured before training

`pxf.couple.probes` on T1031, native side chains:

| quantity | relative change |
|---|---|
| floor (identical input re-encoded) | 0.0 |
| ceiling (masked → visible side chains) | 0.77 |
| chi rotation, 10° | 0.06 |
| chi rotation, 60° | 0.27 |
| chi rotation, 120° | 0.41 |

Torsion rotations are the probe that matters: they preserve every bond length,
every bond angle, the backbone and the sequence, so they land on another *valid*
rotamer. Gaussian coordinate noise and collapse-to-Cα are kept as wiring
diagnostics — they break covalent geometry, so a response to them shows the
encoder reads the block, not that it is sensitive to packing.

Invariances, with the tolerance each deserves:

| invariant | measured | allowed |
|---|---|---|
| nonexistent atom37 slots scrambled | exactly 0 | 0 |
| rigid rotation + translation | ~2e-5 | 1e-3 (float32) |
| padded-row coordinates | 2.6e-4 … 1.2e-3 | 5e-3 |

The last is **not** exact: FaMPNN's encoder leaks a little of a padded row's
coordinates into real residues' node features, independent of length and larger
on a predicted packing than a native one. It does not affect the pilot for a
structural reason — the converter gives every structure `seq_mask = 1` at its own
length and the cycle runs one structure per forward — and
`encode_predicted_packing` warns if that ever stops being true.

`scripts/probe_sb_feedback.py` re-runs all of this on the real panel, against
predicted packings on denoised backbones.

## Two correctness fixes this needed first

**The re-encoding mask.** The converter marks every side-chain slot missing for a
backbone-only proposal, correctly. Passing that same mask into
`encode(..., sidechains=predicted_sc)` left the generated atoms masked anyway,
because `build_atom_mask` multiplies by `1 - missing_atom_mask` — so
`h_packed == h_base` exactly, and the feedback path carried nothing about the
packing. `pxf.couple.visibility` computes post-packing availability instead.
`test_the_old_mask_is_what_made_them_equal` pins the mechanism.

**The evaluator's stage.** `predicted_atom37` assembled from `bb0_dense`
unconditionally, so `--run-feedback` produced a report whose backbone metrics
were still the *uncorrected* proposal's. It now requires an explicit stage once a
cycle has run feedback, and `stage="bb1"` requires `sc1` — a fresh packing on the
corrected backbone — rather than assembling `bb1` with side chains built for
`bb0`.

## The exit decision

`scripts/eval_sb_feedback.py`. Arms: `bb0`, `zero` (wiring), one per checkpoint,
`perturbed` (rotamer-perturbed input to the full arm), and `refine` — the
computational baseline, which spends the second denoiser call on one real EDM
step down the published schedule rather than repeating a deterministic call.
Runtime is measured alongside call counts, because a call plus a 50-step packing
rollout is not the same work as a call.

Criterion: backbone RMSD improvement ≥ `max(0.05 Å, 3% of the strongest
comparable-cost alternative)`, a paired bootstrap interval excluding zero, and no
material side-chain or chemistry regression.

**The interval is protein-level, not cluster-level.** The roadmap asks for
paired protein *and* cluster uncertainty, and the second is not available here:
the la-proteina AFDB manifest carries `afid, split, shard_id, length` and no
cluster assignment, so there is nothing to resample by. The train/val split is
by accession, which guards against train→val leakage; what it does not bound is
correlation *within* the val panel, so if it contains homologues the interval is
optimistic. Quoting it as a cluster-level interval would overstate it. A panel
with cluster labels (or a clustering pass over the 256 val structures) is what
would close this.

**On the criterion's shape at this noise level.** With σ ∈ [0.1, 2] Å the
proposal is already at ~0.32 Å, so `max(0.05 Å, 3%)` is dominated by the
absolute floor and demands a ~15% relative improvement. That is a demanding bar,
and it is the stated one — worth noticing when reading a result, not worth
moving after seeing one. Beating `bb0` demonstrates
practical correction; **claiming a side-chain-specific benefit additionally
requires beating `bb_only` and `generic`.**

Only after that passes does the corrected estimate go into the sampler's normal
update for a one-event rollout test. The two-event training stage is a separate
experiment after that.

## Configuration

`configs/couple_phase2_pilot.yaml`. 2,000 steps (extend to 5,000 only if
promising), lr 1e-4, grad-norm clip 1.0, 50 packing steps, σ ∈ [0.1, 2.0] Å.

That σ window was verified rather than assumed: with `sigma_data = 16 Å` and
`x_noisy = x + σ·ε`, σ is in Ångströms of per-coordinate noise, and [0.1, 2.0] is
steps 305–363 of PXDesign's published 400-step trajectory — 59 discrete values,
15% of the trajectory. `A_SB`'s gate is exactly 1 across it and tapers outside.

The Phase-1 BB→SC policy is **bypass**, held fixed across every arm. No policy
has been selected: the effect is partly sample-specific but how much has not
settled into a choice (shuffled-donor retention on `chi_recovery_20deg`: 20% on
AFDB val, 53% on CASP14/15, 69% on CASP16), and the gate sweep is likewise
unpicked. Bypassing does not prevent testing `A_SB`; it means the packing the
feedback reads is pretrained FaMPNN's, which needs no Phase-1 dependency.
`--init-from-phase1` exists for when a policy is selected — weights-only, with
the optimizer, step counter and EMA reset, because a 2k pilot must start at
step 0.

Activation checkpointing stays off (the driver's default): Protenix recomputes
the forward during backward and the injection hook makes the recomputation
diverge.

## Running it

```bash
# all three arms; bb_only and generic wait on full so they share the cache
VARIANT=full    TAG=full    sbatch scripts/slurm/train_sb_pilot.sh
VARIANT=bb_only TAG=bb_only sbatch --dependency=afterok:$J1 scripts/slurm/train_sb_pilot.sh
VARIANT=generic TAG=generic sbatch --dependency=afterok:$J1 scripts/slurm/train_sb_pilot.sh

# the exit decision, all arms on one held-out panel
ARMS="full=.../full/checkpoints/final.pt \
      bb_only=.../bb_only/checkpoints/final.pt \
      generic=.../generic/checkpoints/final.pt" \
  sbatch scripts/slurm/eval_sb_feedback.sh
```

`--crop-size` must cover the longest structure: the featurizer does not crop the
design region, it refuses one larger than the crop. 512 covers both AFDB
manifests and costs nothing for short structures, since nothing is padded to it.
