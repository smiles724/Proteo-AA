# Stage III: pairing the refinement pass

Why the second backbone forward in Stage III must denoise **the same draw** as the
first, and what was wrong before it did.

## The shape of a Stage III step

Stage III is `B -> S -> B`. One training step runs the backbone module twice:

| Pass | What it is | Loss |
|---|---|---|
| 1 | `B_pre` — denoise the noisy structure with no side-chain input | `mse` |
| 2 | `B_post` — denoise **again**, this time with the side chain's feedback wired in | `bb_post` |

Between them, `S_phi` builds side chains on pass 1's backbone and its summary is fed
back through the armed channels (`s_trunk_refine`, plus the `a`/`q` hooks).

The pair only means something if the **only** difference between the two passes is
that feedback. `bb_post` vs `mse` is supposed to answer *"did the side chain help the
backbone?"*

## What was wrong

`sample_diffusion_training` (`pxdesign_train/generator.py`) draws three things
**inside itself**:

1. the random augmentation (a global rotation + translation of the structure),
2. `sigma` — the noise level, from the EDM log-normal sampler,
3. the Gaussian noise itself.

Stage III called it twice, so **the second call re-randomised all three**. The two
passes denoised different rotations of the structure at different noise levels.

### Consequence 1 — the two losses were not comparable

`mse` and `bb_post` landed on **independent sigmas**. Denoising at `sigma = 0.9` is
far easier than at `sigma = 12`, so their difference reported *which noise level each
pass happened to draw*, not whether the side chain helped. The pair was not a
comparison, and the signal only survived in aggregate over a full ablation.

### Consequence 2 — the feedback arrived under the wrong rotation

This one is worse, because it caps the mechanism rather than just the measurement.

`h_res'` is **not rotation-invariant**. It descends from `q = c_l + W r_noisy`, a
linear map of the noisy **global** coordinates — rotate the structure and it changes.
The same holds for the atom-level `q` feedback.

So the feedback was computed under pass 1's rotation and injected into a forward
living under pass 2's. From pass 2's point of view, the orientation-carrying
components of that summary are **noise**, and the fusion's best response is to learn
to **ignore them** — silently capping how much the channel can carry, in the stage
whose entire purpose is to measure what that channel is worth.

Inference never had this problem: it carries `h_res_prime_inject` across steps of a
single sampling trajectory, where the frame does not change.

## The fix

`sample_diffusion_training` gained a `reuse_draw` parameter and now also returns
`x_noisy`, so the pair can be threaded:

```python
# pass 1
x_gt_aug, x_denoised, sigma, x_noisy = sample_diffusion_training(...)

# pass 2 -- same structure, same rotation, same sigma, same noise
x_gt_aug_post, x_denoised_post, sigma_post, _ = sample_diffusion_training(
    ..., s_trunk=s_trunk_refine, reuse_draw=(x_gt_aug, sigma, x_noisy),
)
```

Two properties worth stating explicitly:

- **It draws nothing.** Toggling this cannot shift the RNG stream, so two runs that
  differ only in this flag stay comparable.
- **It leaves exactly one difference between the passes**: `s_trunk_refine` plus the
  armed `a`/`q` hooks — i.e. the side-chain feedback and nothing else.

## Verification

Single-structure GPU smoke on the full 263.9M model:

| | pass 1 (`mse`) | pass 2 (`bb_post`) |
|---|---:|---:|
| before | 428.2 | 1397.2 |
| after | 135.17 | 135.17 |

**Exact equality after the fix is the expected result, and is a strong check.** The
feedback fusions are zero-initialised, so at step 0 the feedback is identically zero
and pass 2 *should* reproduce pass 1 bit for bit. That it does confirms no other
difference leaks between the passes.

Once training moves off zero-init, `bb_post - mse` becomes a **per-step readout of
what the feedback channel is worth**. Before this change that difference was noise.

Tests: `tests/test_paired_refinement_pass.py` (4). One of them is a characterisation
test pinning the old behaviour — two passes without `reuse_draw` *do* draw
independently — so the distinction stays visible.

## Known limitation, now unblocked

The feedback is still **averaged over the sigma axis** before injection
(`a_sc`, `q_sc`, `h_res_prime_reduced` in `pxdesign_train/model.py`). The in-code
justification for two of those was precisely the mismatch this change removes:

> the refinement pass draws FRESH sigmas, so its row `s` carries no correspondence
> to round-1 row `s`

That is no longer true — row `s` is now row `s` — so the two live `S -> B` channels
(`a_direct_pre`, `q_direct`) **could** be wired per-sigma. Deliberately not done here:

- the benefit is unmeasured, and averaging 8 draws also reduces variance;
- `q_direct` routes through a call-keyed decoder hook (kept consistent with
  activation-checkpoint recompute), so per-sigma is more than dropping a `.mean()`;
- `h_res_prime_reduced` **cannot** be un-averaged without a Protenix submodule change
  — `s_trunk` is sample-shared, with no per-sample axis to inject into. That channel
  is off by default.

Suggested order: run with this fix first and watch whether `bb_post - mse` becomes
non-zero. If the channel is not learning anything, finer granularity is not the
missing piece.
