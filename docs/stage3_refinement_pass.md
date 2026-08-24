# Stage III: chained backbone refinement

Stage III uses one `B -> S -> B` cycle during training. The second backbone pass
continues from the structure predicted by the first pass; it is not another
one-step prediction from the original noisy structure.

## Coordinate and feature flow

```text
x_noisy -- B_pre --> x_denoised^(1)
                         |
                         +-- S_phi --> side-chain feedback (h_res', a, q)
                         |
                         +-- B_post(feedback) --> x_denoised^(2)
```

The two backbone passes therefore have different coordinate inputs:

| pass | coordinate input | side-chain feedback |
|---|---|---|
| `B_pre` | sampled `x_noisy` | off |
| `B_post` | `B_pre`'s `x_denoised` | on |

`B_post` keeps the first pass's augmented target but uses a fixed
`refinement_sigma=2.0`, shared with inference. The value is an EDM residual scale
and refinement-mode condition; the refinement path does **not** sample or add a
Gaussian noise tensor. With backbone `sigma_data=16`, it gives
`c_skip=0.9846` and `c_out=1.9846`, rather than collapsing the update near the
tail of the Karras schedule.

Because one shared DiffusionModule now sees both ordinary noisy inputs and
already-denoised refinement inputs, `B_post` also receives a learned
refinement-pass embedding in
`s_trunk`. The embedding is zero-initialized so a Stage II warm start remains a
no-op, then learns in the backbone phase of alternating training.

## What was wrong before

The implementation went through two intermediate behaviours:

1. `B_post` called the training sampler normally, which produced a new rotation,
   sigma and Gaussian noise. Its coordinate frame did not match the side-chain
   feedback generated after `B_pre`.
2. The paired-pass fix removed the second random draw, but made `B_post` denoise
   the same original `x_noisy` again. That aligned the frames, yet it was still a
   repeated prediction rather than iterative refinement.

The current implementation uses `precomputed_input=(x_gt_aug, sigma, x_denoised)`:

```python
# First backbone pass: ordinary one-step diffusion training input.
x_gt_aug, x_denoised, sigma, x_noisy = sample_diffusion_training(...)

# Second backbone pass: continue from the first prediction; draw/add no noise.
x_gt_aug_post, x_denoised_post, sigma_post, _ = sample_diffusion_training(
    ...,
    s_trunk=s_trunk_refine,
    precomputed_input=(x_gt_aug, sigma, x_denoised),
)
```

No detach is applied to `x_denoised`, so the post-backbone loss can differentiate
through the chained coordinate path as well as through the side-chain feedback
path. Which parameters an optimizer updates is still controlled by Stage III's
alternating-training phase.

## Inference rolls out the learned refinement transition

Inference is deliberately split into two phases. The ordinary EDM trajectory
first generates an initial clean-backbone prediction `B_0`. Euler belongs only
to this generation phase. Co-evolution then rolls out the transition learned by
the one-cycle training forward:

```text
S_0 = S_phi(B_0)
B_1 = B_refine(B_0, feedback_0)

S_1 = S_phi(B_1)
B_2 = B_refine(B_1, feedback_1)
...
```

Every refinement backbone call receives the latest prediction directly, carries
the refinement-pass embedding, and consumes the current side-chain summaries
(`h_res`, `a`, and `q` channels when enabled). The loop neither samples/adds new
noise nor applies an Euler update between `B_k` and `B_(k+1)`.

Training unrolls only one `B -> S -> B` transition at a randomly sampled
timestep; inference repeatedly applies it, so accumulated rollout error remains
possible just as in ordinary one-step diffusion training. `refinement_steps`
controls the rollout length (default: 3).

## How to interpret the losses

The old paired-input implementation produced `mse == bb_post` at initialization
because the feedback fusion was zero-initialized and both backbone passes received
the same coordinate input. That equality was a useful diagnostic for that
intermediate implementation, but it is **not** expected for chained refinement.

Now `B_post` receives a different coordinate tensor even when every feedback
fusion outputs zero. Therefore:

```text
bb_post - mse = effect of another backbone application + effect of SC feedback
```

It is no longer a pure per-step measurement of the feedback channel. To isolate
feedback quality, compare two `B_post` calls that both start from the same
`x_denoised^(1)`, with the feedback channels off versus on.

## Remaining separate choice

This cycle contract does not decide whether `S_phi` itself receives GT or
predicted backbone geometry during training. That is the separate Stage III
teacher-forcing versus Stage IV inference-matching choice. The current stage
routing still sets `predicted_frame=True` for both and is marked with
`TODO(stage-contract)` at both assignment sites; this inference change does not
silently alter that curriculum yet.
