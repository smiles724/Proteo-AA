# Does the side chain actually help the backbone? A controlled Stage III measurement

Stage III exists to close the loop B → S → B: the side chain is supposed to feed
information back and make the backbone better. Until now nothing in the run
measured that. The reported validation metrics (`val_mse`, `val_sc_local`,
`val_aa_acc`) were flat across three separate Stage III runs, which read as
"Stage III does not work" — but those metrics score the **first** backbone pass,
which by construction has not seen the side chain. The quantity that answers the
question was in the logs the whole time and was not being read.

This note records how the comparison was built and what it says.

Status: both runs reached the 24 h wall clock at step ~7 900 of 8 000
(`MAX_STEPS=8000`, `EVAL_INTERVAL=1000`), so §2 reports the full curve. The
sections below were first written at step ~3 200; §2.5 records the final numbers,
and §2.6 the leakage ablation that rules out the obvious alternative explanation.

---

## 1. How the comparison is constructed

### 1.1 The two numbers being compared

Both are the *same* function on the *same* target, differing only in which
prediction is scored:

| log field | `loss.py` | prediction scored | has it seen the side chain? |
|---|---|---|---|
| `mse` | line 355, `_mse_term(pred_coordinate, …)` | `out["x_denoised"]` — first backbone pass (B_pre) | no |
| `bb_post` | line 523, `_mse_term(post_pred_coordinate, …)` | `out["post_pred_coordinate"]` — refinement pass (B_post) | yes |

Same `_mse_term`, same `coordinate_mask`, same augmented ground truth, both in
Å². So

```
gain = (mse - bb_post) / mse
```

is "how much the refinement pass reduces the coordinate error", and it is the
direct measurement of what Stage III is for.

### 1.2 Fix 1 — make the two numbers comparable at all

`sample_diffusion_training` draws the augmentation, the sigma and the Gaussian
noise *inside itself*. Stage III calls it twice, so before `8832c61` the second
call re-randomised all three: `mse` and `bb_post` scored **different rotations of
the structure at different noise levels**. Their ratio reported which sigma each
pass happened to draw, not whether the side chain helped. Denoising at σ≈0.9 and
at σ≈12 are not the same task. It also broke the channel itself: `h_res'`
descends from `q = c_l + W·r_noisy`, a linear map of noisy *global* coordinates,
so computed under one rotation and injected under another its
orientation-carrying components are noise — and the fusion's best response is to
learn to ignore them.

### 1.3 Fix 2 — chain the passes instead of repeating one

`287ebc2` supersedes that fix. The second pass now receives
`precomputed_input=(x_gt_aug, refinement_sigma, x_denoised)`
(`model.py:1977`), which skips the augment/draw/add-noise steps entirely and
denoises the given state directly. Three consequences:

* **Same frame.** `x_gt_aug` is reused, so both passes live in one rotation —
  what fix 1 achieved, now by construction.
* **Chained, not repeated.** The input is `x_denoised`, i.e. **B_pre's own
  prediction**, not the original noisy state. B_post is a refinement operator over
  an already-denoised structure, so `mse → bb_post` reads as "error of the first
  prediction → error after refining it".
* **Fixed conditioning at σ = 2.0.** `torch.full_like(sigma, refinement_sigma)`
  changes only the DiffusionModule conditioning and the EDM `c_skip`/`c_out`; it
  adds **no** noise, and the coordinate input stays B_pre's prediction exactly.
  The value matters: with `sigma_data=16` the Karras tail value (≈0.0077) gives
  `c_skip = 0.99999977`, `c_out = 0.0077`, making repeated refinement nearly an
  identity map and suppressing Ångström-scale corrections. σ=2 gives
  `c_skip = 0.9846`, `c_out = 1.985`, matching the 1–3 Å backbone-error scale. The
  same value is used by training B_post and every inference B_refine call.

Because B_post starts from an already-good prediction it has an easier job than
B_pre, so `bb_post < mse` is **not** by itself evidence of anything — which is
exactly why §1.4 exists.

### 1.4 The control — attribute the gain to the side chain, not to iteration

Even with 1.2 and 1.3, "second pass beats first pass" could simply mean iterative
refinement helps, with the fusion modules learning to exploit a second look
regardless of what they are fed. So two runs, identical in everything except the
two S→B feedback channels:

| | arm | `a_direct_pre` | `q_direct` | `a_bs_concat` | `q_bs` | `bb_context` | `hres_inject` |
|---|---|---|---|---|---|---|---|
| **A** treatment (103961) | `default` | **True** | **True** | True | False | True | False |
| **C** control (103966) | `a-bs` | **False** | **False** | True | False | True | False |

Everything else is shared: same start (Stage II `step52500` + AA head
`step9000`), `lr=1e-5`, crop 384, `iters_to_accumulate=8`,
`refinement_sigma=2.0`, `EVAL_INTERVAL=1000`, `MAX_STEPS=8000`, same seed.

Two mistakes were made and corrected while building this control, both worth
recording because either would have produced a confounded answer:

* **Arm `no` is not the right control.** It also disables `a_bs_concat` and
  `q_bs`, i.e. the B→S direction, so it would have differed from the treatment in
  two ways at once. `a-bs` differs in nothing but the two S→B channels.
* **`a-bs` was unreachable from the CLI.** `SC_ABLATION_ARMS` defines 21 arms but
  `--sc-ablation-arm` hardcoded `choices=` to 7, so the control failed at argparse
  with `invalid choice: 'a-bs'`. `apply_sidechain_ablation_arm` already validates
  against the registry and names the valid arms, so `choices=` was removed and the
  registry is now the single source of truth. (The registry cannot be imported at
  parse time: `parse_args` runs before `_bootstrap_paths`, so `pxdesign_train` and
  its protenix import are not yet on `sys.path`.)

---

## 2. The result

### 2.1 Per-step, over all logged micro-batches

| | A: feedback **ON** | C: feedback **OFF** |
|---|---|---|
| micro-batches | 504 (steps 50–3150) | 496 (steps 50–3100) |
| `bb_post` beats `mse` | **67.9 %** | 45.2 % — a coin flip |
| median gain | **+0.61 %** | +0.00 % |
| first half → second half | −0.14 % → **+3.05 %** | −0.39 % → **+0.09 %** |

### 2.2 On the 308-protein validation set

| step | A: `mse` → `bb_post` | A gain | C: `mse` → `bb_post` | C gain |
|---|---|---|---|---|
| 1000 | 21.07 → 21.05 | +0.1 % | 20.29 → 20.30 | −0.0 % |
| 2000 | 19.38 → 19.06 | +1.7 % | 19.21 → 19.20 | +0.1 % |
| 3000 | 20.63 → 19.49 | **+5.5 %** | 21.21 → 21.19 | +0.1 % |

### 2.3 What this establishes

**The side chain carries usable information into the backbone.** With the two S→B
channels off, the refinement pass does essentially nothing — a coin flip, gain
pinned at zero, and it never leaves zero. With them on, the gain grows from
nothing to +3 % (train) / +5.5 % (validation) over 3 000 steps and is still
climbing. Since the arms differ in nothing else, the gain is attributable to the
feedback channels rather than to iterative refinement.

This is also the correction to the earlier reading of Stage III. "Flat
validation" and "nothing is learning" are not the same claim: the co-evolution
channel was learning the whole time, in a quantity that was not being reported.

### 2.4 What it does not establish

* **Nothing else improves.** `val_sc_local` drifts down slightly and *identically*
  in both arms (A 3.377→3.308, C 3.370→3.285) — correct, since the arms differ
  only in S→B and nothing about S_phi's own supervision changed. `val_aa_acc` is
  flat at ≈0.130 in both, still at the ceiling the frozen-trunk experiment
  established. `val_mse` itself does not improve in either arm (19–21, noisy).
  The side chain improves the *refined* prediction, not the first pass.
* **One step, not a trajectory.** This is a single refinement step at
  `refinement_sigma=2.0`. It says the channel carries information; it does not
  say sampled structures are better. That needs an inference-time comparison.
* **Not converged.** Both runs stopped at the 24 h wall clock, step ~7 900. The
  gain is monotonic and the control flat, but the curve is still climbing at the
  last eval — the plateau, if any, was not reached.
* **Validation noise.** `val_n=308`, not 491 — the Stage III script passes
  `--max-n-token` equal to the crop (384), which rebuilds the index with a
  384-token cap. `val_mse` swings ±9–15 % between evals, which is why a few-percent
  effect is invisible there and has to be read as the A-vs-C *ratio*.

### 2.5 The full curve

![Stage III backbone improvement, with the GT-frame ablation beside it](../runs/figures/stage3_backbone_improvement.png)

```bash
python scripts/plotting/plot_stage3_refinement_gain.py \
  --out runs/figures/stage3_backbone_improvement.png
```

The script re-reads the four training logs, so it stays current as the GT-frame
runs advance. Both arms ran to the wall clock. Per micro-batch over all logged
steps:

| | A: feedback **ON** (103961) | C: feedback **OFF** (103966) |
|---|---|---|
| micro-batches | 1 256 (to step 7 850) | 1 272 (to step 7 950) |
| `bb_post` beats `mse` | **85 %** | 59 % |
| median gain | **+7.28 %** | +0.09 % |
| first half → second half | +1.30 % → **+13.98 %** | +0.00 % → +0.14 % |

On the 308-protein validation set:

| step | A gain | C gain |
|---|---|---|
| 1000 | +0.09 % | −0.05 % |
| 2000 | +1.65 % | +0.05 % |
| 3000 | +5.53 % | +0.09 % |
| 4000 | +8.87 % | +0.14 % |
| 5000 | +12.30 % | +0.05 % |
| 6000 | +14.28 % | +0.10 % |
| 7000 | **+16.24 %** | +0.14 % |

`val_mse` in arm A over the same steps: 21.07, 19.38, 20.63, 20.40, 20.00, 19.82,
19.64 — flat and noisy, i.e. the first pass does not improve. `val_bb_post`:
21.05, 19.06, 19.49, 18.59, 17.54, 16.99, **16.45** — a monotone decline. The
whole effect is in the refined prediction, and the control shows the refinement
pass is inert without the two S→B channels.

### 2.6 Ruling out GT-backbone leakage

The obvious alternative explanation is that the feedback features carry backbone
*geometry* rather than side-chain *content*. A 2x2 over {frame source} x
{feedback} tests it; see `stage3_gtframe_leak_ablation.md` for the design. Result:
handing S_phi the GT backbone does not raise the channel benefit at any matched
step — `G_gt − G_pred` runs −3.4 % to −0.8 % through step 6 000 and +0.15 % at
step 7 000, i.e. the two arms converge. No part of the gain above is attributable
to a GT path; the gain is side-chain content.

See the figure in §2.5.

### 2.7 Reproducing

```bash
CK=…/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt
AA=…/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt

# A — treatment
LOAD_CHECKPOINT=$CK AA_HEAD_CHECKPOINT=$AA EVAL_INTERVAL=1000 \
  CHECKPOINT_INTERVAL=1000 MAX_STEPS=8000 \
  sbatch scripts/training/slurm_stage3_coevolution_monomer.sh

# C — control, identical but for the two S->B channels
LOAD_CHECKPOINT=$CK AA_HEAD_CHECKPOINT=$AA EVAL_INTERVAL=1000 \
  CHECKPOINT_INTERVAL=1000 MAX_STEPS=8000 \
  sbatch scripts/training/slurm_stage3_coevolution_monomer.sh --sc-ablation-arm a-bs
```

The per-step statistic is `(mse - bb_post) / mse` over the training log lines,
split first-half / second-half; the validation version is the `val_mse` and
`val_bb_post` fields of the `val_n=` lines.
