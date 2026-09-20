# Training

FaMPNN ships inference only — upstream [issue #9](https://github.com/richardshuai/fampnn/issues/9)
asks for training code and is unanswered. The objectives and the loop here are
**transcribed from the code the released weights were trained by**:
`allatom_design` (commit `51c9d53`), the research tree the `fampnn` package was
factored out of. The preprint (*Sidechain conditioning and modeling for full-atom
protein sequence design with FAMPNN*,
[bioRxiv 2025.02.13.637498](https://www.biorxiv.org/content/10.1101/2025.02.13.637498v1))
supplies the data schedule; where it and the code disagree, the code wins.

| Here | There |
|---|---|
| `pxf/train/losses.py` | `allatom_design/model/seq_denoiser/sd_loss.py` + the `loss:` block of `configs/seq_denoiser/seq_denoiser.yaml` |
| `pxf/train/step.py` | `SeqDenoiser.forward`, `FAMPNNDenoiser.forward`, and the `not is_sampling` branch of `SidechainDiffusionModule.sidechain_diffusion` (deleted from the release), plus `mini_rollout` |
| `pxf/train/trainer.py` | `lit_sd_model.py` + `optim:`/`trainer:` in the same config |

Everything the release *does* ship — the MAR and EDM interpolants, the encoder,
the denoising MLP, the confidence head, the local-frame transforms — is driven
unchanged. Only the orchestration around them is written here.

## The objective

`L_total = L_seq + L_scn + L_psce`, each with weight 1.0 (which is what
Appendix C.1's "we did not experiment with relative weightings" means).

| Term | What it scores | Reduction |
|---|---|---|
| `L_seq` | label-smoothed (0.1) cross entropy on the residues the interpolant masked, excluding unknown (`X`) residues | sum over masked tokens ÷ **crop length** |
| `L_scn` | EDM-weighted L2 to the clean local-frame side chains | mean over supervised **coordinate components**, then × `1/c_out(σ)²` per example |
| `L_psce` | 33-way cross entropy over per-atom error binned on [0, 4] Å, on a stop-gradient rollout | mean over scored atoms, per example |

Masking follows Appendix C: keep probability `t = √u`, `u ~ U(0,1)` — shipped as
`uniform_sqrt_t`, and the released checkpoints carry exactly that. Sequence and
side chains are masked **separately**: `MAR.drop_sidechains` hides a further
random fraction of the side chains at positions whose *identity* is still
visible, so the encoder routinely sees "identity known, conformation unknown",
which is the regime packing runs in.

## Details that are easy to get wrong

Each has a test, and each is a place where an implementation written from the
paper alone diverges from what the released weights were fitted with.

- **`L_seq` is normalized by the crop length, not the masked count**
  (`seq_loss.per_token_avg: false`). The term is a *sum* over masked positions
  divided by the fixed example size, so a batch the interpolant barely masked
  contributes proportionately little. Dividing by the masked count instead
  rescales every step by a random factor and changes the balance against `L_scn`.
  It also makes `loss_mlm` unreadable as a curve, which is why `mlm_per_token` is
  logged alongside it.
- **`L_scn` supervises every resolved side chain, not only the hidden ones.** Its
  mask is `x_mask · (backbone frame exists)`; `scn_mlm_mask` does not appear. The
  denoiser starts from pure noise in the local frame whatever the *encoder* was
  shown, so a residue whose side chain was visible as context is still a real
  prediction rather than a copy. (`L_psce` **is** restricted to the hidden side
  chains — the two masks genuinely differ, and `hidden_sidechains_only=True`
  restricts `L_scn` too, as an ablation.)
- **Ghost slots are supervised to zero.** `x_mask` removes missing atoms but keeps
  the slots a residue type does not have; their local-frame target is exactly `0`.
  The MLP always emits 33 atoms, and this is what teaches it to put the
  nonexistent ones at the origin. Dropping them leaves those outputs untrained.
- **Structural noise is a model setting.** "The 0.3 Å model" is
  `ProteinFeatures.augment_eps = 0.3` — applied to the encoder's atom14 input in
  train mode, never to the diffusion target (`fampnn_0_3.pt` carries
  `augment_eps: 0.3`, `fampnn_0_0.pt` carries `0.0`). Noising `x` in the data
  pipeline would both corrupt the target and double up with the model's own, so
  `StructureCropDataset` refuses it and `TrainSettings.structural_noise` writes
  `augment_eps` instead.
- **`L_scn` divides by components, not atoms.** The mask is `x_mask`, already
  expanded over xyz, so the denominator is `3 × atoms`. A per-atom or per-residue
  average is a different objective at a different scale.
- **Mask polarity.** `mlm_mask` is `1` where a residue was *kept*. Both sequence
  terms score `1 - mask`.
- **Teacher forcing.** The side-chain denoiser is conditioned on the
  *ground-truth* sequence during training, not on the encoder's argmax.
- **8 noise clones.** The conditioning is cloned `training_batch_size_mult` (= 8)
  times with an independent noise level each, because the MLP is cheap next to
  the encoder.
- **psCE bin edges.** The shipped head builds *lower edges* as
  `linspace(0, 4, 33)` and takes centres at `edge + step/2`. Training labels must
  therefore floor onto the edges; rounding to the nearest centre puts every label
  half a bin low. The top bin runs to `inf`.
- **Confidence stop gradient.** The confidence loss must not reach the main
  model. Verified by asserting encoder gradients are bit-identical with and
  without the term.

## Two upstream gaps this had to work around

1. **The training branch of `SidechainDiffusionModule.sidechain_diffusion` was
   deleted from the release,** along with `mini_rollout`. Both are reconstructed
   in `pxf/train/step.py`: the single denoising step, and the 50-step rollout the
   confidence head scores. The rollout stays in the local frame throughout, as
   the original does — a round trip through global coordinates would need a
   backbone the training batch and the (noise-augmented) encoder view do not
   share. Its packing accuracy, ~1.1 Å on a CASP14 backbone, is the check that
   the reconstruction is the right integrator.
2. **`MAR` is a plain class, not an `nn.Module`,** yet is written like one — it has
   `forward`, calls `super().__init__()`, and reads `self.training`. So it is not
   callable and never receives `model.train()`, which would silently disable
   `drop_sidechains`. `pxf/train/step.py` calls `forward` directly and mirrors the
   mode across. Inference never touches `model.interpolant`, which is why this
   went unnoticed.

One upstream detail is deliberately *not* reproduced: `mini_rollout` ends with an
unconditional `self.train()`, so a rollout drawn during validation leaves dropout
on for everything after it. Here the previous mode is restored.

## What the paper does not specify, and the code does

**Optimizer, learning rate, schedule, weight decay, gradient clipping.** None
appear anywhere in the paper; all of them are in the original config:

```
optim.optimizer: noam
Adam(lr=0, betas=(0.9, 0.98), eps=1e-9)
NoamLR(model_size=128, factor=2, warmup=4000)   # peak lr ≈ 2.8e-3 at step 4000
trainer.gradient_clip_val: 0.0                  # nothing is clipped
trainer.precision: bf16-mixed
```

`optimizer: noam` is the default here and reproduces that. `optimizer: adamw`
(1e-4, 1000-step warmup, constant, grad-norm clip 1.0) is kept for *continuing*
training from released weights on a small set, where the Noam schedule — built
for a 300k-step run from scratch — is the wrong shape. Which one a run used is
recorded in every checkpoint under `optim_settings.source`.

**EMA.** Appendix B.2 describes post-hoc EMA (Karras et al. 2024), with the
length chosen after training — 1% for the 0.3 Å model, 25% for the 0.0 Å one. The
original's checked-in default is instead a plain 0.99 decay (`use_phema: false`),
with `PowerFunctionEMA` available behind `use_phema: true`; which was used for the
release is not recorded in the checkpoints. `pxf/train/ema.py` offers a
`relative_length` EMA that realizes "length = 25% of training" directly, plus
periodic snapshots (`snapshot_every`) that are the prerequisite for true post-hoc
reconstruction. The reconstruction itself (Karras's σ_rel → γ mapping) is **not**
implemented.

**SE(3) augmentation.** The original's dataset centres each example on CA and
applies a random rotation (`se3_augment: true`). Not reproduced here, and it
should not matter: the encoder's features are distances and the GVP layers are
equivariant, and the diffusion target lives in per-residue local frames, so the
whole objective is invariant to it.

## The reduction in `L_scn`

`mse_loss.per_token_avg` in the original config, exposed as
`loss.sidechain_reduction`:

| Value | Formula | What it weights |
|---|---|---|
| `per_token` (default, the original) | `L_b = Σ_iac m d² / Σ_iac m`, then `L = (1/B) Σ_b w_b L_b` | every supervised coordinate equally |
| `fixed_size` | `L_b = Σ_iac m d² / (L·33·3)` | the same sum over a constant, so a crop with few resolved side chains scores lower rather than being renormalized |

Both apply the EDM weight per example, after the reduction. The active choice is
returned in the stats and recorded in every checkpoint under
`train_settings.loss.sidechain_reduction`; `sidechain_mse_local` stays
per-component and unweighted so curves remain comparable across both.

## Presets

From Appendix B.2, in `configs/`:

| | CATH | PDB |
|---|---|---|
| fixed example size | 256 | 1024 |
| batch | 64 | 8/GPU × 4 GPUs |
| grad accumulation | 1 | 4 (effective batch 128) |
| steps | 100k (~6–8 h, 1×H100) | 300k (~3 days, 4×H100) |
| EMA length | 25% (0.0 Å model) | 1% (0.3 Å model) |
| structural noise (`augment_eps`) | 0.0 or 0.3 | 0.0 or 0.3 |

`batch_size` is per-process, so a single-GPU PDB run reproduces the effective
batch with `grad_accum_steps: 16`. **Multi-GPU is not implemented** — there is no
DDP wiring here, so the 4-GPU preset needs either that or accumulation.

## Running

```bash
# continue training from the released 0.0 A weights
python scripts/train.py --pdb-dir <dir> --out runs/ft \
    --init-weights 0.0 --config configs/train_cath.yaml

# only the side-chain denoiser, leaving the sequence encoder frozen
python scripts/train.py --pdb-dir <dir> --out runs/sc --trainable scn_denoiser

# resume (checkpoints from this loop only; the released ones carry no optimizer)
python scripts/train.py --pdb-dir <dir> --out runs/ft --resume runs/ft/checkpoints/step00005000.pt
```

`--noise` sets the encoder's `augment_eps`; omit it to keep whatever the loaded
checkpoint carries, which is what matches the variant being continued.
`--optimizer adamw` switches from the original's Noam schedule to the
fine-tuning one.

**No validation loop is wired up.** The original evaluates each term at *fixed*
noise levels — the sequence term at `t_seq ∈ {0.0 … 0.9}`, the diffusion term at
`t_scd ∈ {0.0 … 0.9}` with the sequence fully masked — so its curves are
comparable across steps rather than averaged over a random draw. The two hooks
that needs are here (`diffusion_loss(..., t_scd=)` and `interpolant.forward(batch,
t=)`); the loop around them is not.

Checkpoints carry `state_dict` + `model_cfg` *and* optimizer/EMA/step, so one file
both resumes this loop and loads directly into upstream inference or
`scripts/pack.py`.

## Coupling: which σ_B the adapters train at

Phase 1–3 train residual adapters conditioned on the backbone noise level,
`A(z, σ_B) = W_out SiLU(W_in [LN(z), e(log σ_B)])`, so the σ_B *distribution* is
part of the objective. `pxf/couple/schedule.py` samples it from the interval the
coupling is intended to run in rather than fixing one value — with a single
training σ_B the noise embedding sees a constant, and the adapter is only
licensed at that σ_B while nothing in the code looks wrong.

The default window is the late end of PXDesign's own 400-step schedule, defined
by the same formula the sampler uses (`InferenceNoiseScheduler`, `s_max` 160,
`s_min` 4e-4, ρ 7, σ_data 16):

```
 step      0        200       280     320      360     400
 sigma  2560       56.0      5.06    1.02     0.126   0.0064
        |-----------|----------|-------|--------|-------|
         coupling off          [ ---- training window ---- ]
                                σ ∈ [0.01, 5.0] = steps 281–395
```

| `sigma.mode` | Draws |
|---|---|
| `trajectory` (default) | a step index uniform over the window, returning that step's exact σ — 115 distinct values, and every one is a σ the sampler really visits |
| `loguniform` | continuous and uniform in `log σ` over the same interval |
| `fixed` | one value; the degenerate case, kept for deployment-matched ablations |

A window too narrow to hold two trajectory steps is an **error**, not a
degenerate run — that failure is the whole reason the module exists. Likewise
`--sigma` outside `fixed` mode is refused rather than silently ignored. Every
run records the distribution in `run_config.json` and in each checkpoint under
`settings.sigma_schedule`, and the training log reports `*_sigma_b_mean` and
`*_sigma_b_range` per window so a collapsed range is visible rather than
inferred.

**Verified.** A 10-step phase-1 run draws 10 distinct `sigma_B` spanning
0.013–3.78 Å, and `L_SC` tracks them — 8.50 at σ 3.78, 2.61 at σ 0.013 — which is
what says the noise level reaches the model rather than only the log. Under the
previous fixed σ = 1.0 all ten rows would have been identical. A 12-step run on
the real PXDesign donor completes in 9.6 s on an H200.

`L_SC` is the **diffusion term alone** — not `L_seq + L_scn + L_psce`. Phase 1
asks whether PXDesign's `a_token` improves side-chain packing, and the sequence
is held fixed, so an MLM term would score a prediction of something already
given and a change in the loss could no longer be read as a change in packing.
`test_l_sc_equals_the_diffusion_term_alone` and
`test_l_sc_carries_no_mlm_or_confidence_term` pin it.

Its **scale changed** when the reduction was corrected against the original
training code: `L_SC` is now the same quantity FaMPNN was fitted with, so an
absolute value from a phase-1/2/3 run predating that (`per_residue`, a
per-residue average) is not comparable with one after it. The relative
comparison each phase actually rests on — adapters live against
`enable_bb_to_sc=False`, at the same σ_B and the same noise draw — is unaffected,
since both arms move together. The coupling stages keep AdamW at a constant low
rate rather than the original's Noam schedule, which is the right shape for a
20k-step adapter fit and is recorded as a departure.

## Verified

- **Overfit**: two CASP14 crops, 120 AdamW steps at 1e-3 — local-frame MSE
  0.0135 → 0.0032 Å² per coordinate, `mlm_per_token` 0.724 → 0.468, masked-token
  accuracy 0.90 over the second half. Kept as a test. `loss_main` over the same
  run goes 0.176 → 0.186, which is the normalization at work rather than a
  failure to learn: the last window happened to mask more residues.
- **Rollout**: the reconstructed `mini_rollout` packs a CASP14 crop at **1.08 Å**
  side-chain RMSD, which is FaMPNN's published packing accuracy — the check that
  the reconstructed integrator is the shipped one.
- **Resume**: step counter, optimizer moments and RNG restored; loss continues
  its trajectory.
- **Stop gradient**: encoder gradients bit-identical with and without `L_psce`.

Read the curve off `mlm_per_token` and `sidechain_mse_local`, not off
`loss_main`: with the original's normalization, `loss_mlm` moves with how much
the interpolant happened to hide in that window.

Not verified: a full-scale 100k/300k run, or that these settings reproduce the
paper's reported numbers.

**Stale, from before the objective was corrected against the original code.** A
400-step overfit under the previous (paper-derived) objective produced a
checkpoint that packed its memorized structure at 0.116 Å side-chain RMSD with
χ-20 0.995. That run is not reproducible with the current defaults — the target
set, both normalizations and the optimizer have all changed — and the number has
not been re-measured.

## Does the coupling work? The held-out measurement

`train_couple.py` reports `L_SC` — FaMPNN's diffusion loss, at randomly drawn
noise levels, **on the structures it is fitting**. That curve cannot answer the
question phase 1 poses. It says the adapter fit something; it does not say the
packing improved, and it says nothing about structures the adapter never saw.
Phase 1 as shipped produced no validation of any kind.

`scripts/eval_couple.py` is the measurement:

```bash
# export the held-out set once (val split; the phase 1 manifest is train split)
python scripts/export_afdb_cifs.py --count 256 --split val \
    --out-dir /hai/scratch/yfsun/afdb_laproteina/cif_val \
    --manifest configs/val_structures_afdb.txt

CHECKPOINT=.../phase1_<jobid>/checkpoints/final.pt \
    OUT=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/phase1 \
    sbatch scripts/slurm/eval_couple.sh
```

The split is disjoint by construction and checked: 2,000 train ids against 256
val ids, intersection 0.

### Two arms, and why that is the right control

Every structure is packed twice — once with the adapters live, once with
`enable_bb_to_sc=False` — at the same σ_B, from the same noise draw, under the
same RNG seed. Because the adapters are **zero-initialized**, the disabled arm
does not approximate the pretrained system, it *is* the pretrained system. And
in phase 1 the backbone proposal is computed before `A_BS` is applied, so both
arms see a bit-identical backbone and the proposal's own error cancels out of the
delta.

Running with no `--checkpoint` is the pipeline's self-test rather than a wasted
job: with untrained adapters the two arms must come out identical to the last
digit. They do.

### σ_B is swept, not sampled

Both adapters take log σ_B as input, so one evaluation point licenses a claim
only at that point. The sweep walks the same window the schedule draws from in
training, and the report is per-σ as well as pooled — an adapter that helps at
low noise and hurts at high noise is a real outcome that pooling hides.

### The frame correction, which is not optional

`sidechain_metrics.score` builds one set of residue frames from the reference and
uses it for both structures, on the stated grounds that the backbone is shared.
That is true for `eval_protenix_sidechain.py`, which packs onto the deposited
backbone. **It is false here, twice over:** the featurizer centers the structure
while FaMPNN's parse of the same file does not, so the two are not in a common
frame at all; and the backbone being packed is a diffusion proposal, not the
deposited one. Scoring without correcting for this reports ~20 Å RMSD on a
perfectly good packing — which is how the problem was found.

`place_on_native_backbone` transfers each residue's side chain through its own
backbone frame, making the shared-backbone assumption true by construction. What
is then measured is side-chain conformation relative to its own backbone: the
standard side-chain accuracy quantity, invariant to global pose, isolating
packing from backbone error. Backbone error is reported separately, per σ, as
Kabsch-superposed `backbone_rmsd`, so "packing improved while the proposal
drifted" stays distinguishable from "both improved".
`test_a_rigidly_moved_prediction_scores_as_its_own_geometry` pins the invariance.

### Smoke-tested end to end

One 34-residue val structure, CPU, 20 pack steps, untrained adapters:

| σ_B | backbone RMSD | side-chain RMSD | rotamer recovery | bad bonds |
|---:|---:|---:|---:|---:|
| 0.010 | 0.014 Å | 1.563 Å | 0.621 | 0.000 |
| 4.881 | 1.101 Å | 2.693 Å | 0.414 | 0.079 |

Both arms identical at every σ, as they must be with zero adapters. The
monotone degradation with σ_B, and a backbone that lands at 0.014 Å when there is
almost no noise to remove, are what say the cycle and the frame handling are
wired up correctly. These are not performance numbers — one short protein, a
truncated rollout.
