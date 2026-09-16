# Training

FaMPNN ships inference only — upstream [issue #9](https://github.com/richardshuai/fampnn/issues/9)
asks for training code and is unanswered. This implements it from the preprint
(*Sidechain conditioning and modeling for full-atom protein sequence design with
FAMPNN*, [bioRxiv 2025.02.13.637498](https://www.biorxiv.org/content/10.1101/2025.02.13.637498v1)),
reusing every training ingredient upstream does ship.

## What the objective is

`L_total = L_MLM + L_diff`, summed with **no relative weighting** — Appendix C.1
states this explicitly ("We did not experiment with relative weightings"). The
confidence head is trained separately on stop-gradient inputs.

| Term | Definition | Paper |
|---|---|---|
| `L_MLM` | cross entropy on the residues the interpolant masked | C.1 |
| `L_diff` | EDM-weighted L2 to the clean local-frame side chains | 4.3.1, D.3.2 |
| `L_conf` | 33-way cross entropy over per-atom error binned on [0, 4] Å | D.4.1 |

Masking follows Appendix C: keep probability `t = √u`, `u ~ U(0,1)` — a concave
schedule that leaves most tokens visible. Upstream already implements it as
`uniform_sqrt_t`, and the released checkpoints were trained with exactly that.

## Details that are easy to get wrong

These are the places where a plausible-looking implementation is wrong, so each
has a test:

- **Mask polarity.** Upstream's `mlm_mask` is `1` where a residue was *kept*. The
  MLM term scores `1 - mask`. Inverting this trains on the visible tokens.
- **Teacher forcing.** Section 4.3.1: the side-chain denoiser is conditioned on
  the *ground-truth* sequence during training, not its own prediction.
- **8 noise clones.** Section 4.3.1: the conditioning is cloned
  `training_batch_size_mult` (= 8 in the released config) times with an
  independent noise level each, because the MLP is cheap next to the encoder.
- **Local frame.** The diffusion target is side chains in the per-residue
  backbone frame (AF2 Algorithm 21), not global coordinates.
- **psCE bin edges.** The shipped head builds *lower edges* as
  `linspace(0, 4, 33)` and takes centres at `edge + step/2`. Training labels must
  therefore `floor(error / 0.125)`; rounding to the nearest centre puts every
  label half a bin low.
- **Confidence stop gradient.** Appendix D.4: the confidence loss must not reach
  the main model. Verified by asserting encoder gradients are bit-identical with
  and without the term.
- **The side-chain target set.** `L_diff` is scored only where the interpolant
  *hid* the side chain: `m_target = (1 - scn_mlm_mask) · seq_mask`. The encoder
  receives visible side chains as input — `encoder_inputs` gates atom37's
  non-backbone slots by `scn_mlm_mask` — so supervising those residues asks the
  denoiser to reproduce coordinates it was just shown. The paper's objective is
  `p(Y_M | Y_M̄)`, and the deployment regime agrees: `sidechain_pack` hides every
  side chain, so the hidden set is the only one the module is ever used on.
  `test_the_encoder_input_and_the_loss_target_never_overlap` pins it.
  `supervise_visible_sidechains=True` restores the unrestricted set as an
  ablation, and `scn_mlm_mask=None` (packing, coupling) supervises everything
  because nothing was visible.

## Two upstream gaps this had to work around

1. **`SidechainDiffusionModule.sidechain_diffusion` ignores `is_sampling`.** It
   only implements the 50-step sampling integrator. Training drives
   `SidechainMLP` directly instead; the integrator is still used, as shipped, for
   the confidence rollout, which is the one place it is the right path.
2. **`MAR` is a plain class, not an `nn.Module`,** yet is written like one — it has
   `forward`, calls `super().__init__()`, and reads `self.training`. So it is not
   callable and never receives `model.train()`. `pxf/train/step.py` calls
   `forward` directly and mirrors the mode across. Inference never touches
   `model.interpolant`, which is why this went unnoticed.

## What the paper does not specify

**Optimizer, learning rate, schedule, weight decay, gradient clipping.** None
appear anywhere in the paper. The defaults here (AdamW, lr 1e-4, 1000-step
warmup, constant, grad-norm clip 1.0) are chosen for *continuing* training from
the released weights, and every checkpoint records them under `optim_settings`
with `source: "not specified in the preprint; chosen for fine-tuning"` so a run
is never ambiguous about what it used.

**Post-hoc EMA.** Appendix B.2 used Karras et al. (2024) post-hoc EMA, choosing
the length after training — 1% for the 0.3 Å model, 25% for the 0.0 Å one.
`pxf/train/ema.py` offers a `relative_length` EMA that realizes "length = 25% of
training" directly, plus periodic snapshots (`snapshot_every`) that are the
prerequisite for true post-hoc reconstruction. The reconstruction itself
(Karras's σ_rel → γ mapping) is **not** implemented.

**Noise placement.** Appendix B.1 says noise is added to "protein structure
examples", which reads as the whole example — so the diffusion target is noised
too. That is consistent with the 0.3 Å model packing worse than the 0.0 Å one
(CASP14 RMSD 0.821 vs 0.745, Table 6). `noise_targets: false` keeps targets clean
if you prefer the other reading.

**The reduction in `L_diff`.** With no training loop released, whether the
per-atom squared error was averaged per atom or per residue cannot be recovered
from the code, and the two are different objectives:

| `sidechain_reduction` | Formula | What it weights |
|---|---|---|
| `per_residue` (default) | `L_i = Σ_a m_ia d²_ia / Σ_a m_ia`, then `L = (1/N) Σ_i w(t) L_i` | every residue once |
| `per_atom` | `L = Σ_ia m_ia w(t) d²_ia / Σ_ia m_ia` | Trp > Phe > Leu > Ala, by atom count |

`per_residue` is the default because the quantity of interest is per-residue
packing quality, which is what every downstream side-chain metric reports; under
`per_atom` a Trp (14 supervised slots) contributes seven times a Ser (2) and the
gradient is dominated by large side chains. Residues with nothing to score —
glycine, an unresolved side chain, a target the interpolant masked out — are
excluded rather than averaged in as zeros, so the loss does not shrink in
proportion to how many glycines a crop happens to contain. The active choice is
returned in the stats and recorded in every checkpoint under
`settings.sidechain_reduction`; `sidechain_mse_local` stays per-atom and
unweighted so curves remain comparable across both.

## Presets

From Appendix B.2, in `configs/`:

| | CATH | PDB |
|---|---|---|
| fixed example size | 256 | 1024 |
| batch | 64 | 8/GPU × 4 GPUs |
| grad accumulation | 1 | 4 (effective batch 128) |
| steps | 100k (~6–8 h, 1×H100) | 300k (~3 days, 4×H100) |
| EMA length | 25% (0.0 Å model) | 1% (0.3 Å model) |
| structural noise | 0.0 or 0.3 Å | 0.0 or 0.3 Å |

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

`L_SC` is the **diffusion term alone** — not `L_MLM + L_diff + L_conf`. Phase 1
asks whether PXDesign's `a_token` improves side-chain packing, and the sequence
is held fixed, so an MLM term would score a prediction of something already
given and a change in the loss could no longer be read as a change in packing.
`test_l_sc_equals_the_diffusion_term_alone` and
`test_l_sc_carries_no_mlm_or_confidence_term` pin it.

## Verified

- **Overfit**: two CASP14 crops, 400 steps — `loss_main` 0.880 → 0.005,
  masked-token accuracy 0.86 → 1.00, both terms falling. Kept as a test.
- **Round trip**: the resulting checkpoint packs the memorized structure at
  **0.116 Å** side-chain RMSD (vs 1.11 Å before training) with χ-20 0.995,
  scored by `scripts/eval_monomer_sidechain.py`.
- **Resume**: step counter, optimizer moments and RNG restored; loss continues
  its trajectory.
- **Stop gradient**: encoder gradients bit-identical with and without `L_conf`.

Not verified: a full-scale 100k/300k run, or that these settings reproduce the
paper's reported numbers. The optimizer is our choice, so reproduction is not
expected without tuning.
