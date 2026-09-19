# E1 and E2: moving the SC → BB correction into the conditioning

The late pilot's answer was specific, and it is worth restating precisely
because these two experiments are built on it. From
[`sb_pilot_results.md`](sb_pilot_results.md): a 60° rotamer flip moves the
representation `A_SB` reads by **29%**, moves the trained adapter's output by
**1.1%**, and moves the backbone by **0.0000 Å**. The adapter converged to
approximately the σ-conditioned bias that its own `generic` control fits
directly. So the optimizer found no use for side-chain information *at that
injection site* — which is a claim about the site, not about the information.

The site was after `layernorm_a` and before `atom_attention_decoder`. Everything
that could use the correction — the atom encoder, all 24 transformer blocks —
had already run. Three decoder blocks were left to re-mix the final token
features, and a correction there can only change how they are re-mixed.

These two experiments move the correction to the **conditioning output**:

```
s_single, z_pair = diffusion_conditioning(...)     <- inject here
s_single = s_single + Δs                              [1, L, c_s]
z_pair   = z_pair   + Δz                              [L, L, c_z]
...
a_token  = atom_attention_encoder(..., z=z_pair, ...)
a_token  = a_token + linear_no_bias_s(layernorm_s(s_single))
a_token  = diffusion_transformer(a=a_token, s=s_single, z=z_pair, ...)
a_token  = layernorm_a(a_token)                    <- the OLD site is here
r_update = atom_attention_decoder(a=a_token, ...)
```

The pretrained `DiffusionConditioning` runs unmodified; its output is added to,
never replaced. `s_trunk`, `z_trunk`, `ref_pos`, the noisy coordinates and the
final `a_token` are untouched.

**`c_s` is 384 and `c_z` is 128; `c_token` is 768.** None of the three is
interchangeable, and the widths are read off the loaded model
(`conditioning_widths`) rather than configured, so a conditioner built against
the wrong one is a shape error at the hook rather than a silent broadcast.

## The two experiments

**E1 — the site alone.** `EarlySingleConditioner` reuses
`pxf.couple.readout.FeedbackReadout` exactly as it stands: same feature groups
in the same order, same LayerNorm, same sequence embedding, same
`full`/`bb_only`/`generic` controls. Only the destination changes. This is the
controlled version of "the site was the problem", and its null result would
mean the representation, not the site, is what the late pilot was limited by.

**E2 — the site and the representation.** `AtomConditioner` encodes the
predicted atoms directly: a learned atom37-slot embedding, residue-local
coordinates, the packer's psCE and a confidence-present indicator, pooled per
residue into `U_i`; and, over a directed 16-nearest-neighbour graph, atom-pair
features pooled into `V_ij`. Two heads produce `Δs` and `Δz`.

E2 deliberately does **not** read `h_packed` or E1's node group. E1 already
tests FaMPNN's node summary at this site; mixing the two would make "what did
the predicted geometry contribute" unanswerable.

The pair branch is the capability the late site did not have at all. `z_pair`
is read by the atom encoder and by every transformer block as an attention
bias, so `V_ij` can assert that *these two residues* interact — which is what a
side chain bridging them is evidence of. `V_ij` is directed and is not
symmetrized: its direction feature is expressed in residue *i*'s frame, so
`V_ji` is a different quantity rather than a redundant copy.

## The arms

`pxf.couple.conditioning.ARMS` names each row; `--sb-arm` selects one. Every
arm freezes PXDesign, FaMPNN and `A_BS`, keeps the sequence fixed, uses 50
packing steps and the bypass BB→SC policy, and trains only the named module.

| arm | reads | injects | trains |
|---|---|---|---|
| `late_full` / `late_bb_only` / `late_generic` | existing readout | `a_token` | readout + late adapter |
| `early_s_full` | existing readout, all groups | `s_single` | readout + single head |
| `early_s_bb_only` | `h_base` and sequence, SC groups zeroed | `s_single` | readout + single head |
| `early_s_generic` | σ only | `s_single` | single head |
| `atom_sz_full` | predicted SC atoms, χ, reliability, sequence | `s_single` + `z_pair` | atom/residue/pair encoders + both heads |
| `atom_sz_bb_only` | predicted BB atoms and sequence | `s_single` + `z_pair` | same architecture |
| `atom_s_full` | as `atom_sz_full` | `s_single` only | encoders + single head |

**The old late `A_SB` is never applied alongside an early arm, and there is no
flag to forget.** The payload's *type* selects the site: a
`ConditioningFeedback` is taken by the conditioning hook and declined by the
decoder hook.

## What the comparison is

Not "does it beat `bb0`". The late pilot already cleared that bar — `full` beat
the proposal by +0.0007 Å with an interval excluding zero — and it meant
nothing, because `bb_only` beat it by +0.0008 Å for a third of the cost.

The claim requires beating the **same-architecture BB-only control** by
≥ max(0.05 Å, 3% of baseline), paired, with no material side-chain or chemistry
regression. Explicitly *not* evidence: a larger residual, stronger perturbation
sensitivity, or any improvement over `bb0` alone.

Three comparisons per experiment:

* `early_s_full` vs `early_s_bb_only` — the SC-specificity test for E1;
* `atom_sz_full` vs `atom_sz_bb_only` — the same test for E2;
* `atom_sz_full` vs `atom_s_full` — whether the pair term earns its cost.

Plus the `perturbed` arm (the candidate reading a rotamer-perturbed packing,
backbone and sequence fixed) and `refine` (one real sampler step, the
comparable-cost baseline that is not feedback at all).

## Why the results are comparable to the late pilot's

The training pool is **not rebuilt**. `pilot_examples` derives the 512
`(structure, σ, seed)` triples deterministically from the manifest, the seed,
`--pool-size` and `--sigmas-per-structure`, so the launcher's defaults
reproduce exactly the 128-structure/512-event pool the late arms used — and the
runs load that run's own upstream cache
(`pxf_sb_pilot/upstream_cache_v2_crop512_pool512.pt`, fingerprint
`aef7838b5c77f925a68df1a4fcf31e39`, 512 states over 128 structures). The frozen
half does not depend on which arm reads it, `UpstreamCache.compatible` refuses a
mismatch rather than warning, and a cache hit means the new arms saw
bit-identical `bb0` and `sc0`.

Everything else is held at the late pilot's values: σ ∈ [0.1, 2.0] Å drawn from
the trajectory (steps 305–363 of 400), lr 1e-4, clip 1.0, 2,000 steps, seed 0,
`SigmaWindow(0.1, 2.0, taper=2.0)`, EMA `relative_length` 0.25, evaluation at
steps 0/500/1000/2000, and the same 64-target × 4-σ held-out AFDB panel.

## Frozen constants, and the two that were specified twice

Widths and neighbourhood limits are constants in `pxf.couple.conditioning` and
recorded in every checkpoint's architecture identity. They are pilot defaults,
not findings; the point of freezing them is that two arms cannot quietly differ.

| | |
|---|---|
| atom37-slot embedding / atom embedding | 16 / 64 |
| residue embedding `U_i` | 128 (from 178 = 64 + 64 + 32 + 12 + 6) |
| edge embedding / pair embedding `V_ij` | 64 / 64 (edge input 58) |
| head hidden / σ embedding | 256 / 64 |
| residue neighbours | ≤ 16 within 20 Å, from CA, directed, self excluded |
| atom pairs per residue pair | ≤ 32 within 12 Å, ≥ 1 side-chain atom |
| distance RBF | 16 centres on [0, 12] Å |

Two quantities were specified twice with different values, and the
disagreement is resolved once, in the module, rather than per call site:

* **RBF width** — `exp[-((d-c)/0.8)²]` and `exp[-(d-c)²/2(0.8)²]` differ by √2 in
  σ. Frozen as the standard Gaussian with σ = 0.8 Å (the second form).
* **Local coordinate scale** — "`q/10`" and "in Å". Frozen as `q/10`.

Both alternatives differ only by a scale the first linear layer can absorb,
which is exactly why leaving the choice implicit would have been a silent
inconsistency between arms rather than a visible one.

Ties are broken deterministically by index (stable sort) in both the residue
graph and the atom-pair selection, so two runs on one structure select the same
edges. Selection runs under `no_grad` on detached geometry — which residues are
neighbours is a choice, not a differentiable quantity. Edge coverage and
truncation counts are logged (`edges`, `edges_truncated`, `atom_pairs`,
`atom_pairs_truncated`).

## Initialization and gradients

Every final projection — `Linear(256, c_s)` and `Linear(256, c_z)` — is
zero-initialized; everything before it is initialized normally. The σ gate is a
*fixed* function equal to 1 across the training window, not learned: the product
of two zero-initialized factors has no gradient in either, and the adapter would
never leave the origin.

So at step 0 the coupled system reproduces PXDesign exactly, and the encoders
have **no gradient until the output projection has moved**. That is the expected
state, not a broken graph, and
`tests/test_couple_conditioning.py::test_the_output_projection_moves_before_the_encoders_do`
checks the projection first and the encoders second for that reason. The one
exception is `early_s_generic`, whose readout groups are all `zeros_like` and
therefore disconnected from the graph — that is what "σ only" means, and an
encoder gradient there would mean the control is reading something.

Activation checkpointing stays off. Protenix recomputes the forward during
backward and the injection hooks make the recomputation diverge
(`CheckpointError: a different number of tensors was saved`). The conditioners
are small and the backbone is frozen, so the memory is not needed.

## Checkpoints refuse the wrong architecture

`load_state_dict(strict=True)` cannot catch the mistake that matters here: E1's
`full` and `bb_only` have **identical parameter shapes by construction** — that
is the point of the controls — so the wrong one loads cleanly and every later
report labels it as the arm it is not. Each checkpoint therefore carries an
architecture identity (`kind`, `version`, `arch`, `variant`, `pair`, `c_s`,
`c_z`), `check_compatible` refuses a mismatch, and the evaluator rebuilds each
arm as the architecture *its own checkpoint* records rather than as the one the
config names.

## Running it

```bash
for v in $(env | sed -n 's/^\(SLURM[A-Za-z_]*\)=.*/\1/p'); do
    [ "$v" = SLURM_CONF ] && continue; unset "$v"; done
unset CUDA_VISIBLE_DEVICES

# E1, all three arms. Run every arm or the candidate's number cannot be read.
for arm in early_s_full early_s_bb_only early_s_generic; do
    ARM=$arm sbatch scripts/slurm/train_early_conditioner.sh
done

# E2, after E1 reads out.
for arm in atom_sz_full atom_sz_bb_only atom_s_full; do
    ARM=$arm sbatch scripts/slurm/train_early_conditioner.sh
done

# One architecture per evaluation run.
R=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond
CONFIG=configs/couple_early_e1.yaml TAG=e1_exit \
ARMS="early_s_full=$R/early_s_full_<job>/checkpoints/final.pt \
      early_s_bb_only=$R/early_s_bb_only_<job>/checkpoints/final.pt \
      early_s_generic=$R/early_s_generic_<job>/checkpoints/final.pt" \
sbatch scripts/slurm/eval_sb_feedback.sh
```

Report both weightings. The late pilot's only gain robust to the EMA choice was
the σ-only control's; `--no-ema` re-scores the same checkpoints raw, and a
candidate whose advantage exists only under weight averaging has not been shown
to have learned a correction.

## What is deliberately not in these two experiments

* **Gate B.** `bs_policy` stays `bypass`. Testing the selected conditioner under
  a frozen Gate B, and then retraining it on Gate-B packings, is a later stage
  with its own regenerated cache — and the two coupling directions are never
  trained jointly.
* **Sampler integration.** Passing the criterion licenses one corrective event
  inside a rollout, and nothing beyond that.
* **A bigger budget.** 2,000 steps, extended to 5,000 only if promising. The
  late pilot's `generic` control was ≥ `full` at every evaluation point, so more
  steps there would have been extending the arm that was behind.
