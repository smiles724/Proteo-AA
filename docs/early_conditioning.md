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
controlled version of "the site was the problem".

What its null result would and would not license: it would rule out *the site
alone* as the explanation, holding the representation, the budget and the data
fixed. It would **not** establish that the representation is the bottleneck —
2,000 steps, the frozen donors' capacity, the 512-event pool and the σ window
are all still live alternatives, and the late pilot already showed that two of
its own arms were not converged at this budget. E2 tests one of those
alternatives; it does not test them all.

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

`verdict()` enforces all of that, and returns one of three outcomes rather than
a boolean. **`incomplete`** is not `do not proceed`: a run that never produced
the evidence has neither shown nor disproved anything, and collapsing the two is
how a missing control becomes a pass. It is returned when the candidate is
ambiguous, when no same-architecture control ran, when a required paired
interval is absent, or when `--no-sidechains` forfeited the chemistry half.
Every condition is binding — including the side-chain tolerances
(`SC_REGRESSION_LIMITS`), which were previously printed underneath the verdict
without affecting it.

One caveat the verdict cannot enforce: side-chain metrics are scored after
transferring predictions onto native residue frames, which measures local
packing accuracy well but says less about attachment geometry and environmental
clashes on the arm's *own* backbone. `lddt_sc_env` and `bad_bond_fraction` are
computed on the predicted structure and partly cover this; a dedicated
per-backbone clash measurement does not exist yet and is not claimed.

Three comparisons per experiment:

* `early_s_full` vs `early_s_bb_only` — the SC-specificity test for E1;
* `atom_sz_full` vs `atom_sz_bb_only` — the same test for E2;
* `atom_sz_full` vs `atom_s_full` — whether the pair term earns its cost.

Plus the `perturbed` arm and `refine` (one real sampler step, the
comparable-cost baseline that is not feedback at all).

The `perturbed` arm **re-encodes**. Substituting perturbed coordinates into the
existing packed state leaves `h_packed` — FaMPNN's node readout for the
*original* structure — untouched, so every arm that reads it (`late_*`,
`early_s_*`) would be handed original node features alongside perturbed χ and
environment features. That is a partial intervention reported as a full one, and
it understates the response in the direction that flatters the arm. `psCE` is
carried over unchanged and the packer is not re-run, so this is explicitly a
**geometry-only** intervention: the confidences describe the packing that was
produced, not the one being read. E2 is unaffected either way, since it
recomputes everything from `coords37`.

Which arm gets perturbed is prespecified via `--candidate`, not taken as "the
first arm whose variant is `full`" — `atom_sz_full` and `atom_s_full` are both
`full`, so that made it depend on flag order.

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

These are **not** equivalent choices that a linear layer absorbs, and saying so
would have been wrong in one of the two cases:

* `q/10` is a genuine reparameterization the first `Linear` can represent — but
  initialization scale and a finite optimization budget are not invariant to it,
  so it is recorded rather than shrugged at.
* RBF bandwidth is **not** absorbable at all. It changes the nonlinear basis
  functions, so the same weights mean something else. σ = 0.8 Å matches the last
  explicit formula given and is frozen on that basis.

Either way, leaving the choice implicit would have been a silent inconsistency
between arms; the checkpoint compatibility check above is what makes it a loud
one.

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
(`CheckpointError: a different number of tensors was saved`).

"The backbone is frozen, so the memory is not needed" would be the wrong reason.
Frozen *weights* do not make *activations* free: the late adapter's gradient
reached three decoder blocks, while `Δz` enters `z_pair`, which every one of the
24 transformer blocks reads as an attention bias — so the retained graph is a
different size. Measured on the longest structure in the training manifest
(`AF-A0A0F9IL42`, L = 510), peak RSS over a loaded-models baseline of 4.40 GB:

| arm | peak RSS | activations |
|---|---|---|
| `late_full` | 5.27 GB | +0.87 |
| `early_s_full` | 5.85 GB | +1.45 |
| `atom_s_full` | 5.89 GB | +1.49 |
| `atom_sz_full` | 6.90 GB | **+2.50** |

So the early site costs ~1.7× the late one and the pair branch ~2.9×, which is
comfortable against an H200 and a 128 GB host allocation but is a measurement
rather than an assumption. These are CPU numbers, so they include the
interpreter and the weights and are an upper bound on device allocation; the
ordering is the transferable part.

## Checkpoints refuse the wrong architecture

`load_state_dict(strict=True)` cannot catch the mistake that matters here: E1's
`full` and `bb_only` have **identical parameter shapes by construction** — that
is the point of the controls — so the wrong one loads cleanly and every later
report labels it as the arm it is not. Each checkpoint therefore carries an
architecture identity (`kind`, `version`, `arch`, `variant`, `pair`, `c_s`,
`c_z`), and the evaluator rebuilds each arm as the architecture *its own
checkpoint* records rather than as the one the config names.

Which of the two checks applies depends on where the comparison has a second
source of truth, and this is worth stating because getting it wrong yields a
check that cannot fire:

* **In the trainer** (`resume`, `initialize_from`) the module is built from the
  run's *config* and then meets a checkpoint. Those are independent, so
  `check_compatible` is a real comparison.
* **In the evaluator** the module is built *from* the recorded identity, so
  comparing the two can only ever agree. The metadata is the sole record of
  which arm produced a given set of weights — there is nothing in the file to
  cross-examine it against. What *is* checkable is the caller's claim:
  `--checkpoint early_s_full=<the control's file>` is a command-line slip that
  otherwise produces a completely self-consistent run with the wrong names on
  the results table. `check_is_the_expected_arm` holds each file to its label,
  and refuses any record whose `(arch, variant, pair)` triple is not a row of
  `ARMS`.

There *is* a second independent source of truth, though, and it is not the
weights: it is **the feature construction the installed code implements**. A
checkpoint records what it was trained against, so the two can be compared, and
the settings split in two:

* **Reconstructed** — `max_neighbours`, `neighbour_radius`, `max_atom_pairs`,
  `atom_pair_radius`, `d_hidden`, `d_noise`, the readout's `sequence` width.
  These are constructor arguments, so a loaded module is rebuilt with the
  checkpoint's values and is the function that was trained. An override of
  today's default is logged, not silently applied.
* **Refused** — `rbf_sigma`, `rbf_centres`, `coordinate_scale`, `pair_types` and
  every layer width. These are not constructor arguments, so a checkpoint
  recording different ones came from code this build does not implement; the
  shapes still match and the weights would load, describing a different function
  of a different input.

`zero_initialized` and `parameters` are excluded: both are properties of a
particular set of weights, and a trained checkpoint necessarily disagrees with a
fresh module on the first.

Separately, `check_arms_comparable` refuses a *set* of arms trained under
different reconstructed settings. Each can load correctly and the set still not
be an experiment — 16 neighbours against 32 differ in receptive field and
capacity, so the delta between them is not attributable to the information, and
nothing in either checkpoint alone can see it.

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

# One architecture per evaluation run. CANDIDATE names the arm the verdict is
# for and the arm the perturbed control perturbs; it is required for E2, where
# atom_sz_full and atom_s_full are both variant='full'.
R=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond
CONFIG=configs/couple_early_e1.yaml TAG=e1_exit CANDIDATE=early_s_full \
ARMS="early_s_full=$R/early_s_full_<job>/checkpoints/final.pt \
      early_s_bb_only=$R/early_s_bb_only_<job>/checkpoints/final.pt \
      early_s_generic=$R/early_s_generic_<job>/checkpoints/final.pt" \
sbatch scripts/slurm/eval_sb_feedback.sh
```

The arm labels are the names in `ARMS`, and each checkpoint is held to the label
it is given — pointing `early_s_full=` at the control's file is refused rather
than reported under the wrong name.

**E2 runs after E1, and the dependency is explicit.** `submit_early_conditioner.sh`
gates E2 on E1's job ids, so the order is real. It is *not* review-gated, and no
Slurm dependency can be: if E1 reads out badly, `scancel` the E2 jobs.

Report both weightings. EMA at `relative_length` 0.25 is the **prespecified**
evaluation policy — fixed before these runs, the same one the late pilot used —
so a gain measured under it is legitimate evidence, not a weaker kind of result.
`--no-ema` re-scores the same checkpoints raw as a *sensitivity check*: it says
how much of the effect depends on the weight-averaging choice, which is worth
knowing because the late pilot's two feature-reading arms changed sign at high σ
under it while its σ-only control did not. A candidate that survives only under
EMA has still met the criterion; it has also shown it is not converged, which
belongs in the writeup rather than in the verdict.

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
