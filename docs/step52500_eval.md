# `fixed_global_decay_from_50k/step52500.pt` — measured results

Checkpoint:
`/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt`
(Stage II-A side-chain warmup, `train_mode=joint`, `global_step=420000`; lineage
`protenix_monomer_backbone/bb_100k_val/step96000` → `fixed_global_from_bb_step96000/step50000` → this run.)

Recorded `sidechain_arch`: `bb_context=True, centre_coord_input=True, frame_aware_head=False,
template_residual=False, type_logits_input=True, edm=False, a_bs_concat=True, q_bs=False`
→ one-step arm, decoded once from the Dunbrack template init.

Validation set throughout: the strict monomer subset of
`recentPDB_low_homology_maxtoken1536`, **491 proteins**, 141 255 tokens,
545 277 supervised side-chain atoms, ≤640 tokens per protein.

---

## 1. Backbone

Stage II freezes the backbone (`trainable_param_keywords=["sidechain_module."]`,
116.56M of 259.24M parameters trainable), and the weights confirm it: all 636
`diffusion_module.` and 96 `design_condition_embedder.` tensors in step52500 are
**byte-identical** to Stage I `bb_100k_val/step96000`. The metrics agree to five
decimals, so Stage II moved the backbone by exactly nothing.

| Metric (491 proteins) | step52500 | Stage I step96000, re-run today | step52500, leaky input |
|---|---|---|---|
| Cα lDDT | **0.6129** | 0.6129 | 0.6728 |
| Cα RMSD (Å) | **3.762** | 3.762 | 3.037 |
| backbone RMSD (Å) | **3.721** | 3.721 | 3.008 |
| TM-score | **0.7764** | 0.7764 | 0.8224 |
| coordinate MSE (Å²) | **23.15** | 23.15 | 15.22 |
| distogram CE | 2.033 | 1.095 | 2.055 |
| smooth-lDDT loss | 0.1855 | — | 0.1695 |

Artifacts: `stage2_backbone_eval/fixed_global_decay_step52500/`,
`.../fixed_global_decay_step52500_leaky_input/`, `.../stage1_step96000_recheck/`
under `/hai/scratch/yfsun/proteo_aa_runs/`.

**Two things this table says that are easy to get wrong.**

*The older Stage I number is not comparable.* The figure and ranking everyone
quotes — `protenix_monomer_backbone/best_checkpoint_eval/92187/`
(`checkpoint_structure_metrics.png` + `checkpoint_ranking.csv`: Cα lDDT 0.7026,
Cα RMSD 2.881, TM 0.8334 at step96000) — was produced 2026-08-03, before commit
`6c7a2c7` (2026-08-06) removed the binder's native side-chain atom rows from the
model input. Re-running the same weights today with that input restored gives
0.6728 / 3.037 / 0.8224 — most of the gap, though not all of it, so other changes
in that commit contribute too. The leakage-safe column is the one to quote; see
"What number each backbone figure actually reports" at the end for which figure
carries which number (`protenix_backbone_train_val_91436.png` carries neither —
it plots losses).

*The distogram number is not a model result.* step52500's `design_distogram_head`
is randomly initialised (see §3), which is the whole of the 1.095 → 2.033
difference.

---

## 2. Side chain

Protocol: GT backbone frames + GT residue types in, side-chain atoms out, scored
with the unweighted masked MSE over supervised side-chain atoms and with
hard-threshold side-chain lDDT (`pxdesign_train/sidechain/lddt.py`). All three
rows come from the same scoring code, so they are directly comparable.

| Set | n | atoms | MSE (Å², atom-wt) | RMSD (Å, atom-wt) | lDDT sc-env | lDDT sc-sc |
|---|---|---|---|---|---|---|
| recentPDB validation | 491 | 545 277 | **1.8206** | **1.3493** | **0.8668** | **0.8265** |
| CASP14 | 26 / 34 | 19 318 | **1.9877** | **1.4099** | **0.8528** | **0.8104** |
| CASP15 | 32 / 45 | 22 137 | **2.2597** | **1.5032** | **0.8413** | **0.7935** |

Per-target / per-protein spread and the macro (per-target) conventions:

| Set | MSE mean/item | MSE median | MSE p90 | RMSD mean/item | lDDT sc-env median |
|---|---|---|---|---|---|
| validation | 1.8244 | 1.7843 | 2.3603 | — | 0.8691 |
| CASP14 | 1.9883 | — | — | 1.403 | 0.8519 |
| CASP15 | 2.3384 | — | — | 1.506 | 0.8481 |

Reference point on the same 491 proteins: the ideal-rotamer Dunbrack template
with no model at all scores **4.7573 Å² / 2.1811 Å**
(`protenix_monomer_sidechain_warmup/template_baseline_491/template_baseline.json`).

### Where step52500 sits in its own sweep

| step | MSE (Å²) | RMSD (Å) | lDDT sc-env | lDDT sc-sc |
|---|---|---|---|---|
| **52500** | **1.8206** | **1.3493** | 0.8668 | 0.8265 |
| 55000 | 1.8221 | 1.3499 | 0.8669 | 0.8266 |
| 57500 | 1.8250 | 1.3509 | 0.8676 | 0.8275 |
| 60000 | 1.8281 | 1.3521 | **0.8681** | **0.8281** |
| 62500 | 1.8267 | 1.3515 | 0.8680 | 0.8280 |
| 65000 | 1.8403 | 1.3566 | 0.8679 | 0.8278 |

The two metrics rank the sweep differently: MSE picks 52500, lDDT picks 60000.
The spread is small (0.02 Å² and 0.0013 lDDT), but the direction is the
disagreement `lddt.py` exists to expose — MSE rewards shorter lever arms, lDDT
rewards contacts that survive.

### CASP coverage caveat

8 of 34 CASP14 and 13 of 45 CASP15 targets are excluded, almost all with
`RuntimeError: Failed to parse CIF` out of Protenix's
`add_missing_atoms_and_residues` (a `KeyError` on a residue id — the domain CIFs
that `casp_natives_to_cif.py` writes carry residue numbering the reference chain
does not cover). One CASP14 target is dropped for length (T1044, 2166 > 1472) and
one for a missing `entity_type`. The excluded sets are **identical** to those of
the `fixed_global/step48000` runs, so the CASP numbers here are directly
comparable to that baseline — but they cover 76% / 71% of each edition, not all
of it. Fixing the domain-CIF numbering would recover the rest.

`--report-train-overlap` reports zero training-index overlap for both editions,
but that check is vacuous on the manifest path: it looks the CASP target name up
as a PDB id. The CASP14 contamination question (CASP14 predates the
`before_2021-09-30` training cutoff) is therefore still open.

---

## 3. AA head

### step52500's own AA head is untrained

Leakage-safe evaluation (`eval_aa_head_strict_backbone.py`, 491 proteins,
141 255 tokens, native labels never shown to the model):

| condition | σ | CE | acc | top-5 | F1 macro |
|---|---|---|---|---|---|
| full_topology_scrub | 0.04 | 3.0077 | 0.0539 | 0.2792 | 0.0399 |
| full_topology_scrub | 0.4 | 3.0006 | 0.0197 | 0.2865 | 0.0243 |
| full_topology_scrub | 4.0 | 2.9960 | 0.0549 | 0.2727 | 0.0301 |
| strict_native | 0.04 | 3.0042 | 0.0557 | 0.2850 | 0.0686 |
| strict_native | 0.4 | 2.9996 | 0.0198 | 0.2822 | 0.0233 |
| strict_native | 4.0 | 2.9958 | 0.0552 | 0.2751 | 0.0371 |
| strict_random | 0.04 | 3.0051 | 0.0604 | 0.2747 | 0.0763 |
| strict_random | 0.4 | 3.0012 | 0.0173 | 0.2757 | 0.0339 |
| strict_random | 4.0 | 2.9954 | 0.0549 | 0.2749 | 0.0368 |

Chance is CE = ln 20 = 2.9957 and accuracy 5%; the empirical majority class (LEU)
is 9.15%. Every cell is at chance, and — decisively — `strict_random`, which
destroys the geometry, scores the same as `strict_native`. The head is not
reading the structure because there is nothing to read: it is at its random
initialisation.

The mechanism is in the donor run's log. `fixed_global_from_bb_step96000`
warm-started with `checkpoint_include_prefixes = ('diffusion_module.',
'design_condition_embedder.')` — "kept 732/743 tensors … missing=419" — so
`design_residue_type_head` and `design_distogram_head` were never loaded, and
Stage II trains neither (`weight_aa=0`, trainable filter `sidechain_module.`).
Weight comparison agrees: 10/10 residue-type-head tensors match the Stage II
donor step50000 and only 2/10 match Stage I step96000.

Current `main` already fixes this for future runs — `sidechain_warmup` now lists
`design_residue_type_head.` in `checkpoint_include_prefixes` — but this
checkpoint predates that and cannot be repaired by grafting (see the comment in
`train_protenix_monomer.py`: an overlaid foreign head drove `aa_ce` to ~50 at
unchanged chance accuracy).

### The trained AA head on this lineage

`aa_head_on_stage2/from_stage2_65000` fits a head to the frozen Stage II model in
the Stage III configuration. Its donor is **step65000 of the same run**, not
step52500. Training-time validation (n=308):

| step | val aa CE | val aa acc | val sc_local (Å²) | val Cα RMSD | val TM |
|---|---|---|---|---|---|
| 1000 | 2.859 | 0.1122 | 5.794 | 3.644 | 0.7538 |
| 5000 | 2.820 | 0.1225 | 5.815 | 3.958 | 0.7338 |
| 9000 | 2.809 | 0.1270 | 5.820 | 3.759 | 0.7455 |

Same leakage-safe evaluation as above, on the full 491 proteins, at step9000:

| condition | σ | CE | acc | top-5 | F1 macro | AUROC macro |
|---|---|---|---|---|---|---|
| full_topology_scrub | 0.04 | 2.8633 | 0.1170 | 0.4187 | 0.0637 | 0.5864 |
| full_topology_scrub | 0.4 | 2.7878 | **0.1523** | 0.4769 | 0.0987 | 0.6509 |
| full_topology_scrub | 4.0 | 2.7742 | 0.1547 | 0.4884 | 0.0966 | 0.6483 |
| strict_native | 0.04 | 2.8783 | 0.1062 | 0.4065 | 0.0622 | 0.5717 |
| strict_native | 0.4 | 2.7940 | **0.1340** | 0.4689 | 0.1068 | 0.6371 |
| strict_native | 4.0 | 2.8041 | 0.1328 | 0.4648 | 0.0921 | 0.6289 |
| strict_random | 0.04 | 2.9298 | 0.0852 | 0.3673 | 0.0962 | 0.4997 |
| strict_random | 0.4 | 2.9838 | **0.0833** | 0.3542 | 0.0669 | 0.4994 |
| strict_random | 4.0 | 2.9874 | 0.0669 | 0.3257 | 0.0367 | 0.5000 |

This head does read the backbone: `strict_native` beats `strict_random` by ~5
accuracy points and AUROC 0.637 vs 0.499 (chance). But it is only ~4 points above
the 9.15% majority-class floor, and the residual topology leak is now small
(15.2% vs 13.4%) rather than the 98%-vs-7% chasm the older `joint` checkpoint
showed.

Two things worth carrying forward. The head plateaus just above the majority-class
floor. And `val_sc_local` is **5.82 Å²** here against 1.82 Å² in §2 — the same
S_φ, scored on *predicted* backbone frames instead of GT ones. The side-chain
result in §2 is a packing result, not a co-design result.

### ProteinMPNN sequence recovery, 491 proteins

| backbone fed to ProteinMPNN | mean | median | token-weighted | p10 | p90 |
|---|---|---|---|---|---|
| ground truth | **0.4559** | 0.4628 | **0.4625** | 0.3670 | 0.5363 |
| predicted, step52500 | **0.3305** | 0.3257 | **0.3338** | 0.2120 | 0.4571 |

Same 491 proteins and 141 255 aligned tokens on both rows, `length_match_fraction
= 1.0`, 1 sequence per target at T = 0.1.

The GT-backbone row is protocol-stable (no model forward pass). The
predicted-backbone row is **not** comparable to the July Stage I MPNN sweep
(step100000: 0.3581 mean / 0.3618 token-weighted): those exports predate the
2026-08-06 leakage fix, exactly as in §1. Since step52500's backbone is
byte-identical to Stage I step96000, 0.3305 is also the honest current number for
Stage I step96000.

---

---

## Two things found while measuring, neither fixed here

**`eval_aa_head_strict_backbone.py` was building the wrong model for an
`aa_head_on_stage2` checkpoint.** The S→B feedback channels come from
`--sc-ablation-arm`, which is not recorded in the checkpoint; that script's arm
default is `no`, correct for `joint` (side chain off entirely) and wrong here —
it dropped `a_token_fusion_pre` and `q_atom_fusion` from the forward pass while
their weights sat in `unexpected=16`. Now reconciled from the checkpoint's own
module set (`adopt_feedback_channels_from_checkpoint`), giving
`missing=0, unexpected=0`. For *this* checkpoint the correction changes nothing —
both fusions' output projections (`mlp.2.weight/bias`) are exactly zero, i.e.
still at their zero init, because `aa_head_on_stage2` freezes everything but
`design_residue_type_head.`. The §3 table is identical to four decimals either
way. The guard matters for any checkpoint where those channels have trained.

**`HResInjector` is not zero-initialised, unlike `ATokenFusion`/`QAtomFusion`.**
In the step9000 checkpoint `hres_injector.proj.1` carries ordinary random weights
(absmax 0.036) and is frozen, so the refinement pass adds a random projection of
`h_res'` to `s_trunk`. It does not affect the numbers above (`weight_aa_post = 0`,
and the reported `aa_logits` come from the primary pass), but it means Stage III
warm-starting from a Stage II checkpoint does **not** begin as a clean no-op the
way the a/q channels deliberately do — worth a look before reading early Stage III
curves. The currently running Stage III job (101203) is on this path.

---

## Figures

Nothing plots step52500 across these three benchmark sets yet. Existing, related:

| Figure | What it shows |
|---|---|
| `runs/figures/edm_vs_onestep.png` | all one-step and EDM arms on one MSE axis; the decay arm is not in it |
| `runs/figures/val_comparison_four_arms.png` | four one-step arms' validation curves, EDM held on its own axis |
| `runs/figures/sidechain_arms_local_vs_global.png` | local vs global side-chain arms |
| `runs/figures/protenix_backbone_train_val_91436.png` | Stage I backbone **loss** curves — see below, it plots none of §1's metrics |
| `protenix_monomer_backbone/best_checkpoint_eval/92187/checkpoint_structure_metrics.png` | Stage I Cα lDDT / TM / Cα RMSD over 50 checkpoints — the source of the "0.70" |
| `runs/figures/aa_head_eval_bb_100k_val_summary.png` | AA-head metrics over Stage I checkpoints |
| `reports/aa_head_inference_style_validation_report.png` | the AA-head leakage investigation |
| `stage2_backbone_eval/*/checkpoint_structure_metrics.png` | single-point backbone plots written by the runs in §1 |
| `eval_aa_head_strict_backbone/fixed_global_decay_step52500/strict_aa_ablation.png` | the §3 chance-level ablation |

The `arm_sweep_decay` sweep (§2 table) has a CSV
(`.../arm_sweep_decay/sweep_fixed_global_decay.csv`) but no figure.

### What number each backbone figure actually reports

This matters because the two figures are easy to confuse with each other and with
§1, and one of them is the origin of the "0.70 Cα lDDT" that circulates.

**`best_checkpoint_eval/92187/checkpoint_structure_metrics.png`** — three panels
(Cα lDDT, TM-score, Cα RMSD) over 50 Stage I checkpoints. It peaks at step96000,
which is exactly the backbone step52500 inherits:

| Panel | step96000 (peak) | step100000 (last point) |
|---|---|---|
| Cα lDDT | **0.7026** | 0.7003 |
| TM-score | 0.8334 | 0.8341 |
| Cα RMSD (Å) | 2.8810 | 2.8771 |

**This entire figure is on the pre-2026-08-06 leaky-input protocol** (it was
written 2026-08-03). Its 0.7026 is therefore closer in kind to §1's "leaky input"
column (0.6728) than to the leakage-safe 0.6129 — and not equal to either, since
re-running the same weights with the old input recovers most but not all of the
gap. Replotted under today's input the whole curve drops by roughly 0.09 lDDT and
step52500 sits at 0.6129 on it. **Do not quote this figure's numbers alongside
§1's.**

**`runs/figures/protenix_backbone_train_val_91436.png`** — five panels of
train/validation **loss** curves, and it reports none of the structural metrics
above. Validation values near step96000: total loss 71.19, backbone MSE 17.74,
LDDT loss 0.1868, distogram loss 1.113, low-sigma fraction 0.4668. Its "LDDT
loss" panel is the differentiable smooth-lDDT training term — **lower is better**,
and it is not Cα lDDT. This is the same confusion `pxdesign_train/sidechain/lddt.py`
documents for the side-chain stage, where the identical `val_lddt` key is computed
on the frozen backbone and says nothing about side chains at all.

Neither figure has been regenerated under the current protocol; doing so is the
cleanest way to make the plotted number and the reported number agree.
