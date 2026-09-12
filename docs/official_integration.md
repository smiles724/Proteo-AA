# Official PXDesign / FAMPNN integration

Integration branch: `feat/official-pxdesign-fampnn`, in
`/hai/users/y/f/yfsun/Proteo-AA-official-pxdesign-fampnn`.
Based on FAMPNN `17120fa`, with pinned Stage III `d15de33` merged in `f92a215`.
The newer Stage III checkout commits are intentionally excluded.

The implementation and small engineering gates pass. Adaptation quality is **not
established**, so PXDesign and FAMPNN remain frozen. The pilot does not justify
pretrained-network updates or production design claims.

## Sources and component loading

Initialize the recorded submodules, then apply the compatibility patch:

```bash
git submodule update --init
git -C PXDesign apply ../patches/pxdesign-embedders-protenix-2.0.patch
```

PXDesign is pinned to official `f788441313c84c3074fe9596ac2433f96b15c763`;
Stage III's gitlink `2202ad0` was not fetchable from the official remote.
Protenix is `c3bfc365b3e1341a11935eddfe7bfdc308092147`.
The modified PXDesign worktree is intentional: the official revision plus the
checked-in compatibility patch is the reproducible source, rather than a private
submodule commit. Component composition checks the imported revisions and patch.
FAMPNN source revision: `aaf788b1502ad95d5c5a84455cfc53f2544f3b45`.

| Component | Validated checkpoint | SHA256 |
| --- | --- | --- |
| Official PXDesign v0.1.0 | `runs/component_donors/pxdesign_v0.1.0.pt` | `b075867bae942dc0c6487173736922b0e2913308c1ba542d227418b6e176478d` |
| SC donor | `/hai/scratch/yfsun/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt` | `60b0bd236c49f9d9d86c9f394886e6b104d84fbd68a40fa56d091c46b10a3c50` |
| FAMPNN | `/hai/users/y/f/yfsun/Protein Project/fampnn/weights/fampnn_0_3.pt` | `8969b3f1f3c941178076c7800952595a18b56fd3828d15bb993d3ef537938a05` |

The shared official copy was inaccessible. The local copy was downloaded from
`https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt`.
SC layout: `edm=false`, `centre_coord_input=true`, `frame_aware_head=false`,
`template_residual=false`, `a_bs_concat=true`, `q_bs=false`, `bb_context=true`,
`type_logits_input=true`. Atom names and CA-based frames remain the donor's.

`pxdesign_train/checkpoints.py` loads official `design_condition_embedder.*` and
`diffusion_module.*`, and SC donor `sidechain_module.*` only. All requested keys
and shapes must match before either donor is written. Prefix normalization uses
the unwrapped model. FAMPNN is strictly loaded through `FaMPNNHead`; its SC
generator and the legacy AA MLP are absent. Feedback initializes separately,
including zero residual output projections. Unknown donor source revisions are
recorded as unknown rather than inferred from a checkpoint filename.

## Runtime and training

`stage4.generate()` uses native PXDesign sampling by default. The shared cycle
supports zero revisions, sequence-only generation, independent packing and
backbone-refinement switches, and separate SC-to-AA and SC-to-BB visibility.
Legacy MLP and `cogenerate(..., sampler_mode="minimal_euler")` remain selectable;
legacy `cogenerate` retains its historical sampler default, so official callers
must select `sampler_mode="pxdesign_native"` explicitly or use `stage4.generate`.

Native sampling uses joint target/binder denoising, then freezes the **generated**
target in its returned frame. Fixed-target adaptation explicitly selects
`initial_target_policy=fixed_context` and experimental `minimal_euler` sampling.
The original target is never silently substituted into a generated complex.

Backbone, query, sequence, packing and feature streams are separated. The native
observer records the coordinates and sigma of its last denoiser call. Packing
uses a separately defined feature pass at conditioning sigma 0.4 on the current
backbone; that pass discards its coordinate prediction. Adaptation uses the same
feature convention. Refinement sigma is not treated as physical corruption sigma.

Generated AA/SC paths hide native supervision, query identities and query SC atoms
before FAMPNN encoding. Generated atom inventories follow committed identities.
The native-type SC auxiliary branch is detached from generated state; final
packing runs on the final sequence and backbone. Ligand/metal context remains in
the structural/physical path; FAMPNN's protein Atom37 input is not a general
ligand interface. Multiple diffusion samples are rejected by the integrated
binding; use one sample and gradient accumulation.

Phases are `baseline`, `sc_adapt`, `feedback_adapt`, `aa_adapt`, and `joint_adapt`.
The migration defaults to SC-only adaptation with both pretrained networks in
evaluation mode. Frozen parameters still permit gradients through input
coordinates. Discrete sequence choices and atom inventories are not differentiable.
Feedback and backbone parameter subsets are explicit; backbone tuning remains
gated by experimental evidence.

Denoising applies EDM weights to per-sample FP32 MSE before averaging. LDDT is
gated before sample reduction. Unweighted structural metrics are reported
separately. Paired refinement has its own loss weight. The native-type SC auxiliary,
generated physical, refinement and denoising terms remain distinct.

## Loading, phase transition and resume

Fresh training composes `--backbone-checkpoint`, `--fampnn-checkpoint` and
`--sidechain-checkpoint`. `--warm-start-checkpoint` preserves a saved integrated
architecture and weights while starting fresh optimizers and counters.
`--resume-checkpoint` restores the saved phase, model, optimizer, scheduler,
counters and RNG. These modes are mutually exclusive. Evaluation reconstructs
from the integrated record and explicitly selects raw or EMA weights.

Checkpoints include effective architecture, donor identities, AA mapping,
SC layout, cycle/policy settings, trainable names, implementation digest, optimizer
and scheduler state, RNG, and EMA when enabled. The main entry point also records
its resolved data/CLI arguments. Old smoke checkpoints predate that last field.
EMA initializes after loading component or warm-start weights.

Example HAI SC adaptation (the wrapper supplies the validated donors):

```bash
bash scripts/training/slurm_stage4_fampnn_binder_hai.sh --dry-run
sbatch scripts/training/slurm_stage4_fampnn_binder_hai.sh
```

Example **engineering** feedback phase transition; do not interpret this as a
recommendation to advance a failed quality gate:

```bash
WARM_START_CHECKPOINT=/path/to/integrated_sc.pt \
STAGE4_PHASE=feedback_adapt TRAIN_ROUNDS=2 INFERENCE_ROUNDS=2 \
sbatch scripts/training/slurm_stage4_fampnn_binder_hai.sh \
  --backbone-refinement-enabled --stage4-sc-to-bb --stage4-sc-to-aa --weight-refine 1
```

For resume, set `RESUME_CHECKPOINT` instead. Preflight and training receive the
same ordered arguments, so CLI overrides win over wrapper/environment defaults.
Pinned submodules are the default source paths; `PROTENIX_CODE_DIR` and
`PXDESIGN_CODE_DIR` can select matching installations. Production data selection
must remain consistent when resuming: exact worker/prefetch replay and per-rank
DDP RNG continuation have **not** been established. Model/optimizer resume parity
is established on a single GPU at optimizer boundaries.

## Evidence and release gates

| Gate | Evidence |
| --- | --- |
| Pinned semantic merge | Compilation and 610 collected tests; focused merge regressions 74 passed, 2 skipped |
| Integrated CPU regressions | Jobs 114890 and final 114895: 630 passed, including released FAMPNN logit/mapping and coordinate-gradient tests |
| GPU component, train/eval/save/resume/export smoke | Job 114891 passed on one H200; [record](validation/official_integration/gpu_smoke_114891.json) |
| Real phase-transition preflight | Job 114894 passed against the SC-adapt checkpoint |
| Adaptation pilots | Jobs 114889 and 114892; [records](validation/official_integration/pilot_114892.json); quality gate remains closed |

The GPU smoke checks exact selected donor tensors, no AA MLP, an SC-only update,
frozen pretrained tensor hashes, native generation with and without packing,
validation, self-contained evaluation reconstruction, explicit feedback warm
start, finite nonzero feedback gradients, a feedback update, feedback checkpoint
resume, and named-atom mmCIF export. Native first random inputs were identical;
maximum repeated FP32 CUDA coordinate difference was 2.288818359375e-5 Angstrom
(tolerance 1e-4). Packing retained its returned backbone exactly. The measured
feedback gradient absolute sum was 2811.92305919528.

Artifacts are under `runs/official_components_smoke/114891/`: SC checkpoint
`checkpoints/step1_smoke.pt`, feedback checkpoint
`feedback_checkpoints/step1_feedback.pt`, `baseline_packed.cif`, `generated.cif`,
and the saved strict input batch. Logs are `logs/official-components-114891.*`.
The retained JSON records carry implementation hashes because validation ran on
working-tree snapshots over merge commit `f92a215` before the changes were committed.

Both pilots use one paired native training example and two PINDER validation
clusters excluded from the smoke training clusters. Each runs 20 SC updates,
then 20 feedback-only updates. PXDesign and FAMPNN hashes remain unchanged.
Every arm reuses the same denoising-derived backbone, target frame, query masks,
decoding budget and round count. Free-generated binders are never assigned
unrelated native training labels. The start checkpoint already has one SC smoke
update, so this is not a pristine donor-quality evaluation.

Seeded pilot 114892, after feedback adaptation (paired backbone MSE, lower is better):

| Arm | Held-out example 0 | Held-out example 1 |
| --- | ---: | ---: |
| No revision/refinement | 2.925470 | 0.231526 |
| Revisions/refinement; neither feedback route | 3.311018 | 0.351110 |
| SC-to-AA only | 3.311019 | 0.351110 |
| SC-to-BB only | 3.309847 | 0.350682 |
| Both | 3.310681 | 0.350301 |

SC-only adaptation changed mean held-out SC auxiliary loss from 4.291085 to
4.400195 in the seeded run (worse). Feedback slightly improved refinement MSE
relative to the matched no-feedback refinement arm, but no refinement remained
better on both examples; SC-to-AA also increased revision AA loss. An earlier
exploratory pilot had mixed feedback effects. Its first feedback checkpoint
recorded the wrong refinement loss weight; the retained JSON annotates that
issue, and the corrected seeded run/checkpoints are authoritative. These small,
unstable results do not establish useful feedback or packer generalization.

Full AlphaProteo-10 free-generation/designability, independent ProteinMPNN/AF2
assessment, production-scale adaptation and multi-GPU validation remain open.
The tools for both legacy AlphaProteo and FAMPNN/co-design evaluation are retained.
The summary parser now distinguishes invalid/missing AF2 scores from scored
failures and populates scalar/list-form means. Source-specific cluster filtering
is preserved; cross-source homology exclusion requires shared clustering.

## Review sequence

Eight commits group the implementation by dependency:

1. `f92a215`: pinned semantic merge.
2. `2c960f3`: selective component loading and self-contained architecture.
3. `39a839b`: native sampling, shared state, target policy, packing and refinement.
4. `fe17b15`: denoising reductions and zero feedback output.
5. `d4ffb72`: frozen training groups, phase transitions and resume.
6. `88d9a9b`: focused component/runtime/loss regression contracts.
7. `3597bcc`: launchers, evaluators, parsing and real GPU smoke.
8. This pilot/report commit: bounded SC-then-feedback experiments and gate evidence.

## Subsequent backbone geometry check

[2026-09-12 investigation](official_backbone_metrics_2026-09-12.md): 400-step native
sampling yielded 0/1086 bad CA pairs in 12 monomers and 0/470 in 11 unique binders.
Twenty-step sampling was inadequate (88.12% and 91.49%). Integrated generation
and co-design evaluation now default to 400 steps. This changes inference
budgets, not the frozen backbone weights or SC-only training phase.
