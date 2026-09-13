# SC-only geometry, complex, and predicted-input adaptation

The phase order is now:

`sc_warmup -> sc_geometry_repair -> sc_complex_adapt -> sc_adapt`

`sc_geometry_repair` is a documented warm start from the 46,000-update EMA of
job 114967. It starts a fresh Adam optimizer, 100-update LR warm-up, scheduler,
and EMA shadow. The side-chain module alone is trainable. Backbone and FAMPNN
parameters are initialized from their donor EMA values, verified after loading,
and remain frozen. The donor checkpoint path, SHA256, step, and `weights=ema`
are recorded; this phase is not represented as an exact continuation.

Repair uses native monomers only, native BB/native residue types for the single
SC prediction, and the existing XPB-masked frozen backbone feature pass. It has
no AA decoding, backbone sampling, feedback, receptor context, or differentiable
clash call. Coordinate and geometry losses act on the same SC output. Existing
rigid augmentation and template/noise initialization are unchanged.

The four covalent terms use the committed
`sidechain/canonical_chemistry.json` registry. Their frozen calibration is in
`runs/sc_geometry_repair/calibration_v1/calibration.yaml`; its subset contains
32 training proteins. The 308 recent-PDB proteins remain validation, and 128
PDBs are held out as a repair-only final test set. This final set is separate
from repair training/validation but is not independent of the warm-up donor's
pretraining, as recorded by the manifest.

Before A/B/C selection, `evaluate_sc_geometry_repair_baseline.py` evaluates the
materialized 46k EMA on the same fixed 308-protein panel. The selector compares C
with both that donor and symmetry-only arm B at matching updates and stochastic
inputs. It requires a substantial pooled 3x-native-RMS covalent-failure reduction,
strict improvement in internal SC bonds against both baselines, bounded regression
in every other class, and preserved explicitly named pooled 20-degree chi metrics.
The selector never reads the final-test manifest.

After selection, `evaluate_sc_geometry_repair_final.py` evaluates the selected
checkpoint once on the reserved 128-protein final test and writes an immutable
`sc_geometry_repair_final_test_v1` report. `sc_complex_adapt` requires both that
report and the `sc_geometry_repair_acceptance_v1` decision, and verifies their
checkpoint, acceptance, and final-manifest hashes.

The native RMS calibration from this registry is:

| Class | count | signed mean | native RMS | 3x tolerance |
|---|---:|---:|---:|---:|
| SC bond | 18,537 | -0.00544 A | 0.03147 A | 0.09442 A |
| Attachment bond | 5,871 | 0.00330 A | 0.01386 A | 0.04157 A |
| SC angle | 17,462 | 0.01060 rad | 0.03669 rad | 0.11007 rad |
| Attachment angle | 17,963 | 0.02531 rad | 0.04801 rad | 0.14404 rad |

These are RMS deviations, not estimates of statistical standard deviation.
The older baked evaluator references agree to approximately `1e-4` at worst,
but repair metrics still use the exact committed registry.

The controlled 2,000-update comparison is an array: A uses ordinary coordinate
loss; B uses symmetry-aware whole-residue coordinate loss; C adds all four
geometry terms after a 200-update ramp. The GPU gate measures each component's
SC-parameter gradient norm on four fixed native examples and writes the chosen
weights before the array starts. Validation and checkpoints occur every 500
updates with fixed item and SC initialization seeds.

The follow-up strength sweep is frozen in
`runs/sc_geometry_repair/sweep_v2/preregistered_gate.json`. Arm D multiplies all
four calibrated weights by 1.5; arm E doubles the two bond weights and retains
the original angle weights. Both reuse the original arm B validation files as
matched controls. The 2k screen requires at least 25% aggregate improvement,
strict improvement in every geometry class, no pooled 20-degree chi regression,
and symmetry RMSD no more than 0.03 A above the 46k EMA donor. A passing arm alone
may resume to 5k. Post-extension validation applies the same gate, using matched
B through step 2000 and frozen B step 2000 later, before opening the final test.

Implementation branch: `feat/sc-adaptation-phases`, based on
`feat/sc-rigid-augmentation` at `6ec0459`. The existing shared rigid transform,
local/reference-coordinate conventions, chemical/model/observed masks,
noncanonical exclusions, invalid-frame handling and per-atom N/CA/C/O masks are
preserved. SC architecture is inherited, including `edm=false`; the launcher has
no architecture overrides.

These are proposed pilot recipes. A successful smoke test establishes execution
compatibility, not an accepted checkpoint or improved packing/design quality.
An operator must select the preceding-phase checkpoint using held-out results.

| Setting | `sc_complex_adapt` | `sc_adapt` |
|---|---|---|
| Accepted parent | Accepted `sc_geometry_repair` | `sc_complex_adapt` |
| Trainable | `sidechain_module.*` | `sidechain_module.*` |
| Source mixture | Monomer .75 / PINDER .25 | Monomer .50 / PINDER .50 |
| Coordinate mixture | Native 1.0 | Native .50 / paired reconstruction .50 |
| Main AA inputs | Native canonical types; no FAMPNN call | Frozen FAMPNN, 4 blocks, temperature 0 |
| Coordinate objective | Observed native SC | Separate native-type auxiliary packing |
| Physical coefficient | 0; separate .01 comparison | .01; separate zero-weight control |
| Backbone/AA losses | All zero | All zero |
| Feedback/revisions/refinement | Disabled | Disabled |

Both use SC LR `1e-5`, Adam `(0.9, .95)`, weight decay 0, BF16 network passes,
FP32 coordinate losses, crop 384, one backbone sample, accumulation 8, 500-update
LR warm-up, clipping 1.0, and explicit native rigid augmentation. The default
pilot is 1,000 optimizer updates; save/evaluate every 500/1,000 updates. The
provisional 5,000–15,000 budget is an explicit override after pilot review. EMA is
disabled in these recipes and validation explicitly reports raw weights.

## Launch and resume

From this checkout, create `logs/training` before `sbatch`. The ordinary phase
launchers require an explicitly selected checkpoint and do not inspect running
jobs.

Direct `sc_warmup -> sc_complex_adapt` transitions are rejected. Select an
accepted repair checkpoint after the controlled repair comparison, then launch
complex adaptation from that checkpoint.

```bash
export PROTEOAA_REPO=/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases
export SC_REPAIR_GATE_DIR=/path/to/passed/gate
sbatch scripts/training/slurm_sc_geometry_repair_arms_hai.sh
sbatch scripts/training/slurm_eval_sc_geometry_repair_baseline_hai.sh

# Submit selection after all three arms and the donor baseline complete, then
# evaluate the selected checkpoint once on the reserved final test.
export SC_REPAIR_RUN_ROOT=/path/to/arms_run
export SC_REPAIR_DONOR_BASELINE=/path/to/donor_baseline.json
sbatch scripts/training/slurm_select_sc_geometry_repair_hai.sh
sbatch scripts/training/slurm_eval_sc_geometry_repair_final_hai.sh

# A 2k checkpoint may be extended to 5k without changing its data identity.
sbatch scripts/training/slurm_extend_sc_geometry_repair_hai.sh

export ACCEPTED_CHECKPOINT=/path/to/accepted_geometry_repair.pt
export SC_REPAIR_ACCEPTANCE="$SC_REPAIR_RUN_ROOT/acceptance.json"
export SC_REPAIR_FINAL_TEST="$SC_REPAIR_RUN_ROOT/final_test/final_test.json"
bash scripts/training/slurm_sc_complex_adapt_hai.sh --dry-run
sbatch scripts/training/slurm_sc_complex_adapt_hai.sh

# Independent physical-loss comparison: same parent, seed and source mixture.
sbatch scripts/training/slurm_sc_complex_adapt_hai.sh --physical-weight .01
# Independent mixture comparison, after assessing the 75/25 pilot.
sbatch scripts/training/slurm_sc_complex_adapt_hai.sh --monomer-fraction .50

export ACCEPTED_CHECKPOINT=/path/to/accepted_sc_complex_adapt.pt
sbatch scripts/training/slurm_sc_predicted_adapt_hai.sh \
  --full-sample-validation-cache /path/to/full400/manifest.json

# Exact resume restores the recipe, optimizer, RNG and consumed sampler cursor.
export RESUME_CHECKPOINT=/path/to/integrated/step500.pt
sbatch scripts/training/slurm_sc_complex_adapt_hai.sh
```

`run_sc_adaptation.sh` is portable: set `PYTHON_BIN`, `PROTEOAA_REPO`,
`FAMPNN_ROOT`, `SC_PHASE`, `OUTPUT_DIR` and `ACCEPTED_CHECKPOINT` (or
`RESUME_CHECKPOINT`) under another scheduler. Data roots/caches, mixture fractions,
LR, accumulation, seed, budgets and intervals have explicit CLI overrides; use
`train_sc_adaptation.py --help`. Unknown overrides fail. Exact resume permits
increasing only the runtime step budget. The data identity excludes runtime
optimization and objective settings while retaining every input, partition,
sampling, and seed setting. A phase transition resets optimizer/counters and
preserves the recorded component architecture and weights. These initial scripts
support one GPU with accumulation and explicitly reject distributed sampling.

## Input and objective contracts

The new behavior is gated by `stage4.adaptation_protocol=sc_only_v1`. Historical
checkpoints retain their original generated-path forward. The new phase resolver
sets the complete destination settings and validates them before data/model
construction. Native-complex physical loss is returned unweighted by the packer
and added exactly once by the trainer. `sidechain.pack_loss` agrees with the
external coefficient. Physical diagnostics are also computed for zero-weight
controls. `sc_warmup` continues to forbid a physical objective.

Each predicted-input item carries `backbone_source` and `input_seed`. Native replay
uses the branch's shared rigid augmentation; the entire complex and its frame
origins/orientations share a transform while local SC targets and `ref_pos` stay
unchanged. Missing native backbone/context atoms remain masked. The main generated
branch never uses native SC observation masks or native SC labels to define its
inventory or attention.

Paired reconstruction adds joint binder/receptor noise at a sampled positive
sigma from `0.4,1,2,4`, retaining clean conditioning features. A frozen denoiser
produces both coordinate states under `no_grad`; its reconstructed receptor is
retained. The common feature pass uses sigma .4 on the resulting coordinates,
and discards the feature-pass coordinate prediction. SC receives the preserved
input backbone. FAMPNN decoding also runs frozen under `no_grad` because these
phases have no coordinate-dependent AA training objective.

The auxiliary branch uses native types and detached current coordinates/features.
Native local geometry transported onto reconstructed frames is a pseudo-target.
The initial per-residue eligibility thresholds are CA displacement ≤3 A and
N–CA/CA–C bond-length error ≤.3 A, plus native-frame/canonical/observed-atom gates.
These thresholds are proposed controls, not validated optima. Excluded residues
are counted; the forward does not retry or replace distorted reconstructions.
Auxiliary outputs never enter generated state. Full samples reject native labels
at the source boundary and omit auxiliary loss and native AA metrics.

## Data and replay

Only whole monomers ≤384 and prepared PINDER holo dimers are used. The existing
binder-fraction limit (.75) and maximum pre-crop complex size (640) are recorded.
The complete binder is retained and receptor cropping uses the native interface.
PINDER uses inverse-cluster weights after filtering, including held-out-cluster
exclusion. Monomer sampling records inverse-cluster or uniform behavior according
to available metadata. Mixtures stay constant throughout each run.

The preparation pass excludes held-out PDB identities across the two sources
before reconstruction or sampling. Shared sequence/UniProt identifiers are checked
when both manifests contain them. `data_audit.json` records missing cross-source
sequence metadata, filtering counts, sampling policies and content fingerprints.
This is not a shared homology-clustering audit, and it does not establish donor
pretraining independence. Do not claim those forms of generalization or add a
second complex source on this evidence alone.

Stable PINDER IDs replace positional sample names. Batches retain source/release,
cluster IDs, chain/atom mappings, retained crop indices, requested/returned provider
indices and retry substitutions. Representative-atom interface coverage uses the
recorded 8 A cutoff, with counts before/after receptor cropping. Native observation
and frame masks remain in supervision records.

Filtering parameters and source manifests are hashed. Python, NumPy and Torch
are seeded before construction. A microstep-indexed stream selects source, item
and coordinate kind reproducibly and isolates worker RNG; prefetch does not
advance the committed cursor. Checkpoints save at optimizer boundaries and restore
the consumed microstep. Evaluation isolates and restores training RNG. Nonfinite
loss/gradients stop before an optimizer update, and checkpoint files are replaced
atomically.

Rigid transforms, reconstruction noise, query order, sequence sampling and main/
auxiliary SC initialization have separate seeds derived from the item seed.
PINDER conversion and validation data fetching have isolated CPU RNG contexts.
The new protocol also enables deterministic Torch algorithms and a deterministic
cuBLAS workspace in training and checkpoint-based evaluation. This matters for
BF16 complex inputs: a fixed-input GPU probe showed varying frozen-backbone
features even without reloading. Deterministic mode removed all observed feature,
SC-input and prediction differences across repeated calls and reloads.

## Full-sample cache and validation

Export audited validation inputs during a native-phase preflight, then generate
backbones separately on a GPU:

```bash
bash scripts/training/slurm_sc_complex_adapt_hai.sh \
  --dry-run --export-validation-inputs --output-dir /path/to/preflight
python scripts/utilities/cache_sc_backbones.py \
  --checkpoint /path/to/integrated_checkpoint.pt \
  --input-manifest /path/to/preflight/native_validation_inputs/manifest.json \
  --output /path/to/full400
```

The cache records the official native sampler, 400 steps, seed, source partition,
parent sample ID, input-manifest hash and backbone checkpoint hash. Native labels
are stripped. Readers verify item hashes, partition and official-donor identity.
Both cached and online inputs use the same fresh positive-sigma feature pass.
`sc_adapt` launch requires a separate full-400-step validation cache. Full samples
are 0% of training by default; the optional stream is capped at 10%, requires a
training-partition cache and a nonzero physical coefficient. Clash-only training
remains experimental and must retain supervised replay.

Validation panels are separate native monomer/complex, paired reconstruction at
each fixed sigma/seed, and full-400-step samples. Packing metrics include
symmetry-aware SC RMSD, χ and all-valid-χ rotamer recovery (<40°), named covalent
bond error (>0.2 A from CCD ideal), clash objective, completeness and counts.
Interface/noninterface and 80–130-residue binder subsets are reported. Per-protein
means exclude empty denominators; pooled estimates and valid-protein counts are
also reported. Zero valid counts mean missing supervision, not a perfect score.
The full-sample panel reports generated geometry without native-reference metrics.
These packing covalent errors are distinct from the historical CA-spacing “坏键率”.

The smoke launcher tests a real PINDER complex, a missing-O derivative, SC-only
updates, frozen component hashes, validation and save/resume, native/paired
generated-input label invariance and a completely unlabeled input interface.
The unlabeled smoke fixture is not a 400-step sample or evidence of deployment
quality. Run actual 400-step validation before accepting predicted-input
adaptation. Advance only on interface improvement with monomer retention, then
predicted-input robustness with sound generated geometry. A lower auxiliary MSE
or clash objective alone does not establish acceptance.

## Implementation verification, 2026-09-12

- CPU regression job `115293`: **676 passed**. Python compilation and shell syntax
  checks also passed.
- GPU smoke `115294`: **passed** on PINDER
  `8phr__X4_UNDEFINED--8phr__W4_UNDEFINED` and a missing-O derivative, with held-out
  PINDER/monomer validation. Dataset reconstruction/fingerprints, checkpoint
  save/resume, frozen-weight invariance, native/paired hidden-label invariance,
  and unlabeled forward/backward passed.
- Only the SC group was trainable (116.56M of 266.51M model parameters). Native and
  paired generated-physical gradient norms were respectively **0.00217** and
  **0.01434**, before multiplication by the physical coefficient.
- Complex rotation-consistency RMSD was **1.545 A** on this minimally trained test
  fixture. This measures the remaining rotation sensitivity; it is not evidence
  that an accepted SC checkpoint is equivariant.
- Fixed-input diagnostic `115292` found identical backbone features, SC inputs
  and predictions across repeated calls/reloads with deterministic algorithms.
  The preceding nondeterministic probe `115290` did not.

The GPU fixture used `114964/step2_gt_warmup.pt`, a smoke checkpoint, rather than
selecting a checkpoint from a current training run. Full 400-step sample quality
and adaptation benefit were **not** established. The initial physical objective
is much smaller than coordinate MSE on this fixture; the .01 coefficient remains
a proposed comparison and should be assessed through held-out geometry and
gradient/objective scales.

Detailed smoke record:
`/hai/scratch/yfsun/proteo_aa_runs/sc_adaptation_smoke/115294/smoke_result.json`.
