# SC-only complex and predicted-input adaptation

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
| Accepted parent | Rigid-augmentation `sc_warmup` | `sc_complex_adapt` |
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

From this checkout, create `logs/training` before `sbatch`. No launcher selects a
checkpoint from a running job or submits the next phase automatically.

```bash
export PROTEOAA_REPO=/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases
export ACCEPTED_CHECKPOINT=/path/to/accepted_rigid_sc_warmup.pt
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
`train_sc_adaptation.py --help`. Unknown overrides fail. Exact resume rejects
changed recipe arguments; a phase transition resets optimizer/counters and
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
