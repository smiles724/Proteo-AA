# Stage IV: FaMPNN sequence head and one-step Proteo-AA packing

Implementation worktree: `/users/yfsun/Proteo-AA-stage4-fampnn`, branch `stage4-fampnn`, based on `847a3f50704adcd02c3fcfbef99047e5c01c378b`. The original `/users/yfsun/Proteo-AA` checkout and its uncommitted experiments are preserved. Reviewed and implemented 10 September 2026.

This is an integration under verification. The frozen FaMPNN boundary and focused regressions have passed CPU checks. The complete real-model optimizer/save-resume/export test has now passed on GPU as HAI job **113676**, on the Stage III donor a production run warm-starts from (see [Running it on HAI](#running-it-on-hai)). No Stage IV training result, matched ProteinMPNN comparison, or binding-quality claim is available yet.

## Implemented behavior

`--training-stage stage4_fampnn` constructs FaMPNN's strictly loaded pretrained `SeqDenoiser.denoiser.seq_design_module` as `aa_head`. The old residue-type MLP is absent from this model and its weights are excluded when warming from earlier checkpoints. Older training stages retain their existing backend.

The upstream source is pinned to `richardshuai/fampnn` revision `aaf788b1502ad95d5c5a84455cfc53f2544f3b45`. The installed release weight is `/users/yfsun/protein-code/fampnn/weights/fampnn_0_3.pt`, SHA-256 `8969b3f1f3c941178076c7800952595a18b56fd3828d15bb993d3ef537938a05`. Loading validates the full checkpoint and retains only its sequence network. FaMPNN's own side-chain diffusion model is not called. Its GVP no-grad behavior is unchanged; the existing distance-feature route carries coordinate gradients.

Proteo-AA retains its donor one-step side-chain module with `sidechain.edm=false`. Generation, observed-target supervision, and fixed receptor context use distinct masks. Inventories come from committed identities, are gated by design ownership and valid frames, and map to Atom37 by atom name. Unobserved chemical atoms remain generatable and are excluded from coordinate loss, including non-finite target placeholders. Receptor and ligand atoms never acquire generated side-chain slots.

`codesign.py` defines the shared state and masking/commit/pack/refine loop. `stage4.py` supplies the actual backbone and packer calls for training and inference. Queried identities become X and their side-chain names, masks, and coordinates are hidden before FaMPNN feature extraction. Generated side chains are transported between old and new backbone frames after each refinement. Every successful exit repacks the final assigned sequence on the final backbone and checks atom-name/inventory consistency and fixed-coordinate invariance.

The default inference budget is three complete revision rounds after initial backbone generation and initial sequence decoding/packing. Final repacking is separate. Each round uses random subsets, occasionally the entire binder, and masked block decoding. Newly assigned identities become visible to later blocks; their side chains become visible after packing. Zero temperature selects argmax; positive temperature samples the normalized canonical 20-class logits. X is excluded. Adaptive stopping is deliberately disabled for the initial comparison. One-round inference requires the evaluator's explicit ablation opt-in.

The tensor core and atom/frame helpers support `[batch,sample,residue,...]`. The existing PXDesign trainer binding explicitly accepts one item at a time, with separate diffusion samples; it rejects a multiple-item tensor instead of borrowing item zero's inventory. The launchers set `--diffusion-batch-size 1` for warmup.

## Training objectives and phases

IV-A freezes the backbone, packer, and feedback modules and trains FaMPNN on generated contexts and queried residues. The frozen modules return to evaluation mode after `model.train()`. IV-B/IV-C enable AA, SC, and a controlled backbone/feedback subset using distinct optimizer groups and joint backward. Initial rates are `1e-5`, `1e-5`, and `1e-6`, respectively. The default controlled backbone subset is the atom-attention decoder. All these settings are experimental starting points.

The objective combines masked AA cross-entropy for initial and revision queries, pre/post backbone losses, generated-state steric loss, and a separate native-AA side-chain auxiliary loss. The auxiliary branch packs from the existing template initialization, uses observed coordinates only as targets, and detaches its backbone features/frames. Its outputs do not enter the generated loop. Discrete AA commitments and atom inventories do not carry gradients.

`--stage4-train-rounds` sets explicit training rollout length; early stopping is off. IV-C currently uses the same generated-context rollout engine as IV-B on denoising-derived initial backbones. A training curriculum mixing full generation-from-noise trajectories, variable-length rollout sampling/truncation, teacher-context curriculum, and adaptive stopping remain follow-up experiments. Inference evaluation already starts binder backbones from noise.

## Data and donor inventory

The actual data root is `/scratch/m000137-pm06/Proteo-AA`. Its `protenix_data` directory is the `PROTENIX_ROOT_DIR`; the older shell setup pointing directly to `/scratch/m000137-pm06` does not resolve these assets correctly.

Available checkpoints under `proteo_aa_runs`:

| Relative checkpoint | Inspection |
| --- | --- |
| `aa_head_warmup/93519/checkpoints/step17000.pt` | No side-chain module |
| `joint_bb_aa_from_aa_head/from_aa_head_step17000/checkpoints/step50000.pt` | No side-chain module |
| `protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt` | Compatible one-step packer |
| `protenix_monomer_sidechain_warmup/frame_residual_resolvedmask_from_bb_step96000/checkpoints/step12000.pt` | Older incompatible local-coordinate parameterization |
| `protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt` | Compatible one-step packer; smoke donor only |
| `stage2_complex_backbone/from_monomer_step96000_protenix_pinder/checkpoints/step50000.pt` | No side-chain module |

No donor on Marlowe is a Stage III coevolution result. The production launcher requires `STAGE3_CHECKPOINT` explicitly and does not relabel the compatible smoke donor as one. The intended donor lives on HAI instead: `proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt`. Job 111408 is crop-448 mixed monomer/PINDER coevolution warm-started from Stage II `step52500` plus AA head `step9000`; it timed out at step 6650 of 30000, so step6000 is its last checkpoint. It is the most-trained coevolution binder state that exists, **not a completed Stage III**, and its own AA head is irrelevant here. Its recorded layout matches the compatible one-step packer below. See `../Proteo-AA-sjm-binder/docs/stage3_binder_run_111408_report.md`.

The compatible donor layout records `bb_context=true`, `centre_coord_input=true`, `frame_aware_head=false`, `template_residual=false`, `type_logits_input=true`, `edm=false`, `a_bs_concat=true`, and `q_bs=false`. The existing layout guards and donor adoption are retained. Warm-start requires complete compatible backbone and packer parameters; newly introduced feedback channels can start from their documented zero initialization. Full resume checks backend/configuration/mapping/mask/phase/cycle/optimizer identity. Switching phases uses a params-only warm start with fresh optimizer state.

PINDER's prepared manifest contains 1,437,458 train rows, 1,810 validation rows, and 1,768 test rows. Stage IV excludes official validation/test cluster IDs from PINDER training before weighting eligible rows. Protenix complex data uses a deterministic hash split by cluster. Complex weights are inverse eligible-row cluster sizes; source-mixture fractions are separate. Existing runtime crop retries can still alter the realized sampling distribution, so weights do not establish perfectly balanced successfully cropped examples.

Metrics use distinct `binder_pinder`, `binder_protenix`, and `monomer_retention` prefixes. Cross-source homology exclusion is **not established** by these independent cluster namespaces, nor between PINDER and the monomer source; a common sequence-clustering manifest is needed for that stronger claim.

## Launch and evaluation

The environment is `/users/yfsun/.venvs/proteoaa-stage4` (Python 3.11, PyTorch 2.7.1). The launcher supplies Protenix, PXDesign, and FaMPNN import paths and `LAYERNORM_TYPE=torch`. It records source/weight/donor hashes in `provenance.json`, installed versions in `environment.txt`, and the effective post-override settings in `arguments.json` and `resolved_config.json`.

```bash
cd /users/yfsun/Proteo-AA-stage4-fampnn
export STAGE3_CHECKPOINT=/absolute/path/to/intended/stage3/checkpoint.pt
export OUTPUT_DIR=/users/yfsun/Proteo-AA-stage4-fampnn/runs/stage4-iva
bash scripts/training/slurm_stage4_fampnn_binder.sh --dry-run --export-stage4-validation
module load slurm
sbatch scripts/training/slurm_stage4_fampnn_binder.sh
```

The bounded launch defaults to IV-A, one training revision, one diffusion sample, 100 steps, and a 30-minute one-GPU job with 192 GiB host memory. This host-memory allowance covers the observed large initialization/CPU-test footprint. These defaults are smoke-scale, not a tuned training prescription. Override `MAX_STEPS`, `TRAIN_ROUNDS`, `STAGE4_PHASE`, `CROP_SIZE`, or CLI options explicitly when setting up a measured run. `--dry-run` resolves configuration, builds data sources/validation loaders, and featurizes an item without constructing the backbone model or starting training.

[Marlowe's Slurm documentation](https://marlowe-research.stanford.edu/documentation/slurm/) directs medium allocations to `batch`. Both Marlowe launchers use `--account=marlowe-m000137-pm06 --partition=batch --qos=medium`. The live account had only medium QoS and no default QoS; `preempt` denied medium. `batch` accepted submission but was down, so job 476535 remained queued and was never observed to run; its log would be `runs/stage4-smoke-476535.log`. The gate was run on HAI instead.

`--export-stage4-validation` saves strict binder batches and a CSV manifest in `validation_batches/`, plus the exact source-specific training cluster sets. Exported identities and clusters follow the actual provider row returned after any crop retry. Then use the same environment as the launcher:

```bash
python scripts/evaluation/eval_stage4_codesign.py \
  --checkpoint /path/to/stage4/checkpoint.pt \
  --fampnn-checkpoint /users/yfsun/protein-code/fampnn/weights/fampnn_0_3.pt \
  --manifest "$OUTPUT_DIR/validation_batches/manifest.csv" \
  --training-clusters "$OUTPUT_DIR/validation_batches/training_clusters.json" \
  --output runs/stage4-generated-comparison \
  --arms A B C D --rounds 1 3 5 --allow-one-round-ablation
```

A/B/C/D disable both feedback channels / enable SC→BB only / enable generated SC→AA only / enable both. All arms keep the same backbone passes, AA calls, mask RNG, receptor context, and final packing. The evaluator exports mmCIF, FASTA, per-sample protocol metadata, conditional recovery/NLL, free sequence recovery, fixed native interface versus non-interface groups, composition, SC clash penalty, peptide C–N distance summaries, and fixed-coordinate error. One versus three versus five rounds intentionally changes compute; compare feedback arms at equal round count.

For the frozen IV-0 sequence comparison on identical supplied predicted states:

```bash
python scripts/evaluation/eval_stage4_fampnn_baseline.py \
  --manifest runs/stage4-generated-comparison/states_manifest.csv \
  --training-clusters "$OUTPUT_DIR/validation_batches/training_clusters.json" \
  --fampnn-checkpoint /users/yfsun/protein-code/fampnn/weights/fampnn_0_3.pt \
  --output runs/stage4-frozen-baseline
```

This compares binder SC hidden versus permitted generated SC at the same query and backbone, and separately performs free sequence decoding. It records an input-state checksum for every comparison. To compare with the reported 33.51% ProteinMPNN recovery, prepare states for exactly that original held-out backbone set and use the same masks/context/scoring protocol; that matched set has not been supplied or identified here. Recovery and NLL are not posterior confidence.

## Running it on HAI

All training data, the PINDER/Protenix assets and every `proteo_aa_runs` checkpoint live on the HAI cluster, so Stage IV runs there. The two launchers above stay Marlowe-native; `slurm_stage4_fampnn_smoke_hai.sh` and `slurm_stage4_fampnn_binder_hai.sh` supply this cluster's Slurm directives, roots, interpreter and donor and then delegate to them, so the preflight, provenance and argument list have one source of truth. The only portability edits to the Marlowe scripts were guarding `module load` (HAI has no module system at all) and turning the smoke launcher's hardcoded paths into the same `PROTEOAA_*` / `PYTHON_BIN` / `DONOR` / `OUTPUT_DIR` overrides the binder launcher already had.

| | HAI | Marlowe |
| --- | --- | --- |
| repo | `/hai/users/y/f/yfsun/Proteo-AA-stage4-fampnn` | `/users/yfsun/Proteo-AA-stage4-fampnn` |
| `PROTEOAA_DATA_ROOT` | `/hai/scratch/yfsun` | `/scratch/m000137-pm06/Proteo-AA` |
| `PROTEOAA_CODE_ROOT` | `/hai/users/y/f/yfsun/Protein Project` | `/users/yfsun/protein-code` |
| interpreter | `conda` env `ml` (Python 3.11, PyTorch 2.7.1+cu126) | `/users/yfsun/.venvs/proteoaa-stage4` |
| Slurm | `--partition=yejin --account=yejin --gres=gpu:h200:1` | `--partition=batch --qos=medium` |

The `ml` environment needed `omegaconf`, `dm-tree`, `torchtyping`, `torch_geometric`, `timm`, `jaxtyping` and `natsort` for FaMPNN's import chain; nothing in the existing torch/CUDA stack changed. The upstream checkout is `$PROTEOAA_CODE_ROOT/fampnn` at the pinned revision with `weights/fampnn_0_3.pt` matching the pinned SHA-256.

**`sbatch` from inside an interactive allocation silently loses the memory request.** A `yejin-interactive` shell exports `SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_NODE` and `SLURM_TRES_PER_TASK`, and `sbatch` inherits them over the script's own `#SBATCH` directives -- a job asking for 192G lands with the shell's 32G and dies far from the cause. Strip `SLURM_*` (keeping `SLURM_CONF`) before submitting and confirm `ReqTRES` with `scontrol show job`.

```bash
mkdir -p logs/training/stage4_fampnn
bash scripts/training/slurm_stage4_fampnn_binder_hai.sh --dry-run --export-stage4-validation
sbatch scripts/training/slurm_stage4_fampnn_smoke_hai.sh          # gate first
sbatch --dependency=afterok:<smoke> scripts/training/slurm_stage4_fampnn_binder_hai.sh
```

The dry run resolves 47,622 monomer rows and 1,178,236 eligible PINDER training rows after cluster-disjoint and binder-token filtering, at a constant 0.25 monomer / 0.75 PINDER mixture, and exported 49 strict held-out binder batches.

The HAI binder launcher replaces the base launcher's 100-step smoke defaults with a measured scale: IV-A, one training revision, crop 384, accumulation 8, gradient clip 1.0, and checkpoint/eval every 2000 steps in a 23:50 slot. Crop 384 is affordable in IV-A only because the frozen backbone, packer and feedback modules retain no autograd graph; **IV-B/IV-C open the packer and the atom-attention decoder and have not been memory-proven at this crop.**

`evaluate()` runs *before* `save_checkpoint()` in the training loop, so a failure in the new binder/monomer validation path at the first eval step discards everything before it with no checkpoint written. Prove the eval path on a short job (`MAX_STEPS=4 EVAL_INTERVAL=2 EVAL_SAMPLES=4 ITERS_TO_ACCUMULATE=1`) rather than discovering it hours into a 24h slot.

## Verification and remaining release gates

Verified: canonical mapping for all 20 residues; Gly/Trp inventories across two items and two samples; missing Lys NZ loss exclusion; query identity/name/mask/coordinate invariance; exact released-logit parity; temperature behavior; fixed receptor/final-state consistency in the shared cycle fixture; checkpoint recomputation uses each refinement call's own feedback; real FaMPNN AA-loss gradients update a real SideChainModule through visible coordinates; IV-A freezes that module. The real-packer gradient test uses a small instance of the same implementation, not the production donor architecture.

The full repository regression suite passed **575 tests** (`runs/stage4-regression.log`, JUnit report `runs/stage4-regression.xml`). The final residue-contiguous mmCIF export also passed a separate readback check. The production launcher dry run passed with 16 monomers, 16 eligible PINDER training rows, and two held-out binder exports (`runs/stage4-launcher-dry-run`). A further 38 affected cycle/trainer tests passed after the final provenance and independent query-RNG changes. The frozen baseline evaluator also passed its complete CLI/artifact check on a labeled synthetic state (`runs/stage4-baseline-cli-check`). These bounded checks do not measure design quality.

Re-run on HAI that suite is 574 passed / 1 failed. `tests/test_bb_atom_mapping.py::test_frames_from_backbone_index_accepts_four_wide` fails on roughly one process in seven: the fixture derives coordinates from `hash(atom_name) % 7`, which `PYTHONHASHSEED` randomizes, so when `hash('N') % 7 == hash('C') % 7` its N and C atoms coincide -- and the new degeneracy guard in `frames_from_backbone_index` correctly reports that frame as invalid. The guard is right and the fixture is seed-dependent; the 575 figure was one lucky seed. It passes at `847a3f5`, and is not yet fixed.

HAI job 113676 ran the gate on the Stage III donor at `stage3_binder_coevolution/111408/checkpoints/step6000.pt`: donor load `unexpected=0`, all loss components finite, both feedback-gradient routes nonzero into the real packer (`aa_to_sc` 1.9e4, `bb_to_sc` 7.8e5), the sampled frozen parameter bit-identical after the optimizer step, save/resume `missing=0, unexpected=0`, and a three-round generation exported to mmCIF. On that one 48-token complex at step 1, `stage4/recovery_pre` was 2.8% and `stage4/recovery_revision` 2.4% -- an initialization reading on a single item, not a measurement. `stage4/sc_aux` was 0.0 there: with `inference_safe_binder` and `backbone_only_binder` the binder has no observed side chains, so the native-AA auxiliary branch has an empty target mask on binder items even though it carries weight 1.0.

That closes the GPU engineering gate. The earlier CPU attempt exposed and fixed a pre-attention feedback hook issue and was stopped after discovering an unintended eight-sample setting; it is not part of this result.

Still open on the training side. HAI job **113677** is the first IV-A run (crop 384, 23:50 slot) and was still in progress when this was written -- no Stage IV training result exists. The periodic `binder_pinder` / `monomer_retention` validation path has not run on a real model at all: it first fires at step 2000, roughly seven hours in, and job **113680** was queued to exercise it in four steps instead.

Still required before selecting an architecture: matched frozen ProteinMPNN/FaMPNN measurement, cluster-audited generated-state validation, full chemical-violation/rotamer and interface metrics, sequence–structure consistency using an independent predictor, matched feedback ablations, and one/three/five-round comparisons. The current clash and bond summaries do not certify chemical validity or binding. Ligands/metals remain available to existing Proteo-AA context/physical terms but are not encoded as generic entities by FaMPNN's protein Atom37 interface.
