#!/bin/bash
#SBATCH --job-name=proteo-aa-stage3-binder
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/stage3_binder/%x-%j.out
#SBATCH --error=logs/training/stage3_binder/%x-%j.err

# Stage III co-evolution on PROTEIN-PROTEIN INTERFACES -- i.e. binder design.
#
# Same stage and the same warm-start contract as
# `slurm_stage3_coevolution_monomer.sh` (job 101203): a Stage II side-chain
# checkpoint supplies the backbone and S_phi, a separate run supplies the AA
# head, both modules then train together with the refinement pass live. The only
# difference is the DATA: `--data-mode mixed_monomer_complex` adds the two
# complex sources under a curriculum instead of training on monomers alone.
#
#   R=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}/proteo_aa_runs
#   LOAD_CHECKPOINT=$R/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt \
#   AA_HEAD_CHECKPOINT=$R/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt \
#   sbatch scripts/training/slurm_stage3_coevolution_binder.sh
#
# On a different machine, set the two roots first (see docs/datasets.md):
#   export PROTEOAA_DATA_ROOT=/my/scratch   PROTEOAA_CODE_ROOT=/my/code
#
# SMOKE=1 runs a handful of steps at a small crop. Run it once at your intended
# CROP_SIZE before committing a 24h slot: the refinement pass runs the backbone
# twice, so peak memory is ~2x Stage II at the same crop, and a complex crop is
# larger than the monomer crop this stage was tuned on.
#
# ---------------------------------------------------------------------------
# KNOWN SAMPLING DEFECT, NOT FIXED HERE. Both complex sources are sampled
# uniformly over INDEX ROWS, and PINDER's rows are cluster-redundant: 1,437,458
# train rows over 40,231 clusters, largest cluster 82,272 rows. The effective
# number of interface families under uniform row sampling is 78 (inverse
# Simpson), and the top 10 clusters take 30% of all draws. Protenix complexes
# are milder but not clean: 188,277 rows / 25,930 clusters -> effective 595.
#
# The fix is per-item weights of 1/cluster_size, which `CurriculumMultiDataset`
# already supports (`per_item_weights`) and which
# `train_protenix_monomer.py:472,484` currently hardcodes to uniform. Until that
# lands, read the mixture fractions below as "share of draws", NOT as "share of
# distinct interfaces seen".
# ---------------------------------------------------------------------------

set -euo pipefail

source /hai/users/s/h/shenjm/miniconda3/etc/profile.d/conda.sh
conda activate proteoaa

# Complex crops vary substantially in shape.  Without expandable CUDA segments,
# the caching allocator can strand tens of GiB in unusable reserved blocks after
# enough differently sized batches (job 107381 had 38.95 GiB reserved but
# unallocated when a 1.41 GiB attention tensor was requested).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- where things live ------------------------------------------------------
# Set PROTEOAA_DATA_ROOT and PROTEOAA_CODE_ROOT and nothing else needs changing;
# every path below derives from them and stays individually overridable.
# docs/datasets.md says what each directory must contain and how to obtain it.
PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
PROTEOAA_CODE_ROOT="${PROTEOAA_CODE_ROOT:-/hai/users/y/f/yfsun/Protein Project}"

# Locating the checkout: probe candidates and take the first that really is one,
# because every single-source answer is wrong in some invocation we actually use.
#   ${BASH_SOURCE[0]}  correct for `bash scripts/training/...`, but sbatch copies
#                      the script to its spool dir, where this yields /var/lib/slurm
#                      (measured: job 104976 died on `mkdir /var/lib/slurm/logs`).
#   SLURM_SUBMIT_DIR   correct under sbatch, but an INTERACTIVE srun/salloc shell
#                      inherits a stale one -- on this cluster a bash job exported
#                      SLURM_SUBMIT_DIR=$HOME, which silently beat the correct
#                      BASH_SOURCE answer.
# Probing for the marker file makes the order harmless.
_repo_marker="scripts/training/train_protenix_monomer.py"
if [[ -z "${REPO_ROOT:-}" ]]; then
  for _cand in \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(pwd)"; do
    if [[ -n "${_cand}" && -f "${_cand}/${_repo_marker}" ]]; then
      REPO_ROOT="${_cand}"
      break
    fi
  done
fi
if [[ -z "${REPO_ROOT:-}" || ! -f "${REPO_ROOT}/${_repo_marker}" ]]; then
  echo "ERROR: could not locate the Proteo-AA checkout (tried the script's own" >&2
  echo "       directory, SLURM_SUBMIT_DIR and \$PWD). Pass REPO_ROOT=/path/to/Proteo-AA." >&2
  exit 2
fi
DATA_ROOT="${DATA_ROOT:-${PROTEOAA_DATA_ROOT}/protenix_data}"
RUNS_ROOT="${RUNS_ROOT:-${PROTEOAA_DATA_ROOT}/proteo_aa_runs}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-${PROTEOAA_CODE_ROOT}/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-${PROTEOAA_CODE_ROOT}/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/stage3_binder_coevolution/${SLURM_JOB_ID:-manual}}"

for d in "${DATA_ROOT}" "${PROTENIX_CODE_DIR}" "${PXDESIGN_CODE_DIR}"; do
  [[ -d "$d" ]] || { echo "ERROR: missing directory ${d} (see docs/datasets.md)" >&2; exit 2; }
done
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: PYTHON_BIN not executable: ${PYTHON_BIN}" >&2; exit 2; }

# --- warm start -------------------------------------------------------------
# Both are REQUIRED for the initial warm start, unlike the monomer script's
# optional-empty default. A Stage II checkpoint carries a chance-level AA head
# (Stage II excluded the head from its warm start and then froze it at random
# init -- measured: CE 3.00 = ln 20, accuracy at the majority-class floor, and
# `strict_random` scoring the same as `strict_native`). Starting Stage III off
# that head wastes the run, so the script refuses rather than letting it happen
# silently. A full resume already contains Stage III's trained AA head and must
# not overlay the original donor again.
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:?set LOAD_CHECKPOINT to a Stage II side-chain checkpoint}"
WARM_START_PARAMS_ONLY="${WARM_START_PARAMS_ONLY:-1}"
if [[ "${WARM_START_PARAMS_ONLY}" == "1" ]]; then
  AA_HEAD_CHECKPOINT="${AA_HEAD_CHECKPOINT:?set AA_HEAD_CHECKPOINT to a run that trained design_residue_type_head}"
else
  AA_HEAD_CHECKPOINT="${AA_HEAD_CHECKPOINT:-}"
fi

# --- complex data sources ---------------------------------------------------
COMPLEX_PROVIDER="${COMPLEX_PROVIDER:-both}"
PINDER_RELEASE="${PINDER_RELEASE:-2024-02}"
PINDER_ROOT="${PINDER_ROOT:-${PROTEOAA_DATA_ROOT}/pinder/${PINDER_RELEASE}}"
PINDER_MANIFEST="${PINDER_MANIFEST:-${PINDER_ROOT}/indices/pinder_ppi_complex.parquet}"
PINDER_CACHE_ROOT="${PINDER_CACHE_ROOT:-/hai/scratch/shenjm/pinder}"
PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-${PINDER_CACHE_ROOT}/cif_cache}"
PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-${PINDER_CACHE_ROOT}/${PINDER_RELEASE}/pdbs}"
PINDER_ARCHIVE="${PINDER_ARCHIVE:-${PINDER_ROOT}/raw/pdbs.zip}"

# Mixture. PINDER carries the larger complex share because 85% of its dimers fit
# a 640 crop whole (median 455 tokens; binder 198) against 41% for Protenix
# complexes (median 756) -- `--complex-max-n-token` is only an INDEX filter, the
# actual crop is CROP_SIZE, so an over-long complex is cropped and can lose the
# interface. Protenix complexes stay non-trivial because they are the only
# complex source carrying MSA features.
#
# With these three numbers the trainer builds:
#   step <= ramp start : monomer 0.50 / protenix_ppi 0.15  / pinder_ppi 0.35
#   step >= ramp end   : monomer 0.25 / protenix_ppi 0.225 / pinder_ppi 0.525
# The monomer share is not decoration -- it is the only source tied to the
# established 491-protein validation, and it holds the fold prior while the
# interface sources pull the model toward the binder task.
STAGE2_START_MONOMER_FRAC="${STAGE2_START_MONOMER_FRAC:-0.50}"
STAGE2_END_MONOMER_FRAC="${STAGE2_END_MONOMER_FRAC:-0.25}"
PINDER_COMPLEX_FRAC="${PINDER_COMPLEX_FRAC:-0.70}"
CURRICULUM_STAGE1_END_STEP="${CURRICULUM_STAGE1_END_STEP:-2000}"
CURRICULUM_STAGE2_START_STEP="${CURRICULUM_STAGE2_START_STEP:-15000}"
COMPLEX_MAX_N_TOKEN="${COMPLEX_MAX_N_TOKEN:-1536}"
COMPLEX_MAX_BINDER_FRACTION="${COMPLEX_MAX_BINDER_FRACTION:-0.75}"

COMPLEX_ARGS=(--data-mode mixed_monomer_complex --complex-provider "${COMPLEX_PROVIDER}")
if [[ "${COMPLEX_PROVIDER}" == "pinder" || "${COMPLEX_PROVIDER}" == "both" ]]; then
  COMPLEX_ARGS+=(
    --pinder-manifest "${PINDER_MANIFEST}"
    --pinder-root "${PINDER_ROOT}"
    --pinder-cif-cache "${PINDER_CIF_CACHE}"
    --pinder-pdb-cache "${PINDER_PDB_CACHE}"
    --pinder-archive "${PINDER_ARCHIVE}"
  )
  for f in "${PINDER_MANIFEST}" "${PINDER_ARCHIVE}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing PINDER input ${f}" >&2; exit 2; }
  done
  mkdir -p "${PINDER_CIF_CACHE}" "${PINDER_PDB_CACHE}"
  for d in "${PINDER_CIF_CACHE}" "${PINDER_PDB_CACHE}"; do
    [[ -w "${d}" ]] || {
      echo "ERROR: PINDER cache is not writable: ${d}" >&2
      exit 2
    }
  done
fi
COMPLEX_ARGS+=(
  --complex-max-n-token "${COMPLEX_MAX_N_TOKEN}"
  --complex-max-binder-fraction "${COMPLEX_MAX_BINDER_FRACTION}"
  --stage2-start-monomer-frac "${STAGE2_START_MONOMER_FRAC}"
  --stage2-end-monomer-frac "${STAGE2_END_MONOMER_FRAC}"
  --pinder-complex-frac "${PINDER_COMPLEX_FRAC}"
  --curriculum-stage1-end-step "${CURRICULUM_STAGE1_END_STEP}"
  --curriculum-stage2-start-step "${CURRICULUM_STAGE2_START_STEP}"
)

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-6}"; LOG_INTERVAL="${LOG_INTERVAL:-1}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-0}"; CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
  TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-64}"
  CROP_SIZE="${CROP_SIZE:-256}"
  ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-1}"
else
  MAX_STEPS="${MAX_STEPS:-30000}"; LOG_INTERVAL="${LOG_INTERVAL:-50}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"; CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
  TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
  # Larger than the monomer stage's 384: a PINDER dimer is a median 455 tokens,
  # so 384 would crop the majority of interfaces. 512 keeps most of them and is
  # the knob to drop first if this OOMs -- prove it with SMOKE=1 CROP_SIZE=512.
  CROP_SIZE="${CROP_SIZE:-512}"
  ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
fi

[[ -f "${LOAD_CHECKPOINT}" ]] || {
  echo "ERROR: checkpoint does not exist: ${LOAD_CHECKPOINT}" >&2
  exit 2
}
if [[ "${WARM_START_PARAMS_ONLY}" == "1" && ! -f "${AA_HEAD_CHECKPOINT}" ]]; then
  echo "ERROR: AA-head checkpoint does not exist: ${AA_HEAD_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs/training/stage3_binder" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOAD_ARGS=(--load-checkpoint "${LOAD_CHECKPOINT}")
if [[ "${WARM_START_PARAMS_ONLY}" == "1" ]]; then
  LOAD_ARGS+=(--warm-start-params-only)
  LOAD_ARGS+=(--load-aa-head-from "${AA_HEAD_CHECKPOINT}")
fi

ABLATION_ARGS=()
if [[ -n "${AA_HEAD_LR:-}" ]]; then
  ABLATION_ARGS+=(--aa-head-lr "${AA_HEAD_LR}")
fi
if [[ "${DETACH_AA_LOGITS_FOR_SIDECHAIN:-0}" == "1" ]]; then
  ABLATION_ARGS+=(--detach-aa-logits-for-sidechain)
fi

echo "stage3-binder  crop=${CROP_SIZE}  provider=${COMPLEX_PROVIDER}"
echo "  monomer ${STAGE2_START_MONOMER_FRAC} -> ${STAGE2_END_MONOMER_FRAC}"
echo "  pinder share of complex mass: ${PINDER_COMPLEX_FRAC}"
echo "  ramp ${CURRICULUM_STAGE1_END_STEP} -> ${CURRICULUM_STAGE2_START_STEP}"
echo "  CUDA allocator : ${PYTORCH_CUDA_ALLOC_CONF}"
if [[ "${COMPLEX_PROVIDER}" == "pinder" || "${COMPLEX_PROVIDER}" == "both" ]]; then
  echo "  PINDER PDB cache: ${PINDER_PDB_CACHE}"
  echo "  PINDER CIF cache: ${PINDER_CIF_CACHE}"
fi
echo "  AA-head lr    : ${AA_HEAD_LR:-same as backbone}"
echo "  detach AA->Sφ : ${DETACH_AA_LOGITS_FOR_SIDECHAIN:-0}"
echo "  backbone+S_phi: ${LOAD_CHECKPOINT}"
if [[ "${WARM_START_PARAMS_ONLY}" == "1" ]]; then
  echo "  AA head       : ${AA_HEAD_CHECKPOINT}"
else
  echo "  resume        : model + AA head + optimizers + schedulers + step counters"
fi
echo "  output        : ${RUN_ROOT}"

"${PYTHON_BIN}" -u scripts/training/train_protenix_monomer.py \
  --training-stage coevolution \
  "${COMPLEX_ARGS[@]}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE}" \
  --max-n-token "${CROP_SIZE}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --max-steps "${MAX_STEPS}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH}" \
  --lr "${LR:-1e-5}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-samples "${EVAL_SAMPLES:-128}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}" \
  "${LOAD_ARGS[@]}" \
  "${ABLATION_ARGS[@]}" \
  "${@}"

# NOTE ON MODEL SELECTION. `build_eval_dataloader` pins the validation set to
# monomers (`eval_args.data_mode = "monomer"`, train_protenix_monomer.py:566),
# so the val_* lines below measure the monomer task even in this run -- they say
# nothing about binder quality. EVAL_SAMPLES is therefore cut to 128 (the monomer
# stage used 491) to spend the time on training instead. Select checkpoints with
# the PINDER binder eval, which scores the binder chain only:
#
#   CHECKPOINT=<ckpt> sbatch scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh
#
# Reference from the previous mixed backbone run (step50000, 0.65/0.175/0.175):
# binder Ca lDDT 0.698, binder Ca RMSD 2.35 A, binder TM 0.637 over 1810 PINDER
# val complexes -- measured BEFORE the 2026-08-06 leakage fix, so re-measure it
# on the current code before treating it as the number to beat.
