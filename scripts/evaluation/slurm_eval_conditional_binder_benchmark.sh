#!/bin/bash
#SBATCH --job-name=proteo-aa-cbdb
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/evaluation/cbdb/%x-%j.out
#SBATCH --error=logs/evaluation/cbdb/%x-%j.err

# A-CODE ConditionalBinderDesignBenchmark (arXiv:2605.03360, Sec. 4.2) on a
# Stage III co-evolution checkpoint.
#
# This is the GENERATION half. It writes designs/*.pdb + designs.csv; designability
# then needs an AF2 initial-guess pass over those PDBs followed by
# scripts/evaluation/score_af2ig_designability.py. See
# docs/conditional_binder_design_benchmark.md.
#
#   SMOKE=1   two targets x one length x two samples with 20 diffusion steps --
#             proves the harness assembles before committing a 24h slot.
#   VALIDATE=1 prepare every input and exit (no GPU needed): checks each target
#             structure is present and every published hotspot resolves.
#
# The paper-scale run is 6 lengths x 64 samples x 6 runnable targets = 2,304
# designs at 1000 Euler steps. Budget accordingly, or split by target with
# TARGETS=... across several jobs; designs.csv is append-only and the driver
# resumes from it.

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
MMCIF_DIR="${MMCIF_DIR:-${DATA_ROOT}/mmcif}"

# Stage III (co-evolution) checkpoint. Required.
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/cbdb/${SLURM_JOB_ID:-manual}}"
TARGETS="${TARGETS:-}"
SMOKE="${SMOKE:-0}"
VALIDATE="${VALIDATE:-0}"

if [[ "${VALIDATE}" != "1" && -z "${CHECKPOINT}" ]]; then
  echo "ERROR: set CHECKPOINT=<stage3 .pt> (or VALIDATE=1 to only prepare inputs)" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ "${SMOKE}" == "1" ]]; then
  LENGTHS="${LENGTHS:-80}"
  SAMPLES_PER_LENGTH="${SAMPLES_PER_LENGTH:-2}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
  TARGETS="${TARGETS:-PDL1 TrkA}"
else
  # Empty = the manifest's paper-scale grid (80..130 by 10, 64 per length).
  LENGTHS="${LENGTHS:-}"
  SAMPLES_PER_LENGTH="${SAMPLES_PER_LENGTH:-}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
fi

mkdir -p "${REPO_ROOT}/logs/evaluation/cbdb" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --mmcif-dir "${MMCIF_DIR}"
  --data-root "${DATA_ROOT}"
  --protenix-code-dir "${PROTENIX_CODE_DIR}"
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --training-stage "${TRAINING_STAGE:-coevolution}"
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}"
  --crop-size "${CROP_SIZE:-640}"
  --dtype "${DTYPE:-bf16}"
)
# OFF by default, which is not a preference: SIDECHAIN_CYCLE=1 currently dies in
# ProtenixDesignTrain._a_token_forward_hook with
#   AttributeError: 'ProtenixDesignTrain' object has no attribute 'a_token_fusion'
# because that hook is registered for the diffusion_internal AA head regardless of
# `sidechain.a_direct`, but its fusion branch calls `self.a_token_fusion`, which is
# only constructed when a_direct is True -- and the config default is
# a_direct=False / a_direct_pre=True. See docs/conditional_binder_design_benchmark.md.
# The benchmark does not need it: AF2-IG refolds from the sequence, so designability
# depends on the binder sequence and backbone, not on S_phi's side chains. Turning
# it on only enriches the written PDBs.
if [[ "${SIDECHAIN_CYCLE:-0}" == "1" ]]; then ARGS+=(--sidechain-cycle); fi
if [[ -n "${CHECKPOINT}" ]]; then ARGS+=(--checkpoint "${CHECKPOINT}"); fi
if [[ -n "${TARGETS}" ]]; then ARGS+=(--targets ${TARGETS}); fi
if [[ -n "${LENGTHS}" ]]; then ARGS+=(--lengths ${LENGTHS}); fi
if [[ -n "${SAMPLES_PER_LENGTH}" ]]; then ARGS+=(--samples-per-length "${SAMPLES_PER_LENGTH}"); fi

if [[ "${VALIDATE}" == "1" ]]; then
  ARGS+=(--validate-only --device cpu)
  # --checkpoint is required by the parser but never read on this path.
  if [[ -z "${CHECKPOINT}" ]]; then ARGS+=(--checkpoint /dev/null); fi
else
  ARGS+=(--device "${DEVICE:-cuda}")
fi

"${PYTHON_BIN}" -u scripts/evaluation/eval_conditional_binder_benchmark.py "${ARGS[@]}" "${@}"

echo
echo "generation done -> ${OUTPUT_DIR}/designs"
echo "next: fold designs/*.pdb with AF2 initial-guess, then"
echo "  ${PYTHON_BIN} scripts/evaluation/score_af2ig_designability.py \\"
echo "      --metrics-csv <af2ig_metrics.csv> --output-dir ${OUTPUT_DIR}"
