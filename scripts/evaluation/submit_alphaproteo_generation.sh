#!/bin/bash
# Submit two matched generation jobs: official PXDesign and one Proteo-AA ckpt.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
PXDESIGN_CHECKPOINT="${PXDESIGN_CHECKPOINT:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}"
PROTEOAA_CHECKPOINT="${PROTEOAA_CHECKPOINT:?set PROTEOAA_CHECKPOINT}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_designability/${RUN_TAG}}"
SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_generate_alphaproteo_designability.sh"

mkdir -p "${REPO_ROOT}/logs/validation/alphaproteo10" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export REPO_ROOT RUN_ROOT
export NUM_DESIGNS_PER_TARGET="${NUM_DESIGNS_PER_TARGET:-1}"
export FIXED_LENGTH="${FIXED_LENGTH:-}"
export LENGTH_MIN="${LENGTH_MIN:-80}" LENGTH_MAX="${LENGTH_MAX:-130}"
export SEED="${SEED:-42}" N_STEP="${N_STEP:-400}"
export TARGETS="${TARGETS:-bhrf1,h1,il17a,il7ra,ir,pdl1,sc2rbd,tnfa,trka,vegfa}"

px_job="$(CHECKPOINT="${PXDESIGN_CHECKPOINT}" \
  MODEL_LABEL=pxdesign_official MODEL_MODE=pxdesign \
  sbatch --parsable --job-name=alpha10-pxdesign "${SCRIPT}")"

aa_job="$(CHECKPOINT="${PROTEOAA_CHECKPOINT}" \
  MODEL_LABEL="${PROTEOAA_LABEL:-proteoaa}" MODEL_MODE=proteoaa \
  AA_READOUTS="${AA_READOUTS:-final,target_sigma,confidence_best}" \
  AA_READOUT_SIGMA="${AA_READOUT_SIGMA:-0.4}" \
  sbatch --parsable --job-name=alpha10-proteoaa "${SCRIPT}")"

cat <<EOF
RUN_ROOT=${RUN_ROOT}
PXDESIGN_GENERATION_JOB=${px_job}
PROTEOAA_GENERATION_JOB=${aa_job}

After both jobs finish, run:
  RUN_ROOT=${RUN_ROOT} \\
  PXDBENCH_DIR=<PXDesignBench-v0.1.2> \\
  PXDBENCH_PYTHON=<pxdbench-env>/bin/python \\
  TOOL_WEIGHTS_ROOT=<tool_weights> \\
  bash scripts/evaluation/submit_alphaproteo_scoring.sh
EOF
