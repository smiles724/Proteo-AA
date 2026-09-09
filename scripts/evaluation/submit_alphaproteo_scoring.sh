#!/bin/bash
# Prepare scoring tasks and submit a bounded slice as a Slurm array.
# Use TASK_START/TASK_COUNT for scheduler-friendly waves; completed tasks resume.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT from generation}"
PXDBENCH_DIR="${PXDBENCH_DIR:?set PXDBENCH_DIR}"
PXDBENCH_PYTHON="${PXDBENCH_PYTHON:?set PXDBENCH_PYTHON}"
TOOL_WEIGHTS_ROOT="${TOOL_WEIGHTS_ROOT:?set TOOL_WEIGHTS_ROOT}"
PREP_PYTHON="${PREP_PYTHON:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}"
TASK_FILE="${TASK_FILE:-${RUN_ROOT}/score_tasks.tsv}"
TASK_START="${TASK_START:-0}"
TASK_COUNT="${TASK_COUNT:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

cd "${REPO_ROOT}"
mkdir -p logs/validation/alphaproteo10 "${RUN_ROOT}/scores"
"${PREP_PYTHON}" scripts/evaluation/prepare_alphaproteo_score_tasks.py \
  --generation-root "${RUN_ROOT}/generation" \
  --score-root "${RUN_ROOT}/scores" \
  --output "${TASK_FILE}" \
  --mpnn-sequences "${MPNN_SEQUENCES:-1}"

n_tasks="$(( $(wc -l < "${TASK_FILE}") - 1 ))"
if (( TASK_START >= n_tasks )); then
  echo "All tasks are covered: TASK_START=${TASK_START}, tasks=${n_tasks}"
  exit 0
fi
task_end="$((TASK_START + TASK_COUNT - 1))"
if (( task_end >= n_tasks )); then task_end="$((n_tasks - 1))"; fi

job_id="$(sbatch --parsable \
  --array="${TASK_START}-${task_end}%${MAX_CONCURRENT}" \
  --export="ALL,TASK_FILE=${TASK_FILE},PXDBENCH_DIR=${PXDBENCH_DIR},PXDBENCH_PYTHON=${PXDBENCH_PYTHON},TOOL_WEIGHTS_ROOT=${TOOL_WEIGHTS_ROOT},MPNN_SEQUENCES=${MPNN_SEQUENCES:-1},MPNN_TEMPERATURE=${MPNN_TEMPERATURE:-0.0001}" \
  scripts/evaluation/slurm_score_alphaproteo_designability.sh)"

echo "SCORING_JOB=${job_id}"
echo "TASK_RANGE=${TASK_START}-${task_end}/${n_tasks}"
echo "Next wave: TASK_START=$((task_end + 1)) with the same command"
