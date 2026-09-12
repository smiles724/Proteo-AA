#!/bin/bash
#SBATCH --job-name=alpha10-summary
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validation/alphaproteo10/%x-%j.out
#SBATCH --error=logs/validation/alphaproteo10/%x-%j.err

set -euo pipefail
source /hai/users/s/h/shenjm/miniconda3/etc/profile.d/conda.sh
conda activate proteoaa

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
TASK_FILE="${TASK_FILE:-${RUN_ROOT}/score_tasks.tsv}"
cd "${REPO_ROOT}"
python scripts/evaluation/summarize_alphaproteo_designability.py \
  --task-file "${TASK_FILE}" \
  --output-dir "${RUN_ROOT}/summary"
