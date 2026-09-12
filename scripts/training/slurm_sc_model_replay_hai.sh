#!/usr/bin/env bash
#SBATCH --job-name=sc-model-replay
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=logs/training/sc-model-replay-%j.out
#SBATCH --error=logs/training/sc-model-replay-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:/hai/users/y/f/yfsun/Protein Project/fampnn"
export PROTENIX_ROOT_DIR=/hai/scratch/yfsun/protenix_data
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OPTIONS=()
if [[ ${SC_DETERMINISTIC:-0} == 1 ]]; then OPTIONS=(--deterministic); fi
exec /hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python "$REPO/scripts/utilities/diagnose_sc_model_replay.py" \
  --checkpoint "$SC_TEST_CHECKPOINT" --batch "$SC_TEST_BATCH" --output "/hai/scratch/yfsun/proteo_aa_runs/sc_data_replay/model-$SLURM_JOB_ID.json" "${OPTIONS[@]}"
