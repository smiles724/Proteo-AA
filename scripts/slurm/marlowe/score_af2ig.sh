#!/bin/bash
#SBATCH --job-name=pxf_af2ig
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/af2ig/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/af2ig/%x-%j.out
set -uo pipefail
TARGET="${TARGET:?set TARGET}"
B="${BENCH_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench}"
DESIGNS_CSV="${DESIGNS_CSV:-$B/designs_all.csv}"
METRICS="${METRICS_DIR:-$B/metrics}"
AA=/users/yfsun/Proteo-AA

cd "$AA"
export AF2_PARAMS_DIR=/users/yfsun/af2_params
export PYTHONPATH="$AA${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} target=$TARGET"
nvidia-smi --query-gpu=name --format=csv,noheader
/users/yfsun/.venvs/af2ig/bin/python scripts/evaluation/fold_af2ig.py \
  --designs-csv "$DESIGNS_CSV" \
  --designs-dir "$B/designs_v1" \
  --inputs-dir "$B/inputs" \
  --metrics-csv "$METRICS/af2ig_${TARGET}.csv" \
  --data-dir "$AF2_PARAMS_DIR" \
  --variants co_design \
  --targets "$TARGET" \
  --num-recycles 3
echo "EXIT=$?"
