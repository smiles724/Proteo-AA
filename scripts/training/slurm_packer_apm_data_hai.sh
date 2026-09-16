#!/bin/bash
#SBATCH --job-name=packer-apm-data
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/training/%x-%A_%a.out
#SBATCH --error=logs/training/%x-%A_%a.err
#
# The torsion packer retrained from scratch on APM's data with APM's objective.
# Array index selects the sequence-conditioning arm:
#   0 = none   (frames + residue type only, APM's released setting)
#   1 = plm    (+ frozen ESM-2 650M, all 34 layers, learned softmax)
# a_token/both are absent on purpose: they need the frozen PXDesign trunk, whose
# features APM's pickles do not carry. That bridge is separate work.
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
APM_REFERENCE=${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}
# dm-tree, lightning-utilities and the torch_scatter shim, installed beside the
# env rather than into it so APM's imports resolve without touching proteoaa.
PYEXTRA=${PYEXTRA:-/hai/scratch/shenjm/pyextra}
PYBIN=${PYBIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}
RUNROOT=${RUNROOT:-/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data}

ARMS=(none plm)
ARM=${ARM:-${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}}
OUT="$RUNROOT/$ARM"

export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:$APM_REFERENCE:$PYEXTRA"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$OUT" logs/training

echo "arm=$ARM out=$OUT node=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$PYBIN" scripts/training/train_packer_apm_data.py \
  --arm "$ARM" --out "$OUT" \
  --max-epochs "${MAX_EPOCHS:-200}" --accum "${ACCUM:-8}" \
  --val-every "${VAL_EVERY:-10}" --val-n "${VAL_N:-100}" \
  --workers 6 --time-limit-h "${TIME_LIMIT_H:-22}" \
  --resume "$@"
