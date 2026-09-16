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
#   0 = none     frames + residue type only (APM's released setting)
#   1 = plm      + frozen ESM-2 650M, all 34 layers, learned softmax
#   2 = a_token  + the frozen PXDesign trunk's token, read from the cache
#   3 = both
#
# Arms 2-3 need A_TOKEN_CACHE, built by scripts/data/build_a_token_cache.py:
# APM's pickles carry no Protenix features, so a_token comes from a separate
# pass over the same chains. At zero coordinate noise with the orientation
# pinned it is a deterministic function of the structure, hence a cache rather
# than a trunk forward per step. The driver refuses those arms without it
# rather than feeding zeros and training `none` under another name.
#
#   sbatch --array=0-1 scripts/training/slurm_packer_apm_data_hai.sh
#   A_TOKEN_CACHE=... sbatch --array=2-3 scripts/training/slurm_packer_apm_data_hai.sh
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
APM_REFERENCE=${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}
# dm-tree, lightning-utilities and the torch_scatter shim, installed beside the
# env rather than into it so APM's imports resolve without touching proteoaa.
PYEXTRA=${PYEXTRA:-/hai/scratch/shenjm/pyextra}
PYBIN=${PYBIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}
RUNROOT=${RUNROOT:-/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data}

ARMS=(none plm a_token both)
ARM=${ARM:-${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}}
OUT="$RUNROOT/$ARM"

A_TOKEN_CACHE=${A_TOKEN_CACHE:-/hai/scratch/shenjm/proteo_aa_runs/a_token_cache}
CACHE_ARGS=()
if [[ "$ARM" == a_token || "$ARM" == both ]]; then
  [[ -d "$A_TOKEN_CACHE/train" && -d "$A_TOKEN_CACHE/val" ]] || {
    echo "arm $ARM needs $A_TOKEN_CACHE/{train,val}; run build_a_token_cache first" >&2
    exit 2; }
  # Skip rather than block: a handful of chains have no cached a_token because
  # their generated mmCIF does not survive Protenix's parse. The count and the
  # names land in the run directory, so the arm difference is auditable.
  CACHE_ARGS=(--a-token-cache "$A_TOKEN_CACHE" --a-token-skip-missing)
fi

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
  --resume "${CACHE_ARGS[@]}" "$@"
