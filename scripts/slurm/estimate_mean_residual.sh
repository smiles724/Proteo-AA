#!/bin/bash
#SBATCH --job-name=pxf_mean_residual
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/%x-%j.out
#
# mu(sigma) for the mean-residual control arm.
#
# STRUCTURES must be the TRAINING manifest. Estimating the mean on an
# evaluation panel would let the control see the set it is scored on, which is
# the confound the arm exists to rule out, so there is no default here and the
# resolved path is recorded in the output JSON.
#
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       CHECKPOINT=.../final.pt OUT=configs/mean_residual_phase1.json \
#       sbatch scripts/slurm/estimate_mean_residual.sh
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the coupling checkpoint}"
OUT="${OUT:?set OUT to the mean-residual JSON path}"
STRUCTURES="${STRUCTURES:-$ROOT/configs/phase1_structures_afdb.txt}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
MAX_PROTEINS="${MAX_PROTEINS:-256}"
N_SIGMA="${N_SIGMA:-5}"
CROP_SIZE="${CROP_SIZE:-512}"
mkdir -p "$(dirname "$OUT")" /hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "checkpoint=${CHECKPOINT}"
echo "training structures=${STRUCTURES} proteins=${MAX_PROTEINS} n_sigma=${N_SIGMA}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

python scripts/estimate_mean_residual.py \
    --checkpoint "$CHECKPOINT" \
    --structures "$STRUCTURES" \
    --pxdesign-donor "$DONOR" \
    --out "$OUT" \
    --max-proteins "$MAX_PROTEINS" \
    --n-sigma "$N_SIGMA" \
    --crop-size "$CROP_SIZE" \
    ${EXTRA_ARGS:-}

echo "done -> $OUT"
