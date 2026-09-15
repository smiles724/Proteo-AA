#!/bin/bash
#SBATCH --job-name=pxf_codesign
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_codesign/%x-%A_%a.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_codesign/%x-%A_%a.out
#
# Co-design branch of the unconditional benchmark: PXDesign backbones in,
# FaMPNN sequence + side chains out. 100 samples per length in {100,200,300}.
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       sbatch --array=0-2 scripts/slurm/codesign_uncond.sh
set -euo pipefail

# sbatch copies this script to /var/lib/slurm/scripts, so BASH_SOURCE does NOT
# point at the repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd
# (the repo root); fall back to BASH_SOURCE only for direct execution.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi
SAMPLES=${SAMPLES:-/hai/scratch/yfsun/proteo_aa_runs/uncond_table1/20260913-bbonly/armF_step2000_bbonly/samples}
OUT=${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_codesign/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
SHARDS=${SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/Protenix:$ROOT/fampnn"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"

echo "node=$(hostname) job=${SLURM_JOB_ID} shard=${SHARD}/${SHARDS} out=$OUT"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

python scripts/codesign_uncond.py \
    --samples-dir "$SAMPLES" \
    --out "$OUT" \
    --fampnn-weights 0.3 \
    --seq-steps 100 \
    --temperature 0.1 \
    --psce-threshold 0.3 \
    --seed 0 \
    --shard-index "$SHARD" \
    --shard-count "$SHARDS" \
    ${EXTRA_ARGS:-}

echo "shard ${SHARD} done -> $OUT"
