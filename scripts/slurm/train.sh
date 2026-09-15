#!/bin/bash
#SBATCH --job-name=pxf_train
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_train/%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_train/%j.out
#
# Continue training FaMPNN's full-atom modules. Objectives and loop are written
# from the preprint (bioRxiv 2025.02.13.637498); see pxf/train/.
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch.
#
# Submit from an interactive job with SLURM_* cleared, or the job silently
# inherits this shell's allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       PDB_DIR=<dir> sbatch scripts/slurm/train.sh
set -euo pipefail

# Derived from this script's location so the repo can be moved or renamed
# (it already has been, into ~/"Proteo-AA old"/). Quote it everywhere: the
# current path contains a space.
# sbatch copies this script to /var/lib/slurm/scripts, so BASH_SOURCE does NOT
# point at the repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd
# (the repo root); fall back to BASH_SOURCE only for direct execution.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi
OUT=${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_train/${SLURM_JOB_ID}}
PDB_DIR=${PDB_DIR:?set PDB_DIR to a directory of training PDBs}
CONFIG=${CONFIG:-$ROOT/configs/train_cath.yaml}
INIT=${INIT:-0.0}
mkdir -p "$OUT"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"

echo "node=$(hostname) job=$SLURM_JOB_ID out=$OUT"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# Checkpoints are ~120 MB each and land under OUT on scratch, never in home.
RESUME_ARG=""
LATEST=$(ls -1 "$OUT"/checkpoints/step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming from $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

python scripts/train.py \
    --pdb-dir "$PDB_DIR" \
    --out "$OUT" \
    --config "$CONFIG" \
    --init-weights "$INIT" \
    ${RESUME_ARG} \
    ${EXTRA_ARGS:-}

echo "outputs under $OUT"
