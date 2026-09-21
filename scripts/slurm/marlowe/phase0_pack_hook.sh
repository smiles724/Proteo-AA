#!/bin/bash
#SBATCH --job-name=pxf_phase0_pack_hook
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_phase0/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_phase0/%x-%j.out
#
# Phase 0: the iterative BB->SC hook on real weights, two chains, and the
# held-out dev complexes in configs/binder_benchmark/dev_complexes.yaml (six
# as of 550bdad). See scripts/phase0_pack_hook_check.py for what each of the
# seven checks is protecting against.
#
# This runs as a batch job because the set is ~30 designs at 100 unmasking
# steps: minutes on a GPU, about four minutes per complex on CPU.
#
# An earlier version of this comment blamed two exit-144 deaths on a
# login-node process not surviving a session ending. That was wrong, and the
# correction is worth keeping because the real cause recurs: both runs were
# killed by `pkill -u $USER -f phase0_pack_hook_check` issued from the same
# compound command that launched the replacement, so the pattern matched the
# launching shell's own command line and it killed itself. Login-node
# survival was never demonstrated either way.
#
# Usage:
#   source /users/yfsun/marlowe_env.sh
#   OUT=/scratch/m000137-pm06/Proteo-AA/pxf/runs/pxf_phase0/run1 \
#       sbatch scripts/slurm/marlowe/phase0_pack_hook.sh
#
# PHASE0_EXTRA_ARGS is a separate variable from EXTRA_ARGS on purpose:
# marlowe_env.sh exports EXTRA_ARGS="--mmcif-dir ..." for a different script
# and argparse would exit 2 on it here.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/couple/binder_residual.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo, or predates pxf/couple/binder_residual;" \
         "set PXF_REPO" >&2
    exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
SEQ_STEPS="${SEQ_STEPS:-100}"
SIGMA_B="${SIGMA_B:-0.429}"
GATE="${GATE:-one}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_phase0"
cd "$ROOT"

PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "seq_steps=${SEQ_STEPS} sigma_b=${SIGMA_B} gate=${GATE}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo "=== the coupling unit tests, on a GPU ==="
python -m pytest tests/test_binder_residual.py tests/test_pack_hook.py \
    -q --no-header 2>&1 | tail -15

echo "=== Phase 0 integration checks ==="
python scripts/phase0_pack_hook_check.py \
    --out "$OUT" \
    --seq-steps "$SEQ_STEPS" \
    --sigma-b "$SIGMA_B" \
    --gate "$GATE" \
    --device auto \
    ${PHASE0_EXTRA_ARGS:-}

echo "done: $OUT/phase0_report.json"
