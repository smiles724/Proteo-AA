#!/bin/bash
#SBATCH --job-name=pxf_bs_seq_sc_preflight
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_bs_seq_sc/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_bs_seq_sc/%x-%j.out
#
# bs_seq_sc_v1: section 7 gradient validation + memory/timing preflight, and
# optionally a short smoke. One job because the Marlowe queue is deep enough
# that three submissions cost more wall time than one run.
#
# NEEDS NO TRAINING DATA. It runs on the 31 held-out dimers, selects nothing,
# adopts no coefficient and keeps no checkpoint, so it does not spend them.
# lambda_seq is REPORTED per structure and deliberately not adopted: section 7
# requires a training-only calibration batch and val is not one.
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
# --device is passed as "cuda", not "auto": pxf.device.select_device treats any
# non-empty string as an explicit request and hands it to torch.device, which
# rejects "auto" outright. Explicit is also the right semantics for a job that
# asked SLURM for a GPU -- if CUDA is unusable this should fail loudly rather
# than quietly spend two hours on CPU. (The check script's own --device default
# is still "auto" and will fail the same way if run without this wrapper.)
#
# PHASE0_EXTRA_ARGS is a separate variable from EXTRA_ARGS on purpose:
# marlowe_env.sh exports EXTRA_ARGS="--mmcif-dir ..." for a different script
# and argparse would exit 2 on it here.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/train/bs_seq_sc.py" ]; then
    echo "ROOT=$ROOT predates pxf/train/bs_seq_sc; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
N_EXAMPLES="${N_EXAMPLES:-4}"
SMOKE_STEPS="${SMOKE_STEPS:-0}"
SIGMA_B="${SIGMA_B:-0.429}"
FAMPNN_VARIANT="${FAMPNN_VARIANT:-0.3}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_bs_seq_sc"
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
echo "fampnn=${FAMPNN_VARIANT} sigma_b=${SIGMA_B} n=${N_EXAMPLES} smoke=${SMOKE_STEPS}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo "=== coupling + mask unit tests, on a GPU ==="
python -m pytest tests/test_binder_residual.py tests/test_binder_masks.py \
    tests/test_shared_prelogit.py tests/test_pack_hook.py -q --no-header 2>&1 | tail -12

echo "=== section 7 + preflight ==="
python scripts/preflight_bs_seq_sc.py \
    --out "$OUT" --n-examples "$N_EXAMPLES" --smoke-steps "$SMOKE_STEPS" \
    --sigma-b "$SIGMA_B" --fampnn-variant "$FAMPNN_VARIANT" --device cuda \
    ${PREFLIGHT_EXTRA_ARGS:-}

echo "done: $OUT/preflight_bs_seq_sc.json"
