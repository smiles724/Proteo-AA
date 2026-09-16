#!/bin/bash
#SBATCH --job-name=pxf_eval_protenix
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_protenix/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_protenix/%x-%j.out
#
# MARLOWE COPY of scripts/slurm/eval_protenix_sidechain.sh. The HAI original is
# left untouched -- that repo is shared and in-place edits would break it.
# Differences from the original, all forced by this cluster:
#
#   header    --partition=yejin --account=yejin --gres=gpu:h200:1
#             -> batch / marlowe-m000137-pm06 / --qos=medium / -G 1.
#             The h200 pin is dropped, not translated: Marlowe's gres is
#             UNTYPED ("gpu:8(S:0-1)"), so "gpu:h200:1" is rejected outright,
#             and all 31 nodes are the same H100 80GB (sm_90), which this
#             torch does have kernels for (arch flags sm_50..sm_90). There is
#             no heterogeneous node to dodge here, so no constraint is needed.
#             --qos=medium is REQUIRED: our association grants only `medium`,
#             and a bare submit fails with "Invalid qos specification".
#             batch is the only usable partition -- preempt sets
#             DenyQos=class,medium and hero needs qos `large`.
#   logs      /hai/scratch/... does not exist here. SBATCH directives are
#             static, so marlowe_env.sh cannot repoint them.
#   PYTHONPATH appended, not overwritten -- see below.
#   mkdir     the second /hai/scratch path is gone.
#   python    no conda; the env is activated by PATH (see PXF_PYTHON_ENV).
#
# Side-chain packing metrics on Protenix's recentPDB low-homology eval split.
# 1,818 entries released 2022-05-04 .. 2023-01-11; the training index is cut at
# 2021-09-30 and the intersection is empty, so this measures generalization
# rather than recall. 1,642 have a supervision mask and are scored.
#
# Usage (source marlowe_env.sh first -- it supplies PXF_REPO, PROTEOAA_ROOT,
# the CCD roots, MASK_ROOT and EXTRA_ARGS/--mmcif-dir):
#
#   source /users/yfsun/marlowe_env.sh
#   OUT=/scratch/m000137-pm06/Proteo-AA/pxf/runs/eval_before \
#       sbatch scripts/slurm/marlowe/eval_protenix_sidechain.sh
#
# Do NOT clear the whole SLURM_* block the way the HAI header suggests. On
# Marlowe SLURM_CONF lives in that namespace, and unsetting it breaks client
# config discovery ("Could not establish a configuration source"). A plain
# login shell here holds no allocation, so there is nothing to shed; if you do
# submit from inside an salloc, clear only the job vars and keep SLURM_CONF:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') OUT=... sbatch ...
set -euo pipefail

# sbatch copies this script elsewhere, so BASH_SOURCE does not point at the
# repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
WEIGHTS="${WEIGHTS:-0.0}"
CHECKPOINT="${CHECKPOINT:-}"
MASK_ROOT="${MASK_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_sidechain/out_eval_fampnn_strictB}"
NUM_STEPS="${NUM_STEPS:-50}"
MAX_TARGETS="${MAX_TARGETS:-0}"
LABEL="${LABEL:-recentPDB_low_homology}"
mkdir -p "$OUT"
cd "$ROOT"

# The env is a relocated conda prefix with no bin/activate and no conda
# install behind it, so it is activated by PATH. Verified sufficient: torch
# 2.7.1+cu126 finds its CUDA runtime through the nvidia-*-cu12 wheels, not
# LD_LIBRARY_PATH. Deliberately NOT loading nvhpc/cudnn modules -- this torch
# ships its own cudnn 9.5.1.17 and a system cudnn on the path can shadow it.
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"

# APPEND. marlowe_env.sh has already put the masks dir and afdb-laproteina/src
# on PYTHONPATH; SideChainMaskSet imports sc_masks by module name and sc_masks
# pulls in afdb_laproteina.constants, so overwriting here would break the run.
# PXDesign is listed explicitly (the HAI original omits it).
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "weights=${WEIGHTS} checkpoint=${CHECKPOINT:-<released>} masks=${MASK_ROOT}"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

CKPT_ARG=""
if [ -n "$CHECKPOINT" ]; then CKPT_ARG="--checkpoint $CHECKPOINT"; fi

python scripts/eval_protenix_sidechain.py \
    --out "$OUT" \
    --label "$LABEL" \
    --weights "$WEIGHTS" \
    ${CKPT_ARG} \
    --mask-root "$MASK_ROOT" \
    --num-steps "$NUM_STEPS" \
    --max-targets "$MAX_TARGETS" \
    ${EXTRA_ARGS:-}

echo "done -> $OUT/sidechain_metrics.json"
