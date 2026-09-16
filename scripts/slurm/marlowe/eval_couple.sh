#!/bin/bash
#SBATCH --job-name=pxf_eval_couple
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_couple/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_couple/%x-%j.out
#
# MARLOWE COPY of scripts/slurm/eval_couple.sh (repo commit 88ae9a7). The HAI
# original is left untouched -- that repo is shared and in-place edits break it.
#
# Two measurements, selected by MODE, submitted with --job-name/--time overrides:
#   MODE=native     GT backbone -> FaMPNN            (pxf_eval_native,   2h)
#   MODE=denoised   PXDesign backbone -> FaMPNN      (pxf_eval_denoised, 6h)
# The header defaults to the longer 6h; sbatch --time wins when passed.
#
# Differences from the HAI original, all forced by this cluster:
#   header    yejin/yejin/gpu:h200:1 -> batch / marlowe-m000137-pm06 /
#             --qos=medium / -G 1. Marlowe's gres is UNTYPED ("gpu:8(S:0-1)")
#             so "gpu:h200:1" is rejected outright; all 31 nodes are the same
#             H100 80GB (sm_90), which this torch has kernels for. --qos=medium
#             is required (our association grants only `medium`) and batch is
#             the only partition allowing it (preempt sets DenyQos=class,medium;
#             hero needs `large`).
#   logs      /hai/scratch/... does not exist here and SBATCH paths are static.
#   PYTHONPATH appended, not overwritten -- marlowe_env.sh already put the mask
#             dir and afdb-laproteina/src there and they must survive.
#   mkdir     the second /hai/scratch path is gone.
#   python    no conda; the env is a relocated conda prefix with no
#             bin/activate, activated by PATH (see PXF_PYTHON_ENV).
#   DONOR / PROTENIX_*_DIR defaults repointed off /hai.
#   STRUCTURES defaults to the REMAPPED val manifest; the original
#             configs/val_structures_afdb.txt lists /hai paths and
#             resolve_structures aborts on the first missing one.
#
#   EXTRA_ARGS -> EVAL_COUPLE_EXTRA_ARGS. marlowe_env.sh exports
#             EXTRA_ARGS="--mmcif-dir ..." for the *protenix side-chain* eval;
#             eval_couple.py has no --mmcif-dir and argparse exits 2 on it.
#             This is the same trap that killed job 488735. Verified against
#             `eval_couple.py --help`: one shared variable cannot serve both.
#
# The val manifest reaches 485 residues, so CROP_SIZE cannot go below that -- a
# crop breaks correspondence with the atom37 side-chain targets and the script
# refuses. Note lowering it does NOT reduce memory: the featurizer emits the
# structure's true token count, so cost follows the structure, not the crop.
#
# Held-out by construction: 2,000 phase-1 train ids vs 256 val ids, intersection 0.
#
# Omit CHECKPOINT to evaluate the untrained adapters. That is the pipeline's own
# sanity check, not a wasted run: the adapters are zero-initialized, so the two
# arms must come out bit-identical and any difference is a seeding/arm-switching
# bug rather than a result.
#
# Usage:
#   source /users/yfsun/marlowe_env.sh
#   MODE=native OUT=.../eval_native \
#     sbatch --job-name=pxf_eval_native --time=02:00:00 \
#            scripts/slurm/marlowe/eval_couple.sh
#   MODE=denoised CHECKPOINT=.../couple_phase1/checkpoints/final.pt \
#     OUT=.../eval_denoised \
#     sbatch --job-name=pxf_eval_denoised --time=06:00:00 \
#            scripts/slurm/marlowe/eval_couple.sh
#
# Do NOT clear the whole SLURM_* block the way the HAI header suggests: on
# Marlowe SLURM_CONF lives in that namespace and unsetting it breaks client
# config discovery. A login shell here holds no allocation, so nothing needs
# shedding; from inside an salloc keep SLURM_CONF:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') MODE=... sbatch ...
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
# denoised: PXDesign proposal -> FaMPNN, coupled vs uncoupled arms.
# native:   deposited backbone -> FaMPNN. No donor, no sigma, no adapters.
MODE="${MODE:-denoised}"
CHECKPOINT="${CHECKPOINT:-}"
# EVAL_STRUCTURES, *not* STRUCTURES. marlowe_env.sh exports
# STRUCTURES=configs/phase1_structures_afdb.marlowe.txt for the *training* job,
# and inheriting it here silently evaluates the coupling on its own training
# data -- a wrong number rather than a crash. Caught live on job 488894, which
# logged structures=...phase1... and was cancelled. Override with
# EVAL_STRUCTURES if you really mean a different set.
STRUCTURES="${EVAL_STRUCTURES:-$ROOT/configs/val_structures_afdb.marlowe.txt}"
CONFIG="${CONFIG:-$ROOT/configs/couple_phase1.yaml}"
DONOR="${DONOR:-/scratch/m000137-pm06/Proteo-AA/pxf/component_donors/pxdesign_v0.1.0.pt}"
CROP_SIZE="${CROP_SIZE:-512}"
PACK_STEPS="${PACK_STEPS:-50}"
N_SIGMA="${N_SIGMA:-5}"
MAX_TARGETS="${MAX_TARGETS:-200}"
FAMPNN_WEIGHTS="${FAMPNN_WEIGHTS:-0.0}"
SEED="${SEED:-0}"
# CODESIGN=1 co-generates sequence and side chains (MPNN -> s_hat -> side-chain
# diffusion) instead of teacher-forcing the deposited sequence. Side-chain
# geometry is then scored only where s_hat matches the native identity, and
# sequence recovery is reported alongside.
CODESIGN="${CODESIGN:-0}"
SEQ_TEMPERATURE="${SEQ_TEMPERATURE:-0.0}"
CODESIGN_ARGS=""
if [ "$CODESIGN" = "1" ]; then
    CODESIGN_ARGS="--codesign --seq-temperature $SEQ_TEMPERATURE"
fi
mkdir -p "$OUT"
cd "$ROOT"

# Held-out guard: enforce the disjointness the measurement depends on rather
# than assume it. Refuses rather than reporting a contaminated number.
TRAIN_MANIFEST="$ROOT/configs/phase1_structures_afdb.marlowe.txt"
if [ -f "$TRAIN_MANIFEST" ]; then
    "${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}/bin/python" - "$STRUCTURES" "$TRAIN_MANIFEST" <<'PYGUARD'
import pathlib, sys
def ids(p):
    return {pathlib.Path(l).name for l in pathlib.Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")}
ev, tr = ids(sys.argv[1]), ids(sys.argv[2])
both = ev & tr
if both:
    sys.exit(
        f"REFUSING: {len(both)} of {len(ev)} eval structures are in the phase-1 "
        f"training manifest (e.g. {sorted(both)[:3]}). This would not be a "
        "held-out measurement. Set EVAL_STRUCTURES to a disjoint set."
    )
print(f"held-out guard: {len(ev)} eval ids, 0 overlap with {len(tr)} training ids")
PYGUARD
fi

# Relocated conda prefix, no bin/activate and no conda install behind it, so it
# is activated by PATH. torch 2.7.1+cu126 finds its CUDA runtime through the
# nvidia-*-cu12 wheels. Deliberately NOT loading nvhpc/cudnn modules -- this
# torch ships its own cudnn 9.5.1.17 and a system one can shadow it.
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"

export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} mode=${MODE} out=${OUT}"
echo "checkpoint=${CHECKPOINT:-<untrained adapters>} structures=${STRUCTURES}"
echo "crop=${CROP_SIZE} pack_steps=${PACK_STEPS} n_sigma=${N_SIGMA} targets=${MAX_TARGETS}"
echo "codesign=${CODESIGN} seq_temperature=${SEQ_TEMPERATURE}"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

if [ "$MODE" = "native" ]; then
    # No backbone is generated, so the donor, the crop and the sigma sweep have
    # nothing to act on; passing them would only imply they were used.
    python scripts/eval_couple.py \
        --mode native \
        --structures "$STRUCTURES" \
        --out "$OUT" \
        --pack-steps "$PACK_STEPS" \
        --max-targets "$MAX_TARGETS" \
        --fampnn-weights "$FAMPNN_WEIGHTS" \
        --seed "$SEED" \
        ${CODESIGN_ARGS} \
        ${EVAL_COUPLE_EXTRA_ARGS:-}
else
    CKPT_ARG=""
    if [ -n "$CHECKPOINT" ]; then CKPT_ARG="--checkpoint $CHECKPOINT"; fi
    python scripts/eval_couple.py \
        --mode denoised \
        --structures "$STRUCTURES" \
        --out "$OUT" \
        --config "$CONFIG" \
        --pxdesign-donor "$DONOR" \
        ${CKPT_ARG} \
        --crop-size "$CROP_SIZE" \
        --pack-steps "$PACK_STEPS" \
        --n-sigma "$N_SIGMA" \
        --max-targets "$MAX_TARGETS" \
        --fampnn-weights "$FAMPNN_WEIGHTS" \
        --seed "$SEED" \
        ${CODESIGN_ARGS} \
        ${EVAL_COUPLE_EXTRA_ARGS:-}
fi

echo "done -> $OUT/couple_metrics.json"
