#!/bin/bash
#SBATCH --job-name=pxf_couple
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_couple/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_couple/%x-%j.out
#
# MARLOWE COPY of scripts/slurm/train_couple.sh. The HAI original is left
# untouched -- that repo is shared and in-place edits would break it.
# Differences from the original, all forced by this cluster:
#
#   header    --partition=yejin --account=yejin --gres=gpu:h200:1
#             -> batch / marlowe-m000137-pm06 / --qos=medium / -G 1.
#             The h200 pin is dropped, not translated: Marlowe's gres is
#             UNTYPED ("gpu:8(S:0-1)"), so "gpu:h200:1" is rejected outright,
#             and all 31 nodes are the same H100 80GB (sm_90), which this
#             torch does have kernels for (arch flags sm_50..sm_90).
#             --qos=medium is REQUIRED -- our association grants only
#             `medium`, and batch is the only partition that allows it
#             (preempt sets DenyQos=class,medium; hero needs `large`).
#   logs      /hai/scratch/... does not exist here, and SBATCH directives are
#             static so marlowe_env.sh cannot repoint them.
#   PYTHONPATH appended, not overwritten -- see below.
#   defaults  DONOR / PROTENIX_*_DIR / OUT repointed off /hai.
#   python    no conda; the env is activated by PATH (see PXF_PYTHON_ENV).
#
# NOTE ON MEMORY: HAI's H200 has 141GB, Marlowe's H100 has 80GB. CROP_SIZE
# cannot be lowered to compensate -- the AFDB manifest reaches 510 residues
# and a crop breaks correspondence with the atom37 side-chain targets, so the
# script refuses anything under 510. If this OOMs, the lever is batch/
# accumulation or activation checkpointing in configs/couple_phase1.yaml, not
# the crop.
#
# Train the PXDesign <-> FaMPNN coupling adapters. Both donors stay frozen and
# the adapters are zero-initialized, so step 0 reproduces the two pretrained
# models exactly and any gain is attributable to the coupling.
#
#   PHASE=1  A_BS on L_SC
#   PHASE=2  A_SB on L_BB, packing detached
#   PHASE=3  both, alternating one objective per step
#
# STRUCTURES is required -- no data source is defaulted anywhere in this
# pipeline. It must be the REMAPPED manifest; the original
# configs/phase1_structures_afdb.txt lists /hai paths and resolve_structures
# aborts on the first missing one. marlowe_env.sh already points it at
# configs/phase1_structures_afdb.marlowe.txt.
#
# Usage:
#   source /users/yfsun/marlowe_env.sh
#   PHASE=1 OUT=/scratch/m000137-pm06/Proteo-AA/pxf/runs/couple_phase1 \
#       sbatch scripts/slurm/marlowe/train_couple.sh
#
# Do NOT clear the whole SLURM_* block the way the HAI header suggests. On
# Marlowe SLURM_CONF lives in that namespace and unsetting it breaks client
# config discovery. A plain login shell holds no allocation, so there is
# nothing to shed; from inside an salloc, keep SLURM_CONF:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') PHASE=1 sbatch ...
#
# Noise range:
#   SIGMA_MODE=trajectory|loguniform|fixed  (default trajectory)
#   SIGMA_MIN / SIGMA_MAX                   move the coupling window, Angstroms
#   SIGMA                                   only with SIGMA_MODE=fixed
#
# Phases chain by dependency, each warm-starting from the previous:
#   J1=$(PHASE=1 sbatch --parsable scripts/slurm/marlowe/train_couple.sh)
#   J2=$(PHASE=2 RESUME=<phase1 final.pt> sbatch --parsable \
#            --dependency=afterok:$J1 scripts/slurm/marlowe/train_couple.sh)
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

PHASE="${PHASE:?set PHASE to 1, 2 or 3}"
STRUCTURES="${STRUCTURES:?set STRUCTURES to the remapped .marlowe.txt manifest (no default)}"
BACKBONE="${BACKBONE:-pxdesign}"
DONOR="${DONOR:-/scratch/m000137-pm06/Proteo-AA/pxf/component_donors/pxdesign_v0.1.0.pt}"
OUT="${OUT:-/scratch/m000137-pm06/Proteo-AA/pxf/runs/couple_phase${PHASE}_${SLURM_JOB_ID:-local}}"
CROP_SIZE="${CROP_SIZE:-384}"
# sigma_B is sampled from the late end of PXDesign's trajectory, not fixed: both
# adapters are conditioned on log sigma_B, so one training value would only
# license deployment at that value. Override the window, not a single sigma.
SIGMA_MODE="${SIGMA_MODE:-trajectory}"
SIGMA_MIN="${SIGMA_MIN:-}"
SIGMA_MAX="${SIGMA_MAX:-}"
SIGMA="${SIGMA:-}"   # only read when SIGMA_MODE=fixed
FAMPNN_WEIGHTS="${FAMPNN_WEIGHTS:-0.0}"
mkdir -p "$OUT"
cd "$ROOT"

# Relocated conda prefix with no bin/activate and no conda install behind it,
# so it is activated by PATH. torch 2.7.1+cu126 finds its CUDA runtime through
# the nvidia-*-cu12 wheels. Deliberately NOT loading nvhpc/cudnn modules --
# this torch ships its own cudnn 9.5.1.17 and a system one can shadow it.
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"

# APPEND -- marlowe_env.sh has already put the masks dir and
# afdb-laproteina/src on PYTHONPATH and they must survive.
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} phase=${PHASE} backbone=${BACKBONE}"
echo "structures=${STRUCTURES} crop=${CROP_SIZE} sigma_mode=${SIGMA_MODE} out=${OUT}"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list()[-2:])"

# Resume this phase's own run if it was interrupted; RESUME warm-starts from a
# previous phase instead.
RESUME_ARG=""
LATEST=$(ls -1 "$OUT"/checkpoints/step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming interrupted run from $LATEST"
    RESUME_ARG="--resume $LATEST"
elif [ -n "${RESUME:-}" ]; then
    echo "warm-starting from $RESUME"
    RESUME_ARG="--resume $RESUME"
fi

DONOR_ARG=""
if [ "$BACKBONE" = "pxdesign" ]; then DONOR_ARG="--pxdesign-donor $DONOR"; fi

# NOTE: this forwards COUPLE_EXTRA_ARGS, not EXTRA_ARGS. marlowe_env.sh exports
# EXTRA_ARGS="--mmcif-dir ..." for the EVAL job -- that flag is the seam around
# pxf/train/protenix.py:DEFAULT_MMCIF_DIR and exists only on
# eval_protenix_sidechain.py. train_couple.py has no --mmcif-dir and argparse
# exits 2 on it, which is exactly how job 488735 died 10s in. The HAI original
# forwards EXTRA_ARGS here safely only because nothing sets it globally there;
# on Marlowe one shared variable cannot serve both jobs, so they get separate
# names. Pass extra train_couple.py flags as COUPLE_EXTRA_ARGS.

# Unset flags are omitted entirely so the config's own sigma block stays in
# charge; the script refuses --sigma outside fixed mode rather than ignoring it.
SIGMA_ARGS="--sigma-mode $SIGMA_MODE"
if [ -n "$SIGMA_MIN" ]; then SIGMA_ARGS="$SIGMA_ARGS --sigma-min $SIGMA_MIN"; fi
if [ -n "$SIGMA_MAX" ]; then SIGMA_ARGS="$SIGMA_ARGS --sigma-max $SIGMA_MAX"; fi
if [ -n "$SIGMA" ]; then
    if [ "$SIGMA_MODE" != "fixed" ]; then
        echo "SIGMA=$SIGMA is only read when SIGMA_MODE=fixed (got $SIGMA_MODE);" \
             "set SIGMA_MIN/SIGMA_MAX to move the sampling window" >&2
        exit 2
    fi
    SIGMA_ARGS="$SIGMA_ARGS --sigma $SIGMA"
fi

python scripts/train_couple.py \
    --config "configs/couple_phase${PHASE}.yaml" \
    --structures "$STRUCTURES" \
    --out "$OUT" \
    --backbone "$BACKBONE" \
    ${DONOR_ARG} \
    --crop-size "$CROP_SIZE" \
    ${SIGMA_ARGS} \
    --fampnn-weights "$FAMPNN_WEIGHTS" \
    ${RESUME_ARG} \
    ${COUPLE_EXTRA_ARGS:-}

echo "phase ${PHASE} done -> $OUT"
