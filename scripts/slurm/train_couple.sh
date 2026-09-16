#!/bin/bash
#SBATCH --job-name=pxf_couple
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_couple/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_couple/%x-%j.out
#
# Train the PXDesign <-> FaMPNN coupling adapters. Both donors stay frozen and
# the adapters are zero-initialized, so step 0 reproduces the two pretrained
# models exactly and any gain is attributable to the coupling.
#
#   PHASE=1  A_BS on L_SC
#   PHASE=2  A_SB on L_BB, packing detached
#   PHASE=3  both, alternating one objective per step
#
# STRUCTURES is required -- no data source is defaulted anywhere in this pipeline.
# For --backbone pxdesign it must hold .cif files, and CROP_SIZE must be at least
# the longest structure, because a crop breaks correspondence with the atom37
# side-chain targets (the script checks and refuses).
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       PHASE=1 STRUCTURES=/path/to/cifs sbatch scripts/slurm/train_couple.sh
#
# Noise range:
#   SIGMA_MODE=trajectory|loguniform|fixed  (default trajectory)
#   SIGMA_MIN / SIGMA_MAX                   move the coupling window, Angstroms
#   SIGMA                                   only with SIGMA_MODE=fixed
#
# Phases chain by dependency, each warm-starting from the previous:
#   J1=$(... PHASE=1 sbatch --parsable scripts/slurm/train_couple.sh)
#   J2=$(... PHASE=2 RESUME=<phase1 final.pt> sbatch --parsable \
#            --dependency=afterok:$J1 scripts/slurm/train_couple.sh)
set -euo pipefail

# sbatch copies this script to /var/lib/slurm/scripts, so BASH_SOURCE does not
# point at the repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

PHASE="${PHASE:?set PHASE to 1, 2 or 3}"
STRUCTURES="${STRUCTURES:?set STRUCTURES to a directory of .cif files (no default)}"
BACKBONE="${BACKBONE:-pxdesign}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
OUT="${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_couple/phase${PHASE}_${SLURM_JOB_ID}}"
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

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"

echo "node=$(hostname) job=${SLURM_JOB_ID} phase=${PHASE} backbone=${BACKBONE}"
echo "structures=${STRUCTURES} crop=${CROP_SIZE} sigma_mode=${SIGMA_MODE} out=${OUT}"
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
    ${EXTRA_ARGS:-}

echo "phase ${PHASE} done -> $OUT"
