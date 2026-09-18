#!/bin/bash
#SBATCH --job-name=sb_audit
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/%x-%j.out
#
# Experiment 1: trace where side-chain information stops reaching the backbone
# coordinates. Four stages -- h_res, A_SB output, coordinates, quality -- with
# the first failing stage being the only one worth fixing.
#
# The perturbed arm RE-ENCODES from scratch. Substituting rotated coordinates
# into the packed state while keeping h_packed leaves the 128-dim node group
# unperturbed, which understates the intervention badly.
set -euo pipefail
ROOT="${PXF_REPO:-${SLURM_SUBMIT_DIR:?set PXF_REPO}}"
cd "$ROOT"
source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the full-variant A_SB}"
OUT="${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/audit_${SLURM_JOB_ID:-local}}"
python scripts/audit_sb_sensitivity.py \
    --structures "${STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}" \
    --out "$OUT" \
    --pxdesign-donor "${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}" \
    --checkpoint "$CHECKPOINT" \
    --crop-size "${CROP_SIZE:-512}" \
    --pack-steps "${PACK_STEPS:-50}" \
    --sigmas ${SIGMAS:-0.847 1.939} \
    --max-targets "${MAX_TARGETS:-16}" \
    --perturb-degrees "${PERTURB_DEGREES:-60}" \
    --seed "${SEED:-0}" \
    ${EXTRA_ARGS:-}
echo "audit done -> $OUT"
