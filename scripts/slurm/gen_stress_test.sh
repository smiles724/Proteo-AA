#!/bin/bash
#SBATCH --job-name=gen_stress
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_gen_stress/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_gen_stress/%x-%j.out
#
# One-event SC -> BB feedback on target-conditioned PXDesign generations.
#
# Scale in the order the plan sets: one target x one seed x one event x three
# arms first, then 4 x 2, then the full panel. MAX_TARGETS/SEEDS/EVENTS control
# that; nothing else needs to change between stages.
#
# A generated backbone has no native counterpart, so no metric here scores it
# against the original partner. Refolding is self-consistency and is emitted per
# UNIQUE sequence, since the arms share theirs.
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

R=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot
OUT="${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_gen_stress/${TAG:-stage1}_${SLURM_JOB_ID:-local}}"
python scripts/gen_stress_test.py \
    --prepared "${PREPARED:-$ROOT/configs/gen_stress_prepared.parquet}" \
    --out "$OUT" \
    --pxdesign-donor "${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}" \
    --checkpoint "full=${FULL_CKPT:-$R/full_118215/checkpoints/final.pt}" \
    --checkpoint "bb_only=${BBONLY_CKPT:-$R/bb_only_118216/checkpoints/final.pt}" \
    --crop-size "${CROP_SIZE:-512}" \
    --pack-steps "${PACK_STEPS:-50}" \
    --n-step "${N_STEP:-200}" \
    --events ${EVENTS:-0.85} \
    --max-targets "${MAX_TARGETS:-1}" \
    --seeds ${SEEDS:-0} \
    --pool "${POOL:-pdb}" \
    ${EXTRA_ARGS:-}
echo "gen stress done -> $OUT"
