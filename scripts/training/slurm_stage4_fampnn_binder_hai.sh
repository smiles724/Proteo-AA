#!/bin/bash
#SBATCH --job-name=stage4-fampnn-binder
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/stage4_fampnn/%x-%j.out
#SBATCH --error=logs/training/stage4_fampnn/%x-%j.err

# HAI-cluster bindings for Stage IV binder training. The preflight, provenance
# and full argument list live in `slurm_stage4_fampnn_binder.sh`; this file
# supplies this cluster's Slurm directives, roots, interpreter, Stage III donor
# and a measured-run scale in place of that launcher's 100-step smoke defaults.
#
#   mkdir -p logs/training/stage4_fampnn
#   bash scripts/training/slurm_stage4_fampnn_binder_hai.sh --dry-run   # login node
#   sbatch scripts/training/slurm_stage4_fampnn_binder_hai.sh
#
# The migration starts with SC adaptation and both pretrained networks frozen.
# Phase transitions require WARM_START_CHECKPOINT and explicit revision/feedback
# controls; RESUME_CHECKPOINT restores the recorded phase and optimizer state.
# Memory at production crop sizes remains to be measured for each phase.
set -euo pipefail

export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-official-pxdesign-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/hai/users/y/f/yfsun/Protein Project}
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}

# Stage III supplies sidechain_module only. Backbone and feedback have separate origins.
export BACKBONE_CHECKPOINT=${BACKBONE_CHECKPOINT:-$PROTEOAA_REPO/runs/component_donors/pxdesign_v0.1.0.pt}
export SC_CHECKPOINT=${SC_CHECKPOINT:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage4_fampnn_binder/${SLURM_JOB_ID:-dry-run}}

export STAGE4_PHASE=${STAGE4_PHASE:-sc_adapt}
export TRAIN_ROUNDS=${TRAIN_ROUNDS:-0}
export INFERENCE_ROUNDS=${INFERENCE_ROUNDS:-0}
export CROP_SIZE=${CROP_SIZE:-384}
export COMPLEX_MAX_N_TOKEN=${COMPLEX_MAX_N_TOKEN:-640}
export MAX_STEPS=${MAX_STEPS:-30000}
export EVAL_SAMPLES=${EVAL_SAMPLES:-64}

# The base launcher consumes --dry-run only in the first position. Keep it
# there while placing cluster defaults before explicit CLI overrides.
RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run)
fi
# Accumulation and clipping follow the Stage III binder run by default.
# CHECKPOINT_INTERVAL defaults below EVAL_INTERVAL because validation runs
# before checkpoint saving; an earlier save bounds work lost on failure.
# 500, not 1000: IV-A measures 5.7 s/step here, so 1000 is ~95 min unsaved,
# and jobs 113677/113714 both died on a bad crop at steps 800 and 150 with
# nothing on disk. At ~3.1 GiB a checkpoint and ~15k steps per 24h slot that
# is ~93 GiB per run, against 2.9 TiB free on scratch.
exec bash "$PROTEOAA_REPO/scripts/training/slurm_stage4_fampnn_binder.sh" "${RUN_OPTIONS[@]}" \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE:-8}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-500}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-2000}" \
  --num-workers "${NUM_WORKERS:-4}" "$@"
