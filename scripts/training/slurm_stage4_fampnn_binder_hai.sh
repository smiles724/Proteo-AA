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
# STAGE4_PHASE=IV-A trains only the FaMPNN sequence network, on contexts the
# frozen backbone and packer generate. With the backbone, packer and feedback
# modules all frozen, no autograd graph is retained through them, which is why
# a 384 crop is affordable here even though Stage III needed 512 to be a 2x
# Stage II memory bet. IV-B/IV-C open the packer and the atom-attention decoder
# and have NOT been memory-proven at this crop -- re-prove it before switching.
#
# Every knob below is an override, e.g. STAGE4_PHASE=IV-B CROP_SIZE=256 sbatch ...
set -euo pipefail

export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-stage4-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/hai/users/y/f/yfsun/Protein Project}
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}

# The intended complete Stage III co-evolution donor. Job 111408 (crop 448,
# mixed monomer/PINDER, warm-started from Stage II step52500 + AA head step9000)
# timed out at step 6650 of 30000, so step6000 is its last checkpoint -- it is
# the most-trained co-evolution binder state that exists, not a finished Stage
# III. It carries what Stage IV needs from a donor: the backbone, the one-step
# global-coordinate packer (edm=false), and the feedback modules. Its AA head is
# irrelevant here; Stage IV drops it for FaMPNN. See
# ../Proteo-AA-sjm-binder/docs/stage3_binder_run_111408_report.md.
export STAGE3_CHECKPOINT=${STAGE3_CHECKPOINT:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage4_fampnn_binder/${SLURM_JOB_ID:-dry-run}}

export STAGE4_PHASE=${STAGE4_PHASE:-IV-A}
export TRAIN_ROUNDS=${TRAIN_ROUNDS:-1}
export INFERENCE_ROUNDS=${INFERENCE_ROUNDS:-3}
export CROP_SIZE=${CROP_SIZE:-384}
export COMPLEX_MAX_N_TOKEN=${COMPLEX_MAX_N_TOKEN:-640}
export MAX_STEPS=${MAX_STEPS:-30000}
export EVAL_SAMPLES=${EVAL_SAMPLES:-64}

# Appended last, so these win over the base launcher's smoke-scale values for
# the same options. Accumulation and clipping follow the Stage III binder run.
#
# CHECKPOINT_INTERVAL stays BELOW EVAL_INTERVAL on purpose. `run()` evaluates
# before it saves, so anything that raises in validation discards every step
# since the last checkpoint -- which, at equal intervals, is all of them.
exec bash "$PROTEOAA_REPO/scripts/training/slurm_stage4_fampnn_binder.sh" "$@" \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE:-8}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-1000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-2000}" \
  --num-workers "${NUM_WORKERS:-4}"
