#!/bin/bash
#SBATCH --job-name=stage4-ligandmpnn-binder
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/stage4_ligandmpnn/%x-%j.out
#SBATCH --error=logs/training/stage4_ligandmpnn/%x-%j.err

# Stage IV with the LigandMPNN sequence backend, HAI-native.
#
#   mkdir -p logs/training/stage4_ligandmpnn
#   bash scripts/training/slurm_stage4_ligandmpnn_binder_hai.sh --dry-run   # login node
#   sbatch scripts/training/slurm_stage4_ligandmpnn_binder_hai.sh
#
# This is one file rather than the base/wrapper pair the FaMPNN launchers use:
# those are Marlowe-native with a HAI wrapper on top, and nothing here runs on
# Marlowe. Every knob is an override, e.g. STAGE4_PHASE=IV-F sbatch ...
#
# PHASES
#   IV-A  train the sequence head only, on contexts the frozen generator makes.
#   IV-F  the mirror image: FREEZE the head, train the packer and the backbone
#         subset against it. Gradient still flows through the frozen head into
#         the coordinates -- that route is the objective, and it is asserted by
#         tests/test_stage4_phase_iv_f.py, not assumed.
#   IV-B  both.
#
# MEMORY: IV-A retains no autograd graph through the frozen generator, which is
# why crop 384 fits. IV-F and IV-B open the packer and the atom-attention
# decoder and are NOT memory-proven at this crop -- start smaller and watch.
set -euo pipefail

export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/s/h/shenjm/Proteo-AA}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}

# Upstream checkout and released weights. The head verifies the revision and
# that the tree is clean, and records the weight SHA-256 in the checkpoint, so
# these are provenance inputs and not merely paths.
export LIGANDMPNN_SOURCE=${LIGANDMPNN_SOURCE:-/hai/users/s/h/shenjm/tools/LigandMPNN}
export LIGANDMPNN_CHECKPOINT=${LIGANDMPNN_CHECKPOINT:-/hai/users/s/h/shenjm/tools/ligandmpnn_weights/ligandmpnn_v_32_010_25.pt}

# Same Stage III donor as the FaMPNN runs, so the two backends differ in the
# sequence network and nothing else. Job 111408 timed out at step 6650 of
# 30000, so step6000 is its last checkpoint: the most-trained co-evolution
# binder state that exists, NOT a finished Stage III. Its own AA head is
# dropped -- Stage IV supplies the sequence network.
export STAGE3_CHECKPOINT=${STAGE3_CHECKPOINT:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/shenjm/proteo_aa_runs/stage4_ligandmpnn_binder/${SLURM_JOB_ID:-dry-run}}

STAGE4_PHASE=${STAGE4_PHASE:-IV-A}
TRAIN_ROUNDS=${TRAIN_ROUNDS:-1}
INFERENCE_ROUNDS=${INFERENCE_ROUNDS:-3}

export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
# Not `torch`: the fused CUDA LayerNorm needs ninja and a matching toolchain,
# and this path never uses it.
export LAYERNORM_TYPE=${LAYERNORM_TYPE:-openfold}
export USE_DEEPSPEED_EVO_ATTENTION=${USE_DEEPSPEED_EVO_ATTENTION:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
# LigandMPNN is imported BY PATH from the pinned checkout (see
# pxdesign_train/aa/ligandmpnn_head.py), so it is deliberately absent here: on
# PYTHONPATH its `data_utils` would shadow nothing but would drag in prody.
export PYTHONPATH=$PROTEOAA_REPO:$PROTEOAA_REPO/Protenix:$PROTEOAA_REPO/PXDesign

cd "$PROTEOAA_REPO"
mkdir -p "$OUTPUT_DIR" logs/training/stage4_ligandmpnn

[[ -f $STAGE3_CHECKPOINT ]] || { echo "ERROR: donor missing: $STAGE3_CHECKPOINT" >&2; exit 2; }
[[ -f $LIGANDMPNN_CHECKPOINT ]] || { echo "ERROR: weights missing: $LIGANDMPNN_CHECKPOINT" >&2; exit 2; }
[[ -f $LIGANDMPNN_SOURCE/model_utils.py ]] || { echo "ERROR: upstream checkout missing: $LIGANDMPNN_SOURCE" >&2; exit 2; }

RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run --device cpu)
fi

# CHECKPOINT_INTERVAL stays BELOW EVAL_INTERVAL on purpose: validation runs
# before the checkpoint save in the training loop, so a failure in the eval
# path discards everything since the last save. Jobs 113677 and 113714 both
# died before their first checkpoint and lost the whole run.
"$PYTHON_BIN" scripts/training/train_protenix_monomer.py \
  --training-stage stage4_ligandmpnn --stage4-phase "$STAGE4_PHASE" \
  --stage4-train-rounds "$TRAIN_ROUNDS" --stage4-inference-rounds "$INFERENCE_ROUNDS" \
  --load-checkpoint "$STAGE3_CHECKPOINT" --warm-start-params-only \
  --ligandmpnn-checkpoint "$LIGANDMPNN_CHECKPOINT" \
  --ligandmpnn-source "$LIGANDMPNN_SOURCE" \
  --protenix-code-dir "$PROTEOAA_REPO/Protenix" --pxdesign-code-dir "$PROTEOAA_REPO/PXDesign" \
  --data-root "$PROTEOAA_DATA_ROOT/protenix_data" --data-mode mixed_monomer_complex --complex-provider pinder \
  --pinder-root "$PROTEOAA_DATA_ROOT/pinder/2024-02" \
  --pinder-archive "$PROTEOAA_DATA_ROOT/pinder/2024-02/raw/pdbs.zip" \
  --pinder-manifest "$PROTEOAA_DATA_ROOT/pinder/2024-02/indices/pinder_ppi_complex.parquet" \
  --pinder-cif-cache "$OUTPUT_DIR/pinder_cif_cache" \
  --crop-size "${CROP_SIZE:-384}" --max-n-token "${CROP_SIZE:-384}" \
  --complex-max-n-token "${COMPLEX_MAX_N_TOKEN:-640}" \
  --diffusion-batch-size 1 --dtype bf16 \
  --stage2-start-monomer-frac 0.25 --stage2-end-monomer-frac 0.25 \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE:-8}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --max-steps "${MAX_STEPS:-30000}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-500}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-2000}" --eval-samples "${EVAL_SAMPLES:-64}" \
  --eval-num-workers 0 --num-workers "${NUM_WORKERS:-4}" \
  --output-dir "$OUTPUT_DIR" "$@" "${RUN_OPTIONS[@]}"
