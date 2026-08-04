#!/bin/bash
# Demo invocation for PXDesign-d training.
#
# This script is a TEMPLATE — it does not run training out of the box. You
# need to:
#   1. Write a Python driver that builds your ComplexProviders. Each provider
#      yields one real protein complex per __getitem__ (see
#      `pxdesign_train/runner/data.py` for the protocol).
#   2. Wrap each provider in a DesignSourceDataset.
#   3. Construct CurriculumMultiDataset, CurriculumSchedule, TrainerComponents.
#   4. Call `train_from_components()`.
#
# A worked example with synthetic data lives in
# `tests/test_trainer_integration.py`.

set -euo pipefail

# Where Protenix and PXDesign live on this machine. Both must be on PYTHONPATH
# because we re-use their model components.
export PXDESIGN_TRAIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export REPO_ROOT="$(cd "$PXDESIGN_TRAIN_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/Protenix:$REPO_ROOT/PXDesign:$PXDESIGN_TRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Protenix's fused LayerNorm needs ninja + a CUDA toolchain. Set to "torch"
# to fall back to plain `torch.nn.LayerNorm`. Use the fused kernels in
# production (faster + lower memory).
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

# Run YOUR driver here. Example:
#   python my_train_driver.py \
#       --output_dir ./runs/$(date +%Y%m%d_%H%M%S) \
#       --pdb_root  /path/to/weightedPDB \
#       --afdb_root /path/to/afdb_monomers \
#       --max_steps 100000 \
#       --crop_size 640 \
#       --batch_size 64 \
#       --lr 5e-4
#
# The driver wires data providers + curriculum and then calls
# `pxdesign_train.runner.train.train_from_components`.

echo "This is a template. Write your driver per the comments above."
echo "Working integration example: PXDesign-train/tests/test_trainer_integration.py"
exit 1
