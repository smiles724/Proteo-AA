#!/bin/bash
# Demo invocation for PXDesign-d *fine-tuning*.
#
# Fine-tuning is just training started from an existing checkpoint with
# `load_strict=False` and a much lower LR. The Python driver below shows the
# pattern; copy it, adjust to your data / checkpoint paths, then run.

set -euo pipefail

export PXDESIGN_TRAIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export REPO_ROOT="$(cd "$PXDESIGN_TRAIN_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/Protenix:$REPO_ROOT/PXDesign:$PXDESIGN_TRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"   # use fused kernels in prod

# Where to load the warm-start checkpoint from. Two reasonable choices:
#   - PXDesign-d release: ${RELEASE_DATA}/checkpoint/pxdesign_v0.1.0.pt
#   - Protenix base:      ${PROTENIX_ROOT}/release_data/checkpoint/protenix_base_default_v1.0.0.pt
# (The first is more aligned with the report; the second is a shortcut.)
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-/path/to/pxdesign_v0.1.0.pt}"

# Run YOUR driver. A working example outline:
cat <<'PYDRIVER' >&2
# my_finetune_driver.py
from protenix.config.config import parse_configs
from protenix.data.pipeline.dataset import BaseSingleDataset

from pxdesign_train.configs.configs_train import training_configs
from pxdesign_train.data import (
    CurriculumMultiDataset,
    CurriculumSchedule,
)
from pxdesign_train.runner import (
    DesignSourceDataset,
    ProtenixComplexProvider,
    TrainerComponents,
    finetune_from_components,
    make_finetune_configs,
    select_protenix_chain_2,
)

configs = parse_configs(training_configs, arg_str='')
configs = make_finetune_configs(configs, lr=1e-5, warmup_steps=200, max_steps=5_000)

# Build a Protenix BaseSingleDataset that DOES NOT crop (crop_size=0). Our
# DesignCropper handles the design-aware crop.
base = BaseSingleDataset(
    name='your_finetune_set',
    indices_fpath='/path/to/your/indices.csv',
    bioassembly_dict_dir='/path/to/bioassembly_pkls',
    cropping_configs={'crop_size': 0, 'method_weights': [1.0, 0.0, 0.0]},
    # ... (other required BaseSingleDataset args from your data_configs)
)
provider = ProtenixComplexProvider(base, binder_selector_fn=select_protenix_chain_2())
src = DesignSourceDataset(provider, source_name='pdb', crop_size=640)

multi = CurriculumMultiDataset(
    datasets=[src],
    source_names=['pdb'],
    per_item_weights=[[1.0] * len(src)],  # or use Protenix's get_weighted_pdb_weight
)
schedule = CurriculumSchedule(
    stage1={'pdb': 1.0},
    stage2={'pdb': 1.0},
    stage1_end_step=0,
    stage2_start_step=0,
)
components = TrainerComponents(
    train_dataset=multi, schedule=schedule, train_samples_per_epoch=1_000,
)

finetune_from_components(
    configs, components,
    load_checkpoint_path='/path/to/pxdesign_v0.1.0.pt',
    checkpoint_dir='./finetune_ckpts',
)
PYDRIVER

echo "Demo driver outline printed above. Adapt and save as my_finetune_driver.py, then run:"
echo "    python my_finetune_driver.py"
exit 1
