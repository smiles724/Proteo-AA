#!/bin/bash
#SBATCH --job-name=alpha10-af2ig
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/alphaproteo10/%x-%A_%a.out
#SBATCH --error=logs/validation/alphaproteo10/%x-%A_%a.err

# One Slurm-array element evaluates one model/target/sequence-arm directory.
# This must run in a separate PXDesignBench environment based on Protenix 0.5.

set -euo pipefail

TASK_FILE="${TASK_FILE:?set TASK_FILE from prepare_alphaproteo_score_tasks.py}"
PXDBENCH_DIR="${PXDBENCH_DIR:?set PXDBENCH_DIR to PXDesignBench v0.1.2}"
PXDBENCH_PYTHON="${PXDBENCH_PYTHON:?set PXDBENCH_PYTHON to the pxdbench environment python}"
TOOL_WEIGHTS_ROOT="${TOOL_WEIGHTS_ROOT:?set TOOL_WEIGHTS_ROOT}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}"

[[ -f "${TASK_FILE}" ]] || { echo "ERROR: missing ${TASK_FILE}" >&2; exit 2; }
[[ -f "${PXDBENCH_DIR}/pxdbench/run.py" ]] || { echo "ERROR: invalid PXDBENCH_DIR" >&2; exit 2; }
[[ -x "${PXDBENCH_PYTHON}" ]] || { echo "ERROR: invalid PXDBENCH_PYTHON" >&2; exit 2; }
[[ -f "${TOOL_WEIGHTS_ROOT}/af2/params_model_1.npz" ]] || { echo "ERROR: AF2 weights missing" >&2; exit 2; }
[[ -d "${TOOL_WEIGHTS_ROOT}/mpnn/vanilla_model_weights" ]] || { echo "ERROR: ProteinMPNN weights missing" >&2; exit 2; }

line="$(sed -n "$((TASK_INDEX + 2))p" "${TASK_FILE}")"
[[ -n "${line}" ]] || { echo "ERROR: no task ${TASK_INDEX} in ${TASK_FILE}" >&2; exit 2; }
IFS=$'\t' read -r _ model_label target sequence_arm input_dir output_dir use_gt_seq n_backbones expected_sequences <<<"${line}"

if [[ "${FORCE:-0}" != "1" && -s "${output_dir}/sample_level_output.csv" ]]; then
  echo "skip completed task=${TASK_INDEX} ${model_label}/${target}/${sequence_arm}"
  exit 0
fi

mkdir -p "${output_dir}"
export TOOL_WEIGHTS_ROOT
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export USE_DEEPSPEED_EVO_ATTENTION="${USE_DEEPSPEED_EVO_ATTENTION:-false}"

echo "task=${TASK_INDEX} model=${model_label} target=${target} arm=${sequence_arm}"
echo "input=${input_dir} expected_backbones=${n_backbones} expected_sequences=${expected_sequences}"

"${PXDBENCH_PYTHON}" "${PXDBENCH_DIR}/pxdbench/run.py" \
  --data_dir "${input_dir}" \
  --dump_dir "${output_dir}" \
  --is_mmcif true \
  --seed "${SCORE_SEED:-2025}" \
  --binder_chains Z \
  --binder.num_seqs "${MPNN_SEQUENCES:-1}" \
  --binder.tools.mpnn.temperature "${MPNN_TEMPERATURE:-0.0001}" \
  --binder.tools.af2.use_initial_guess true \
  --binder.tools.af2.use_binder_template true \
  --binder.eval_complex true \
  --binder.eval_binder_monomer true \
  --binder.eval_protenix_mini false \
  --binder.eval_protenix false \
  --binder.eval_diversity false \
  --binder.use_gt_seq "${use_gt_seq}"
