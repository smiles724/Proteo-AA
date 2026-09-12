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

# PXDesignBench launches ProteinMPNN and AF2 through a bare `python3`
# subprocess. Put the selected environment first so those subprocesses do not
# silently fall back to the system Python.
PXDBENCH_BIN="$(dirname "${PXDBENCH_PYTHON}")"
export PATH="${PXDBENCH_BIN}:${PATH}"
"${PXDBENCH_BIN}/python3" -c 'import colabdesign' || {
  echo "ERROR: colabdesign is not importable in ${PXDBENCH_BIN}" >&2
  exit 2
}

line="$(sed -n "$((TASK_INDEX + 2))p" "${TASK_FILE}")"
[[ -n "${line}" ]] || { echo "ERROR: no task ${TASK_INDEX} in ${TASK_FILE}" >&2; exit 2; }
IFS=$'\t' read -r _ model_label target sequence_arm input_dir output_dir use_gt_seq n_backbones expected_sequences <<<"${line}"

if [[ "${FORCE:-0}" != "1" && -s "${output_dir}/sample_level_output.csv" ]]; then
  echo "skip completed task=${TASK_INDEX} ${model_label}/${target}/${sequence_arm}"
  exit 0
fi

mkdir -p "${output_dir}"
export TOOL_WEIGHTS_ROOT
# This scoring path disables Protenix and uses AF2 as the independent
# structure predictor. Avoid importing/compiling Protenix's unused fused CUDA
# LayerNorm extension; it is both unnecessary and unsafe across array workers.
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-openfold}"
export USE_DEEPSPEED_EVO_ATTENTION="${USE_DEEPSPEED_EVO_ATTENTION:-false}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/hai/scratch/shenjm/triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

# AF2 runs on JAX. jaxlib is a `+cuda12.cudnn91` build and needs libcudnn.so.9,
# but the env's `nvidia-cudnn-cu12` is 8.9.2.26 (libcudnn.so.8), pinned there by
# torch 2.3.1+cu121. Without this path JAX's CUDA backend fails to initialize
# and silently falls back to CPU: single-chain targets then score very slowly
# and multi-chain targets die with "UNIMPLEMENTED: unsupported operand type
# BF16 in op dot", because AF2-multimer runs in bfloat16 and the CPU backend has
# no bf16 dot. cuDNN 9 lives in its own prefix so torch keeps its cuDNN 8.
JAX_CUDNN_LIB_DIR="${JAX_CUDNN_LIB_DIR:-/hai/scratch/shenjm/pxdbench_cudnn9/nvidia/cudnn/lib}"
[[ -f "${JAX_CUDNN_LIB_DIR}/libcudnn.so.9" ]] || {
  echo "ERROR: no libcudnn.so.9 in ${JAX_CUDNN_LIB_DIR}" >&2
  exit 2
}
export LD_LIBRARY_PATH="${JAX_CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Fail fast rather than produce CPU-fallback scores that look like real results.
"${PXDBENCH_PYTHON}" - <<'PY' || { echo "ERROR: JAX cannot use the GPU" >&2; exit 2; }
import sys
import jax, jax.numpy as jnp
backend = jax.default_backend()
if backend != "gpu":
    sys.exit(f"jax.default_backend()={backend!r}, devices={jax.devices()}")
x = jnp.ones((256, 256), dtype=jnp.bfloat16)
float((x @ x).sum())  # AF2-multimer needs a working bf16 dot
print(f"jax backend={backend} devices={jax.devices()}")
PY

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
