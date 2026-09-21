#!/bin/bash
#SBATCH --job-name=pxd_poscontrol
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxd_official/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxd_official/%x-%j.out
#
# Audit section 7's positive control, re-run on MARLOWE in the freshly built
# official env. Released checkpoint, no FaMPNN, no adapters, no refolding.
# The gate is 12/12 interfaces clash-free; a clean import proves nothing.
set -uo pipefail
PX=/users/yfsun/pxdesign_pristine
ENV=/users/yfsun/.venvs/pxdesign_official
DATA=/scratch/m000137-pm06/Proteo-AA/pxf/official_release_data
OUT=/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench/poscontrol_pdl1_pristine
mkdir -p "$DATA/ccd_cache" "$OUT"
export PROTENIX_DATA_ROOT_DIR="$DATA/ccd_cache"
export CUTLASS_PATH="${CUTLASS_PATH:-$HOME/cutlass}"
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
cd "$PX"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import protenix,torch;print('protenix',protenix.__version__ if hasattr(protenix,'__version__') else '?','torch',torch.__version__,'cuda',torch.cuda.is_available())"

"$ENV/bin/pxdesign" infer \
  -i ./examples/PDL1_quick_start.yaml \
  -o "$OUT" \
  --N_sample 4 --N_step 400 --dtype bf16 \
  --eta_type const --eta_min 2.5 --eta_max 2.5 \
  --seeds 101,102,103
echo "PXD_EXIT=$?"
find "$OUT" -name "*.cif" | head -20
echo "n_cif=$(find "$OUT" -name '*.cif' | wc -l)"
