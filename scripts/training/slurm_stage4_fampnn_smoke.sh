#!/bin/bash
#SBATCH --job-name=proteo-fampnn-smoke
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --account=marlowe-m000137-pm06
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=/users/yfsun/Proteo-AA-stage4-fampnn/runs/stage4-smoke-%j.log
set -euo pipefail
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75
module load mps
PROTEO_REPO=/users/yfsun/Proteo-AA-stage4-fampnn
export PROTEOAA_DATA_ROOT=/scratch/m000137-pm06/Proteo-AA
export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROTEO_REPO:/users/yfsun/protein-code/Protenix:/users/yfsun/protein-code/11/PXDesign:/users/yfsun/protein-code/fampnn
cd "$PROTEO_REPO"
/users/yfsun/.venvs/proteoaa-stage4/bin/python scripts/utilities/smoke_stage4_fampnn.py \
  --donor "$PROTEOAA_DATA_ROOT/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt" \
  --fampnn-checkpoint /users/yfsun/protein-code/fampnn/weights/fampnn_0_3.pt \
  --data-root "$PROTEOAA_DATA_ROOT" --output "$PROTEO_REPO/runs/stage4-smoke-$SLURM_JOB_ID"
