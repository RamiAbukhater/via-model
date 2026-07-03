#!/usr/bin/env bash
# Generic single-GPU training job on Expanse (gpu-shared partition).
#
#   sbatch hpc/train.sbatch train.train_belief
#   sbatch hpc/train.sbatch train.train_world_model
#   sbatch hpc/train.sbatch train.train_goal
#   sbatch hpc/train.sbatch train.train_decision
#   sbatch --time=24:00:00 hpc/train.sbatch eval.eval_libero --suite libero_spatial
#
# EDIT --account below to your allocation ID (shown by `expanse-client user`).
#
#SBATCH --job-name=via
#SBATCH --account=REPLACE_WITH_ALLOCATION
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

SCRATCH="/expanse/lustre/scratch/$USER/temp_project"
source "$SCRATCH/miniconda3/bin/activate" via
export HF_HOME="$SCRATCH/hf_cache"
export MUJOCO_GL=egl          # headless GPU rendering for robosuite/MuJoCo
export WANDB_DIR="$SCRATCH"

MODULE="$1"; shift || true
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
srun python -m "$MODULE" --config configs/expanse.yaml --device cuda "$@"
