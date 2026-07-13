#!/usr/bin/env bash
# Run one VIA stage inside a DSMLP GPU pod. From the repo root, in a pod:
#
#   bash hpc/dsmlp_train.sh train.train_belief
#   bash hpc/dsmlp_train.sh train.train_world_model
#   bash hpc/dsmlp_train.sh train.train_goal
#   bash hpc/dsmlp_train.sh train.train_decision
#   bash hpc/dsmlp_train.sh eval.uncertainty_analysis
#   bash hpc/dsmlp_train.sh eval.eval_libero --suite libero_spatial --episodes 10
#
# Always passes --resume: DSMLP pods have a session time limit, so if a run
# is killed mid-training, relaunch a pod and rerun the same command — it
# warm-starts from the last per-epoch checkpoint in ~/via-checkpoints.
#
# Recommended pod (goal/decision stages load Phi-3, ~8 GB fp16 + headroom):
#   launch-scipy-ml.sh -g 1 -c 8 -m 32 -v 2080ti     # belief / world model
#   launch-scipy-ml.sh -g 1 -c 8 -m 32 -v a5000      # goal / decision / eval
set -euo pipefail

source "$HOME/envs/via/bin/activate"
export HF_HOME="$HOME/via-hf"
export MUJOCO_GL=egl          # headless rendering; try osmesa if EGL fails
mkdir -p logs

MODULE="$1"; shift || true
LOG="logs/$(echo "$MODULE" | tr '.' '-')-$(date +%m%d-%H%M).log"
echo "logging to $LOG"
python -m "$MODULE" --config configs/dsmlp.yaml --device cuda --resume "$@" 2>&1 | tee "$LOG"
