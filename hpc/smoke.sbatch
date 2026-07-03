#!/usr/bin/env bash
# 30-minute sanity job on the gpu-debug partition: verifies the real encoders
# load on a GPU node, the test suite passes, and one LIBERO episode steps in
# headless MuJoCo. Run this FIRST, before spending allocation on training.
#
#   sbatch hpc/smoke.sbatch
#
#SBATCH --job-name=via-smoke
#SBATCH --account=REPLACE_WITH_ALLOCATION
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

SCRATCH="/expanse/lustre/scratch/$USER/temp_project"
source "$SCRATCH/miniconda3/bin/activate" via
export HF_HOME="$SCRATCH/hf_cache"
export MUJOCO_GL=egl

echo "=== 1. test suite (stub encoders) ==="
python -m pytest -q

echo "=== 2. real encoders on GPU ==="
python - <<'EOF'
import torch
from via.perception import SigLIPPerception
from via.goal import Phi3LanguageEncoder
p = SigLIPPerception().cuda()
out = p(torch.rand(1, 2, 3, 224, 224, device="cuda"))
assert out.shape == (1, 2, 196, 768), out.shape
l = Phi3LanguageEncoder().cuda()
tok, mask = l(["pick up the red block"])
assert tok.shape[-1] == 3072, tok.shape
print("encoders OK:", out.shape, tok.shape)
EOF

echo "=== 3. LIBERO dataset loader ==="
python - <<'EOF'
import os
from via.data.libero import LiberoTrajectoryDataset
ds = LiberoTrajectoryDataset(
    os.path.expandvars("/expanse/lustre/scratch/$USER/temp_project/datasets/libero"),
    clip_len=12,
)
s = ds[0]
print("clips:", len(ds), "| frames", tuple(s["frames"].shape),
      "| instruction:", s["instruction"])
EOF

echo "=== 4. one headless LIBERO episode (untrained model) ==="
python - <<'EOF'
from eval.eval_libero import build_model
from train import common
from via.data.libero import LiberoEnvRunner
cfg = common.load_config("configs/expanse.yaml")
model = build_model(cfg, "cuda", smoke=True)   # untrained weights; tests plumbing only
runner = LiberoEnvRunner("libero_spatial")
out = runner.run_episode(model, task_id=0, max_steps=20, device="cuda")
print("episode ran, success =", out["success"], "| task:", out["instruction"])
EOF

echo "ALL SMOKE CHECKS PASSED"
