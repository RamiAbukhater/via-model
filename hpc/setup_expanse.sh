#!/usr/bin/env bash
# One-time environment setup on SDSC Expanse. Run on a LOGIN node:
#   bash hpc/setup_expanse.sh
#
# Puts everything big (conda env, HF models, datasets, checkpoints) on the
# Lustre scratch filesystem — home directories are small and slow.
set -euo pipefail

SCRATCH="/expanse/lustre/scratch/$USER/temp_project"
mkdir -p "$SCRATCH"/{hf_cache,datasets,checkpoints,results}

# --- Miniconda (self-managed; avoids drift in the cluster's module stack) ---
if [ ! -d "$SCRATCH/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$SCRATCH/miniconda3"
fi
source "$SCRATCH/miniconda3/bin/activate"

if ! conda env list | grep -q "^via "; then
    conda create -y -n via python=3.10
fi
conda activate via

# --- Python deps (pip torch wheels bundle CUDA; no cluster CUDA module needed) ---
pip install --upgrade pip
pip install "torch>=2.4" "transformers>=4.44" h5py pyyaml wandb pytest pillow matplotlib

# --- LIBERO + MuJoCo stack (for closed-loop eval; offline training works without it) ---
if [ ! -d "$SCRATCH/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$SCRATCH/LIBERO"
fi
pip install -e "$SCRATCH/LIBERO"
pip install robosuite==1.4.1 bddl easydict

# --- Pre-download frozen encoders on the login node (compute nodes shouldn't
#     depend on download speed; cache goes to scratch) ---
export HF_HOME="$SCRATCH/hf_cache"
python - <<'EOF'
from transformers import SiglipVisionModel, AutoModel, AutoTokenizer
SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print("encoders cached OK")
EOF

# --- LIBERO datasets (~ a few GB per suite) ---
python "$SCRATCH/LIBERO/benchmark_scripts/download_libero_datasets.py" \
    --download-dir "$SCRATCH/datasets/libero" --datasets libero_spatial

echo
echo "Done. Add to ~/.bashrc:"
echo "  export SCRATCH=$SCRATCH"
echo "  export HF_HOME=\$SCRATCH/hf_cache"
echo "  source \$SCRATCH/miniconda3/bin/activate via"
echo "Then run 'wandb login' once, and edit configs/expanse.yaml paths if needed."
