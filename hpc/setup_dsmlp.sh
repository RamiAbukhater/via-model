#!/usr/bin/env bash
# One-time environment setup on UCSD DSMLP. Run INSIDE a pod (CPU pod is fine):
#
#   ssh <ucsd-username>@dsmlp-login.ucsd.edu       # UCSD VPN if off-campus
#   git clone <your-repo-url> via && cd via
#   launch-scipy-ml.sh -c 4 -m 16                  # CPU pod for setup
#   bash hpc/setup_dsmlp.sh
#
# Everything persists in your home directory across pods. Rough footprint:
# venv ~7 GB, HF models ~9 GB, libero_spatial ~5 GB — well under quota, but
# check with `df -h ~` if you have other large files.
set -euo pipefail

ENV="$HOME/envs/via"
mkdir -p "$HOME"/{envs,via-data,via-checkpoints,via-hf}

# --- Python env (venv in home; DSMLP images' preinstalled torch is often old) ---
if [ ! -d "$ENV" ]; then
    python3 -m venv "$ENV"
fi
source "$ENV/bin/activate"
pip install --upgrade pip
pip install "torch>=2.4" "transformers>=4.44" h5py pyyaml wandb pytest pillow matplotlib

# --- LIBERO (dataset download + offline API; sim eval needs working EGL/OSMesa) ---
if [ ! -d "$HOME/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HOME/LIBERO"
fi
pip install -e "$HOME/LIBERO"
pip install robosuite==1.4.1 bddl easydict

# --- Pre-download frozen encoders into the persistent HF cache ---
export HF_HOME="$HOME/via-hf"
python - <<'EOF'
from transformers import SiglipVisionModel, AutoModel, AutoTokenizer
SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print("encoders cached OK")
EOF

# --- LIBERO demonstrations ---
python "$HOME/LIBERO/benchmark_scripts/download_libero_datasets.py" \
    --download-dir "$HOME/via-data/libero" --datasets libero_spatial

echo
echo "Done. Add to ~/.bashrc:"
echo '  export HF_HOME=$HOME/via-hf'
echo '  source $HOME/envs/via/bin/activate'
echo "Then run 'wandb login' once."
