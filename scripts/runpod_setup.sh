#!/bin/bash
# =============================================================================
# RunPod H200 Training Setup Script for nanochat with KAN
# =============================================================================
# Usage: bash runpod_setup.sh
#
# This script sets up the environment on a RunPod H200 instance:
# 1. Installs uv package manager
# 2. Clones and sets up the nanochat repository
# 3. Installs Python dependencies (with GPU support)
# 4. Builds the Rust tokenizer
# 5. Downloads training data
#
# After running this script, use runpod_train.sh to start training.
# =============================================================================

set -e  # Exit on error

# Configuration
DEPTH=26
DEVICE_BATCH_SIZE=16
LEARNING_RATE=1.5e-4
GRAD_ACCUM=4
NUM_SHARDS=300
REPO_URL="https://github.com/stchakwdev/nanochat.git"

# Baantu Research ASCII Art Banner
cat << 'EOF'

================================================================================

██████╗  █████╗  █████╗ ███╗   ██╗████████╗██╗   ██╗
██╔══██╗██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝██║   ██║
██████╔╝███████║███████║██╔██╗ ██║   ██║   ██║   ██║
██╔══██╗██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║
██████╔╝██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝

██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║
██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

        Hybrid KAN-Transformer Language Models
        Advancing AI Through Novel Architectures

================================================================================

EOF

echo "Model: d${DEPTH} (~750M params)"
echo "Batch size: ${DEVICE_BATCH_SIZE} x ${GRAD_ACCUM} = $((DEVICE_BATCH_SIZE * GRAD_ACCUM))"
echo "Data shards: ${NUM_SHARDS}"
echo "============================================================================"

# Step 1: Install uv package manager
echo -e "\n[1/6] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the new PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "uv version: $(uv --version)"

# Step 2: Clone repository
echo -e "\n[2/6] Cloning nanochat..."
if [ ! -d "nanochat" ]; then
    git clone ${REPO_URL}
else
    echo "Repository already exists, pulling latest..."
    cd nanochat && git pull && cd ..
fi
cd nanochat

# Step 3: Install Python dependencies
echo -e "\n[3/6] Installing Python dependencies..."
uv venv
source .venv/bin/activate
uv sync --extra gpu

# Step 4: Install Rust and build tokenizer
echo -e "\n[4/6] Building Rust tokenizer..."
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "Rust version: $(rustc --version)"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Step 5: Verify GPU
echo -e "\n[5/6] Verifying GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: True')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA not available!')
    exit(1)
"

# Step 6: Download training data
echo -e "\n[6/6] Downloading training data..."
echo "Training tokenizer..."
python -m scripts.tok_train --max_chars=2000000000

echo "Downloading ${NUM_SHARDS} data shards (this may take a while)..."
python -m nanochat.dataset -n ${NUM_SHARDS}

echo -e "\n=============================================="
echo "   Setup Complete!"
echo "=============================================="
echo ""
echo "To start training, run:"
echo "  cd nanochat"
echo "  source .venv/bin/activate"
echo "  bash scripts/runpod_train.sh"
echo ""
echo "Or with HuggingFace upload:"
echo "  HF_REPO=your-username/nanochat-d26-kan bash scripts/runpod_train.sh"
echo "=============================================="
