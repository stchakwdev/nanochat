#!/bin/bash
# =============================================================================
# RunPod H200 Training Script for nanochat with KAN
# =============================================================================
# Usage: bash runpod_train.sh
#        HF_REPO=username/repo-name bash runpod_train.sh
#
# This script:
# 1. Sets up environment variables for optimal GPU performance
# 2. Runs base model training with KAN layers
# 3. Evaluates the trained model
# 4. Generates a training report
# 5. Optionally uploads to HuggingFace Hub
#
# Prerequisites: Run runpod_setup.sh first!
# =============================================================================

set -e  # Exit on error

# Configuration
DEPTH=26
DEVICE_BATCH_SIZE=16
LEARNING_RATE=1.5e-4
GRAD_ACCUM=4
WANDB_RUN_NAME="${WANDB_RUN:-nanochat-d26-kan}"
HF_REPO="${HF_REPO:-}"

# Environment variables for optimal GPU performance
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN="${WANDB_RUN_NAME}"

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
echo "Learning rate: ${LEARNING_RATE}"
echo "W&B run: ${WANDB_RUN_NAME}"
if [ -n "${HF_REPO}" ]; then
    echo "HuggingFace repo: ${HF_REPO}"
fi
echo "============================================================================"

# Ensure we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    if [ -d "nanochat" ]; then
        cd nanochat
    else
        echo "ERROR: Cannot find nanochat directory. Run runpod_setup.sh first!"
        exit 1
    fi
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Run runpod_setup.sh first!"
    exit 1
fi

# Verify GPU before starting
echo -e "\n[Pre-check] Verifying GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Start training
echo -e "\n[1/4] Starting training..."
echo "This will take approximately 8-10 hours for d26..."
echo ""

START_TIME=$(date +%s)

python -m scripts.base_train \
    --depth=${DEPTH} \
    --device_batch_size=${DEVICE_BATCH_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --gradient_accumulation_steps=${GRAD_ACCUM}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "\nTraining completed in $((DURATION / 3600))h $((DURATION % 3600 / 60))m"

# Evaluate model
echo -e "\n[2/4] Running evaluation..."
python -m scripts.base_eval || echo "Warning: Evaluation had issues, continuing..."

# Generate report
echo -e "\n[3/4] Generating training report..."
python -m nanochat.report generate || echo "Warning: Report generation had issues, continuing..."

# Upload to HuggingFace (if configured)
echo -e "\n[4/4] Finalizing..."
if [ -n "${HF_REPO}" ]; then
    echo "Uploading to HuggingFace: ${HF_REPO}"
    python -m scripts.upload_to_hf --repo_id="${HF_REPO}" --source=base
    echo "Model uploaded successfully!"
else
    echo "Skipping HuggingFace upload (HF_REPO not set)"
    echo "To upload later, run:"
    echo "  python -m scripts.upload_to_hf --repo_id=YOUR_USERNAME/nanochat-d26-kan --source=base"
fi

echo -e "\n=============================================="
echo "   Training Complete!"
echo "=============================================="
echo ""
echo "Checkpoint location: ~/.cache/nanochat/base_checkpoints/d${DEPTH}/"
echo "Training duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo ""
echo "IMPORTANT: Remember to terminate your RunPod instance to stop billing!"
echo "=============================================="
