# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a full-stack implementation of an LLM like ChatGPT in a single, minimal, dependency-lite codebase. It trains models from scratch through tokenization, pretraining, midtraining, SFT, and optional RL on a single 8XH100 node.

## Development Commands

### Environment Setup
```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies (GPU version)
uv sync --extra gpu

# Install dependencies (CPU version)
uv sync --extra cpu

# Activate virtual environment
source .venv/bin/activate

# Build Rust tokenizer (required before running)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v -s

# Run specific test file
python -m pytest tests/test_rustbpe.py -v -s

# Run specific test
python -m pytest tests/test_engine.py::test_specific_function -v -s
```

### Training Commands

**Full Pipeline (Speedrun - ~$100, 4 hours on 8XH100):**
```bash
bash speedrun.sh
```

**Individual Training Stages:**
```bash
# Train tokenizer (run first, required for all other steps)
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Base model pretraining (single GPU)
python -m scripts.base_train

# Base model pretraining (distributed, 8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Midtraining (teach conversation format and tool use)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Supervised Fine-tuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Reinforcement Learning (optional, GSM8K only)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

### Evaluation Commands
```bash
# Evaluate base model on CORE metric
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Evaluate base model bits per byte
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

# Evaluate chat model on all tasks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Evaluate specific checkpoint (mid/sft/rl)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# Evaluate only specific task (e.g., GSM8K)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K
```

### Inference Commands
```bash
# Chat via CLI (interactive)
python -m scripts.chat_cli

# Chat via CLI (single prompt)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Launch web UI (ChatGPT-style interface)
python -m scripts.chat_web
```

### Data Management
```bash
# Download pretraining data shards (n = number of shards, each ~250MB)
python -m nanochat.dataset -n 240
```

## Architecture Overview

### Training Pipeline Flow
nanochat follows a standard LLM training pipeline:
1. **Tokenizer Training** → BPE tokenizer with vocab_size=65536
2. **Base Pretraining** → Causal language modeling on FineWeb dataset
3. **Midtraining** → Learn conversation format, special tokens, tool use
4. **Supervised Finetuning (SFT)** → Task-specific training on mixed datasets
5. **Reinforcement Learning (Optional)** → Currently only GSM8K

### Model Architecture (GPT class)
The core model is in `nanochat/gpt.py`:
- **Architecture**: Decoder-only Transformer with modern improvements
- **Key Features**:
  - Rotary embeddings (RoPE) instead of absolute positional embeddings
  - QK normalization (queries and keys are normalized)
  - Untied weights (separate token embedding and lm_head)
  - ReLU² activation in MLP layers
  - No bias in linear layers, no learnable params in RMSNorm
  - Multi-Query Attention (MQA) / Grouped Query Attention (GQA) support
  - Logits softcapping (logits = 15 * tanh(logits / 15))
- **Model sizing**: Depth determines model size via formula:
  - `num_layers = depth`
  - `model_dim = depth * 64` (aspect ratio of 64)
  - `num_heads = max(1, (model_dim + 127) // 128)` (head_dim = 128)
  - Example: d20 = 20 layers, 1280 dim, 10 heads = 561M params
  - Example: d32 = 32 layers, 2048 dim, 16 heads = 1.9B params

### Optimizer Setup (Dual Optimizer Strategy)
Two optimizers are used together (see `GPT.setup_optimizers`):
1. **DistAdamW** for embedding and lm_head parameters
   - Unembedding LR: 0.004 (scaled by √(768/model_dim))
   - Embedding LR: 0.2 (scaled by √(768/model_dim))
2. **DistMuon** for Transformer layer matrix parameters
   - Matrix LR: 0.02
   - Momentum: 0.95

### Data Loading Architecture
**Tokenizing Distributed Data Loader** (`nanochat/dataloader.py`):
- Streams text from parquet files (FineWeb dataset)
- Tokenizes on-the-fly with multi-threading
- Distributes across DDP ranks automatically
- Uses deque-based token buffer for efficient streaming
- Yields batches of shape (B, T) where B=batch_size, T=sequence_length

### Inference Engine (`nanochat/engine.py`)
**KV Cache Management**:
- Implements efficient autoregressive generation with KV caching
- Supports batch generation with shared prefill
- Dynamically grows cache when needed
- Handles three attention scenarios:
  1. Training (causal attention, no cache)
  2. Single query inference (attend to all cached KV)
  3. Chunked inference (prefix + causal within chunk)

**Tool Use (Calculator)**:
- Models can use `<|python_start|>...<|python_end|>` for calculations
- Engine evaluates expressions safely and injects results
- Results wrapped in `<|output_start|>...<|output_end|>`

### Configuration System (`nanochat/configurator.py`)
Uses a "poor man's configurator" approach:
- Define config variables as module-level globals in scripts
- Override via config files: `python script.py config/file.py`
- Override via CLI args: `python script.py --batch_size=32`
- No configuration objects or classes needed

### Task System (`tasks/common.py`)
All evaluation datasets inherit from `Task` base class:
- **Task**: Base class with slicing support (start, stop, step)
- **TaskMixture**: Combines multiple tasks with deterministic shuffling (for SFT)
- **TaskSequence**: Concatenates tasks sequentially (for curriculum learning)
- Available tasks: ARC, MMLU, GSM8K, HumanEval, SmolTalk, SpellingBee, CustomJSON

### Checkpoint Management
Checkpoints are organized by phase:
- **base**: After pretraining
- **mid**: After midtraining
- **sft**: After supervised finetuning
- **rl**: After reinforcement learning (optional)

Load with: `load_model(phase, device, model_tag=None, step=None)`

## Key Hyperparameters

### Memory Management
If you OOM (Out of Memory), adjust `--device_batch_size`:
- Default: 32 for d20, 16 for d26, 8 for d32
- Reduce to 16 → 8 → 4 → 2 → 1 until it fits
- Code automatically compensates with gradient accumulation

### Scaling Model Size
To train larger models (e.g., GPT-2 grade d26):
1. Increase `--depth=26` (or higher)
2. Decrease `--device_batch_size=16` to avoid OOM
3. Download more data shards:
   - Calculate: params × 20 (Chinchilla) × 4.8 (chars/token) ÷ 250M (chars/shard)
   - Example d26: ~750M params → 72B tokens → 345B chars → 1380 shards
4. Use same device_batch_size for midtraining

### Data:Param Ratio
Default follows Chinchilla: 20× params in tokens
- Control via `--target_param_data_ratio=20`
- Can override with `--num_iterations` or `--target_flops`

## Important Design Principles

1. **Simplicity over configurability**: This is not a framework. It's a cohesive, minimal baseline designed to be forked and modified, not to be exhaustively configurable.

2. **No abstraction layers**: Direct PyTorch code, no model factories, no giant config objects, no if-then-else monsters.

3. **Single file focus**: Most logic is self-contained in single files (gpt.py, engine.py, dataloader.py) that can be read top-to-bottom.

4. **Distributed training is default**: All training scripts assume multi-GPU via torchrun but degrade gracefully to single GPU.

5. **Checkpoint compatibility**: Models from any stage (base/mid/sft/rl) can be loaded for evaluation or further training.

## Special Tokens

The tokenizer uses these special tokens (learned during midtraining):
- `<|bos|>`: Beginning of sequence
- `<|user_start|>`, `<|user_end|>`: User message delimiters
- `<|assistant_start|>`, `<|assistant_end|>`: Assistant message delimiters
- `<|python_start|>`, `<|python_end|>`: Calculator tool delimiters
- `<|output_start|>`, `<|output_end|>`: Tool output delimiters

## CPU/MPS Development

For development without GPUs, use much smaller models:
```bash
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 \
  --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
```

See `dev/runcpu.sh` for a complete CPU-friendly example.

## Rust Tokenizer (rustbpe)

nanochat includes a custom Rust BPE tokenizer for efficient training:
- Faster than Python minbpe, simpler than HuggingFace tokenizers
- Trains vocab from scratch, exports to tiktoken-compatible format
- Must be built before first use: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`

## Report Generation

After training, `report.md` is generated with:
- System info and runtime stats
- Evaluation metrics (CORE, ARC, MMLU, GSM8K, HumanEval, ChatCORE)
- Training samples and loss curves
- Codebase statistics (lines, files, tokens, dependencies)

Generate manually: `python -m nanochat.report generate`

## Environment Variables

- `NANOCHAT_BASE_DIR`: Where to store checkpoints/data (default: `~/.cache/nanochat`)
- `OMP_NUM_THREADS=1`: Disable OpenMP threading (recommended)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Better CUDA memory management
- `WANDB_RUN`: Wandb run name for logging (use "dummy" to disable)

## Customization

To add personality/identity to your model:
1. Generate synthetic conversations (see `dev/gen_synthetic_data.py`)
2. Save as JSONL with format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
3. Add to midtraining or SFT task mixture via `CustomJSON(filepath=path)`

To add new abilities (e.g., counting letters in words):
1. Create synthetic training data for the skill
2. Mix into SFT training via TaskMixture
3. Optionally add as evaluation task in `tasks/` directory

---

## Baantu Research: Hybrid KAN-Transformer

This fork integrates **Kolmogorov-Arnold Networks (KAN)** into nanochat, creating a hybrid architecture where the last 2 transformer layers use KAN instead of standard MLPs.

### Architecture Overview

```
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
```

**Key Changes:**
- **KAN_MLP class** (`nanochat/gpt.py`): Replaces standard MLP with KAN layers
- **Hybrid strategy**: Last 2 layers use `KAN_MLP`, all others use standard `MLP`
- **Package**: Uses `efficient-kan` from [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan)

```python
# Layer distribution for d26 model (26 layers):
# Layers 0-23:  Standard MLP (ReLU² activation)
# Layers 24-25: KAN_MLP (learnable activation functions)
```

### Why KAN?

Kolmogorov-Arnold Networks use learnable activation functions on edges (instead of fixed activations on nodes), potentially offering:
- Better function approximation with fewer parameters
- Improved interpretability
- Different inductive biases than standard MLPs

### New Dependencies

```toml
# pyproject.toml additions:
"efficient-kan @ git+https://github.com/AthanasiosDelis/faster-kan.git"
"huggingface-hub>=0.20.0"
"safetensors>=0.4.0"
```

### New Scripts

| Script | Purpose |
|--------|---------|
| `scripts/runpod_setup.sh` | One-time RunPod H200 environment setup |
| `scripts/runpod_train.sh` | Training + evaluation + HuggingFace upload |
| `scripts/upload_to_hf.py` | Upload trained models to HuggingFace Hub |

### HuggingFace Upload

Upload trained models to HuggingFace Hub:

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Upload after training
python -m scripts.upload_to_hf --repo_id=username/nanochat-d26-kan --source=base

# Options:
#   --repo_id       HuggingFace repo (required)
#   --source        Checkpoint source: base, mid, sft, rl (default: base)
#   --private       Create private repo (default: false)
#   --step          Specific checkpoint step (default: latest)
```

### RunPod H200 Training Plan

**Target Model:** d26 (~750M params) Hybrid KAN-Transformer

**Cost Estimate:** ~$25-35 USD
- H200 on-demand: ~$3.59/hr
- Estimated runtime: 8-10 hours
- Storage: 200GB Network Volume

**Training Configuration:**
```bash
DEPTH=26                    # ~750M params
DEVICE_BATCH_SIZE=16        # Conservative for KAN memory usage
LEARNING_RATE=1.5e-4        # Slightly lower for KAN stability
GRAD_ACCUM=4                # Effective batch = 64
NUM_SHARDS=300              # ~75B chars of training data
```

**Quick Start on RunPod:**

```bash
# 1. Clone the branch
git clone -b baantu-research-branding https://github.com/stchakwdev/nanochat.git
cd nanochat

# 2. Run setup (installs deps, builds tokenizer, downloads data)
bash scripts/runpod_setup.sh

# 3. Start training with HuggingFace upload
source .venv/bin/activate
export HF_TOKEN=your_token_here
HF_REPO=stchakman/nanochat-d26-kan bash scripts/runpod_train.sh
```

**Monitoring:**
- Watch for loss spikes → reduce LR to 1e-4
- Monitor VRAM via `nvidia-smi`
- Check W&B dashboard for training curves

### Environment Variables (Extended)

```bash
# Standard nanochat
export NANOCHAT_BASE_DIR=~/.cache/nanochat
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN=nanochat-d26-kan

# HuggingFace upload
export HF_TOKEN=hf_xxxxxxxxxxxxx
export HF_REPO=username/model-name  # For runpod_train.sh
```

### Checkpoint Flow

```
Training Pipeline:
  base_train.py → base_checkpoints/d26/
       ↓
  base_eval.py → Evaluation metrics
       ↓
  upload_to_hf.py → HuggingFace Hub (stchakman/nanochat-d26-kan)
```

### Branch Information

- **Branch:** `baantu-research-branding`
- **GitHub:** https://github.com/stchakwdev/nanochat/tree/baantu-research-branding
- **HuggingFace:** https://huggingface.co/stchakman

### Future Work

- [ ] Train d26 Hybrid KAN-Transformer on RunPod H200
- [ ] Compare KAN vs MLP performance on downstream tasks
- [ ] Experiment with more KAN layers (3-4 instead of 2)
- [ ] Midtraining and SFT with KAN architecture
- [ ] Publish results and model weights on HuggingFace
