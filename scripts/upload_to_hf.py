"""
Upload trained nanochat model to Hugging Face Hub.

Usage:
    python -m scripts.upload_to_hf --repo_id=username/nanochat-d12-kan
    python -m scripts.upload_to_hf --repo_id=username/nanochat-d12-kan --checkpoint_dir=~/.cache/nanochat/base_checkpoints/d12
    python -m scripts.upload_to_hf --repo_id=username/nanochat-d12-kan --private=true
"""
import os
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_largest_model, find_last_step, load_checkpoint

# Configuration (can be overridden via command line)
repo_id = None  # Required: e.g., "username/nanochat-d12-kan"
checkpoint_dir = None  # Optional: defaults to base_checkpoints with largest model
step = None  # Optional: defaults to latest step
private = False  # Whether to create a private repo
source = "base"  # Which checkpoint source: base, mid, sft, rl


BAANTU_ASCII_ART = """
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
"""


def get_model_card(meta_data: dict, repo_id: str) -> str:
    """Generate a HuggingFace model card."""
    config = meta_data.get("model_config", {})

    return f"""---
license: mit
library_name: pytorch
tags:
  - nanochat
  - transformer
  - kan
  - hybrid-kan-transformer
  - baantu-research
---

# {repo_id.split('/')[-1]}

```
{BAANTU_ASCII_ART}
```

A hybrid KAN-Transformer language model trained by [Baantu Research](https://github.com/stchakwdev) using [nanochat](https://github.com/stchakwdev/nanochat).

## Model Details

- **Organization**: Baantu Research
- **Architecture**: Hybrid KAN-Transformer (last 2 layers use Kolmogorov-Arnold Networks)
- **Layers**: {config.get('n_layer', 'N/A')}
- **Hidden Size**: {config.get('n_embd', 'N/A')}
- **Attention Heads**: {config.get('n_head', 'N/A')}
- **KV Heads**: {config.get('n_kv_head', 'N/A')}
- **Vocab Size**: {config.get('vocab_size', 'N/A')}
- **Max Sequence Length**: {config.get('sequence_len', 'N/A')}

## Training Details

- **Framework**: nanochat (PyTorch)
- **Validation BPB**: {meta_data.get('val_bpb', 'N/A')}
- **GPU**: NVIDIA H200 (RunPod)

## Usage

```python
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", device="cuda", phase="eval")
```

## About Baantu Research

Baantu Research is focused on advancing AI through novel architectures, exploring
Kolmogorov-Arnold Networks (KAN), Transformer architectures, and efficient deep learning.

## Citation

If you use this model, please cite the Baantu Research nanochat repository:

```bibtex
@misc{{baantu-nanochat,
  author = {{Baantu Research}},
  title = {{nanochat: Hybrid KAN-Transformer Language Models}},
  year = {{2024}},
  publisher = {{GitHub}},
  url = {{https://github.com/stchakwdev/nanochat}}
}}
```
"""


def upload_model():
    """Upload model to HuggingFace Hub."""
    global repo_id, checkpoint_dir, step, private, source

    if repo_id is None:
        raise ValueError("repo_id is required. Set it via --repo_id=username/model-name")

    # Resolve checkpoint directory
    if checkpoint_dir is None:
        base_dir = get_base_dir()
        source_dirs = {
            "base": "base_checkpoints",
            "mid": "mid_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        checkpoints_dir = os.path.join(base_dir, source_dirs[source])
        model_tag = find_largest_model(checkpoints_dir)
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
        print(f"Using checkpoint directory: {checkpoint_dir}")
    else:
        checkpoint_dir = os.path.expanduser(checkpoint_dir)

    # Resolve step
    global_step = step
    if global_step is None:
        global_step = find_last_step(checkpoint_dir)
        print(f"Using latest step: {global_step}")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_dir} at step {global_step}...")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, global_step, device="cpu", load_optimizer=False)

    # Remove torch.compile prefix if present
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Convert bfloat16 to float32 for compatibility
    model_data = {
        k: v.float() if v.dtype == torch.bfloat16 else v
        for k, v in model_data.items()
    }

    # Create HF repo
    api = HfApi()
    print(f"Creating repository: {repo_id} (private={private})...")
    create_repo(repo_id, private=private, exist_ok=True)

    # Upload files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save model as safetensors
        safetensors_path = tmpdir / "model.safetensors"
        print(f"Converting to safetensors: {safetensors_path}")
        save_file(model_data, safetensors_path)

        # Save config
        config_path = tmpdir / "config.json"
        config_data = {
            "model_type": "nanochat-gpt",
            "architecture": "hybrid-kan-transformer",
            **meta_data.get("model_config", {}),
            "step": global_step,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Save model card
        readme_path = tmpdir / "README.md"
        with open(readme_path, "w") as f:
            f.write(get_model_card(meta_data, repo_id))

        # Upload all files
        print(f"Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            commit_message=f"Upload nanochat model (step {global_step})",
        )

    print(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    from nanochat.configurator import overrides_from_argv
    overrides_from_argv(globals())
    upload_model()
