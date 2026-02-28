#!/bin/bash
# GCP instance setup script for Unsloth fine-tuning
# Run this on the hackathon-vm-ai-aint-interested instance

set -e

echo "=== Setting up Unsloth fine-tuning environment ==="

# Check GPU availability
echo "Checking GPU..."
nvidia-smi || { echo "No GPU detected. Check instance configuration."; exit 1; }

# Show GPU info for config decisions
echo ""
echo "GPU Memory Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install Unsloth (handles torch, transformers, peft, bitsandbytes internally)
echo "Installing Unsloth and dependencies..."
pip install --upgrade pip
pip install -r infra/requirements.txt

# Verify installation
echo ""
echo "Verifying Unsloth installation..."
python -c "
from unsloth import FastLanguageModel
import torch
print(f'Unsloth loaded successfully')
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Generate training data:  python data/generate_synthetic.py"
echo "  2. Run fine-tuning:         python infra/fine_tuning/train.py"
echo "  3. Evaluate:                python infra/fine_tuning/eval/compare.py"
echo "  4. Export model:            python infra/deployment/export_model.py"
