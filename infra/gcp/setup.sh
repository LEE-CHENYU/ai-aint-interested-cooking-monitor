#!/bin/bash
# GCP instance setup script for fine-tuning
# Run this on the hackathon-vm-ai-aint-interested instance

set -e

echo "=== Setting up fine-tuning environment ==="

# Check GPU availability
echo "Checking GPU..."
nvidia-smi || { echo "No GPU detected. Check instance configuration."; exit 1; }

# Install Python dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r infra/requirements.txt

# Verify CUDA + PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true

echo "=== Setup complete ==="
