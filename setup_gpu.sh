#!/bin/bash

# setup_gpu.sh
# Script to set up the environment on a rented GPU instance (e.g., Lambda Labs, RunPod)

set -e  # Exit on error

echo "Starting GPU environment setup..."

# 1. System updates (optional, might need sudo)
if command -v apt-get &> /dev/null; then
    echo "Updating system packages..."
    sudo apt-get update && sudo apt-get install -y python3-venv git
fi

# 2. Create virtual environment
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install GPU-specific optimizations (optional but recommended)
# Flash Attention requires CUDA toolkit to be installed
# pip install flash-attn --no-build-isolation

echo "Setup complete! Activate the environment with: source .venv/bin/activate"
