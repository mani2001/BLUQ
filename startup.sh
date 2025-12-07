#!/bin/bash
# Thunder Compute Startup Script for BLUQ
# Run this script on a fresh GPU instance to set up the environment

set -e  # Exit on any error

echo "=============================================="
echo "BLUQ Environment Setup - Thunder Compute"
echo "=============================================="

# Git configuration
GIT_USERNAME="mani2001"
GIT_EMAIL="anirudh@example.com"  # Update this with your email

echo "[1/6] Configuring Git..."
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"
git config --global credential.helper store
echo "Git configured for: $GIT_USERNAME <$GIT_EMAIL>"

# Clone repository
REPO_URL="https://github.com/mani2001/BLUQ.git"
REPO_DIR="$HOME/BLUQ"

echo "[2/6] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "Repository ready at: $REPO_DIR"

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
cd "$REPO_DIR"
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping creation..."
else
    python3 -m venv .venv
fi
echo "Virtual environment created at: $REPO_DIR/.venv"

# Activate virtual environment
echo "[4/6] Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated"

# Upgrade pip
echo "[5/6] Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "[6/6] Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  cd ~/BLUQ && source .venv/bin/activate"
echo ""
echo "To verify GPU is ready:"
echo "  python verify_gpu_ready.py"
echo ""
echo "To run the benchmark:"
echo "  python run_benchmark.py --mode long --tasks qa rc --models gemma-2b"
echo ""

# Verify setup
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA: Available"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "CUDA: Not available (will use CPU)"
fi

echo ""
echo "Ready to run benchmarks!"
