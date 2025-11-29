#!/bin/bash
# Quick Start Script for A100 GPU Benchmarking
# Optimized for maximum speed

set -e

echo "========================================="
echo "BLUQ Benchmark - A100 Optimized"
echo "========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "✓ Virtual environment activated"
echo "✓ CUDA optimizations enabled"
echo ""

# Check GPU
echo "GPU Information:"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
echo ""

# Generate default configs first
echo "Generating configuration files..."
python generate_configs.py
echo ""

# Ask user what to run
echo "What would you like to run?"
echo ""
echo "1. Quick Test (1 model, 1 task, 100 samples) - ~2 minutes"
echo "2. Fast Benchmark (3 models, 3 tasks, 1000 samples each) - ~15 minutes"
echo "3. Full Benchmark (All models, all tasks) - ~2 hours"
echo "4. Custom (you specify)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
  1)
    echo ""
    echo "Running Quick Test..."
    echo "Model: SmolLM-135M (fastest)"
    echo "Task: QA (MMLU)"
    echo "Samples: 100"
    echo ""
    python run_single_task.py \
      --task qa \
      --model smollm-135m \
      --num-samples 100 \
      --output-dir ./results/quick_test
    ;;
    
  2)
    echo ""
    echo "Running Fast Benchmark..."
    echo "Models: SmolLM-135M, SmolLM-360M, TinyLlama-1.1B"
    echo "Tasks: QA, RC, CI"
    echo "Samples: 1000 per task"
    echo ""
    python run_benchmark.py \
      --tasks qa rc ci \
      --models smollm-135m smollm-360m tinyllama-1.1b \
      --data-config dataset_config_fast.yaml \
      --output-dir ./results/fast_benchmark
    ;;
    
  3)
    echo ""
    echo "Running Full Benchmark..."
    echo "This will take approximately 2 hours"
    echo ""
    read -p "Are you sure? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
      python run_benchmark.py \
        --tasks qa rc ci drs ds \
        --output-dir ./results/full_benchmark
    else
      echo "Cancelled."
    fi
    ;;
    
  4)
    echo ""
    echo "Custom run - use run_benchmark.py with your own parameters"
    echo ""
    echo "Example:"
    echo "  python run_benchmark.py --tasks qa rc --models phi-2 gemma-2b"
    echo ""
    echo "Available models:"
    echo "  - smollm-135m (fastest)"
    echo "  - smollm-360m"
    echo "  - openelm-270m"
    echo "  - tinyllama-1.1b"
    echo "  - smollm-1.7b"
    echo "  - qwen-1.8b"
    echo "  - phi-2"
    echo "  - gemma-2b"
    echo "  - gemma-2b-it"
    echo ""
    echo "Available tasks:"
    echo "  - qa (Question Answering - MMLU)"
    echo "  - rc (Reading Comprehension - CosmosQA)"
    echo "  - ci (Commonsense Inference - HellaSwag)"
    echo "  - drs (Dialogue Response Selection)"
    echo "  - ds (Document Summarization)"
    ;;
    
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
