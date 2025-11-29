#!/bin/bash
# Run all three benchmark levels sequentially
# 1. Quick Test (~2 minutes)
# 2. Fast Benchmark (~15 minutes)
# 3. Full Benchmark (~2 hours)

set -e

echo "========================================="
echo "BLUQ - Complete Benchmark Suite"
echo "Running all three benchmark levels"
echo "========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo "✓ Virtual environment activated"
echo "✓ CUDA optimizations enabled"
echo ""

# Display GPU info
echo "GPU Information:"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./results/complete_run_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT_DIR}"

echo "Results will be saved to: ${BASE_OUTPUT_DIR}"
echo ""

# ============================================
# LEVEL 1: Quick Test
# ============================================
echo "========================================="
echo "LEVEL 1: Quick Test"
echo "========================================="
echo "Model: SmolLM-135M"
echo "Task: QA (MMLU)"
echo "Samples: 100"
echo "Expected time: ~2 minutes"
echo ""

START_TIME=$(date +%s)

python run_benchmark.py \
  --tasks qa \
  --models smollm-135m \
  --quick-test \
  --model-config model_config_a100_optimized.yaml \
  --output-dir "${BASE_OUTPUT_DIR}/1_quick_test"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✓ Level 1 completed in ${DURATION} seconds"
echo ""

# ============================================
# LEVEL 2: Fast Benchmark
# ============================================
echo "========================================="
echo "LEVEL 2: Fast Benchmark"
echo "========================================="
echo "Models: SmolLM-135M, SmolLM-360M, TinyLlama-1.1B"
echo "Tasks: QA, RC, CI"
echo "Samples: 1000 per task"
echo "Expected time: ~15 minutes"
echo ""

START_TIME=$(date +%s)

# Use default configs with reduced samples via command line
python run_benchmark.py \
  --tasks qa rc ci \
  --models smollm-135m smollm-360m tinyllama-1.1b \
  --data-config dataset_config_fast.yaml \
  --model-config model_config_a100_optimized.yaml \
  --output-dir "${BASE_OUTPUT_DIR}/2_fast_benchmark"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))
echo ""
echo "✓ Level 2 completed in ${MINUTES}m ${SECONDS}s"
echo ""

# ============================================
# LEVEL 3: Full Benchmark
# ============================================
echo "========================================="
echo "LEVEL 3: Full Benchmark"
echo "========================================="
echo "Models: All 17 models (135M to 2.7B)"
echo "Tasks: All 5 tasks (QA, RC, CI, DRS, DS)"
echo "Expected time: ~2 hours"
echo ""
echo "Starting full benchmark..."
echo ""

START_TIME=$(date +%s)

python run_benchmark.py \
  --tasks qa rc ci drs ds \
  --model-config model_config_a100_optimized.yaml \
  --output-dir "${BASE_OUTPUT_DIR}/3_full_benchmark"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo ""
echo "✓ Level 3 completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================="
echo "ALL BENCHMARKS COMPLETED!"
echo "========================================="
echo ""
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo ""
echo "Summary:"
echo "  - Level 1 (Quick Test): ✓"
echo "  - Level 2 (Fast Benchmark): ✓"
echo "  - Level 3 (Full Benchmark): ✓"
echo ""
echo "Next steps:"
echo "  1. View results: ls -lh ${BASE_OUTPUT_DIR}"
echo "  2. Check comparison report: cat ${BASE_OUTPUT_DIR}/3_full_benchmark/comparison_report.txt"
echo "  3. Analyze CSV summaries in ${BASE_OUTPUT_DIR}/3_full_benchmark/"
echo ""
echo "========================================="
