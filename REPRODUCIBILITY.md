# Reproducibility Guide

This document provides detailed instructions to reproduce the benchmark results in the BLUQ project.

## Hardware Requirements

### GPU (Recommended for Production Benchmarks)

- **Primary configuration**: NVIDIA A100 80GB GPU (used for all published benchmarks)
- **Alternative GPUs**: Any CUDA-compatible GPU with 16GB+ VRAM
  - A100 40GB: Supported with reduced batch sizes
  - RTX 4090/3090: 24GB VRAM, use batch_size=2-4 for 7B models
  - RTX 3080/4080: 16GB VRAM, limit to 2B models or smaller batch sizes

### Apple Silicon (Mac)

- **Supported**: M1/M2/M3 chips with MPS (Metal Performance Shaders)
- **Memory recommendations**:
  - 8GB RAM: Use smollm-135m, smollm-360m
  - 16GB RAM: Use tinyllama-1.1b, phi-2
  - 32GB+ RAM: Use stablelm-2-1.6b, gemma-2b, phi-2
  - 64GB+ RAM: Use gemma-2-9b-it, mistral-7b
- **Note**: MPS is 2-5x slower than A100 but still faster than CPU
- **Limitations**: GPU profiling disabled on MPS (CUDA-only feature)

### CPU Fallback

- **Supported**: Any x86_64 or ARM64 CPU
- **Memory**: 16GB+ RAM recommended
- **Note**: Significantly slower (10-50x) than GPU, not recommended for full benchmarks

## Software Requirements

### Core Dependencies

- **Python**: 3.10 or higher (tested on 3.10, 3.11)
- **Operating System**: Linux (Ubuntu 20.04+), macOS (12.0+), Windows (WSL2)
- **CUDA**: 11.8+ (for NVIDIA GPUs)

### Python Packages

All dependencies are specified in `requirements.txt`:

```bash
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# PyTorch and transformers
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# HuggingFace datasets
datasets>=2.10.0,<2.15.0

# Configuration and serialization
pyyaml>=6.0
dataclasses-json>=0.5.9

# Progress bars and utilities
tqdm>=4.65.0
coloredlogs>=15.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# System utilities
psutil>=5.9.0

# Testing (optional)
pytest>=7.3.0
pytest-cov>=4.1.0
scikit-learn>=1.2.0

# GPU monitoring (optional, CUDA only)
nvidia-ml-py>=12.535.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/BLUQ.git
cd BLUQ

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation and compute environment
python verify_gpu_ready.py
```

## Random Seeds and Reproducibility

### Default Seed

- **Random seed**: `42` (default across all experiments)
- **Location**: `configs/dataset_config.yaml` (line 1: `seed: 42`)
- **Usage**: Controls dataset sampling, calibration/test splits, and demonstration selection

### Seed Propagation

The seed is propagated to:
- NumPy random number generator
- Dataset sampling (stratified for MMLU, random for others)
- Calibration/test split (50/50)
- Demonstration selection (per-task few-shot examples)

### Model Inference

- **Temperature**: `1.0` (no sampling, deterministic greedy decoding)
- **Note**: Model inference is deterministic given the same input prompts and model weights

## Hyperparameters

### Conformal Prediction

- **alpha**: `0.1` (target 90% coverage rate, i.e., 1-alpha = 0.9)
- **calibration_ratio**: `0.5` (50/50 split between calibration and test sets)
- **Conformal methods**:
  - `lac`: Least Ambiguous Classifiers (score = 1 - P(true_class))
  - `aps`: Adaptive Prediction Sets (score = cumulative prob to reach true class)

### Sampling

- **num_samples**: `10000` (for full production benchmarks)
- **Test mode**: `100` (for quick testing/debugging with `--mode short`)

### Model Configuration

- **dtype**: `float16` (primary), `float32` (secondary for comparison)
- **max_length**: `2048` tokens
- **temperature**: `1.0` (deterministic greedy decoding)
- **batch_size**: Dynamic, auto-detected based on GPU tier
  - A100 80GB: 16-32 (for <1.5B params), 4-8 (1.5B-3B), 2-4 (7B)
  - A100 40GB: 8-16 (for <1.5B params), 2-4 (1.5B-3B), 1-2 (7B)
  - RTX 3090/4090: 4-8 (for <1.5B params), 1-2 (7B)
  - MPS/CPU: 2-4 (conservative for memory)
- **Override**: Use `--max-batch-size N` to manually specify

### Prompting

- **Strategy**: `base` (default, simple question-answer format)
- **Alternative strategies**: `shared_instruction`, `task_specific`
- **Demonstrations**: Task-specific few-shot examples
  - QA (MMLU): 5 demonstrations
  - RC (CosmosQA): 5 demonstrations
  - CI (HellaSwag): 5 demonstrations
  - DRS (HaluDial): 3 demonstrations
  - DS (HaluSum): 1 demonstration

### Answer Format

All tasks standardized to 6-option multiple choice:
- **A-D**: Original options (4 options for most tasks)
- **E**: "I don't know" (epistemic uncertainty)
- **F**: "None of the above" (aleatoric uncertainty)

## Dataset Details

### Tasks and Datasets

1. **qa** (Question Answering)
   - Dataset: MMLU (Massive Multitask Language Understanding)
   - HuggingFace: `cais/mmlu`
   - Sampling: Stratified by subject (57 subjects across 4 categories)
   - Split: `test` (primary), `validation`, `dev` (fallback)

2. **rc** (Reading Comprehension)
   - Dataset: CosmosQA
   - HuggingFace: `cosmos_qa`
   - Sampling: Random
   - Split: `train`, `validation`

3. **ci** (Commonsense Inference)
   - Dataset: HellaSwag
   - HuggingFace: `Rowan/hellaswag`
   - Sampling: Random
   - Split: `train`, `validation`

4. **drs** (Dialogue Response Selection)
   - Dataset: HaluDial (from HaluEval)
   - HuggingFace: `pminervini/HaluEval` (config: `dialogue`)
   - Sampling: Random
   - Split: `data`

5. **ds** (Document Summarization)
   - Dataset: HaluSum (from HaluEval)
   - HuggingFace: `pminervini/HaluEval` (config: `summarization`)
   - Sampling: Random
   - Split: `data`

### Data Download

All datasets are automatically downloaded from HuggingFace on first use:
- **Cache location**: `./data/cache` (configurable in `configs/dataset_config.yaml`)
- **Note**: First run requires internet connection and may take 10-30 minutes depending on bandwidth
- **Storage**: Approximately 5-10GB for all datasets

## Models

### Supported Models

All models are automatically downloaded from HuggingFace:

1. **tinyllama-1.1b**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. **phi-2**: `microsoft/phi-2` (2.7B params)
3. **stablelm-2-1.6b**: `stabilityai/stablelm-2-1_6b`
4. **gemma-2b-it**: `google/gemma-2b-it`
5. **gemma-2-2b-it**: `google/gemma-2-2b-it`
6. **gemma-2-9b-it**: `google/gemma-2-9b-it`
7. **mistral-7b**: `mistralai/Mistral-7B-v0.1`
8. **mistral-7b-instruct**: `mistralai/Mistral-7B-Instruct-v0.2`
9. **qwen-1.8b**: `Qwen/Qwen-1_8B` (known tokenizer issue)

### Model Cache

- **Location**: `~/.cache/huggingface/hub` (default HuggingFace cache)
- **Storage**: 5-30GB per model (depends on model size and dtype)
- **Override**: Set `HF_HOME` environment variable

## Reproducing Benchmark Results

### Quick Test (100 samples, 1-2 minutes per task)

```bash
# Test single model on single task
python run_benchmark.py --mode short --tasks qa --models tinyllama-1.1b

# Test multiple tasks
python run_benchmark.py --mode short --tasks qa rc ci --models tinyllama-1.1b
```

### Full Production Benchmark (10,000 samples)

#### Single Model (1-3 hours depending on model size)

```bash
# TinyLlama-1.1B (float16, ~25 minutes)
python run_benchmark.py --mode long --tasks qa rc ci drs ds --models tinyllama-1.1b --dtypes float16

# Phi-2 (float16, ~90 minutes)
python run_benchmark.py --mode long --tasks qa rc ci drs ds --models phi-2 --dtypes float16

# Mistral-7B-Instruct (float16, ~3 hours)
python run_benchmark.py --mode long --tasks qa rc ci drs ds --models mistral-7b-instruct --dtypes float16
```

#### Multiple Models (5-20 hours depending on selection)

```bash
# Small models (<2B params, ~4-6 hours total)
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models tinyllama-1.1b phi-2 stablelm-2-1.6b --dtypes float16

# All tested models (15-20 hours on A100 80GB)
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models tinyllama-1.1b phi-2 stablelm-2-1.6b gemma-2-9b-it mistral-7b mistral-7b-instruct \
  --dtypes float16
```

#### Dtype Comparison (float16 vs float32)

```bash
# Run both dtypes for comparison (2x runtime)
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models phi-2 --dtypes float16 float32
```

### Checkpoint and Resume

For long-running benchmarks, use `--resume` to recover from interruptions:

```bash
# Start a benchmark
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models tinyllama-1.1b phi-2 stablelm-2-1.6b --dtypes float16

# If interrupted, resume with same command + --resume flag
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models tinyllama-1.1b phi-2 stablelm-2-1.6b --dtypes float16 --resume
```

The checkpoint file (`outputs/results/checkpoint.json`) tracks completed (model, task, dtype) combinations and skips them on resume.

### Custom Configurations

```bash
# Custom sample count
python run_benchmark.py --mode custom --num-samples 5000 --tasks qa --models phi-2

# Custom batch size (for memory-constrained GPUs)
python run_benchmark.py --mode long --tasks qa --models mistral-7b \
  --max-batch-size 2

# Disable dynamic batch sizing
python run_benchmark.py --mode long --tasks qa --models phi-2 \
  --no-dynamic-batch --max-batch-size 8

# Multiple prompting strategies
python run_benchmark.py --mode short --tasks qa --models phi-2 \
  --strategies base shared_instruction task_specific

# Disable GPU profiling (slight speedup)
python run_benchmark.py --mode long --tasks qa --models phi-2 --no-profiling
```

## Expected Runtime

Runtimes on **NVIDIA A100 80GB** with 10,000 samples per task (5 tasks total):

| Model | Parameters | float16 | float32 | Notes |
|-------|-----------|---------|---------|-------|
| tinyllama-1.1b | 1.1B | 25 min | 56 min | Batch size 16 |
| phi-2 | 2.7B | 87 min | 63 min | Batch size 8 |
| stablelm-2-1.6b | 1.6B | 62 min | 28 min | Batch size 8 |
| gemma-2-2b-it | 2B | 61 min | N/A | Batch size 8 |
| gemma-2-9b-it | 9B | 168 min | 160 min | Batch size 4 |
| mistral-7b | 7B | ~180 min | ~180 min | Batch size 2 |
| mistral-7b-instruct | 7B | ~180 min | ~180 min | Batch size 2 |

**Notes**:
- Runtimes include model loading, dataset processing, inference, conformal calibration, and result saving
- float32 can be faster for some models due to hardware optimizations
- Actual runtime varies by GPU utilization, dataset download speed, and system load
- MPS (Apple Silicon): Expect 2-5x longer runtimes

## Output Structure

Results are saved to `outputs/results/` with the following structure:

```
outputs/results/
├── checkpoint.json                          # Resume checkpoint
├── config_YYYYMMDD_HHMMSS.json             # Run configuration
├── summary_YYYYMMDD_HHMMSS.json            # Overall summary
├── figures/                                 # Visualizations
│   ├── dashboard_float16.png
│   ├── heatmap_accuracy_float16.png
│   └── ...
├── logs/                                    # Detailed logs
│   └── benchmark_YYYYMMDD_HHMMSS.log
└── {model-name}/                           # Per-model subdirectories
    ├── config_{dtype}_YYYYMMDD_HHMMSS.json
    ├── results_{dtype}_YYYYMMDD_HHMMSS.json
    ├── summary_{dtype}_YYYYMMDD_HHMMSS.json
    ├── gpu_profile_{dtype}_YYYYMMDD_HHMMSS.json
    └── probabilities/                       # Raw probability arrays
        └── probs_{task}_{dtype}_{strategy}_YYYYMMDD_HHMMSS.npz
```

### Key Output Files

1. **results_{dtype}_YYYYMMDD_HHMMSS.json**: Raw results for all tasks, including:
   - Accuracy, coverage rate, average set size
   - Per-task, per-strategy, per-conformal-method metrics
   - Inference time, number of samples

2. **summary_{dtype}_YYYYMMDD_HHMMSS.json**: Aggregated statistics:
   - Overall accuracy, coverage, set size
   - Coverage guarantee met ratio
   - Total runtime

3. **probabilities/probs_*.npz**: Raw probability arrays for post-hoc analysis:
   - `cal_probs`: Calibration set probabilities (n_cal, 6)
   - `test_probs`: Test set probabilities (n_test, 6)
   - `cal_labels`, `test_labels`: True labels
   - Metadata: model, task, dtype, strategy, timestamp

4. **gpu_profile_{dtype}_YYYYMMDD_HHMMSS.json**: GPU utilization metrics (CUDA only):
   - Memory usage over time
   - Utilization percentages
   - Peak memory, average utilization

## Verifying Results

### Expected Metrics

For a correctly running benchmark, you should see:

1. **Coverage Rate**: 85-95% (target is 90% with alpha=0.1)
   - Lower than 85%: Model may be miscalibrated or task is too difficult
   - Higher than 95%: Model may be too conservative (large prediction sets)

2. **Average Set Size**: 2-5 options
   - Closer to 1: Model is very confident (may undercover)
   - Closer to 6: Model is very uncertain (may overcover)
   - Ideal: 2-3 (informative prediction sets)

3. **Accuracy**: Varies by model and task
   - Small models (1-2B): 25-40%
   - Medium models (7B): 30-50%
   - Note: Random baseline is 16.67% (6 options)

4. **Coverage Guarantee Met**: At least 80% of tasks should meet the 90% guarantee
   - Lower values indicate poor calibration or insufficient calibration samples

### Comparing Results

To verify your results match the published benchmarks:

1. **Check model versions**: Ensure you're using the same HuggingFace model IDs
2. **Check seed**: Default seed=42 should be used (in `configs/dataset_config.yaml`)
3. **Check sample count**: Full benchmarks use 10,000 samples per task
4. **Allow for minor variations**: Due to non-determinism in GPU operations, expect 1-2% variation in metrics

### Example Verification

```bash
# Run the same configuration as published results
python run_benchmark.py --mode long --tasks qa rc ci drs ds \
  --models phi-2 --dtypes float16 --seed 42

# Compare with published results
cat outputs/results/phi-2/summary_float16_*.json

# Expected output (approximate):
# {
#   "overall_accuracy": 0.30-0.35,
#   "overall_coverage": 0.88-0.92,
#   "overall_set_size": 3.5-4.5,
#   "guarantee_met_ratio": 0.6-0.8
# }
```

### Post-Hoc Analysis

Load raw probabilities for custom analysis:

```python
import numpy as np

# Load probabilities
data = np.load('outputs/results/phi-2/probabilities/probs_qa_float16_base_*.npz')

# Access arrays
cal_probs = data['cal_probs']      # (5000, 6) for 50/50 split
test_probs = data['test_probs']    # (5000, 6)
cal_labels = data['cal_labels']    # (5000,)
test_labels = data['test_labels']  # (5000,)

# Recompute metrics with custom conformal methods
# ... your custom analysis code ...
```

## Common Issues and Solutions

### Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python run_benchmark.py --mode long --tasks qa --models mistral-7b \
  --max-batch-size 1
```

### Dataset Download Timeout

**Issue**: CosmosQA download sometimes times out

**Solution**: Retry or download manually
```bash
# Retry the same command
python run_benchmark.py --mode long --tasks rc --models tinyllama-1.1b --resume

# Or pre-download datasets
python -c "from datasets import load_dataset; load_dataset('cosmos_qa')"
```

### Slow MPS Performance

**Solution**: Use smaller models or reduce batch size
```bash
python run_benchmark.py --mode short --tasks qa --models tinyllama-1.1b \
  --max-batch-size 2
```

### Tokenizer Warnings

Some models (e.g., Qwen) may produce tokenizer warnings. These are non-fatal but may affect results. Check CLAUDE.md for known issues.

## Citation

If you use this code or reproduce these benchmarks, please cite:

```
@inproceedings{ye2024benchmarking,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang, Mingming and Pang, Jianhui and Wang, Longyue and Wong, Derek F and Yilmaz, Emine and Shi, Shuming and Tu, Zhaopeng},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  year={2024}
}
```

## Support

For issues or questions:
1. Check CLAUDE.md for known issues and TODOs
2. Check README.md for project overview and quick start
3. Open an issue on the GitHub repository
