# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLUQ (Benchmarking Language models via Uncertainty Quantification) evaluates Small Language Models (SLMs) using conformal prediction for uncertainty quantification. It implements the methodology from "Benchmarking LLMs via Uncertainty Quantification" (Ye et al., 2024).

## Supported Platforms

- **NVIDIA GPUs (CUDA)**: Full support with GPU profiling and dynamic batch sizing
- **Apple Silicon (MPS)**: Full support for Mac M1/M2/M3 chips using PyTorch MPS backend
- **CPU**: Fallback support (significantly slower)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Verify compute environment (GPU/MPS/CPU)
python verify_gpu_ready.py

# Run single task evaluation (for debugging/testing)
python run_single_task.py --task qa --model tinyllama-1.1b --num-samples 100

# Run short benchmark (100 samples - for testing)
python run_benchmark.py --mode short --tasks qa --models tinyllama-1.1b

# Run full benchmark (10,000 samples - production)
python run_benchmark.py --mode long --tasks qa rc ci drs ds --models tinyllama-1.1b phi-2

# Run tests
pytest tests/ -v

# Run single test file
pytest tests/test_conformal.py -v
```

## Mac-Specific Instructions

### Requirements for Apple Silicon (M1/M2/M3)

1. **Python**: Use Python 3.10+ (recommended via Homebrew or pyenv)
2. **PyTorch with MPS**: Install PyTorch with MPS support:
   ```bash
   pip install torch torchvision torchaudio
   ```
3. **Memory**: Models are loaded into unified memory. Recommended:
   - 8GB RAM: Use smollm-135m, smollm-360m
   - 16GB RAM: Use tinyllama-1.1b, phi-1.5
   - 32GB+ RAM: Use phi-2, gemma-2b, stablelm-2-1.6b

### Running on Mac

The benchmark automatically detects MPS and configures appropriately:

```bash
# Verify MPS is available
python verify_gpu_ready.py

# Run a quick test (smaller model recommended for Mac)
python run_benchmark.py --mode short --tasks qa --models tinyllama-1.1b --max-batch-size 4

# For limited memory, use smaller batch sizes
python run_benchmark.py --mode short --tasks qa --models smollm-135m --max-batch-size 2
```

### Mac Performance Notes

- GPU profiling is disabled on MPS (CUDA-only feature)
- Use `float32` dtype for better MPS compatibility if you encounter issues
- Reduce batch size if you encounter memory pressure
- MPS is generally 2-5x slower than A100 but still faster than CPU

## Architecture

### Entry Points
- `run_benchmark.py` - Main benchmark runner with checkpointing, GPU profiling, and multi-model/task support
- `run_single_task.py` - Single model-task evaluation for debugging

### Core Pipeline (src/)

**Data Layer (`src/data/`)**
- `DatasetLoaderFactory` creates task-specific loaders for 5 tasks: qa (MMLU), rc (CosmosQA), ci (HellaSwag), drs (HaluDial), ds (HaluSum)
- `DatasetProcessor` standardizes all tasks to 6-option format (A-D + E "I don't know" + F "None of the above")
- `DataSplitter` creates calibration/test splits (default 50/50)

**Model Layer (`src/models/`)**
- `ModelLoader` handles HuggingFace model loading with dtype/device configuration
- `InferenceEngine` runs batched inference with dynamic batch sizing
- `ProbabilityExtractor` extracts next-token probabilities for answer options

**Prompting (`src/prompting/`)**
- `PromptBuilder` constructs prompts with 3 strategies: base, shared_instruction, task_specific
- `DemonstrationManager` selects few-shot examples per task

**Conformal Prediction (`src/conformal/`)**
- `LACScorer` - Least Ambiguous Classifiers: score = 1 - P(true_class)
- `APSScorer` - Adaptive Prediction Sets: score = cumulative prob to reach true class
- `PredictionSetGenerator` calibrates thresholds and generates prediction sets with coverage guarantees

**Evaluation (`src/evaluation/`)**
- Computes accuracy, coverage rate, average set size, and whether 90% coverage guarantee is met

### Key Data Flow
1. Load dataset → Process to 6-option format → Split calibration/test
2. Load model → Build prompts with demonstrations → Run inference
3. Extract probabilities → Calibrate conformal threshold on calibration set
4. Generate prediction sets on test set → Compute metrics

### Available Models
tinyllama-1.1b, phi-2, stablelm-2-1.6b, gemma-2b, qwen-1.8b

### Available Tasks
- qa: Question Answering (MMLU)
- rc: Reading Comprehension (CosmosQA)
- ci: Commonsense Inference (HellaSwag)
- drs: Dialogue Response Selection (HaluDial)
- ds: Document Summarization (HaluSum)

## Configuration

- `configs/model_config.yaml` - Model IDs, dtypes, device settings
- `configs/dataset_config.yaml` - Task configs, sample counts, demonstration counts

## Output Structure

Results are saved to `outputs/` with:
- `results_*.json` - Raw benchmark results
- `summary_*.json` - Aggregated statistics
- `checkpoint.json` - Resume checkpoint for interrupted runs
- `probabilities/` - Raw probability arrays (.npz) for post-hoc analysis
- `figures/` - Visualization plots (dashboard, heatmaps, radar charts)
- `logs/` - Detailed run logs

### Raw Probability Files

Each run saves raw probabilities to `outputs/probabilities/probs_{model}_{task}_{dtype}_{strategy}_{timestamp}.npz` containing:
- `cal_probs`: Calibration set probabilities (n_cal, 6)
- `test_probs`: Test set probabilities (n_test, 6)
- `cal_labels`: Calibration set true labels (n_cal,)
- `test_labels`: Test set true labels (n_test,)
- `option_letters`: ['A', 'B', 'C', 'D', 'E', 'F']
- Metadata: model, task, dtype, strategy, timestamp, alpha, calibration_ratio

Load with: `data = np.load('probs_*.npz'); cal_probs = data['cal_probs']`

## Key Implementation Details

- Conformal prediction threshold: quantile at ceil((n+1)(1-alpha))/n of calibration scores
- Default alpha=0.1 targets 90% coverage guarantee
- Dynamic batch sizing based on GPU memory (auto-detected by GPU tier)
- Checkpoint/resume support via `--resume` flag

## TODOs

### Pending Benchmarks
- [ ] Qwen-1.8B: Tokenizer compatibility issue ("Adding unknown special tokens is not supported")
- [ ] TinyLlama-1.1B float32 RC task: Network timeout during dataset download

### Completed Benchmarks (A100 80GB, 10k samples)
- TinyLlama-1.1B: float16 5/5, float32 4/5
- Phi-2: float16 5/5
- StableLM-2-1.6B: float16 5/5
- Gemma-2B-IT: float16 5/5
- Gemma-2-2B-IT: float16 5/5
- Gemma-2-9B-IT: float16 5/5