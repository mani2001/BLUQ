# BLUQ Execution Guide for A100

## Quick Validation Run (100 samples, 5 models)

### Pre-flight Checklist

```bash
# 1. Clone repo (on A100 instance)
git clone https://github.com/YOUR_USERNAME/BLUQ.git
cd BLUQ

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install pynvml for GPU utilization monitoring
pip install pynvml

# 4. Verify GPU is ready
python verify_gpu_ready.py
```

---

## Run Models One-by-One (Memory Safe)

Run each model separately to avoid OOM. Results accumulate in `./outputs/quick_validation/`.

### Model 1: TinyLlama (1.1B)
```bash
python run_full_benchmark.py \
    --mode short \
    --models tinyllama-1.1b \
    --tasks qa rc ci drs ds \
    --strategies base \
    --output-dir ./outputs/quick_validation \
    --resume
```

### Model 2: Phi-2 (2.7B)
```bash
python run_full_benchmark.py \
    --mode short \
    --models phi-2 \
    --tasks qa rc ci drs ds \
    --strategies base \
    --output-dir ./outputs/quick_validation \
    --resume
```

### Model 3: StableLM-2 (1.6B)
```bash
python run_full_benchmark.py \
    --mode short \
    --models stablelm-2-1.6b \
    --tasks qa rc ci drs ds \
    --strategies base \
    --output-dir ./outputs/quick_validation \
    --resume
```

### Model 4: Gemma-2B
```bash
python run_full_benchmark.py \
    --mode short \
    --models gemma-2b \
    --tasks qa rc ci drs ds \
    --strategies base \
    --output-dir ./outputs/quick_validation \
    --resume
```

### Model 5: Qwen-1.8B
```bash
python run_full_benchmark.py \
    --mode short \
    --models qwen-1.8b \
    --tasks qa rc ci drs ds \
    --strategies base \
    --output-dir ./outputs/quick_validation \
    --resume
```

---

## One-Liner (All 5 Models Sequentially)

```bash
for model in tinyllama-1.1b phi-2 stablelm-2-1.6b gemma-2b qwen-1.8b; do
    echo "========== Running $model =========="
    python run_full_benchmark.py \
        --mode short \
        --models $model \
        --tasks qa rc ci drs ds \
        --strategies base \
        --output-dir ./outputs/quick_validation \
        --resume
done
```

---

## Push Results to GitHub

After all models complete:

```bash
# Add results
git add outputs/quick_validation/

# Commit
git commit -m "Quick validation results: 5 models x 5 tasks x 100 samples

Models: tinyllama-1.1b, phi-2, stablelm-2-1.6b, gemma-2b, qwen-1.8b
Tasks: qa, rc, ci, drs, ds
Samples: 100 per task
Strategy: base

🤖 Generated with Claude Code"

# Push
git push origin main
```

---

## Pull Results Locally (For Analysis)

On your local machine:

```bash
cd BLUQ
git pull origin main
```

Results will be in:
- `outputs/quick_validation/results_*.json` - Raw results
- `outputs/quick_validation/summary_*.json` - Summary stats
- `outputs/quick_validation/checkpoint.json` - Progress checkpoint
- `outputs/quick_validation/figures/` - Visualizations (if generated)

---

## Expected Output Structure

```
outputs/quick_validation/
├── checkpoint.json              # Resume checkpoint
├── config_YYYYMMDD_HHMMSS.json
├── results_YYYYMMDD_HHMMSS.json
├── summary_YYYYMMDD_HHMMSS.json
├── gpu_profile_YYYYMMDD_HHMMSS.json  # GPU profiling report
└── figures/
    ├── accuracy_by_model.png
    ├── coverage_by_task.png
    └── ...
```

---

## GPU Profiling Output

The benchmark automatically logs:
- **Timing per operation**: model loading, inference, probability extraction
- **GPU memory usage**: before/after each operation, peak usage
- **Throughput**: samples/second for inference
- **Bottleneck detection**: identifies slow operations (>20% of total time)

### Sample profiling output:
```
================================================================================
GPU PROFILING SUMMARY
================================================================================

Total Operations: 15
Total Time: 45230.50ms (45.23s)

--- Operations Breakdown ---

  [inference_calibration]
    Count: 5
    Total: 25000.00ms (55.3%)
    Avg: 5000.00ms | Min: 4500.00ms | Max: 5500.00ms
    Avg Memory Delta: +0.0MB

  [model_loading]
    Count: 5
    Total: 15000.00ms (33.2%)
    Avg: 3000.00ms | Min: 2800.00ms | Max: 3200.00ms
    Avg Memory Delta: +2500.0MB

  [inference_test]
    Count: 5
    Total: 4000.00ms (8.8%)
    ...

--- Memory Statistics ---
  Peak Allocated: 5500.0MB
  Avg Allocated: 4200.0MB
  Avg GPU Utilization: 75.2%

--- Bottlenecks & Recommendations ---

  [WARN] inference_calibration (55.3% of total time)
     -> Inference timing is reasonable.

  [WARN] model_loading (33.2% of total time)
     -> Model loading is expected to be slow. Consider caching or using quantization.
================================================================================
```

### Disable profiling (if overhead is a concern):
```bash
python run_full_benchmark.py --mode short --models MODEL --no-profiling ...
```

---

## Troubleshooting

### If a model fails (OOM, timeout, etc.)
Just re-run the same command with `--resume`. It will skip completed models.

### If CosmosQA (rc task) fails
Skip it:
```bash
python run_full_benchmark.py --mode short --models MODEL --tasks qa ci drs ds ...
```

### Check GPU memory
```bash
nvidia-smi
```

### Clear GPU cache if needed
```python
import torch
torch.cuda.empty_cache()
```

---

## Estimated Time

| Model | ~Time (100 samples x 5 tasks) |
|-------|------------------------------|
| TinyLlama-1.1B | 5-10 min |
| Phi-2 | 10-15 min |
| StableLM-2-1.6B | 8-12 min |
| Gemma-2B | 10-15 min |
| Qwen-1.8B | 8-12 min |

**Total: ~45-60 minutes** for all 5 models
