# A100 Speed Optimization Guide

## 🚀 Your Setup

**GPU**: NVIDIA A100 80GB PCIe
**Memory**: 85.09 GB
**Status**: ✅ Ready for benchmarking

---

## ⚡ Speed Optimizations Applied

### 1. **Large Batch Sizes**
With 80GB of memory, we can process many samples simultaneously:
- **SmolLM-135M**: Batch size 128 (vs default 1)
- **SmolLM-360M**: Batch size 64
- **TinyLlama-1.1B**: Batch size 32
- **Phi-2/Gemma-2B**: Batch size 16

**Expected speedup**: 10-50x depending on model size

### 2. **FP16 Precision**
All models use `float16` instead of `float32`:
- **Memory savings**: 50%
- **Speed improvement**: 2-3x on A100
- **Accuracy impact**: Minimal (< 0.1%)

### 3. **CUDA Optimizations**
Environment variables set in `quick_start_a100.sh`:
```bash
export CUDA_LAUNCH_BLOCKING=0  # Async kernel launches
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better memory allocation
```

### 4. **Optimized Model Loading**
- `use_fast_tokenizer: true` - Faster tokenization
- `use_cache: true` - KV cache for faster generation
- `device: "cuda"` - Force GPU usage

### 5. **Reduced I/O**
- `save_logits: false` - Skip saving large logit files
- Only save essential results

---

## 📊 Expected Performance

### Quick Test (100 samples, 1 model)
- **Time**: ~2 minutes
- **Purpose**: Verify everything works

### Short Benchmark (100 samples per task)
- **Time**: ~5-10 minutes
- **Purpose**: Get quick results for testing
- **Models**: TinyLlama-1.1B, Phi-2
- **Tasks**: QA, RC, CI, DRS, DS

### Long Benchmark (10,000 samples per task) - ACTUAL RESULTS
- **Time**: ~92 minutes (1.5 hours)
- **Purpose**: Production-quality evaluation
- **Models**: TinyLlama-1.1B, Phi-2
- **Tasks**: 5 tasks (QA, RC, CI, DRS, DS)
- **Conformal Methods**: LAC and APS
- **Results**:
  - Overall Accuracy: 31.7%
  - Overall Coverage: 94.1%
  - Average Set Size: 5.34

### Full Benchmark (All models, all tasks)
- **Time**: ~4-6 hours
- **Purpose**: Complete evaluation across all supported models
- **Models**: 16+ models (135M to 2.7B parameters)
- **Tasks**: 5 tasks (QA, RC, CI, DRS, DS)

---

## 🎯 Quick Start Commands

### Option 1: Interactive Script (Recommended)
```bash
./quick_start_a100.sh
```

### Option 2: Direct Commands

**Quick Test:**
```bash
source .venv/bin/activate
python run_single_task.py --task qa --model smollm-135m --num-samples 100
```

**Fast Benchmark:**
```bash
source .venv/bin/activate
python run_benchmark.py \
  --tasks qa rc ci \
  --models smollm-135m smollm-360m tinyllama-1.1b \
  --data-config dataset_config_fast.yaml
```

**Full Benchmark:**
```bash
source .venv/bin/activate
python run_benchmark.py --tasks qa rc ci drs ds
```

---

## 💡 Additional Speed Tips

### 1. **Start with Smallest Models**
Models are ordered by speed in the config:
- Tier 1 (Ultra-fast): 135M-450M parameters
- Tier 2 (Fast): 1.1B-1.8B parameters  
- Tier 3 (Medium): 2B-2.7B parameters

### 2. **Use Fewer Tasks Initially**
- `qa` (MMLU) - Most diverse, good for testing
- `rc` (CosmosQA) - Reading comprehension
- `ci` (HellaSwag) - Commonsense reasoning

### 3. **Monitor GPU Usage**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 4. **Reduce Sample Sizes for Testing**
Edit `dataset_config_fast.yaml` to use even fewer samples:
```yaml
num_samples: 100  # Instead of 1000
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
Reduce batch sizes in `model_config_a100_optimized.yaml`:
```yaml
inference_config:
  batch_size: 8  # Reduce from 16/32/64
```

### Slow Download Speeds
Models are downloaded from HuggingFace on first run. This is one-time:
- SmolLM-135M: ~300MB
- TinyLlama-1.1B: ~2GB
- Phi-2: ~5GB

### CUDA Errors
Ensure you're using the virtual environment:
```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📈 Monitoring Progress

Results are saved to:
```
results/
├── models/
│   ├── smollm-135m/
│   │   ├── qa_results.json
│   │   └── ...
│   └── ...
├── summary_accuracy.csv
├── summary_set_size.csv
└── comparison_report.txt
```

View results in real-time:
```bash
tail -f results/comparison_report.txt
```

---

## 🎓 Understanding the Results

### Key Metrics

1. **Accuracy**: % of correct predictions (higher is better)
2. **Coverage**: % of true answers in prediction sets (should be ≥90%)
3. **Set Size**: Average # of options in prediction set (smaller is better)
4. **ECE**: Expected Calibration Error (lower is better)

### What to Look For

- **High accuracy + small set size** = Confident and correct model
- **High accuracy + large set size** = Correct but uncertain model
- **Low accuracy + small set size** = Overconfident model (dangerous!)
- **Coverage < 90%** = Conformal prediction not working properly

### Actual Results from Long Benchmark (10,000 samples)

| Task | Accuracy | Set Size (LAC) | Interpretation |
|------|----------|----------------|----------------|
| **DRS** | 45-49% | 4.8-5.0 | Best performance - pattern matching |
| **DS** | 48% | 3.9-4.3 | Good accuracy, smallest sets |
| **QA/RC/CI** | ~21% | 5.1-5.3 | Near-random, high uncertainty |

**Key Finding**: SLMs show appropriate uncertainty - large prediction sets when accuracy is low, smaller sets when accuracy is higher.

---

## 🚀 Next Steps

1. **Run Quick Test** to verify setup
2. **Run Fast Benchmark** to get initial results
3. **Analyze results** in `results/` directory
4. **Run Full Benchmark** for complete evaluation
5. **Compare with paper results** in `paper.pdf`

---

## 📚 Files Created

- `model_config_a100_optimized.yaml` - Optimized model configuration
- `dataset_config_fast.yaml` - Fast dataset configuration  
- `quick_start_a100.sh` - Interactive quick start script
- `A100_OPTIMIZATION_GUIDE.md` - This file

---

**Ready to start?** Run:
```bash
./quick_start_a100.sh
```
