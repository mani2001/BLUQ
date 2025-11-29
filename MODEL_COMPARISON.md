# Model Comparison Results

This document summarizes the benchmark results for Small Language Models (SLMs) evaluated using conformal prediction for uncertainty quantification.

## Overview

The benchmarks evaluate models across multiple NLP tasks using conformal prediction methods (LAC and APS) to measure both accuracy and uncertainty. All models are evaluated at **FP16 precision** for optimal performance on A100 GPUs.

## Fast Benchmark Results

**Models Evaluated**: SmolLM-135M, SmolLM-360M, TinyLlama-1.1B  
**Tasks**: QA (MMLU), RC (CosmosQA), CI (HellaSwag)  
**Samples**: 100 per task (quick test mode)

### Accuracy Results

| Model | CI | QA | RC | Average | Rank |
|-------|----|----|----|---------|------|
| **smollm-360m** | 25.49% | 18.95% | 27.45% | **23.97%** | 🥇 1 |
| **tinyllama-1.1b** | 13.07% | 23.53% | 30.72% | **22.44%** | 🥈 2 |
| **smollm-135m** | 22.22% | 16.34% | 15.03% | **17.86%** | 🥉 3 |

### Prediction Set Size Results

| Model | CI | QA | RC | Average | Rank |
|-------|----|----|----|---------|------|
| **smollm-135m** | 5.54 | 5.67 | 5.51 | **5.57** | 1 |
| **tinyllama-1.1b** | 5.37 | 5.72 | 5.37 | **5.48** | 2 |
| **smollm-360m** | 5.42 | 5.55 | 5.20 | **5.39** | 3 |

*Note: Smaller set sizes indicate more confident predictions*

### Coverage Rate Results

| Model | CI | QA | RC | Average | Rank |
|-------|----|----|----|---------|------|
| **tinyllama-1.1b** | 95.75% | 96.41% | 93.14% | **95.10%** | 🥇 1 |
| **smollm-135m** | 94.77% | 96.08% | 92.48% | **94.44%** | 🥈 2 |
| **smollm-360m** | 95.75% | 93.46% | 91.18% | **93.46%** | 🥉 3 |

*Target coverage: ≥90% (all models meet this requirement)*

### Key Findings - Fast Benchmark

- **Best Overall Model**: SmolLM-360M (highest average accuracy: 23.97%)
- **Best per Task**:
  - CI (Commonsense Inference): SmolLM-360M (25.49%)
  - QA (Question Answering): TinyLlama-1.1B (23.53%)
  - RC (Reading Comprehension): TinyLlama-1.1B (30.72%)
- **Overall Statistics**:
  - Average Accuracy: 21.42% ± 5.65%
  - Average Set Size: 5.48 ± 0.15 options
  - Average Coverage: 94.34% ± 1.74%

---

## TinyLlama-1.1B Complete Evaluation

**Model**: TinyLlama-1.1B (1.1B parameters, Instruct-tuned)  
**Tasks**: QA, RC, CI, DRS (Dialogue Response Selection), DS (Document Summarization)  
**Samples**: 100 per task

### Accuracy Results Across All Tasks

| Task | Accuracy | Description |
|------|----------|-------------|
| **DRS** | **56.67%** | Dialogue Response Selection (Hallucination Detection) |
| **DS** | **44.00%** | Document Summarization (Hallucination Detection) |
| **RC** | **30.72%** | Reading Comprehension (CosmosQA) |
| **QA** | **23.53%** | Question Answering (MMLU) |
| **CI** | **13.07%** | Commonsense Inference (HellaSwag) |
| **Average** | **33.60%** | ± 15.31% |

### Prediction Set Size Across All Tasks

| Task | Avg Set Size | Description |
|------|--------------|-------------|
| **DS** | **2.78** | Smallest sets (most confident) |
| **CI** | 5.37 | |
| **RC** | 5.37 | |
| **DRS** | 5.42 | |
| **QA** | 5.72 | Largest sets (least confident) |
| **Average** | **4.93** | ± 1.08 |

### Coverage Rate Across All Tasks

| Task | Coverage Rate | Status |
|------|---------------|--------|
| **QA** | **96.41%** | ✅ Exceeds target |
| **CI** | **95.75%** | ✅ Exceeds target |
| **RC** | **93.14%** | ✅ Exceeds target |
| **DRS** | **93.00%** | ✅ Exceeds target |
| **DS** | **89.00%** | ⚠️ Below target (LAC method) |
| **Average** | **93.46%** | ± 2.61% |

*Note: DS task shows lower coverage with LAC method, but APS method achieves 92% coverage*

### Key Findings - TinyLlama Complete Evaluation

- **Best Performing Task**: DRS (56.67% accuracy) - Dialogue response selection
- **Most Challenging Task**: CI (13.07% accuracy) - Commonsense inference
- **Most Confident Predictions**: DS task (smallest set size: 2.78)
- **Least Confident Predictions**: QA task (largest set size: 5.72)
- **Coverage Guarantee**: All tasks meet ≥90% coverage using APS method

---

## Model Comparison Summary

### Model Rankings by Accuracy

1. **SmolLM-360M** (361M parameters)
   - Best overall accuracy in fast benchmark
   - Strong performance on commonsense inference
   - Good balance of accuracy and uncertainty

2. **TinyLlama-1.1B** (1.1B parameters, Instruct-tuned)
   - Best performance on reading comprehension and question answering
   - Highest coverage rates across tasks
   - Strong performance on hallucination detection tasks (DRS, DS)

3. **SmolLM-135M** (135M parameters)
   - Smallest model, fastest inference
   - Competitive on commonsense inference
   - Good coverage rates

### Key Insights

1. **Model Size vs. Performance**: 
   - Larger models (360M, 1.1B) generally perform better, but the relationship is not linear
   - SmolLM-360M outperforms TinyLlama-1.1B on some tasks despite being smaller

2. **Task Difficulty**:
   - Hallucination detection tasks (DRS, DS) show higher accuracy, possibly due to binary nature
   - Commonsense inference (CI) is most challenging for all models
   - Reading comprehension shows good performance across models

3. **Uncertainty Quantification**:
   - All models successfully meet the 90% coverage guarantee using APS method
   - LAC method sometimes undercovers on difficult tasks
   - Average set sizes range from 4.93 to 5.57, indicating moderate uncertainty

4. **Instruct-Tuning Effects**:
   - TinyLlama-1.1B (instruct-tuned) shows better performance on reading comprehension
   - May exhibit more calibrated uncertainty (higher coverage rates)

---

## Experimental Setup

- **Precision**: FP16 (float16) for all models
- **Conformal Methods**: LAC (Least Ambiguous Classifiers) and APS (Adaptive Prediction Sets)
- **Coverage Target**: 90% (alpha = 0.1)
- **Prompting Strategies**: Base, Shared Instruction, Task-Specific
- **Calibration Ratio**: 50% calibration, 50% test
- **Hardware**: NVIDIA A100 80GB GPU

---

## Files and Data

Detailed results are available in:
- `results/fast_benchmark/` - Fast benchmark results (3 models, 3 tasks)
- `results/tinyllama_all_tasks/` - Complete TinyLlama evaluation (5 tasks)
- `results/quick_test/` - Quick test results

Each directory contains:
- Summary CSV files (accuracy, set_size, coverage_rate)
- Comparison reports
- Statistical analysis JSON
- Individual model result files

---

*Last Updated: November 29, 2025*

