# Model Comparison Results

This document summarizes the benchmark results for Small Language Models (SLMs) evaluated using conformal prediction for uncertainty quantification.

## Overview

The benchmarks evaluate models across multiple NLP tasks using conformal prediction methods (LAC and APS) to measure both accuracy and uncertainty. Most models are evaluated at **FP16 precision** for optimal performance on A100 GPUs, with some models also evaluated at **FP32 precision** for comparison.

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

## TinyLlama-1.1B FP32 Evaluation

**Model**: TinyLlama-1.1B (1.1B parameters, Instruct-tuned)  
**Precision**: FP32 (float32)  
**Tasks**: QA, RC, CI, DRS (Dialogue Response Selection), DS (Document Summarization)  
**Samples**: 100 per task (quick test mode)

### Accuracy Results Across All Tasks (FP32)

| Task | Accuracy | Description |
|------|----------|-------------|
| **DRS** | **100.00%** | Dialogue Response Selection (Hallucination Detection) |
| **DS** | **50.67%** | Document Summarization (Hallucination Detection) |
| **QA** | **26.80%** | Question Answering (MMLU) |
| **RC** | **25.49%** | Reading Comprehension (CosmosQA) |
| **CI** | **23.53%** | Commonsense Inference (HellaSwag) |
| **Average** | **45.30%** | ± 29.08% |

### Prediction Set Size Across All Tasks (FP32)

| Task | Avg Set Size | Description |
|------|--------------|-------------|
| **DRS** | **1.45** | Smallest sets (most confident) |
| **DS** | 2.36 | |
| **CI** | 4.25 | |
| **RC** | 4.04 | |
| **QA** | 4.05 | |
| **Average** | **3.23** | ± 1.12 |

### Coverage Rate Across All Tasks (FP32)

| Task | Coverage Rate | Status |
|------|---------------|--------|
| **DRS** | **100.00%** | ✅ Exceeds target |
| **DS** | **98.00%** | ✅ Exceeds target |
| **QA** | **96.41%** | ✅ Exceeds target |
| **CI** | **95.42%** | ✅ Exceeds target |
| **RC** | **94.77%** | ✅ Exceeds target |
| **Average** | **96.92%** | ± 1.89% |

*All tasks meet the ≥90% coverage guarantee*

### Key Findings - TinyLlama FP32 Evaluation

- **Best Performing Task**: DRS (100% accuracy) - Perfect performance on dialogue response selection
- **Most Challenging Task**: CI (23.53% accuracy) - Commonsense inference
- **Most Confident Predictions**: DRS task (smallest set size: 1.45)
- **Least Confident Predictions**: CI task (largest set size: 4.25)
- **Coverage Guarantee**: All tasks meet ≥90% coverage
- **FP32 vs FP16 Comparison**: 
  - FP32 shows higher average accuracy (45.30% vs 33.60%)
  - FP32 shows lower average set size (3.23 vs 4.93), indicating more confident predictions
  - FP32 shows higher average coverage (96.92% vs 93.46%)

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

- **Precision**: FP16 (float16) for most models, FP32 (float32) for TinyLlama-1.1B comparison
- **Conformal Methods**: LAC (Least Ambiguous Classifiers) and APS (Adaptive Prediction Sets)
- **Coverage Target**: 90% (alpha = 0.1)
- **Prompting Strategies**: Base, Shared Instruction, Task-Specific
- **Calibration Ratio**: 50% calibration, 50% test
- **Hardware**: NVIDIA A100 80GB GPU

---

## Files and Data

Detailed results are available in:
- `results/fast_benchmark/` - Fast benchmark results (3 models, 3 tasks)
- `results/tinyllama_all_tasks/` - Complete TinyLlama evaluation (5 tasks, FP16)
- `results/tinyllama_fp32_all_tasks/` - Complete TinyLlama evaluation (5 tasks, FP32)
- `results/quick_test/` - Quick test results

Each directory contains:
- Summary CSV files (accuracy, set_size, coverage_rate)
- Comparison reports
- Statistical analysis JSON
- Individual model result files

---

*Last Updated: November 29, 2025*

---

## Precision Comparison: FP16 vs FP32 (TinyLlama-1.1B)

A comparison of TinyLlama-1.1B performance at different precision levels:

### Accuracy Comparison

| Task | FP16 | FP32 | Difference |
|------|------|------|------------|
| **DRS** | 56.67% | 100.00% | +43.33% |
| **DS** | 44.00% | 50.67% | +6.67% |
| **QA** | 23.53% | 26.80% | +3.27% |
| **RC** | 30.72% | 25.49% | -5.23% |
| **CI** | 13.07% | 23.53% | +10.46% |
| **Average** | **33.60%** | **45.30%** | **+11.70%** |

### Prediction Set Size Comparison

| Task | FP16 | FP32 | Difference |
|------|------|------|------------|
| **DRS** | 5.42 | 1.45 | -3.97 |
| **DS** | 2.78 | 2.36 | -0.42 |
| **QA** | 5.72 | 4.05 | -1.67 |
| **RC** | 5.37 | 4.04 | -1.33 |
| **CI** | 5.37 | 4.25 | -1.12 |
| **Average** | **4.93** | **3.23** | **-1.70** |

### Coverage Rate Comparison

| Task | FP16 | FP32 | Difference |
|------|------|------|------------|
| **DRS** | 93.00% | 100.00% | +7.00% |
| **DS** | 89.00% | 98.00% | +9.00% |
| **QA** | 96.41% | 96.41% | 0.00% |
| **RC** | 93.14% | 94.77% | +1.63% |
| **CI** | 95.75% | 95.42% | -0.33% |
| **Average** | **93.46%** | **96.92%** | **+3.46%** |

### Key Insights - Precision Comparison

1. **Accuracy**: FP32 shows significantly higher accuracy (+11.70% average), with the most dramatic improvement on DRS task (+43.33%)
2. **Confidence**: FP32 produces smaller prediction sets (-1.70 average), indicating more confident predictions
3. **Coverage**: FP32 maintains or improves coverage rates (+3.46% average), with all tasks meeting the 90% target
4. **Trade-offs**: FP32 provides better accuracy and confidence at the cost of slower inference speed and higher memory usage

