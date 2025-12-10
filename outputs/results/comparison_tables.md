# BLUQ Benchmark Results

Benchmarking Small Language Models via Uncertainty Quantification

---

## Summary Statistics

**Total models evaluated:** 8

- tinyllama-1.1b: 5 tasks, dtypes: ['float16', 'float32']
- stablelm-2-1.6b: 5 tasks, dtypes: ['float16', 'float32']
- gemma-2b-it: 5 tasks, dtypes: ['float16']
- gemma-2-2b-it: 5 tasks, dtypes: ['float16']
- phi-2: 5 tasks, dtypes: ['float16', 'float32']
- mistral-7b: 5 tasks, dtypes: ['float16', 'float32']
- mistral-7b-instruct: 5 tasks, dtypes: ['float16', 'float32']
- gemma-2-9b-it: 5 tasks, dtypes: ['float16', 'float32']

**Coverage guarantee (90%) achievement:**

- LAC: 32/70 (45.7%)
- APS: 54/70 (77.1%)

---

# Results (float16)

## Accuracy by Model and Task (float16)

| Model | Size | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 1.1B | 21.9% | 20.9% | 20.3% | 20.0% | 22.7% | **21.2%** |
| **stablelm-2-1.6b** | 1.6B | 20.5% | 21.9% | 22.2% | 21.3% | 23.6% | **21.9%** |
| **gemma-2b-it** | 2.0B | 24.3% | 22.9% | 22.1% | 22.8% | 24.6% | **23.3%** |
| **gemma-2-2b-it** | 2.0B | 22.3% | 25.7% | 25.0% | 23.8% | 26.8% | **24.7%** |
| **phi-2** | 2.7B | 21.7% | 23.8% | 22.4% | 22.0% | 32.4% | **24.4%** |
| **mistral-7b** | 7.0B | 27.1% | 30.6% | 28.3% | 25.2% | 55.4% | **33.3%** |
| **mistral-7b-instruct** | 7.0B | 26.5% | 30.3% | 27.1% | 25.1% | 55.7% | **32.9%** |
| **gemma-2-9b-it** | 9.0B | 70.8% | 30.7% | 30.3% | 28.8% | 38.5% | **39.8%** |

---

## Coverage Rate by Model and Task (LAC, float16)

Target: 90% coverage

| Model | QA | RC | CI | DRS | DS | Mean | Meets 90% |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 89.3%* | 90.7% | 90.2% | 89.5%* | 90.2% | **90.0%** | 3/5 |
| **stablelm-2-1.6b** | 89.6%* | 90.5% | 90.0% | 89.5%* | 90.3% | **90.0%** | 3/5 |
| **gemma-2b-it** | 89.5%* | 90.4% | 90.4% | 90.2% | 90.1% | **90.1%** | 4/5 |
| **gemma-2-2b-it** | 89.8%* | 89.8%* | 90.2% | 90.4% | 90.0% | **90.1%** | 3/5 |
| **phi-2** | 90.3% | 90.9% | 90.0%* | 89.6%* | 89.8%* | **90.1%** | 2/5 |
| **mistral-7b** | 89.9%* | 89.2%* | 90.2% | 89.2%* | 89.9%* | **89.7%** | 1/5 |
| **mistral-7b-instruct** | 89.9%* | 89.0%* | 89.9%* | 90.0% | 89.9%* | **89.7%** | 1/5 |
| **gemma-2-9b-it** | 90.6% | 89.8%* | 90.3% | 90.2% | 91.2% | **90.4%** | 4/5 |

*Coverage below 90% target

## Coverage Rate by Model and Task (APS, float16)

Target: 90% coverage

| Model | QA | RC | CI | DRS | DS | Mean | Meets 90% |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 90.2% | 90.9% | 92.0% | 89.8%* | 90.9% | **90.8%** | 4/5 |
| **stablelm-2-1.6b** | 98.2% | 90.2% | 92.0% | 90.0%* | 90.6% | **92.2%** | 4/5 |
| **gemma-2b-it** | 89.9%* | 90.7% | 98.5% | 90.8% | 89.7%* | **91.9%** | 3/5 |
| **gemma-2-2b-it** | 97.6% | 98.2% | 98.4% | 98.5% | 98.2% | **98.2%** | 5/5 |
| **phi-2** | 97.7% | 90.9% | 98.3% | 98.2% | 89.3%* | **94.9%** | 4/5 |
| **mistral-7b** | 90.5% | 89.0%* | 91.0% | 89.2%* | 90.7% | **90.1%** | 3/5 |
| **mistral-7b-instruct** | 89.9%* | 89.4%* | 92.0% | 88.8%* | 93.2% | **90.7%** | 2/5 |
| **gemma-2-9b-it** | 93.6% | 90.2% | 90.4% | 98.2% | 90.0%* | **92.5%** | 4/5 |

*Coverage below 90% target

---

## Average Prediction Set Size (LAC, float16)

Smaller set size = higher certainty (better)

| Model | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 5.17 | 5.18 | 5.26 | 5.16 | 4.66 | **5.08** |
| **stablelm-2-1.6b** | 5.26 | 5.28 | 5.25 | 5.24 | 5.03 | **5.21** |
| **gemma-2b-it** | 4.72 | 5.27 | 5.32 | 5.26 | 4.65 | **5.04** |
| **gemma-2-2b-it** | 5.35 | 5.27 | 5.31 | 5.36 | 5.12 | **5.28** |
| **phi-2** | 5.39 | 5.24 | 5.24 | 5.25 | 5.17 | **5.26** |
| **mistral-7b** | 4.95 | 4.87 | 4.96 | 4.97 | 4.04 | **4.76** |
| **mistral-7b-instruct** | 5.01 | 4.66 | 4.82 | 4.94 | 2.99 | **4.49** |
| **gemma-2-9b-it** | 2.60 | 4.78 | 4.91 | 4.95 | 4.15 | **4.28** |

## Average Prediction Set Size (APS, float16)

Smaller set size = higher certainty (better)

| Model | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 4.95 | 5.00 | 5.17 | 4.92 | 4.36 | **4.88** |
| **stablelm-2-1.6b** | 5.85 | 5.04 | 5.19 | 4.89 | 4.85 | **5.16** |
| **gemma-2b-it** | 4.74 | 5.17 | 5.86 | 5.16 | 4.55 | **5.09** |
| **gemma-2-2b-it** | 5.84 | 5.87 | 5.87 | 5.86 | 5.86 | **5.86** |
| **phi-2** | 5.84 | 5.17 | 5.85 | 5.83 | 4.94 | **5.52** |
| **mistral-7b** | 4.98 | 4.89 | 4.86 | 4.82 | 3.90 | **4.69** |
| **mistral-7b-instruct** | 5.00 | 4.89 | 5.17 | 4.87 | 4.00 | **4.79** |
| **gemma-2-9b-it** | 3.60 | 4.86 | 4.99 | 5.85 | 4.40 | **4.74** |

---

## LAC vs APS Comparison (float16)

| Model | LAC Coverage | APS Coverage | LAC Set Size | APS Set Size | Winner |
|---|---|---|---|---|---|
| **tinyllama-1.1b** | 90.0% | 90.8% | 5.08 | 4.88 | APS |
| **stablelm-2-1.6b** | 90.0% | 92.2% | 5.21 | 5.16 | APS |
| **gemma-2b-it** | 90.1% | 91.9% | 5.04 | 5.09 | LAC |
| **gemma-2-2b-it** | 90.1% | 98.2% | 5.28 | 5.86 | LAC |
| **phi-2** | 90.1% | 94.9% | 5.26 | 5.52 | LAC |
| **mistral-7b** | 89.7% | 90.1% | 4.76 | 4.69 | APS |
| **mistral-7b-instruct** | 89.7% | 90.7% | 4.49 | 4.79 | APS |
| **gemma-2-9b-it** | 90.4% | 92.5% | 4.28 | 4.74 | LAC |

---

## Base vs Instruction-Tuned Comparison (float16)

### Mistral-7B Family

| Task | Method | Base Acc | Inst Acc | Base SetSize | Inst SetSize | Inst More Uncertain |
|---|---|---|---|---|---|---|
| QA | LAC | 27.1% | 26.5% | 4.95 | 5.01 | Yes |
| RC | LAC | 30.6% | 30.3% | 4.87 | 4.66 | No |
| CI | LAC | 28.3% | 27.1% | 4.96 | 4.82 | No |
| DRS | LAC | 25.2% | 25.1% | 4.97 | 4.94 | No |
| DS | LAC | 55.4% | 55.7% | 4.04 | 2.99 | No |
| QA | APS | 27.1% | 26.5% | 4.98 | 5.00 | Yes |
| RC | APS | 30.6% | 30.3% | 4.89 | 4.89 | No |
| CI | APS | 28.3% | 27.1% | 4.86 | 5.17 | Yes |
| DRS | APS | 25.2% | 25.1% | 4.82 | 4.87 | Yes |
| DS | APS | 55.4% | 55.7% | 3.90 | 4.00 | Yes |


# Results (float32)

## Accuracy by Model and Task (float32)

| Model | Size | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 1.1B | 22.1% | 22.0% | 20.6% | 21.4% | 22.6% | **21.8%** |
| **stablelm-2-1.6b** | 1.6B | 22.3% | 28.2% | 30.4% | 22.6% | 23.9% | **25.5%** |
| **phi-2** | 2.7B | 26.6% | 30.6% | 23.3% | 23.5% | 33.0% | **27.4%** |
| **mistral-7b** | 7.0B | 39.2% | 51.6% | 45.0% | 38.4% | 62.6% | **47.4%** |
| **mistral-7b-instruct** | 7.0B | 38.0% | 52.3% | 45.0% | 39.2% | 62.6% | **47.4%** |
| **gemma-2-9b-it** | 9.0B | 70.7% | 81.5% | 79.5% | 63.9% | 63.7% | **71.9%** |

---

## Coverage Rate by Model and Task (LAC, float32)

Target: 90% coverage

| Model | QA | RC | CI | DRS | DS | Mean | Meets 90% |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 89.4%* | 89.7%* | 90.2% | 89.7%* | 89.8%* | **89.8%** | 1/5 |
| **stablelm-2-1.6b** | 90.3% | 90.1% | 88.6%* | 89.5%* | 90.0% | **89.7%** | 3/5 |
| **phi-2** | 89.2%* | 89.5%* | 89.1%* | 89.8%* | 89.8%* | **89.5%** | 0/5 |
| **mistral-7b** | 90.1% | 89.9%* | 90.3% | 90.9% | 91.3% | **90.5%** | 4/5 |
| **mistral-7b-instruct** | 89.7%* | 89.1%* | 90.2% | 90.0% | 89.8%* | **89.8%** | 2/5 |
| **gemma-2-9b-it** | 90.7% | 89.6%* | 89.7%* | 89.5%* | 88.9%* | **89.7%** | 1/5 |

*Coverage below 90% target

## Coverage Rate by Model and Task (APS, float32)

Target: 90% coverage

| Model | QA | RC | CI | DRS | DS | Mean | Meets 90% |
|---|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 99.7% | 89.8%* | 100.0% | 98.5% | 97.2% | **97.0%** | 4/5 |
| **stablelm-2-1.6b** | 91.7% | 89.9%* | 90.3% | 89.5%* | 90.6% | **90.4%** | 3/5 |
| **phi-2** | 98.2% | 90.7% | 98.0% | 97.9% | 89.2%* | **94.8%** | 4/5 |
| **mistral-7b** | 89.5%* | 90.9% | 91.0% | 91.1% | 91.6% | **90.8%** | 4/5 |
| **mistral-7b-instruct** | 90.1% | 91.0% | 90.5% | 90.5% | 93.9% | **91.2%** | 5/5 |
| **gemma-2-9b-it** | 93.7% | 94.9% | 93.8% | 90.1% | 92.4% | **93.0%** | 5/5 |

*Coverage below 90% target

---

## Average Prediction Set Size (LAC, float32)

Smaller set size = higher certainty (better)

| Model | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 5.14 | 5.10 | 5.22 | 5.19 | 4.37 | **5.01** |
| **stablelm-2-1.6b** | 5.18 | 5.17 | 4.92 | 5.32 | 4.99 | **5.12** |
| **phi-2** | 5.22 | 5.04 | 5.16 | 5.24 | 5.17 | **5.17** |
| **mistral-7b** | 4.17 | 3.72 | 4.17 | 4.27 | 3.64 | **3.99** |
| **mistral-7b-instruct** | 4.23 | 3.27 | 3.74 | 4.30 | 2.56 | **3.62** |
| **gemma-2-9b-it** | 2.60 | 1.49 | 1.73 | 3.67 | 2.13 | **2.33** |

## Average Prediction Set Size (APS, float32)

Smaller set size = higher certainty (better)

| Model | QA | RC | CI | DRS | DS | Mean |
|---|---|---|---|---|---|---|
| **tinyllama-1.1b** | 5.96 | 4.94 | 6.00 | 5.90 | 5.26 | **5.61** |
| **stablelm-2-1.6b** | 5.19 | 4.97 | 4.91 | 4.89 | 4.84 | **4.96** |
| **phi-2** | 5.84 | 4.97 | 5.84 | 5.84 | 4.92 | **5.48** |
| **mistral-7b** | 4.29 | 4.11 | 3.97 | 4.18 | 3.54 | **4.02** |
| **mistral-7b-instruct** | 4.58 | 4.26 | 4.13 | 4.49 | 3.83 | **4.26** |
| **gemma-2-9b-it** | 3.60 | 3.33 | 3.17 | 4.08 | 3.12 | **3.46** |

---

## LAC vs APS Comparison (float32)

| Model | LAC Coverage | APS Coverage | LAC Set Size | APS Set Size | Winner |
|---|---|---|---|---|---|
| **tinyllama-1.1b** | 89.8% | 97.0% | 5.01 | 5.61 | APS |
| **stablelm-2-1.6b** | 89.7% | 90.4% | 5.12 | 4.96 | APS |
| **phi-2** | 89.5% | 94.8% | 5.17 | 5.48 | APS |
| **mistral-7b** | 90.5% | 90.8% | 3.99 | 4.02 | LAC |
| **mistral-7b-instruct** | 89.8% | 91.2% | 3.62 | 4.26 | APS |
| **gemma-2-9b-it** | 89.7% | 93.0% | 2.33 | 3.46 | APS |

---

## Base vs Instruction-Tuned Comparison (float32)

### Mistral-7B Family

| Task | Method | Base Acc | Inst Acc | Base SetSize | Inst SetSize | Inst More Uncertain |
|---|---|---|---|---|---|---|
| QA | LAC | 39.2% | 38.0% | 4.17 | 4.23 | Yes |
| RC | LAC | 51.6% | 52.3% | 3.72 | 3.27 | No |
| CI | LAC | 45.0% | 45.0% | 4.17 | 3.74 | No |
| DRS | LAC | 38.4% | 39.2% | 4.27 | 4.30 | Yes |
| DS | LAC | 62.6% | 62.6% | 3.64 | 2.56 | No |
| QA | APS | 39.2% | 38.0% | 4.29 | 4.58 | Yes |
| RC | APS | 51.6% | 52.3% | 4.11 | 4.26 | Yes |
| CI | APS | 45.0% | 45.0% | 3.97 | 4.13 | Yes |
| DRS | APS | 38.4% | 39.2% | 4.18 | 4.49 | Yes |
| DS | APS | 62.6% | 62.6% | 3.54 | 3.83 | Yes |

