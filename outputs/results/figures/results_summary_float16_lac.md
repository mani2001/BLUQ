# BLUQ Benchmark Results (float16) - LAC

## Accuracy (%)

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 22.3 | 25.7 | 25.0 | 23.8 | 26.8 | **24.7** |
| gemma-2-9b-it | 70.8 | 30.7 | 30.3 | 28.8 | 38.5 | **39.8** |
| gemma-2b-it | 24.3 | 22.9 | 22.1 | 22.8 | 24.6 | **23.3** |
| mistral-7b | 27.1 | 30.6 | 28.3 | 25.2 | 55.4 | **33.3** |
| phi-2 | 22.5 | 24.6 | 22.3 | 22.6 | 35.3 | **25.5** |
| stablelm-2-1.6b | 20.5 | 21.9 | 22.2 | 21.3 | 23.6 | **21.9** |
| tinyllama-1.1b | 21.9 | 20.9 | 20.3 | 20.0 | 22.7 | **21.2** |

## Coverage Rate (%)

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 89.8 | 89.8 | 90.2 | 90.4 | 90.0 | **90.1** |
| gemma-2-9b-it | 90.6 | 89.8 | 90.3 | 90.2 | 91.2 | **90.4** |
| gemma-2b-it | 89.5 | 90.4 | 90.4 | 90.2 | 90.1 | **90.1** |
| mistral-7b | 89.9 | 89.2 | 90.2 | 89.2 | 89.9 | **89.7** |
| phi-2 | 90.1 | 90.7 | 89.4 | 89.6 | 89.5 | **89.9** |
| stablelm-2-1.6b | 89.6 | 90.5 | 90.0 | 89.5 | 90.3 | **90.0** |
| tinyllama-1.1b | 89.3 | 90.7 | 90.2 | 89.5 | 90.2 | **90.0** |

## Average Set Size

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 5.35 | 5.27 | 5.31 | 5.36 | 5.12 | **5.28** |
| gemma-2-9b-it | 2.60 | 4.78 | 4.91 | 4.95 | 4.15 | **4.28** |
| gemma-2b-it | 4.72 | 5.27 | 5.32 | 5.26 | 4.65 | **5.04** |
| mistral-7b | 4.95 | 4.87 | 4.96 | 4.97 | 4.04 | **4.76** |
| phi-2 | 5.30 | 5.20 | 5.17 | 5.25 | 5.02 | **5.19** |
| stablelm-2-1.6b | 5.26 | 5.28 | 5.25 | 5.24 | 5.03 | **5.21** |
| tinyllama-1.1b | 5.17 | 5.18 | 5.26 | 5.16 | 4.66 | **5.08** |