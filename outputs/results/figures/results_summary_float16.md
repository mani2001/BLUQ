# BLUQ Benchmark Results (float16)

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
| gemma-2-2b-it | 93.7 | 94.0 | 94.3 | 94.5 | 94.1 | **94.1** |
| gemma-2-9b-it | 92.1 | 90.0 | 90.4 | 94.2 | 90.6 | **91.5** |
| gemma-2b-it | 89.7 | 90.5 | 94.5 | 90.5 | 89.9 | **91.0** |
| mistral-7b | 90.2 | 89.1 | 90.6 | 89.2 | 90.3 | **89.9** |
| phi-2 | 94.5 | 93.0 | 94.1 | 94.1 | 91.4 | **93.4** |
| stablelm-2-1.6b | 93.9 | 90.4 | 91.0 | 89.8 | 90.5 | **91.1** |
| tinyllama-1.1b | 89.8 | 90.8 | 91.1 | 89.7 | 90.5 | **90.4** |

## Average Set Size

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 5.60 | 5.57 | 5.59 | 5.61 | 5.49 | **5.57** |
| gemma-2-9b-it | 3.10 | 4.82 | 4.95 | 5.40 | 4.27 | **4.51** |
| gemma-2b-it | 4.73 | 5.22 | 5.59 | 5.21 | 4.60 | **5.07** |
| mistral-7b | 4.96 | 4.88 | 4.91 | 4.90 | 3.97 | **4.73** |
| phi-2 | 5.60 | 5.39 | 5.53 | 5.57 | 5.18 | **5.46** |
| stablelm-2-1.6b | 5.55 | 5.16 | 5.22 | 5.06 | 4.94 | **5.19** |
| tinyllama-1.1b | 5.06 | 5.09 | 5.21 | 5.04 | 4.51 | **4.98** |