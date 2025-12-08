# BLUQ Benchmark Results (float16) - APS

## Accuracy (%)

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 22.3 | 25.7 | 25.0 | 23.8 | 26.8 | **24.7** |
| gemma-2-9b-it | 70.8 | 30.7 | 30.3 | 28.8 | 38.5 | **39.8** |
| gemma-2b-it | 24.3 | 22.9 | 22.1 | 22.8 | 24.6 | **23.3** |
| phi-2 | 22.5 | 24.6 | 22.3 | 22.6 | 35.3 | **25.5** |
| stablelm-2-1.6b | 20.5 | 21.9 | 22.2 | 21.3 | 23.6 | **21.9** |
| tinyllama-1.1b | 21.9 | 20.9 | 20.3 | 20.0 | 22.7 | **21.2** |

## Coverage Rate (%)

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 97.6 | 98.2 | 98.4 | 98.5 | 98.2 | **98.2** |
| gemma-2-9b-it | 93.6 | 90.2 | 90.4 | 98.2 | 90.0 | **92.5** |
| gemma-2b-it | 89.9 | 90.7 | 98.5 | 90.8 | 89.7 | **91.9** |
| phi-2 | 98.8 | 95.4 | 98.7 | 98.7 | 93.3 | **97.0** |
| stablelm-2-1.6b | 98.2 | 90.2 | 92.0 | 90.0 | 90.6 | **92.2** |
| tinyllama-1.1b | 90.2 | 90.9 | 92.0 | 89.8 | 90.9 | **90.8** |

## Average Set Size

| Model | Question Answering | Reading Comprehension | Commonsense Inference | Dialogue Response | Document Summarization | Avg |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | 5.84 | 5.87 | 5.87 | 5.86 | 5.86 | **5.86** |
| gemma-2-9b-it | 3.60 | 4.86 | 4.99 | 5.85 | 4.40 | **4.74** |
| gemma-2b-it | 4.74 | 5.17 | 5.86 | 5.16 | 4.55 | **5.09** |
| phi-2 | 5.91 | 5.58 | 5.90 | 5.89 | 5.34 | **5.72** |
| stablelm-2-1.6b | 5.85 | 5.04 | 5.19 | 4.89 | 4.85 | **5.16** |
| tinyllama-1.1b | 4.95 | 5.00 | 5.17 | 4.92 | 4.36 | **4.88** |