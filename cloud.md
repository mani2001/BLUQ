# Cloud Benchmarking Notes

## Completed Benchmarks

All benchmarks run on NVIDIA A100 80GB PCIe with 10,000 samples.

| Model | Parameters | Tasks Completed | Notes |
|-------|------------|-----------------|-------|
| TinyLlama-1.1B | 1.1B | 5/5 (float16), 4/5 (float32) | float32 RC failed - network timeout |
| Phi-2 | 2.7B | 5/5 | Two complete runs |
| StableLM-2-1.6B | 1.6B | 5/5 | |
| Gemma-2B-IT | 2B | 3/5 | Only RC, CI, DRS |
| Gemma-2-2B-IT | 2B | 5/5 | |
| Gemma-2-9B-IT | 9B | 4/5 | QA failed during model unload |

## Pending Work

### Not Yet Run
- Qwen-1.8B: Tokenizer compatibility issue ("Adding unknown special tokens is not supported")

### Partial Runs to Complete
- TinyLlama-1.1B float32: RC task (network timeout)
- Gemma-2B-IT: QA and DS tasks
- Gemma-2-9B-IT: QA task (errored during save, probabilities exist)

## RunPod Instance Details

- GPU: NVIDIA A100 80GB PCIe
- SSH: `ssh root@38.128.232.57 -p 14166 -i ~/.ssh/id_rsa`
- Working directory: `~/BLUQ`
- Virtual environment: `.venv`
