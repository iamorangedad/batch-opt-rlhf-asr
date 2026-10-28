# CPU memory comparison report

- Generated (UTC): `2026-10-28T20:54:44.922112+00:00`
- Device: **CPU** (`CUDA_VISIBLE_DEVICES=` forced empty)
- Model: `openai/whisper-tiny` + LoRA
- Data: `benchmark/data/librispeech_tiny`
- Isolation: **one fresh Python process per framework**
- Metric: process **peak RSS** (`cpu_peak_rss_mb`) and delta vs baseline

## Results

| rank | name | framework | algorithm | peak RSS (MiB) | ΔRSS (MiB) | baseline (MiB) | elapsed (s) | status |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | trl_orpo | trl | orpo | 1264.8 | 765.0 | 499.8 | 11.67 | ok |
| 2 | verl_grpo | verl | grpo | 1436.4 | 936.6 | 499.8 | 14.92 | ok |
| 3 | trl_grpo | trl | grpo | 1436.8 | 936.9 | 499.9 | 15.72 | ok |
| 4 | openrlhf_ppo | openrlhf | ppo | 1464.9 | 965.1 | 499.9 | 15.82 | ok |
| 5 | native_orpo | native_orpo | orpo | 1480.5 | 596.0 | 884.4 | 20.66 | ok |

## Notes

- Peak RSS includes Python + PyTorch + model weights + activations for that process.
- Online RL (GRPO/PPO) usually keeps a frozen reference copy, so RSS tends to be higher than offline ORPO.
- Native ORPO uses HuggingFace Trainer; one epoch over the tiny set (~12 steps) may differ from the fixed 5-step adapters.
- This measures **process memory**, not host-wide pressure from other agents.

JSON: `/workspace/outputs/benchmark/cpu_memory_compare/cpu_memory_report.json`
