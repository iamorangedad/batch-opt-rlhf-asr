# Framework comparison (Whisper RL fine-tuning benchmark)

## Overview

| Dimension | TRL | OpenRLHF adapter | verl adapter | Native ORPO |
|---|---|---|---|---|
| Primary algorithm | GRPO / ORPO | PPO | GRPO | ORPO |
| Learning mode | online / offline | online | online | offline |
| Getting started | easy | medium | medium | easy (in-repo) |
| Distributed (this adapter) | single node | single node (Ray-ready) | single node (phased) | DDP (legacy script) |
| Production scale-out | Accelerate | Ray + DeepSpeed + vLLM | HybridFlow + FSDP | custom |
| Typical use | algorithm prototype, HF stack | role-separated RLHF | industrial colocated RL | preference baseline |

## Algorithm tips

- **Have reference text / can compute WER/CER**: prefer **GRPO** (TRL / verl); no Critic / RM needed.
- **Have human or model preference pairs**: use **ORPO** (TRL ORPO or Native ORPO).
- **Need OpenRLHF-style role layout**: use the **PPO adapter**, then migrate to upstream OpenRLHF.

## Fair comparison checklist

1. Same `openai/whisper-tiny` (or same base)
2. Same train/eval JSONL
3. Same `reward:` weights
4. Similar `num_generations`, temperature, `max_steps`
5. Same `benchmark/scripts/evaluate.py`

## Aligned outputs

Each framework writes:

- `summary.json`: `framework`, `algorithm`, `eval_wer`, `eval_cer`, `steps`
- `train_metrics.jsonl`: step-wise loss / reward
- `checkpoint-final/`: loadable weights or LoRA

Aggregate with `compare_results.py`.
