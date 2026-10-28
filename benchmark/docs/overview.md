# Benchmark overview: comparing against mainstream RL stacks

## Goal

The repository core is an **in-house Whisper ORPO trainer** (`finetune.py`).
This benchmark suite compares that baseline with mainstream RL frameworks under a
shared **audio dataset, reward function, and evaluation metrics**.

## Why a separate benchmark tree?

- Day-to-day training should stay focused on the in-house ORPO path.
- Side-by-side evaluation against TRL / OpenRLHF / verl needs its own adapters,
  configs, and run scripts.
- ASR has **verifiable rewards** (WER/CER), which fit GRPO / PPO-style RLVR and
  make fair cross-framework comparison practical.
- Whisper is seq2seq; most LLM-RL frameworks are not Whisper-ready out of the box,
  so comparison adapters must be explicit.

## What to run when

| Goal | Where to go |
|---|---|
| Train / ship the in-house ORPO model | Root [`README.md`](../../README.md), `finetune.py` |
| Compare vs TRL / OpenRLHF / verl | This `benchmark/` suite |
| Use in-house trainer as a benchmark baseline | `frameworks/native_orpo/` |

## Recommended comparison flow

```text
Prepare data (prepare_data / generate_demo_data)
        ↓
Set up framework env (frameworks/*/setup_env.sh via uv)
        ↓
Smoke test (run.sh smoke)
        ↓
Full train (run.sh grpo|ppo|orpo)  # include native_orpo for the in-house baseline
        ↓
evaluate.py + compare_results.py
```

## Smoke success criteria

- Each framework `run.sh smoke` exits 0
- Produces `summary.json`
- `eval_wer` / `eval_cer` are finite numbers (values on demo tones are not meaningful)
