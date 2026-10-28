# Benchmark overview: RL fine-tuning for Whisper

## Goal

Compare RL engineering frameworks for Whisper fine-tuning under a shared
**audio dataset, reward function, and evaluation metrics**.

## Why multiple frameworks?

- ASR has **verifiable rewards** (WER/CER), which fit GRPO / PPO-style RLVR.
- Industrial post-training stacks (TRL / OpenRLHF / verl) emphasize different strengths.
- Whisper is seq2seq; most LLM-RL frameworks are not Whisper-ready out of the box,
  so adapters and test workflows must be explicit.

## Recommended experiment flow

```text
Prepare data (prepare_data / generate_demo_data)
        ↓
Set up framework env (frameworks/*/setup_env.sh via uv)
        ↓
Smoke test (run.sh smoke)
        ↓
Full train (run.sh grpo|ppo|orpo)
        ↓
evaluate.py + compare_results.py
```

## Smoke success criteria

- Each framework `run.sh smoke` exits 0
- Produces `summary.json`
- `eval_wer` / `eval_cer` are finite numbers (values on demo tones are not meaningful)
