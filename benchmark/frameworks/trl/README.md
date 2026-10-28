# TRL (HuggingFace Transformer Reinforcement Learning)

Best fit for Whisper RL **prototyping and baseline comparison**: integrates with
`transformers` / `peft` / `accelerate`.

This directory ships two adapters:

| Script | Algorithm | Data | Notes |
|---|---|---|---|
| `train_grpo.py` | GRPO | `*_rl.jsonl` | Online sampling + WER/CER reward |
| `train_orpo.py` | ORPO | `*_pref.jsonl` | Offline chosen/rejected |

> Upstream TRL `GRPOTrainer` targets decoder-only models. Here we implement the
> **same recipe** for Whisper encoder-decoder so frameworks can be compared fairly.

---

## 1. Environment setup (uv)

```bash
# Recommended: sync the trl extra from the repo root
bash benchmark/frameworks/trl/setup_env.sh
# equivalent: uv sync --extra trl

source .venv/bin/activate   # optional; or use uv run
```

Check:

```bash
uv run python -c "import torch,transformers,peft,trl; print('ok', torch.__version__)"
```

---

## 2. Prepare data and configs

```bash
# Pipeline validation data (synthetic audio)
python benchmark/scripts/generate_demo_data.py

# Or real data
python benchmark/scripts/prepare_data.py \
  --hf_dataset librispeech_asr --hf_config clean \
  --hf_split validation --max_samples 200 \
  --out_dir benchmark/data/librispeech
```

Configs: `benchmark/configs/frameworks/trl_grpo.yaml` / `trl_orpo.yaml`.

---

## 3. How to run tests

### 3.1 Smoke (CI / env check)

```bash
bash benchmark/frameworks/trl/run.sh smoke
```

Expect:

- 2 GRPO steps + 2 ORPO steps
- `outputs/benchmark/trl_*/summary.json`

### 3.2 Small GRPO run

```bash
bash benchmark/frameworks/trl/run.sh grpo
# or
python benchmark/frameworks/trl/train_grpo.py \
  --config benchmark/configs/frameworks/trl_grpo.yaml \
  --max_steps 50
```

Flow:

1. Load Whisper (optional LoRA)
2. Sample `num_generations` hypotheses per clip
3. Score with blended WER/CER reward; group-normalize advantages
4. Maximize `A * log_prob` with KL penalty to the reference
5. Eval WER/CER and write summary

### 3.3 ORPO preference optimization

```bash
bash benchmark/frameworks/trl/run.sh orpo
```

### 3.4 Evaluate and compare

```bash
python benchmark/scripts/evaluate.py \
  --model_path outputs/benchmark/trl_grpo/checkpoint-final \
  --base_model openai/whisper-tiny \
  --eval_path benchmark/data/demo/eval_rl.jsonl

python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## Key hyperparameters (GRPO)

| Param | Default | Meaning |
|---|---|---|
| `grpo.num_generations` | 4 | Group size G per clip |
| `grpo.temperature` | 0.7 | Sampling temperature |
| `grpo.beta` | 0.04 | KL coefficient |
| `training.learning_rate` | 3e-6 | Policy learning rate |
| `reward.wer_weight` | 0.7 | WER weight |

---

## FAQ

- **OOM**: lower `num_generations` / batch size, or use `whisper-tiny` + LoRA.
- **No CUDA**: CPU smoke works; use a GPU for real training.
- **Want upstream TRL Trainer**: plug the same reward into a decoder-only ASR-LLM;
  for pure Whisper use these adapters.
