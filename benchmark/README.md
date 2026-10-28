# Whisper RL Fine-tuning Benchmark

Optional comparison suite for this repository.

The **core product** of the repo is the in-house Whisper ORPO trainer
(`finetune.py` + `src/`). This `benchmark/` tree exists to **compare that
baseline against mainstream RL frameworks** (TRL / OpenRLHF / verl) under a
shared data protocol, reward, and eval metrics.

It is **not required** for day-to-day ORPO training. For the main project, see
the [root README](../README.md).

For each framework it provides:

1. **Environment setup** (**uv** + `setup_env.sh` / `pyproject.toml`)
2. **Test scripts** (train / smoke / evaluate)
3. **Test workflow** (data prep → train → metric comparison)

Python dependencies are managed with uv; see [`docs/uv_env.md`](docs/uv_env.md).

> Note: Whisper is **encoder-decoder**, while OpenRLHF / verl / some TRL trainers
> target decoder-only LLMs. This benchmark ships a runnable Whisper adapter loop
> per framework under a shared reward and data protocol; production scale-out can
> later plug in native Ray / vLLM / FSDP backends.

## Relation to the core trainer

| Piece | Role |
|---|---|
| `finetune.py` + `src/orpo_trainer.py` | Primary in-house ORPO training |
| `benchmark/frameworks/native_orpo/` | Thin wrapper that runs the in-house trainer as a baseline |
| `benchmark/frameworks/trl/` | Mainstream TRL-style GRPO / ORPO comparison |
| `benchmark/frameworks/openrlhf/` | Mainstream OpenRLHF-style PPO comparison |
| `benchmark/frameworks/verl/` | Mainstream verl-style GRPO comparison |

Use the core trainer for production preference fine-tuning; use this suite when
you need side-by-side numbers against common RL stacks.

## Supported frameworks

| Framework | Algorithm | Use case | Entry |
|---|---|---|---|
| **TRL** (HuggingFace) | GRPO / ORPO | Fastest prototype, HF ecosystem | `benchmark/frameworks/trl/` |
| **OpenRLHF** | PPO (Actor/Ref/Reward) | Distributed RLHF-style | `benchmark/frameworks/openrlhf/` |
| **verl** (HybridFlow) | GRPO (single-controller) | Industrial colocated RL | `benchmark/frameworks/verl/` |
| **Native ORPO** | ORPO | In-repo baseline (`finetune.py`) | `benchmark/frameworks/native_orpo/` |

## Layout

```text
benchmark/
├── README.md                 # this file
├── common/                   # shared reward, metrics, data, config
├── configs/                  # shared YAML configs
├── data/demo/                # auto-generated pipeline validation data
├── docs/                     # overview and framework comparison
├── frameworks/
│   ├── trl/
│   ├── openrlhf/
│   ├── verl/
│   └── native_orpo/
└── scripts/                  # data prep, eval, one-shot compare
```

## Quick start

### 0. Set up with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
bash scripts/setup_uv.sh all
# per framework: bash scripts/setup_uv.sh trl|openrlhf|verl
```

### 1. Generate demo data

```bash
uv run python benchmark/scripts/generate_demo_data.py
```

> Demo audio is short sine tones — only for validating the **data → train → eval**
> pipeline. For real experiments use `prepare_data.py` with LibriSpeech / Common Voice.

### 2. Per-framework smoke tests

```bash
bash benchmark/frameworks/trl/run.sh smoke
bash benchmark/frameworks/openrlhf/run.sh smoke
bash benchmark/frameworks/verl/run.sh smoke
```

Or one-shot:

```bash
bash benchmark/scripts/run_benchmark.sh smoke trl,openrlhf,verl
```

### 3. Compare results

```bash
uv run python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

Each framework writes:

- `outputs/benchmark/<framework>/train_metrics.jsonl`
- `outputs/benchmark/<framework>/summary.json`
- `outputs/benchmark/<framework>/eval_predictions.jsonl`
- `outputs/benchmark/<framework>/checkpoint-final/`

## Unified data protocol

### Online RL (GRPO / PPO)

```json
{"audio":{"path":"path/to.wav"},"duration":3.2,"reference":"ground truth text","language":"en","id":"utt-0"}
```

### Offline preference (ORPO / DPO)

```json
{"audio":{"path":"path/to.wav"},"duration":3.2,"chosen":"good hyp","rejected":"bad hyp","reference":"ground truth text","language":"en","id":"utt-0"}
```

## Unified reward

Online RL uses a verifiable rule-based reward (no reward model):

\[
r = 1 - (w_{wer}\cdot WER + w_{cer}\cdot CER) - \lambda_{len}\cdot L_{len} - \lambda_{rep}\cdot L_{rep}
\]

See `benchmark/common/rewards.py`; weights are tunable under YAML `reward:`.

## Real data preparation

```bash
# Existing JSONL
python benchmark/scripts/prepare_data.py \
  --input /path/to/asr.jsonl \
  --out_dir benchmark/data/my_asr

# HuggingFace LibriSpeech example
python benchmark/scripts/prepare_data.py \
  --hf_dataset librispeech_asr --hf_config clean \
  --hf_split train.100 --max_samples 1000 \
  --out_dir benchmark/data/librispeech
```

Then update `data.train_path` / `data.eval_path` in the framework YAML.

## Framework workflow docs

- [TRL](frameworks/trl/README.md)
- [OpenRLHF](frameworks/openrlhf/README.md)
- [verl](frameworks/verl/README.md)
- [Native ORPO](frameworks/native_orpo/README.md)
- [Framework comparison](docs/framework_comparison.md)
- [Latest comparison report](docs/comparison_report.md)
- [Data format](docs/data_format.md)
- [uv environment](docs/uv_env.md)
- [Overview](docs/overview.md)

## Evaluation

```bash
uv run python benchmark/scripts/evaluate.py \
  --model_path outputs/benchmark/trl_grpo/checkpoint-final \
  --base_model openai/whisper-tiny \
  --eval_path benchmark/data/demo/eval_rl.jsonl
```

Primary metrics: **WER**, **CER**; also inspect `reward_mean` / `kl` / `loss` in training logs.
