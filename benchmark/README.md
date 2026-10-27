# Whisper 强化学习微调基准（RL Fine-tuning Benchmark）

面向 **音频 ASR / Whisper** 的强化学习微调基准套件。为业界常见 RL 框架各自提供：

1. **环境搭建**（**uv** + `setup_env.sh` / `pyproject.toml`）
2. **测试脚本**（训练 / 冒烟 / 评测）
3. **测试流程**（数据准备 → 训练 → 指标对比）

Python 依赖统一用 uv 管理，参见 [`docs/uv_env.md`](docs/uv_env.md)。

> 说明：Whisper 是 **encoder-decoder**，而 OpenRLHF / verl / 部分 TRL trainer 主要面向 decoder-only LLM。本基准在统一奖励与数据协议上，为每个框架提供 **可运行的 Whisper 适配训练环**；大规模生产部署时可再挂接各框架原生的 Ray / vLLM / FSDP 后端。

## 支持的框架

| 框架 | 算法 | 场景 | 入口 |
|---|---|---|---|
| **TRL**（HuggingFace） | GRPO / ORPO | 原型最快、HF 生态 | `benchmark/frameworks/trl/` |
| **OpenRLHF** | PPO（Actor/Ref/Reward） | 分布式 RLHF 风格 | `benchmark/frameworks/openrlhf/` |
| **verl**（HybridFlow） | GRPO（单控制器相位） | 工业级 colocated RL | `benchmark/frameworks/verl/` |
| **Native ORPO** | ORPO | 本仓库原有实现基线 | `benchmark/frameworks/native_orpo/` |

## 目录结构

```text
benchmark/
├── README.md                 # 本文件
├── common/                   # 共享：奖励、指标、数据、配置
├── configs/                  # 统一 YAML 配置
├── data/demo/                # 自动生成的流水线验证数据
├── docs/                     # 总览与框架对比
├── frameworks/
│   ├── trl/
│   ├── openrlhf/
│   ├── verl/
│   └── native_orpo/
└── scripts/                  # 数据准备、评测、一键对比
```

## 快速开始（推荐路径）

### 0. 用 uv 搭建环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
bash scripts/setup_uv.sh all
# 按框架: bash scripts/setup_uv.sh trl|openrlhf|verl
```

### 1. 生成 demo 数据

```bash
uv run python benchmark/scripts/generate_demo_data.py
```

> demo 音频是短正弦波，仅用于验证 **数据→训练→评测** 流水线。真实实验请用 `prepare_data.py` 接入 LibriSpeech / Common Voice。

### 2. 按框架冒烟测试

```bash
bash benchmark/frameworks/trl/run.sh smoke
bash benchmark/frameworks/openrlhf/run.sh smoke
bash benchmark/frameworks/verl/run.sh smoke
```

或一键：

```bash
bash benchmark/scripts/run_benchmark.sh smoke trl,openrlhf,verl
```

### 3. 对比结果

```bash
uv run python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

每个框架输出：

- `outputs/benchmark/<framework>/train_metrics.jsonl`
- `outputs/benchmark/<framework>/summary.json`
- `outputs/benchmark/<framework>/eval_predictions.jsonl`
- `outputs/benchmark/<framework>/checkpoint-final/`

## 统一数据协议

### 在线 RL（GRPO / PPO）

```json
{"audio":{"path":"path/to.wav"},"duration":3.2,"reference":"ground truth text","language":"en","id":"utt-0"}
```

### 离线偏好（ORPO / DPO）

```json
{"audio":{"path":"path/to.wav"},"duration":3.2,"chosen":"good hyp","rejected":"bad hyp","reference":"ground truth text","language":"en","id":"utt-0"}
```

## 统一奖励

在线 RL 默认使用可验证规则奖励（无需奖励模型）：

\[
r = 1 - (w_{wer}\cdot WER + w_{cer}\cdot CER) - \lambda_{len}\cdot L_{len} - \lambda_{rep}\cdot L_{rep}
\]

实现见 `benchmark/common/rewards.py`，可在 YAML `reward:` 段调权重。

## 真实数据准备

```bash
# 已有 JSONL
python benchmark/scripts/prepare_data.py \
  --input /path/to/asr.jsonl \
  --out_dir benchmark/data/my_asr

# HuggingFace LibriSpeech 示例
python benchmark/scripts/prepare_data.py \
  --hf_dataset librispeech_asr --hf_config clean \
  --hf_split train.100 --max_samples 1000 \
  --out_dir benchmark/data/librispeech
```

然后改对应框架 YAML 中的 `data.train_path` / `data.eval_path`。

## 各框架测试流程文档

- [TRL](frameworks/trl/README.md)
- [OpenRLHF](frameworks/openrlhf/README.md)
- [verl](frameworks/verl/README.md)
- [Native ORPO](frameworks/native_orpo/README.md)
- [框架对比](docs/framework_comparison.md)
- [数据格式](docs/data_format.md)
- [uv 环境管理](docs/uv_env.md)

## 评测

```bash
uv run python benchmark/scripts/evaluate.py \
  --model_path outputs/benchmark/trl_grpo/checkpoint-final \
  --base_model openai/whisper-tiny \
  --eval_path benchmark/data/demo/eval_rl.jsonl
```

主指标：**WER**、**CER**；辅看训练曲线中的 `reward_mean` / `kl` / `loss`。
