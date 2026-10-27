# TRL（HuggingFace Transformer Reinforcement Learning）

最适合做 Whisper RL **原型与基准对照** 的框架：与 `transformers` / `peft` / `accelerate` 一体。

本目录提供两套适配：

| 脚本 | 算法 | 数据 | 说明 |
|---|---|---|---|
| `train_grpo.py` | GRPO | `*_rl.jsonl` | 在线采样 + WER/CER 奖励 |
| `train_orpo.py` | ORPO | `*_pref.jsonl` | 离线 chosen/rejected |

> TRL 官方 `GRPOTrainer` 面向 decoder-only。此处实现 **同配方** 的 Whisper encoder-decoder GRPO 环，便于与其它框架公平对比。

---

## 1. 搭建测试环境

```bash
# 方式 A：专用 venv
bash benchmark/frameworks/trl/setup_env.sh
source .venvs/trl/bin/activate

# 方式 B：当前环境直接装
pip install -r benchmark/frameworks/trl/requirements.txt
```

检查：

```bash
python -c "import torch,transformers,peft,trl; print('ok', torch.__version__)"
```

---

## 2. 准备测试脚本与数据

```bash
# 流水线验证数据（合成音频）
python benchmark/scripts/generate_demo_data.py

# 或真实数据
python benchmark/scripts/prepare_data.py \
  --hf_dataset librispeech_asr --hf_config clean \
  --hf_split validation --max_samples 200 \
  --out_dir benchmark/data/librispeech
```

配置：`benchmark/configs/frameworks/trl_grpo.yaml` / `trl_orpo.yaml`。

---

## 3. 如何进行测试

### 3.1 冒烟（CI / 环境验证）

```bash
bash benchmark/frameworks/trl/run.sh smoke
```

期望：

- 跑通 2 步 GRPO + 2 步 ORPO
- 写出 `outputs/benchmark/trl_*/summary.json`

### 3.2 GRPO 正式小跑

```bash
bash benchmark/frameworks/trl/run.sh grpo
# 或
python benchmark/frameworks/trl/train_grpo.py \
  --config benchmark/configs/frameworks/trl_grpo.yaml \
  --max_steps 50
```

流程：

1. 加载 Whisper（可选 LoRA）
2. 对每条音频采样 `num_generations` 条假设
3. 用 WER/CER 混合奖励打分，组内标准化 advantage
4. 最大化 `A · logπ`，并加对 reference 的 KL 惩罚
5. 在 eval 集上算 WER/CER，落盘 summary

### 3.3 ORPO 偏好优化

```bash
bash benchmark/frameworks/trl/run.sh orpo
```

### 3.4 评测与对比

```bash
python benchmark/scripts/evaluate.py \
  --model_path outputs/benchmark/trl_grpo/checkpoint-final \
  --base_model openai/whisper-tiny \
  --eval_path benchmark/data/demo/eval_rl.jsonl

python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## 关键超参（GRPO）

| 参数 | 默认 | 含义 |
|---|---|---|
| `grpo.num_generations` | 4 | 每条音频采样组大小 G |
| `grpo.temperature` | 0.7 | 采样温度 |
| `grpo.beta` | 0.04 | KL 系数 |
| `training.learning_rate` | 3e-6 | 策略学习率 |
| `reward.wer_weight` | 0.7 | WER 权重 |

---

## 常见问题

- **OOM**：减小 `num_generations`、batch size，或换 `whisper-tiny` + LoRA。
- **无 CUDA**：可 CPU 冒烟，但正式训练请用 GPU。
- **想用官方 TRL Trainer**：可把同一奖励函数接到 decoder-only ASR-LLM；纯 Whisper 请用本适配脚本。
