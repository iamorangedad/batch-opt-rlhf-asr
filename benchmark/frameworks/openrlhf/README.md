# OpenRLHF 风格流水线（Whisper ASR）

OpenRLHF 是面向生产的 RLHF 框架（Ray 调度 + DeepSpeed/Megatron + vLLM rollout）。  
本基准提供 **角色对齐的 Whisper PPO 适配器**：Actor / Reference / Reward 三分法与 OpenRLHF 一致，便于对照测试流程；完整 Ray+vLLM 集群可作为后续扩展。

---

## 1. 搭建测试环境（uv）

```bash
bash benchmark/frameworks/openrlhf/setup_env.sh
# 等价: uv sync --extra openrlhf

source .venv/bin/activate   # 可选；也可用 uv run

# 可选：完整 OpenRLHF（decoder-only / 多机）
# uv pip install openrlhf vllm
```

---

## 2. 准备测试脚本与数据

```bash
python benchmark/scripts/generate_demo_data.py
```

配置：`benchmark/configs/frameworks/openrlhf.yaml`

脚本：

- `train_ppo.py` — colocated PPO
- `run.sh` — setup / smoke / ppo

---

## 3. 如何进行测试

### 3.1 冒烟

```bash
bash benchmark/frameworks/openrlhf/run.sh smoke
```

### 3.2 PPO 训练

```bash
bash benchmark/frameworks/openrlhf/run.sh ppo
# 或
python benchmark/frameworks/openrlhf/train_ppo.py \
  --config benchmark/configs/frameworks/openrlhf.yaml \
  --max_steps 30
```

训练环（对齐 OpenRLHF 概念）：

```text
Rollout Worker  →  Actor.generate(G 条假设)
Reward          →  WER/CER 规则奖励（可替换为 RM）
Reference       →  冻结策略算 KL
PPO Update      →  clipped surrogate + KL coef
```

### 3.3 结果

输出目录：`outputs/benchmark/openrlhf/`

```bash
python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## 与官方 OpenRLHF 的关系

| 组件 | 本适配器 | 官方 OpenRLHF |
|---|---|---|
| 调度 | 单进程 | Ray Actor Pool |
| Rollout | HF `generate` | vLLM / 异步生成 |
| 训练引擎 | PyTorch AdamW | DeepSpeed ZeRO / Megatron |
| 模型 | Whisper seq2seq | 主要为 Causal LM |
| 奖励 | 规则 WER/CER | RM 或规则 |

当你的 ASR 模型是 **decoder-only audio-LLM** 时，可直接切到官方 OpenRLHF，复用同一 `reference` JSONL 与奖励函数接口。

---

## 关键配置

| 参数 | 含义 |
|---|---|
| `openrlhf.num_generations` | rollout 组大小 |
| `openrlhf.clip_eps` | PPO clip |
| `openrlhf.kl_coef` | KL 惩罚 |
| `openrlhf.ppo_epochs` | 每批 PPO 更新轮数 |
| `openrlhf.use_ray` | 预留开关（默认 false） |
