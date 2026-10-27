# verl（HybridFlow）风格流水线（Whisper ASR）

verl（Volcano Engine RL）以 **单控制器 + colocated rollout/train 换引擎** 著称。  
本基准实现一个 **相位清晰** 的 Whisper GRPO 控制器，对应 HybridFlow 的：

```text
rollout_phase → reward_phase → update_phase
```

便于在同一套 ASR 数据/奖励上对比 verl 式调度与其它框架。

---

## 1. 搭建测试环境

```bash
bash benchmark/frameworks/verl/setup_env.sh
source .venvs/verl/bin/activate

# 可选完整 verl（decoder-only + FSDP/Megatron + vLLM）
# pip install verl
```

或：

```bash
pip install -r benchmark/frameworks/verl/requirements.txt
```

---

## 2. 准备测试脚本与数据

```bash
python benchmark/scripts/generate_demo_data.py
```

配置：`benchmark/configs/frameworks/verl.yaml`  
脚本：`train_grpo.py`、`run.sh`

---

## 3. 如何进行测试

### 3.1 冒烟

```bash
bash benchmark/frameworks/verl/run.sh smoke
```

### 3.2 GRPO 训练

```bash
bash benchmark/frameworks/verl/run.sh grpo
```

控制器逻辑（见 `SingleController`）：

1. **rollout_phase**：策略采样 G 条转写  
2. **reward_phase**：WER/CER 奖励 + group-relative advantage  
3. **update_phase**：策略梯度 + KL 到 reference  

指标写入 `outputs/benchmark/verl/train_metrics.jsonl`，字段含 `"phase": "rollout->reward->update"`。

### 3.3 对比

```bash
python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## 与官方 verl 的关系

| 能力 | 本适配器 | 官方 verl |
|---|---|---|
| 控制模型 | 单控制器相位 API | HybridFlow 单控制器 |
| 权重换布局 | 无（同引擎） | 3D-HybridEngine reshard |
| Rollout 引擎 | HF generate | vLLM / SGLang |
| 并行 | 单卡/简单 DDP 可扩 | FSDP2 / Megatron 3D |
| Whisper | ✅ | 需自定义 worker |

本适配器用于 **基准流程与算法对照**；上大规模 decoder-only ASR-LLM 时，把同一奖励函数接到 verl reward manager 即可。
