# Native ORPO（本仓库自带）

基于仓库根目录 `finetune.py` + `src/orpo_trainer.py` 的 Whisper ORPO 实现，作为 **离线偏好优化基线**。

## 1. 搭建环境

```bash
bash benchmark/frameworks/native_orpo/run.sh setup
```

## 2. 准备数据

偏好数据为 JSONL，每行：

```json
{"audio":{"path":"..."},"duration":1.2,"chosen":"correct transcript","rejected":"wrong transcript","language":"en"}
```

生成 demo：

```bash
python benchmark/scripts/generate_demo_data.py
```

## 3. 运行测试

```bash
# 冒烟
bash benchmark/frameworks/native_orpo/run.sh smoke

# 正式训练
bash benchmark/frameworks/native_orpo/run.sh train
```

## 4. 与其它框架对比

| 项 | Native ORPO | TRL ORPO |
|---|---|---|
| 算法 | ORPO | ORPO |
| 数据 | chosen/rejected | chosen/rejected |
| 入口 | `finetune.py` | `benchmark/frameworks/trl/train_orpo.py` |
| 在线采样 | 否 | 否 |

在线 RL（需要生成再打分）请用 TRL-GRPO / OpenRLHF-PPO / verl-GRPO。
