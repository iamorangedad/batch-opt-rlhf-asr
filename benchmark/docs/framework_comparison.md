# 框架对比（Whisper RL 微调基准）

## 总览

| 维度 | TRL | OpenRLHF 适配 | verl 适配 | Native ORPO |
|---|---|---|---|---|
| 主算法 | GRPO / ORPO | PPO | GRPO | ORPO |
| 学习范式 | 在线 / 离线 | 在线 | 在线 | 离线 |
| 上手难度 | 低 | 中 | 中 | 低（本仓） |
| 分布式能力（本适配） | 单机 | 单机（可扩 Ray） | 单机（相位化） | DDP（原脚本） |
| 生产扩展 | Accelerate | Ray+DeepSpeed+vLLM | HybridFlow+FSDP | 自研 |
| 典型用途 | 算法原型、HF 生态 | 角色分离 RLHF | 工业 colocated RL | 偏好基线 |

## 算法选择建议

- **有 reference 文本、可算 WER/CER**：优先 **GRPO**（TRL / verl），无需 Critic / RM。  
- **有人工/模型偏好对**：用 **ORPO**（TRL ORPO 或 Native ORPO）。  
- **需要对齐 OpenRLHF 工程角色**：用本仓 **PPO 适配**，再迁官方 OpenRLHF。  

## 公平对比约定

为了让基准可复现，建议锁定：

1. 同一 `openai/whisper-tiny`（或同一 base）  
2. 同一 train/eval JSONL  
3. 同一 `reward:` 权重  
4. 相近的 `num_generations`、温度、`max_steps`  
5. 同一评测脚本 `benchmark/scripts/evaluate.py`

## 输出对齐

每个框架都写：

- `summary.json`：`framework`, `algorithm`, `eval_wer`, `eval_cer`, `steps`  
- `train_metrics.jsonl`：逐步 loss / reward  
- `checkpoint-final/`：可加载权重或 LoRA  

用 `compare_results.py` 汇总。
