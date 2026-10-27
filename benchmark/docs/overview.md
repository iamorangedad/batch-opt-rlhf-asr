# 基准总览：用 RL 微调 Whisper

## 目标

在统一的 **音频数据、奖励函数、评测指标** 下，对比不同强化学习工程框架微调 Whisper 的流程与效果。

## 为什么需要多框架基准？

- ASR 有 **可验证奖励**（WER/CER），天然适合 GRPO / PPO 等 RLVR 方法。  
- 工业界主流 post-training 栈（TRL / OpenRLHF / verl）能力侧重点不同。  
- Whisper 是 seq2seq，多数 LLM-RL 框架不能开箱即用，需要明确 **适配层与测试流程**。

## 推荐实验流程

```text
准备数据 (prepare_data / generate_demo_data)
        ↓
选择框架环境 (frameworks/*/setup_env.sh)
        ↓
冒烟 smoke (run.sh smoke)
        ↓
正式训练 (run.sh grpo|ppo|orpo)
        ↓
evaluate.py + compare_results.py
```

## 成功标准（冒烟）

- 各框架 `run.sh smoke` 退出码 0  
- 产出 `summary.json`  
- `eval_wer` / `eval_cer` 为有限数值（demo 数据上数值本身无业务意义）
