# 数据格式说明

## 在线 RL：`*_rl.jsonl`

一行一个样本：

```json
{
  "audio": {"path": "benchmark/data/demo/audio/train_000.wav"},
  "duration": 1.5,
  "reference": "the quick brown fox jumps over the lazy dog",
  "language": "en",
  "id": "train-0"
}
```

字段：

| 字段 | 必需 | 说明 |
|---|---|---|
| `audio.path` / `audio_path` | 是 | 本地 wav/flac |
| `reference` | 是 | 标准转写（奖励与评测） |
| `language` | 否 | 默认配置语言 |
| `duration` | 否 | 秒；过滤长短音频用 |
| `id` | 否 | 追踪用 |

## 离线偏好：`*_pref.jsonl`

在 RL 字段基础上增加：

```json
{
  "audio": {"path": "..."},
  "duration": 1.5,
  "reference": "ground truth",
  "chosen": "ground truth or better hyp",
  "rejected": "worse hyp",
  "language": "en",
  "id": "train-0"
}
```

兼容本仓库原有 ORPO `finetune.py` 读取格式。

## 生成方式

```bash
# 流水线 demo
python benchmark/scripts/generate_demo_data.py

# 从已有 JSONL / HF 数据集
python benchmark/scripts/prepare_data.py --help
```
