# Data format

## Online RL: `*_rl.jsonl`

One sample per line:

```json
{
  "audio": {"path": "benchmark/data/demo/audio/train_000.wav"},
  "duration": 1.5,
  "reference": "the quick brown fox jumps over the lazy dog",
  "language": "en",
  "id": "train-0"
}
```

| Field | Required | Description |
|---|---|---|
| `audio.path` / `audio_path` | yes | Local wav/flac path |
| `reference` | yes | Ground-truth transcript (reward + eval) |
| `language` | no | Defaults to config language |
| `duration` | no | Seconds; used to filter short/long clips |
| `id` | no | Tracking id |

## Offline preference: `*_pref.jsonl`

Same as RL rows, plus:

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

Compatible with the repo's original ORPO `finetune.py` format.

## How to generate

```bash
# Pipeline demo
python benchmark/scripts/generate_demo_data.py

# From existing JSONL / HF datasets
python benchmark/scripts/prepare_data.py --help
```
