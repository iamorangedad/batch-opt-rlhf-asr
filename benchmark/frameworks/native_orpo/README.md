# Native ORPO (in-repo baseline)

Whisper ORPO based on root `finetune.py` + `src/orpo_trainer.py`, used as the
**offline preference-optimization baseline**.

## 1. Environment setup (uv)

```bash
bash benchmark/frameworks/native_orpo/run.sh setup
# equivalent: uv sync
```

## 2. Prepare data

Preference JSONL, one object per line:

```json
{"audio":{"path":"..."},"duration":1.2,"chosen":"correct transcript","rejected":"wrong transcript","language":"en"}
```

Generate demo data:

```bash
python benchmark/scripts/generate_demo_data.py
```

## 3. Run tests

```bash
# smoke
bash benchmark/frameworks/native_orpo/run.sh smoke

# full train
bash benchmark/frameworks/native_orpo/run.sh train
```

## 4. Comparison with other frameworks

| Item | Native ORPO | TRL ORPO |
|---|---|---|
| Algorithm | ORPO | ORPO |
| Data | chosen/rejected | chosen/rejected |
| Entry | `finetune.py` | `benchmark/frameworks/trl/train_orpo.py` |
| Online sampling | no | no |

For online RL (generate then score), use TRL-GRPO / OpenRLHF-PPO / verl-GRPO.
