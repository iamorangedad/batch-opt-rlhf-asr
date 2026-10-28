# verl (HybridFlow)-style pipeline (Whisper ASR)

verl (Volcano Engine RL) is known for a **single controller + colocated
rollout/train engine swap**. This benchmark implements a **phase-clear** Whisper
GRPO controller matching HybridFlow's:

```text
rollout_phase → reward_phase → update_phase
```

so verl-style scheduling can be compared on the same ASR data and reward.

---

## 1. Environment setup (uv)

```bash
bash benchmark/frameworks/verl/setup_env.sh
# equivalent: uv sync

source .venv/bin/activate   # optional; or use uv run

# Optional full verl (decoder-only + FSDP/Megatron + vLLM)
# uv pip install verl
```

---

## 2. Prepare scripts and data

```bash
python benchmark/scripts/generate_demo_data.py
```

Config: `benchmark/configs/frameworks/verl.yaml`  
Scripts: `train_grpo.py`, `run.sh`

---

## 3. How to run tests

### 3.1 Smoke

```bash
bash benchmark/frameworks/verl/run.sh smoke
```

### 3.2 GRPO training

```bash
bash benchmark/frameworks/verl/run.sh grpo
```

Controller logic (`SingleController`):

1. **rollout_phase**: sample G transcripts from the policy  
2. **reward_phase**: WER/CER reward + group-relative advantage  
3. **update_phase**: policy gradient + KL to reference  

Metrics go to `outputs/benchmark/verl/train_metrics.jsonl` with
`"phase": "rollout->reward->update"`.

### 3.3 Compare

```bash
python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## Relation to upstream verl

| Capability | This adapter | Upstream verl |
|---|---|---|
| Control model | single-controller phase API | HybridFlow single controller |
| Weight reshard | none (same engine) | 3D-HybridEngine reshard |
| Rollout engine | HF generate | vLLM / SGLang |
| Parallelism | single GPU / simple DDP | FSDP2 / Megatron 3D |
| Whisper | yes | needs custom worker |

Use this adapter for **benchmark workflow and algorithm comparison**. For large
decoder-only ASR-LLMs, attach the same reward function to verl's reward manager.
