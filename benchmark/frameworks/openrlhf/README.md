# OpenRLHF-style pipeline (Whisper ASR)

OpenRLHF is a production RLHF stack (Ray + DeepSpeed/Megatron + vLLM rollout).
This benchmark provides a **role-aligned Whisper PPO adapter**: Actor / Reference /
Reward match OpenRLHF concepts for comparable test workflows; a full Ray+vLLM
cluster can be added later.

---

## 1. Environment setup (uv)

```bash
bash benchmark/frameworks/openrlhf/setup_env.sh
# equivalent: uv sync --extra openrlhf

source .venv/bin/activate   # optional; or use uv run

# Optional full OpenRLHF (decoder-only / multi-node)
# uv pip install openrlhf vllm
```

---

## 2. Prepare scripts and data

```bash
python benchmark/scripts/generate_demo_data.py
```

Config: `benchmark/configs/frameworks/openrlhf.yaml`

Scripts:

- `train_ppo.py` — colocated PPO
- `run.sh` — setup / smoke / ppo

---

## 3. How to run tests

### 3.1 Smoke

```bash
bash benchmark/frameworks/openrlhf/run.sh smoke
```

### 3.2 PPO training

```bash
bash benchmark/frameworks/openrlhf/run.sh ppo
# or
python benchmark/frameworks/openrlhf/train_ppo.py \
  --config benchmark/configs/frameworks/openrlhf.yaml \
  --max_steps 30
```

Training loop (OpenRLHF-aligned concepts):

```text
Rollout Worker  →  Actor.generate(G hypotheses)
Reward          →  WER/CER rule reward (swap for an RM if needed)
Reference       →  frozen policy for KL
PPO Update      →  clipped surrogate + KL coef
```

### 3.3 Results

Output dir: `outputs/benchmark/openrlhf/`

```bash
python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

---

## Relation to upstream OpenRLHF

| Component | This adapter | Upstream OpenRLHF |
|---|---|---|
| Scheduling | single process | Ray Actor Pool |
| Rollout | HF `generate` | vLLM / async generation |
| Train engine | PyTorch AdamW | DeepSpeed ZeRO / Megatron |
| Model | Whisper seq2seq | mostly Causal LM |
| Reward | rule WER/CER | RM or rules |

If your ASR model is a **decoder-only audio-LLM**, switch to upstream OpenRLHF and
reuse the same `reference` JSONL and reward interface.

---

## Key config

| Param | Meaning |
|---|---|
| `openrlhf.num_generations` | rollout group size |
| `openrlhf.clip_eps` | PPO clip |
| `openrlhf.kl_coef` | KL penalty |
| `openrlhf.ppo_epochs` | PPO epochs per batch |
| `openrlhf.use_ray` | reserved flag (default false) |
