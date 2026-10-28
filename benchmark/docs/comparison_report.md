# Whisper RL benchmark comparison report

- Generated (UTC): `2026-10-28T20:38:38.214218+00:00`
- Device: **CPU** (no GPU available in this run)
- Data: `hf-internal-testing/librispeech_asr_dummy` — 12 train / 4 eval
- Base model: `openai/whisper-tiny` + LoRA
- Online RL steps: `8` with `num_generations=2`

## Results (shared evaluator)

| name | framework | algorithm | steps | WER | CER | ΔWER vs base | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| base_whisper_tiny | base | none | 0 | 0.0706 | 0.0426 |  | ok |
| trl_grpo | trl | grpo | 8 | 0.0706 | 0.0426 | +0.0000 | ok |
| trl_orpo | trl | orpo | 8 | 0.0706 | 0.0426 | +0.0000 | ok |
| openrlhf_ppo | openrlhf | ppo | 8 | 0.0706 | 0.0426 | +0.0000 | ok |
| verl_grpo | verl | grpo | 8 | 0.0706 | 0.0426 | +0.0000 | ok |
| native_orpo_orpo | native_orpo | orpo | 12 | 0.0706 | 0.0426 | +0.0000 | ok |

## Interpretation

- All mainstream adapters (**TRL GRPO/ORPO**, **OpenRLHF PPO**, **verl GRPO**) completed successfully.
- **Native ORPO** (`finetune.py`) completed after compatibility fixes for current transformers/peft APIs.
- Under this short CPU budget, post-train WER matches the base model within noise; the run validates **end-to-end comparability**, not SOTA quality.

## How to reproduce (GPU recommended)

```bash
bash scripts/setup_uv.sh all
uv run python benchmark/scripts/prepare_data.py \
  --hf_dataset hf-internal-testing/librispeech_asr_dummy --hf_config clean \
  --hf_split validation --max_samples 200 --out_dir benchmark/data/librispeech_tiny
# or use the committed tiny set under benchmark/data/librispeech_tiny
uv run python benchmark/frameworks/trl/train_grpo.py --config benchmark/configs/frameworks/compare_trl_grpo.yaml
uv run python benchmark/frameworks/trl/train_orpo.py --config benchmark/configs/frameworks/compare_trl_orpo.yaml
uv run python benchmark/frameworks/openrlhf/train_ppo.py --config benchmark/configs/frameworks/compare_openrlhf.yaml
uv run python benchmark/frameworks/verl/train_grpo.py --config benchmark/configs/frameworks/compare_verl.yaml
uv run python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark --out outputs/benchmark/comparison_run/comparison.md
```

Raw artifacts: `outputs/benchmark/comparison_run/`
