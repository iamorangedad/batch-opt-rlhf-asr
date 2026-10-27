#!/usr/bin/env bash
# Wrapper around the repository's existing ORPO trainer (finetune.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
MODE="${1:-smoke}"
EXTRA=("${@:2}")

ensure_data() {
  if [[ ! -f benchmark/data/demo/train_pref.jsonl ]]; then
    python benchmark/scripts/generate_demo_data.py
  fi
}

case "${MODE}" in
  setup)
    python -m pip install -U pip
    python -m pip install torch transformers datasets librosa soundfile peft pyyaml tqdm numpy zhconv
    ;;
  smoke)
    ensure_data
    echo "[native_orpo] Smoke uses the preference JSONL produced by generate_demo_data.py"
    echo "[native_orpo] Launching finetune.py for 1 epoch on demo data..."
    python finetune.py \
      --base_model openai/whisper-tiny \
      --train_data benchmark/data/demo/train_pref.jsonl \
      --test_data benchmark/data/demo/eval_pref.jsonl \
      --output_dir outputs/benchmark/native_orpo \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --logging_steps 1 \
      --save_steps 10 \
      --num_workers 0 \
      --fp16 False \
      --use_adalora False \
      --enable_lora True \
      --is_test False \
      "${EXTRA[@]}" || {
        echo "[native_orpo] Note: legacy finetune.py may need src/log.py and import fixes."
        echo "Prefer TRL ORPO adapter for a green-field smoke path:"
        echo "  bash benchmark/frameworks/trl/run.sh orpo --smoke"
        exit 1
      }
    ;;
  train)
    ensure_data
    python finetune.py \
      --base_model openai/whisper-tiny \
      --train_data benchmark/data/demo/train_pref.jsonl \
      --test_data benchmark/data/demo/eval_pref.jsonl \
      --output_dir outputs/benchmark/native_orpo \
      "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: $0 {setup|smoke|train} [extra args]"
    exit 1
    ;;
esac
