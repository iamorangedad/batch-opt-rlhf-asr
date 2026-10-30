#!/usr/bin/env bash
# Container entrypoint for Jetson Orin smoke / train / shell / check.
# Uses the L4T image's system python3 + torch (do not call uv sync here).
set -euo pipefail

ROOT="${WORKDIR:-/workspace}"
cd "${ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/jetson}"
MODE="${1:-smoke}"
shift || true

log() { echo "[jetson-entrypoint] $*"; }

ensure_demo_data() {
  if [[ ! -f benchmark/data/demo/train_pref.jsonl ]]; then
    log "Generating demo data..."
    python3 benchmark/scripts/generate_demo_data.py
  fi
}

# Shared low-memory knobs for Orin Nano-class boards.
finetune_jetson() {
  local out="$1"
  shift
  mkdir -p "${out}"
  python3 finetune.py \
    --base_model openai/whisper-tiny \
    --train_data benchmark/data/demo/train_pref.jsonl \
    --test_data benchmark/data/demo/eval_pref.jsonl \
    --output_dir "${out}" \
    --language en \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_steps 5 \
    --logging_steps 1 \
    --save_steps 20 \
    --eval_steps 20 \
    --num_workers 0 \
    --max_audio_len 10 \
    --fp16 True \
    --use_8bit False \
    --use_adalora False \
    --enable_lora True \
    --is_test False \
    --optim adamw_torch \
    "$@"
}

case "${MODE}" in
  check|gpu)
    exec bash deploy/jetson/scripts/check_gpu.sh "$@"
    ;;
  metrics)
    log "Running metrics/rewards unit test..."
    exec python3 benchmark/tests/test_metrics_rewards.py "$@"
    ;;
  smoke)
    log "GPU check..."
    bash deploy/jetson/scripts/check_gpu.sh
    ensure_demo_data
    log "Running metrics unit test..."
    python3 benchmark/tests/test_metrics_rewards.py
    log "Running short Whisper ORPO smoke → ${OUTPUT_DIR}/native_orpo"
    finetune_jetson "${OUTPUT_DIR}/native_orpo" "$@"
    log "Smoke finished → ${OUTPUT_DIR}/native_orpo"
    ;;
  train)
    ensure_demo_data
    log "Short Jetson training run → ${OUTPUT_DIR}/train"
    finetune_jetson "${OUTPUT_DIR}/train" \
      --gradient_accumulation_steps 8 \
      --max_audio_len 15 \
      "$@"
    ;;
  shell|bash)
    exec bash "$@"
    ;;
  *)
    # Arbitrary command: docker compose run asr-jetson python3 finetune.py ...
    exec "${MODE}" "$@"
    ;;
esac
