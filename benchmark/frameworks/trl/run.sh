#!/usr/bin/env bash
# Entry point for the TRL Whisper RL benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MODE="${1:-smoke}"
EXTRA=("${@:2}")

ensure_data() {
  if [[ ! -f benchmark/data/demo/train_rl.jsonl ]]; then
    echo "[trl] Demo data missing; generating..."
    python benchmark/scripts/generate_demo_data.py
  fi
}

case "${MODE}" in
  smoke)
    ensure_data
    echo "[trl] Running GRPO smoke test..."
    python benchmark/frameworks/trl/train_grpo.py --smoke --max_steps 2 "${EXTRA[@]}"
    echo "[trl] Running ORPO smoke test..."
    python benchmark/frameworks/trl/train_orpo.py --smoke --max_steps 2 "${EXTRA[@]}"
    ;;
  grpo)
    ensure_data
    python benchmark/frameworks/trl/train_grpo.py "${EXTRA[@]}"
    ;;
  orpo)
    ensure_data
    python benchmark/frameworks/trl/train_orpo.py "${EXTRA[@]}"
    ;;
  setup)
    bash benchmark/frameworks/trl/setup_env.sh
    ;;
  *)
    echo "Usage: $0 {setup|smoke|grpo|orpo} [extra args]"
    exit 1
    ;;
esac
