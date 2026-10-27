#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
MODE="${1:-smoke}"
EXTRA=("${@:2}")

ensure_data() {
  [[ -f benchmark/data/demo/train_rl.jsonl ]] || python benchmark/scripts/generate_demo_data.py
}

case "${MODE}" in
  setup) bash benchmark/frameworks/verl/setup_env.sh ;;
  smoke)
    ensure_data
    python benchmark/frameworks/verl/train_grpo.py --smoke --max_steps 2 "${EXTRA[@]}"
    ;;
  grpo|train)
    ensure_data
    python benchmark/frameworks/verl/train_grpo.py "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: $0 {setup|smoke|grpo} [extra args]"
    exit 1
    ;;
esac
