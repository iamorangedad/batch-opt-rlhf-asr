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
  setup) bash benchmark/frameworks/openrlhf/setup_env.sh ;;
  smoke)
    ensure_data
    python benchmark/frameworks/openrlhf/train_ppo.py --smoke --max_steps 2 "${EXTRA[@]}"
    ;;
  ppo|train)
    ensure_data
    python benchmark/frameworks/openrlhf/train_ppo.py "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: $0 {setup|smoke|ppo} [extra args]"
    exit 1
    ;;
esac
