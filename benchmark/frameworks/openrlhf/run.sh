#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
MODE="${1:-smoke}"
EXTRA=("${@:2}")

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  if [[ ! -d "${ROOT}/.venv" ]]; then
    uv sync --extra openrlhf
  fi
}

ensure_data() {
  [[ -f benchmark/data/demo/train_rl.jsonl ]] || uv run python benchmark/scripts/generate_demo_data.py
}

case "${MODE}" in
  setup) bash benchmark/frameworks/openrlhf/setup_env.sh ;;
  smoke)
    ensure_uv
    ensure_data
    uv run python benchmark/frameworks/openrlhf/train_ppo.py --smoke --max_steps 2 "${EXTRA[@]}"
    ;;
  ppo|train)
    ensure_uv
    ensure_data
    uv run python benchmark/frameworks/openrlhf/train_ppo.py "${EXTRA[@]}"
    ;;
  *)
    echo "Usage: $0 {setup|smoke|ppo} [extra args]"
    exit 1
    ;;
esac
