#!/usr/bin/env bash
# Bootstrap the project Python environment with uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[uv] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

EXTRA="${1:-all}"

echo "[uv] Python pin: $(cat .python-version 2>/dev/null || echo '>=3.10')"
uv python install "$(cat .python-version 2>/dev/null || echo 3.12)"

case "${EXTRA}" in
  base)
    uv sync
    ;;
  trl)
    uv sync --extra trl
    ;;
  openrlhf)
    uv sync --extra openrlhf
    ;;
  verl|native_orpo)
    uv sync
    ;;
  all|*)
    uv sync --extra all --group dev
    ;;
esac

echo "[uv] Environment ready at ${ROOT}/.venv"
echo "  Activate:  source .venv/bin/activate"
echo "  Or run:    uv run <command>"
echo "  Smoke:     uv run bash benchmark/scripts/run_benchmark.sh smoke trl"
