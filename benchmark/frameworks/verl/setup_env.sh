#!/usr/bin/env bash
# Setup the verl-style Whisper RL environment with uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[verl] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

echo "[verl] Syncing project env with uv (base deps)"
uv python install "$(cat .python-version 2>/dev/null || echo 3.12)"
uv sync

echo "[verl] Environment ready at ${ROOT}/.venv"
echo "Optional full verl: uv pip install verl"
echo "Activate with: source .venv/bin/activate"
echo "Or run via:    uv run ..."
echo "Smoke test:    bash benchmark/frameworks/verl/run.sh smoke"
