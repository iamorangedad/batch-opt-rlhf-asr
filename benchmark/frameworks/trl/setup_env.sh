#!/usr/bin/env bash
# Setup the TRL Whisper RL benchmark environment with uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[trl] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

echo "[trl] Syncing project env with uv (extra: trl)"
uv python install "$(cat .python-version 2>/dev/null || echo 3.12)"
uv sync --extra trl

echo "[trl] Environment ready at ${ROOT}/.venv"
echo "Activate with: source .venv/bin/activate"
echo "Or run via:    uv run ..."
echo "Smoke test:    bash benchmark/frameworks/trl/run.sh smoke"
