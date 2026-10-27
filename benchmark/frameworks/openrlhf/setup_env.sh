#!/usr/bin/env bash
# Setup the OpenRLHF-style Whisper RL environment with uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[openrlhf] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

echo "[openrlhf] Syncing project env with uv (extra: openrlhf)"
uv python install "$(cat .python-version 2>/dev/null || echo 3.12)"
uv sync --extra openrlhf

echo "[openrlhf] Environment ready at ${ROOT}/.venv"
echo "Optional full OpenRLHF stack: uv pip install openrlhf vllm"
echo "Activate with: source .venv/bin/activate"
echo "Or run via:    uv run ..."
echo "Smoke test:    bash benchmark/frameworks/openrlhf/run.sh smoke"
