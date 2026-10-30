#!/usr/bin/env bash
# Idempotent Cloud Agent / CI install for batch-opt-rlhf-asr (x86_64 hosts).
# Not for Jetson — use scripts/setup_jetson_orin.sh on aarch64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi
if [[ -x "${HOME}/.local/bin/uv" ]] && [[ ! -e /usr/local/bin/uv ]]; then
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv || true
fi

uv python install "$(cat .python-version 2>/dev/null || echo 3.12)"
uv sync --extra all --group dev

echo "[cloud-install] ready: uv run python -c 'import torch; print(torch.__version__)'"
