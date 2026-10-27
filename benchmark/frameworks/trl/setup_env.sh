#!/usr/bin/env bash
# Setup a dedicated virtualenv for the TRL Whisper RL benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="${ROOT}/.venvs/trl"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

echo "[trl] Creating venv at ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REQ}"

echo "[trl] Environment ready."
echo "Activate with: source ${ENV_DIR}/bin/activate"
echo "Smoke test:    bash benchmark/frameworks/trl/run.sh smoke"
