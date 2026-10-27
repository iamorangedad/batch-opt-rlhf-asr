#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="${ROOT}/.venvs/verl"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

echo "[verl] Creating venv at ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REQ}"

echo "[verl] Base environment ready."
echo "Optional full verl: pip install verl"
echo "Activate: source ${ENV_DIR}/bin/activate"
echo "Smoke:    bash benchmark/frameworks/verl/run.sh smoke"
