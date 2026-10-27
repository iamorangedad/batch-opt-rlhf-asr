#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="${ROOT}/.venvs/openrlhf"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

echo "[openrlhf] Creating venv at ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REQ}"

echo "[openrlhf] Base environment ready."
echo "Optional full OpenRLHF:"
echo "  pip install openrlhf"
echo "Activate: source ${ENV_DIR}/bin/activate"
echo "Smoke:    bash benchmark/frameworks/openrlhf/run.sh smoke"
