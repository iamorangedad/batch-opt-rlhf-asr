#!/usr/bin/env bash
# Bootstrap batch-opt-rlhf-asr on NVIDIA Jetson Orin (aarch64 + JetPack).
#
# Generic PyPI torch wheels are CPU/x86 — on Jetson we install CUDA torch from
# the Jetson AI Lab index, then uv-sync the rest without replacing torch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JP6_TORCH_INDEX="${JP6_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.8.0}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.23.0}"
EXTRA="${1:-all}"

die() { echo "[jetson] ERROR: $*" >&2; exit 1; }
info() { echo "[jetson] $*"; }

arch="$(uname -m)"
[[ "${arch}" == "aarch64" ]] || die "expected aarch64, got ${arch} (this script is Jetson-only)"

if [[ ! -f /etc/nv_tegra_release ]] && [[ ! -e /usr/lib/aarch64-linux-gnu/tegra ]]; then
  info "warning: no Tegra markers found; continuing anyway (arch is aarch64)"
fi

if [[ -f /etc/nv_tegra_release ]]; then
  info "Tegra release: $(tr -d '\n' </etc/nv_tegra_release)"
fi

info "Installing system packages (sudo)..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  ffmpeg \
  libsndfile1 \
  libopenblas-dev \
  python3-pip \
  python3-venv \
  ca-certificates

if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi
# Keep uv on PATH for non-interactive shells.
if [[ -x "${HOME}/.local/bin/uv" ]] && [[ ! -e /usr/local/bin/uv ]]; then
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv || true
fi

PY_PIN="$(cat .python-version 2>/dev/null || echo 3.10)"
info "Python pin: ${PY_PIN}"
# JetPack typically ships 3.10; prefer matching the system when uv cannot fetch.
if ! uv python install "${PY_PIN}"; then
  info "uv python install ${PY_PIN} failed; falling back to system python3"
fi

info "Creating / syncing venv skeleton..."
uv venv --python "${PY_PIN}" .venv || uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

info "Installing Jetson CUDA PyTorch from ${JP6_TORCH_INDEX}"
info "  ${TORCH_SPEC} ${TORCHVISION_SPEC}"
python -m pip install --upgrade pip
python -m pip install 'numpy<2'
python -m pip install --upgrade \
  --index-url "${JP6_TORCH_INDEX}" \
  "${TORCH_SPEC}" "${TORCHVISION_SPEC}"

info "uv sync project deps without replacing torch..."
case "${EXTRA}" in
  base) uv sync --no-install-package torch ;;
  trl) uv sync --extra trl --no-install-package torch ;;
  openrlhf) uv sync --extra openrlhf --no-install-package torch ;;
  verl|native_orpo) uv sync --no-install-package torch ;;
  all|*) uv sync --extra all --group dev --no-install-package torch ;;
esac

info "CUDA smoke check..."
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name)
    print("total_memory_mb", round(p.total_memory / (1024 ** 2), 1))
    x = torch.randn(64, 64, device="cuda")
    print("matmul_ok", float((x @ x.T).sum()))
else:
    raise SystemExit("CUDA not available — Jetson torch wheel mismatch?")
PY

info "Environment ready."
info "  Activate: source .venv/bin/activate"
info "  GPU compare: uv run python benchmark/scripts/run_gpu_memory_compare.py"
info "  Docs: benchmark/docs/jetson_orin.md"
