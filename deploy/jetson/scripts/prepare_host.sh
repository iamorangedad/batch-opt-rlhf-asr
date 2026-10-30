#!/usr/bin/env bash
# One-time Jetson host preparation checks for Docker / optional k3s GPU jobs.
# Run on the Orin board (not on an x86 CI runner).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FAIL=0

ok() { echo "[ok] $*"; }
warn() { echo "[warn] $*"; }
bad() { echo "[fail] $*"; FAIL=1; }

echo "=== Jetson host preparation ==="
echo "repo: ${ROOT}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  bad "Expected aarch64 Jetson host, got $(uname -m)"
else
  ok "architecture aarch64"
fi

if [[ -f /etc/nv_tegra_release ]]; then
  ok "L4T release: $(tr -d '\r' </etc/nv_tegra_release | head -1)"
else
  warn "Missing /etc/nv_tegra_release — is this a Jetson with JetPack?"
fi

if command -v docker >/dev/null 2>&1; then
  ok "docker: $(docker --version)"
else
  bad "docker not installed"
fi

if docker info 2>/dev/null | grep -qi nvidia; then
  ok "Docker reports NVIDIA runtime capability"
else
  warn "Docker does not clearly list nvidia — install NVIDIA Container Toolkit"
  warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  warn "  then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# Swap helps Orin Nano 8GB under Whisper + transformers
SWAP_TOTAL="$(free -m | awk '/Swap:/ {print $2}')"
if [[ "${SWAP_TOTAL}" -ge 4096 ]]; then
  ok "swap ${SWAP_TOTAL} MiB"
else
  warn "swap is ${SWAP_TOTAL} MiB; recommend ≥ 8192 MiB on Orin Nano"
  warn "  e.g. create a swapfile or enable zram before long training jobs"
fi

DISK_AVAIL_G="$(df -BG --output=avail "${ROOT}" | tail -1 | tr -dc '0-9')"
if [[ "${DISK_AVAIL_G}" -ge 32 ]]; then
  ok "disk free ~${DISK_AVAIL_G}G at repo"
else
  warn "only ~${DISK_AVAIL_G}G free; images + HF cache need headroom"
fi

# Optional k3s presence
if command -v k3s >/dev/null 2>&1 || command -v kubectl >/dev/null 2>&1; then
  ok "kubectl/k3s present — see deploy/jetson/k8s/ for smoke Job manifests"
  if kubectl get runtimeclass nvidia >/dev/null 2>&1; then
    ok "RuntimeClass/nvidia exists"
  else
    warn "RuntimeClass/nvidia missing — apply deploy/jetson/k8s/runtimeclass.yaml after toolkit setup"
  fi
else
  warn "k3s/kubectl not found (optional; Docker path still works)"
fi

echo
echo "Next:"
echo "  1) Match JETSON_BASE_IMAGE to this JetPack (see deploy/jetson/README.md)"
echo "  2) bash deploy/jetson/scripts/build_and_run.sh smoke"
echo

if [[ "${FAIL}" -ne 0 ]]; then
  echo "Host preparation reported failures."
  exit 1
fi
echo "Host preparation checks finished (warnings may remain)."
