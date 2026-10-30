#!/usr/bin/env bash
# Sanity-check NVIDIA / CUDA / PyTorch on Jetson (host or container).
set -euo pipefail

echo "=== platform ==="
uname -a
echo "arch=$(uname -m)"

echo "=== L4T / JetPack hints ==="
if [[ -f /etc/nv_tegra_release ]]; then
  cat /etc/nv_tegra_release
else
  echo "(no /etc/nv_tegra_release — may be a generic container userspace)"
fi
if [[ -f /etc/nv_tegra_release ]] || [[ -d /usr/local/cuda ]]; then
  ls -d /usr/local/cuda* 2>/dev/null || true
fi

echo "=== nvidia-smi (may be limited on some JetPack versions) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi returned non-zero (common on some L4T setups)"
else
  echo "nvidia-smi not found"
fi

echo "=== tegrastats snapshot (host Jetson) ==="
if command -v tegrastats >/dev/null 2>&1; then
  timeout 3 tegrastats || true
else
  echo "tegrastats not found (ok inside slim containers)"
fi

echo "=== PyTorch CUDA ==="
python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("FAILED: cannot import torch:", e)
    sys.exit(1)

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul_ok:", float(y.mean()))
else:
    print("FAILED: torch.cuda.is_available() is False")
    sys.exit(2)
print("GPU check passed")
PY
