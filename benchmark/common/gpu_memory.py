"""GPU memory sampling helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _device_index(device: Optional[Any] = None) -> int:
    import torch

    if device is None:
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    if isinstance(device, int):
        return device
    if isinstance(device, str) and device.startswith("cuda"):
        # "cuda" or "cuda:0"
        parts = device.split(":")
        return int(parts[1]) if len(parts) > 1 else 0
    if hasattr(device, "index") and device.index is not None:
        return int(device.index)
    return 0


def bytes_to_mb(num_bytes: int) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 2)


@dataclass
class GpuMemoryTracker:
    """Track current and peak CUDA memory for a training/eval run."""

    device: Any = None
    enabled: bool = field(init=False, default=False)
    device_index: int = field(init=False, default=0)
    device_name: Optional[str] = field(init=False, default=None)
    total_memory_mb: Optional[float] = field(init=False, default=None)
    peak_allocated_mb: float = field(init=False, default=0.0)
    peak_reserved_mb: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.enabled = cuda_available()
        if not self.enabled:
            return
        import torch

        self.device_index = _device_index(self.device)
        props = torch.cuda.get_device_properties(self.device_index)
        self.device_name = props.name
        self.total_memory_mb = bytes_to_mb(props.total_memory)
        self.reset()

    def reset(self) -> None:
        if not self.enabled:
            return
        import torch

        torch.cuda.synchronize(self.device_index)
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0
        self.sample()

    def sample(self) -> Dict[str, Any]:
        """Sample current + peak memory; update running peaks."""
        if not self.enabled:
            return {
                "gpu_available": False,
                "gpu_allocated_mb": None,
                "gpu_reserved_mb": None,
                "gpu_peak_allocated_mb": None,
                "gpu_peak_reserved_mb": None,
            }
        import torch

        torch.cuda.synchronize(self.device_index)
        allocated = bytes_to_mb(torch.cuda.memory_allocated(self.device_index))
        reserved = bytes_to_mb(torch.cuda.memory_reserved(self.device_index))
        peak_alloc = bytes_to_mb(
            torch.cuda.max_memory_allocated(self.device_index)
        )
        peak_reserved = bytes_to_mb(
            torch.cuda.max_memory_reserved(self.device_index)
        )
        self.peak_allocated_mb = max(self.peak_allocated_mb, peak_alloc)
        self.peak_reserved_mb = max(self.peak_reserved_mb, peak_reserved)
        return {
            "gpu_available": True,
            "gpu_device_index": self.device_index,
            "gpu_device_name": self.device_name,
            "gpu_total_memory_mb": self.total_memory_mb,
            "gpu_allocated_mb": allocated,
            "gpu_reserved_mb": reserved,
            "gpu_peak_allocated_mb": self.peak_allocated_mb,
            "gpu_peak_reserved_mb": self.peak_reserved_mb,
        }

    def summary(self) -> Dict[str, Any]:
        snap = self.sample()
        if not self.enabled:
            return {
                "gpu_available": False,
                "gpu_peak_allocated_mb": None,
                "gpu_peak_reserved_mb": None,
                "gpu_device_name": None,
                "gpu_total_memory_mb": None,
            }
        return {
            "gpu_available": True,
            "gpu_device_name": snap["gpu_device_name"],
            "gpu_device_index": snap["gpu_device_index"],
            "gpu_total_memory_mb": snap["gpu_total_memory_mb"],
            "gpu_peak_allocated_mb": snap["gpu_peak_allocated_mb"],
            "gpu_peak_reserved_mb": snap["gpu_peak_reserved_mb"],
        }
