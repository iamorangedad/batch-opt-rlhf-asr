"""Process / host memory sampling for CPU-side benchmark compares."""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def bytes_to_mb(num_bytes: float) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 2)


def _read_status_kb(keys: tuple[str, ...]) -> Dict[str, Optional[int]]:
    """Read selected fields from /proc/self/status (kB)."""
    out: Dict[str, Optional[int]] = {k: None for k in keys}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                for key in keys:
                    if line.startswith(key + ":"):
                        # e.g. "VmRSS:	 123456 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            out[key] = int(parts[1])
    except OSError:
        pass
    return out


def current_rss_mb() -> float:
    vals = _read_status_kb(("VmRSS",))
    if vals["VmRSS"] is not None:
        return bytes_to_mb(vals["VmRSS"] * 1024)
    # Fallback: ru_maxrss is peak, not current; still better than nothing on non-Linux.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB; macOS: bytes
    if usage > 10_000_000:  # heuristic: treat as bytes
        return bytes_to_mb(usage)
    return bytes_to_mb(usage * 1024)


def peak_rss_mb() -> float:
    """OS-reported high-water RSS for this process."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is kilobytes; macOS is bytes.
    if os.uname().sysname == "Darwin":
        return bytes_to_mb(usage)
    return bytes_to_mb(usage * 1024)


def host_meminfo_mb() -> Dict[str, Optional[float]]:
    info: Dict[str, Optional[float]] = {
        "host_mem_total_mb": None,
        "host_mem_available_mb": None,
    }
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["host_mem_total_mb"] = bytes_to_mb(int(line.split()[1]) * 1024)
                elif line.startswith("MemAvailable:"):
                    info["host_mem_available_mb"] = bytes_to_mb(
                        int(line.split()[1]) * 1024
                    )
    except OSError:
        pass
    return info


@dataclass
class CpuMemoryTracker:
    """Track process RSS / peak RSS during a training or eval run."""

    baseline_rss_mb: float = field(init=False, default=0.0)
    peak_rss_mb: float = field(init=False, default=0.0)
    peak_sampled_rss_mb: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.baseline_rss_mb = current_rss_mb()
        self.peak_sampled_rss_mb = self.baseline_rss_mb
        self.peak_rss_mb = peak_rss_mb()
        self.sample()

    def sample(self) -> Dict[str, Any]:
        rss = current_rss_mb()
        os_peak = peak_rss_mb()
        self.peak_sampled_rss_mb = max(self.peak_sampled_rss_mb, rss)
        self.peak_rss_mb = max(self.peak_rss_mb, os_peak, self.peak_sampled_rss_mb)
        host = host_meminfo_mb()
        return {
            "cpu_rss_mb": rss,
            "cpu_peak_rss_mb": self.peak_rss_mb,
            "cpu_baseline_rss_mb": self.baseline_rss_mb,
            "cpu_rss_delta_mb": round(rss - self.baseline_rss_mb, 2),
            "cpu_peak_rss_delta_mb": round(self.peak_rss_mb - self.baseline_rss_mb, 2),
            **host,
        }

    def summary(self) -> Dict[str, Any]:
        snap = self.sample()
        return {
            "cpu_baseline_rss_mb": snap["cpu_baseline_rss_mb"],
            "cpu_peak_rss_mb": snap["cpu_peak_rss_mb"],
            "cpu_peak_rss_delta_mb": snap["cpu_peak_rss_delta_mb"],
            "host_mem_total_mb": snap["host_mem_total_mb"],
            "host_mem_available_mb": snap["host_mem_available_mb"],
        }
