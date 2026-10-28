#!/usr/bin/env python3
"""Compare summary.json files produced by framework adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def find_summaries(root: Path) -> List[Path]:
    return sorted(root.glob("**/summary.json"))


def _fmt_mb(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outputs_dir",
        default="outputs/benchmark",
        help="Root directory containing per-framework run outputs",
    )
    p.add_argument("--out", default=None, help="Optional markdown report path")
    args = p.parse_args()

    root = Path(args.outputs_dir)
    paths = find_summaries(root)
    if not paths:
        print(f"No summary.json under {root}")
        return

    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(path.parent),
                "framework": data.get("framework"),
                "algorithm": data.get("algorithm"),
                "eval_wer": data.get("eval_wer"),
                "eval_cer": data.get("eval_cer"),
                "steps": data.get("steps"),
                "device": data.get("device"),
                "smoke": data.get("smoke"),
                "gpu_peak_allocated_mb": data.get("gpu_peak_allocated_mb"),
                "gpu_peak_reserved_mb": data.get("gpu_peak_reserved_mb"),
                "gpu_total_memory_mb": data.get("gpu_total_memory_mb"),
                "gpu_device_name": data.get("gpu_device_name"),
                "gpu_available": data.get("gpu_available"),
            }
        )

    headers = [
        "framework",
        "algorithm",
        "eval_wer",
        "eval_cer",
        "steps",
        "gpu_peak_alloc_mb",
        "gpu_peak_reserved_mb",
        "gpu_total_mb",
        "gpu_name",
        "smoke",
        "path",
    ]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        vals = [
            str(r["framework"]),
            str(r["algorithm"]),
            f"{r['eval_wer']:.4f}" if isinstance(r["eval_wer"], (int, float)) else "",
            f"{r['eval_cer']:.4f}" if isinstance(r["eval_cer"], (int, float)) else "",
            str(r["steps"]),
            _fmt_mb(r["gpu_peak_allocated_mb"]),
            _fmt_mb(r["gpu_peak_reserved_mb"]),
            _fmt_mb(r["gpu_total_memory_mb"]),
            str(r["gpu_device_name"] or ""),
            str(r["smoke"]),
            str(r["path"]),
        ]
        print(" | ".join(vals))
        lines.append("| " + " | ".join(vals) + " |")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
