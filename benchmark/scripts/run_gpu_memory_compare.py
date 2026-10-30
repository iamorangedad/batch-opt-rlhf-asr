#!/usr/bin/env python3
"""Run isolated per-framework jobs and compare GPU peak VRAM on this host.

Each framework is launched in a fresh subprocess so CUDA allocations from one
job do not inflate the next. Requires CUDA (e.g. Jetson Orin or discrete GPU).

Companion to ``run_cpu_memory_compare.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


JOBS = [
    {
        "name": "trl_grpo",
        "framework": "trl",
        "algorithm": "grpo",
        "outdir": "outputs/benchmark/gpu_mem_trl_grpo",
        "cmd": [
            "python",
            "benchmark/frameworks/trl/train_grpo.py",
            "--config",
            "benchmark/configs/frameworks/jetson_trl_grpo.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/gpu_mem_trl_grpo",
        ],
    },
    {
        "name": "trl_orpo",
        "framework": "trl",
        "algorithm": "orpo",
        "outdir": "outputs/benchmark/gpu_mem_trl_orpo",
        "cmd": [
            "python",
            "benchmark/frameworks/trl/train_orpo.py",
            "--config",
            "benchmark/configs/frameworks/jetson_trl_orpo.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/gpu_mem_trl_orpo",
        ],
    },
    {
        "name": "openrlhf_ppo",
        "framework": "openrlhf",
        "algorithm": "ppo",
        "outdir": "outputs/benchmark/gpu_mem_openrlhf",
        "cmd": [
            "python",
            "benchmark/frameworks/openrlhf/train_ppo.py",
            "--config",
            "benchmark/configs/frameworks/jetson_openrlhf.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/gpu_mem_openrlhf",
        ],
    },
    {
        "name": "verl_grpo",
        "framework": "verl",
        "algorithm": "grpo",
        "outdir": "outputs/benchmark/gpu_mem_verl",
        "cmd": [
            "python",
            "benchmark/frameworks/verl/train_grpo.py",
            "--config",
            "benchmark/configs/frameworks/jetson_verl.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/gpu_mem_verl",
        ],
    },
    {
        "name": "native_orpo",
        "framework": "native_orpo",
        "algorithm": "orpo",
        "outdir": "outputs/benchmark/gpu_mem_native_orpo",
        "cmd": [
            "python",
            "finetune.py",
            "--base_model",
            "openai/whisper-tiny",
            "--train_data",
            "benchmark/data/librispeech_tiny/train_pref.jsonl",
            "--test_data",
            "benchmark/data/librispeech_tiny/eval_pref.jsonl",
            "--output_dir",
            "outputs/benchmark/gpu_mem_native_orpo",
            "--num_train_epochs",
            "1",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--logging_steps",
            "1",
            "--save_steps",
            "50",
            "--num_workers",
            "0",
            "--fp16",
            "True",
            "--use_adalora",
            "False",
            "--enable_lora",
            "True",
            "--is_test",
            "False",
            "--learning_rate",
            "1e-4",
            "--language",
            "en",
            "--use_8bit",
            "False",
            "--optim",
            "adamw_torch",
        ],
    },
]


def _require_cuda() -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch is required for GPU memory compare") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Run this on a GPU host (e.g. Jetson Orin).\n"
            "See benchmark/docs/jetson_orin.md"
        )
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_device_name": props.name,
        "gpu_total_memory_mb": round(props.total_memory / (1024.0 * 1024.0), 2),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "arch": os.uname().machine,
    }


def run_job(job: dict, report_dir: Path) -> dict:
    outdir = REPO_ROOT / job["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / f"{job['name']}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Do not force-hide CUDA; optionally pin to device 0.
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            job["cmd"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = round(time.time() - t0, 2)

    summary_path = outdir / "summary.json"
    if not summary_path.exists():
        candidates = list(outdir.glob("**/summary.json"))
        summary_path = candidates[0] if candidates else summary_path

    row = {
        "name": job["name"],
        "framework": job["framework"],
        "algorithm": job["algorithm"],
        "exit_code": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "summary_path": str(summary_path.relative_to(REPO_ROOT))
        if summary_path.exists()
        else None,
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row.update(
            {
                "steps": summary.get("steps"),
                "eval_wer": summary.get("eval_wer"),
                "eval_cer": summary.get("eval_cer"),
                "device": summary.get("device"),
                "cpu_peak_rss_mb": summary.get("cpu_peak_rss_mb"),
                "cpu_peak_rss_delta_mb": summary.get("cpu_peak_rss_delta_mb"),
                "gpu_available": summary.get("gpu_available"),
                "gpu_device_name": summary.get("gpu_device_name"),
                "gpu_total_memory_mb": summary.get("gpu_total_memory_mb"),
                "gpu_peak_allocated_mb": summary.get("gpu_peak_allocated_mb"),
                "gpu_peak_reserved_mb": summary.get("gpu_peak_reserved_mb"),
            }
        )
    return row


def write_report(rows: list[dict], report_dir: Path, host: dict) -> Path:
    report = {
        "title": "GPU memory comparison across Whisper RL frameworks",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "protocol": {
            "device": "cuda",
            "base_model": "openai/whisper-tiny",
            "data": "benchmark/data/librispeech_tiny",
            "configs": "benchmark/configs/frameworks/jetson_*.yaml",
            "online_rl_steps": 5,
            "isolation": "fresh subprocess per framework",
            "metric": "torch CUDA peak allocated / reserved",
        },
        "results": rows,
    }
    (report_dir / "gpu_memory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    ranked = sorted(
        [r for r in rows if isinstance(r.get("gpu_peak_allocated_mb"), (int, float))],
        key=lambda r: r["gpu_peak_allocated_mb"],
    )

    lines = [
        "# GPU memory comparison report",
        "",
        f"- Generated (UTC): `{report['created_at_utc']}`",
        f"- Device: **{host.get('gpu_device_name')}** "
        f"({host.get('gpu_total_memory_mb')} MiB total, arch `{host.get('arch')}`)",
        f"- Torch: `{host.get('torch_version')}` CUDA `{host.get('cuda_version')}`",
        "- Model: `openai/whisper-tiny` + LoRA",
        "- Data: `benchmark/data/librispeech_tiny`",
        "- Configs: Jetson memory-conservative (`jetson_*.yaml`, fp16)",
        "- Isolation: **one fresh Python process per framework**",
        "- Metric: CUDA **peak allocated** / peak reserved",
        "",
        "## Results",
        "",
        "| rank | name | framework | algorithm | peak alloc (MiB) | peak reserved (MiB) | "
        "eval WER | elapsed (s) | status |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in enumerate(ranked, start=1):
        status = "ok" if r.get("exit_code") == 0 else f"fail({r.get('exit_code')})"
        wer = r.get("eval_wer")
        wer_s = f"{wer:.4f}" if isinstance(wer, (int, float)) else "-"
        lines.append(
            "| {rank} | {name} | {fw} | {algo} | {alloc:.1f} | {res:.1f} | {wer} | "
            "{elapsed} | {status} |".format(
                rank=i,
                name=r["name"],
                fw=r["framework"],
                algo=r["algorithm"],
                alloc=r["gpu_peak_allocated_mb"],
                res=r.get("gpu_peak_reserved_mb") or 0.0,
                wer=wer_s,
                elapsed=r.get("elapsed_sec"),
                status=status,
            )
        )

    missing_gpu = [
        r
        for r in rows
        if r.get("exit_code") == 0 and not isinstance(r.get("gpu_peak_allocated_mb"), (int, float))
    ]
    failed = [r for r in rows if r.get("exit_code") != 0]
    if missing_gpu:
        lines += ["", "## Jobs without GPU stats", ""]
        for r in missing_gpu:
            lines.append(f"- `{r['name']}` (exit ok but `gpu_peak_allocated_mb` missing)")
    if failed:
        lines += ["", "## Failed jobs", ""]
        for r in failed:
            lines.append(f"- `{r['name']}` exit={r['exit_code']} log=`{r['log']}`")

    lines += [
        "",
        "## Notes",
        "",
        "- Peak allocated is the best cross-framework apples-to-apples VRAM signal.",
        "- Peak reserved can be higher due to the CUDA caching allocator.",
        "- On Jetson, unified memory is shared with the OS; close desktop apps before runs.",
        "- CPU-only hosts should use `run_cpu_memory_compare.py` instead.",
        "",
        f"JSON: `{report_dir / 'gpu_memory_report.json'}`",
    ]
    text = "\n".join(lines) + "\n"
    md_path = report_dir / "gpu_memory_report.md"
    docs_path = REPO_ROOT / "benchmark/docs/gpu_memory_report.md"
    md_path.write_text(text, encoding="utf-8")
    docs_path.write_text(text, encoding="utf-8")
    return docs_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jobs",
        default="trl_grpo,trl_orpo,openrlhf_ppo,verl_grpo,native_orpo",
        help="Comma-separated job names from JOBS",
    )
    p.add_argument(
        "--allow-cpu-fail",
        action="store_true",
        help="Skip the CUDA preflight (jobs will still need CUDA to fill GPU fields).",
    )
    args = p.parse_args()

    host = {
        "gpu_device_name": None,
        "gpu_total_memory_mb": None,
        "torch_version": None,
        "cuda_version": None,
        "arch": os.uname().machine,
    }
    if not args.allow_cpu_fail:
        host = _require_cuda()

    wanted = {x.strip() for x in args.jobs.split(",") if x.strip()}
    jobs = [j for j in JOBS if j["name"] in wanted]

    report_dir = REPO_ROOT / "outputs/benchmark/gpu_memory_compare"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for job in jobs:
        print(f"==== running {job['name']} ====", flush=True)
        row = run_job(job, report_dir)
        rows.append(row)
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "exit_code": row["exit_code"],
                    "gpu_peak_allocated_mb": row.get("gpu_peak_allocated_mb"),
                    "gpu_peak_reserved_mb": row.get("gpu_peak_reserved_mb"),
                    "eval_wer": row.get("eval_wer"),
                    "elapsed_sec": row.get("elapsed_sec"),
                }
            ),
            flush=True,
        )

    docs = write_report(rows, report_dir, host)
    print(f"Wrote {docs}")
    print((report_dir / "gpu_memory_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    if Path(sys.executable).name:
        for job in JOBS:
            if job["cmd"][0] == "python":
                job["cmd"][0] = sys.executable
    main()
