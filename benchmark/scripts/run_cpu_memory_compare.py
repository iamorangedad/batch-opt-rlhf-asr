#!/usr/bin/env python3
"""Run isolated per-framework jobs and compare CPU peak RSS on this host.

Each framework is launched in a fresh subprocess so RSS is not inflated by
previous imports / leftover tensors from earlier jobs.
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
        "outdir": "outputs/benchmark/cpu_mem_trl_grpo",
        "cmd": [
            "python",
            "benchmark/frameworks/trl/train_grpo.py",
            "--config",
            "benchmark/configs/frameworks/compare_trl_grpo.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/cpu_mem_trl_grpo",
        ],
    },
    {
        "name": "trl_orpo",
        "framework": "trl",
        "algorithm": "orpo",
        "outdir": "outputs/benchmark/cpu_mem_trl_orpo",
        "cmd": [
            "python",
            "benchmark/frameworks/trl/train_orpo.py",
            "--config",
            "benchmark/configs/frameworks/compare_trl_orpo.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/cpu_mem_trl_orpo",
        ],
    },
    {
        "name": "openrlhf_ppo",
        "framework": "openrlhf",
        "algorithm": "ppo",
        "outdir": "outputs/benchmark/cpu_mem_openrlhf",
        "cmd": [
            "python",
            "benchmark/frameworks/openrlhf/train_ppo.py",
            "--config",
            "benchmark/configs/frameworks/compare_openrlhf.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/cpu_mem_openrlhf",
        ],
    },
    {
        "name": "verl_grpo",
        "framework": "verl",
        "algorithm": "grpo",
        "outdir": "outputs/benchmark/cpu_mem_verl",
        "cmd": [
            "python",
            "benchmark/frameworks/verl/train_grpo.py",
            "--config",
            "benchmark/configs/frameworks/compare_verl.yaml",
            "--max_steps",
            "5",
            "--output_dir",
            "outputs/benchmark/cpu_mem_verl",
        ],
    },
    {
        "name": "native_orpo",
        "framework": "native_orpo",
        "algorithm": "orpo",
        "outdir": "outputs/benchmark/cpu_mem_native_orpo",
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
            "outputs/benchmark/cpu_mem_native_orpo",
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
            "False",
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


def run_job(job: dict, report_dir: Path) -> dict:
    outdir = REPO_ROOT / job["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / f"{job['name']}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force CPU for this comparison.
    env["CUDA_VISIBLE_DEVICES"] = ""

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
    # native ORPO writes under output_dir/<model_basename>/summary.json
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
                "cpu_baseline_rss_mb": summary.get("cpu_baseline_rss_mb"),
                "cpu_peak_rss_mb": summary.get("cpu_peak_rss_mb"),
                "cpu_peak_rss_delta_mb": summary.get("cpu_peak_rss_delta_mb"),
                "host_mem_total_mb": summary.get("host_mem_total_mb"),
                "gpu_available": summary.get("gpu_available"),
                "gpu_peak_allocated_mb": summary.get("gpu_peak_allocated_mb"),
            }
        )
    return row


def write_report(rows: list[dict], report_dir: Path) -> Path:
    report = {
        "title": "CPU memory comparison across Whisper RL frameworks",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "device": "cpu",
            "base_model": "openai/whisper-tiny",
            "data": "benchmark/data/librispeech_tiny",
            "online_rl_steps": 5,
            "isolation": "fresh subprocess per framework",
            "metric": "process peak RSS (ru_maxrss / VmRSS)",
        },
        "results": rows,
    }
    (report_dir / "cpu_memory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Sort by peak RSS when available.
    ranked = sorted(
        [r for r in rows if isinstance(r.get("cpu_peak_rss_mb"), (int, float))],
        key=lambda r: r["cpu_peak_rss_mb"],
    )

    lines = [
        "# CPU memory comparison report",
        "",
        f"- Generated (UTC): `{report['created_at_utc']}`",
        "- Device: **CPU** (`CUDA_VISIBLE_DEVICES=` forced empty)",
        "- Model: `openai/whisper-tiny` + LoRA",
        "- Data: `benchmark/data/librispeech_tiny`",
        "- Isolation: **one fresh Python process per framework**",
        "- Metric: process **peak RSS** (`cpu_peak_rss_mb`) and delta vs baseline",
        "",
        "## Results",
        "",
        "| rank | name | framework | algorithm | peak RSS (MiB) | ΔRSS (MiB) | baseline (MiB) | elapsed (s) | status |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in enumerate(ranked, start=1):
        status = "ok" if r.get("exit_code") == 0 else f"fail({r.get('exit_code')})"
        lines.append(
            "| {rank} | {name} | {fw} | {algo} | {peak:.1f} | {delta:.1f} | {base:.1f} | {elapsed} | {status} |".format(
                rank=i,
                name=r["name"],
                fw=r["framework"],
                algo=r["algorithm"],
                peak=r["cpu_peak_rss_mb"],
                delta=r.get("cpu_peak_rss_delta_mb") or 0.0,
                base=r.get("cpu_baseline_rss_mb") or 0.0,
                elapsed=r.get("elapsed_sec"),
                status=status,
            )
        )

    failed = [r for r in rows if r.get("exit_code") != 0]
    if failed:
        lines += ["", "## Failed jobs", ""]
        for r in failed:
            lines.append(f"- `{r['name']}` exit={r['exit_code']} log=`{r['log']}`")

    lines += [
        "",
        "## Notes",
        "",
        "- Peak RSS includes Python + PyTorch + model weights + activations for that process.",
        "- Online RL (GRPO/PPO) usually keeps a frozen reference copy, so RSS tends to be higher than offline ORPO.",
        "- Native ORPO uses HuggingFace Trainer; one epoch over the tiny set (~12 steps) may differ from the fixed 5-step adapters.",
        "- This measures **process memory**, not host-wide pressure from other agents.",
        "",
        f"JSON: `{report_dir / 'cpu_memory_report.json'}`",
    ]
    md_path = report_dir / "cpu_memory_report.md"
    # Also publish under docs for the repo.
    docs_path = REPO_ROOT / "benchmark/docs/cpu_memory_report.md"
    text = "\n".join(lines) + "\n"
    md_path.write_text(text, encoding="utf-8")
    docs_path.write_text(text, encoding="utf-8")
    return docs_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jobs",
        default="trl_grpo,trl_orpo,openrlhf_ppo,verl_grpo,native_orpo",
        help="Comma-separated job names from JOBS",
    )
    args = p.parse_args()
    wanted = {x.strip() for x in args.jobs.split(",") if x.strip()}
    jobs = [j for j in JOBS if j["name"] in wanted]

    report_dir = REPO_ROOT / "outputs/benchmark/cpu_memory_compare"
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
                    "cpu_peak_rss_mb": row.get("cpu_peak_rss_mb"),
                    "cpu_peak_rss_delta_mb": row.get("cpu_peak_rss_delta_mb"),
                    "elapsed_sec": row.get("elapsed_sec"),
                }
            ),
            flush=True,
        )

    docs = write_report(rows, report_dir)
    print(f"Wrote {docs}")
    print((report_dir / "cpu_memory_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    # Prefer uv-run python if invoked via uv; otherwise current interpreter.
    if Path(sys.executable).name:
        for job in JOBS:
            if job["cmd"][0] == "python":
                job["cmd"][0] = sys.executable
    main()
