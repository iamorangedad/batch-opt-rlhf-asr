# Jetson Orin test environment

Run the Whisper RL benchmark on **NVIDIA Jetson Orin** (AGX / NX / Nano) so
GPU VRAM and WER numbers are real — Cloud Agent VMs are x86 CPU-only.

## Why Jetson

| Host | Arch | GPU | Role |
|---|---|---|---|
| Cursor Cloud Agent | x86_64 | none | pipeline smoke, CPU RSS compare |
| Jetson Orin | aarch64 | iGPU (Ampere) | GPU VRAM + quality compare |

Open PR work already records `gpu_peak_*` fields; they stay `null` until a
CUDA device is present. Orin fills those fields.

## Prerequisites

1. JetPack **6.1+** (recommended **6.2**, CUDA 12.6) installed on the Orin.
2. Disk: ≥ **32 GiB** free (models + uv cache + outputs).
3. Swap: for Orin Nano / 8 GB modules, add **8–16 GiB** swap before training.
4. Network: HuggingFace + PyPI (or a mirror) reachable.

Check:

```bash
uname -m                    # expect aarch64
cat /etc/nv_tegra_release   # JetPack / L4T stamp
nvidia-smi || tegrastats --interval 1000
```

## Bootstrap (on the Jetson)

```bash
git clone https://github.com/iamorangedad/batch-opt-rlhf-asr.git
cd batch-opt-rlhf-asr
git checkout cursor/develop-7e36   # or main once GPU-tracking PRs land

bash scripts/setup_jetson_orin.sh
# optional: JP6_TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
```

What the script does:

1. Confirms `aarch64` + Jetson markers.
2. Installs system packages (`libopenblas-dev`, audio libs, `uv`).
3. Creates `.venv` with the pinned Python when possible.
4. Installs **Jetson CUDA PyTorch** from the Jetson AI Lab index (not generic PyPI).
5. Runs `uv sync --extra all --group dev --no-install-package torch` so uv does
   not overwrite the Jetson wheel with an x86/CPU build.
6. Prints a CUDA smoke check.

Verify:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vram_mb", round(torch.cuda.get_device_properties(0).total_memory/1024**2,1) if torch.cuda.is_available() else None)
PY
```

## Memory-conservative configs

Jetson unified memory is shared with the OS. Use the `jetson_*` YAML overlays
(batch size 1, grad accum 1, fp16 when stable, short max_steps):

```text
benchmark/configs/frameworks/jetson_trl_grpo.yaml
benchmark/configs/frameworks/jetson_trl_orpo.yaml
benchmark/configs/frameworks/jetson_openrlhf.yaml
benchmark/configs/frameworks/jetson_verl.yaml
```

Suggested starting points by module memory:

| Module | Unified mem | Start with |
|---|---|---|
| Orin Nano 8 GB | ~8 GiB | `whisper-tiny`, LoRA, `max_steps≤5`, GRPO `num_generations=2` |
| Orin NX 16 GB | ~16 GiB | same; try `per_device_train_batch_size=2` |
| AGX Orin 32/64 GB | 32–64 GiB | can raise steps / generations carefully |

Always prefer **fp16** over bf16 on Orin iGPU unless you have verified bf16.

## GPU memory comparison

Isolated subprocess per framework (same protocol as CPU RSS compare):

```bash
uv run python benchmark/scripts/run_gpu_memory_compare.py
# subset:
uv run python benchmark/scripts/run_gpu_memory_compare.py --jobs trl_orpo,native_orpo
```

Outputs:

- `outputs/benchmark/gpu_memory_compare/gpu_memory_report.md`
- `benchmark/docs/gpu_memory_report.md` (copied for the repo)
- per-job `summary.json` with `gpu_peak_allocated_mb` / `gpu_peak_reserved_mb`

## Smoke + full compare on Orin

```bash
# pipeline smoke (demo sine tones)
bash benchmark/scripts/run_benchmark.sh smoke trl

# tiny LibriSpeech quality / VRAM compare (Jetson configs)
uv run python benchmark/frameworks/trl/train_grpo.py \
  --config benchmark/configs/frameworks/jetson_trl_grpo.yaml

uv run python benchmark/scripts/compare_results.py --outputs_dir outputs/benchmark
```

## Cursor self-hosted worker (optional)

Cloud Agents cannot boot Jetson hardware. To let a Cursor agent run **on** the
Orin:

1. On the Jetson, install and start the Cursor cloud-agent / private worker for
   repo `iamorangedad/batch-opt-rlhf-asr`.
2. Confirm the worker appears connected for this repository in Cursor Cloud.
3. Start a Cloud Agent targeted at that private worker; it will execute on the
   Orin and see CUDA.

Until that worker is online for this repo, use the scripts above over SSH.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` | Wrong wheel (PyPI CPU). Re-run `setup_jetson_orin.sh`; do not `uv sync` without `--no-install-package torch`. |
| OOM / process killed | Lower `max_steps`, `num_generations`, enable swap; close desktop GUI. |
| `Illegal instruction` | JetPack / torch mismatch; use the index matching your JetPack (jp6/cu126 for 6.2). |
| Audio import errors | `sudo apt-get install -y libsndfile1 ffmpeg` |
| uv wants to replace torch | Always sync with `--no-install-package torch` on Jetson. |

## Related docs

- [uv environment](uv_env.md)
- [CPU memory report](cpu_memory_report.md) (cloud baseline)
- [Framework comparison](framework_comparison.md)
