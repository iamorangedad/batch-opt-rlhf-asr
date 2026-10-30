# Jetson Orin test environment (Docker / Kubernetes)

Materials for running GPU smoke tests and short Whisper ORPO / RL
benchmarks on **NVIDIA Jetson Orin** edge boards (typically **Orin Nano**
or Orin NX). There is no product named “Jetson Nano Orin”; use **Orin Nano**
if you meant the Nano-class Orin board.

> Scope: **validation / smoke / small-batch fine-tuning** on-device.
> Full multi-framework RL (OpenRLHF + Ray, large Whisper) is better run on
> discrete-GPU servers. On Jetson prefer `whisper-tiny` + LoRA + the
> memory-safe config in [`configs/jetson_orin_nano.yaml`](configs/jetson_orin_nano.yaml).

## What you get

| Path | Purpose |
|---|---|
| [`scripts/prepare_host.sh`](scripts/prepare_host.sh) | Host checks: JetPack, NVIDIA Container Toolkit, GPU |
| [`scripts/check_gpu.sh`](scripts/check_gpu.sh) | Inside-container / host CUDA + torch sanity |
| [`scripts/build_and_run.sh`](scripts/build_and_run.sh) | Build image and run smoke / train / shell |
| [`docker/`](docker/) | Dockerfile (L4T PyTorch), compose, entrypoint |
| [`k8s/`](k8s/) | Namespace, RuntimeClass, PVC, smoke Job for k3s |
| [`configs/jetson_orin_nano.yaml`](configs/jetson_orin_nano.yaml) | Low-memory training overrides |

## Hardware assumptions

| Item | Recommendation |
|---|---|
| Board | Jetson Orin Nano (8 GB preferred) or Orin NX |
| Arch | `aarch64` (arm64) |
| OS / SDK | JetPack 5.1.x (L4T R35) or JetPack 6.x (L4T R36) |
| Disk | ≥ 32 GB free (images + Hugging Face cache) |
| Swap | 8–16 GB zram/swap strongly recommended on 8 GB boards |
| Power | Adequate PSU / MAXN or equivalent power mode |

## Host preparation (once per board)

```bash
# On the Jetson (Ubuntu / JetPack)
cd /path/to/batch-opt-rlhf-asr
bash deploy/jetson/scripts/prepare_host.sh
```

The script verifies:

1. `uname -m` is `aarch64`
2. `nvidia-smi` / `tegrastats` / L4T release are present
3. NVIDIA Container Toolkit is installed and Docker uses the NVIDIA runtime
4. Optional: k3s + RuntimeClass `nvidia` for Kubernetes jobs

Manual toolkit install (if needed):

```bash
# Official NVIDIA Container Toolkit for Jetson / L4T
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
# follow current NVIDIA docs for your JetPack version, then:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirm GPU inside a container:

```bash
sudo docker run --rm --runtime nvidia --network host \
  nvcr.io/nvidia/l4t-cuda:12.2.12-runtime \
  nvidia-smi || true
# On older JP5 images, tegrastats / device query may be more reliable than nvidia-smi.
bash deploy/jetson/scripts/check_gpu.sh
```

## Docker path (recommended first)

### 1. Pick a JetPack-matched base image

Edit `JETSON_BASE_IMAGE` in [`docker/Dockerfile`](docker/Dockerfile) or pass a build arg.

| JetPack | Example base |
|---|---|
| JP 5.1.x (R35.x) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| JP 6.0 / 6.1 (R36.x) | `nvcr.io/nvidia/l4t-jetpack:r36.2.0` then install torch via [jetson-containers](https://github.com/dusty-nv/jetson-containers) / NGC JP6 PyTorch tags |

**Critical:** the container’s L4T / CUDA userspace must match the host JetPack.
Do not use x86_64 CUDA images on Jetson.

### 2. Build and smoke

```bash
bash deploy/jetson/scripts/build_and_run.sh smoke
```

Equivalent compose:

```bash
cd deploy/jetson/docker
docker compose build
docker compose run --rm asr-jetson smoke
```

Smoke does:

1. GPU / torch check
2. Demo data generation (if missing)
3. Native ORPO or unit metrics smoke with Jetson overrides

### 3. Short training run

```bash
bash deploy/jetson/scripts/build_and_run.sh train
# or interactive shell
bash deploy/jetson/scripts/build_and_run.sh shell
```

Compose mounts:

- repo root → `/workspace`
- HF cache → `~/.cache/huggingface`
- outputs → `./outputs`

## Kubernetes path (k3s on Jetson)

Single-node k3s is enough for device jobs.

```bash
# On Jetson, after prepare_host.sh and image build/load
export IMAGE=batch-opt-rlhf-asr:jetson
# If you build on the board:
sudo k3s ctr images import <(docker save "${IMAGE}")

kubectl apply -f deploy/jetson/k8s/namespace.yaml
kubectl apply -f deploy/jetson/k8s/runtimeclass.yaml
kubectl apply -f deploy/jetson/k8s/configmap.yaml
kubectl apply -f deploy/jetson/k8s/pvc.yaml
kubectl apply -f deploy/jetson/k8s/job-smoke.yaml

kubectl -n asr-jetson logs -f job/whisper-orpo-smoke
kubectl -n asr-jetson wait --for=condition=complete job/whisper-orpo-smoke --timeout=3600s
```

The Job requests one NVIDIA GPU via `nvidia.com/gpu: 1` and
`runtimeClassName: nvidia`. Adjust the image pull policy if you only have a
local import (see comments in `job-smoke.yaml`).

### Multi-board notes

- Prefer one Job per Orin; do not schedule Ray / OpenRLHF multi-node on Nano-class memory.
- For a small lab cluster, label nodes (`jetson-orin-nano`, `jetson-orin-nx`) and pin Jobs with `nodeSelector`.
- Keep PVC on local path / NVMe; Whisper checkpoints + HF cache dominate disk.

## Memory & model guidance (Orin Nano)

| Setting | Suggested |
|---|---|
| Model | `openai/whisper-tiny` (default) |
| Batch | `per_device_train_batch_size: 1` |
| Accumulation | `4–8` |
| Precision | `fp16: true` (bf16 also OK on Orin Ampere; FP16 is safer under memory pressure) |
| Checkpointing | `gradient_checkpointing: true` |
| Framework | Native ORPO or TRL ORPO first; skip OpenRLHF/Ray smoke on 8 GB |

Use [`configs/jetson_orin_nano.yaml`](configs/jetson_orin_nano.yaml) as the
benchmark overlay (merged by entrypoint / Job command).

## Suggested validation matrix

| Test | Command | Pass criteria |
|---|---|---|
| Host GPU | `prepare_host.sh` | toolkit + device visible |
| Container GPU | `check_gpu.sh` inside image | `torch.cuda.is_available()` |
| Metrics unit | `python benchmark/tests/test_metrics_rewards.py` | exit 0 |
| Demo pipeline | `build_and_run.sh smoke` | `summary.json` written |
| Tiny train | `build_and_run.sh train` | checkpoint under `outputs/` |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown runtime specified nvidia` | Toolkit not configured | `nvidia-ctk runtime configure --runtime=docker` + restart |
| CUDA version mismatch | Image JetPack ≠ host | Match `l4t-pytorch` / JetPack tag |
| OOM / killed | 4–8 GB unified memory | tiny model, batch 1, more swap, close GUI |
| `nvidia.com/gpu` unschedulable | Device plugin / runtime missing | install NVIDIA k8s device plugin; set RuntimeClass |
| pip installs x86 torch | Wrong platform wheel | Keep torch from L4T image; install other deps only |
| Slow first run | HF model download | Pre-seed cache volume / ConfigMap mount |

## Relation to the rest of the repo

- Core trainer: [`finetune.py`](../../finetune.py) + [`README.md`](../../README.md)
- Optional RL comparison suite: [`benchmark/README.md`](../../benchmark/README.md)
- This tree only adds **edge GPU packaging**; it does not change training algorithms.
