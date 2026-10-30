# Jetson Orin (aarch64) image for Whisper ORPO / benchmark smoke tests.
#
# IMPORTANT: Base L4T / JetPack must match the host board firmware.
# Override at build time, e.g.:
#   docker build --build-arg JETSON_BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 ...
#
# Do not use x86_64 CUDA base images on Jetson.

ARG JETSON_BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
FROM ${JETSON_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    WORKDIR=/workspace

WORKDIR /workspace

# Audio + build helpers. L4T images already ship CUDA + a Jetson-built torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libsndfile1-dev \
      git \
      curl \
      ca-certificates \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install project deps WITHOUT replacing the Jetson torch / torchvision /
# torchaudio that ship in the L4T image. Generic PyPI CUDA wheels are wrong
# for Tegra.
COPY pyproject.toml README.md ./
COPY src ./src
COPY finetune.py ./
COPY benchmark ./benchmark
COPY scripts ./scripts
COPY deploy/jetson ./deploy/jetson

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install \
      "transformers>=4.40.0" \
      "datasets>=2.18.0" \
      "accelerate>=0.28.0" \
      "peft>=0.10.0" \
      "librosa>=0.10.0" \
      "soundfile>=0.12.0" \
      "pyyaml>=6.0" \
      "tqdm>=4.66.0" \
      "numpy>=1.24.0" \
      "jiwer>=3.0.0" \
      "zhconv>=1.4.0" \
      "trl>=0.9.0" \
 && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

COPY deploy/jetson/docker/entrypoint.sh /usr/local/bin/jetson-entrypoint.sh
RUN chmod +x /usr/local/bin/jetson-entrypoint.sh

RUN mkdir -p /data/hf-cache /data/outputs

ENTRYPOINT ["/usr/local/bin/jetson-entrypoint.sh"]
CMD ["smoke"]
