#!/usr/bin/env bash
# Build the Jetson image and run smoke | train | check | shell | metrics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="${ROOT}/deploy/jetson/docker/docker-compose.yml"
MODE="${1:-smoke}"
shift || true

export IMAGE="${IMAGE:-batch-opt-rlhf-asr:jetson}"
export JETSON_BASE_IMAGE="${JETSON_BASE_IMAGE:-nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3}"
export HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

mkdir -p "${ROOT}/outputs" "${HF_CACHE}"

cd "${ROOT}"

echo "[build_and_run] IMAGE=${IMAGE}"
echo "[build_and_run] JETSON_BASE_IMAGE=${JETSON_BASE_IMAGE}"
echo "[build_and_run] mode=${MODE}"

docker compose -f "${COMPOSE_FILE}" build

case "${MODE}" in
  build)
    echo "[build_and_run] image built: ${IMAGE}"
    ;;
  smoke|train|check|gpu|metrics|shell|bash)
    docker compose -f "${COMPOSE_FILE}" run --rm --entrypoint /usr/local/bin/jetson-entrypoint.sh \
      asr-jetson "${MODE}" "$@"
    ;;
  *)
    docker compose -f "${COMPOSE_FILE}" run --rm --entrypoint /usr/local/bin/jetson-entrypoint.sh \
      asr-jetson "${MODE}" "$@"
    ;;
esac
