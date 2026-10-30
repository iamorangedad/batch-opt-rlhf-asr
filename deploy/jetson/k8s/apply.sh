#!/usr/bin/env bash
# Apply Jetson smoke manifests in order (single-node k3s).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=asr-jetson

kubectl apply -f "${DIR}/namespace.yaml"
kubectl apply -f "${DIR}/runtimeclass.yaml"
kubectl apply -f "${DIR}/configmap.yaml"
kubectl apply -f "${DIR}/pvc.yaml"

# Recreate Job if it already exists (Jobs are immutable).
kubectl -n "${NS}" delete job whisper-orpo-smoke --ignore-not-found
kubectl apply -f "${DIR}/job-smoke.yaml"

echo "Waiting for job/whisper-orpo-smoke ..."
kubectl -n "${NS}" wait --for=condition=complete job/whisper-orpo-smoke --timeout=3600s
kubectl -n "${NS}" logs job/whisper-orpo-smoke
