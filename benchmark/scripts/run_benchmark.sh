#!/usr/bin/env bash
# Run smoke or full benchmarks across frameworks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

MODE="${1:-smoke}"   # smoke | full
FRAMEWORKS="${2:-trl,openrlhf,verl}"

IFS=',' read -r -a FW_ARR <<< "${FRAMEWORKS}"

echo "[benchmark] Preparing demo data..."
python benchmark/scripts/generate_demo_data.py

run_one() {
  local fw="$1"
  echo "============================================================"
  echo "[benchmark] Framework=${fw} mode=${MODE}"
  echo "============================================================"
  case "${fw}" in
    trl)
      if [[ "${MODE}" == "smoke" ]]; then
        bash benchmark/frameworks/trl/run.sh smoke
      else
        bash benchmark/frameworks/trl/run.sh grpo
        bash benchmark/frameworks/trl/run.sh orpo
      fi
      ;;
    openrlhf)
      if [[ "${MODE}" == "smoke" ]]; then
        bash benchmark/frameworks/openrlhf/run.sh smoke
      else
        bash benchmark/frameworks/openrlhf/run.sh ppo
      fi
      ;;
    verl)
      if [[ "${MODE}" == "smoke" ]]; then
        bash benchmark/frameworks/verl/run.sh smoke
      else
        bash benchmark/frameworks/verl/run.sh grpo
      fi
      ;;
    native_orpo)
      if [[ "${MODE}" == "smoke" ]]; then
        bash benchmark/frameworks/native_orpo/run.sh smoke
      else
        bash benchmark/frameworks/native_orpo/run.sh train
      fi
      ;;
    *)
      echo "Unknown framework: ${fw}"
      exit 1
      ;;
  esac
}

for fw in "${FW_ARR[@]}"; do
  run_one "${fw}"
done

echo "[benchmark] Comparing results..."
python benchmark/scripts/compare_results.py \
  --outputs_dir outputs/benchmark \
  --out outputs/benchmark/comparison.md || true

echo "[benchmark] Done."
