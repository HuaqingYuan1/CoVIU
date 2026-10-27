#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(ga gadiff npo manifold viker)
fi

for method in "${METHODS[@]}"; do
  echo "===== Running MLLMU baseline: ${method} ====="
  bash "${ROOT}/scripts/run_mllmu_baseline.sh" "${method}"
done

echo "All requested MLLMU baselines finished: ${METHODS[*]}"
