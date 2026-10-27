#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/token_counterfactual_augmented_all_lora.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi
python -u validate_counterfactual_gain.py --config "${CONFIG_PATH}" "${@:2}"
