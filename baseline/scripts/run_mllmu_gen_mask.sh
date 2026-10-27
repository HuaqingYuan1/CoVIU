#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"


CONFIG="${1:-${CONFIG:-config/mllmu_gen_mask.yaml}}"
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${ROOT}/${CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

echo ">>> config: ${CONFIG}"

python data_process/MLLMU_gen_mask.py --config "${CONFIG}" --weight-only "${@:2}"

