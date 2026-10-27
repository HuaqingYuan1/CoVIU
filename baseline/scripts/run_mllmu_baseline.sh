#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {ga|gadiff|ga_diff|npo|manifold|viker} [config.yaml] [extra args...]" >&2
  exit 1
fi

METHOD="${1}"
shift

case "${METHOD}" in
  ga)
    SCRIPT="baselines/MLLMU_GA.py"
    DEFAULT_CONFIG="config/mllmu_ga.yaml"
    ;;
  gadiff|ga_diff)
    SCRIPT="baselines/MLLMU_GA_Diff.py"
    DEFAULT_CONFIG="config/mllmu_ga_diff.yaml"
    ;;
  manifold)
    SCRIPT="baselines/MLLMU_manifold.py"
    DEFAULT_CONFIG="config/mllmu_manifold.yaml"
    ;;
  npo)
    SCRIPT="baselines/MLLMU_NPO.py"
    DEFAULT_CONFIG="config/mllmu_npo.yaml"
    ;;
  viker)
    SCRIPT="baselines/MLLMU_ViKeR.py"
    DEFAULT_CONFIG="config/mllmu_viker.yaml"
    ;;
  *)
    echo "Unknown baseline: ${METHOD}" >&2
    exit 1
    ;;
esac

CONFIG="${DEFAULT_CONFIG}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG="${1}"
  shift
fi
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi


LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"
CURRENT_TIME="$(date "+%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${METHOD}_${CURRENT_TIME}.log}"

CONFIG_CUDA_VISIBLE_DEVICES="$(
python - "${CONFIG}" <<'PY'
import sys

try:
    import yaml
except ImportError:
    print("")
    raise SystemExit

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
value = cfg.get("cuda_visible_devices")
print("" if value is None else str(value))
PY
)"

if [[ -n "${CONFIG_CUDA_VISIBLE_DEVICES}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CONFIG_CUDA_VISIBLE_DEVICES}"
fi

echo ">>> method: ${METHOD}"
echo ">>> script: ${SCRIPT}"
echo ">>> config: ${CONFIG}"
echo ">>> log:    ${LOG_FILE}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python -u "${SCRIPT}" --config "${CONFIG}" "$@" 2>&1 | tee "${LOG_FILE}"
