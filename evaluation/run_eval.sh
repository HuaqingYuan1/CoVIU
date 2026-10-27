#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -u "${EVAL_ROOT}/run_eval.py" "$@"
