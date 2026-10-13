#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONHASHSEED="0"
export PYTHONDONTWRITEBYTECODE="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${FRESH_ACCURACY_CEILING_PYTHON:-python}"

CONFIG_PATH="${REPO_ROOT}/configs/fresh_seed_accuracy_ceiling_confirmation_v1.yaml"
SOURCE_PATH="${REPO_ROOT}/analyse/fresh_seed_accuracy_ceiling_confirmation.py"
TEST_PATH="${REPO_ROOT}/tests/test_fresh_seed_accuracy_ceiling_confirmation.py"

RESULT_DIR="${REPO_ROOT}/results/posthoc/fresh-seed-accuracy-ceiling-confirmation-global4-v1"
ARCHIVE_PATH="${RESULT_DIR}.tar.gz"
ARCHIVE_SHA_PATH="${ARCHIVE_PATH}.sha256"

MODE="formal"
PREFLIGHT_LOG_DIR=""

usage() {
    printf 'Usage: bash %s [--formal|--dry-run]\n' "${BASH_SOURCE[0]}"
}

cleanup() {
    local status=$?
    if [[ -n "${PREFLIGHT_LOG_DIR}" && -d "${PREFLIGHT_LOG_DIR}" ]]; then
        rm -rf -- "${PREFLIGHT_LOG_DIR}"
    fi
    return "${status}"
}
trap cleanup EXIT

case "${1:-}" in
    "")
        ;;
    --formal)
        MODE="formal"
        shift
        ;;
    --dry-run)
        MODE="dry-run"
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
for required_path in "${CONFIG_PATH}" "${SOURCE_PATH}" "${TEST_PATH}"; do
    if [[ ! -f "${required_path}" ]]; then
        printf 'Required fresh-seed confirmation file is missing: %s\n' \
            "${required_path}" >&2
        exit 1
    fi
done
if [[ "${MODE}" == "formal" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'Formal fresh-seed confirmation requires the frozen CUDA evaluator environment.\n' >&2
    exit 1
fi

PREFLIGHT_LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dvae-fresh-accuracy-ceiling-preflight.XXXXXX")"

run_logged() {
    local log_path="$1"
    shift
    "$@" 2>&1 | tee "${log_path}"
}

printf '[Fresh-seed accuracy ceiling] Environment audit\n'
run_logged \
    "${PREFLIGHT_LOG_DIR}/environment_audit.log" \
    "${PYTHON_BIN}" - <<'PY'
import json
import platform
import sys

import numpy
import pytest
import scipy
import torch
import yaml

if torch.cuda.is_initialized():
    raise SystemExit("preflight environment audit initialized CUDA")

print(json.dumps({
    "cuda_initialized": False,
    "numpy": numpy.__version__,
    "platform": platform.platform(),
    "pytest": pytest.__version__,
    "python": sys.version.split()[0],
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "yaml": yaml.__version__,
}, sort_keys=True))
PY

printf '[Fresh-seed accuracy ceiling] Python syntax compilation\n'
run_logged \
    "${PREFLIGHT_LOG_DIR}/py_compile.log" \
    "${PYTHON_BIN}" -m py_compile "${SOURCE_PATH}" "${TEST_PATH}"

printf '[Fresh-seed accuracy ceiling] Focused tests\n'
run_logged \
    "${PREFLIGHT_LOG_DIR}/pytest_focused.log" \
    "${PYTHON_BIN}" -m pytest -q "${TEST_PATH}"

printf '[Fresh-seed accuracy ceiling] Synthetic smoke\n'
run_logged \
    "${PREFLIGHT_LOG_DIR}/synthetic_smoke.log" \
    "${PYTHON_BIN}" "${SOURCE_PATH}" \
    --config "${CONFIG_PATH}" \
    --synthetic-smoke

printf '[Fresh-seed accuracy ceiling] Dry-run protocol validation\n'
run_logged \
    "${PREFLIGHT_LOG_DIR}/dry_run.log" \
    "${PYTHON_BIN}" "${SOURCE_PATH}" \
    --config "${CONFIG_PATH}" \
    --dry-run

if [[ "${MODE}" == "dry-run" ]]; then
    printf '[Fresh-seed accuracy ceiling] Dry-run complete; no formal result path was created.\n'
    exit 0
fi

# Persistent outputs are created only after every CPU-safe preflight passes.
# Existing formal results, archives, and sidecars are immutable.
for target_path in "${RESULT_DIR}" "${ARCHIVE_PATH}" "${ARCHIVE_SHA_PATH}"; do
    if [[ -e "${target_path}" || -L "${target_path}" ]]; then
        printf 'Formal output already exists; refusing to overwrite: %s\n' \
            "${target_path}" >&2
        exit 1
    fi
done

CONFIG_SHA256="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
SOURCE_SHA256="$("${PYTHON_BIN}" "${SOURCE_PATH}" --source-identity)"
if [[ ! "${CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'Config identity command returned an invalid SHA-256.\n' >&2
    exit 1
fi
if [[ ! "${SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'Source identity command returned an invalid SHA-256: %s\n' \
        "${SOURCE_SHA256}" >&2
    exit 1
fi

printf '[Fresh-seed accuracy ceiling] Starting the frozen formal experiment\n'
"${PYTHON_BIN}" "${SOURCE_PATH}" \
    --config "${CONFIG_PATH}" \
    --output "${RESULT_DIR}" \
    --config-sha256 "${CONFIG_SHA256}" \
    --source-sha256 "${SOURCE_SHA256}" \
    --preflight-log-dir "${PREFLIGHT_LOG_DIR}" \
    --formal
