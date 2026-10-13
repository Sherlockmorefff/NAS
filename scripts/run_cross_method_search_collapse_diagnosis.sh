#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED="0"
export PYTHONDONTWRITEBYTECODE="1"
export MPLBACKEND="Agg"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${CROSS_METHOD_COLLAPSE_PYTHON:-python}"
CONFIG="${REPO_ROOT}/configs/cross_method_search_collapse_diagnosis_v1.yaml"
ANALYSIS="${REPO_ROOT}/analyse/cross_method_search_collapse_diagnosis.py"
TESTS="${REPO_ROOT}/tests/test_cross_method_search_collapse_diagnosis.py"
OUTPUT="${REPO_ROOT}/results/posthoc/cross-method-search-collapse-diagnosis-global4-v1"

for path in "${CONFIG}" "${ANALYSIS}" "${TESTS}"; do
    if [[ ! -f "${path}" ]]; then
        printf 'Required diagnosis file is missing: %s\n' "${path}" >&2
        exit 1
    fi
done
for target in "${OUTPUT}" "${OUTPUT}.tar.gz" "${OUTPUT}.tar.gz.sha256"; do
    if [[ -e "${target}" || -L "${target}" ]]; then
        printf 'Formal diagnosis output already exists; refusing overwrite: %s\n' "${target}" >&2
        exit 1
    fi
done

printf '[collapse diagnosis] Python syntax compilation\n'
"${PYTHON_BIN}" -m py_compile "${ANALYSIS}" "${TESTS}"
printf '[collapse diagnosis] Focused tests\n'
"${PYTHON_BIN}" -m pytest -q "${TESTS}"
printf '[collapse diagnosis] Synthetic smoke\n'
"${PYTHON_BIN}" "${ANALYSIS}" --config "${CONFIG}" --synthetic-smoke
printf '[collapse diagnosis] Full zero-GNN formal diagnosis\n'
"${PYTHON_BIN}" "${ANALYSIS}" --config "${CONFIG}" --output "${OUTPUT}" --formal
