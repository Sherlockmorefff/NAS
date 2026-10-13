#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED="0"
export PYTHONDONTWRITEBYTECODE="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${SOFT_METRIC_PYTHON:-python}"

CONFIG_PATH="${REPO_ROOT}/configs/joint_configuration_soft_metric_gate_v2.yaml"
SOURCE_PATH="${REPO_ROOT}/analyse/joint_configuration_soft_metric_gate.py"
TEST_PATH="${REPO_ROOT}/tests/test_joint_configuration_soft_metric_gate.py"
REUSED_JOINT_V1="${REPO_ROOT}/analyse/joint_configuration_metric_gate.py"
REUSED_PERFORMANCE_V2="${REPO_ROOT}/analyse/performance_metric_quick_gate.py"
REUSED_PAIR_AUDIT_V1="${REPO_ROOT}/analyse/joint_pair_power_projection_audit.py"
RESULT_DIR="${REPO_ROOT}/results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
ARCHIVE_PATH="${RESULT_DIR}.tar.gz"
ARCHIVE_SHA_PATH="${ARCHIVE_PATH}.sha256"
RESULT_BASENAME="$(basename -- "${RESULT_DIR}")"
LOCK_PATH="${TMPDIR:-/tmp}/dvae-joint-configuration-soft-metric-gate-global4-v2.lock"

MODE="formal"
CURRENT_STAGE="runner_initialization"
RUN_ACTIVE=0
RUN_SUCCEEDED=0
FINALIZING=0
CONFIG_SHA256=""
SOURCE_SHA256=""

usage() {
    printf 'Usage: bash %s [--dry-run]\n' "${BASH_SOURCE[0]}"
}

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command is unavailable: %s\n' "${command_name}" >&2
        return 1
    fi
}

path_has_entries() {
    local directory="$1"
    [[ -d "${directory}" ]] &&
        [[ -n "$(find "${directory}" -mindepth 1 -print -quit 2>/dev/null)" ]]
}

assert_targets_available() {
    if [[ -e "${RESULT_DIR}" || -L "${RESULT_DIR}" ]]; then
        if [[ ! -d "${RESULT_DIR}" ]] || [[ -L "${RESULT_DIR}" ]]; then
            printf 'Formal result path exists and is not a reusable empty directory: %s\n' \
                "${RESULT_DIR}" >&2
            return 1
        fi
        if path_has_entries "${RESULT_DIR}"; then
            printf 'Formal result directory is non-empty; refusing to overwrite: %s\n' \
                "${RESULT_DIR}" >&2
            return 1
        fi
    fi
    if [[ -e "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" ]]; then
        printf 'Formal archive already exists; refusing to overwrite: %s\n' \
            "${ARCHIVE_PATH}" >&2
        return 1
    fi
    if [[ -e "${ARCHIVE_SHA_PATH}" || -L "${ARCHIVE_SHA_PATH}" ]]; then
        printf 'Formal archive checksum already exists; refusing to overwrite: %s\n' \
            "${ARCHIVE_SHA_PATH}" >&2
        return 1
    fi
}

assert_no_matching_formal_process() {
    if pgrep -f '[j]oint_configuration_soft_metric_gate\.py.*--formal' \
        >/dev/null 2>&1; then
        printf 'Another formal soft joint-metric gate process is already running.\n' >&2
        return 1
    fi
}

common_python_args() {
    printf '%s\0' \
        --config "${CONFIG_PATH}" \
        --output "${RESULT_DIR}" \
        --config-sha256 "${CONFIG_SHA256}" \
        --source-sha256 "${SOURCE_SHA256}"
}

run_gate_mode() {
    local mode_flag="$1"
    shift
    local -a shared_args=()
    while IFS= read -r -d '' argument; do
        shared_args+=("${argument}")
    done < <(common_python_args)
    "${PYTHON_BIN}" "${SOURCE_PATH}" "${shared_args[@]}" "${mode_flag}" "$@"
}

verify_source_snapshot() {
    local relative_path
    local -a required_paths=(
        'configs/joint_configuration_soft_metric_gate_v2.yaml'
        'analyse/joint_configuration_soft_metric_gate.py'
        'scripts/run_joint_configuration_soft_metric_gate.sh'
        'tests/test_joint_configuration_soft_metric_gate.py'
        'analyse/joint_configuration_metric_gate.py'
        'analyse/performance_metric_quick_gate.py'
        'analyse/joint_pair_power_projection_audit.py'
        'imported_local_modules.json'
        'sha256_manifest.json'
    )
    for relative_path in "${required_paths[@]}"; do
        if [[ ! -f "${RESULT_DIR}/source_snapshot/${relative_path}" ]]; then
            printf 'Required source snapshot file is missing: %s\n' \
                "${relative_path}" >&2
            return 1
        fi
    done
}

verify_common_outputs() {
    local relative_path
    local -a required_paths=(
        state.json frozen_protocol.json input_integrity.json data_inventory.json
        split_manifest.json pair_weight_summary.json metric_training_audit.json
        geometry_metrics.json knn_locality.json metrics_per_fold.csv
        oof_predictions.csv best_of_4_batches.json best_of_4_results.json
        branch_ablation.json shuffle_controls.json sensitivity_30seed.json
        decision.json final_report.md environment.json git_status.txt git_diff.patch
        manifest.json source_snapshot/imported_local_modules.json
        source_snapshot/sha256_manifest.json logs/py_compile.log
        logs/pytest_focused.log logs/synthetic_smoke.log logs/run.log
    )
    for relative_path in "${required_paths[@]}"; do
        if [[ ! -f "${RESULT_DIR}/${relative_path}" ]]; then
            printf 'Required bundle output is missing: %s\n' "${relative_path}" >&2
            return 1
        fi
    done
}

verify_success_outputs() {
    verify_common_outputs
    verify_source_snapshot
    [[ -f "${RESULT_DIR}/COMPLETE" ]]
    [[ ! -e "${RESULT_DIR}/FAILED" ]]
    [[ ! -e "${RESULT_DIR}/error.json" ]]
}

verify_failure_outputs() {
    verify_common_outputs
    verify_source_snapshot
    [[ -f "${RESULT_DIR}/FAILED" ]]
    [[ -f "${RESULT_DIR}/error.json" ]]
    [[ ! -e "${RESULT_DIR}/COMPLETE" ]]
}

refresh_manifest_twice() {
    run_gate_mode --refresh-manifest >>"${RESULT_DIR}/logs/run.log" 2>&1
    run_gate_mode --refresh-manifest >/dev/null 2>&1
}

assert_package_payload_safe() {
    local forbidden
    forbidden="$(find "${RESULT_DIR}" -type f \( \
        -name '*.pyc' -o \
        -name '*.pyo' -o \
        -name '*.pt' -o \
        -name '*.pth' -o \
        -name '*.ckpt' -o \
        -name '*.onnx' -o \
        -name '*.joblib' \
    \) -print -quit 2>/dev/null)"
    if [[ -n "${forbidden}" ]]; then
        printf 'Forbidden cache/model payload detected: %s\n' "${forbidden}" >&2
        return 1
    fi
    if [[ -n "$(find "${RESULT_DIR}" -type d \( \
        -name '__pycache__' -o -name '.pytest_cache' \
    \) -print -quit 2>/dev/null)" ]]; then
        printf 'Cache directory detected in formal result payload.\n' >&2
        return 1
    fi
    if [[ -n "$(find "${RESULT_DIR}" -type d \( \
        -name '.venv' -o -name 'venv' -o -name 'site-packages' \
    \) -print -quit 2>/dev/null)" ]]; then
        printf 'Python environment directory detected in formal result payload.\n' >&2
        return 1
    fi
}

package_bundle() {
    local archive_tmp sha_tmp archive_hash
    archive_tmp="${ARCHIVE_PATH}.tmp.$$"
    sha_tmp="${ARCHIVE_SHA_PATH}.tmp.$$"

    if [[ -e "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" || \
          -e "${ARCHIVE_SHA_PATH}" || -L "${ARCHIVE_SHA_PATH}" ]]; then
        printf 'Refusing to overwrite an existing formal archive target.\n' >&2
        return 1
    fi
    if [[ -e "${archive_tmp}" || -L "${archive_tmp}" || \
          -e "${sha_tmp}" || -L "${sha_tmp}" ]]; then
        printf 'Temporary package target already exists; refusing to overwrite it.\n' >&2
        return 1
    fi

    assert_package_payload_safe
    if ! tar \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='*.onnx' \
        --exclude='*.joblib' \
        -C "$(dirname -- "${RESULT_DIR}")" \
        -czf "${archive_tmp}" \
        "${RESULT_BASENAME}"; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        return 1
    fi

    if ! tar -tzf "${archive_tmp}" \
        "${RESULT_BASENAME}/source_snapshot/sha256_manifest.json" \
        >/dev/null; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        return 1
    fi
    if ! tar -tzf "${archive_tmp}" \
        "${RESULT_BASENAME}/source_snapshot/imported_local_modules.json" \
        >/dev/null; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        return 1
    fi

    archive_hash="$(sha256sum "${archive_tmp}" | awk '{print $1}')"
    printf '%s  %s\n' "${archive_hash}" "$(basename -- "${ARCHIVE_PATH}")" \
        >"${sha_tmp}"

    if [[ -e "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" || \
          -e "${ARCHIVE_SHA_PATH}" || -L "${ARCHIVE_SHA_PATH}" ]]; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        printf 'Archive target appeared during packaging; refusing to overwrite it.\n' >&2
        return 1
    fi

    mv -- "${archive_tmp}" "${ARCHIVE_PATH}"
    if ! mv -- "${sha_tmp}" "${ARCHIVE_SHA_PATH}"; then
        rm -f -- "${ARCHIVE_PATH}" "${sha_tmp}"
        return 1
    fi
}

write_failure_bundle() {
    local exit_code="$1"
    local failure_message
    failure_message="runner stage ${CURRENT_STAGE} failed with exit code ${exit_code}"

    mkdir -p -- "${RESULT_DIR}/logs"
    : >>"${RESULT_DIR}/logs/py_compile.log"
    : >>"${RESULT_DIR}/logs/pytest_focused.log"
    : >>"${RESULT_DIR}/logs/synthetic_smoke.log"
    : >>"${RESULT_DIR}/logs/run.log"
    rm -f -- "${RESULT_DIR}/COMPLETE"
    if ! verify_failure_outputs >/dev/null 2>&1; then
        run_gate_mode --write-failure-bundle "${failure_message}" \
            >>"${RESULT_DIR}/logs/run.log" 2>&1
    fi
    verify_failure_outputs
    refresh_manifest_twice
    verify_failure_outputs
}

on_exit() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    set +u
    set +o pipefail

    if [[ "${RUN_ACTIVE}" -eq 1 && "${RUN_SUCCEEDED}" -eq 0 && \
          "${FINALIZING}" -eq 0 ]]; then
        FINALIZING=1
        if write_failure_bundle "${exit_code}"; then
            package_bundle || \
                printf 'Could not package the completed failure bundle.\n' >&2
        else
            printf 'The program could not generate a complete failure bundle.\n' >&2
        fi
    fi
    exit "${exit_code}"
}

parse_arguments() {
    if [[ $# -gt 1 ]]; then
        usage >&2
        return 2
    fi
    if [[ $# -eq 1 ]]; then
        if [[ "$1" != '--dry-run' ]]; then
            usage >&2
            return 2
        fi
        MODE="dry-run"
    fi
}

environment_preflight() {
    local conda_candidate
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
       ! "${PYTHON_BIN}" -c \
           'import numpy, scipy, yaml, pytest, sklearn, torch, gpytorch' >/dev/null 2>&1; then
        conda_candidate="$(dirname -- "${REPO_ROOT}")/.conda/envs/nas/bin/python"
        if [[ -x "${conda_candidate}" ]] && \
           "${conda_candidate}" -c \
               'import numpy, scipy, yaml, pytest, sklearn, torch, gpytorch' >/dev/null 2>&1; then
            PYTHON_BIN="${conda_candidate}"
        else
            printf '%s\n' \
                'Warning: the selected Python is missing one or more formal dependencies; the preflight phase will fail closed and bundle the error.' \
                >&2
        fi
    fi

    require_command "${PYTHON_BIN}"
    require_command git
    require_command find
    require_command pgrep
    require_command flock
    require_command tar
    require_command sha256sum
    require_command awk

    [[ -f "${CONFIG_PATH}" ]]
    [[ -f "${SOURCE_PATH}" ]]
    [[ -f "${TEST_PATH}" ]]
    [[ -f "${REUSED_JOINT_V1}" ]]
    [[ -f "${REUSED_PERFORMANCE_V2}" ]]
    [[ -f "${REUSED_PAIR_AUDIT_V1}" ]]
    [[ -f "${REPO_ROOT}/AGENTS.md" ]]

    CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
    SOURCE_SHA256="$("${PYTHON_BIN}" - \
        "${SOURCE_PATH}" "${BASH_SOURCE[0]}" "${TEST_PATH}" \
        "${REUSED_JOINT_V1}" "${REUSED_PERFORMANCE_V2}" \
        "${REUSED_PAIR_AUDIT_V1}" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

digest = sha256()
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"
}

main() {
    parse_arguments "$@"
    cd -- "${REPO_ROOT}"

    require_command find
    assert_targets_available
    environment_preflight
    assert_no_matching_formal_process

    exec 9>"${LOCK_PATH}"
    if ! flock -n 9; then
        printf 'Another soft joint-metric runner holds the formal lock.\n' >&2
        return 1
    fi
    assert_targets_available

    if [[ "${MODE}" == 'dry-run' ]]; then
        CURRENT_STAGE="dry_run"
        run_gate_mode --dry-run
        if [[ -e "${RESULT_DIR}" || -L "${RESULT_DIR}" ]]; then
            printf 'Dry-run unexpectedly created the formal result directory.\n' >&2
            return 1
        fi
        return 0
    fi

    mkdir -p -- "${RESULT_DIR}/logs"
    : >"${RESULT_DIR}/logs/py_compile.log"
    : >"${RESULT_DIR}/logs/pytest_focused.log"
    : >"${RESULT_DIR}/logs/synthetic_smoke.log"
    : >"${RESULT_DIR}/logs/run.log"
    RUN_ACTIVE=1

    CURRENT_STAGE="py_compile"
    "${PYTHON_BIN}" -m py_compile "${SOURCE_PATH}" "${TEST_PATH}" \
        >"${RESULT_DIR}/logs/py_compile.log" 2>&1

    CURRENT_STAGE="focused_pytest"
    "${PYTHON_BIN}" -m pytest -q "${TEST_PATH}" \
        >"${RESULT_DIR}/logs/pytest_focused.log" 2>&1

    CURRENT_STAGE="synthetic_smoke"
    run_gate_mode --synthetic-smoke \
        >"${RESULT_DIR}/logs/synthetic_smoke.log" 2>&1

    CURRENT_STAGE="formal_gate"
    run_gate_mode --formal --preflight-ready \
        >"${RESULT_DIR}/logs/run.log" 2>&1

    CURRENT_STAGE="formal_output_verification"
    rm -f -- "${RESULT_DIR}/FAILED" "${RESULT_DIR}/error.json"
    touch "${RESULT_DIR}/COMPLETE"
    verify_success_outputs

    CURRENT_STAGE="manifest_refresh"
    refresh_manifest_twice
    verify_success_outputs

    CURRENT_STAGE="archive_packaging"
    package_bundle

    RUN_SUCCEEDED=1
    printf 'Soft joint-configuration metric gate completed.\n'
    printf 'Result directory: %s\n' "${RESULT_DIR}"
    printf 'Archive: %s\n' "${ARCHIVE_PATH}"
    printf 'SHA-256: %s\n' "${ARCHIVE_SHA_PATH}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

main "$@"
