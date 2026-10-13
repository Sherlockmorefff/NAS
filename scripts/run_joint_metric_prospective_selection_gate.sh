#!/usr/bin/env bash
set -Eeuo pipefail

# Scorer replay and candidate generation are explicitly CPU-bound by the Python
# protocol.  CUDA visibility is intentionally preserved for the subsequently
# authorized high-regime GNN evaluation stage.
export PROSPECTIVE_SCORER_DEVICE="cpu"
export PYTHONHASHSEED="0"
export PYTHONDONTWRITEBYTECODE="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${JOINT_METRIC_PROSPECTIVE_PYTHON:-python}"

CONFIG_PATH="${REPO_ROOT}/configs/joint_metric_prospective_selection_gate_v1.yaml"
SOURCE_PATH="${REPO_ROOT}/analyse/joint_metric_prospective_selection_gate.py"
TEST_PATH="${REPO_ROOT}/tests/test_joint_metric_prospective_selection_gate.py"

SOFT_RESULT_ROOT="${REPO_ROOT}/results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
SOFT_SNAPSHOT_ROOT="${SOFT_RESULT_ROOT}/source_snapshot"
FROZEN_SOFT_CONFIG="${SOFT_SNAPSHOT_ROOT}/configs/joint_configuration_soft_metric_gate_v2.yaml"
FROZEN_SOFT_SOURCE="${SOFT_SNAPSHOT_ROOT}/analyse/joint_configuration_soft_metric_gate.py"
FROZEN_JOINT_SOURCE="${SOFT_SNAPSHOT_ROOT}/analyse/joint_configuration_metric_gate.py"
FROZEN_PERFORMANCE_SOURCE="${SOFT_SNAPSHOT_ROOT}/analyse/performance_metric_quick_gate.py"
FROZEN_PAIR_AUDIT_SOURCE="${SOFT_SNAPSHOT_ROOT}/analyse/joint_pair_power_projection_audit.py"
FROZEN_SOFT_MANIFEST="${SOFT_SNAPSHOT_ROOT}/sha256_manifest.json"

DIAGNOSIS_RESULT_ROOT="${REPO_ROOT}/results/posthoc/joint-configuration-cross-factor-locality-diagnosis-global4-v1"

HIGH_REGIME_ROOT="${REPO_ROOT}/results/search/high-regime-prior-model-blind-validation-global4-v1"
HIGH_SNAPSHOT_ROOT="${HIGH_REGIME_ROOT}/source_snapshot"
FROZEN_HIGH_PLAN_SOURCE="${HIGH_SNAPSHOT_ROOT}/analyse/high_regime_prior_model_blind_validation.py"
FROZEN_HIGH_EVALUATOR_SOURCE="${HIGH_SNAPSHOT_ROOT}/analyse/run_high_regime_prior_model_blind_validation_labels.py"
FROZEN_HIGH_ANALYSIS_SOURCE="${HIGH_SNAPSHOT_ROOT}/analyse/analyse_high_regime_prior_model_blind_validation.py"
FROZEN_EVAL_UTILS_SOURCE="${HIGH_SNAPSHOT_ROOT}/eval_utils.py"
FROZEN_ARCH_DIAGNOSTICS_SOURCE="${HIGH_SNAPSHOT_ROOT}/architecture_diagnostics.py"
FROZEN_HIGH_SNAPSHOT_AUDIT="${HIGH_REGIME_ROOT}/source_snapshot_audit.json"

RESULT_DIR="${REPO_ROOT}/results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1"
ARCHIVE_PATH="${RESULT_DIR}.tar.gz"
ARCHIVE_SHA_PATH="${ARCHIVE_PATH}.sha256"
RESULT_BASENAME="$(basename -- "${RESULT_DIR}")"
LOCK_PATH="${TMPDIR:-/tmp}/dvae-joint-metric-prospective-selection-gate-global4-v1-rerun1.lock"

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

announce_stage() {
    printf '[prospective selection gate] %s\n' "$1"
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
        if path_has_entries "${RESULT_DIR}"; then
            printf 'Formal result directory is non-empty; refusing to overwrite: %s\n' \
                "${RESULT_DIR}" >&2
        else
            printf 'Formal result path already exists; refusing to reuse it: %s\n' \
                "${RESULT_DIR}" >&2
        fi
        return 1
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
    if pgrep -f \
        '[j]oint_metric_prospective_selection_gate\.py.*--formal' \
        >/dev/null 2>&1; then
        printf 'Another formal prospective selection gate is already running.\n' >&2
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
        'configs/joint_metric_prospective_selection_gate_v1.yaml'
        'analyse/joint_metric_prospective_selection_gate.py'
        'scripts/run_joint_metric_prospective_selection_gate.sh'
        'tests/test_joint_metric_prospective_selection_gate.py'
        'configs/joint_configuration_soft_metric_gate_v2.yaml'
        'analyse/joint_configuration_soft_metric_gate.py'
        'analyse/joint_configuration_metric_gate.py'
        'analyse/performance_metric_quick_gate.py'
        'analyse/joint_pair_power_projection_audit.py'
        'analyse/high_regime_prior_model_blind_validation.py'
        'analyse/run_high_regime_prior_model_blind_validation_labels.py'
        'analyse/analyse_high_regime_prior_model_blind_validation.py'
        'eval_utils.py'
        'architecture_diagnostics.py'
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
        state.json
        frozen_protocol.json
        input_integrity.json
        engineering_fix.json
        scorer_replay_identity.json
        pre_eval/candidate_manifest.json
        pre_eval/predictions.csv
        pre_eval/selections.json
        pre_eval/scorer_identity.json
        pre_eval/freeze_manifest.json
        pre_eval/pre_eval_root_sha256.txt
        evaluation/task_manifest.json
        evaluation/raw_seed_results.csv
        evaluation/config_3seed_statistics.csv
        evaluation/training_audit.json
        selection_results_per_pool.csv
        prospective_metrics.json
        paired_pool_comparisons.json
        bootstrap_results.json
        decision.json
        final_report.md
        environment.json
        git_status_before.txt
        git_status_after.txt
        git_diff.patch
        source_snapshot/imported_local_modules.json
        source_snapshot/sha256_manifest.json
        manifest.json
        logs/py_compile.log
        logs/pytest_focused.log
        logs/synthetic_smoke.log
        logs/scorer_replay.log
        logs/candidate_generation.log
        logs/evaluation.log
        logs/analysis.log
        logs/runner.log
    )
    for relative_path in "${required_paths[@]}"; do
        if [[ ! -f "${RESULT_DIR}/${relative_path}" ]]; then
            printf 'Required prospective bundle output is missing: %s\n' \
                "${relative_path}" >&2
            return 1
        fi
    done
}

verify_pre_eval_marker_consistency() {
    "${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
classification = str(decision.get("classification", ""))
marker = root / "PRE_EVAL_SELECTION_FROZEN"
requires_freeze = {
    "PROSPECTIVE_LOW_DISAGREEMENT",
    "PROSPECTIVE_EVALUATION_INCOMPLETE",
    "METRIC_SELECTION_PROSPECTIVE_REJECTED",
    "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY",
    "METRIC_GP_PROSPECTIVE_SIGNAL",
}
forbids_freeze = {
    "PROSPECTIVE_INPUT_INVALID",
    "SCORER_REPLAY_IDENTITY_MISMATCH",
    "CANDIDATE_GENERATION_FAILED",
}
if classification in requires_freeze and not marker.is_file():
    raise SystemExit(f"missing pre-evaluation freeze marker for {classification}")
if classification in forbids_freeze and marker.exists():
    raise SystemExit(f"unexpected pre-evaluation freeze marker for {classification}")
if classification not in requires_freeze | forbids_freeze:
    raise SystemExit(f"unexpected successful classification: {classification}")
PY
}

verify_success_outputs() {
    verify_common_outputs
    verify_source_snapshot
    verify_pre_eval_marker_consistency
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
    run_gate_mode --refresh-manifest >>"${RESULT_DIR}/logs/runner.log" 2>&1
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
        printf 'Cache directory detected in the prospective bundle.\n' >&2
        return 1
    fi
    if [[ -n "$(find "${RESULT_DIR}" -type d \( \
        -name '.venv' -o -name 'venv' -o -name 'site-packages' \
    \) -print -quit 2>/dev/null)" ]]; then
        printf 'Python environment directory detected in the prospective bundle.\n' >&2
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
    if ! tar -tzf "${archive_tmp}" \
        "${RESULT_BASENAME}/source_snapshot/analyse/joint_configuration_soft_metric_gate.py" \
        >/dev/null; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        return 1
    fi
    if ! tar -tzf "${archive_tmp}" \
        "${RESULT_BASENAME}/source_snapshot/analyse/high_regime_prior_model_blind_validation.py" \
        >/dev/null; then
        rm -f -- "${archive_tmp}" "${sha_tmp}"
        return 1
    fi
    if ! tar -tzf "${archive_tmp}" \
        "${RESULT_BASENAME}/engineering_fix.json" \
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

    mkdir -p -- \
        "${RESULT_DIR}/logs" "${RESULT_DIR}/pre_eval" \
        "${RESULT_DIR}/evaluation"
    : >>"${RESULT_DIR}/logs/py_compile.log"
    : >>"${RESULT_DIR}/logs/pytest_focused.log"
    : >>"${RESULT_DIR}/logs/synthetic_smoke.log"
    : >>"${RESULT_DIR}/logs/scorer_replay.log"
    : >>"${RESULT_DIR}/logs/candidate_generation.log"
    : >>"${RESULT_DIR}/logs/evaluation.log"
    : >>"${RESULT_DIR}/logs/analysis.log"
    : >>"${RESULT_DIR}/logs/runner.log"
    rm -f -- "${RESULT_DIR}/COMPLETE"
    if ! verify_failure_outputs >/dev/null 2>&1; then
        run_gate_mode --write-failure-bundle "${failure_message}" \
            >>"${RESULT_DIR}/logs/runner.log" 2>&1
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
           'import numpy, scipy, yaml, pytest, sklearn, torch, gpytorch' \
           >/dev/null 2>&1; then
        conda_candidate="$(dirname -- "${REPO_ROOT}")/.conda/envs/nas/bin/python"
        if [[ -x "${conda_candidate}" ]] && \
           "${conda_candidate}" -c \
               'import numpy, scipy, yaml, pytest, sklearn, torch, gpytorch' \
               >/dev/null 2>&1; then
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
    [[ -f "${FROZEN_SOFT_CONFIG}" ]]
    [[ -f "${FROZEN_SOFT_SOURCE}" ]]
    [[ -f "${FROZEN_JOINT_SOURCE}" ]]
    [[ -f "${FROZEN_PERFORMANCE_SOURCE}" ]]
    [[ -f "${FROZEN_PAIR_AUDIT_SOURCE}" ]]
    [[ -f "${FROZEN_SOFT_MANIFEST}" ]]
    [[ -f "${DIAGNOSIS_RESULT_ROOT}/COMPLETE" ]]
    [[ -f "${FROZEN_HIGH_PLAN_SOURCE}" ]]
    [[ -f "${FROZEN_HIGH_EVALUATOR_SOURCE}" ]]
    [[ -f "${FROZEN_HIGH_ANALYSIS_SOURCE}" ]]
    [[ -f "${FROZEN_EVAL_UTILS_SOURCE}" ]]
    [[ -f "${FROZEN_ARCH_DIAGNOSTICS_SOURCE}" ]]
    [[ -f "${FROZEN_HIGH_SNAPSHOT_AUDIT}" ]]
    [[ -f "${REPO_ROOT}/AGENTS.md" ]]

    CONFIG_SHA256="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
    SOURCE_SHA256="$("${PYTHON_BIN}" - \
        "${SOURCE_PATH}" "${BASH_SOURCE[0]}" "${TEST_PATH}" \
        "${CONFIG_PATH}" "${FROZEN_SOFT_CONFIG}" "${FROZEN_SOFT_SOURCE}" \
        "${FROZEN_JOINT_SOURCE}" "${FROZEN_PERFORMANCE_SOURCE}" \
        "${FROZEN_PAIR_AUDIT_SOURCE}" "${FROZEN_HIGH_PLAN_SOURCE}" \
        "${FROZEN_HIGH_EVALUATOR_SOURCE}" "${FROZEN_HIGH_ANALYSIS_SOURCE}" \
        "${FROZEN_EVAL_UTILS_SOURCE}" "${FROZEN_ARCH_DIAGNOSTICS_SOURCE}" <<'PY'
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
        printf 'Another prospective selection gate runner holds the formal lock.\n' >&2
        return 1
    fi
    assert_targets_available

    if [[ "${MODE}" == 'dry-run' ]]; then
        CURRENT_STAGE="dry_run"
        announce_stage 'validating the frozen protocol without replay, candidate generation, or GNN training'
        run_gate_mode --dry-run
        if [[ -e "${RESULT_DIR}" || -L "${RESULT_DIR}" ]]; then
            printf 'Dry-run unexpectedly created the formal result directory.\n' >&2
            return 1
        fi
        return 0
    fi

    mkdir -p -- \
        "${RESULT_DIR}/logs" "${RESULT_DIR}/pre_eval" \
        "${RESULT_DIR}/evaluation"
    : >"${RESULT_DIR}/logs/py_compile.log"
    : >"${RESULT_DIR}/logs/pytest_focused.log"
    : >"${RESULT_DIR}/logs/synthetic_smoke.log"
    : >"${RESULT_DIR}/logs/scorer_replay.log"
    : >"${RESULT_DIR}/logs/candidate_generation.log"
    : >"${RESULT_DIR}/logs/evaluation.log"
    : >"${RESULT_DIR}/logs/analysis.log"
    : >"${RESULT_DIR}/logs/runner.log"
    RUN_ACTIVE=1

    CURRENT_STAGE="py_compile"
    announce_stage 'checking Python syntax'
    "${PYTHON_BIN}" -m py_compile "${SOURCE_PATH}" "${TEST_PATH}" \
        >"${RESULT_DIR}/logs/py_compile.log" 2>&1

    CURRENT_STAGE="focused_pytest"
    announce_stage 'running focused CPU-safe tests'
    "${PYTHON_BIN}" -m pytest -q "${TEST_PATH}" \
        >"${RESULT_DIR}/logs/pytest_focused.log" 2>&1

    CURRENT_STAGE="synthetic_smoke"
    announce_stage 'running the synthetic smoke test'
    run_gate_mode --synthetic-smoke \
        >"${RESULT_DIR}/logs/synthetic_smoke.log" 2>&1

    CURRENT_STAGE="formal_gate"
    announce_stage 'replaying frozen scorers and freezing all 8 candidate pools before any new validation'
    announce_stage '96 GNN runs are permitted only if the frozen M2/B3 disagreement gate passes'
    run_gate_mode --formal --preflight-ready \
        >"${RESULT_DIR}/logs/runner.log" 2>&1

    CURRENT_STAGE="formal_output_verification"
    announce_stage 'verifying the prospective bundle and immutable pre-evaluation state'
    rm -f -- "${RESULT_DIR}/FAILED" "${RESULT_DIR}/error.json"
    touch "${RESULT_DIR}/COMPLETE"
    verify_success_outputs

    CURRENT_STAGE="manifest_refresh"
    refresh_manifest_twice
    verify_success_outputs

    CURRENT_STAGE="archive_packaging"
    announce_stage 'creating the source-complete archive and SHA-256 sidecar'
    package_bundle

    RUN_SUCCEEDED=1
    printf 'Soft joint metric prospective selection gate completed.\n'
    printf 'Result directory: %s\n' "${RESULT_DIR}"
    printf 'Archive: %s\n' "${ARCHIVE_PATH}"
    printf 'SHA-256: %s\n' "${ARCHIVE_SHA_PATH}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

main "$@"
