#!/usr/bin/env bash
set -euo pipefail

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-0}"
RUN_ONLY="${RUN_ONLY:-}"

LOG_DIR="logs/run_all_phase4_gmm"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

TASK_KEYS=(
  "global4_bo"
  "global4_tpe"
  "hybrid_cond7_tpe"
  "layer_cond19_tpe"
)
TASK_SCRIPTS=(
  "./run_pipeline_global4_bo_gmm.sh"
  "./run_pipeline_global4_tpe_gmm.sh"
  "./run_pipeline_hybrid_cond7_tpe_gmm.sh"
  "./run_pipeline_layer_cond19_tpe_gmm.sh"
)
TASK_DONE_FILES=(
  "results/bo_phase4_global4_bo_gmm/history_final.json"
  "results/bo_phase4_tpe_global4_gmm/history_final.json"
  "results/bo_phase4_tpe_hybrid_cond7_gmm/history_final.json"
  "results/bo_phase4_tpe_layer_cond19_gmm/history_final.json"
)

IFS=',' read -r -a RUN_ONLY_KEYS <<< "${RUN_ONLY}"
EXTRA_ARGS=("$@")

SUCCEEDED_TASKS=()
SKIPPED_TASKS=()
FAILED_TASKS=()
OVERALL_STATUS=0
RUN_START_TS="$(date +%s)"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

run_only_contains() {
  local key="$1"
  local item
  if [[ -z "${RUN_ONLY}" ]]; then
    return 0
  fi
  for item in "${RUN_ONLY_KEYS[@]}"; do
    if [[ "${item}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_run_only() {
  local item key found
  if [[ -z "${RUN_ONLY}" ]]; then
    return 0
  fi
  for item in "${RUN_ONLY_KEYS[@]}"; do
    found=0
    for key in "${TASK_KEYS[@]}"; do
      if [[ "${item}" == "${key}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      log_line "ERROR: unknown RUN_ONLY task key: ${item}"
      log_line "Valid keys: ${TASK_KEYS[*]}"
      exit 2
    fi
  done
}

print_command() {
  local script="$1"
  shift
  printf 'bash %q' "${script}"
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

run_task() {
  local key="$1"
  local script="$2"
  local done_file="$3"
  local start_ts end_ts elapsed exit_code

  if ! run_only_contains "${key}"; then
    SKIPPED_TASKS+=("${key}:run_only")
    log_line "SKIP task=${key} reason=RUN_ONLY"
    return 0
  fi

  if [[ "${SKIP_IF_DONE}" == "1" && -f "${done_file}" ]]; then
    SKIPPED_TASKS+=("${key}:done")
    log_line "SKIP task=${key} reason=history_final_exists path=${done_file}"
    return 0
  fi

  log_line "START task=${key}"
  log_line "script=${script}"
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    log_line "command=$(print_command "${script}" "${EXTRA_ARGS[@]}")"
  else
    log_line "command=$(print_command "${script}")"
  fi
  start_ts="$(date +%s)"

  if [[ "${DRY_RUN}" == "1" ]]; then
    exit_code=0
    log_line "DRY_RUN task=${key} not executed"
  elif [[ ! -f "${script}" ]]; then
    exit_code=127
    log_line "ERROR task=${key} script not found: ${script}"
  else
    set +e
    if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
      bash "${script}" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
      bash "${script}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    exit_code="${PIPESTATUS[0]}"
    set -e
  fi

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  log_line "END task=${key} elapsed_seconds=${elapsed} exit_code=${exit_code}"

  if [[ "${exit_code}" -eq 0 ]]; then
    SUCCEEDED_TASKS+=("${key}")
    return 0
  fi

  FAILED_TASKS+=("${key}:${exit_code}")
  OVERALL_STATUS=1
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    log_line "CONTINUE_ON_ERROR=1; continuing after failed task=${key}"
    return 0
  fi

  log_line "CONTINUE_ON_ERROR=0; exiting after failed task=${key}"
  exit "${exit_code}"
}

validate_run_only

log_line "run_all_phase4_gmm started"
log_line "log_file=${LOG_FILE}"
log_line "cwd=$(pwd)"
log_line "DRY_RUN=${DRY_RUN} CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR} SKIP_IF_DONE=${SKIP_IF_DONE} RUN_ONLY=${RUN_ONLY}"
log_line "extra_args=${EXTRA_ARGS[*]:-}"
log_line "planned_tasks=${TASK_KEYS[*]}"

for idx in "${!TASK_KEYS[@]}"; do
  run_task "${TASK_KEYS[$idx]}" "${TASK_SCRIPTS[$idx]}" "${TASK_DONE_FILES[$idx]}"
done

TOTAL_ELAPSED="$(($(date +%s) - RUN_START_TS))"
log_line "SUMMARY succeeded=${SUCCEEDED_TASKS[*]:-none}"
log_line "SUMMARY skipped=${SKIPPED_TASKS[*]:-none}"
log_line "SUMMARY failed=${FAILED_TASKS[*]:-none}"
log_line "SUMMARY total_elapsed_seconds=${TOTAL_ELAPSED}"

if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
  exit "${OVERALL_STATUS}"
fi
