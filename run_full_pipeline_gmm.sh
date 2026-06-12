#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_IF_DONE="${SKIP_IF_DONE:-0}"

SEED="${SEED:-42}"
CORA_ROOT="${CORA_ROOT:-/tmp/Cora}"
TRAIN_VERSION="${TRAIN_VERSION:-pipeline}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/joint_search}"

NUM_SAMPLES="${NUM_SAMPLES:-3000}"
SKIP_PROB="${SKIP_PROB:-0.35}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"

PHASE2_N_INIT="${PHASE2_N_INIT:-10}"
PHASE2_N_ITER="${PHASE2_N_ITER:-40}"

PHASE3_N_INIT="${PHASE3_N_INIT:-15}"
PHASE3_N_ITER="${PHASE3_N_ITER:-50}"

PHASE4_BO_N_INIT="${PHASE4_BO_N_INIT:-20}"
PHASE4_BO_N_ITER="${PHASE4_BO_N_ITER:-60}"
PHASE4_TPE_N_TRIALS="${PHASE4_TPE_N_TRIALS:-80}"

SEARCH_EVAL_EPOCHS="${SEARCH_EVAL_EPOCHS:-100}"
SEARCH_PATIENCE="${SEARCH_PATIENCE:-20}"

GMM_INIT_TRIALS="${GMM_INIT_TRIALS:-20}"
GMM_TOP_FRAC="${GMM_TOP_FRAC:-0.3}"
GMM_N_COMPONENTS="${GMM_N_COMPONENTS:-4}"
GMM_WEIGHT_TEMP="${GMM_WEIGHT_TEMP:-8.0}"

FINAL_N_SEEDS="${FINAL_N_SEEDS:-10}"
FINAL_EVAL_EPOCHS="${FINAL_EVAL_EPOCHS:-200}"
FINAL_PATIENCE="${FINAL_PATIENCE:-30}"

LOG_DIR="logs/run_full_pipeline_gmm"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_full_$(date +%Y%m%d_%H%M%S).log"

SUCCEEDED_STEPS=()
SKIPPED_STEPS=()
FAILED_STEPS=()
OVERALL_STATUS=0
PIPELINE_START_TS="$(date +%s)"

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

print_command() {
  local arg
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

run_step() {
  local key="$1"
  local done_file="$2"
  shift 2
  local start_ts end_ts elapsed exit_code

  if [[ "${SKIP_IF_DONE}" == "1" && -n "${done_file}" && -f "${done_file}" ]]; then
    SKIPPED_STEPS+=("${key}:done")
    log_line "SKIP step=${key} reason=done path=${done_file}"
    return 0
  fi

  log_line "START step=${key}"
  log_line "done_file=${done_file:-none}"
  log_line "command=$(print_command "$@")"
  start_ts="$(date +%s)"

  if [[ "${DRY_RUN}" == "1" ]]; then
    exit_code=0
    log_line "DRY_RUN step=${key} not executed"
  else
    set +e
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    exit_code="${PIPESTATUS[0]}"
    set -e
  fi

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  log_line "END step=${key} elapsed_seconds=${elapsed} exit_code=${exit_code}"

  if [[ "${exit_code}" -eq 0 ]]; then
    SUCCEEDED_STEPS+=("${key}")
    return 0
  fi

  FAILED_STEPS+=("${key}:${exit_code}")
  OVERALL_STATUS=1
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    log_line "CONTINUE_ON_ERROR=1; continuing after failed step=${key}"
    return 0
  fi

  log_line "CONTINUE_ON_ERROR=0; exiting after failed step=${key}"
  exit "${exit_code}"
}

dataset_path() {
  printf 'data/mini_gnn_dataset_%s.pkl' "$1"
}

checkpoint_path() {
  printf '%s/joint_model_%s_%s_best.pth' "${CHECKPOINT_DIR}" "${TRAIN_VERSION}" "$1"
}

phase3_dir() {
  printf 'results/bo_phase3_%s' "$1"
}

phase4_tpe_dir() {
  printf 'results/bo_phase4_tpe_%s_gmm' "$1"
}

log_line "run_full_pipeline_gmm started"
log_line "log_file=${LOG_FILE}"
log_line "cwd=$(pwd)"
log_line "PYTHON_BIN=${PYTHON_BIN}"
log_line "DRY_RUN=${DRY_RUN} CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR} SKIP_IF_DONE=${SKIP_IF_DONE}"
log_line "TRAIN_VERSION=${TRAIN_VERSION} CHECKPOINT_DIR=${CHECKPOINT_DIR}"
log_line "NUM_SAMPLES=${NUM_SAMPLES} TRAIN_EPOCHS=${TRAIN_EPOCHS} SEARCH_EVAL_EPOCHS=${SEARCH_EVAL_EPOCHS}"

for mode in global4 hybrid_cond7 layer_cond19; do
  data_file="$(dataset_path "${mode}")"
  ckpt_file="$(checkpoint_path "${mode}")"

  run_step "generate_data_${mode}" "${data_file}" \
    "${PYTHON_BIN}" generate_mini_data.py \
      --hp_mode "${mode}" \
      --num_samples "${NUM_SAMPLES}" \
      --skip_prob "${SKIP_PROB}" \
      --output "${data_file}" \
      --seed "${SEED}"

  run_step "train_joint_${mode}" "${ckpt_file}" \
    "${PYTHON_BIN}" train_joint.py \
      --hp_mode "${mode}" \
      --data "${data_file}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --version "${TRAIN_VERSION}" \
      --epochs "${TRAIN_EPOCHS}" \
      --batch_size "${TRAIN_BATCH_SIZE}" \
      --seed "${SEED}"
done

GLOBAL4_CKPT="$(checkpoint_path global4)"

run_step "phase2_arch_only" "results/bo_phase2/history_final.json" \
  "${PYTHON_BIN}" bo_phase2.py \
    --checkpoint "${GLOBAL4_CKPT}" \
    --vae_hp_mode global4 \
    --cora_root "${CORA_ROOT}" \
    --output results/bo_phase2 \
    --version arch_only \
    --n_init "${PHASE2_N_INIT}" \
    --n_iter "${PHASE2_N_ITER}" \
    --eval_epochs "${SEARCH_EVAL_EPOCHS}" \
    --patience "${SEARCH_PATIENCE}" \
    --seed "${SEED}"

for mode in global4 hybrid_cond7 layer_cond19; do
  ckpt_file="$(checkpoint_path "${mode}")"
  p3_dir="$(phase3_dir "${mode}")"

  run_step "phase3_${mode}" "${p3_dir}/history_final.json" \
    "${PYTHON_BIN}" bo_phase3.py \
      --hp_mode "${mode}" \
      --checkpoint "${ckpt_file}" \
      --warm_start results/bo_phase2/best_z_arch_final.pt \
      --cora_root "${CORA_ROOT}" \
      --output "${p3_dir}" \
      --version "${mode}_phase3" \
      --n_init "${PHASE3_N_INIT}" \
      --n_iter "${PHASE3_N_ITER}" \
      --eval_epochs "${SEARCH_EVAL_EPOCHS}" \
      --patience "${SEARCH_PATIENCE}" \
      --seed "${SEED}"
done

run_step "phase4_global4_bo_gmm" "results/bo_phase4_global4_bo_gmm/history_final.json" \
  "${PYTHON_BIN}" bo_phase4.py \
    --hp_mode global4 \
    --checkpoint "$(checkpoint_path global4)" \
    --warm_start "$(phase3_dir global4)/best_z_final.pt" \
    --cora_root "${CORA_ROOT}" \
    --output results/bo_phase4_global4_bo_gmm \
    --version global4_bo_gmm \
    --n_init "${PHASE4_BO_N_INIT}" \
    --n_iter "${PHASE4_BO_N_ITER}" \
    --eval_epochs "${SEARCH_EVAL_EPOCHS}" \
    --patience "${SEARCH_PATIENCE}" \
    --gmm_init_history "$(phase3_dir global4)/history_final.json" \
    --gmm_init_trials "${GMM_INIT_TRIALS}" \
    --gmm_top_frac "${GMM_TOP_FRAC}" \
    --gmm_n_components "${GMM_N_COMPONENTS}" \
    --gmm_weight_temp "${GMM_WEIGHT_TEMP}" \
    --gmm_save_summary \
    --seed "${SEED}"

for mode in global4 hybrid_cond7 layer_cond19; do
  run_step "phase4_${mode}_tpe_gmm" "$(phase4_tpe_dir "${mode}")/history_final.json" \
    "${PYTHON_BIN}" bo_phase4_tpe.py \
      --hp_mode "${mode}" \
      --checkpoint "$(checkpoint_path "${mode}")" \
      --warm_start "$(phase3_dir "${mode}")/best_z_final.pt" \
      --cora_root "${CORA_ROOT}" \
      --output "$(phase4_tpe_dir "${mode}")" \
      --version "${mode}_tpe_gmm" \
      --n_trials "${PHASE4_TPE_N_TRIALS}" \
      --eval_epochs "${SEARCH_EVAL_EPOCHS}" \
      --patience "${SEARCH_PATIENCE}" \
      --gmm_init_history "$(phase3_dir "${mode}")/history_final.json" \
      --gmm_init_trials "${GMM_INIT_TRIALS}" \
      --gmm_top_frac "${GMM_TOP_FRAC}" \
      --gmm_n_components "${GMM_N_COMPONENTS}" \
      --gmm_weight_temp "${GMM_WEIGHT_TEMP}" \
      --gmm_save_summary \
      --seed "${SEED}"
done

run_step "final_eval_global4_bo_gmm" "results/final_eval_global4_bo_gmm/final_results_global4.json" \
  "${PYTHON_BIN}" final_eval.py \
    --hp_mode global4 \
    --checkpoint "$(checkpoint_path global4)" \
    --auto_best_z results/bo_phase4_global4_bo_gmm/best_z_final.pt \
    --output results/final_eval_global4_bo_gmm \
    --version final_global4_bo_gmm \
    --cora_root "${CORA_ROOT}" \
    --n_seeds "${FINAL_N_SEEDS}" \
    --eval_epochs "${FINAL_EVAL_EPOCHS}" \
    --patience "${FINAL_PATIENCE}" \
    --seed "${SEED}"

for mode in global4 hybrid_cond7 layer_cond19; do
  run_step "final_eval_${mode}_tpe_gmm" "results/final_eval_${mode}_tpe_gmm/final_results_${mode}.json" \
    "${PYTHON_BIN}" final_eval.py \
      --hp_mode "${mode}" \
      --checkpoint "$(checkpoint_path "${mode}")" \
      --auto_best_z "$(phase4_tpe_dir "${mode}")/best_z_final.pt" \
      --output "results/final_eval_${mode}_tpe_gmm" \
      --version "final_${mode}_tpe_gmm" \
      --cora_root "${CORA_ROOT}" \
      --n_seeds "${FINAL_N_SEEDS}" \
      --eval_epochs "${FINAL_EVAL_EPOCHS}" \
      --patience "${FINAL_PATIENCE}" \
      --seed "${SEED}"
done

TOTAL_ELAPSED="$(($(date +%s) - PIPELINE_START_TS))"
log_line "SUMMARY succeeded=${SUCCEEDED_STEPS[*]:-none}"
log_line "SUMMARY skipped=${SKIPPED_STEPS[*]:-none}"
log_line "SUMMARY failed=${FAILED_STEPS[*]:-none}"
log_line "SUMMARY total_elapsed_seconds=${TOTAL_ELAPSED}"

if [[ "${#FAILED_STEPS[@]}" -gt 0 ]]; then
  exit "${OVERALL_STATUS}"
fi
