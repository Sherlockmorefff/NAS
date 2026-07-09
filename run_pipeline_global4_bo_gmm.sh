#!/usr/bin/env bash
set -euo pipefail

HP_MODE="global4"
SEARCH_DIM=16
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="${VERSION:-${HP_MODE}_bo_gmm}"
DATA_PATH="${DATA_PATH:-data/mini_gnn_dataset_${HP_MODE}.pkl}"
CHECKPOINT="${CHECKPOINT:-results/joint_search/joint_model_${HP_MODE}_best.pth}"
OUTPUT="${OUTPUT:-results/bo_phase4_${HP_MODE}_bo_gmm}"
GMM_HISTORY="${GMM_HISTORY:-results/bo_phase3_${HP_MODE}/history_final.json}"
GP_CHECKPOINT="${GP_CHECKPOINT:-results/gp_predictor/${HP_MODE}/accuracy_gp_offline.pt}"

if [[ ! -f "${GMM_HISTORY}" && -f "results/bo_phase3/history_final.json" ]]; then
  GMM_HISTORY="results/bo_phase3/history_final.json"
fi

if [[ -f "${GMM_HISTORY}" ]]; then
  GMM_HISTORY_ARGS=(--gmm_init_history "${GMM_HISTORY}")
else
  echo "ERROR: history not found; offline GP training cannot proceed: ${GMM_HISTORY}" >&2
  exit 1
fi

echo "HP_MODE=${HP_MODE} SEARCH_DIM=${SEARCH_DIM}"
echo "DATA_PATH=${DATA_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT=${OUTPUT}"

"${PYTHON_BIN}" surrogate/train_accuracy_gp.py \
  --history_paths "${GMM_HISTORY}" \
  --hp_mode "${HP_MODE}" \
  --arch_nz 12 \
  --z_bound 2.5 \
  --holdout_frac 0.2 \
  --seed 42 \
  --checkpoint "${CHECKPOINT}" \
  --dataset Cora \
  --eval_epochs 100 \
  --patience 20 \
  --output "${GP_CHECKPOINT}" \
  --prediction_output "$(dirname "${GP_CHECKPOINT}")/offline_holdout_predictions.csv" \
  --metrics_output "$(dirname "${GP_CHECKPOINT}")/offline_metrics.json"

"${PYTHON_BIN}" bo_phase4.py \
  --hp_mode "${HP_MODE}" \
  --checkpoint "${CHECKPOINT}" \
  --gp_checkpoint "${GP_CHECKPOINT}" \
  --warm_start "$(dirname "${GMM_HISTORY}")/best_z_final.pt" \
  --output "${OUTPUT}" \
  --version "${VERSION}" \
  --gmm_init_trials 20 \
  --gmm_top_frac 0.3 \
  --gmm_n_components 4 \
  --gmm_weight_temp 8.0 \
  --gmm_save_summary \
  "${GMM_HISTORY_ARGS[@]}" \
  "$@"
