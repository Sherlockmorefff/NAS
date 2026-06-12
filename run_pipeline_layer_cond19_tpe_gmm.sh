#!/usr/bin/env bash
set -euo pipefail

HP_MODE="layer_cond19"
SEARCH_DIM=31
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="${VERSION:-${HP_MODE}_tpe_gmm}"
DATA_PATH="${DATA_PATH:-data/mini_gnn_dataset_${HP_MODE}.pkl}"
CHECKPOINT="${CHECKPOINT:-results/joint_search/joint_model_${HP_MODE}_best.pth}"
OUTPUT="${OUTPUT:-results/bo_phase4_tpe_${HP_MODE}_gmm}"
GMM_HISTORY="${GMM_HISTORY:-results/bo_phase3_${HP_MODE}/history_final.json}"

if [[ ! -f "${GMM_HISTORY}" && -f "results/bo_phase3/history_final.json" ]]; then
  GMM_HISTORY="results/bo_phase3/history_final.json"
fi

GMM_HISTORY_ARGS=()
if [[ -f "${GMM_HISTORY}" ]]; then
  GMM_HISTORY_ARGS=(--gmm_init_history "${GMM_HISTORY}")
else
  echo "WARNING: GMM history not found; running without --gmm_init_history: ${GMM_HISTORY}" >&2
fi

echo "HP_MODE=${HP_MODE} SEARCH_DIM=${SEARCH_DIM}"
echo "DATA_PATH=${DATA_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT=${OUTPUT}"

"${PYTHON_BIN}" bo_phase4_tpe.py \
  --hp_mode "${HP_MODE}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --version "${VERSION}" \
  --gmm_init_trials 20 \
  --gmm_top_frac 0.3 \
  --gmm_n_components 4 \
  --gmm_weight_temp 8.0 \
  --gmm_save_summary \
  "${GMM_HISTORY_ARGS[@]}" \
  "$@"
