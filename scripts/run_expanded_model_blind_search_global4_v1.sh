#!/usr/bin/env bash
set -euo pipefail

cd /home/fangxy/D-VAE
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES

set +e
conda run -n nas python -B analyse/expanded_model_blind_search_global4_v1.py \
  --config configs/expanded_model_blind_search_global4_v1.json \
  --runner-enter
enter_rc=$?
if (( enter_rc != 0 )); then
  exit "${enter_rc}"
fi

conda run -n nas python -B analyse/expanded_model_blind_search_global4_v1.py \
  --config configs/expanded_model_blind_search_global4_v1.json \
  --formal-run
formal_rc=$?

conda run -n nas python -B analyse/expanded_model_blind_search_global4_v1.py \
  --config configs/expanded_model_blind_search_global4_v1.json \
  --runner-exit
exit_rc=$?

if (( formal_rc != 0 )); then
  exit "${formal_rc}"
fi
exit "${exit_rc}"
