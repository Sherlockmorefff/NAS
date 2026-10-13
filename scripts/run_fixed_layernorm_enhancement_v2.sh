#!/usr/bin/env bash
set -euo pipefail

cd /home/fangxy/D-VAE
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES

set +e
python -B analyse/fixed_layernorm_enhancement_v2.py \
  --config configs/fixed_layernorm_enhancement_global4_v2.json \
  --runner-enter
runner_enter_rc=$?
if [[ ${runner_enter_rc} -ne 0 ]]; then
  python -B analyse/fixed_layernorm_enhancement_v2.py \
    --config configs/fixed_layernorm_enhancement_global4_v2.json \
    --record-preformal-failure
  exit "${runner_enter_rc}"
fi

python -B analyse/fixed_layernorm_enhancement_v2.py \
  --config configs/fixed_layernorm_enhancement_global4_v2.json \
  --formal-run
formal_rc=$?
python -B analyse/fixed_layernorm_enhancement_v2.py \
  --config configs/fixed_layernorm_enhancement_global4_v2.json \
  --runner-exit
runner_exit_rc=$?
if [[ ${runner_exit_rc} -ne 0 && ${formal_rc} -ne 0 ]]; then
  python -B analyse/fixed_layernorm_enhancement_v2.py \
    --config configs/fixed_layernorm_enhancement_global4_v2.json \
    --record-preformal-failure
fi
exit "${formal_rc}"
