#!/usr/bin/env bash
set -euo pipefail

cd /home/fangxy/D-VAE
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export D_VAE_REPO_ROOT=/home/fangxy/D-VAE
unset CUDA_VISIBLE_DEVICES

python_bin=/home/fangxy/.conda/envs/nas/bin/python
analyser=analyse/layernorm_collapse_diagnostic_global4_v1.py
result_root=/home/fangxy/D-VAE/results/posthoc/layernorm-collapse-diagnostic-global4-v1
snapshot_analyser="${result_root}/source_snapshot/analyse/layernorm_collapse_diagnostic_global4_v1.py"
tmux_name=layernorm_collapse_diagnostic_v1
process_log="${result_root}/formal/process.log"

"${python_bin}" -B "${analyser}" preflight --result-root "${result_root}" >/dev/null
run_id="layernorm-collapse-diagnostic-v1-$("${python_bin}" -c 'import uuid; print(uuid.uuid4())')"
formal_command="cd /home/fangxy/D-VAE && exec env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 D_VAE_REPO_ROOT=/home/fangxy/D-VAE ${python_bin} -B ${snapshot_analyser} formal-run --result-root ${result_root} --run-id ${run_id} >${process_log} 2>&1"
tmux new-session -d -s "${tmux_name}" "${formal_command}"

printf '{"run_id":"%s","result_root":"%s","tmux":"%s"}\n' "${run_id}" "${result_root}" "${tmux_name}"
