#!/usr/bin/env bash
set -euo pipefail
cd /home/fangxy/D-VAE
session=ac_open_graph_mix_20261012v1
run_dir=/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1
python_bin=/home/fangxy/.conda/envs/nas/bin/python
prompt=/home/fangxy/D-VAE/artifacts/ac_open_graph_mix_20261012v1/PROMPT.txt
mkdir -p "$run_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' 'Existing dedicated session; inspect status instead of duplicate launch.'; exit 2
fi
if [[ -f "$run_dir/COMPLETE.json" ]]; then printf '%s\n' 'Already complete; no extra experiment.'; exit 0; fi
tmux new-session -d -s "$session" -n controller
for window in acceptance logs; do tmux new-window -t "$session" -n "$window"; done
for window in controller acceptance logs; do tmux set-window-option -t "$session:$window" remain-on-exit on; done
env_prefix="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8"
recover=''
if [[ -f "$run_dir/freeze.json" ]]; then recover='--recover'; fi
cmd="cd /home/fangxy/D-VAE; $env_prefix; $python_bin -u -B -m analyse.ac_open_graph_mix run $recover --prompt $prompt --run-dir $run_dir >> $run_dir/controller.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $run_dir/controller_exit_code.txt; exit \$rc"
tmux send-keys -t "$session:controller" "$cmd" Enter
cmd="cd /home/fangxy/D-VAE; $env_prefix; $python_bin -u -B -m analyse.ac_open_graph_mix timer --run-dir $run_dir >> $run_dir/acceptance.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $run_dir/acceptance_exit_code.txt; exit \$rc"
tmux send-keys -t "$session:acceptance" "$cmd" Enter
tmux send-keys -t "$session:logs" "tail -F $run_dir/controller.log" Enter
printf '%s\n' "$session"
