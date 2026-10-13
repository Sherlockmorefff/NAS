#!/usr/bin/env bash
set -euo pipefail
cd /home/fangxy/D-VAE
session=acq_optimizer_offline_20261012v1
run_dir=/home/fangxy/D-VAE/results/acq_optimizer_offline_20261012v1
python_bin=/home/fangxy/.conda/envs/nas/bin/python
prompt_path=/home/fangxy/D-VAE/artifacts/acq_optimizer_offline_20261012v1/PROMPT.txt
mkdir -p "$run_dir"
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' 'Existing session; inspect status and resume exactly, refusing duplicate launch.'
  exit 2
fi
if [[ -f "$run_dir/COMPLETE.json" ]]; then printf '%s\n' 'Already complete; refusing additional runs.'; exit 0; fi
action=run
if [[ -f "$run_dir/PROTOCOL.json" ]]; then action=resume; fi
tmux new-session -d -s "$session" -n controller
tmux set-option -t "$session" remain-on-exit on
for window in worker0 worker1 acceptance; do tmux new-window -t "$session" -n "$window"; done
for window in controller worker0 worker1 acceptance; do
  case "$window" in
    controller) args="$action --prompt $prompt_path";;
    worker0) args='supervisor --worker 0';;
    worker1) args='supervisor --worker 1';;
    acceptance) args='acceptance';;
  esac
  cmd="cd /home/fangxy/D-VAE; export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; $python_bin -u -B -m analyse.acq_optimizer_offline $args --run-dir $run_dir >> $run_dir/$window.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $run_dir/${window}_exit_code.txt; exit \$rc"
  tmux send-keys -t "$session:$window" "$cmd" Enter
done
printf '%s\n' "$session"
