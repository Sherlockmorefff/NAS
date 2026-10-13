#!/usr/bin/env bash
set -euo pipefail
cd /home/fangxy/D-VAE
session=wi_schur_B_20261011v1
run_dir=/home/fangxy/D-VAE/results/wi_schur_B_pool300_20261011v1
python_bin=/home/fangxy/.conda/envs/nas/bin/python
test -f "$run_dir/freeze.json"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing session found; inspect instead of double launch: $session"
  tmux list-windows -t "$session"
  exit 1
fi
tmux new-session -d -s "$session" -n pipeline "cd /home/fangxy/D-VAE; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; $python_bin -u -B -m analyse.wi_schur_B_pool300 run --run-dir $run_dir >> $run_dir/controller.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $run_dir/controller_exit_code.txt; exit \$rc"
tmux set-option -t "$session" remain-on-exit on
tmux new-window -t "$session" -n acceptance "cd /home/fangxy/D-VAE; $python_bin -u -B -m analyse.wi_schur_B_pool300 accept --run-dir $run_dir >> $run_dir/acceptance.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $run_dir/acceptance_exit_code.txt; exit \$rc"
tmux new-window -t "$session" -n logs "tail -n 20 -F $run_dir/controller.log $run_dir/worker0.log $run_dir/worker1.log"
tmux list-windows -t "$session"
