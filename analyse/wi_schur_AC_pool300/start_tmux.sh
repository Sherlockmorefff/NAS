#!/usr/bin/env bash
set -euo pipefail
cd /home/fangxy/D-VAE
session=wi_schur_AC_20261012v1
root=/home/fangxy/D-VAE/results/wi_schur_AC_pool300_20261012v1
py=/home/fangxy/.conda/envs/nas/bin/python
test -f "$root/freeze.json"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing A/C session; inspect, do not double launch: $session"
  tmux list-windows -t "$session"
  exit 1
fi
tmux new-session -d -s "$session" -n controller
tmux set-window-option -t "$session:controller" remain-on-exit on
tmux send-keys -t "$session:controller" "cd /home/fangxy/D-VAE; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; $py -u -B -m analyse.wi_schur_AC_pool300 run --run-dir $root >> $root/controller.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $root/controller_exit_code.txt; exit \$rc" C-m
tmux new-window -t "$session" -n acceptance
tmux set-window-option -t "$session:acceptance" remain-on-exit on
tmux send-keys -t "$session:acceptance" "cd /home/fangxy/D-VAE; $py -u -B -m analyse.wi_schur_AC_pool300 accept --run-dir $root >> $root/acceptance.log 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > $root/acceptance_exit_code.txt; exit \$rc" C-m
tmux new-window -t "$session" -n logs "tail -n 20 -F $root/controller.log $root/worker0.log $root/worker1.log $root/gp_worker0.log $root/gp_worker1.log"
tmux list-windows -t "$session"
