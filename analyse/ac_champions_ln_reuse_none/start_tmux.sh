#!/usr/bin/env bash
set -euo pipefail
cd /home/fangxy/D-VAE
SESSION=ac_champions_ln_reuse_none_20261013v1
RUN_ROOT=/home/fangxy/D-VAE/results/ac_champions_ln_reuse_none_20261013v1
PYTHON_BIN=/home/fangxy/.conda/envs/nas/bin/python
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "This experiment tmux session already exists; inspect status and use identity-matched recovery."
  exit 1
fi
test -f "$RUN_ROOT/freeze.json"
tmux new-session -d -s "$SESSION" -n controller "cd /home/fangxy/D-VAE; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 $PYTHON_BIN -u -B -m analyse.ac_champions_ln_reuse_none run --root $RUN_ROOT >> $RUN_ROOT/controller.log 2>&1; rc=\$?; echo \$rc > $RUN_ROOT/controller_exit_code.txt; exec bash"
tmux new-window -t "$SESSION" -n acceptance "cd /home/fangxy/D-VAE; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PYTHON_BIN -u -B -m analyse.ac_champions_ln_reuse_none timer --root $RUN_ROOT >> $RUN_ROOT/acceptance.log 2>&1; rc=\$?; echo \$rc > $RUN_ROOT/acceptance_exit_code.txt; exec bash"
tmux new-window -t "$SESSION" -n logs "tail -F $RUN_ROOT/controller.log $RUN_ROOT/worker0.log $RUN_ROOT/worker1.log"
echo "Started $SESSION"
