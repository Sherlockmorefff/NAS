#!/usr/bin/env bash
set -euo pipefail

cd /home/fangxy/D-VAE
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

python analyse/boundary_training_ceiling_audit.py \
  --config configs/boundary_training_ceiling_audit_global4_v1.json \
  --run
