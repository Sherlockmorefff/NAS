#!/bin/bash

# ================================================================
# GNN-NAS 联合搜索自动化流水线脚本 (适配 8 维极致压缩空间)
# ================================================================

# 设置：遇到任何错误立刻终止脚本
set -e

# 设置 Cora 数据集默认路径
CORA_ROOT="/tmp/Cora"

echo "================================================================"
echo "  🚀 启动 GNN-NAS 联合搜索自动化流水线 "
echo "================================================================"

# ── 1. 验证 jacobian_utils Bug 修复（应全部 PASS，不再报维度错误）──
echo "=> 正在验证 jacobian_utils..."
python jacobian_utils.py

# ── 2. 验证 eval_utils v2（hidden_dim + l2 反归一化 + 训练自测）──
echo "=> 正在验证 eval_utils..."
python eval_utils.py --cora_root "${CORA_ROOT}"

# ── 3. Phase 3 搜索空间 v2（SEARCH_DIM=12）───────────────────
echo "=> 启动 Phase 3..."
python bo_phase3.py \
  --checkpoint results/joint_search/joint_model_ep50.pth \
  --warm_start results/bo_phase2_nz8/best_z_arch_final.pt \
  --cora_root  "${CORA_ROOT}" \
  --n_init 15 --n_iter 50 \
  --output results/bo_phase3_nz8

# ── 4. Phase 4 搜索空间 v2 + GPND-NAS（SEARCH_DIM=12）──────────
echo "=> 启动 Phase 4..."
python bo_phase4.py \
  --checkpoint results/joint_search/joint_model_ep50.pth \
  --warm_start results/bo_phase2_nz8/best_z_arch_final.pt \
  --cora_root  "${CORA_ROOT}" \
  --n_init 20 --n_iter 60 \
  --n_probes 8 \
  --output results/bo_phase4_nz8

# ── 5. 最终多种子评估（含 v2 候选）──────────────────────────
#    Phase 3/4 跑完后，将实际发现的最优 LR/hd/l2 填入
#    final_eval.py 的 CANDIDATES 列表，再执行：
echo "=> 启动 最终评估..."
python final_eval.py \
  --cora_root "${CORA_ROOT}" \
  --n_seeds 10 \
  --output results/final_eval_nz8