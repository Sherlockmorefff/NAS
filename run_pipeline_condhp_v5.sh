#!/bin/bash
# run_pipeline_condhp_v5.sh
# ================================================================
# GNN-NAS 条件参数 v5 完整流水线
# Search Space v3: max_n=7, nvt=8, nz=12, GCNII
# Conditional HP: [lr, dropout, gat_heads, sage_aggr]
# ================================================================
set -e

CORA_ROOT="/tmp/Cora"
VERSION="condhp_v5"
VAE_CKPT="results/joint_search/joint_model_${VERSION}_ep100.pth"

PHASE2_OUT="results/bo_phase2_${VERSION}"
PHASE3_OUT="results/bo_phase3_${VERSION}"
PHASE4_OUT="results/bo_phase4_${VERSION}"

echo "================================================================"
echo "  启动 GNN-NAS 条件参数 v5 完整流水线"
echo "  VERSION=${VERSION}"
echo "  CORA_ROOT=${CORA_ROOT}"
echo "================================================================"

echo "=== Step 1: 生成 v5 条件参数训练数据集 ==="
python generate_mini_data.py \
  --output data/mini_gnn_dataset_v5.pkl \
  --num_samples 3000 \
  --skip_prob 0.35 \
  --hp_dim 4 \
  --seed 42
# → data/mini_gnn_dataset_v5.pkl

echo "=== Step 2: 训练 v5 Joint VAE ==="
python train_joint.py \
  --data data/mini_gnn_dataset_v5.pkl \
  --checkpoint_dir results/joint_search \
  --version "${VERSION}" \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --beta_max 0.01 \
  --free_bits 2.0 \
  --seed 42
# → results/joint_search/joint_model_condhp_v5_ep100.pth

echo "=== Step 3: 验证可微解码器 ==="
python dvae_differentiable.py
# 预期：type_dim=48  edge_dim=36  p_dim=84  全部通过

echo "=== Step 4: 验证 Jacobian 工具 ==="
python jacobian_utils.py
# 预期：J.shape=(84,12)  probes.shape=(12,12)  全部通过

echo "=== Step 5: 验证评估工具 ==="
python eval_utils.py
# 预期：GCNII / GAT heads / SAGE aggr 构建均通过

echo "=== Step 6: 启动 Phase 2 架构搜索 ==="
python bo_phase2.py \
  --checkpoint "${VAE_CKPT}" \
  --cora_root "${CORA_ROOT}" \
  --n_init 10 \
  --n_iter 40 \
  --output "${PHASE2_OUT}" \
  --sigma 0.8 \
  --eval_epochs 100 \
  --patience 20 \
  --version "${VERSION}" \
  --seed 42
# → results/bo_phase2_condhp_v5/best_z_arch_final.pt

echo "=== Step 7: 启动 Phase 3 NAS + 条件 HP 搜索 ==="
python bo_phase3.py \
  --checkpoint "${VAE_CKPT}" \
  --warm_start "${PHASE2_OUT}/best_z_arch_final.pt" \
  --cora_root "${CORA_ROOT}" \
  --n_init 15 \
  --n_iter 50 \
  --output "${PHASE3_OUT}" \
  --sigma_arch 0.8 \
  --eval_epochs 100 \
  --patience 20 \
  --novelty_w 0.1 \
  --jacobian_n 12 \
  --version "${VERSION}" \
  --seed 42
# → results/bo_phase3_condhp_v5/best_z_search_final.pt

echo "=== Step 8: 启动 Phase 4 条件参数 GPND-NAS 搜索 ==="
python bo_phase4.py \
  --checkpoint "${VAE_CKPT}" \
  --warm_start "${PHASE2_OUT}/best_z_arch_final.pt" \
  --cora_root "${CORA_ROOT}" \
  --n_init 20 \
  --n_iter 60 \
  --output "${PHASE4_OUT}" \
  --sigma_arch 0.8 \
  --eval_epochs 100 \
  --patience 20 \
  --novelty_w 0.15 \
  --tau_gumbel 0.3 \
  --n_probes 12 \
  --lhs_oversample 3 \
  --gcnii_alpha 0.1 \
  --gcnii_theta 0.5 \
  --version "${VERSION}" \
  --seed 42
# → results/bo_phase4_condhp_v5/best_z_final.pt

echo "================================================================"
echo "  条件参数 v5 NAS 流水线完成"
echo "  VAE checkpoint : ${VAE_CKPT}"
echo "  Phase 2 output : ${PHASE2_OUT}"
echo "  Phase 3 output : ${PHASE3_OUT}"
echo "  Phase 4 output : ${PHASE4_OUT}"
echo "================================================================"