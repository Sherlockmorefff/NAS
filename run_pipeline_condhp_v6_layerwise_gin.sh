#!/bin/bash
# run_pipeline_condhp_v6_layerwise_gin.sh
# ================================================================
# GNN-NAS 条件参数 v6 完整流水线
# Search Space v3: max_n=7, nvt=8, nz=12, GCNII
# Layer-wise Conditional HP:
#   [lr, dropout, gat_heads_i, sage_aggr_i, gin_eps_i], i=0..4
# ================================================================
set -e

CORA_ROOT="/tmp/Cora"
VERSION="condhp_v6_layerwise_gin"
HP_DIM=17
DATA_PATH="data/mini_gnn_dataset_v6_layerwise_gin.pkl"
VAE_CKPT="results/joint_search/joint_model_${VERSION}_ep100.pth"

PHASE2_OUT="results/bo_phase2_${VERSION}"
PHASE3_OUT="results/bo_phase3_${VERSION}"
PHASE4_OUT="results/bo_phase4_${VERSION}"
FINAL_OUT="results/final_eval_${VERSION}"

echo "================================================================"
echo "  启动 GNN-NAS 条件参数 v6 完整流水线"
echo "  VERSION=${VERSION}"
echo "  HP_DIM=${HP_DIM}"
echo "  CORA_ROOT=${CORA_ROOT}"
echo "================================================================"

echo "=== Step 1: 生成 v6 逐层条件参数训练数据集 ==="
python generate_mini_data.py \
  --output "${DATA_PATH}" \
  --num_samples 3000 \
  --skip_prob 0.35 \
  --hp_dim "${HP_DIM}" \
  --seed 42

echo "=== Step 2: 训练 v6 Joint VAE ==="
python train_joint.py \
  --data "${DATA_PATH}" \
  --checkpoint_dir results/joint_search \
  --version "${VERSION}" \
  --hp_dim "${HP_DIM}" \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --beta_max 0.01 \
  --free_bits 2.0 \
  --seed 42

echo "=== Step 3: 验证可微解码器 ==="
python dvae_differentiable.py

echo "=== Step 4: 验证 Jacobian 工具 ==="
python jacobian_utils.py

echo "=== Step 5: 验证评估工具 ==="
python eval_utils.py

echo "=== Step 6: 启动 Phase 2 架构搜索 ==="
python bo_phase2.py \
  --checkpoint "${VAE_CKPT}" \
  --hp_dim "${HP_DIM}" \
  --cora_root "${CORA_ROOT}" \
  --n_init 10 \
  --n_iter 40 \
  --output "${PHASE2_OUT}" \
  --sigma 0.8 \
  --eval_epochs 100 \
  --patience 20 \
  --version "${VERSION}" \
  --seed 42

echo "=== Step 7: 启动 Phase 3 NAS + 逐层条件 HP 搜索 ==="
python bo_phase3.py \
  --checkpoint "${VAE_CKPT}" \
  --hp_dim "${HP_DIM}" \
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

echo "=== Step 8: 启动 Phase 4 条件核函数 + 逐层条件参数 GPND-NAS 搜索 ==="
python bo_phase4.py \
  --checkpoint "${VAE_CKPT}" \
  --hp_dim "${HP_DIM}" \
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
  --use_conditional_kernel \
  --cond_kernel_tau 0.3 \
  --cond_kernel_mask_weight 1.0 \
  --gcnii_alpha 0.1 \
  --gcnii_theta 0.5 \
  --version "${VERSION}" \
  --seed 42

echo "=== Step 9: 最终评估，自动读取 Phase 4 best_z_final.pt ==="
python final_eval.py \
  --cora_root "${CORA_ROOT}" \
  --n_seeds 10 \
  --eval_epochs 200 \
  --patience 30 \
  --output "${FINAL_OUT}" \
  --version "${VERSION}_final" \
  --checkpoint "${VAE_CKPT}" \
  --hp_dim "${HP_DIM}" \
  --auto_best_z "${PHASE4_OUT}/best_z_final.pt" \
  --hidden_dim 64 \
  --weight_decay 5e-4 \
  --gcnii_alpha 0.1 \
  --gcnii_theta 0.5 \
  --seed 42

echo "================================================================"
echo "  条件参数 v6 NAS 流水线完成"
echo "  VAE checkpoint : ${VAE_CKPT}"
echo "  Phase 2 output : ${PHASE2_OUT}"
echo "  Phase 3 output : ${PHASE3_OUT}"
echo "  Phase 4 output : ${PHASE4_OUT}"
echo "  Final eval out : ${FINAL_OUT}"
echo "================================================================"
