#!/bin/bash
# run_pipeline_v3.sh
# ================================================================
# GNN-NAS v3 完整流水线 (max_n=7, nvt=8, nz=12, GCNII)
# ================================================================
set -e
CORA_ROOT="/tmp/Cora"

echo "=== Step 1: 生成 v3 训练数据集 ==="
python generate_mini_data.py
# → data/mini_gnn_dataset_v3.pkl

echo "=== Step 2: 训练 v3 VAE ==="
# train_joint.py 中需要将数据路径更新为 v3 数据集
# 以及将 ArchArgs.nz = 12（已在 nas_space.py 中作为默认值）
python train_joint.py
# → results/joint_search/joint_model_ep100.pth

echo "=== Step 3: 验证可微解码器 ==="
python dvae_differentiable.py
# 预期：type_dim=48  edge_dim=36  p_dim=84  全部 ✅

echo "=== Step 4: 验证 Jacobian 工具 ==="
python jacobian_utils.py
# 预期：J.shape=(84,12)  probes.shape=(12,12)  全部 ✅

echo "=== Step 5: 验证评估工具 ==="
python eval_utils.py
# 预期：GCNII arch / plain arch shape 均 ✅

echo "=== Step 6: 启动 Phase 4 v3 搜索 ==="
python bo_phase4.py \
  --checkpoint results/joint_search/joint_model_ep100.pth \
  --cora_root  "${CORA_ROOT}" \
  --n_init 20 --n_iter 60 \
  --n_probes 12 \
  --gcnii_alpha 0.1 \
  --gcnii_theta 0.5 \
  --output results/bo_phase4_v3

echo "=== 完成 ==="