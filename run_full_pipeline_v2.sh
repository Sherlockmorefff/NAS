#!/bin/bash
# ==============================================================================
# GNN-NAS 完整搜索与评估流水线 (Phase 4 + Final Eval)
# 版本: Search Space v2 + 舒尔补 + 消融对齐 v6
# ==============================================================================

# 遇到错误立即停止
set -e

# ==================== 1. 全局环境变量与路径配置 ====================
export CUDA_VISIBLE_DEVICES=0                  # 指定使用的 GPU
DATA_ROOT="/tmp/Cora"                          # 数据集路径

# 【已修复】修改为真实存在且训练最充分的 100 epoch 权重文件
VAE_CKPT="results/joint_search/joint_model_ep100.pth" 

PHASE4_OUT="results/bo_phase4_schur"           # Phase 4 搜索结果输出目录
FINAL_OUT="results/final_eval_schur_v6"        # 最终评估输出目录

echo "============================================================"
echo "🚀 启动 GNN-NAS 完整流水线 (BO 搜索 -> 最终独立评估)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ==================== 2. 执行 Phase 4: 贝叶斯联合搜索 ====================
echo -e "\n>>> [阶段 1/2] 正在运行 Phase 4 (舒尔补冷启动 + GPND-NAS 探索)..."
mkdir -p ${PHASE4_OUT}

python bo_phase4.py \
    --checkpoint ${VAE_CKPT} \
    --cora_root ${DATA_ROOT} \
    --output ${PHASE4_OUT} \
    --n_init 20 \
    --n_iter 60 \
    --lhs_oversample 5 \
    --novelty_w 0.15 \
    --sigma_arch 0.8 \
    --eval_epochs 100 \
    --gpnd_alpha 0.2 \
    --gpnd_beta 0.3 \
    --gpnd_gamma 0.3 \
    --gpnd_delta 0.2 2>&1 | tee ${PHASE4_OUT}/search.log

echo -e "\n✅ Phase 4 搜索完成！"

# ==================== 3. 智能交互暂停 (Human-in-the-loop) ====================
echo "============================================================"
echo "⚠️  【需要人工介入】搜索阶段已结束！"
echo "请查看上方日志中打印出的 'Phase 4 v5... Final' 结果框。"
echo "请将最新搜出的以下 4 个超参数："
echo "  - LR"
echo "  - Dropout"
echo "  - Hidden"
echo "  - L2"
echo "手动修改并保存到 final_eval.py 文件的 NAS_v2_Phase4_SAGE_GCN 字典中。"
echo "============================================================"

# 脚本在此处暂停，等待用户按下回车键
read -p "修改并保存 final_eval.py 后，请按 [Enter] 键继续执行最终评估..."

# ==================== 4. 执行 Final Eval: 多种子独立评估 ====================
echo -e "\n>>> [阶段 2/2] 正在运行 Final Eval (10 Seeds 严谨验证)..."
mkdir -p ${FINAL_OUT}

python final_eval.py \
    --cora_root ${DATA_ROOT} \
    --n_seeds 10 \
    --eval_epochs 200 \
    --patience 30 \
    --output ${FINAL_OUT} 2>&1 | tee ${FINAL_OUT}/final_evaluation_v6.log

# ==================== 5. 流程结束与归档提示 ====================
echo "============================================================"
echo "🎉 完整流水线全部执行完毕！"
echo "最终核心产出物路径："
echo "  - 搜索日志    : ${PHASE4_OUT}/search.log"
echo "  - 评估汇总JSON: ${FINAL_OUT}/final_results_v6.json"
echo "  - 3x3消融图表 : ${FINAL_OUT}/final_comparison_v6.png"
echo "============================================================"