"""
train_joint.py  v10  (Search Space v3 + layer-wise 条件参数 mask)
====================================================================
v10 变更：
  - 支持 hp_dim=17 的 layer-wise 条件参数：
      [lr, dropout, gat_heads_i, sage_aggr_i, gin_eps_i]
  - hp_dim 从数据集自动推断；hp_dim=4 的 v5 数据仍兼容。

v9 变更：
  - HP 4 维改为 [lr_norm, dropout_norm, gat_heads_norm, sage_aggr_norm]。
  - 训练时根据每个 DAG 的算子类型 mask 非活跃条件参数：
      GATConv 缺席时不监督 gat_heads，SAGEConv 缺席时不监督 sage_aggr。
  - 保留 v8 的 DVAE.loss() 智能解包修复。
"""

import os
import sys
import json
import time
import logging
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# 导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nas_space import JointSpaceVAE, condition_masks_from_graphs, HP_DIM_LAYERWISE

# ============================================================
# 日志系统
# ============================================================
def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime('%m%d_%H%M')
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)

    log_filename = f"train_{version}_{ts}.log"
    log_filepath = os.path.join(log_subdir, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(log_filepath, encoding='utf-8')
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_filepath

def save_args_json(args, log_filepath: str):
    json_path = log_filepath.replace('.log', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path

# ============================================================
# 损失函数 (v9: 条件参数 mask + 智能解包)
# ============================================================
def compute_loss(model, arch_out, mu_a, log_a, hp_out, mu_h, log_h,
                 target_graphs, target_hps, hp_mask, beta, free_bits, device):
    """
    计算联合 VAE 损失。
    """
    # 1. 架构重构与 KL 损失
    arch_loss_tuple = model.arch_vae.loss(mu_a, log_a, target_graphs)
    
    # 智能解包：大多数 DVAE 返回 (total_loss, recon_loss, kld_loss)
    if len(arch_loss_tuple) == 3:
        _, recon_arch, kl_arch_raw = arch_loss_tuple
    elif len(arch_loss_tuple) == 2:
        recon_arch, kl_arch_raw = arch_loss_tuple
    else:
        # 兜底：假设按标准约定索引 1 和 2 是 recon 和 kl
        recon_arch = arch_loss_tuple[1]
        kl_arch_raw = arch_loss_tuple[2]
    
    # 2. 条件 HP 重建损失：只监督当前架构真正生效的条件位
    recon_hp = torch.sum(((hp_out - target_hps) * hp_mask) ** 2)
    
    # 3. HP KL 散度计算
    kl_hp_raw = -0.5 * torch.sum(1 + log_h - mu_h.pow(2) - log_h.exp())
    
    bs = target_hps.size(0)
    arch_nz = mu_a.size(1)
    hp_nz   = mu_h.size(1)
    
    # 修正 Free Bits：按维度加权
    free_limit_arch = free_bits * bs * arch_nz
    free_limit_hp   = free_bits * bs * hp_nz
    
    kl_arch = torch.clamp(kl_arch_raw - free_limit_arch, min=0.0)
    kl_hp   = torch.clamp(kl_hp_raw - free_limit_hp, min=0.0)
    
    total_recon = recon_arch + recon_hp
    total_kl    = kl_arch + kl_hp
    total_loss  = total_recon + beta * total_kl
    
    return total_loss, recon_arch, recon_hp, kl_arch_raw, kl_hp_raw

# ============================================================
# 训练循环
# ============================================================
def train_one_epoch(model, loader, optimizer, beta, args, device):
    model.train()
    total_loss, total_ra, total_rh, total_ka, total_kh = 0, 0, 0, 0, 0
    
    for g_batch, hp_batch in loader:
        hp_batch = hp_batch.to(device)
        hp_mask  = condition_masks_from_graphs(
            g_batch, device=device, hp_dim=hp_batch.shape[1])
        hp_input = hp_batch * hp_mask
        optimizer.zero_grad()
        
        # 前向传播 (arch_out 在训练时自由解码的结果不参与 loss 计算)
        (arch_out, mu_a, log_a), (hp_out, mu_h, log_h) = model(g_batch, hp_input)
        
        # 计算损失
        loss, ra, rh, ka, kh = compute_loss(
            model, arch_out, mu_a, log_a, hp_out, mu_h, log_h,
            g_batch, hp_batch, hp_mask, beta, args.free_bits, device
        )
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ra   += ra.item()
        total_rh   += rh.item()
        total_ka   += ka.item()
        total_kh   += kh.item()
        
    n = len(loader)
    return total_loss/n, total_ra/n, total_rh/n, total_ka/n, total_kh/n

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='train_joint.py v10')
    parser.add_argument('--data', type=str,
                        default='data/mini_gnn_dataset_v6_layerwise_gin.pkl')
    parser.add_argument('--checkpoint_dir', type=str, default='results/joint_search')
    parser.add_argument('--version', type=str, default='v3_final')
    parser.add_argument('--log_dir', type=str, default='logs/train_joint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta_max', type=float, default=0.01)
    parser.add_argument('--free_bits', type=float, default=2.0)
    parser.add_argument('--hp_dim', type=int, default=None,
                        help='HP 维度；默认从数据集自动推断')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. 初始化日志
    logger, log_path = setup_logger(args.log_dir, 'train_joint', args.version)
    save_args_json(args, log_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info(f"train_joint.py v10 (Search Space v3 + layer-wise conditional HP mask) | Device: {device}")
    logger.info(f"Data: {args.data} | Beta Max: {args.beta_max}")
    logger.info("=" * 60)

    # 2. 加载数据
    if not os.path.exists(args.data):
        logger.error(f"数据集未找到: {args.data}，请先运行 generate_mini_data.py --hp_dim {HP_DIM_LAYERWISE}")
        return

    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)

    inferred_hp_dim = len(dataset[0][2])
    if args.hp_dim is not None and args.hp_dim != inferred_hp_dim:
        logger.error(f"--hp_dim={args.hp_dim} 与数据集 hp_dim={inferred_hp_dim} 不一致")
        return
    hp_dim = inferred_hp_dim
    logger.info(f"HP dim inferred from dataset: {hp_dim}")
    
    # 转换为 (igraph, tensor) 格式
    processed_data = []
    import igraph
    for types, adj, hp in dataset:
        g = igraph.Graph(directed=True)
        g.add_vertices(len(types))
        g.vs['type'] = types
        edges = [(i, j) for i in range(len(types)) for j in range(len(types)) if adj[i][j] == 1]
        g.add_edges(edges)
        processed_data.append((g, torch.tensor(hp, dtype=torch.float32)))

    # 划分训练集/验证集
    indices = list(range(len(processed_data)))
    np.random.seed(args.seed); np.random.shuffle(indices)
    split = int(0.9 * len(processed_data))
    train_idx, val_idx = indices[:split], indices[split:]

    def collate_fn(batch):
        return [b[0] for b in batch], torch.stack([b[1] for b in batch])

    train_loader = DataLoader(Subset(processed_data, train_idx), 
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(Subset(processed_data, val_idx), 
                              batch_size=args.batch_size, collate_fn=collate_fn)

    # 3. 初始化模型
    class ArchArgs:
        max_n = 7; num_vertex_type = 8; nz = 12; bidirectional = True
        hs = 501; START_TYPE = 0; END_TYPE = 1

    model = JointSpaceVAE(
        ArchArgs(), hp_latent_dim=hp_dim, hp_input_dim=hp_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. 训练循环
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # KL 退火逻辑
        if epoch <= 20: beta = 0.0
        elif epoch <= 80: beta = args.beta_max * (epoch - 20) / 60
        else: beta = args.beta_max

        loss, ra, rh, ka, kh = train_one_epoch(model, train_loader, optimizer, beta, args, device)
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:>3d}/{args.epochs} | Loss: {loss:.2f} | "
                        f"RA: {ra:.2f} RH: {rh:.4f} | KA: {ka:.1f} KH: {kh:.1f} | Beta: {beta:.4f}")

        # 保存 Checkpoint
        if epoch == args.epochs:
            save_path = os.path.join(args.checkpoint_dir, f"joint_model_{args.version}_ep{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"模型已保存至: {save_path}")

    logger.info("✅ 训练完成。")

if __name__ == '__main__':
    main()
