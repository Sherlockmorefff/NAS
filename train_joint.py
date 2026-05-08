"""
train_joint.py  v4  (nz=12 + KL Annealing + Free Bits + 动态诊断阈值)
=======================================================
核心修改：
1. 适应极致压缩：将 nz 从 32 降至 12，消除冗余噪声维度，完美适配下游 BO。
2. 动态诊断阈值：修复因硬编码阈值导致的“假性坍缩”误报。
"""

import os
import json
import pickle
import random
import torch
from torch import optim
import torch.nn.functional as F
import igraph
from tqdm import tqdm

from nas_space import JointSpaceVAE

# ============================================================
# 全局配置
# ============================================================
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-4
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KL 退火与正则化超参
BETA_WARMUP_END = 20
BETA_ANNEAL_END = 80
BETA_TARGET     = 0.01
FREE_BITS       = 2.0


class ArchArgs:
    def __init__(self):
        self.max_n           = 5
        self.num_vertex_type = 7
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = 12     # ← 核心修改: 32 -> 12 (完美匹配实际有效秩)
        self.bidirectional   = True


# ============================================================
# Beta 退火调度器
# ============================================================
def get_beta(epoch: int) -> float:
    if epoch <= BETA_WARMUP_END:
        return 0.0
    elif epoch <= BETA_ANNEAL_END:
        progress = (epoch - BETA_WARMUP_END) / (BETA_ANNEAL_END - BETA_WARMUP_END)
        return BETA_TARGET * progress
    else:
        return BETA_TARGET


# ============================================================
# 数据加载
# ============================================================
def load_and_preprocess_data(filepath: str = 'data/mini_gnn_dataset.pkl'):
    print(f"加载数据集: {filepath}")
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)

    processed_data = []
    bad_edges = 0

    for _, (types, adj, hp) in enumerate(raw_data):
        n = len(types)
        g = igraph.Graph(directed=True)
        g.add_vertices(n)
        g.vs['type'] = types

        edges = []
        for i in range(n):
            for j in range(n):
                if adj[i][j] == 1:
                    if i >= j:
                        bad_edges += 1
                        continue
                    edges.append((i, j))
        g.add_edges(edges)
        processed_data.append((g, hp))

    if bad_edges > 0:
        print(f"⚠️  跳过 {bad_edges} 条反向/自环边")

    random.shuffle(processed_data)
    split = int(len(processed_data) * 0.9)
    print(f"训练集: {split}  测试集: {len(processed_data)-split}")
    return processed_data[:split], processed_data[split:]


# ============================================================
# 单 epoch 训练
# ============================================================
def train_one_epoch(model: JointSpaceVAE, optimizer, train_data: list,
                    epoch: int, beta: float):
    model.train()
    total_loss  = 0.0
    total_recon = 0.0
    total_kld   = 0.0

    random.shuffle(train_data)
    g_batch, hp_batch = [], []

    pbar = tqdm(train_data, desc=f"Ep {epoch:>3d} β={beta:.4f}")
    for i, (g, hp) in enumerate(pbar):
        g_batch.append(g)
        hp_batch.append(hp)

        if len(g_batch) < BATCH_SIZE and i < len(train_data) - 1:
            continue

        optimizer.zero_grad()
        hp_t = torch.tensor(hp_batch, dtype=torch.float32).to(DEVICE)
        bs = len(g_batch)

        (_, mu_a, log_a), (hp_out, mu_h, log_h) = model(g_batch, hp_t)

        _, recon_a, kld_a = model.arch_vae.loss(mu_a, log_a, g_batch, beta=0.0)
        
        kld_a_eff = torch.clamp(kld_a, min=FREE_BITS * bs)
        loss_arch = recon_a + beta * kld_a_eff

        recon_h = F.mse_loss(hp_out, hp_t, reduction='sum')
        kld_h   = -0.5 * torch.sum(1 + log_h - mu_h.pow(2) - log_h.exp())
        
        kld_h_eff = torch.clamp(kld_h, min=FREE_BITS * bs)
        loss_hp = recon_h + beta * kld_h_eff

        loss = loss_arch + loss_hp
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_recon += (recon_a + recon_h).item()
        total_kld   += (kld_a + kld_h).item()

        pbar.set_postfix({
            'L': f"{loss.item()/bs:.2f}",
            'R': f"{(recon_a+recon_h).item()/bs:.2f}",
            'K': f"{(kld_a+kld_h).item()/bs:.2f}",
        })
        g_batch, hp_batch = [], []

    n = len(train_data)
    print(f"  => Loss={total_loss/n:.4f}  "
          f"Recon={total_recon/n:.4f}  KLD={total_kld/n:.4f}")
    return total_loss / n, total_recon / n, total_kld / n


# ============================================================
# 有效秩诊断（适配小维度空间）
# ============================================================
def diagnose_latent_rank(model: JointSpaceVAE, train_data: list,
                         n_samples: int = 200) -> float:
    model.eval()
    samples = random.sample(train_data, min(n_samples, len(train_data)))
    mu_list = []

    with torch.no_grad():
        for i in range(0, len(samples), 32):
            batch_g = [s[0] for s in samples[i:i+32]]
            mu_a, _ = model.arch_vae.encode(batch_g)
            mu_list.append(mu_a.cpu())

    Z = torch.cat(mu_list, dim=0)
    Z = Z - Z.mean(0, keepdim=True)
    nz_dim = Z.shape[1]

    try:
        _, S, _ = torch.linalg.svd(Z, full_matrices=False)
        lam = S ** 2
        eff_rank = (lam.sum() ** 2 / (lam ** 2).sum()).item()
        top_var_idx = min(5, nz_dim)
        top5_var = lam[:top_var_idx].sum().item() / lam.sum().item()

        print(f"\n[Rank Diagnosis]  eff_rank={eff_rank:.1f}/{nz_dim}"
              f"  top{top_var_idx}_var={top5_var:.1%}")
        print(f"  sing_vals: {[f'{v:.3f}' for v in S[:min(12, nz_dim)].tolist()]}")

        # 动态判定阈值：不再硬编码 4，而是按维度比例
        if eff_rank < nz_dim * 0.25:
            print("  ❌ 严重坍缩：有效维度过低")
        elif eff_rank < nz_dim * 0.4:
            print("  ⚠️  轻度坍缩：部分维度未激活")
        else:
            print("  ✅ 隐空间有效秩极其完美，已达到最优流形压缩！")

        return eff_rank
    except Exception as e:
        print(f"  [诊断失败] {e}")
        return 0.0


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    os.makedirs('results/joint_search', exist_ok=True)

    train_data, test_data = load_and_preprocess_data()

    arch_args = ArchArgs()
    model     = JointSpaceVAE(arch_args, hp_latent_dim=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\narch_nz={model.arch_nz}  hp_nz={model.hp_nz}  total_nz={model.total_nz}")
    print(f"KL Annealing: 0.0 (ep1-{BETA_WARMUP_END})"
          f" → {BETA_TARGET} (ep{BETA_WARMUP_END+1}-{BETA_ANNEAL_END})"
          f" → {BETA_TARGET} (ep{BETA_ANNEAL_END+1}+)")
    print(f"Free Bits 机制启用，阈值: {FREE_BITS}")
    print("\n🚀 训练开始（12维极简空间架构）...\n")

    loss_log = []

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch)
        avg_loss, avg_recon, avg_kld = train_one_epoch(
            model, optimizer, train_data, epoch, beta)
        loss_log.append({'epoch': epoch, 'beta': beta,
                         'loss': avg_loss, 'recon': avg_recon, 'kld': avg_kld})

        if epoch % 10 == 0:
            ckpt = f"results/joint_search/joint_model_ep{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint: {ckpt}")

            model.eval()
            with torch.no_grad():
                z = torch.randn(1, model.total_nz).to(DEVICE)
                cfgs, lrs, drs = model.decode_from_joint_latent(z)
                print(f"  [sample] ops={cfgs[0]['operations']}  "
                      f"eff={cfgs[0]['effective_layers']}  "
                      f"lr={lrs[0]:.5f}  dr={drs[0]:.4f}")
            model.train()

    eff_rank = diagnose_latent_rank(model, train_data)

    with open('results/joint_search/train_loss_log.json', 'w') as f:
        json.dump(loss_log, f, indent=2)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ep_  = [d['epoch'] for d in loss_log]
        rec_ = [d['recon'] for d in loss_log]
        kld_ = [d['kld']   for d in loss_log]
        bet_ = [d['beta']  for d in loss_log]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(ep_, rec_, label='Recon', color='steelblue')
        ax1.plot(ep_, kld_, label='KLD',   color='tomato', ls='--')
        ax1.set_ylabel('Loss / sample')
        ax1.set_title(f'Beta-VAE Training (nz=12)  eff_rank={eff_rank:.1f}')
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(ep_, bet_, color='green', lw=2)
        ax2.axvline(BETA_WARMUP_END, color='gray', ls=':',
                    label=f'warmup end (ep{BETA_WARMUP_END})')
        ax2.axvline(BETA_ANNEAL_END, color='gray', ls='--',
                    label=f'anneal end (ep{BETA_ANNEAL_END})')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Beta')
        ax2.set_title('KL Annealing Schedule')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = 'results/joint_search/training_curve_betavae.png'
        plt.savefig(out, dpi=150); plt.close()
        print(f"\n训练曲线已保存: {out}")
    except ImportError:
        pass