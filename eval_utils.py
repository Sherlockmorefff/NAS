"""
eval_utils.py  v3  (Search Space v3：GCNII + 5层算子支持)
==========================================================
v3 变更（相对 v2）：
  - OP_REG 加入 'GCNII' → GCN2Conv（PyG）
  - DynamicGNN：
      * 当 config 包含 GCNII 算子时，自动添加 input_proj (in_ch → hidden_dim)
        以确保 x_0（初始残差向量）与所有层的特征维度一致
      * 所有 GCNII 层以递增 layer_idx 初始化（论文推荐的 θ_l = log(θ/l+1) 依赖）
      * GCNII 层 forward 时接收 (x, x_0, edge_index)
  - 搜索空间 v3 支持最多 5 个算子节点（max_n=7 意味着 ops 列表长度最多为 5）

搜索空间向量布局（SEARCH_DIM=16）：
  [0:12]  z_arch   — 架构隐变量 (v3: nz=12)
  [12]    log_lr   — 学习率（归一化）
  [13]    dropout  — Dropout（归一化）
  [14]    hidden   — 隐藏层维度（归一化）
  [15]    l2       — L2 正则（归一化）

v2 变更保留不变（CosineAnnealingLR、patience=20 早停）。
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList, Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GCN2Conv

# ============================================================
# 评估超参
# ============================================================
MAX_EPOCHS = 150
PATIENCE   = 20
HIDDEN     = 64

# OP_REG 仅包含"标准"算子（ctor 签名为 (in_ch, out_ch)）
# GCNII 由 DynamicGNN 特殊处理（不同的 ctor 签名）
OP_REG = {
    'GCNConv':  GCNConv,
    'GATConv':  GATConv,
    'SAGEConv': SAGEConv,
    'GINConv':  GINConv,
}

# ============================================================
# 搜索空间 v2/v3：反归一化表
# ============================================================
HIDDEN_DIM_OPTIONS = [16, 32, 64, 128, 256, 512]
L2_OPTIONS         = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]


def norm_to_hidden(x: float) -> int:
    idx = min(int(x * len(HIDDEN_DIM_OPTIONS)), len(HIDDEN_DIM_OPTIONS) - 1)
    return HIDDEN_DIM_OPTIONS[idx]


def norm_to_l2(x: float) -> float:
    idx = min(int(x * len(L2_OPTIONS)), len(L2_OPTIONS) - 1)
    return L2_OPTIONS[idx]


# ============================================================
# 动态 GNN 模型（v3：支持 GCNII）
# ============================================================
class DynamicGNN(torch.nn.Module):
    """
    根据 config dict 动态构建 GNN。

    支持算子：GCNConv, GATConv, SAGEConv, GINConv, Identity, GCNII。

    GCNII 设计要点
    --------------
    GCN2Conv(channels, alpha, theta, layer) 的 forward 签名为
        forward(x, x_0, edge_index)
    其中 x 和 x_0 必须具有相同维度 `channels`。

    因此，当 config 包含 GCNII 时：
      1. 增加 input_proj (Linear: in_ch → hidden_dim)，将原始特征投影至
         统一宽度，并将其记为 x_0（初始残差锚点）。
      2. 所有后续层均以 hidden_dim 为输入/输出宽度。
      3. 对非 GCNII 的算子（GCNConv 等），prev 维度从 hidden_dim 开始，
         因此它们也以 hidden_dim 作为 in_channels，行为与原逻辑一致。

    Parameters
    ----------
    gcnii_alpha : float
        GCNII 初始残差系数 α（默认 0.1，论文推荐范围 0.1–0.5）
    gcnii_theta : float
        GCNII 恒等映射系数 θ（默认 0.5，与 layer_idx 共同决定 θ_l）
    """

    def __init__(self, config: dict, in_ch: int, out_ch: int,
                 dropout: float = 0.5, hidden_dim: int = HIDDEN,
                 gcnii_alpha: float = 0.1, gcnii_theta: float = 0.5):
        super().__init__()
        self.ops        = config['operations']
        self.dropout    = dropout
        self.hidden_dim = hidden_dim

        # ── 检测是否包含 GCNII ──────────────────────────────
        self.has_gcnii = 'GCNII' in self.ops

        # ── 若含 GCNII，需要显式输入投影到 hidden_dim ──────
        if self.has_gcnii:
            self.input_proj = Linear(in_ch, hidden_dim)
            prev = hidden_dim
        else:
            self.input_proj = None
            prev = in_ch

        # ── 构建算子层列表 ───────────────────────────────────
        self.layers      = ModuleList()
        gcnii_layer_idx  = 1   # GCNII 要求从 1 开始递增的层编号

        for op in self.ops:
            if op == 'Identity':
                self.layers.append(None)

            elif op == 'GINConv':
                mlp = Sequential(
                    Linear(prev, hidden_dim), ReLU(),
                    Linear(hidden_dim, hidden_dim))
                self.layers.append(GINConv(mlp))
                prev = hidden_dim

            elif op == 'GCNII':
                # GCN2Conv(channels, alpha, theta, layer, shared_weights)
                # 注意：channels 必须等于 prev（已通过 input_proj 保证）
                self.layers.append(GCN2Conv(
                    channels       = hidden_dim,
                    alpha          = gcnii_alpha,
                    theta          = gcnii_theta,
                    layer          = gcnii_layer_idx,
                    shared_weights = True,
                ))
                gcnii_layer_idx += 1
                prev = hidden_dim

            else:
                # 标准算子：GCNConv / GATConv / SAGEConv
                self.layers.append(OP_REG[op](prev, hidden_dim))
                prev = hidden_dim

        self.clf = Linear(prev, out_ch)

    def forward(self, x, edge_index):
        # ── GCNII 需要初始残差 x_0 ──────────────────────────
        if self.has_gcnii:
            x   = F.relu(self.input_proj(x))
            x_0 = x   # 锚点：在所有 GCNII 层中保持不变
        else:
            x_0 = None

        for layer, op in zip(self.layers, self.ops):
            if op == 'Identity' or layer is None:
                continue
            elif op == 'GCNII':
                # GCN2Conv.forward(x, x_0, edge_index)
                x = F.dropout(
                    F.relu(layer(x, x_0, edge_index)),
                    self.dropout, self.training)
            else:
                x = F.dropout(
                    F.relu(layer(x, edge_index)),
                    self.dropout, self.training)

        return self.clf(x)


# ============================================================
# 核心评估函数
# ============================================================
def train_and_eval_arch(
    config: dict,
    data,
    in_ch: int,
    out_ch: int,
    lr: float           = 1e-3,
    dropout: float      = 0.5,
    device              = None,
    hidden_dim: int     = HIDDEN,
    weight_decay: float = 5e-4,
    gcnii_alpha: float  = 0.1,
    gcnii_theta: float  = 0.5,
    max_epochs: int     = MAX_EPOCHS,
    patience: int       = PATIENCE,
    seed: int           = None,
) -> tuple:
    """
    训练一个 DynamicGNN 并返回最优验证集准确率。

    Returns
    -------
    (best_val_acc: float, is_valid: bool)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config is None or config.get('effective_layers', 0) == 0:
        return 0.0, False

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        gnn = DynamicGNN(
            config, in_ch, out_ch,
            dropout     = dropout,
            hidden_dim  = hidden_dim,
            gcnii_alpha = gcnii_alpha,
            gcnii_theta = gcnii_theta,
        ).to(device)
    except Exception as e:
        print(f"  [DynamicGNN build failed] {e}")
        return 0.0, False

    optimizer = torch.optim.Adam(
        gnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    best_val   = 0.0
    no_improve = 0

    for epoch in range(max_epochs):
        gnn.train()
        optimizer.zero_grad()
        out  = gnn(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        gnn.eval()
        with torch.no_grad():
            pred    = gnn(data.x, data.edge_index).argmax(dim=1)
            val_acc = (pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
                       / data.val_mask.sum().item())

        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val, True


# ============================================================
# 带 z_arch 解码的完整评估入口
# ============================================================
def eval_z_search(
    vae,
    z_search: torch.Tensor,
    data,
    in_ch: int,
    out_ch: int,
    arch_nz: int,
    log_lr_min: float,
    log_lr_max: float,
    dropout_min: float,
    dropout_max: float,
    device,
    has_hidden: bool    = False,
    has_l2: bool        = False,
    gcnii_alpha: float  = 0.1,
    gcnii_theta: float  = 0.5,
    n_trials: int       = 3,
    max_epochs: int     = MAX_EPOCHS,
    patience: int       = PATIENCE,
) -> tuple:
    """
    搜索空间 v3 向量布局（SEARCH_DIM=16）：
      [0:arch_nz]      z_arch              (arch_nz=12)
      [arch_nz]        log_lr_norm
      [arch_nz+1]      dropout_norm
      [arch_nz+2]      hidden_dim_norm     (has_hidden=True)
      [arch_nz+3]      l2_norm             (has_l2=True)

    Returns
    -------
    (val_acc, lr_val, dropout_val, hidden_dim_val, l2_val, is_valid)
    """
    log_lr_n = float(z_search[arch_nz])
    drop_n   = float(z_search[arch_nz + 1])
    log_lr   = log_lr_n * (log_lr_max - log_lr_min) + log_lr_min
    dropout  = drop_n   * (dropout_max - dropout_min) + dropout_min
    lr_val   = float(np.clip(10 ** log_lr, 1e-6, 1.0))
    dr_val   = float(np.clip(dropout, 0.0, 0.9))

    if has_hidden and z_search.shape[0] > arch_nz + 2:
        hidden_val = norm_to_hidden(float(z_search[arch_nz + 2]))
    else:
        hidden_val = HIDDEN

    if has_l2 and z_search.shape[0] > arch_nz + 3:
        l2_val = norm_to_l2(float(z_search[arch_nz + 3]))
    else:
        l2_val = 5e-4

    z_arch = z_search[:arch_nz]
    config = _decode_arch(vae, z_arch, device, n_trials)

    val_acc, is_valid = train_and_eval_arch(
        config, data, in_ch, out_ch,
        lr           = lr_val,
        dropout      = dr_val,
        hidden_dim   = hidden_val,
        weight_decay = l2_val,
        gcnii_alpha  = gcnii_alpha,
        gcnii_theta  = gcnii_theta,
        device       = device,
        max_epochs   = max_epochs,
        patience     = patience,
    )

    return val_acc, lr_val, dr_val, hidden_val, l2_val, is_valid


def _decode_arch(vae, z_arch: torch.Tensor, device, n_trials: int = 3):
    from collections import Counter
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(device))
            g      = graphs[0]
            ops    = [vae.op_mapping.get(g.vs[i]['type'], 'Identity')
                      for i in range(1, g.vcount()-1)]
            edges  = g.get_edgelist()
            n_eff  = sum(1 for op in ops if op != 'Identity')
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append((key, {'effective_layers': n_eff,
                                      'operations': ops, 'edges': edges}))
    if not results:
        return None
    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(c for k, c in results if k == best_key)


# ============================================================
# 快速自测
# ============================================================
if __name__ == '__main__':
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    import os

    print("eval_utils v3 self-test...")
    device = torch.device('cpu')

    # ── 测试 DynamicGNN（无需 Cora）────────────────────────
    print("\n[Unit] DynamicGNN shape test (no dataset needed)")

    # 模拟简单图数据
    from torch_geometric.data import Data
    import torch
    n_nodes   = 10
    x_dummy   = torch.randn(n_nodes, 1433)
    edge_idx  = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
    data_dummy = Data(x=x_dummy, edge_index=edge_idx)

    # 测试包含 GCNII 的架构
    config_gcnii = {
        'operations': ['GCNConv', 'GCNII', 'GCNConv', 'GCNII', 'Identity'],
        'effective_layers': 4,
    }
    gnn_gcnii = DynamicGNN(config_gcnii, in_ch=1433, out_ch=7,
                            hidden_dim=64).to(device)
    out = gnn_gcnii(data_dummy.x, data_dummy.edge_index)
    assert out.shape == (n_nodes, 7), f"GCNII arch shape: {out.shape}"
    print(f"  GCNII arch output shape: {out.shape}  ✅")

    # 测试无 GCNII 的架构（与 v2 行为一致）
    config_plain = {
        'operations': ['GCNConv', 'GATConv', 'SAGEConv', 'GINConv', 'Identity'],
        'effective_layers': 4,
    }
    gnn_plain = DynamicGNN(config_plain, in_ch=1433, out_ch=7,
                            hidden_dim=64).to(device)
    out2 = gnn_plain(data_dummy.x, data_dummy.edge_index)
    assert out2.shape == (n_nodes, 7)
    print(f"  Plain arch output shape: {out2.shape}  ✅")

    # ── 可选：Cora 上的完整训练测试 ────────────────────────
    cora_root = '/tmp/Cora'
    if os.path.exists(cora_root):
        ds   = Planetoid(root=os.path.dirname(cora_root),
                          name='Cora', transform=T.NormalizeFeatures())
        data = ds[0]
        in_ch, out_ch = ds.num_features, ds.num_classes

        # 测试 GCNII 完整训练流程
        config_test = {
            'operations': ['GCNConv', 'GCNII', 'GCNConv'],
            'effective_layers': 3,
        }
        val, ok = train_and_eval_arch(
            config_test, data, in_ch, out_ch,
            lr=0.01, dropout=0.5, hidden_dim=64,
            gcnii_alpha=0.1, gcnii_theta=0.5,
            device=device, max_epochs=50, patience=10, seed=0)
        print(f"\n  [Cora] GCNII arch: val={val:.4f}  ok={ok}")

        print(f"\n  norm_to_hidden: {[norm_to_hidden(v) for v in [0,.2,.4,.6,.8,1.]]}")
        print(f"  norm_to_l2:     {[norm_to_l2(v) for v in [0,.25,.5,.75,1.]]}")

    print("\n✅ eval_utils v3 OK")