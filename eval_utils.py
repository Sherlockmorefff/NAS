"""
eval_utils.py  v5  (网络隔离自测版)
=====================================================================
v5 变更：
  - 将自测模块（__main__）中的 Test 4 改为使用随机 Dummy 数据，
    彻底切断了对 GitHub 的网络下载依赖，避免服务器 timeout 报错。
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch.nn import ModuleList, ModuleDict, Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GCN2Conv

# ============================================================
# 评估超参
# ============================================================
MAX_EPOCHS = 150
PATIENCE   = 20
HIDDEN     = 64

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
# 修改前: L2_OPTIONS = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
# 修改后:
L2_OPTIONS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

def norm_to_hidden(x: float) -> int:
    idx = min(int(x * len(HIDDEN_DIM_OPTIONS)), len(HIDDEN_DIM_OPTIONS) - 1)
    return HIDDEN_DIM_OPTIONS[idx]

def norm_to_l2(x: float) -> float:
    idx = min(int(x * len(L2_OPTIONS)), len(L2_OPTIONS) - 1)
    return L2_OPTIONS[idx]

# ============================================================
# 动态 GNN 模型
# ============================================================
class DynamicGNN(torch.nn.Module):
    def __init__(self, config: dict, in_ch: int, out_ch: int,
                 dropout: float = 0.5, hidden_dim: int = HIDDEN,
                 gcnii_alpha: float = 0.1, gcnii_theta: float = 0.5):
        super().__init__()
        self.ops        = config['operations']
        self.dropout    = dropout
        self.hidden_dim = hidden_dim
        self.in_ch      = in_ch
        self.out_ch     = out_ch

        self.edges = list(config.get('edges', []))
        n_ops = len(self.ops)
        if not self.edges:
            self.edges = [(i, i + 1) for i in range(n_ops + 1)]

        self.n_nodes = n_ops + 2
        self._predecessors = defaultdict(list)
        for src, dst in self.edges:
            self._predecessors[dst].append(src)

        self.has_gcnii = 'GCNII' in self.ops
        if self.has_gcnii:
            self.input_proj = Linear(in_ch, hidden_dim)
        else:
            self.input_proj = None

        self.layers     = ModuleList()
        gcnii_layer_idx = 1   

        node_out_dim = {0: in_ch}

        for i, op in enumerate(self.ops):
            node_idx = i + 1  
            preds    = self._predecessors.get(node_idx, [node_idx - 1])
            in_dim = node_out_dim.get(preds[0], hidden_dim) if preds else in_ch

            if op == 'Identity':
                self.layers.append(None)
                node_out_dim[node_idx] = in_dim  

            elif op == 'GINConv':
                mlp = Sequential(
                    Linear(in_dim, hidden_dim), ReLU(),
                    Linear(hidden_dim, hidden_dim))
                self.layers.append(GINConv(mlp))
                node_out_dim[node_idx] = hidden_dim

            elif op == 'GCNII':
                self.layers.append(GCN2Conv(
                    channels       = hidden_dim,
                    alpha          = gcnii_alpha,
                    theta          = gcnii_theta,
                    layer          = gcnii_layer_idx,
                    shared_weights = True,
                ))
                gcnii_layer_idx += 1
                node_out_dim[node_idx] = hidden_dim

            else:
                self.layers.append(OP_REG[op](in_dim, hidden_dim))
                node_out_dim[node_idx] = hidden_dim

        end_node_idx = self.n_nodes - 1
        end_preds    = self._predecessors.get(end_node_idx, [self.n_nodes - 2])
        end_in_dim   = node_out_dim.get(end_preds[0], hidden_dim) if end_preds else hidden_dim
        self.clf     = Linear(end_in_dim, out_ch)
        self._node_out_dim = node_out_dim

    def forward(self, x, edge_index):
        if self.has_gcnii and self.input_proj is not None:
            x_0 = F.relu(self.input_proj(x))   
        else:
            x_0 = None

        node_states: dict = {0: x}

        for i, (layer, op) in enumerate(zip(self.layers, self.ops)):
            node_idx = i + 1
            preds    = self._predecessors.get(node_idx, [node_idx - 1])

            agg = None
            for pred in preds:
                pred_feat = node_states.get(pred, x)  
                if agg is None:
                    agg = pred_feat
                else:
                    if agg.shape[1] != pred_feat.shape[1]:
                        min_dim = min(agg.shape[1], pred_feat.shape[1])
                        agg       = agg[:, :min_dim]
                        pred_feat = pred_feat[:, :min_dim]
                    agg = agg + pred_feat

            if agg is None:
                agg = x  

            if op == 'Identity' or layer is None:
                out = agg
            elif op == 'GCNII':
                if agg.shape[1] != self.hidden_dim and self.input_proj is not None:
                    agg = F.relu(self.input_proj(agg))
                out = F.dropout(
                    F.relu(layer(agg, x_0, edge_index)),
                    self.dropout, self.training)
            else:
                out = F.dropout(
                    F.relu(layer(agg, edge_index)),
                    self.dropout, self.training)

            node_states[node_idx] = out

        end_node_idx = self.n_nodes - 1
        end_preds    = self._predecessors.get(end_node_idx, [self.n_nodes - 2])
        agg_end = None
        for pred in end_preds:
            feat = node_states.get(pred, list(node_states.values())[-1])
            if agg_end is None:
                agg_end = feat
            else:
                if agg_end.shape[1] != feat.shape[1]:
                    min_dim  = min(agg_end.shape[1], feat.shape[1])
                    agg_end  = agg_end[:, :min_dim]
                    feat     = feat[:, :min_dim]
                agg_end = agg_end + feat

        if agg_end is None:
            agg_end = list(node_states.values())[-1]

        if agg_end.shape[1] != self.clf.in_features:
            if agg_end.shape[1] > self.clf.in_features:
                agg_end = agg_end[:, :self.clf.in_features]
            else:
                pad = torch.zeros(agg_end.shape[0],
                                  self.clf.in_features - agg_end.shape[1],
                                  device=agg_end.device)
                agg_end = torch.cat([agg_end, pad], dim=1)

        return self.clf(agg_end)

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
    track_test: bool    = False,
) -> tuple:
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config is None or config.get('effective_layers', 0) == 0:
        return (0.0, False, 0.0) if track_test else (0.0, False)

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
        return (0.0, False, 0.0) if track_test else (0.0, False)

    optimizer = torch.optim.Adam(
        gnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    best_val   = 0.0
    best_test  = 0.0
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

            if track_test:
                test_acc = (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
                            / data.test_mask.sum().item())

        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
            if track_test:
                best_test = test_acc   
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if track_test:
        return best_val, True, best_test
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
        track_test   = False,
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
    from torch_geometric.data import Data
    import os

    print("eval_utils v5 self-test (Network Isolated)...")
    device = torch.device('cpu')

    # ── 测试 1：拓扑重建（含跳跃边） ─────────────────────
    print("\n[Test 1] Topology skip-connection")
    n_nodes  = 10
    x_dummy  = torch.randn(n_nodes, 1433)
    edge_idx = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
    data_d   = Data(x=x_dummy, edge_index=edge_idx)

    config_skip = {
        'operations': ['GCNConv', 'SAGEConv', 'GCNConv'],
        'effective_layers': 3,
        'edges': [(0,1),(1,2),(2,3),(3,4),(1,3)],  
    }
    gnn_skip = DynamicGNN(config_skip, in_ch=1433, out_ch=7, hidden_dim=64).to(device)
    out_skip = gnn_skip(data_d.x, data_d.edge_index)
    assert out_skip.shape == (n_nodes, 7), f"Skip topo shape: {out_skip.shape}"
    print(f"  Skip-conn output shape: {out_skip.shape}  ✅")
    assert 1 in gnn_skip._predecessors[3], "Skip edge 1→3 missing!"
    print(f"  predecessors[3] = {gnn_skip._predecessors[3]}  ✅")

    # ── 测试 2：GCNII 投影不污染主数据流 ─────────────────
    print("\n[Test 2] GCNII x_0 isolation (no main stream pollution)")
    config_gcnii = {
        'operations': ['GCNConv', 'GCNII', 'GCNConv', 'GCNII', 'Identity'],
        'effective_layers': 4,
        'edges': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)],
    }
    gnn_gcnii = DynamicGNN(config_gcnii, in_ch=1433, out_ch=7, hidden_dim=64).to(device)
    assert gnn_gcnii.input_proj is not None
    assert gnn_gcnii.input_proj.in_features == 1433
    assert gnn_gcnii.input_proj.out_features == 64
    out2 = gnn_gcnii(data_d.x, data_d.edge_index)
    assert out2.shape == (n_nodes, 7), f"GCNII shape: {out2.shape}"
    print(f"  GCNII output shape: {out2.shape}  ✅")
    print(f"  input_proj: {gnn_gcnii.input_proj.in_features} → "
          f"{gnn_gcnii.input_proj.out_features}  ✅")

    # ── 测试 3：无 GCNII 时无 input_proj ─────────────────
    print("\n[Test 3] No input_proj when no GCNII")
    config_plain = {
        'operations': ['GCNConv', 'GATConv', 'SAGEConv', 'GINConv', 'Identity'],
        'effective_layers': 4,
        'edges': [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)],
    }
    gnn_plain = DynamicGNN(config_plain, in_ch=1433, out_ch=7, hidden_dim=64).to(device)
    assert gnn_plain.input_proj is None
    out3 = gnn_plain(data_d.x, data_d.edge_index)
    assert out3.shape == (n_nodes, 7)
    print(f"  Plain arch output shape: {out3.shape}  ✅")

    # ── 测试 4：train_and_eval_arch track_test ────────────
    print("\n[Test 4] train_and_eval_arch track_test=True return value")
    
    # [v5 修改] 使用纯随机数据替代网络下载，防止 timeout
    n_test = 50
    in_c, out_c = 1433, 7
    x_c = torch.randn(n_test, in_c)
    ei_c = torch.randint(0, n_test, (2, 100))
    y_c = torch.randint(0, out_c, (n_test,))
    
    tm = torch.zeros(n_test, dtype=torch.bool); tm[:20] = True
    vm = torch.zeros(n_test, dtype=torch.bool); vm[20:35] = True
    xm = torch.zeros(n_test, dtype=torch.bool); xm[35:] = True
    
    data_c = Data(x=x_c, edge_index=ei_c, y=y_c, train_mask=tm, val_mask=vm, test_mask=xm)

    config_t = {
        'operations': ['GCNConv', 'SAGEConv', 'Identity'],
        'effective_layers': 2,
        'edges': [(0,1),(1,2),(2,3),(3,4)],
    }
    
    result = train_and_eval_arch(
        config_t, data_c, in_c, out_c,
        lr=1e-3, dropout=0.5, hidden_dim=64,
        device=device, max_epochs=10, patience=5,
        seed=0, track_test=True)
    
    assert len(result) == 3, f"track_test=True should return 3-tuple: {result}"
    val, ok, test = result
    print(f"  track_test=True → val={val:.4f}  ok={ok}  test={test:.4f}  ✅")

    result2 = train_and_eval_arch(
        config_t, data_c, in_c, out_c,
        lr=1e-3, dropout=0.5, hidden_dim=64,
        device=device, max_epochs=10, patience=5,
        seed=0, track_test=False)
    
    assert len(result2) == 2, f"track_test=False should return 2-tuple: {result2}"
    print(f"  track_test=False → val={result2[0]:.4f}  ok={result2[1]}  ✅")

    print(f"\n  norm_to_hidden: {[norm_to_hidden(v) for v in [0,.2,.4,.6,.8,1.]]}")
    print(f"  norm_to_l2:     {[norm_to_l2(v) for v in [0,.25,.5,.75,1.]]}")

    print("\n✅ eval_utils v5 OK")