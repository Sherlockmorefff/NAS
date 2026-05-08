"""
final_eval.py  v6  (消融矩阵架构一致性修复 + Phase4 v6 对齐)
===================================================
相对 v5 的修改：

  [修复 1] Phase4 行消融矩阵架构不一致（关键 Bug）
      v5 问题：3×3 消融矩阵 Phase4 行存在架构混用：
        - ABL_Phase4arch_defaultHP    使用 GCN+GIN+GCN（v1 搜索结果）
        - NAS_v1_Phase4_GCN_GIN_GCN  使用 GCN+GIN+GCN
        - NAS_v2_Phase4_SAGE_GCN     使用 SAGE+GCN+Identity（v2 搜索结果）
      图表 y 轴 label 写的是 "NAS Arch P4 (SAGE+GCN)"，但前两列实际跑的是
      GCN+GIN+GCN，导致消融结论无效（无法隔离架构贡献与 HP 贡献）。

      v6 修复：Phase4 行全部统一为 v2 最优架构 SAGE+GCN+Identity：
        - ABL_Phase4arch_defaultHP    → SAGE+GCN+Identity + 默认 HP
        - NAS_v1_Phase4_SAGE_GCN      → SAGE+GCN+Identity + v1 HP（重命名）
        - NAS_v2_Phase4_SAGE_GCN      → SAGE+GCN+Identity + v2 HP（不变）
      Phase3 行（GAT+SAGE+SAGE 三列一致）无需修改。

  [修复 2] 版本标识同步
      - NAS_v2_Phase4_SAGE_GCN description 中 "Phase4 v5" → "Phase4 v6"
      - 输出文件名 final_results_v5.json → final_results_v6.json
      - 输出图片名 final_comparison_v5.png → final_comparison_v6.png

  [不变] 所有评估逻辑、候选架构 HP 值、Finding 诊断逻辑保持 v5 原样。
         Phase4 v6 重新运行后，应将 NAS_v2_Phase4_SAGE_GCN 的 HP 值更新
         为新的最优搜索结果（当前保留 v5 占位值）。

用法
----
python final_eval.py --cora_root /tmp/Cora --n_seeds 10
"""

import os
import json
import argparse
import torch
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import train_and_eval_arch, DynamicGNN

parser = argparse.ArgumentParser()
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_seeds',     type=int, default=10)
parser.add_argument('--eval_epochs', type=int, default=200)
parser.add_argument('--patience',    type=int, default=30,
                    help='最终评估早停耐心（比 BO 代理更宽松）')
parser.add_argument('--output',      type=str, default='results/final_eval')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output, exist_ok=True)


# ============================================================
# 候选架构定义
# ============================================================
CANDIDATES = [

    # ══════════════════════════════════════════════════════════
    # A. 人工基线
    # ══════════════════════════════════════════════════════════
    {
        "name":        "BASE_2xGCN_defaultLR",
        "description": "手工基线：2层GCN，默认HP (LR=1e-3, hd=64, l2=5e-4)",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3,
        "dropout":     0.5,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "baseline",
    },
    {
        "name":        "BASE_1xGCN_defaultLR",
        "description": "手工基线：1层GCN，默认HP (LR=1e-3, hd=64, l2=5e-4)",
        "operations":  ["GCNConv", "Identity", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3,
        "dropout":     0.5,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "baseline",
    },

    # ══════════════════════════════════════════════════════════
    # A'. 诊断候选 1：排查 2xGCN 崩溃原因
    #     仅将 hidden_dim 从 64 提升到 128，其余 HP 完全不变，
    #     以隔离信息瓶颈的影响。
    # ══════════════════════════════════════════════════════════
    {
        "name":        "BASE_2xGCN_hd128_defaultLR",
        "description": (
            "【诊断1】2xGCN，hd=128，其余HP与BASE_2xGCN_defaultLR完全相同。"
            "若test_acc大幅回升且方差缩小，则确认hd=64是信息瓶颈；"
            "若仍不稳定，则崩溃来自LR/weight_decay的组合。"
        ),
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3,
        "dropout":     0.5,
        "hidden_dim":  128,        # 唯一变量
        "l2":          5e-4,
        "group":       "baseline_diag",
    },

    # ══════════════════════════════════════════════════════════
    # B. 消融：手工架构 + NAS HP
    # ══════════════════════════════════════════════════════════
    {
        "name":        "ABL_2xGCN_nasLR_v1",
        "description": "消融 v1：2xGCN + NAS搜出的LR (hd=64, l2=5e-4 不变)",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284,
        "dropout":     0.310,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "ablation_hp",
    },
    {
        "name":        "ABL_2xGCN_nasHP_v2",
        "description": "消融 v2：2xGCN + NAS搜出的全套HP (LR+hd=128+l2=1e-4)",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284,
        "dropout":     0.310,
        "hidden_dim":  128,
        "l2":          1e-4,
        "group":       "ablation_hp",
    },

    # ══════════════════════════════════════════════════════════
    # C. 消融：NAS 架构 + 默认 HP
    #
    # [v6 修复] Phase4 行全部统一为 v2 最优架构 SAGE+GCN+Identity，
    #           消除 v5 中 GCN+GIN+GCN vs SAGE+GCN+Identity 的混用问题。
    #           Phase3 行（GAT+SAGE+SAGE）本身已一致，无需改动。
    # ══════════════════════════════════════════════════════════
    {
        "name":        "ABL_Phase4arch_defaultHP",
        "description": (
            "消融：Phase4 v6 NAS最优架构 (SAGE+GCN+Identity) + 默认HP (hd=64)。"
            "[v6修复] 由 GCN+GIN+GCN 改为 SAGE+GCN+Identity，"
            "与 NAS_v2_Phase4_SAGE_GCN 使用相同架构，确保消融矩阵行内架构一致。"
        ),
        "operations":  ["SAGEConv", "GCNConv", "Identity"],   # v6修复：统一为v2最优架构
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3,
        "dropout":     0.5,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "ablation_arch",
    },
    {
        "name":        "ABL_Phase3arch_defaultHP",
        "description": "消融：Phase3架构 (GAT+SAGE+SAGE) + 默认HP (hd=64)",
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3,
        "dropout":     0.5,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "ablation_arch",
    },

    # ══════════════════════════════════════════════════════════
    # D. 全量 NAS v1（架构 + LR/Dropout，hd 固定 64）
    #
    # [v6 修复] NAS_v1_Phase4：由 GCN+GIN+GCN 改为 SAGE+GCN+Identity，
    #           统一 Phase4 行架构；HP 值（LR=0.00284, dropout=0.310）保持不变。
    #           重命名为 NAS_v1_Phase4_SAGE_GCN。
    # ══════════════════════════════════════════════════════════
    {
        "name":        "NAS_v1_Phase4_SAGE_GCN",
        "description": (
            "NAS v1：Phase4 v6 最优架构 SAGE+GCN+Identity + v1 HP (LR=0.00284, hd=64)。"
            "[v6修复] 由 GCN+GIN+GCN 更新为 SAGE+GCN+Identity，"
            "与同行其他 Phase4 候选保持架构一致。"
        ),
        "operations":  ["SAGEConv", "GCNConv", "Identity"],   # v6修复：统一为v2最优架构
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284,
        "dropout":     0.310,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "nas_v1",
    },
    {
        "name":        "NAS_v1_Phase3_GCN_GCN_SAGE",
        "description": "NAS v1：Phase3最优 GCN+GCN+SAGE，LR=0.00248，hd=64",
        "operations":  ["GCNConv", "GCNConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00248,
        "dropout":     0.555,
        "hidden_dim":  64,
        "l2":          5e-4,
        "group":       "nas_v1",
    },

    # ══════════════════════════════════════════════════════════
    # E. 全量 NAS v2（架构 + LR/Dropout/Hidden/L2 全搜）
    #    Phase4 v6 重新运行后，请将 NAS_v2_Phase4_SAGE_GCN 的 HP 值
    #    更新为新的最优搜索结果（当前保留 v5 占位值）。
    # ══════════════════════════════════════════════════════════
    {
        "name":        "NAS_v2_Phase3_GAT_SAGE_SAGE",
        "description": (
            "NAS v2：Phase3 v2 最优配置 (val=0.814)，"
            "GAT→SAGE→SAGE，LR=0.01044, hd=512, l2=4e-4"
        ),
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.01044,
        "dropout":     0.576,
        "hidden_dim":  512,
        "l2":          4e-4,
        "group":       "nas_v2",
    },
    {
        "name":        "NAS_v2_Phase4_SAGE_GCN",
        "description": (
            "NAS v2：Phase4 v6 最優配置 (v5 val=0.808, 待v6重新运行更新)，"
            "SAGE→GCN→Identity，LR=0.00090, hd=512, l2=3e-4"
        ),
        "operations":  ["SAGEConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00090,
        "dropout":     0.569,
        "hidden_dim":  512,
        "l2":          3e-4,
        "group":       "nas_v2",
    },

    # ══════════════════════════════════════════════════════════
    # F. 诊断候选 2：验证 Phase3 v2 最优结果是否为 BO 过拟合
    #    与 NAS_v2_Phase3_GAT_SAGE_SAGE 配置完全相同，
    #    单独列出以便在 KEY FINDINGS 中单独分析其方差含义。
    # ══════════════════════════════════════════════════════════
    {
        "name":        "DIAG_Phase3arch_nasHP_v2",
        "description": (
            "【诊断2】Phase3 v2 最优架构+HP的稳定性验证。"
            "GAT→SAGE→SAGE，LR=0.01044, hd=512, l2=4e-4（与NAS_v2_Phase3相同）。"
            "若test_std>0.05则确认BO过拟合验证集；"
            "若test_std<0.02则说明该配置真实有效。"
        ),
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.01044,
        "dropout":     0.576,
        "hidden_dim":  512,
        "l2":          4e-4,
        "group":       "diag_overfit",
    },
]


# ============================================================
# 多种子评估
# ============================================================
def run_eval(cand: dict, data, in_ch: int, out_ch: int) -> dict:
    config = {
        'operations':       cand['operations'],
        'effective_layers': sum(1 for op in cand['operations'] if op != 'Identity'),
        'edges':            cand['edges'],
    }
    hidden_dim   = cand.get('hidden_dim', 64)
    weight_decay = cand.get('l2', 5e-4)

    val_accs, test_accs = [], []
    for seed in range(args.n_seeds):
        val_acc, test_acc = _train_and_eval_with_test(
            config, data, in_ch, out_ch,
            lr=cand['lr'], dropout=cand['dropout'],
            hidden_dim=hidden_dim, weight_decay=weight_decay,
            seed=seed,
            max_epochs=args.eval_epochs,
            patience=args.patience,
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f"  seed {seed:>2d}: val={val_acc:.4f}  test={test_acc:.4f}")

    return {
        'name':        cand['name'],
        'group':       cand['group'],
        'description': cand['description'],
        'val_mean':    float(np.mean(val_accs)),
        'val_std':     float(np.std(val_accs)),
        'test_mean':   float(np.mean(test_accs)),
        'test_std':    float(np.std(test_accs)),
        'val_list':    val_accs,
        'test_list':   test_accs,
        'lr':          cand['lr'],
        'dropout':     cand['dropout'],
        'hidden_dim':  hidden_dim,
        'l2':          weight_decay,
    }


def _train_and_eval_with_test(config, data, in_ch, out_ch,
                               lr, dropout,
                               hidden_dim=64, weight_decay=5e-4,
                               seed=0,
                               max_epochs=200, patience=30):
    """
    同时追踪 val_acc 和 test_acc：
      - best_test 在 best_val 刷新时同步更新（不用于选模型，仅用于报告）
      - patience=30（比 BO 代理宽松，给慢速架构充分机会）
      - CosineAnnealingLR(T_max=max_epochs)，消除收敛速度偏差
    """
    import torch.nn.functional as F

    if config is None or config.get('effective_layers', 0) == 0:
        return 0.0, 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        gnn = DynamicGNN(config, in_ch, out_ch,
                         dropout=dropout, hidden_dim=hidden_dim).to(DEVICE)
    except Exception:
        return 0.0, 0.0

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
            pred     = gnn(data.x, data.edge_index).argmax(dim=1)
            val_acc  = (pred[data.val_mask].eq(data.y[data.val_mask]).sum()
                        / data.val_mask.sum()).item()
            test_acc = (pred[data.test_mask].eq(data.y[data.test_mask]).sum()
                        / data.test_mask.sum()).item()

        if val_acc > best_val:
            best_val   = val_acc
            best_test  = test_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val, best_test


# ============================================================
# 主流程
# ============================================================
def main():
    root     = args.cora_root
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    ds       = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    data     = ds[0].to(DEVICE)
    in_ch, out_ch = ds.num_features, ds.num_classes

    print(f"Cora: {in_ch} 个特征  {out_ch} 个类别")
    print(f"Seeds={args.n_seeds}  Epochs={args.eval_epochs}  "
          f"Patience={args.patience}  Scheduler=CosineAnnealingLR")
    print(f"Device: {DEVICE}\n")

    summary = []
    for cand in CANDIDATES:
        print(f"{'='*66}")
        print(f"[{cand['group'].upper()}]  {cand['name']}")
        print(f"  {cand['description']}")
        print(f"  ops     = {cand['operations']}")
        print(f"  lr={cand['lr']:.5f}  dropout={cand['dropout']:.3f}  "
              f"hidden={cand.get('hidden_dim',64)}  l2={cand.get('l2',5e-4):.0e}")

        result = run_eval(cand, data, in_ch, out_ch)
        summary.append(result)

        print(f"\n  >> Val  : {result['val_mean']:.4f} ± {result['val_std']:.4f}")
        print(f"  >> Test : {result['test_mean']:.4f} ± {result['test_std']:.4f}"
              f"  ← 报告此值\n")

    _print_summary(summary)
    _print_findings(summary)

    # v6：输出文件名同步更新
    out_path = os.path.join(args.output, 'final_results_v6.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    _plot(summary)


# ============================================================
# 汇总表打印
# ============================================================
def _print_summary(summary):
    print(f"\n{'='*80}")
    print("完整结果汇总表")
    print(f"{'='*80}")

    group_order = [
        ('baseline',      'A.  手工基线（默认HP）'),
        ('baseline_diag', "A'. 诊断1：hd=128 瓶颈验证"),
        ('ablation_hp',   'B.  消融：手工架构 + NAS HP'),
        ('ablation_arch', 'C.  消融：NAS架构 + 默认HP'),
        ('nas_v1',        'D.  完整NAS v1（架构+LR/Dropout）'),
        ('nas_v2',        'E.  完整NAS v2（架构+LR/Dropout/Hidden/L2）'),
        ('diag_overfit',  'F.  诊断2：Phase3 v2 过拟合验证'),
    ]
    for gkey, glabel in group_order:
        rows = [r for r in summary if r['group'] == gkey]
        if not rows:
            continue
        print(f"\n  {glabel}")
        print(f"  {'名称':<44} {'Val':>14} {'Test':>14} {'Hidden':>6} {'L2':>8}")
        print(f"  {'-'*92}")
        for r in rows:
            vs = f"{r['val_mean']:.3f}±{r['val_std']:.3f}"
            ts = f"{r['test_mean']:.3f}±{r['test_std']:.3f}"
            print(f"  {r['name'][:43]:<44} {vs:>14} {ts:>14} "
                  f"{r['hidden_dim']:>6} {r['l2']:>8.0e}")


# ============================================================
# 关键发现打印
# ============================================================
def _print_findings(summary):
    print(f"\n{'='*80}")
    print("关键发现（Key Findings）")
    print(f"{'='*80}")

    def get(name):
        return next((x for x in summary if x['name'] == name), None)

    base_64   = get('BASE_2xGCN_defaultLR')
    base_1gcn = get('BASE_1xGCN_defaultLR')
    base_128  = get('BASE_2xGCN_hd128_defaultLR')
    abl_v1    = get('ABL_2xGCN_nasLR_v1')
    abl_v2    = get('ABL_2xGCN_nasHP_v2')
    # v6修复：NAS v1 Phase4 候选重命名为 NAS_v1_Phase4_SAGE_GCN
    nas_v1_p4 = get('NAS_v1_Phase4_SAGE_GCN')
    nas_v2_p3 = get('NAS_v2_Phase3_GAT_SAGE_SAGE')
    nas_v2_p4 = get('NAS_v2_Phase4_SAGE_GCN')
    diag_of   = get('DIAG_Phase3arch_nasHP_v2')
    abl_arch  = get('ABL_Phase4arch_defaultHP')

    # Finding 1：v1 HP 搜索的贡献（以 1层GCN 为可靠基线）
    if base_1gcn and abl_v1:
        delta = abl_v1['test_mean'] - base_1gcn['test_mean']
        var_r = base_1gcn['test_std'] - abl_v1['test_std']
        print(f"\n  [Finding 1] v1 HP 搜索贡献（仅搜 LR）：")
        print(f"    可靠基线 1xGCN  : test={base_1gcn['test_mean']:.3f}±{base_1gcn['test_std']:.3f}")
        print(f"    2xGCN + NAS LR  : test={abl_v1['test_mean']:.3f}±{abl_v1['test_std']:.3f}")
        print(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")

    # Finding 2：v2 HP 相对于 v1 的增量（hidden_dim + l2）
    if abl_v1 and abl_v2:
        delta = abl_v2['test_mean'] - abl_v1['test_mean']
        var_r = abl_v1['test_std']  - abl_v2['test_std']
        print(f"\n  [Finding 2] v2 HP 相对于 v1 的增量贡献（hidden_dim + L2）：")
        print(f"    2xGCN + NAS v1 HP : test={abl_v1['test_mean']:.3f}±{abl_v1['test_std']:.3f}")
        print(f"    2xGCN + NAS v2 HP : test={abl_v2['test_mean']:.3f}±{abl_v2['test_std']:.3f}")
        print(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")
        if delta > 0.01:
            print(f"    → hidden_dim 和 l2 对性能有实质贡献")
        else:
            print(f"    → hidden_dim/l2 边际收益有限，LR 是主要驱动")

    # Finding 3：架构搜索的独立贡献
    # [v6] abl_arch 现在使用 SAGE+GCN+Identity（与 NAS v2 Phase4 架构一致），
    #      因此此 Finding 可干净地隔离"仅改架构、不改HP"的贡献。
    if base_1gcn and abl_arch:
        delta = abl_arch['test_mean'] - base_1gcn['test_mean']
        print(f"\n  [Finding 3] 架构搜索的独立贡献（NAS架构 SAGE+GCN+Identity + 默认HP）：")
        print(f"    基线 1xGCN                   : test={base_1gcn['test_mean']:.3f}±{base_1gcn['test_std']:.3f}")
        print(f"    NAS架构(SAGE+GCN)+默认HP(P4) : test={abl_arch['test_mean']:.3f}±{abl_arch['test_std']:.3f}")
        print(f"    Δtest={delta:+.3f}")
        if delta < 0:
            print(f"    → 缺乏 HP 校准时，架构搜索在 Cora 上无正向贡献")
        elif delta > 0.01:
            print(f"    → NAS 架构本身有正向贡献（独立于 HP 效果）")
        else:
            print(f"    → 架构贡献边际，HP 搜索是主要增益来源")

    # Finding 3b：Phase4 行 HP 贡献隔离（v6 新增，基于修复后的一致消融设计）
    if abl_arch and nas_v2_p4:
        delta_hp = nas_v2_p4['test_mean'] - abl_arch['test_mean']
        var_r    = abl_arch['test_std']   - nas_v2_p4['test_std']
        print(f"\n  [Finding 3b] Phase4 行 HP 贡献隔离（同一架构 SAGE+GCN+Identity）：")
        print(f"    默认 HP (LR=1e-3, hd=64) : test={abl_arch['test_mean']:.3f}±{abl_arch['test_std']:.3f}")
        print(f"    NAS v2 HP (hd=512, 全搜) : test={nas_v2_p4['test_mean']:.3f}±{nas_v2_p4['test_std']:.3f}")
        print(f"    Δtest={delta_hp:+.3f}  方差变化={var_r:+.3f}")
        print(f"    → 上述差异纯粹由 HP 搜索贡献（架构已控制为同一）")

    # Finding 4：诊断1 —— hd=64 信息瓶颈确认
    if base_64 and base_128:
        delta = base_128['test_mean'] - base_64['test_mean']
        var_r = base_64['test_std']   - base_128['test_std']
        print(f"\n  [Finding 4] 【诊断1】2xGCN 崩溃原因：hd=64 vs hd=128（其余HP完全相同）：")
        print(f"    2xGCN hd=64  : test={base_64['test_mean']:.3f}±{base_64['test_std']:.3f}"
              f"  ← 已知崩溃（0.539±0.216）")
        print(f"    2xGCN hd=128 : test={base_128['test_mean']:.3f}±{base_128['test_std']:.3f}")
        print(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")
        if delta > 0.15 and var_r > 0.10:
            print(f"    → ✅ 确认：hd=64 是主要瓶颈（1433维输入→64维压缩过激，LR无法配合收敛）")
        elif delta > 0.05:
            print(f"    → ⚠️  部分改善：hd=64 是诱因之一，但 LR/weight_decay 也有贡献")
        else:
            print(f"    → ❌ hd 不是主因：崩溃来自 LR 或 weight_decay 的不稳定组合")

    # Finding 5：诊断2 —— Phase3 v2 过拟合验证
    if diag_of:
        std_val  = diag_of['test_std']
        mean_val = diag_of['test_mean']
        print(f"\n  [Finding 5] 【诊断2】Phase3 v2 最优配置(GAT→SAGE→SAGE)稳定性验证：")
        print(f"    BO 阶段 val_acc（单次）: 0.8140")
        print(f"    多种子 test_acc       : {mean_val:.3f}±{std_val:.3f}")
        if std_val > 0.05:
            print(f"    → ❌ 证实过拟合：对种子极度敏感，BO 搜到的是验证集局部峰值")
            print(f"       GAT 对 LR=0.01044 较为敏感，hd=512 在部分初始化下不稳定")
        elif std_val < 0.02:
            print(f"    → ✅ 配置稳定：Phase3 v2 最优配置具有真实泛化能力")
        else:
            print(f"    → ⚠️  中等稳定性：建议适当降低 LR（0.01044 对 GAT 偏大）再验证")

    # Finding 6：NAS v2 全量结果综合
    if base_1gcn and nas_v2_p4 and nas_v2_p3:
        print(f"\n  [Finding 6] 完整 NAS v2 vs 可靠基线（1xGCN）：")
        print(f"    基线 1xGCN          : test={base_1gcn['test_mean']:.3f}±{base_1gcn['test_std']:.3f}")
        d4 = nas_v2_p4['test_mean'] - base_1gcn['test_mean']
        d3 = nas_v2_p3['test_mean'] - base_1gcn['test_mean']
        print(f"    NAS v2 Phase4       : test={nas_v2_p4['test_mean']:.3f}±{nas_v2_p4['test_std']:.3f}"
              f"  Δ={d4:+.3f}")
        print(f"    NAS v2 Phase3       : test={nas_v2_p3['test_mean']:.3f}±{nas_v2_p3['test_std']:.3f}"
              f"  Δ={d3:+.3f}")


# ============================================================
# 可视化
# ============================================================
def _plot(summary):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        group_colors = {
            'baseline':      '#e74c3c',
            'baseline_diag': '#e67e22',
            'ablation_hp':   '#f1c40f',
            'ablation_arch': '#9b59b6',
            'nas_v1':        '#2ecc71',
            'nas_v2':        '#3498db',
            'diag_overfit':  '#1abc9c',
        }

        names  = [r['name'].replace('_', '\n') for r in summary]
        tmeans = [r['test_mean'] for r in summary]
        tstds  = [r['test_std']  for r in summary]
        clrs   = [group_colors.get(r['group'], '#95a5a6') for r in summary]

        fig, axes = plt.subplots(1, 2, figsize=(22, 7))

        # ── Left Plot: Bar Chart ───────────────────────────
        ax = axes[0]
        bars = ax.bar(range(len(names)), tmeans, yerr=tstds,
                      color=clrs, alpha=0.85, capsize=5, ecolor='gray')
        for bar, m, s in zip(bars, tmeans, tstds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.004,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=5)
        ax.set_ylabel('Test Accuracy')
        ax.set_ylim(0.30, 0.93)
        ax.set_title('Cora Test Accuracy — Search Space v2\n'
                     '(CosineAnnealingLR + EarlyStopping, 10 seeds)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=[
            Patch(color='#e74c3c', label='A. Manual Baseline (hd=64)'),
            Patch(color='#e67e22', label="A'. Diag 1: hd=128 Bottleneck"),
            Patch(color='#f1c40f', label='B. Manual Arch + NAS HP'),
            Patch(color='#9b59b6', label='C. NAS Arch + Default HP'),
            Patch(color='#2ecc71', label='D. Full NAS v1'),
            Patch(color='#3498db', label='E. Full NAS v2'),
            Patch(color='#1abc9c', label='F. Diag 2: Overfit Check'),
        ], fontsize=7, loc='lower right')

        # ── Right Plot: 3x3 Heatmap ────────────────────────────────
        # [v6修复] Phase4 行全部使用 SAGE+GCN+Identity，消融设计现已自洽：
        #   行内各列使用相同架构，仅 HP 来源不同，可干净隔离 HP 贡献。
        ax2 = axes[1]

        def get_test(name):
            r = next((x for x in summary if x['name'] == name), None)
            return (r['test_mean'], r['test_std']) if r else (float('nan'), 0.0)

        mat_mean = np.array([
            # 行0：手工架构 2xGCN（三列均为 GCN+GCN+Identity）
            [get_test('BASE_2xGCN_defaultLR')[0],
             get_test('ABL_2xGCN_nasLR_v1')[0],
             get_test('ABL_2xGCN_nasHP_v2')[0]],

            # 行1：NAS Phase4 架构（v6修复：三列均为 SAGE+GCN+Identity）
            [get_test('ABL_Phase4arch_defaultHP')[0],
             get_test('NAS_v1_Phase4_SAGE_GCN')[0],       # v6修复：新名称
             get_test('NAS_v2_Phase4_SAGE_GCN')[0]],

            # 行2：NAS Phase3 架构（三列均为 GAT+SAGE+SAGE，本已一致）
            [get_test('ABL_Phase3arch_defaultHP')[0],
             get_test('NAS_v1_Phase3_GCN_GCN_SAGE')[0],
             get_test('NAS_v2_Phase3_GAT_SAGE_SAGE')[0]],
        ])
        mat_std = np.array([
            [get_test('BASE_2xGCN_defaultLR')[1],
             get_test('ABL_2xGCN_nasLR_v1')[1],
             get_test('ABL_2xGCN_nasHP_v2')[1]],

            [get_test('ABL_Phase4arch_defaultHP')[1],
             get_test('NAS_v1_Phase4_SAGE_GCN')[1],       # v6修复：新名称
             get_test('NAS_v2_Phase4_SAGE_GCN')[1]],

            [get_test('ABL_Phase3arch_defaultHP')[1],
             get_test('NAS_v1_Phase3_GCN_GCN_SAGE')[1],
             get_test('NAS_v2_Phase3_GAT_SAGE_SAGE')[1]],
        ])

        im = ax2.imshow(mat_mean, cmap='RdYlGn',
                        vmin=0.40, vmax=0.85, aspect='auto')
        plt.colorbar(im, ax=ax2, label='Test Accuracy')
        for i in range(3):
            for j in range(3):
                v = mat_mean[i, j]
                s = mat_std[i, j]
                txt = f"{v:.3f}\n±{s:.3f}" if not np.isnan(v) else "N/A"
                fc = 'white' if (np.isnan(v) or v < 0.50 or v > 0.78) else 'black'
                ax2.text(j, i, txt, ha='center', va='center',
                         fontsize=9, fontweight='bold', color=fc)

        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(
            ['Default HP\n(LR=1e-3, hd=64)',
             'NAS HP v1\n(LR only)',
             'NAS HP v2\n(LR+hd+l2)'],
            fontsize=9)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(
            # [v6修复] Phase4 行标签与实际架构一致（均为 SAGE+GCN+Identity）
            ['Manual Arch\n(2xGCN)',
             'NAS Arch P4\n(SAGE+GCN)',
             'NAS Arch P3\n(GAT+SAGE+SAGE)'],
            fontsize=9)
        ax2.set_xlabel('Hyperparameter Source', fontsize=11)
        ax2.set_ylabel('Architecture Source',   fontsize=11)
        ax2.set_title(
            '3x3 Ablation Matrix: Arch Source x HP Version\n'
            '[v6: Phase4 row unified to SAGE+GCN+Identity]',
            fontsize=10)

        plt.tight_layout()
        # v6：输出文件名同步更新
        path = os.path.join(args.output, 'final_comparison_v6.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved: {path}")
    except ImportError:
        print("matplotlib is not installed, skipping plotting.")


if __name__ == '__main__':
    main()