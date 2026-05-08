"""
final_eval.py  v7  (架构名解耦 + eval_utils 统一 + GCNII 透传 + 工业级日志)
============================================================================
v7 变更（相对 v6）：

  [修复 1] 消除硬编码架构名查询（消融矩阵安全性）
      v6 问题：_print_findings 中大量使用
               get('NAS_v1_Phase4_SAGE_GCN') 等字符串查询，
               一旦 CANDIDATES 里的名称发生变更（重命名/重组）则静默失效。
      v7 修复：引入 get_best_by_group(summary, group) 和
               get_nth_by_group(summary, group, n)，
               所有发现分析全部通过 group 属性安全提取数据。
               _plot 的 3×3 热图也改为按 group + 位置索引动态取值。

  [修复 2] 透传 GCNII 参数（run_eval 解耦）
      v6 问题：run_eval 未提取 cand 里的 gcnii_alpha/gcnii_theta，
               所有候选统一使用 eval_utils 默认值（0.1/0.5），
               含 GCNII 算子的候选无法自定义这两个关键参数。
      v7 修复：run_eval 安全提取 cand.get('gcnii_alpha', 0.1) 等，
               并透传给 train_and_eval_arch。

  [修复 3] 删除冗余 _train_and_eval_with_test
      v6 问题：_train_and_eval_with_test 与 eval_utils.train_and_eval_arch
               存在大量重复代码（早停 / CosineAnnealing / test 追踪）。
      v7 修复：直接调用 train_and_eval_arch(track_test=True)，
               删除 _train_and_eval_with_test，消除代码双份维护风险。

  [新增 4] 工业级日志系统
      - setup_logger 控制台 + 文件双端输出
      - 日志路径：{log_dir}/final_eval/train_{version}_{MMDD_HHMM}.log
      - 新增参数：--version, --log_dir, --seed,
                  --weight_decay, --hidden_dim (占位参数已在各候选中定义),
                  --gcnii_alpha, --gcnii_theta
      - 完整参数 .json 持久化
      - 启动打印所有参数，结束打印总耗时
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, '/mnt/project')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import train_and_eval_arch, DynamicGNN


# ============================================================
# 日志系统
# ============================================================
def setup_logger(log_dir: str, script_name: str, version: str):
    ts        = datetime.now().strftime('%m%d_%H%M')
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)

    log_filename = f"train_{version}_{ts}.log"
    log_filepath = os.path.join(log_subdir, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_filepath, encoding='utf-8')
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_filepath}")
    return logger, log_filepath


def save_args_json(args, log_filepath: str) -> str:
    json_path = log_filepath.replace('.log', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path


# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description='final_eval v7')
# 原有参数（不变）
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_seeds',     type=int, default=10)
parser.add_argument('--eval_epochs', type=int, default=200)
parser.add_argument('--patience',    type=int, default=30)
parser.add_argument('--output',      type=str, default='results/final_eval')
# 新增参数
parser.add_argument('--version',     type=str, default='v3_final')
parser.add_argument('--log_dir',     type=str, default='logs/')
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='全局默认 weight_decay（各候选中若有 l2 字段则优先使用）')
parser.add_argument('--hidden_dim',  type=int, default=64,
                    help='全局默认 hidden_dim（各候选中若有 hidden_dim 字段则优先）')
parser.add_argument('--gcnii_alpha', type=float, default=0.1)
parser.add_argument('--gcnii_theta', type=float, default=0.5)
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output, exist_ok=True)

logger, log_filepath = setup_logger(args.log_dir, 'final_eval', args.version)
save_args_json(args, log_filepath)


# ============================================================
# 候选架构定义（结构不变，仅新增 gcnii_alpha/theta 占位字段）
# ============================================================
CANDIDATES = [
    # ══ A. 人工基线 ══════════════════════════════════════════
    {
        "name":        "BASE_2xGCN_defaultLR",
        "description": "手工基线：2层GCN，默认HP",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3, "dropout": 0.5, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "baseline",
    },
    {
        "name":        "BASE_1xGCN_defaultLR",
        "description": "手工基线：1层GCN，默认HP",
        "operations":  ["GCNConv", "Identity", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3, "dropout": 0.5, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "baseline",
    },

    # ══ A'. 诊断候选 1：hd=64 信息瓶颈验证 ════════════════════
    {
        "name":        "BASE_2xGCN_hd128_defaultLR",
        "description": "【诊断1】2xGCN，hd=128，其余HP同BASE_2xGCN_defaultLR",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3, "dropout": 0.5, "hidden_dim": 128, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "baseline_diag",
    },

    # ══ B. 消融：手工架构 + NAS HP ════════════════════════════
    {
        "name":        "ABL_2xGCN_nasLR_v1",
        "description": "消融 v1：2xGCN + NAS搜出的LR",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284, "dropout": 0.310, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "ablation_hp",
    },
    {
        "name":        "ABL_2xGCN_nasHP_v2",
        "description": "消融 v2：2xGCN + NAS全套HP",
        "operations":  ["GCNConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284, "dropout": 0.310, "hidden_dim": 128, "l2": 1e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "ablation_hp",
    },

    # ══ C. 消融：NAS架构 + 默认HP ══════════════════════════════
    {
        "name":        "ABL_Phase4arch_defaultHP",
        "description": "消融：Phase4 v7 NAS最优架构 (SAGE+GCN+Identity) + 默认HP",
        "operations":  ["SAGEConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3, "dropout": 0.5, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "ablation_arch",
    },
    {
        "name":        "ABL_Phase3arch_defaultHP",
        "description": "消融：Phase3架构 (GAT+SAGE+SAGE) + 默认HP",
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          1e-3, "dropout": 0.5, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "ablation_arch",
    },

    # ══ D. 全量 NAS v1（架构 + LR/Dropout）════════════════════
    {
        "name":        "NAS_v1_Phase4_SAGE_GCN",
        "description": "NAS v1：Phase4最优架构SAGE+GCN+Identity + v1 HP",
        "operations":  ["SAGEConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00284, "dropout": 0.310, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "nas_v1",
    },
    {
        "name":        "NAS_v1_Phase3_GCN_GCN_SAGE",
        "description": "NAS v1：Phase3最优 GCN+GCN+SAGE，LR=0.00248，hd=64",
        "operations":  ["GCNConv", "GCNConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00248, "dropout": 0.555, "hidden_dim": 64, "l2": 5e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "nas_v1",
    },

    # ══ E. 全量 NAS v2（架构 + LR/Dropout/Hidden/L2 全搜）═════
    {
        "name":        "NAS_v2_Phase3_GAT_SAGE_SAGE",
        "description": "NAS v2：Phase3 v2 最优 GAT→SAGE→SAGE，LR=0.01044",
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.01044, "dropout": 0.576, "hidden_dim": 512, "l2": 4e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "nas_v2",
    },
    {
        "name":        "NAS_v2_Phase4_SAGE_GCN",
        "description": "NAS v2：Phase4 v7 最优 SAGE→GCN→Identity，LR=0.00090",
        "operations":  ["SAGEConv", "GCNConv", "Identity"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.00090, "dropout": 0.569, "hidden_dim": 512, "l2": 3e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "nas_v2",
    },

    # ══ F. 诊断候选 2：Phase3 v2 过拟合验证 ════════════════════
    {
        "name":        "DIAG_Phase3arch_nasHP_v2",
        "description": "【诊断2】Phase3 v2 最优配置稳定性验证",
        "operations":  ["GATConv", "SAGEConv", "SAGEConv"],
        "edges":       [(0,1),(1,2),(2,3),(3,4)],
        "lr":          0.01044, "dropout": 0.576, "hidden_dim": 512, "l2": 4e-4,
        "gcnii_alpha": 0.1,  "gcnii_theta": 0.5,
        "group":       "diag_overfit",
    },
]


# ============================================================
# Group-based 安全查询函数（v7 核心：消除硬编码名称）
# ============================================================
def get_best_by_group(summary: list, group: str):
    """
    按 group 字段查询 test_mean 最高的候选。
    替代原有的 get(hardcoded_name) 查询方式。
    """
    rows = [r for r in summary if r.get('group') == group]
    if not rows:
        return None
    return max(rows, key=lambda r: r.get('test_mean', 0.0))


def get_nth_by_group(summary: list, group: str, n: int):
    """
    按 group 字段查询第 n 个候选（0-indexed，按 CANDIDATES 顺序）。
    用于 3×3 消融矩阵等需要指定位置的场景。
    """
    rows = [r for r in summary if r.get('group') == group]
    return rows[n] if len(rows) > n else None


def get_all_by_group(summary: list, group: str) -> list:
    """按 group 字段返回该组所有候选（按 CANDIDATES 顺序）。"""
    return [r for r in summary if r.get('group') == group]


def _safe_test(entry) -> tuple:
    """安全提取 (test_mean, test_std)，若 entry 为 None 返回 (nan, 0)。"""
    if entry is None:
        return float('nan'), 0.0
    return entry.get('test_mean', float('nan')), entry.get('test_std', 0.0)


# ============================================================
# 多种子评估（v7：删除 _train_and_eval_with_test，直接调用 eval_utils）
# ============================================================
def run_eval(cand: dict, data, in_ch: int, out_ch: int) -> dict:
    """
    多种子评估单个候选架构。

    v7 修复：
      - 完全删除 _train_and_eval_with_test（与 eval_utils 重复）
      - 直接调用 train_and_eval_arch(track_test=True) 获取 val+test
      - 从 cand 字典安全提取 gcnii_alpha/gcnii_theta（v7 透传修复）
    """
    config = {
        'operations':       cand['operations'],
        'effective_layers': sum(1 for op in cand['operations'] if op != 'Identity'),
        'edges':            cand['edges'],
    }
    hidden_dim   = cand.get('hidden_dim', args.hidden_dim)
    weight_decay = cand.get('l2', args.weight_decay)
    gcnii_alpha  = cand.get('gcnii_alpha', args.gcnii_alpha)   # v7 透传
    gcnii_theta  = cand.get('gcnii_theta', args.gcnii_theta)   # v7 透传

    val_accs, test_accs = [], []

    for seed in range(args.n_seeds):
        # v7：直接调用 eval_utils 的 train_and_eval_arch(track_test=True)
        val_acc, is_valid, test_acc = train_and_eval_arch(
            config       = config,
            data         = data,
            in_ch        = in_ch,
            out_ch       = out_ch,
            lr           = cand['lr'],
            dropout      = cand['dropout'],
            hidden_dim   = hidden_dim,
            weight_decay = weight_decay,
            gcnii_alpha  = gcnii_alpha,
            gcnii_theta  = gcnii_theta,
            device       = DEVICE,
            max_epochs   = args.eval_epochs,
            patience     = args.patience,
            seed         = seed,
            track_test   = True,
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        logger.info(f"  seed {seed:>2d}: val={val_acc:.4f}  test={test_acc:.4f}"
                    f"  [{'valid' if is_valid else 'INVALID'}]")

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
        'gcnii_alpha': gcnii_alpha,
        'gcnii_theta': gcnii_theta,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    root     = args.cora_root
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    ds       = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    data     = ds[0].to(DEVICE)
    in_ch, out_ch = ds.num_features, ds.num_classes

    logger.info(f"Cora: {in_ch} 个特征  {out_ch} 个类别")
    logger.info(f"Seeds={args.n_seeds}  Epochs={args.eval_epochs}  "
                f"Patience={args.patience}  Scheduler=CosineAnnealingLR")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"gcnii_alpha(default)={args.gcnii_alpha}  "
                f"gcnii_theta(default)={args.gcnii_theta}")

    summary = []
    for cand in CANDIDATES:
        logger.info(f"\n{'='*66}")
        logger.info(f"[{cand['group'].upper()}]  {cand['name']}")
        logger.info(f"  {cand['description']}")
        logger.info(f"  ops={cand['operations']}")
        logger.info(f"  lr={cand['lr']:.5f}  dropout={cand['dropout']:.3f}  "
                    f"hidden={cand.get('hidden_dim',64)}  l2={cand.get('l2',5e-4):.0e}  "
                    f"gcnii_α={cand.get('gcnii_alpha',0.1)}  "
                    f"gcnii_θ={cand.get('gcnii_theta',0.5)}")

        result = run_eval(cand, data, in_ch, out_ch)
        summary.append(result)

        logger.info(f"\n  >> Val  : {result['val_mean']:.4f} ± {result['val_std']:.4f}")
        logger.info(f"  >> Test : {result['test_mean']:.4f} ± {result['test_std']:.4f}"
                    f"  ← 报告此值")

    _print_summary(summary)
    _print_findings(summary)

    out_path = os.path.join(args.output, 'final_results_v7.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {out_path}")

    _plot(summary)


# ============================================================
# 汇总表打印
# ============================================================
def _print_summary(summary):
    logger.info(f"\n{'='*80}")
    logger.info("完整结果汇总表")
    logger.info(f"{'='*80}")

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
        rows = get_all_by_group(summary, gkey)
        if not rows:
            continue
        logger.info(f"\n  {glabel}")
        logger.info(f"  {'名称':<44} {'Val':>14} {'Test':>14} {'Hidden':>6} {'L2':>8}")
        logger.info(f"  {'-'*92}")
        for r in rows:
            vs = f"{r['val_mean']:.3f}±{r['val_std']:.3f}"
            ts = f"{r['test_mean']:.3f}±{r['test_std']:.3f}"
            logger.info(f"  {r['name'][:43]:<44} {vs:>14} {ts:>14} "
                        f"{r['hidden_dim']:>6} {r['l2']:>8.0e}")


# ============================================================
# 关键发现打印（v7：全部使用 group-based 安全查询）
# ============================================================
def _print_findings(summary):
    logger.info(f"\n{'='*80}")
    logger.info("关键发现（Key Findings）—— v7 group-based 安全查询")
    logger.info(f"{'='*80}")

    # ── 按 group + 位置索引提取所有需要的候选 ─────────────────
    baseline     = get_all_by_group(summary, 'baseline')
    diag1        = get_all_by_group(summary, 'baseline_diag')
    ablation_hp  = get_all_by_group(summary, 'ablation_hp')
    ablation_arch = get_all_by_group(summary, 'ablation_arch')
    nas_v1       = get_all_by_group(summary, 'nas_v1')
    nas_v2       = get_all_by_group(summary, 'nas_v2')
    diag_of      = get_all_by_group(summary, 'diag_overfit')

    # 取各组关键候选（按 CANDIDATES 中的顺序）
    # baseline[0] = 2xGCN defaultLR, baseline[1] = 1xGCN defaultLR
    base_2gcn  = baseline[0] if len(baseline) > 0 else None
    base_1gcn  = baseline[1] if len(baseline) > 1 else None
    base_diag  = diag1[0]    if len(diag1) > 0   else None
    # ablation_hp[0] = NAS LR only, ablation_hp[1] = NAS full HP
    abl_hp_v1  = ablation_hp[0] if len(ablation_hp) > 0 else None
    abl_hp_v2  = ablation_hp[1] if len(ablation_hp) > 1 else None
    # ablation_arch[0] = Phase4 arch, ablation_arch[1] = Phase3 arch
    abl_arch_p4 = ablation_arch[0] if len(ablation_arch) > 0 else None
    # nas_v2[0] = Phase3, nas_v2[1] = Phase4
    nas_v2_p3  = nas_v2[0] if len(nas_v2) > 0 else None
    nas_v2_p4  = nas_v2[1] if len(nas_v2) > 1 else None
    diag_ovf   = diag_of[0] if len(diag_of) > 0 else None

    # ── Finding 1：v1 HP 搜索贡献 ─────────────────────────
    if base_1gcn and abl_hp_v1:
        delta = abl_hp_v1['test_mean'] - base_1gcn['test_mean']
        var_r = base_1gcn['test_std'] - abl_hp_v1['test_std']
        logger.info(f"\n  [Finding 1] v1 HP 搜索贡献（仅搜 LR）：")
        logger.info(f"    可靠基线 1xGCN  : test={base_1gcn['test_mean']:.3f}"
                    f"±{base_1gcn['test_std']:.3f}")
        logger.info(f"    2xGCN + NAS LR  : test={abl_hp_v1['test_mean']:.3f}"
                    f"±{abl_hp_v1['test_std']:.3f}")
        logger.info(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")

    # ── Finding 2：v2 HP 相对于 v1 增量 ───────────────────
    if abl_hp_v1 and abl_hp_v2:
        delta = abl_hp_v2['test_mean'] - abl_hp_v1['test_mean']
        var_r = abl_hp_v1['test_std']  - abl_hp_v2['test_std']
        logger.info(f"\n  [Finding 2] v2 HP 相对于 v1 增量（hidden_dim + L2）：")
        logger.info(f"    2xGCN + NAS v1 HP : test={abl_hp_v1['test_mean']:.3f}"
                    f"±{abl_hp_v1['test_std']:.3f}")
        logger.info(f"    2xGCN + NAS v2 HP : test={abl_hp_v2['test_mean']:.3f}"
                    f"±{abl_hp_v2['test_std']:.3f}")
        logger.info(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")
        if delta > 0.01:
            logger.info(f"    → hidden_dim 和 l2 对性能有实质贡献")
        else:
            logger.info(f"    → hidden_dim/l2 边际收益有限，LR 是主要驱动")

    # ── Finding 3：架构搜索独立贡献 ───────────────────────
    if base_1gcn and abl_arch_p4:
        delta = abl_arch_p4['test_mean'] - base_1gcn['test_mean']
        logger.info(f"\n  [Finding 3] 架构搜索独立贡献（NAS架构 + 默认HP）：")
        logger.info(f"    基线 1xGCN          : test={base_1gcn['test_mean']:.3f}"
                    f"±{base_1gcn['test_std']:.3f}")
        logger.info(f"    NAS架构(P4)+默认HP  : test={abl_arch_p4['test_mean']:.3f}"
                    f"±{abl_arch_p4['test_std']:.3f}")
        logger.info(f"    Δtest={delta:+.3f}")
        if delta < 0:
            logger.info(f"    → 缺乏 HP 校准时，架构搜索无正向贡献")
        elif delta > 0.01:
            logger.info(f"    → NAS 架构本身有正向贡献（独立于 HP 效果）")
        else:
            logger.info(f"    → 架构贡献边际，HP 搜索是主要增益来源")

    # ── Finding 3b：Phase4 行 HP 贡献隔离 ─────────────────
    if abl_arch_p4 and nas_v2_p4:
        delta_hp = nas_v2_p4['test_mean'] - abl_arch_p4['test_mean']
        var_r    = abl_arch_p4['test_std'] - nas_v2_p4['test_std']
        logger.info(f"\n  [Finding 3b] Phase4 行 HP 贡献隔离（同一架构）：")
        logger.info(f"    默认 HP  : test={abl_arch_p4['test_mean']:.3f}"
                    f"±{abl_arch_p4['test_std']:.3f}")
        logger.info(f"    NAS v2 HP: test={nas_v2_p4['test_mean']:.3f}"
                    f"±{nas_v2_p4['test_std']:.3f}")
        logger.info(f"    Δtest={delta_hp:+.3f}  方差变化={var_r:+.3f}")
        logger.info(f"    → 上述差异纯粹由 HP 搜索贡献（架构已控制为同一）")

    # ── Finding 4：诊断1 hd=64 信息瓶颈 ──────────────────
    if base_2gcn and base_diag:
        delta = base_diag['test_mean'] - base_2gcn['test_mean']
        var_r = base_2gcn['test_std']  - base_diag['test_std']
        logger.info(f"\n  [Finding 4] 【诊断1】2xGCN 崩溃原因（hd=64 vs hd=128）：")
        logger.info(f"    2xGCN hd=64  : test={base_2gcn['test_mean']:.3f}"
                    f"±{base_2gcn['test_std']:.3f}  ← 已知崩溃")
        logger.info(f"    2xGCN hd=128 : test={base_diag['test_mean']:.3f}"
                    f"±{base_diag['test_std']:.3f}")
        logger.info(f"    Δtest={delta:+.3f}  方差变化={var_r:+.3f}")
        if delta > 0.15 and var_r > 0.10:
            logger.info(f"    → ✅ 确认：hd=64 是主要瓶颈")
        elif delta > 0.05:
            logger.info(f"    → ⚠️  部分改善：hd=64 是诱因之一")
        else:
            logger.info(f"    → ❌ hd 不是主因：崩溃来自 LR/weight_decay")

    # ── Finding 5：诊断2 Phase3 过拟合验证 ────────────────
    if diag_ovf:
        std_val  = diag_ovf['test_std']
        mean_val = diag_ovf['test_mean']
        logger.info(f"\n  [Finding 5] 【诊断2】Phase3 v2 最优配置稳定性验证：")
        logger.info(f"    BO 阶段 val_acc（单次）: 0.8140")
        logger.info(f"    多种子 test_acc       : {mean_val:.3f}±{std_val:.3f}")
        if std_val > 0.05:
            logger.info(f"    → ❌ 证实过拟合：种子极度敏感，BO 搜到验证集局部峰值")
        elif std_val < 0.02:
            logger.info(f"    → ✅ 配置稳定：具有真实泛化能力")
        else:
            logger.info(f"    → ⚠️  中等稳定性：建议适当降低 LR 再验证")

    # ── Finding 6：NAS v2 全量综合 ────────────────────────
    if base_1gcn and nas_v2_p4 and nas_v2_p3:
        logger.info(f"\n  [Finding 6] 完整 NAS v2 vs 可靠基线（1xGCN）：")
        logger.info(f"    基线 1xGCN    : test={base_1gcn['test_mean']:.3f}"
                    f"±{base_1gcn['test_std']:.3f}")
        d4 = nas_v2_p4['test_mean'] - base_1gcn['test_mean']
        d3 = nas_v2_p3['test_mean'] - base_1gcn['test_mean']
        logger.info(f"    NAS v2 Phase4 : test={nas_v2_p4['test_mean']:.3f}"
                    f"±{nas_v2_p4['test_std']:.3f}  Δ={d4:+.3f}")
        logger.info(f"    NAS v2 Phase3 : test={nas_v2_p3['test_mean']:.3f}"
                    f"±{nas_v2_p3['test_std']:.3f}  Δ={d3:+.3f}")


# ============================================================
# 可视化（v7：3×3 矩阵改为 group+位置索引取值，消除硬编码名称）
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

        # ── 左图：柱状图 ──────────────────────────────────
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
        ax.set_title('Cora Test Accuracy — Search Space v3\n'
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

        # ── 右图：3×3 消融热图（v7: group-based 取值）────────
        # 布局：
        #   行 0: 手工架构 2xGCN
        #         → baseline[0], ablation_hp[0], ablation_hp[1]
        #   行 1: NAS Phase4 架构 (SAGE+GCN+Identity)
        #         → ablation_arch[0], nas_v1[0], nas_v2[1]
        #   行 2: NAS Phase3 架构 (GAT+SAGE+SAGE)
        #         → ablation_arch[1], nas_v1[1], nas_v2[0]
        ax2 = axes[1]

        def g(group, n):
            """group+位置索引取 (mean, std)，安全返回 (nan, 0)。"""
            return _safe_test(get_nth_by_group(summary, group, n))

        mat_mean = np.array([
            # 行0: 手工架构 2xGCN
            [g('baseline',      0)[0], g('ablation_hp', 0)[0], g('ablation_hp', 1)[0]],
            # 行1: NAS Phase4 架构（SAGE+GCN）
            [g('ablation_arch', 0)[0], g('nas_v1',      0)[0], g('nas_v2',      1)[0]],
            # 行2: NAS Phase3 架构（GAT+SAGE+SAGE）
            [g('ablation_arch', 1)[0], g('nas_v1',      1)[0], g('nas_v2',      0)[0]],
        ])
        mat_std = np.array([
            [g('baseline',      0)[1], g('ablation_hp', 0)[1], g('ablation_hp', 1)[1]],
            [g('ablation_arch', 0)[1], g('nas_v1',      0)[1], g('nas_v2',      1)[1]],
            [g('ablation_arch', 1)[1], g('nas_v1',      1)[1], g('nas_v2',      0)[1]],
        ])

        im = ax2.imshow(mat_mean, cmap='RdYlGn',
                        vmin=0.40, vmax=0.85, aspect='auto')
        plt.colorbar(im, ax=ax2, label='Test Accuracy')
        for i in range(3):
            for j in range(3):
                v = mat_mean[i, j]
                s = mat_std[i, j]
                txt = f"{v:.3f}\n±{s:.3f}" if not np.isnan(v) else "N/A"
                fc  = 'white' if (np.isnan(v) or v < 0.50 or v > 0.78) else 'black'
                ax2.text(j, i, txt, ha='center', va='center',
                         fontsize=9, fontweight='bold', color=fc)

        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(
            ['Default HP\n(LR=1e-3, hd=64)',
             'NAS HP v1\n(LR only)',
             'NAS HP v2\n(LR+hd+l2)'], fontsize=9)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(
            ['Manual Arch\n(2xGCN)',
             'NAS Arch P4\n(SAGE+GCN)',
             'NAS Arch P3\n(GAT+SAGE+SAGE)'], fontsize=9)
        ax2.set_xlabel('Hyperparameter Source', fontsize=11)
        ax2.set_ylabel('Architecture Source',   fontsize=11)
        ax2.set_title(
            '3x3 Ablation Matrix: Arch Source × HP Version\n'
            '[v7: group-based indexing, no hardcoded names]', fontsize=10)

        plt.tight_layout()
        path = os.path.join(args.output, 'final_comparison_v7.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"final_eval.py  v7  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={DEVICE}  Seeds={args.n_seeds}  "
                f"Epochs={args.eval_epochs}  Patience={args.patience}")
    logger.info(f"version={args.version}  seed={args.seed}")

    main()

    elapsed = time.time() - start_time
    logger.info(f"\n✅ 运行完成  总耗时: {elapsed/60:.1f} min")