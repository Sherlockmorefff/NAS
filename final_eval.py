"""
final_eval.py  v9  (面向 condHP v5 的非冗余消融实验)
============================================================================
v9 变更（相对 v8）：

  [重构 1] 去除旧 v1/v2/诊断型冗余候选
      v8 仍保留 Phase3/Phase4 旧硬编码架构、旧 NAS v1/v2 HP 和诊断项，
      与当前 condHP v5 的 best_z_final.pt 不在同一实验语义下。
      v9 改为只保留必要人工基线，并从 Phase4 best_z 自动生成可归因消融：
        - 手工基线：Manual GCN + default HP
        - HP-only：Manual GCN + searched global HP
        - Arch-only：Decoded NAS arch + default HP
        - Cond-only：Decoded NAS arch + default global HP + searched conditional HP
        - GlobalHP-only：Decoded NAS arch + searched global HP + default conditional HP
        - Full：Decoded NAS arch + searched global HP + searched conditional HP

  [重构 2] 条件参数消融自动去重
      当架构不含 GAT/SAGE，或搜索到的条件参数等于默认值时，
      cond-only / globalHP-only 中会出现重复配置。v9 自动跳过这些重复候选。

  [重构 3] 输出与可视化改为 condHP 2×2 归因矩阵
      在 decoded NAS 架构固定时，比较：
        行：默认全局 HP / 搜索全局 HP
        列：默认条件 HP / 搜索条件 HP
      更直接地回答条件参数是否带来额外收益。

v8 变更（相对 v7）：

  [新增 1] 自动读取 Phase4 条件参数搜索结果
      - 默认读取 results/bo_phase4_condhp_v5/best_z_final.pt
      - 默认读取 results/joint_search/joint_model_condhp_v5_ep100.pth
      - 使用 JointSpaceVAE 解码 z[:12] 得到 GNN 架构
      - 使用 z[12:16] 反归一化得到 [lr, dropout, gat_heads, sage_aggr]
      - 按架构条件 mask 仅在包含 GAT/SAGE 时激活对应条件参数
      - 自动候选追加到 CANDIDATES 后，不破坏原有人工候选和消融矩阵

  [修复 2] final_eval 透传条件 HP
      v7 问题：run_eval 未把 cand 里的 gat_heads/sage_aggr 传给
               train_and_eval_arch，导致最终评估时条件参数失效。
      v8 修复：run_eval 安全提取 gat_heads/sage_aggr 并透传。

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
from collections import Counter
from datetime import datetime

sys.path.insert(0, '/mnt/project')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from nas_space import JointSpaceVAE
from eval_utils import (
    train_and_eval_arch,
    condition_mask_from_config,
    norm_to_gat_heads,
    norm_to_sage_aggr,
)


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
parser = argparse.ArgumentParser(description='final_eval v9')
# 原有参数（不变）
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_seeds',     type=int, default=10)
parser.add_argument('--eval_epochs', type=int, default=200)
parser.add_argument('--patience',    type=int, default=30)
parser.add_argument('--output',      type=str, default='results/final_eval')
# 新增参数
parser.add_argument('--version',     type=str, default='condhp_v5_final')
parser.add_argument('--log_dir',     type=str, default='logs/')
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='全局默认 weight_decay（各候选中若有 l2 字段则优先使用）')
parser.add_argument('--hidden_dim',  type=int, default=64,
                    help='全局默认 hidden_dim（各候选中若有 hidden_dim 字段则优先）')
parser.add_argument('--gcnii_alpha', type=float, default=0.1)
parser.add_argument('--gcnii_theta', type=float, default=0.5)
# v9：Phase4 条件参数 best_z 自动消融
parser.add_argument('--checkpoint',  type=str,
                    default='results/joint_search/joint_model_condhp_v5_ep100.pth',
                    help='用于解码 best_z_final.pt 的 JointSpaceVAE checkpoint')
parser.add_argument('--auto_best_z', type=str,
                    default='results/bo_phase4_condhp_v5/best_z_final.pt',
                    help='Phase4 条件参数搜索输出的 best_z_final.pt')
parser.add_argument('--disable_auto_best', action='store_true',
                    help='禁用自动追加 Phase4 best_z_final.pt 候选')
parser.add_argument('--auto_name',   type=str, default='NAS_condHP_full')
parser.add_argument('--auto_decode_trials', type=int, default=10)
parser.add_argument('--arch_nz',     type=int, default=12)
parser.add_argument('--log_lr_min',  type=float, default=-4.0)
parser.add_argument('--log_lr_max',  type=float, default=-1.5)
parser.add_argument('--dropout_min', type=float, default=0.1)
parser.add_argument('--dropout_max', type=float, default=0.6)
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ARCH_NZ = args.arch_nz
os.makedirs(args.output, exist_ok=True)

logger, log_filepath = setup_logger(args.log_dir, 'final_eval', args.version)
save_args_json(args, log_filepath)


# ============================================================
# 候选架构定义（v9：仅保留必要人工基线，其余消融由 best_z 动态生成）
# ============================================================
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.5
DEFAULT_GAT_HEADS = 1
DEFAULT_SAGE_AGGR = 'mean'
DEFAULT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]
MANUAL_2GCN_OPS = ["GCNConv", "GCNConv", "Identity"]
MANUAL_1GCN_OPS = ["GCNConv", "Identity", "Identity"]


def make_candidate(
    name: str,
    group: str,
    description: str,
    operations: list,
    edges: list,
    lr: float = DEFAULT_LR,
    dropout: float = DEFAULT_DROPOUT,
    hidden_dim: int = None,
    l2: float = None,
    gat_heads: int = DEFAULT_GAT_HEADS,
    sage_aggr: str = DEFAULT_SAGE_AGGR,
    **extra,
) -> dict:
    cand = {
        "name":        name,
        "description": description,
        "operations":  operations,
        "edges":       edges,
        "lr":          lr,
        "dropout":     dropout,
        "hidden_dim":  args.hidden_dim if hidden_dim is None else hidden_dim,
        "l2":          args.weight_decay if l2 is None else l2,
        "gcnii_alpha": args.gcnii_alpha,
        "gcnii_theta": args.gcnii_theta,
        "gat_heads":   gat_heads,
        "sage_aggr":   sage_aggr,
        "group":       group,
    }
    cand.update(extra)
    return cand


CANDIDATES = [
    make_candidate(
        name="BASE_2xGCN_defaultHP",
        group="baseline",
        description="人工基线：2层GCN + 默认全局HP",
        operations=MANUAL_2GCN_OPS,
        edges=DEFAULT_EDGES,
    ),
    make_candidate(
        name="BASE_1xGCN_defaultHP",
        group="baseline",
        description="人工参考：1层GCN + 默认全局HP",
        operations=MANUAL_1GCN_OPS,
        edges=DEFAULT_EDGES,
    ),
]


# ============================================================
# v9：Phase4 best_z_final.pt 自动消融候选
# ============================================================
class ArchArgs:
    max_n           = 7
    num_vertex_type = 8
    START_TYPE      = 0
    END_TYPE        = 1
    hs              = 501
    nz              = ARCH_NZ
    bidirectional   = True


def _torch_load(path: str, map_location):
    """兼容不同 torch 版本的安全加载入口。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_vae_for_auto(ckpt_path: str):
    model = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).to(DEVICE)
    state = _torch_load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Auto VAE loaded: {ckpt_path}  (arch_nz={ARCH_NZ}, hp_dim=4)")
    return model


def decode_arch_from_z(vae, z_arch: torch.Tensor, n_trials: int = 10):
    """
    多次解码并取众数架构，和 Phase4 decode_arch 保持一致。
    """
    results = []
    z_arch = z_arch.detach().float()
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(DEVICE))
            g = graphs[0]
            ops = [vae.op_mapping.get(g.vs[i]['type'], 'Identity')
                   for i in range(1, g.vcount() - 1)]
            edges = g.get_edgelist()
            n_eff = sum(1 for op in ops if op != 'Identity')
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append((key, {
                    'effective_layers': n_eff,
                    'operations':       ops,
                    'edges':            edges,
                }))
    if not results:
        return None

    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(c for k, c in results if k == best_key)


def z_to_condhp(z_search: torch.Tensor, config: dict):
    """
    将 Phase4 搜索向量 z[12:16] 反归一化为条件 HP。
    z[14]/z[15] 仅在架构含 GATConv/SAGEConv 时激活。
    """
    if z_search.shape[0] < ARCH_NZ + 4:
        raise ValueError(
            f"best_z 维度不足：需要至少 {ARCH_NZ + 4} 维，实际 {z_search.shape[0]} 维")

    log_lr_n = float(z_search[ARCH_NZ])
    drop_n   = float(z_search[ARCH_NZ + 1])
    log_lr   = log_lr_n * (args.log_lr_max - args.log_lr_min) + args.log_lr_min
    dropout  = drop_n * (args.dropout_max - args.dropout_min) + args.dropout_min
    lr       = float(np.clip(10 ** np.clip(log_lr, -6, 0), 1e-6, 1.0))
    dr       = float(np.clip(dropout, 0.0, 0.9))

    mask = condition_mask_from_config(config)
    gat_heads = norm_to_gat_heads(float(z_search[ARCH_NZ + 2])) \
        if mask['gat_heads'] else 1
    sage_aggr = norm_to_sage_aggr(float(z_search[ARCH_NZ + 3])) \
        if mask['sage_aggr'] else 'mean'

    return lr, dr, gat_heads, sage_aggr, mask


def candidate_signature(cand: dict) -> tuple:
    edges = tuple(tuple(e) for e in cand['edges'])
    return (
        tuple(cand['operations']),
        edges,
        round(float(cand['lr']), 12),
        round(float(cand['dropout']), 12),
        int(cand.get('hidden_dim', args.hidden_dim)),
        round(float(cand.get('l2', args.weight_decay)), 12),
        int(cand.get('gat_heads', DEFAULT_GAT_HEADS)),
        cand.get('sage_aggr', DEFAULT_SAGE_AGGR),
        round(float(cand.get('gcnii_alpha', args.gcnii_alpha)), 12),
        round(float(cand.get('gcnii_theta', args.gcnii_theta)), 12),
    )


def append_unique_candidate(target: list, cand: dict, seen: set) -> bool:
    sig = candidate_signature(cand)
    if sig in seen:
        logger.info(f"跳过冗余消融候选: {cand['name']}  ({cand['description']})")
        return False
    seen.add(sig)
    target.append(cand)
    return True


def build_auto_ablation_candidates(existing_candidates=None):
    if args.disable_auto_best:
        logger.info("Auto best_z ablation disabled by --disable_auto_best")
        return []

    if not os.path.exists(args.auto_best_z):
        logger.warning(f"未找到 auto_best_z，跳过自动消融候选: {args.auto_best_z}")
        return []
    if not os.path.exists(args.checkpoint):
        logger.warning(f"未找到 checkpoint，跳过自动消融候选: {args.checkpoint}")
        return []

    vae = load_vae_for_auto(args.checkpoint)
    z_obj = _torch_load(args.auto_best_z, map_location='cpu')
    z_search = torch.as_tensor(z_obj, dtype=torch.float32).view(-1)

    config = decode_arch_from_z(
        vae, z_search[:ARCH_NZ], n_trials=args.auto_decode_trials)
    if config is None:
        logger.warning(f"best_z 架构解码失败，跳过自动消融候选: {args.auto_best_z}")
        return []

    lr, dropout, gat_heads, sage_aggr, mask = z_to_condhp(z_search, config)

    source_meta = {
        'source_best_z': args.auto_best_z,
        'source_checkpoint': args.checkpoint,
        'condition_mask': mask,
    }
    nas_ops = config['operations']
    nas_edges = config['edges']

    added = []
    seen = {candidate_signature(c) for c in (existing_candidates or [])}

    append_unique_candidate(added, make_candidate(
        name="ABL_manualArch_searchGlobalHP",
        group="hp_on_manual",
        description="HP-only：2xGCN 手工架构 + Phase4 搜索出的 lr/dropout",
        operations=MANUAL_2GCN_OPS,
        edges=DEFAULT_EDGES,
        lr=lr,
        dropout=dropout,
        **source_meta,
    ), seen)

    append_unique_candidate(added, make_candidate(
        name="ABL_NASArch_defaultHP",
        group="arch_only",
        description="Arch-only：Phase4 解码架构 + 默认全局HP + 默认条件HP",
        operations=nas_ops,
        edges=nas_edges,
        **source_meta,
    ), seen)

    has_nondefault_cond = (
        (mask['gat_heads'] and gat_heads != DEFAULT_GAT_HEADS) or
        (mask['sage_aggr'] and sage_aggr != DEFAULT_SAGE_AGGR)
    )

    if has_nondefault_cond:
        append_unique_candidate(added, make_candidate(
            name="ABL_NASArch_condHPOnly",
            group="cond_only",
            description="Cond-only：Phase4 解码架构 + 默认 lr/dropout + 搜索条件HP",
            operations=nas_ops,
            edges=nas_edges,
            gat_heads=gat_heads,
            sage_aggr=sage_aggr,
            **source_meta,
        ), seen)

        append_unique_candidate(added, make_candidate(
            name="ABL_NASArch_globalHPOnly",
            group="arch_global_hp",
            description="GlobalHP-only：Phase4 解码架构 + 搜索 lr/dropout + 默认条件HP",
            operations=nas_ops,
            edges=nas_edges,
            lr=lr,
            dropout=dropout,
            **source_meta,
        ), seen)
    else:
        logger.info("搜索到的条件HP与默认值一致或无激活条件参数，跳过 cond-only/globalHP-only 重复消融")

    append_unique_candidate(added, make_candidate(
        name=args.auto_name,
        group="full_condhp",
        description="Full：Phase4 解码架构 + 搜索 lr/dropout + 搜索条件HP",
        operations=nas_ops,
        edges=nas_edges,
        lr=lr,
        dropout=dropout,
        gat_heads=gat_heads,
        sage_aggr=sage_aggr,
        **source_meta,
    ), seen)

    logger.info("已生成 Phase4 condHP v5 消融候选：")
    logger.info(f"  ops={nas_ops}")
    logger.info(f"  lr={lr:.5f}  dropout={dropout:.3f}  "
                f"gat_heads={gat_heads}  sage_aggr={sage_aggr}")
    logger.info(f"  mask={mask}")
    logger.info(f"  added={len(added)}")
    return added


# ============================================================
# Group-based 安全查询函数
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
    按 group 字段查询第 n 个候选（0-indexed，按候选列表顺序）。
    用于保留人工基线的固定顺序。
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
# 多种子评估（v9：条件参数透传 + eval_utils 统一）
# ============================================================
def run_eval(cand: dict, data, in_ch: int, out_ch: int) -> dict:
    """
    多种子评估单个候选架构。

    v9 保留：
      - 透传条件参数 gat_heads/sage_aggr
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
    gcnii_alpha  = cand.get('gcnii_alpha', args.gcnii_alpha)
    gcnii_theta  = cand.get('gcnii_theta', args.gcnii_theta)
    gat_heads    = cand.get('gat_heads', 1)
    sage_aggr    = cand.get('sage_aggr', 'mean')

    val_accs, test_accs = [], []

    for seed in range(args.n_seeds):
        # 直接调用 eval_utils 的 train_and_eval_arch(track_test=True)
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
            gat_heads    = gat_heads,
            sage_aggr    = sage_aggr,
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
        'gat_heads':   gat_heads,
        'sage_aggr':   sage_aggr,
        'operations':  cand['operations'],
        'edges':       cand['edges'],
        'source_best_z': cand.get('source_best_z'),
        'source_checkpoint': cand.get('source_checkpoint'),
        'condition_mask': cand.get('condition_mask'),
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

    candidates = list(CANDIDATES)
    candidates.extend(build_auto_ablation_candidates(existing_candidates=candidates))

    summary = []
    for cand in candidates:
        logger.info(f"\n{'='*66}")
        logger.info(f"[{cand['group'].upper()}]  {cand['name']}")
        logger.info(f"  {cand['description']}")
        logger.info(f"  ops={cand['operations']}")
        logger.info(f"  lr={cand['lr']:.5f}  dropout={cand['dropout']:.3f}  "
                    f"hidden={cand.get('hidden_dim',64)}  l2={cand.get('l2',5e-4):.0e}  "
                    f"gcnii_α={cand.get('gcnii_alpha',0.1)}  "
                    f"gcnii_θ={cand.get('gcnii_theta',0.5)}  "
                    f"gat_heads={cand.get('gat_heads',1)}  "
                    f"sage_aggr={cand.get('sage_aggr','mean')}")
        if cand.get('source_best_z'):
            logger.info(f"  source_best_z={cand['source_best_z']}")

        result = run_eval(cand, data, in_ch, out_ch)
        summary.append(result)

        logger.info(f"\n  >> Val  : {result['val_mean']:.4f} ± {result['val_std']:.4f}")
        logger.info(f"  >> Test : {result['test_mean']:.4f} ± {result['test_std']:.4f}"
                    f"  ← 报告此值")

    _print_summary(summary)
    _print_findings(summary)

    out_path = os.path.join(args.output, 'final_results_v9.json')
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
        ('baseline',       'A. 人工基线'),
        ('hp_on_manual',   'B. HP-only：手工架构 + 搜索全局HP'),
        ('arch_only',      'C. Arch-only：NAS架构 + 默认HP'),
        ('cond_only',      'D. Cond-only：NAS架构 + 搜索条件HP'),
        ('arch_global_hp', 'E. GlobalHP-only：NAS架构 + 搜索全局HP'),
        ('full_condhp',    'F. Full：NAS架构 + 搜索全局HP + 搜索条件HP'),
    ]
    for gkey, glabel in group_order:
        rows = get_all_by_group(summary, gkey)
        if not rows:
            continue
        logger.info(f"\n  {glabel}")
        logger.info(f"  {'名称':<44} {'Val':>14} {'Test':>14} "
                    f"{'Hidden':>6} {'L2':>8} {'Heads':>5} {'Aggr':>6}")
        logger.info(f"  {'-'*108}")
        for r in rows:
            vs = f"{r['val_mean']:.3f}±{r['val_std']:.3f}"
            ts = f"{r['test_mean']:.3f}±{r['test_std']:.3f}"
            logger.info(f"  {r['name'][:43]:<44} {vs:>14} {ts:>14} "
                        f"{r['hidden_dim']:>6} {r['l2']:>8.0e} "
                        f"{r.get('gat_heads', 1):>5} {r.get('sage_aggr', 'mean'):>6}")


# ============================================================
# 关键发现打印（v9：condHP 非冗余消融归因）
# ============================================================
def _print_findings(summary):
    logger.info(f"\n{'='*80}")
    logger.info("关键发现（Key Findings）—— condHP v5 非冗余消融")
    logger.info(f"{'='*80}")

    base_2gcn = get_nth_by_group(summary, 'baseline', 0)
    base_1gcn = get_nth_by_group(summary, 'baseline', 1)
    hp_manual = get_best_by_group(summary, 'hp_on_manual')
    arch_default = get_best_by_group(summary, 'arch_only')
    cond_only = get_best_by_group(summary, 'cond_only')
    global_only = get_best_by_group(summary, 'arch_global_hp')
    full = get_best_by_group(summary, 'full_condhp')

    def line(label, entry):
        if entry is None:
            logger.info(f"    {label:<24}: N/A")
            return
        logger.info(f"    {label:<24}: test={entry['test_mean']:.3f}"
                    f"±{entry['test_std']:.3f}")

    def delta(label, target, ref):
        if target is None or ref is None:
            return
        d = target['test_mean'] - ref['test_mean']
        logger.info(f"    {label:<24}: Δtest={d:+.3f}")

    logger.info("\n  [Finding 1] 人工基线稳定性：")
    line("2xGCN default", base_2gcn)
    line("1xGCN default", base_1gcn)
    delta("1xGCN - 2xGCN", base_1gcn, base_2gcn)

    if hp_manual:
        logger.info("\n  [Finding 2] 全局 HP 搜索贡献（固定手工架构）：")
        line("Manual default HP", base_2gcn)
        line("Manual searched HP", hp_manual)
        delta("global HP only", hp_manual, base_2gcn)

    if arch_default:
        logger.info("\n  [Finding 3] 架构搜索贡献（默认 HP 控制）：")
        line("Manual arch", base_2gcn)
        line("NAS arch default HP", arch_default)
        delta("arch only", arch_default, base_2gcn)

    if global_only:
        logger.info("\n  [Finding 4] NAS 架构上的全局 HP 增量：")
        line("NAS arch default HP", arch_default)
        line("NAS arch global HP", global_only)
        delta("global HP on NAS", global_only, arch_default)

    if cond_only:
        logger.info("\n  [Finding 5] 条件 HP 单独贡献（默认 lr/dropout）：")
        line("NAS arch default cond", arch_default)
        line("NAS arch searched cond", cond_only)
        delta("conditional HP only", cond_only, arch_default)

    if full:
        logger.info("\n  [Finding 6] 完整 condHP 搜索结果：")
        line("Full condHP", full)
        if global_only:
            delta("condHP marginal", full, global_only)
        elif arch_default:
            delta("searched HP total", full, arch_default)
        if base_2gcn:
            delta("full vs 2xGCN", full, base_2gcn)
        logger.info(f"    ops                     : {full['operations']}")
        logger.info(f"    lr/dropout              : {full['lr']:.5f} / {full['dropout']:.3f}")
        logger.info(f"    gat_heads/sage_aggr     : {full.get('gat_heads', 1)} / "
                    f"{full.get('sage_aggr', 'mean')}")
        logger.info(f"    condition_mask          : {full.get('condition_mask')}")
    else:
        logger.warning("\n  未生成完整 condHP 候选；请确认 Phase4 best_z_final.pt 和 VAE checkpoint 是否存在。")


# ============================================================
# 可视化（v9：非冗余候选柱状图 + condHP 2×2 消融矩阵）
# ============================================================
def _plot(summary):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        group_colors = {
            'baseline':       '#7f8c8d',
            'hp_on_manual':   '#f39c12',
            'arch_only':      '#8e44ad',
            'cond_only':      '#16a085',
            'arch_global_hp': '#2980b9',
            'full_condhp':    '#c0392b',
        }

        names  = [r['name'].replace('_', '\n') for r in summary]
        tmeans = [r['test_mean'] for r in summary]
        tstds  = [r['test_std']  for r in summary]
        clrs   = [group_colors.get(r['group'], '#95a5a6') for r in summary]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # ── 左图：柱状图 ──────────────────────────────────
        ax = axes[0]
        bars = ax.bar(range(len(names)), tmeans, yerr=tstds,
                      color=clrs, alpha=0.85, capsize=5, ecolor='gray')
        for bar, m, s in zip(bars, tmeans, tstds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.004,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel('Test Accuracy')
        ax.set_ylim(0.30, 0.93)
        ax.set_title('Cora Test Accuracy — condHP v5 ablation\n'
                     '(CosineAnnealingLR + EarlyStopping)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=[
            Patch(color='#7f8c8d', label='Manual baseline'),
            Patch(color='#f39c12', label='HP-only on manual arch'),
            Patch(color='#8e44ad', label='Arch-only'),
            Patch(color='#16a085', label='Cond-only'),
            Patch(color='#2980b9', label='GlobalHP-only'),
            Patch(color='#c0392b', label='Full condHP'),
        ], fontsize=7, loc='lower right')

        # ── 右图：固定 NAS 架构后的 2×2 HP 消融热图 ─────────
        ax2 = axes[1]

        def best(group):
            return _safe_test(get_best_by_group(summary, group))

        mat_mean = np.array([
            [best('arch_only')[0],      best('cond_only')[0]],
            [best('arch_global_hp')[0], best('full_condhp')[0]],
        ])
        mat_std = np.array([
            [best('arch_only')[1],      best('cond_only')[1]],
            [best('arch_global_hp')[1], best('full_condhp')[1]],
        ])

        im = ax2.imshow(mat_mean, cmap='RdYlGn',
                        vmin=0.40, vmax=0.85, aspect='auto')
        plt.colorbar(im, ax=ax2, label='Test Accuracy')
        for i in range(2):
            for j in range(2):
                v = mat_mean[i, j]
                s = mat_std[i, j]
                txt = f"{v:.3f}\n±{s:.3f}" if not np.isnan(v) else "N/A"
                fc  = 'white' if (np.isnan(v) or v < 0.50 or v > 0.78) else 'black'
                ax2.text(j, i, txt, ha='center', va='center',
                         fontsize=9, fontweight='bold', color=fc)

        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(
            ['Default conditional HP\n(heads=1, aggr=mean)',
             'Searched conditional HP'], fontsize=8)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(
            ['Default global HP\n(lr=1e-3, dropout=0.5)',
             'Searched global HP'], fontsize=8)
        ax2.set_xlabel('Conditional HP source', fontsize=11)
        ax2.set_ylabel('Global HP source', fontsize=11)
        ax2.set_title(
            '2x2 HP ablation on decoded NAS architecture\n'
            '[N/A means the candidate was redundant and skipped]', fontsize=10)

        plt.tight_layout()
        path = os.path.join(args.output, 'final_comparison_v9.png')
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
    logger.info(f"final_eval.py  v9  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={DEVICE}  Seeds={args.n_seeds}  "
                f"Epochs={args.eval_epochs}  Patience={args.patience}")
    logger.info(f"version={args.version}  seed={args.seed}")

    main()

    elapsed = time.time() - start_time
    logger.info(f"\n✅ 运行完成  总耗时: {elapsed/60:.1f} min")
