"""
bo_phase3.py  v7  (ARCH_NZ=12 对齐 + GCNII 兼容 + 工业级日志)
===============================================================
v7 变更（相对 v6）：

  [对齐 1] ARCH_NZ: 8 → 12，SEARCH_DIM: 12 → 16
      ArchArgs 强制更新：max_n=7, num_vertex_type=8, nz=12
      全部 HP 切片索引通过 ARCH_NZ 常量动态计算

  [对齐 2] GCNII 兼容
      eval_z_search 调用时透传 gcnii_alpha / gcnii_theta

  [新增 3] 工业级日志系统
      - setup_logger 控制台 + 文件双端输出
      - 新增参数：--version, --log_dir, --seed,
                  --weight_decay, --hidden_dim,
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
import urllib.request
import warnings
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, '/mnt/project')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*Input.*not contained.*unit cube.*")

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from nas_space import JointSpaceVAE
from eval_utils import (
    eval_z_search,
    DynamicGNN,
    HP_DIM_LAYERWISE,
    norm_to_gat_heads,
    norm_to_sage_aggr,
    decode_layerwise_conditional_hp,
    condition_mask_from_config,
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
parser = argparse.ArgumentParser(description='bo_phase3 v7')
# 原有参数（保持不变）
parser.add_argument('--checkpoint',  type=str,
                    default='results/joint_search/joint_model_v3_ep100.pth')
parser.add_argument('--warm_start',  type=str,
                    default='results/bo_phase2/best_z_arch_final.pt')
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_init',      type=int, default=15)
parser.add_argument('--n_iter',      type=int, default=50)
parser.add_argument('--output',      type=str, default='results/bo_phase3')
parser.add_argument('--sigma_arch',  type=float, default=0.8)
parser.add_argument('--eval_epochs', type=int, default=100)
parser.add_argument('--novelty_w',   type=float, default=0.1)
parser.add_argument('--jacobian_n',  type=int, default=12)   # v7: 8→12
parser.add_argument('--log_lr_min',  type=float, default=-4.0)
parser.add_argument('--log_lr_max',  type=float, default=-1.5)
parser.add_argument('--dropout_min', type=float, default=0.1)
parser.add_argument('--dropout_max', type=float, default=0.6)
# 新增参数
parser.add_argument('--version',     type=str, default='v3_final')
parser.add_argument('--log_dir',     type=str, default='logs/')
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden_dim',  type=int,   default=64)
parser.add_argument('--gcnii_alpha', type=float, default=0.1)
parser.add_argument('--gcnii_theta', type=float, default=0.5)
parser.add_argument('--patience',    type=int,   default=20)
parser.add_argument('--hp_dim',      type=int,   default=4,
                    choices=[4, HP_DIM_LAYERWISE],
                    help='HP 维度：4=v5 type-wise，17=v6 layer-wise + GIN')
args = parser.parse_args()

# ── 全局常量（v7：ARCH_NZ=12）────────────────────────────
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ARCH_NZ    = 12   # v7 核心修正：8 → 12
HP_DIM     = args.hp_dim
SEARCH_DIM = ARCH_NZ + HP_DIM

os.makedirs(args.output, exist_ok=True)

# ── 日志初始化 ────────────────────────────────────────────
logger, log_filepath = setup_logger(args.log_dir, 'bo_phase3', args.version)
save_args_json(args, log_filepath)


# ============================================================
# HP 反归一化
# ============================================================
def z_to_hp(z_search: torch.Tensor, config: dict = None):
    log_lr_n = float(z_search[ARCH_NZ])
    drop_n   = float(z_search[ARCH_NZ + 1])
    log_lr   = log_lr_n * (args.log_lr_max - args.log_lr_min) + args.log_lr_min
    dropout  = drop_n   * (args.dropout_max - args.dropout_min) + args.dropout_min
    lr       = float(np.clip(10 ** log_lr, 1e-6, 1.0))
    dr       = float(np.clip(dropout, 0.0, 0.9))

    if HP_DIM >= HP_DIM_LAYERWISE:
        gat_heads, sage_aggr, gin_eps = decode_layerwise_conditional_hp(
            z_search, ARCH_NZ, config)
        return lr, dr, gat_heads, sage_aggr, gin_eps

    mask = condition_mask_from_config(config)
    gat_heads = norm_to_gat_heads(float(z_search[ARCH_NZ + 2])) \
        if mask['gat_heads'] and z_search.shape[0] > ARCH_NZ + 2 else 1
    sage_aggr = norm_to_sage_aggr(float(z_search[ARCH_NZ + 3])) \
        if mask['sage_aggr'] and z_search.shape[0] > ARCH_NZ + 3 else 'mean'

    return lr, dr, gat_heads, sage_aggr, None


def sample_hp_norm():
    return torch.rand(HP_DIM)


# ============================================================
# ArchArgs — v3 对齐
# ============================================================
class ArchArgs:
    def __init__(self):
        self.max_n           = 7    # v3
        self.num_vertex_type = 8    # v3
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = ARCH_NZ   # 12
        self.bidirectional   = True


def load_vae(ckpt_path):
    model = JointSpaceVAE(
        ArchArgs(), hp_latent_dim=HP_DIM, hp_input_dim=HP_DIM).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"VAE loaded: {ckpt_path}  (arch_nz={ARCH_NZ})")
    return model


# ============================================================
# Cora 加载
# ============================================================
_CORA_FILES = [
    "ind.cora.x","ind.cora.tx","ind.cora.allx",
    "ind.cora.y","ind.cora.ty","ind.cora.ally",
    "ind.cora.graph","ind.cora.test.index",
]
_MIRRORS = [
    "https://gitee.com/jiajiewu/planetoid/raw/master/data",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
]


def _dl(url, dest, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 10: return False
        open(dest,"wb").write(data); return True
    except Exception: return False


def load_cora(root):
    raw = os.path.join(root, "raw")
    miss = [f for f in _CORA_FILES if not os.path.exists(os.path.join(raw, f))]
    if miss:
        os.makedirs(raw, exist_ok=True)
        for f in miss:
            ok = any(_dl(f"{m}/{f}", os.path.join(raw, f)) for m in _MIRRORS)
            logger.info(f"  {f}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root)=="Cora" else root
    ds = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    return ds[0].to(DEVICE), ds.num_features, ds.num_classes


# ============================================================
# 评估函数（v7：透传 gcnii_alpha/theta）
# ============================================================
def eval_z(vae, z_search: torch.Tensor, data, in_ch, out_ch, epochs=None):
    result = eval_z_search(
        vae, z_search, data, in_ch, out_ch,
        arch_nz     = ARCH_NZ,
        log_lr_min  = args.log_lr_min,
        log_lr_max  = args.log_lr_max,
        dropout_min = args.dropout_min,
        dropout_max = args.dropout_max,
        device      = DEVICE,
        has_hidden  = True,
        has_l2      = True,
        use_conditional_params = True,
        default_hidden_dim     = args.hidden_dim,
        default_weight_decay   = args.weight_decay,
        gcnii_alpha = args.gcnii_alpha,   # v7 新增
        gcnii_theta = args.gcnii_theta,   # v7 新增
        return_layerwise = HP_DIM >= HP_DIM_LAYERWISE,
    )
    if HP_DIM >= HP_DIM_LAYERWISE:
        return result
    acc, lr_v, dr_v, heads_v, aggr_v, ok = result
    return acc, lr_v, dr_v, heads_v, aggr_v, None, ok


# ============================================================
# 新颖性计算（基于 Jacobian 正交方向）
# ============================================================
OP_REG = {'GCNConv': GCNConv, 'GATConv': GATConv,
          'SAGEConv': SAGEConv, 'GINConv': GINConv}


def _soft_vec(vae, z_arch):
    graphs = vae.arch_vae.decode(z_arch.unsqueeze(0))
    g      = graphs[0]
    vec    = torch.zeros(vae.arch_vae.max_n * vae.arch_vae.nvt)
    for i, v in enumerate(g.vs):
        t = v['type']
        if 0 <= t < vae.arch_vae.nvt:
            vec[i * vae.arch_vae.nvt + t] = 1.0
    return vec


def compute_novelty(vae, z_cands, X_obs, n_probes=12, eps=1e-3):
    actual_n = min(n_probes, ARCH_NZ)
    scores   = torch.zeros(z_cands.shape[0])
    with torch.no_grad():
        for m in range(z_cands.shape[0]):
            z  = z_cands[m, :ARCH_NZ].to(DEVICE)
            pd = torch.nn.functional.normalize(
                torch.randn(actual_n, ARCH_NZ, device=DEVICE), dim=1)
            J  = torch.stack([
                (_soft_vec(vae, z + eps*pd[d]) - _soft_vec(vae, z - eps*pd[d]))
                / (2*eps) for d in range(actual_n)], dim=1).cpu()
            try:
                _, _, Vh = torch.linalg.svd(J, full_matrices=False)
                n_o = min(actual_n // 2, 4)
                vo  = torch.nn.functional.normalize(
                    (Vh[:n_o].unsqueeze(-1) * pd[:n_o].cpu()).sum(0, keepdim=True),
                    dim=1).squeeze(0)
            except Exception:
                continue
            if X_obs.shape[0] == 0:
                scores[m] = 1.0; continue
            diffs  = z_cands[m, :ARCH_NZ].cpu() - X_obs[:, :ARCH_NZ]
            scores[m] = (diffs * vo).sum(1).abs().min().item()
    mx = scores.max()
    return scores / mx if mx > 1e-8 else scores


# ============================================================
# BO 采集函数
# ============================================================
def _minmax(X):
    xmin = X.min(0).values
    xrng = (X.max(0).values - xmin).clamp(min=1e-8)
    return (X - xmin) / xrng, xmin, xrng


def optimize_acq(X_obs, Y_obs, vae, novelty_w):
    X_n, Xmin, Xrng = _minmax(X_obs)
    Y_n = (Y_obs - Y_obs.mean()) / Y_obs.std().clamp(min=1e-6)

    gp  = SingleTaskGP(X_n.double(), Y_n.double())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    logei  = qLogExpectedImprovement(gp, best_f=Y_n.double().max())
    bounds = torch.zeros(2, SEARCH_DIM, dtype=torch.double)
    bounds[1] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand, _ = optimize_acqf(logei, bounds=bounds, q=1,
                                num_restarts=10, raw_samples=512)

    if novelty_w > 0:
        extra    = torch.rand(64, SEARCH_DIM, dtype=torch.double)
        all_c    = torch.cat([cand, extra]).float()
        all_orig = all_c * Xrng + Xmin
        all_norm = (all_orig - Xmin) / Xrng
        with torch.no_grad():
            ei_s = logei(all_norm.double().unsqueeze(1)).float()
        nov_s = compute_novelty(vae, all_orig, X_obs, args.jacobian_n)
        ei_n  = (ei_s - ei_s.min()) / (ei_s.max()-ei_s.min()).clamp(min=1e-8)
        best  = (ei_n + novelty_w * nov_s).argmax().item()
        return all_orig[best]
    return cand.squeeze(0).float() * Xrng + Xmin


# ============================================================
# 解码辅助
# ============================================================
def decode_arch(vae, z_arch: torch.Tensor, n_trials=5):
    from collections import Counter
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(DEVICE))
            g = graphs[0]
            ops   = [vae.op_mapping.get(g.vs[i]['type'], 'Identity')
                     for i in range(1, g.vcount()-1)]
            edges = g.get_edgelist()
            n_eff = sum(1 for op in ops if op != 'Identity')
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append((key, {'effective_layers': n_eff,
                                      'operations': ops, 'edges': edges}))
    if not results: return None
    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(c for k, c in results if k == best_key)


# ============================================================
# 主 BO 循环
# ============================================================
def run_bo(vae, data, in_ch, out_ch):
    history, X_obs, Y_obs = [], [], []

    # ── Warm Start ────────────────────────────────────────
    warm_pts = []
    if os.path.exists(args.warm_start):
        z_arch_best = torch.load(args.warm_start, map_location='cpu',
                                 weights_only=True)
        if z_arch_best.shape[0] > ARCH_NZ:
            z_arch_best = z_arch_best[:ARCH_NZ]
        elif z_arch_best.shape[0] < ARCH_NZ:
            z_arch_best = torch.cat(
                [z_arch_best, torch.zeros(ARCH_NZ - z_arch_best.shape[0])])
        n_warm = min(5, args.n_init // 3)
        for _ in range(n_warm):
            z_arch = z_arch_best + torch.randn(ARCH_NZ) * 0.3
            z_hp   = sample_hp_norm()
            warm_pts.append(torch.cat([z_arch, z_hp]))
        logger.info(f"Warm start: {n_warm} pts near Phase 2 best")

    n_rand   = args.n_init - len(warm_pts)
    rand_pts = [torch.cat([torch.randn(ARCH_NZ) * args.sigma_arch,
                           sample_hp_norm()])
                for _ in range(n_rand)]
    init_pts = warm_pts + rand_pts

    logger.info(f"\n[Step 1] Init: {args.n_init} pts  "
                f"({len(warm_pts)} warm + {n_rand} random)")
    logger.info(f"  ARCH_NZ={ARCH_NZ}  SEARCH_DIM={SEARCH_DIM}")
    logger.info(f"  LR: 10^[{args.log_lr_min},{args.log_lr_max}]  "
                f"Dropout: [{args.dropout_min},{args.dropout_max}]")
    if HP_DIM >= HP_DIM_LAYERWISE:
        logger.info("  conditional HP: [lr, dropout, gat_heads_i, sage_aggr_i, gin_eps_i]")
        logger.info("  layer-wise activation for GATConv/SAGEConv/GINConv")
    else:
        logger.info("  conditional HP: [lr, dropout, gat_heads, sage_aggr]")
        logger.info("  gat_heads active only for GATConv; sage_aggr active only for SAGEConv")
    logger.info(f"  gcnii_alpha={args.gcnii_alpha}  gcnii_theta={args.gcnii_theta}")
    logger.info(f"  Eval: max_epochs=150  patience={args.patience}  CosineAnnealingLR")

    for i, z in enumerate(tqdm(init_pts, desc='Init')):
        acc, lr_v, dr_v, heads_v, aggr_v, gin_v, ok = eval_z(vae, z, data, in_ch, out_ch)
        X_obs.append(z); Y_obs.append(acc)
        history.append({'step': i, 'type': 'init',
                        'val_acc': acc, 'lr': lr_v, 'dropout': dr_v,
                        'gat_heads': heads_v, 'sage_aggr': aggr_v,
                        'gin_eps': gin_v,
                        'valid': ok})
        logger.info(f"  init {i:>2d}: val={acc:.4f}  lr={lr_v:.5f}  "
                    f"drop={dr_v:.3f}  heads={heads_v}  sage={aggr_v}  "
                    f"gin={gin_v}  "
                    f"[{'valid' if ok else 'INVALID'}]")

    if not any(h['valid'] for h in history):
        logger.error("All init invalid."); return

    best_acc = max(Y_obs)
    logger.info(f"\nInit done. Best={best_acc:.4f}")

    logger.info(f"\n[Step 2] BO: {args.n_iter} iters  "
                f"novelty_w={args.novelty_w}  SEARCH_DIM={SEARCH_DIM}")

    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs).unsqueeze(-1)
        try:
            z_next = optimize_acq(X_t, Y_t, vae, args.novelty_w).cpu()
        except Exception as e:
            logger.warning(f"  [iter {it}] acqf error: {e}, random fallback")
            z_next = torch.cat([torch.randn(ARCH_NZ) * args.sigma_arch,
                                sample_hp_norm()])

        acc, lr_v, dr_v, heads_v, aggr_v, gin_v, ok = eval_z(vae, z_next, data, in_ch, out_ch)
        X_obs.append(z_next); Y_obs.append(acc)

        if acc > best_acc:
            best_acc = acc; tag = "<-- NEW BEST 🎯"
        else:
            tag = f"(best={best_acc:.4f})"
        logger.info(f"  iter {it:>3d}: val={acc:.4f}  lr={lr_v:.5f}  "
                    f"drop={dr_v:.3f}  heads={heads_v}  sage={aggr_v}  "
                    f"gin={gin_v}  "
                    f"{tag}{'  [invalid]' if not ok else ''}")

        history.append({'step': args.n_init+it, 'type': 'bo',
                        'val_acc': acc, 'lr': lr_v, 'dropout': dr_v,
                        'gat_heads': heads_v, 'sage_aggr': aggr_v,
                        'gin_eps': gin_v,
                        'best': best_acc, 'valid': ok})
        if (it+1) % 10 == 0:
            _save(history, X_obs, Y_obs, f'step{it}')

    # ── 最终结果 ─────────────────────────────────────────
    best_idx = Y_obs.index(max(Y_obs))
    best_z   = X_obs[best_idx]
    best_cfg = decode_arch(vae, best_z[:ARCH_NZ], n_trials=10)
    lr_b, dr_b, heads_b, aggr_b, gin_b = z_to_hp(best_z, best_cfg)

    logger.info("\n" + "="*66)
    logger.info(f"          Phase 3 v7 (nz={ARCH_NZ}, SEARCH_DIM={SEARCH_DIM}) Final")
    logger.info("="*66)
    logger.info(f"  Best val_acc  : {max(Y_obs):.4f}")
    logger.info(f"  GNN config    : {best_cfg}")
    logger.info(f"  LR={lr_b:.5f}  Dropout={dr_b:.3f}  "
                f"GAT heads={heads_b}  SAGE aggr={aggr_b}")
    logger.info(f"  GIN eps={gin_b}")
    logger.info(f"  gcnii_alpha={args.gcnii_alpha}  gcnii_theta={args.gcnii_theta}")
    logger.info("="*66)

    _save(history, X_obs, Y_obs, 'final')
    _plot(history)
    _compare(history)


def _save(history, X_obs, Y_obs, suffix):
    with open(os.path.join(args.output, f'history_{suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(X_obs[Y_obs.index(max(Y_obs))],
               os.path.join(args.output, f'best_z_search_{suffix}.pt'))
    logger.info(f"  Saved: {suffix}")


def _plot(history):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        steps  = [h['step']    for h in history]
        accs   = [h['val_acc'] for h in history]
        valids = [h.get('valid', True) for h in history]
        bests, cur = [], 0.0
        for h in history: cur = max(cur, h['val_acc']); bests.append(cur)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        vs = [s for s,v in zip(steps,valids) if v]
        va = [a for a,v in zip(accs, valids) if v]
        ix = [s for s,v in zip(steps,valids) if not v]
        ia = [a for a,v in zip(accs, valids) if not v]
        ax.scatter(vs, va, alpha=0.6, color='steelblue', label='Valid', zorder=3)
        ax.scatter(ix, ia, alpha=0.4, color='red', marker='x', label='Invalid', zorder=3)
        ax.plot(steps, bests, 'r-', lw=2, label='Best so far')
        ax.axvline(x=args.n_init-0.5, color='gray', ls='--', label='Init|BO')
        ax.set(xlabel='Step', ylabel='Val Accuracy',
               title=f'Phase 3 v7 (nz={ARCH_NZ}, SEARCH_DIM={SEARCH_DIM})')
        ax.legend(); ax.grid(alpha=0.3)

        ax2 = axes[1]
        lrs_v  = [h['lr']      for h in history if h.get('valid')]
        drs_v  = [h['dropout'] for h in history if h.get('valid')]
        accs_v = [h['val_acc'] for h in history if h.get('valid')]
        sc = ax2.scatter(lrs_v, drs_v, c=accs_v, cmap='RdYlGn',
                         alpha=0.7, s=60, vmin=0.4, vmax=0.85)
        plt.colorbar(sc, ax=ax2, label='Val Accuracy')
        ax2.set_xscale('log')
        ax2.set(xlabel='LR (log)', ylabel='Dropout',
                title='HP Landscape (LR vs Dropout)')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(args.output, 'convergence_phase3_v7.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")


def _compare(h3):
    p2 = 'results/bo_phase2/history_final.json'
    if not os.path.exists(p2): return
    best2 = max(e['val_acc'] for e in json.load(open(p2)))
    best3 = max(e['val_acc'] for e in h3)
    delta = best3 - best2
    logger.info(f"\n[Comparison]  Phase2: {best2:.4f}  "
                f"Phase3v7: {best3:.4f}  Δ={delta:+.4f}")


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    logger.info("=" * 65)
    logger.info(f"bo_phase3.py  v7  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)
    logger.info(f"Device={DEVICE}  ARCH_NZ={ARCH_NZ}  SEARCH_DIM={SEARCH_DIM}")
    logger.info(f"seed={args.seed}  version={args.version}")

    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")
    logger.info(f"Search space v3: z_arch({ARCH_NZ}d) + hp({HP_DIM}d) "
                f"= {SEARCH_DIM}d")

    run_bo(vae, data, in_ch, out_ch)

    elapsed = time.time() - start_time
    logger.info(f"\n✅ 运行完成  总耗时: {elapsed/60:.1f} min")
