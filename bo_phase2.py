"""
bo_phase2.py  v8  (对齐 Search Space v3 + GCNII 支持 + 工业级日志)
====================================================================
v8 变更（相对 v7）：

  [对齐 1] ArchArgs 强制更新为 v3 空间
      max_n=7, num_vertex_type=8, nz=12
      （JointSpaceVAE.__init__ 会强制覆盖，此处保持与 v3 一致）

  [对齐 2] GCNII 兼容
      评估函数调用时补充传入 gcnii_alpha=0.1, gcnii_theta=0.5

  [新增 3] 工业级日志系统
      - setup_logger 实现控制台 + 文件双端输出
      - 日志路径：{log_dir}/bo_phase2/train_{version}_{MMDD_HHMM}.log
      - 新增参数：--version, --log_dir, --seed,
                  --weight_decay, --dropout, --hidden_dim,
                  --gcnii_alpha, --gcnii_theta
      - 完整参数字典持久化为 .json
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

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from nas_space import JointSpaceVAE
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
parser = argparse.ArgumentParser(description='bo_phase2 v8')
# 原有参数（不变）
parser.add_argument('--checkpoint',  type=str,
                    default='results/joint_search/joint_model_v3_ep100.pth')
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_init',      type=int, default=10)
parser.add_argument('--n_iter',      type=int, default=40)
parser.add_argument('--output',      type=str, default='results/bo_phase2')
parser.add_argument('--sigma',       type=float, default=0.8)
parser.add_argument('--eval_epochs', type=int, default=100)
# 新增参数
parser.add_argument('--version',     type=str, default='v3_final')
parser.add_argument('--log_dir',     type=str, default='logs/')
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout',     type=float, default=0.5)
parser.add_argument('--hidden_dim',  type=int,   default=64)
parser.add_argument('--gcnii_alpha', type=float, default=0.1)
parser.add_argument('--gcnii_theta', type=float, default=0.5)
parser.add_argument('--patience',    type=int,   default=20)
args = parser.parse_args()

# ── 全局常量 ─────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output, exist_ok=True)

# ── 日志初始化 ────────────────────────────────────────────
logger, log_filepath = setup_logger(args.log_dir, 'bo_phase2', args.version)
save_args_json(args, log_filepath)


# ============================================================
# ArchArgs — v3 对齐（max_n=7, nvt=8, nz=12）
# ============================================================
class ArchArgs:
    def __init__(self):
        self.max_n           = 7    # v3
        self.num_vertex_type = 8    # v3
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = 12   # v3
        self.bidirectional   = True


def load_vae(checkpoint_path: str) -> JointSpaceVAE:
    model = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"VAE loaded: {checkpoint_path}  (arch_nz={model.arch_nz})")
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
    "https://cdn.jsdelivr.net/gh/kimiyoung/planetoid@master/data",
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
]


def _try_download_file(url, dest, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 10: return False
        with open(dest, "wb") as f: f.write(data)
        return True
    except Exception: return False


def load_cora(root):
    raw_dir = os.path.join(root, "raw")
    missing = [f for f in _CORA_FILES
               if not os.path.exists(os.path.join(raw_dir, f))
               or os.path.getsize(os.path.join(raw_dir, f)) < 10]
    if missing:
        logger.info(f"下载 {len(missing)} 个 Cora 文件...")
        os.makedirs(raw_dir, exist_ok=True)
        for fname in missing:
            dest = os.path.join(raw_dir, fname)
            ok = any(_try_download_file(f"{m}/{fname}", dest) for m in _MIRRORS)
            logger.info(f"  {fname}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset  = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    return dataset[0].to(DEVICE), dataset.num_features, dataset.num_classes


HIDDEN_DIM = 64


def evaluate_z_arch(vae, z_arch, data, in_ch, out_ch):
    """评估单个架构隐向量（仅搜索架构，HP 固定默认值）。"""
    with torch.no_grad():
        z_hp    = torch.zeros(1, vae.hp_nz, device=DEVICE)
        z_total = torch.cat([z_arch.unsqueeze(0), z_hp], dim=-1)
        configs, _, _ = vae.decode_from_joint_latent(z_total)
    config = configs[0]

    # [v8 对齐] 补充 gcnii_alpha / gcnii_theta
    val_acc, is_valid = train_and_eval_arch(
        config, data, in_ch, out_ch,
        lr           = 1e-3,
        dropout      = args.dropout,
        hidden_dim   = args.hidden_dim,
        weight_decay = args.weight_decay,
        gcnii_alpha  = args.gcnii_alpha,
        gcnii_theta  = args.gcnii_theta,
        device       = DEVICE,
        max_epochs   = args.eval_epochs,
        patience     = args.patience,
        track_test   = False,
    )
    return val_acc, is_valid


def _minmax_scale(X: torch.Tensor):
    X_min   = X.min(dim=0).values
    X_range = (X.max(dim=0).values - X_min).clamp(min=1e-8)
    return (X - X_min) / X_range, X_min, X_range


def fit_gp_and_optimize_logei(X_obs, Y_obs, arch_nz, X_min, X_range):
    X = X_obs.double()
    Y = Y_obs.double()
    gp  = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    log_ei = qLogExpectedImprovement(gp, best_f=Y.max())
    bounds = torch.zeros(2, arch_nz, dtype=torch.double)
    bounds[1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidate, _ = optimize_acqf(log_ei, bounds=bounds, q=1,
                                     num_restarts=10, raw_samples=1024)
    return candidate.squeeze(0).float() * X_range + X_min


# ============================================================
# 主 BO 循环
# ============================================================
def run_bo(vae, data, in_ch, out_ch):
    arch_nz = vae.arch_nz   # 12
    history, X_obs, Y_obs = [], [], []

    logger.info(f"\n[Step 1] Random init: {args.n_init} points  arch_nz={arch_nz}")
    logger.info(f"  Eval: max_epochs={args.eval_epochs}  "
                f"patience={args.patience}  CosineAnnealingLR")
    logger.info(f"  dropout={args.dropout}  hidden_dim={args.hidden_dim}  "
                f"wd={args.weight_decay}")
    logger.info(f"  gcnii_alpha={args.gcnii_alpha}  gcnii_theta={args.gcnii_theta}")

    for i in tqdm(range(args.n_init), desc='Init'):
        z   = torch.randn(arch_nz, device=DEVICE) * args.sigma
        acc, valid = evaluate_z_arch(vae, z, data, in_ch, out_ch)
        X_obs.append(z.cpu()); Y_obs.append(acc)
        history.append({
            'step': i, 'type': 'init', 'val_acc': acc, 'valid': valid})
        logger.info(f"  init {i:>2d}: val_acc={acc:.4f}  "
                    f"[{'valid' if valid else 'INVALID'}]")

    if not any(h['valid'] for h in history):
        logger.error("All init invalid."); return

    best_acc = max(Y_obs)
    logger.info(f"\nInit done. Best val_acc={best_acc:.4f}")

    logger.info(f"\n[Step 2] BO: {args.n_iter} iters")
    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs).unsqueeze(-1)
        Y_norm = (Y_t - Y_t.mean()) / Y_t.std().clamp(min=1e-6)
        X_norm, X_min, X_range = _minmax_scale(X_t)

        try:
            z_next = fit_gp_and_optimize_logei(
                X_norm, Y_norm, arch_nz, X_min, X_range).to(DEVICE)
        except Exception as e:
            logger.warning(f"  [iter {it}] GP optimize failed: {e}, random fallback")
            z_next = torch.randn(arch_nz, device=DEVICE) * args.sigma

        acc, valid = evaluate_z_arch(vae, z_next, data, in_ch, out_ch)
        X_obs.append(z_next.cpu()); Y_obs.append(acc)

        if acc > best_acc:
            best_acc = acc; tag = "<-- NEW BEST 🎯"
        else:
            tag = f"(best={best_acc:.4f})"

        logger.info(f"  iter {it:>3d}: val_acc={acc:.4f}  {tag}"
                    f"{'  [invalid]' if not valid else ''}")
        history.append({
            'step': args.n_init+it, 'type': 'bo',
            'val_acc': acc, 'best': best_acc, 'valid': valid})

        if (it+1) % 10 == 0:
            _save(history, X_obs, Y_obs, f'step{it}')

    # ── 最终结果 ─────────────────────────────────────────
    best_idx = Y_obs.index(max(Y_obs))
    best_z   = X_obs[best_idx].to(DEVICE)
    with torch.no_grad():
        z_total = torch.cat([best_z.unsqueeze(0),
                             torch.zeros(1, vae.hp_nz, device=DEVICE)], dim=-1)
        configs, _, _ = vae.decode_from_joint_latent(z_total)

    logger.info("\n" + "="*55)
    logger.info("       Phase 2 v8 (nz=12) BO Final Result")
    logger.info("="*55)
    logger.info(f"  Best val_acc : {max(Y_obs):.4f}")
    logger.info(f"  GNN config   : {configs[0]}")
    logger.info("="*55)

    _save(history, X_obs, Y_obs, 'final')
    _plot(history)


def _save(history, X_obs, Y_obs, suffix):
    with open(os.path.join(args.output, f'history_{suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(X_obs[Y_obs.index(max(Y_obs))],
               os.path.join(args.output, f'best_z_arch_{suffix}.pt'))
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

        v_s = [s for s,v in zip(steps,valids) if v]
        v_a = [a for a,v in zip(accs, valids) if v]
        i_s = [s for s,v in zip(steps,valids) if not v]
        i_a = [a for a,v in zip(accs, valids) if not v]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(v_s, v_a, alpha=0.6, color='steelblue', label='Valid arch', zorder=3)
        ax.scatter(i_s, i_a, alpha=0.4, color='red', marker='x', label='Invalid', zorder=3)
        ax.plot(steps, bests, 'r-', lw=2, label='Best so far')
        ax.axvline(x=args.n_init - 0.5, color='gray', ls='--', label='Init|BO')
        ax.set_xlabel('Step'); ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'Phase 2 v8 BO (nz=12, fixed HP)  best={max(bests):.4f}')
        ax.legend(); ax.grid(alpha=0.3)

        path = os.path.join(args.output, 'convergence_nz12_v8.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"bo_phase2.py  v8  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"Device={DEVICE}  arch_nz=12  seed={args.seed}")
    logger.info(f"checkpoint: {args.checkpoint}")

    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes  "
                f"train={data.train_mask.sum().item()}  "
                f"val={data.val_mask.sum().item()}")

    run_bo(vae, data, in_ch, out_ch)

    elapsed = time.time() - start_time
    logger.info(f"\n✅ 运行完成  总耗时: {elapsed/60:.1f} min")