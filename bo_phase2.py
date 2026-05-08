"""
bo_phase2.py  v7  (nz=8 + CosineAnnealingLR + Early Stopping)
================================================================
修改：全面对齐 8 维空间，更新了类属性和画图标签。
"""

import os, sys, json, argparse, urllib.request
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from nas_space import JointSpaceVAE
from eval_utils import train_and_eval_arch, DynamicGNN   # ← 新增

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  type=str,
                    default='results/joint_search/joint_model_ep100.pth') # ← 匹配你最新训练的 100 轮模型
parser.add_argument('--cora_root',   type=str, default='/tmp/Cora')
parser.add_argument('--n_init',      type=int, default=10)
parser.add_argument('--n_iter',      type=int, default=40)
parser.add_argument('--output',      type=str, default='results/bo_phase2')
parser.add_argument('--sigma',       type=float, default=0.8)
parser.add_argument('--eval_epochs', type=int, default=100)
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output, exist_ok=True)


class ArchArgs:
    def __init__(self):
        self.max_n           = 5
        self.num_vertex_type = 7
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = 8    # ← 核心修改：32 → 8
        self.bidirectional   = True


def load_vae(checkpoint_path: str) -> JointSpaceVAE:
    model = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"VAE loaded: {checkpoint_path}  (arch_nz=8)")
    return model


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
        print(f"Downloading {len(missing)} Cora files...")
        os.makedirs(raw_dir, exist_ok=True)
        for fname in missing:
            dest = os.path.join(raw_dir, fname)
            for mirror in _MIRRORS:
                if _try_download_file(f"{mirror}/{fname}", dest):
                    break
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset  = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    return dataset[0].to(DEVICE), dataset.num_features, dataset.num_classes


HIDDEN_DIM = 64


def evaluate_z_arch(vae, z_arch, data, in_ch, out_ch, eval_epochs=None):
    with torch.no_grad():
        z_hp    = torch.zeros(1, vae.hp_nz, device=DEVICE)
        z_total = torch.cat([z_arch.unsqueeze(0), z_hp], dim=-1)
        configs, _, _ = vae.decode_from_joint_latent(z_total)
    config = configs[0]

    val_acc, is_valid = train_and_eval_arch(
        config, data, in_ch, out_ch,
        lr=1e-3, dropout=0.5, device=DEVICE)
    return val_acc, is_valid


def _minmax_scale(X: torch.Tensor):
    X_min   = X.min(dim=0).values
    X_range = (X.max(dim=0).values - X_min).clamp(min=1e-8)
    return (X - X_min) / X_range, X_min, X_range


def fit_gp_and_optimize_logei(X_obs, Y_obs, arch_nz, sigma, X_min, X_range):
    X = X_obs.double()
    Y = Y_obs.double()
    gp  = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    log_ei = qLogExpectedImprovement(gp, best_f=Y.max())
    bounds = torch.zeros(2, arch_nz, dtype=torch.double)
    bounds[1] = 1.0
    candidate, _ = optimize_acqf(log_ei, bounds=bounds, q=1,
                                  num_restarts=10, raw_samples=1024)
    return candidate.squeeze(0).float() * X_range + X_min


def run_bo(vae, data, in_ch, out_ch):
    arch_nz = vae.arch_nz   # 8
    history, X_obs, Y_obs = [], [], []

    print(f"\n[Step 1] Random init: {args.n_init} points  arch_nz={arch_nz}")
    print(f"  Eval: max_epochs=150  patience=20  CosineAnnealingLR")
    for i in tqdm(range(args.n_init), desc='Init'):
        z = torch.randn(arch_nz, device=DEVICE) * args.sigma
        acc, valid = evaluate_z_arch(vae, z, data, in_ch, out_ch)
        X_obs.append(z.cpu()); Y_obs.append(acc)
        history.append({'step': i, 'type': 'init', 'val_acc': acc, 'valid': valid})
        print(f"  init {i:>2d}: val_acc={acc:.4f}  [{'valid' if valid else 'INVALID'}]")

    if not any(h['valid'] for h in history):
        print("All init invalid."); return
    best_acc = max(Y_obs)
    print(f"\nInit done. Best val_acc={best_acc:.4f}")

    print(f"\n[Step 2] BO: {args.n_iter} iters")
    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs).unsqueeze(-1)
        Y_norm = (Y_t - Y_t.mean()) / Y_t.std().clamp(min=1e-6)
        X_norm, X_min, X_range = _minmax_scale(X_t)

        z_next = fit_gp_and_optimize_logei(
            X_norm, Y_norm, arch_nz, args.sigma, X_min, X_range
        ).to(DEVICE)

        acc, valid = evaluate_z_arch(vae, z_next, data, in_ch, out_ch)
        X_obs.append(z_next.cpu()); Y_obs.append(acc)

        if acc > best_acc: best_acc = acc; tag = "<-- NEW BEST"
        else:              tag = f"(best={best_acc:.4f})"
        print(f"  iter {it:>3d}: val_acc={acc:.4f}  {tag}"
              f"{'  [invalid]' if not valid else ''}")
        history.append({'step': args.n_init+it, 'type': 'bo',
                        'val_acc': acc, 'best': best_acc, 'valid': valid})
        if (it+1) % 10 == 0:
            _save(history, X_obs, Y_obs, f'step{it}')

    best_idx = Y_obs.index(max(Y_obs))
    best_z   = X_obs[best_idx].to(DEVICE)
    with torch.no_grad():
        z_total = torch.cat([best_z.unsqueeze(0),
                             torch.zeros(1, vae.hp_nz, device=DEVICE)], dim=-1)
        configs, _, _ = vae.decode_from_joint_latent(z_total)

    print("\n" + "="*55)
    print("       Phase 2 (nz=8) BO Final Result")
    print("="*55)
    print(f"  Best val_acc : {max(Y_obs):.4f}")
    print(f"  GNN config   : {configs[0]}")
    print("="*55)

    _save(history, X_obs, Y_obs, 'final')
    _plot(history)


def _save(history, X_obs, Y_obs, suffix):
    with open(os.path.join(args.output, f'history_{suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(X_obs[Y_obs.index(max(Y_obs))],
               os.path.join(args.output, f'best_z_arch_{suffix}.pt'))


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
        ax.set_title('Phase 2 BO (nz=8, fixed HP)')
        ax.legend(); ax.grid(alpha=0.3)

        path = os.path.join(args.output, 'convergence_nz8.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {path}")
    except ImportError: pass


if __name__ == '__main__':
    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    print(f"Cora: {in_ch} features, {out_ch} classes  "
          f"train={data.train_mask.sum().item()}  val={data.val_mask.sum().item()}")
    run_bo(vae, data, in_ch, out_ch)