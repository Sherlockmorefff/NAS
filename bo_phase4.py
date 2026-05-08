"""
bo_phase4.py  v7  (Search Space v3: ARCH_NZ=12, SEARCH_DIM=16, GCNII)
======================================================================
v7 变更（相对 v6）：
  - ARCH_NZ  : 8  → 12  (适配 v3 VAE)
  - SEARCH_DIM: 12 → 16  (12 arch + 4 HP)
  - ArchArgs : max_n=7, num_vertex_type=8
  - eval_z   : 传递 gcnii_alpha / gcnii_theta（使用默认值）
  - HP 切片索引全部通过 ARCH_NZ 常量动态计算，无需手动修改

⚠️  需要使用 v3 VAE 权重（train_joint.py 重新训练后生成）。
"""

import os, sys, json, argparse, warnings, urllib.request
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/mnt/project')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning)

from scipy.stats import qmc
from scipy.stats import norm as scipy_norm

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from nas_space import JointSpaceVAE
from dvae_differentiable import build_differentiable_dvae
from jacobian_utils import analyze_latent_smoothness
from acqf_geometric import GPNDNASNovelty, make_acqf_and_optimize
from eval_utils import eval_z_search, DynamicGNN, norm_to_hidden, norm_to_l2

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',      type=str,
                    default='results/joint_search/joint_model_v3_ep100.pth')
parser.add_argument('--warm_start',      type=str,
                    default='results/bo_phase2/best_z_arch_final.pt')
parser.add_argument('--n_eval_seeds', type=int, default=3,
                    help='每个候选架构的评估种子数（推荐 3，单次为 1）')
parser.add_argument('--cora_root',       type=str, default='/tmp/Cora')
parser.add_argument('--n_init',          type=int, default=20)
parser.add_argument('--n_iter',          type=int, default=60)
parser.add_argument('--output',          type=str, default='results/bo_phase4_v3')
parser.add_argument('--sigma_arch',      type=float, default=0.8)
parser.add_argument('--eval_epochs',     type=int, default=100)
parser.add_argument('--novelty_w',       type=float, default=0.15)
parser.add_argument('--tau_gumbel',      type=float, default=0.3)
parser.add_argument('--n_probes',        type=int, default=12)   # ≤ ARCH_NZ=12
parser.add_argument('--log_lr_min',      type=float, default=-4.0)
parser.add_argument('--log_lr_max',      type=float, default=-1.5)
parser.add_argument('--dropout_min',     type=float, default=0.1)
parser.add_argument('--dropout_max',     type=float, default=0.6)
parser.add_argument('--lhs_oversample',  type=int, default=3)
parser.add_argument('--stagnation_k',    type=int, default=12)
parser.add_argument('--skip_smoothness', action='store_true')
parser.add_argument('--gpnd_alpha',      type=float, default=0.2)
parser.add_argument('--gpnd_beta',       type=float, default=0.3)
parser.add_argument('--gpnd_gamma',      type=float, default=0.3)
parser.add_argument('--gpnd_delta',      type=float, default=0.2)
parser.add_argument('--gcnii_alpha',     type=float, default=0.1)
parser.add_argument('--gcnii_theta',     type=float, default=0.5)
args = parser.parse_args()

DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ARCH_NZ    = 12    # ← v3: 8 → 12
SEARCH_DIM = ARCH_NZ + 4   # 16

os.makedirs(args.output, exist_ok=True)


# ============================================================
# HP 反归一化
# ============================================================
def z_to_hp(z_search: torch.Tensor):
    """从搜索向量中提取并反归一化 HP 值。
    切片索引全部由 ARCH_NZ 动态决定，无需手动维护。
    """
    log_lr_n = float(z_search[ARCH_NZ])
    drop_n   = float(z_search[ARCH_NZ + 1])
    log_lr   = log_lr_n * (args.log_lr_max - args.log_lr_min) + args.log_lr_min
    dropout  = drop_n   * (args.dropout_max - args.dropout_min) + args.dropout_min
    lr       = float(np.clip(10 ** np.clip(log_lr, -6, 0), 1e-6, 1.0))
    dr       = float(np.clip(dropout, 0.0, 0.9))

    hd = norm_to_hidden(float(z_search[ARCH_NZ + 2])) \
         if z_search.shape[0] > ARCH_NZ + 2 else 64
    l2 = norm_to_l2(float(z_search[ARCH_NZ + 3])) \
         if z_search.shape[0] > ARCH_NZ + 3 else 5e-4

    return lr, dr, hd, l2


def sample_hp_norm():
    return torch.rand(4)


# ============================================================
# VAE / Cora 加载
# ============================================================
class ArchArgs:
    """v3 ArchArgs — JointSpaceVAE.__init__ 会强制覆盖 max_n/nvt，
    但保留这里方便直接阅读。"""
    max_n           = 7    # ← v3
    num_vertex_type = 8    # ← v3
    START_TYPE      = 0
    END_TYPE        = 1
    hs              = 501
    nz              = ARCH_NZ   # 12
    bidirectional   = True


def load_vae(ckpt):
    model = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"VAE loaded: {ckpt}  (arch_nz={ARCH_NZ}, max_n=7, nvt=8)")
    return model


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
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = r.read()
        if len(d)<10: return False
        open(dest,"wb").write(d); return True
    except Exception: return False


def load_cora(root):
    raw = os.path.join(root, "raw")
    miss = [f for f in _CORA_FILES if not os.path.exists(os.path.join(raw, f))]
    if miss:
        os.makedirs(raw, exist_ok=True)
        for f in miss:
            ok = any(_dl(f"{m}/{f}", os.path.join(raw, f)) for m in _MIRRORS)
            print(f"  {f}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root)=="Cora" else root
    ds = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    return ds[0].to(DEVICE), ds.num_features, ds.num_classes


# ============================================================
# 评估函数
# ============================================================
def eval_z(vae, z_search, data, in_ch, out_ch, epochs=None):
    return eval_z_search(
        vae, z_search, data, in_ch, out_ch,
        arch_nz     = ARCH_NZ,
        log_lr_min  = args.log_lr_min,
        log_lr_max  = args.log_lr_max,
        dropout_min = args.dropout_min,
        dropout_max = args.dropout_max,
        device      = DEVICE,
        has_hidden  = True,
        has_l2      = True,
        gcnii_alpha = args.gcnii_alpha,
        gcnii_theta = args.gcnii_theta,
    )


# ============================================================
# 架构解码辅助
# ============================================================
def decode_arch(vae, z_arch, n_trials=5):
    from collections import Counter
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(DEVICE))
            g = graphs[0]
            ops = [vae.op_mapping.get(g.vs[i]['type'], 'Identity')
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
# 舒尔补贪心初始点选择（算法本体不变，自动适配新维度）
# ============================================================
def schur_greedy_select(
    X_norm: np.ndarray,
    Y_cand: np.ndarray,
    n_target: int,
    noise_var: float = 1e-4,
    lengthscale: float = 0.5,
    outputscale: float = 1.0,
) -> list:
    """Matern-5/2 ARD 核 + 舒尔补贪心方差最大化的初始点选择。
    算法与 v6 完全相同，核矩阵自动适配 SEARCH_DIM=16 输入。
    """
    N, D = X_norm.shape

    if N <= n_target:
        print(f"  [Schur Greedy] 候选池 ({N}) ≤ n_target ({n_target})，全部返回")
        return list(range(N))

    X      = X_norm.astype(np.float64)
    ls     = np.full(D, lengthscale, dtype=np.float64)
    diff   = (X[:, None, :] - X[None, :, :]) / ls
    r2     = (diff ** 2).sum(-1)
    r      = np.sqrt(np.maximum(r2, 0.0))
    sqrt5r = np.sqrt(5.0) * r
    K = outputscale * (1.0 + sqrt5r + (5.0 / 3.0) * r2) * np.exp(-sqrt5r)
    K += noise_var * np.eye(N)

    print(f"  [Schur Greedy] K ({N}×{N})  D={D}  ls={lengthscale}  noise={noise_var:.0e}")

    seed_idx  = int(np.argmax(Y_cand))
    selected  = [seed_idx]
    remaining = list(range(N))
    remaining.remove(seed_idx)

    L = np.array([[np.sqrt(max(K[seed_idx, seed_idx], 1e-12))]], dtype=np.float64)

    for step in range(1, n_target):
        if not remaining:
            break
        rem        = np.array(remaining, dtype=np.int32)
        K_S_rem    = K[np.ix_(selected, rem)]
        v          = np.linalg.solve(L, K_S_rem)
        K_rem_diag = K[rem, rem]
        post_var   = np.maximum(K_rem_diag - (v ** 2).sum(axis=0), 0.0)

        best_local  = int(np.argmax(post_var))
        best_global = remaining[best_local]
        var_best    = post_var[best_local]

        selected.append(best_global)
        remaining.remove(best_global)

        k_new = K[selected[:-1], best_global]
        v_new = np.linalg.solve(L, k_new)
        ell   = np.sqrt(max(var_best, 1e-12))
        s     = len(selected)
        L_new = np.zeros((s, s), dtype=np.float64)
        L_new[:s-1, :s-1] = L
        L_new[s-1, :s-1]  = v_new
        L_new[s-1, s-1]   = ell
        L = L_new

        if step % 5 == 0 or step == n_target - 1:
            print(f"    step {step:>2d}/{n_target-1}  "
                  f"|S|={len(selected)}  max_post_var={var_best:.5f}")

    print(f"  [Schur Greedy] 完成：选出 {len(selected)} / {N} 个初始点")
    return selected


# ============================================================
# LHS 初始化（不变，SEARCH_DIM 自动生效）
# ============================================================
def lhs_init(vae, data, in_ch, out_ch, n_target: int, oversample: int = 3):
    n_sample = n_target * oversample
    print(f"\n[LHS Init] {n_sample} 样本 → 目标 {n_target} 有效点  "
          f"(SEARCH_DIM={SEARCH_DIM}, ARCH_NZ={ARCH_NZ})")

    sampler  = qmc.LatinHypercube(d=SEARCH_DIM, seed=42)
    lhs_raw  = sampler.random(n=n_sample)
    arch_pts = scipy_norm.ppf(
        np.clip(lhs_raw[:, :ARCH_NZ], 0.001, 0.999)
    ) * args.sigma_arch
    hp_pts   = lhs_raw[:, ARCH_NZ:]
    z_pts    = np.concatenate([arch_pts, hp_pts], axis=1)

    warm_pts = []
    if os.path.exists(args.warm_start):
        z_best = torch.load(args.warm_start, map_location='cpu', weights_only=True)
        # Warm start 来自旧 Phase 2（nz=8），自动截断或填充至 ARCH_NZ=12
        if z_best.shape[0] > ARCH_NZ:
            z_best = z_best[:ARCH_NZ]
        elif z_best.shape[0] < ARCH_NZ:
            z_best = torch.cat([z_best, torch.zeros(ARCH_NZ - z_best.shape[0])])
        n_warm = min(3, n_target // 4)
        for _ in range(n_warm):
            warm_pts.append(np.concatenate([
                (z_best + torch.randn(ARCH_NZ) * 0.3).numpy(),
                np.random.rand(4)
            ]))
        print(f"  + {n_warm} warm-start 点 (旧 Phase 2 best, 填充至 nz={ARCH_NZ})")

    all_pts = np.array(warm_pts + list(z_pts))

    X_all, Y_all, hist_all = [], [], []
    pbar = tqdm(all_pts, desc='LHS eval')
    for raw in pbar:
        z = torch.tensor(raw, dtype=torch.float32)
        acc, lr_v, dr_v, hd_v, l2_v, ok = eval_z(vae, z, data, in_ch, out_ch)
        X_all.append(z)
        Y_all.append(acc)
        hist_all.append({
            'step': len(hist_all) - 1, 'type': 'lhs',
            'val_acc': acc, 'lr': lr_v, 'dropout': dr_v,
            'hidden': hd_v, 'l2': l2_v, 'valid': ok
        })
        if ok:
            pbar.set_postfix({'valid_best': f"{max(Y_all):.4f}"})

    valid_idx   = [i for i, h in enumerate(hist_all) if h['valid']]
    invalid_idx = [i for i, h in enumerate(hist_all) if not h['valid']]
    best_so_far = max(Y_all) if Y_all else 0.0
    print(f"\n  LHS: {len(valid_idx)}/{len(X_all)} 有效  best={best_so_far:.4f}")

    if len(valid_idx) < 2:
        print("  ⚠️  有效点不足 2，返回全部点")
        return X_all, Y_all, hist_all

    if len(valid_idx) <= n_target:
        print(f"  ⚠️  有效点 ({len(valid_idx)}) ≤ n_target ({n_target})，直接返回")
        return X_all, Y_all, hist_all

    X_valid = torch.stack([X_all[i] for i in valid_idx]).numpy()
    Y_valid = np.array([Y_all[i] for i in valid_idx])
    X_min   = X_valid.min(axis=0)
    X_range = np.maximum(X_valid.max(axis=0) - X_min, 1e-8)
    X_norm  = (X_valid - X_min) / X_range

    greedy_local = schur_greedy_select(
        X_norm, Y_valid, n_target=n_target,
        noise_var=1e-4, lengthscale=0.5, outputscale=1.0)

    selected_global = [valid_idx[i] for i in greedy_local]
    final_idx = sorted(set(selected_global + invalid_idx))
    X_obs   = [X_all[i]    for i in final_idx]
    Y_obs   = [Y_all[i]    for i in final_idx]
    history = [hist_all[i] for i in final_idx]

    n_valid_kept = sum(1 for h in history if h['valid'])
    print(f"  Schur Greedy → 保留 {len(X_obs)} 点  "
          f"(有效={n_valid_kept}  无效={len(invalid_idx)})")
    return X_obs, Y_obs, history


# ============================================================
# 主 BO 循环
# ============================================================
def run_bo(vae, dvae_diff, data, in_ch, out_ch):
    if not args.skip_smoothness:
        print("\n[Smoothness Pre-analysis]")
        try:
            analyze_latent_smoothness(
                dvae_diff, n_pairs=5, nz=ARCH_NZ,
                tau=args.tau_gumbel, method='random_proj', device='cpu')
        except Exception as e:
            print(f"  error: {e}")

    X_obs, Y_obs, history = lhs_init(
        vae, data, in_ch, out_ch,
        n_target=args.n_init, oversample=args.lhs_oversample)

    if not any(h['valid'] for h in history):
        print("❌ All init invalid."); return

    best_acc     = max(Y_obs)
    stagnant_cnt = 0
    print(f"\nInit complete. Best={best_acc:.4f}  N={len(X_obs)}")
    print(f"Eval: max_epochs=150  patience=20  CosineAnnealingLR")

    safe_n_probes = min(args.n_probes, ARCH_NZ)
    gpnd = GPNDNASNovelty(
        dvae_diff,
        arch_nz  = ARCH_NZ,
        tau      = args.tau_gumbel,
        n_probes = safe_n_probes,
        alpha    = args.gpnd_alpha,
        beta     = args.gpnd_beta,
        gamma    = args.gpnd_gamma,
        delta    = args.gpnd_delta,
    )

    print(f"\n[Step 2] BO: {args.n_iter} iters  "
          f"novelty_w={args.novelty_w}  SEARCH_DIM={SEARCH_DIM}")

    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs, dtype=torch.float32).unsqueeze(-1)

        if stagnant_cnt >= args.stagnation_k:
            best_z_all = X_obs[Y_obs.index(best_acc)]
            best_hp    = best_z_all[ARCH_NZ:].clone()
            new_hp     = (1.0 - best_hp + torch.randn(4) * 0.1).clamp(0.0, 1.0)
            z_next     = torch.cat([
                torch.randn(ARCH_NZ) * args.sigma_arch * 0.5, new_hp])
            stagnant_cnt = 0
            tag_extra = " [RESTART→new HP]"
        else:
            try:
                z_next = make_acqf_and_optimize(
                    X_t, Y_t, gpnd,
                    search_dim     = SEARCH_DIM,
                    novelty_weight = args.novelty_w,
                    num_restarts   = 8,
                    raw_samples    = 256,
                    n_extra        = 128,
                ).cpu()
            except Exception as e:
                print(f"  [iter {it}] acqf error: {e}, random fallback")
                z_next = torch.cat([
                    torch.randn(ARCH_NZ) * args.sigma_arch, sample_hp_norm()])
            tag_extra = ""

        acc, lr_v, dr_v, hd_v, l2_v, ok = eval_z(vae, z_next, data, in_ch, out_ch)
        X_obs.append(z_next.cpu()); Y_obs.append(acc)

        if acc > best_acc:
            best_acc = acc; stagnant_cnt = 0
            tag = "<-- NEW BEST 🎯"
        else:
            stagnant_cnt += 1
            tag = f"(best={best_acc:.4f}  stag={stagnant_cnt})"

        print(f"  iter {it:>3d}: val={acc:.4f}  lr={lr_v:.5f}  "
              f"drop={dr_v:.3f}  hd={hd_v}  l2={l2_v:.0e}  "
              f"{tag}{'  [invalid]' if not ok else ''}{tag_extra}")

        history.append({'step': args.n_init+it, 'type': 'bo',
                        'val_acc': acc, 'lr': lr_v, 'dropout': dr_v,
                        'hidden': hd_v, 'l2': l2_v,
                        'best': best_acc, 'valid': ok})

        if (it+1) % 10 == 0:
            _save(history, X_obs, Y_obs, f'step{it}')

    best_idx = Y_obs.index(max(Y_obs))
    best_z   = X_obs[best_idx]
    best_cfg = decode_arch(vae, best_z[:ARCH_NZ], n_trials=10)
    lr_b, dr_b, hd_b, l2_b = z_to_hp(best_z)

    print("\n" + "="*70)
    print(f"  Phase 4 v7 (nz={ARCH_NZ}, GPND-NAS, SEARCH_DIM={SEARCH_DIM}) Final")
    print("="*70)
    print(f"  Best val_acc  : {max(Y_obs):.4f}")
    print(f"  GNN config    : {best_cfg}")
    print(f"  LR={lr_b:.5f}  Dropout={dr_b:.3f}  Hidden={hd_b}  L2={l2_b:.0e}")
    print(f"  gcnii_alpha={args.gcnii_alpha}  gcnii_theta={args.gcnii_theta}")
    print("="*70)

    _save(history, X_obs, Y_obs, 'final')
    _compare(history)


def _save(history, X_obs, Y_obs, suffix):
    with open(os.path.join(args.output, f'history_{suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(X_obs[Y_obs.index(max(Y_obs))],
               os.path.join(args.output, f'best_z_{suffix}.pt'))
    print(f"  Saved: {suffix}")


def _compare(h4):
    print("\n[Phase Comparison]")
    best4 = max(e['val_acc'] for e in h4)
    for phase, path in [
        ('Phase 2', 'results/bo_phase2/history_final.json'),
        ('Phase 3', 'results/bo_phase3/history_final.json'),
    ]:
        if os.path.exists(path):
            bprev = max(e['val_acc'] for e in json.load(open(path)))
            d = best4 - bprev
            print(f"  vs {phase}: {bprev:.4f} → {best4:.4f}  "
                  f"{'✅ +' if d>0 else '❌ '}{d:.4f}")


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"arch_nz={ARCH_NZ}  search_dim={SEARCH_DIM}  "
          f"novelty_w={args.novelty_w}  tau={args.tau_gumbel}")
    print(f"hidden_dim: {{16,32,64,128,256,512}}  l2: {{1e-4..5e-4}}")
    print(f"GCNII: alpha={args.gcnii_alpha}  theta={args.gcnii_theta}")

    vae       = load_vae(args.checkpoint)
    dvae_diff = build_differentiable_dvae(vae)

    print(f"\nDVAE: p_dim={dvae_diff.p_dim}  (expected 84)")
    assert dvae_diff.p_dim == 84, f"p_dim mismatch: {dvae_diff.p_dim} != 84"
    dvae_diff.check_gradient_flow(nz=ARCH_NZ)

    data, in_ch, out_ch = load_cora(args.cora_root)
    print(f"Cora: {in_ch} features  {out_ch} classes")

    run_bo(vae, dvae_diff, data, in_ch, out_ch)