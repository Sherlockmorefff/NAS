"""
jacobian_utils.py  v3  (Search Space v3: default nz=12)
========================================================
v3 変更：
  - analyze_latent_smoothness のデフォルト nz: 8 → 12
  - __main__ テストを nz=12/n_probes=12 に更新
  - random_proj の actual_n = min(n_probes, nz) ロジックは変更なし
    (n_probes デフォルト 16 → min(16, 12) = 12 に自動クランプ)
  v2 の次元修正バグフィックスはそのまま保持。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
import warnings


# ======================================================================
# Jacobian 計算
# ======================================================================

def compute_jacobian(
    dvae_diff,
    z: torch.Tensor,
    tau: float = 0.3,
    method: str = 'finite_diff',
    eps: float = 1e-3,
    n_probes: int = 16,
) -> Optional[torch.Tensor]:
    try:
        if method == 'autograd':
            return {'matrix': _jacobian_autograd(dvae_diff, z, tau),
                    'probes': None, 'mode': 'autograd'}
        elif method == 'finite_diff':
            return {'matrix': _jacobian_finite_diff(dvae_diff, z, tau, eps),
                    'probes': None, 'mode': 'finite_diff'}
        elif method == 'random_proj':
            return _jacobian_random_proj(dvae_diff, z, tau, eps, n_probes)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        warnings.warn(f"Jacobian computation failed: {e}")
        return None


def _jacobian_autograd(dvae_diff, z, tau):
    def fn(z_):
        return dvae_diff.soft_decode(z_.unsqueeze(0), tau=tau,
                                     deterministic=True).squeeze(0)
    z_req = z.detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(fn, z_req, create_graph=False)
    return J.detach()


def _jacobian_finite_diff(dvae_diff, z, tau, eps):
    nz = z.shape[0]
    z0 = z.detach()
    with torch.no_grad():
        p0 = dvae_diff.soft_decode(
            z0.unsqueeze(0), tau=tau, deterministic=True).squeeze(0)
    P_dim = p0.shape[0]
    J     = torch.zeros(P_dim, nz, device=z.device)
    for d in range(nz):
        z_pos = z0.clone(); z_pos[d] += eps
        z_neg = z0.clone(); z_neg[d] -= eps
        with torch.no_grad():
            p_pos = dvae_diff.soft_decode(
                z_pos.unsqueeze(0), tau=tau, deterministic=True).squeeze(0)
            p_neg = dvae_diff.soft_decode(
                z_neg.unsqueeze(0), tau=tau, deterministic=True).squeeze(0)
        J[:, d] = (p_pos - p_neg) / (2 * eps)
    return J


def _jacobian_random_proj(dvae_diff, z, tau, eps, n_probes):
    """
    随机投影 Jacobian 估计。
    actual_n = min(n_probes, nz) — 防止 n_probes > nz 时的维度不匹配。
    """
    nz        = z.shape[0]
    actual_n  = min(n_probes, nz)
    P_dim     = dvae_diff.p_dim
    z0        = z.detach()

    raw    = torch.randn(actual_n, nz, device=z.device)
    Q, _   = torch.linalg.qr(raw.T)
    probes = Q.T                          # (actual_n, nz)

    J_approx = torch.zeros(P_dim, actual_n, device=z.device)
    for i, v in enumerate(probes):
        v  = F.normalize(v, dim=0)
        zp = z0 + eps * v
        zm = z0 - eps * v
        with torch.no_grad():
            pp = dvae_diff.soft_decode(
                zp.unsqueeze(0), tau=tau, deterministic=True).squeeze(0)
            pm = dvae_diff.soft_decode(
                zm.unsqueeze(0), tau=tau, deterministic=True).squeeze(0)
        J_approx[:, i] = (pp - pm) / (2 * eps)

    return {'matrix': J_approx, 'probes': probes, 'mode': 'random_proj'}


# ======================================================================
# SVD 分解 + 方向提取
# ======================================================================

def svd_decompose(
    J_input,
    top_k_orth: int = 4,
    top_k_tang: int = 4,
    min_singular_ratio: float = 0.01,
) -> Dict:
    if isinstance(J_input, dict):
        J      = J_input['matrix']
        probes = J_input.get('probes')
        mode   = J_input.get('mode', 'unknown')
    else:
        J      = J_input
        probes = None
        mode   = 'direct'

    nz = probes.shape[1] if probes is not None else J.shape[1]
    k  = probes.shape[0] if probes is not None else J.shape[1]

    try:
        U, S, Vh = torch.linalg.svd(J.float(), full_matrices=False)
    except RuntimeError as e:
        warnings.warn(f"SVD failed: {e}")
        return _empty_svd_result(nz, top_k_orth, top_k_tang)

    threshold     = S[0] * min_singular_ratio
    rank_estimate = int((S > threshold).sum().item())
    condition_num = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf')

    n_orth       = min(top_k_orth, Vh.shape[0])
    v_orth_probe = F.normalize(Vh[:n_orth], dim=-1)

    valid_mask   = S > threshold
    valid_idx    = valid_mask.nonzero(as_tuple=False).squeeze(-1)
    tang_idx     = valid_idx[-top_k_tang:] if len(valid_idx) >= top_k_tang else valid_idx
    v_tang_probe = (F.normalize(Vh[tang_idx], dim=-1) if len(tang_idx) > 0
                    else torch.zeros(1, Vh.shape[-1], device=J.device))

    if probes is not None:
        v_orth = F.normalize(v_orth_probe @ probes, dim=-1)
        v_tang = F.normalize(v_tang_probe @ probes, dim=-1)
    else:
        v_orth = v_orth_probe
        v_tang = v_tang_probe

    return {
        'v_orthogonal':     v_orth,
        'v_tangent':        v_tang,
        'singular_values':  S.detach(),
        'rank_estimate':    rank_estimate,
        'condition_number': condition_num,
    }


def _empty_svd_result(nz, top_k_orth, top_k_tang):
    return {
        'v_orthogonal':     None,
        'v_tangent':        None,
        'singular_values':  None,
        'rank_estimate':    0,
        'condition_number': float('nan'),
    }


def compute_jacobian_svd(
    dvae_diff,
    z: torch.Tensor,
    tau: float = 0.3,
    method: str = 'finite_diff',
    eps: float = 1e-3,
    n_probes: int = 16,
    top_k_orth: int = 4,
    top_k_tang: int = 4,
) -> Dict:
    nz = z.shape[0]
    J_result = compute_jacobian(dvae_diff, z, tau=tau, method=method,
                                eps=eps, n_probes=n_probes)
    if J_result is None:
        return _empty_svd_result(nz, top_k_orth, top_k_tang)
    return svd_decompose(J_result, top_k_orth=top_k_orth, top_k_tang=top_k_tang)


class JacobianCache:
    def __init__(self, tol: float = 0.1, max_size: int = 200):
        self.tol      = tol
        self.max_size = max_size
        self._keys: List[torch.Tensor] = []
        self._vals: List[Dict]         = []

    def get(self, z: torch.Tensor) -> Optional[Dict]:
        if not self._keys:
            return None
        keys_mat = torch.stack(self._keys)
        dists    = torch.norm(keys_mat - z.unsqueeze(0), dim=-1)
        min_dist, idx = dists.min(0)
        if min_dist.item() < self.tol:
            return self._vals[idx.item()]
        return None

    def put(self, z: torch.Tensor, result: Dict):
        if len(self._keys) >= self.max_size:
            self._keys.pop(0)
            self._vals.pop(0)
        self._keys.append(z.detach().clone())
        self._vals.append(result)

    def get_or_compute(
        self, z, dvae_diff,
        tau=0.3, method='finite_diff', n_probes=16, top_k_orth=4,
    ) -> Dict:
        cached = self.get(z)
        if cached is not None:
            return cached
        result = compute_jacobian_svd(
            dvae_diff, z, tau=tau, method=method,
            n_probes=n_probes, top_k_orth=top_k_orth)
        self.put(z, result)
        return result

    @property
    def size(self):
        return len(self._keys)


def analyze_latent_smoothness(
    dvae_diff,
    n_pairs: int    = 10,
    nz: int         = 12,   # ← v3: 8 → 12
    n_steps: int    = 6,
    tau: float      = 0.3,
    method: str     = 'random_proj',
    device: str     = 'cpu',
) -> Dict:
    """
    沿随机插值路径计算 Jacobian，分析隐空间平滑度。
    n_probes 自动 clamp 到 min(16, nz)，支持任意 nz。
    """
    cond_nums   = []
    rank_ests   = []
    sing_val_0  = []

    n_probes_actual = min(16, nz)
    print(f"Analyzing latent smoothness ({n_pairs} pairs × {n_steps} steps)  "
          f"nz={nz}  n_probes={n_probes_actual}  method={method}")

    for i in range(n_pairs):
        z1 = torch.randn(nz, device=device) * 0.5
        z2 = torch.randn(nz, device=device) * 0.5
        for alpha in np.linspace(0, 1, n_steps):
            z   = (1 - alpha) * z1 + alpha * z2
            res = compute_jacobian_svd(
                dvae_diff, z, tau=tau, method=method,
                n_probes=n_probes_actual, top_k_orth=4)
            if res['singular_values'] is not None:
                S = res['singular_values']
                cond_nums.append(res['condition_number'])
                rank_ests.append(res['rank_estimate'])
                sing_val_0.append(S[0].item())

    result = {}
    if cond_nums:
        result['cond_mean']    = float(np.mean(cond_nums))
        result['cond_std']     = float(np.std(cond_nums))
        result['rank_mean']    = float(np.mean(rank_ests))
        result['sing_val_max'] = float(np.mean(sing_val_0))

        print(f"\n  Condition number : {result['cond_mean']:.2f} ± {result['cond_std']:.2f}")
        print(f"  Effective rank   : {result['rank_mean']:.1f}")
        print(f"  Avg max sing val : {result['sing_val_max']:.4f}")

        if result['cond_mean'] < 10:
            print("  ⚠️  Low condition number: latent space may be too flat")
        elif result['cond_mean'] > 1000:
            print("  ✅ High condition number: clear orthogonal directions exist")
        else:
            print("  ✅ Moderate condition number: usable for geometric BO")
    else:
        print("  ❌ All Jacobian computations failed")

    return result


# ======================================================================
# 单元测试
# ======================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from nas_space import JointSpaceVAE
    from legacy.geometric_acquisition.dvae_differentiable import (
        build_differentiable_dvae,
    )

    class ArchArgs:
        max_n=7; num_vertex_type=8; START_TYPE=0; END_TYPE=1
        hs=501; nz=12; bidirectional=True

    print("=" * 60)
    print("jacobian_utils v3 Unit Test (nz=12, n_probes=16→12)")
    print("=" * 60)

    vae       = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).eval()
    dvae_diff = build_differentiable_dvae(vae)

    assert dvae_diff.p_dim == 84, f"Expected p_dim=84, got {dvae_diff.p_dim}"
    print(f"p_dim={dvae_diff.p_dim} ✅")

    z = torch.randn(12)

    print("\n[Test 1] finite_diff Jacobian (nz=12):")
    J_dict = compute_jacobian(dvae_diff, z, method='finite_diff', eps=1e-3)
    J = J_dict['matrix']
    print(f"  J.shape = {J.shape}  expected ({dvae_diff.p_dim}, 12)")
    assert J.shape == (dvae_diff.p_dim, 12)
    print("  ✅")

    print("\n[Test 2] random_proj (n_probes=16, nz=12 → actual_n=12):")
    J2_dict  = compute_jacobian(dvae_diff, z, method='random_proj', n_probes=16)
    J2       = J2_dict['matrix']
    probes2  = J2_dict['probes']
    # actual_n = min(16, 12) = 12
    print(f"  J2.shape     = {J2.shape}        expected ({dvae_diff.p_dim}, 12)")
    print(f"  probes.shape = {probes2.shape}   expected (12, 12)")
    assert J2.shape    == (dvae_diff.p_dim, 12), f"Got {J2.shape}"
    assert probes2.shape == (12, 12),             f"Got {probes2.shape}"
    print("  ✅")

    print("\n[Test 3] SVD decompose:")
    res = svd_decompose(J2_dict, top_k_orth=4, top_k_tang=4)
    print(f"  v_orthogonal shape  : {res['v_orthogonal'].shape}")
    print(f"  singular_values[:5] : {res['singular_values'][:5].tolist()}")
    print(f"  condition_number    : {res['condition_number']:.2f}")
    print(f"  rank_estimate       : {res['rank_estimate']}")
    print("  ✅")

    print("\n[Test 4] analyze_latent_smoothness (random_proj, nz=12):")
    analyze_latent_smoothness(dvae_diff, n_pairs=3, nz=12,
                               method='random_proj', device='cpu')

    print("\n✅ All jacobian_utils v3 tests passed.")
