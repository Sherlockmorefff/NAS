"""
acqf_geometric.py  v3  (ARCH_NZ=8 + random_proj 维度修复)
==========================================================
修复：JacobianLogDet.compute 中当 n_probes > nz 时，
     QR 分解只产出 nz 个正交向量，但 J_cols 拼出 n_probes 列，
     SVD 不报错但结果含全零列，影响 logD 精度（无崩溃但有偏差）。
     现统一使用 actual_n = min(n_probes, nz)。
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Optional, Dict, List
from botorch.models import SingleTaskGP
from botorch.acquisition.logei import qLogExpectedImprovement


ARCH_NZ = 8


# ======================================================================
# 组件 1：logD —— Jacobian 行列式（流形拉伸）
# ======================================================================

class JacobianLogDet:
    def __init__(self, dvae_diff, tau: float = 0.3,
                 n_probes: int = 16, eps: float = 1e-3):
        self.dvae_diff = dvae_diff
        self.tau       = tau
        self.n_probes  = n_probes
        self.eps       = eps
        self._cache: Dict = {}

    def compute(self, z: torch.Tensor) -> float:
        key = _z_key(z)
        if key in self._cache:
            return self._cache[key]

        arch_nz  = self.dvae_diff.dvae.nz
        # ── 修复：actual_n 不超过 arch_nz ──────────────────────
        actual_n = min(self.n_probes, arch_nz)
        z_model  = z[:arch_nz].to(self.dvae_diff.get_device()).detach()

        raw    = torch.randn(actual_n, arch_nz, device=z_model.device)
        Q, _   = torch.linalg.qr(raw.T)   # Q: (arch_nz, actual_n)
        probes = Q.T                        # probes: (actual_n, arch_nz)

        J_cols = []
        for v in probes:
            v  = F.normalize(v, dim=0)
            zp = z_model + self.eps * v
            zm = z_model - self.eps * v
            with torch.no_grad():
                pp = self.dvae_diff.soft_decode(
                    zp.unsqueeze(0), tau=self.tau, deterministic=True).squeeze(0)
                pm = self.dvae_diff.soft_decode(
                    zm.unsqueeze(0), tau=self.tau, deterministic=True).squeeze(0)
            J_cols.append((pp - pm) / (2 * self.eps))

        J = torch.stack(J_cols, dim=1).cpu()   # (p_dim, actual_n)

        try:
            _, S, _ = torch.linalg.svd(J.float(), full_matrices=False)
            logD = -torch.log(S.clamp(min=1e-8)).sum().item()
        except Exception:
            logD = 0.0

        self._cache[key] = logD
        return logD

    def batch_compute(self, z_list: List[torch.Tensor]) -> np.ndarray:
        return np.array([self.compute(z) for z in z_list])


# ======================================================================
# 组件 2：logPz —— 隐空间密度
# ======================================================================

class LatentDensityModel:
    def __init__(self, arch_nz: int = ARCH_NZ):
        self.arch_nz = arch_nz
        self.beta    = None
        self.loc     = None
        self.scale   = None

    def fit(self, X_obs: torch.Tensor):
        import scipy.stats
        Z = X_obs[:, :self.arch_nz].cpu().numpy()
        N, D = Z.shape

        if N < 5:
            self.beta  = np.ones(D) * 2.0
            self.loc   = Z.mean(0)
            self.scale = Z.std(0) + 1e-6
            return

        self.beta  = np.zeros(D)
        self.loc   = np.zeros(D)
        self.scale = np.zeros(D)

        def fmin_wrapper(func, x0, args, disp):
            import scipy.optimize
            return scipy.optimize.fmin(func, [2.0, 0.0, 1.0], args,
                                       xtol=1e-8, ftol=1e-8, disp=0)

        for i in range(D):
            try:
                b, l, s = scipy.stats.gennorm.fit(Z[:, i], optimizer=fmin_wrapper)
                self.beta[i]  = float(b)
                self.loc[i]   = float(l)
                self.scale[i] = float(max(abs(s), 1e-6))
            except Exception:
                self.beta[i]  = 2.0
                self.loc[i]   = float(Z[:, i].mean())
                self.scale[i] = float(Z[:, i].std() + 1e-6)

    def log_density(self, z: torch.Tensor) -> float:
        if self.beta is None:
            return 0.0
        import scipy.stats
        z_np  = z[:self.arch_nz].cpu().numpy()
        p     = scipy.stats.gennorm.pdf(z_np, self.beta, self.loc, self.scale)
        log_p = np.log(np.clip(p, 1e-300, None)).sum()
        return float(log_p) if np.isfinite(log_p) else -1000.0

    def novelty_score(self, z: torch.Tensor) -> float:
        return -self.log_density(z)


# ======================================================================
# 组件 3：logPe —— 重建空间新颖性
# ======================================================================

class ReconstructionNovelty:
    def __init__(self, dvae_diff, tau: float = 0.3,
                 arch_nz: int = ARCH_NZ, max_cache: int = 200):
        self.dvae_diff = dvae_diff
        self.tau       = tau
        self.arch_nz   = arch_nz
        self.max_cache = max_cache
        self._p_obs: Optional[torch.Tensor] = None

    def update(self, X_obs: torch.Tensor):
        device = self.dvae_diff.get_device()
        n = min(len(X_obs), self.max_cache)
        Z_arch = X_obs[-n:, :self.arch_nz].to(device)

        with torch.no_grad():
            p_list = []
            for i in range(0, len(Z_arch), 16):
                p = self.dvae_diff.soft_decode(
                    Z_arch[i:i+16], tau=self.tau, deterministic=True)
                p_list.append(p.cpu())
        self._p_obs = torch.cat(p_list, dim=0)

    def novelty(self, z: torch.Tensor) -> float:
        if self._p_obs is None or len(self._p_obs) == 0:
            return 1.0

        device = self.dvae_diff.get_device()
        z_arch = z[:self.arch_nz].to(device)

        with torch.no_grad():
            p_cand = self.dvae_diff.soft_decode(
                z_arch.unsqueeze(0), tau=self.tau,
                deterministic=True).squeeze(0).cpu()

        dists = torch.norm(self._p_obs - p_cand.unsqueeze(0), dim=-1)
        return float(dists.min().item())

    def batch_novelty(self, z_list: List[torch.Tensor]) -> np.ndarray:
        return np.array([self.novelty(z) for z in z_list])


# ======================================================================
# 组件 4：HP 多样性奖励
# ======================================================================

class HPDiversityBonus:
    def __init__(self, hp_start_idx: int = ARCH_NZ, bandwidth: float = 0.15):
        self.start    = hp_start_idx
        self.bw       = bandwidth
        self._hp_obs: Optional[np.ndarray] = None

    def update(self, X_obs: torch.Tensor):
        self._hp_obs = X_obs[:, self.start:].cpu().numpy()

    def bonus(self, z: torch.Tensor) -> float:
        if self._hp_obs is None or len(self._hp_obs) < 3:
            return 0.0
        hp_cand = z[self.start:].cpu().numpy()
        diff  = self._hp_obs - hp_cand[np.newaxis, :]
        dists = (diff**2).sum(-1)
        kde   = np.exp(-dists / (2 * self.bw**2)).mean()
        return float(-np.log(max(kde, 1e-10)))


# ======================================================================
# GPND-NAS 综合新颖性得分
# ======================================================================

class GPNDNASNovelty:
    def __init__(self, dvae_diff, arch_nz: int = ARCH_NZ,
                 tau: float = 0.3, n_probes: int = 16,
                 alpha: float = 0.2, beta: float = 0.3,
                 gamma: float = 0.3, delta: float = 0.2):
        self.arch_nz = arch_nz
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.delta   = delta

        self.logD_calc  = JacobianLogDet(dvae_diff, tau=tau, n_probes=n_probes)
        self.logPz_calc = LatentDensityModel(arch_nz=arch_nz)
        self.logPe_calc = ReconstructionNovelty(dvae_diff, tau=tau, arch_nz=arch_nz)
        self.hp_bonus   = HPDiversityBonus(hp_start_idx=arch_nz)

        self._n_obs_last = 0

    def update(self, X_obs: torch.Tensor):
        n = len(X_obs)
        if n == self._n_obs_last:
            return
        self._n_obs_last = n
        self.logPz_calc.fit(X_obs)
        self.logPe_calc.update(X_obs)
        self.hp_bonus.update(X_obs)

    def score(self, z: torch.Tensor) -> float:
        s_logD  = self.logD_calc.compute(z)
        s_logPz = self.logPz_calc.novelty_score(z)
        s_logPe = self.logPe_calc.novelty(z)
        s_hp    = self.hp_bonus.bonus(z)
        return (self.alpha * s_logD + self.beta * s_logPz
                + self.gamma * s_logPe + self.delta * s_hp)

    def batch_score(self, z_list: List[torch.Tensor],
                    normalize: bool = True) -> np.ndarray:
        scores = np.array([self.score(z) for z in z_list])
        if normalize:
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min > 1e-8:
                scores = (scores - s_min) / (s_max - s_min)
        return scores


# ======================================================================
# GeometricLogEI
# ======================================================================

class GeometricLogEI:
    def __init__(self, gp_model, best_f: float,
                 gpnd_novelty: GPNDNASNovelty,
                 X_obs: torch.Tensor,
                 X_min: torch.Tensor, X_range: torch.Tensor,
                 novelty_weight: float = 0.15):
        self.gp      = gp_model
        self.best_f  = best_f
        self.gpnd    = gpnd_novelty
        self.X_obs   = X_obs
        self.X_min   = X_min.float()
        self.X_range = X_range.float()
        self.lam     = novelty_weight
        self._logei  = qLogExpectedImprovement(gp_model, best_f=best_f)

    def _logei_score(self, z_norm_batch: torch.Tensor) -> torch.Tensor:
        X = z_norm_batch.double().unsqueeze(1)
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = self._logei(X)
        return s.float()

    def score_candidates(self, z_norm_candidates: torch.Tensor) -> torch.Tensor:
        N = len(z_norm_candidates)
        ei_scores = self._logei_score(z_norm_candidates)

        if self.lam <= 0 or N == 0:
            return ei_scores

        z_orig = z_norm_candidates.float() * self.X_range + self.X_min
        z_list = [z_orig[i] for i in range(N)]
        nov    = torch.tensor(
            self.gpnd.batch_score(z_list, normalize=True),
            dtype=torch.float32)

        return ei_scores + self.lam * nov

    def optimize(self, search_dim: int,
                 num_restarts: int = 10, raw_samples: int = 512,
                 n_extra: int = 128) -> torch.Tensor:
        from botorch.optim import optimize_acqf

        bounds = torch.zeros(2, search_dim, dtype=torch.double)
        bounds[1] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cand_opt, _ = optimize_acqf(
                    self._logei, bounds=bounds, q=1,
                    num_restarts=num_restarts, raw_samples=raw_samples)
                cand_opt = cand_opt.squeeze(0).float()
            except Exception as e:
                warnings.warn(f"optimize_acqf failed: {e}")
                cand_opt = torch.rand(search_dim)

        extra_norm = torch.rand(n_extra, search_dim)
        all_cands  = torch.cat([cand_opt.unsqueeze(0), extra_norm], dim=0)
        all_scores = self.score_candidates(all_cands)

        best_idx    = all_scores.argmax().item()
        z_next_norm = all_cands[best_idx]
        return z_next_norm.float() * self.X_range + self.X_min


# ======================================================================
# 便捷工厂函数
# ======================================================================

def make_acqf_and_optimize(
    X_obs: torch.Tensor,
    Y_obs: torch.Tensor,
    gpnd_novelty: GPNDNASNovelty,
    search_dim: int,
    novelty_weight: float = 0.15,
    num_restarts: int = 8,
    raw_samples: int = 256,
    n_extra: int = 128,
) -> torch.Tensor:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X_min   = X_obs.min(0).values
    X_range = (X_obs.max(0).values - X_min).clamp(min=1e-8)
    X_norm  = (X_obs - X_min) / X_range
    Y_norm  = (Y_obs - Y_obs.mean()) / Y_obs.std().clamp(min=1e-6)

    gp  = SingleTaskGP(X_norm.double(), Y_norm.double())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    gpnd_novelty.update(X_obs)

    acqf = GeometricLogEI(
        gp_model       = gp,
        best_f         = Y_norm.double().max(),
        gpnd_novelty   = gpnd_novelty,
        X_obs          = X_obs,
        X_min          = X_min,
        X_range        = X_range,
        novelty_weight = novelty_weight,
    )

    return acqf.optimize(
        search_dim   = search_dim,
        num_restarts = num_restarts,
        raw_samples  = raw_samples,
        n_extra      = n_extra,
    )


def _z_key(z: torch.Tensor) -> str:
    return str(z.cpu().numpy().round(3).tobytes())