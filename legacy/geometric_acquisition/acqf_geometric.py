"""
acqf_geometric.py  v6  (Conditional ARD Kernel + ARCH_NZ=12 对齐)
===========================================================
v6 变更（相对 v5）：

  [新增 1] 显式 Conditional ARD Kernel
      - GP 不再只依赖条件特征映射，而是在核函数内部计算条件距离。
      - 对 arch 维度和全局 HP 维度始终计算 ARD 距离。
      - 对 GAT/SAGE/GIN 条件 HP 维度，只在两个点都激活该条件维度时
        计入距离：
            d_hp_i(x,x') = (hp_i - hp'_i) * sqrt(w_i(x) * w_i(x'))
      - inactive HP 维度对 GP 距离贡献为 0，避免随机 inactive 值污染核距离。
      - 条件 mask 由 DVAE soft_decode(z_arch) 的算子类型预测得到，
        训练点与候选点都会计算 mask。

v5 变更（相对 v4）：

  [新增 1] 条件核函数（conditional kernel）支持
      - 对 z_arch 使用 DVAE soft_decode 得到每层 GAT/SAGE/GIN 激活概率。
      - 条件 HP 维度先乘对应激活概率，再进入 GP 核函数。
      - 额外拼接激活概率作为结构条件特征，使“激活模式不同”的架构
        在核空间中保持可区分。
      - 该实现等价于在原搜索变量 x 上使用
            k_cond(x, x') = k_base(phi_cond(x), phi_cond(x'))
        其中 phi_cond 是可微条件特征映射。
      - 默认关闭；由 bo_phase4.py 的 --use_conditional_kernel 启用。

v4 变更（相对 v3）：

  [修复 1] JacobianLogDet.compute 奇异值爆炸防护
      v3 问题：直接对所有奇异值取 -log，当极小奇异值（接近 0）存在时，
               -log(S_min) → +∞，导致 logD 爆炸，污染采集函数得分。
      v4 修复：定义 threshold = S[0] * 0.01（相对于最大奇异值的 1%），
               筛选 active_S = S[S > threshold]，仅对有效奇异值求 logD。
               若 active_S 为空，则 logD = 0.0（安全回退）。

  [修复 2] ARCH_NZ 常量：8 → 12（对齐 Search Space v3）
      HPDiversityBonus.hp_start_idx 默认值随之更新。
      LatentDensityModel.arch_nz 默认值随之更新。
      ReconstructionNovelty.arch_nz 默认值随之更新。

  v3 其他修复保留不变（actual_n = min(n_probes, nz) 防维度溢出）。
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Optional, Dict, List
from botorch.models import SingleTaskGP
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.priors import GammaPrior


ARCH_NZ = 12   # v4: 8 → 12，对齐 Search Space v3
MAX_OP_NODES = 5
HP_DIM_LAYERWISE = 2 + 3 * MAX_OP_NODES
GAT_TYPE = 3
SAGE_TYPE = 4
GIN_TYPE = 5


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
        # 防止 n_probes > arch_nz 时 QR 分解维度溢出
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

            # ── [v4 修复] 奇异值爆炸防护 ────────────────────────
            # 定义相对阈值：最大奇异值的 1%
            # 极小奇异值（噪声维度）的 -log 会趋向 +∞，必须过滤
            threshold = S[0] * 0.01
            active_S  = S[S > threshold]

            if len(active_S) > 0:
                logD = -torch.log(active_S.clamp(min=1e-8)).sum().item()
            else:
                logD = 0.0   # 全部奇异值低于阈值时安全回退

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
# 组件 6：Conditional ARD Kernel
# ======================================================================

class ConditionalARDKernel(Kernel):
    """
    Conditional ARD Matern-5/2 kernel.

    对原搜索向量 x=[z_arch, hp] 直接定义条件距离：

        d(x, x')^2 =
            sum_arch/global_hp ((x_i - x'_i) / l_i)^2
          + sum_cond_hp (((hp_i - hp'_i) * sqrt(w_i(x) w_i(x'))) / l_i)^2

    当任一点未激活条件维度 i 时，w_i=0，该维度距离贡献为 0。
    """

    has_lengthscale = True

    def __init__(
        self,
        dvae_diff,
        x_min: torch.Tensor,
        x_range: torch.Tensor,
        arch_nz: int = ARCH_NZ,
        hp_dim: int = 4,
        tau: float = 0.3,
        active_weight: float = 1.0,
        binary_masks: bool = True,
        detach_masks: bool = False,
        **kwargs,
    ):
        super().__init__(ard_num_dims=arch_nz + hp_dim, **kwargs)
        # Keep DVAE as a plain reference: it defines conditional masks, but its
        # parameters must not become GP hyperparameters during MLL fitting.
        self.__dict__["dvae_diff"] = dvae_diff
        self.arch_nz = arch_nz
        self.hp_dim = hp_dim
        self.tau = tau
        self.active_weight = float(active_weight)
        self.binary_masks = binary_masks
        self.detach_masks = detach_masks
        self.register_buffer("x_min", x_min.float())
        self.register_buffer("x_range", x_range.float().clamp(min=1e-8))

    def _op_masks(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Return HP masks with shape (..., N, hp_dim).

        The first two HP dimensions, lr/dropout, are always active. Conditional
        dimensions are active only when the decoded layer/operator requires them.
        """
        leading_shape = x_norm.shape[:-1]
        flat = x_norm.reshape(-1, x_norm.shape[-1]).float()

        x_min = self.x_min.to(flat.device)
        x_range = self.x_range.to(flat.device)
        z_orig = flat * x_range + x_min
        z_arch = z_orig[:, :self.arch_nz].to(self.dvae_diff.get_device())

        if self.binary_masks or self.detach_masks:
            with torch.no_grad():
                p = self.dvae_diff.soft_decode(
                    z_arch, tau=self.tau, deterministic=True)
        else:
            p = self.dvae_diff.soft_decode(
                z_arch, tau=self.tau, deterministic=True)
        type_probs = p[:, :self.dvae_diff.type_dim].reshape(
            -1, self.dvae_diff.max_n - 1, self.dvae_diff.nvt)
        op_probs = type_probs[:, :MAX_OP_NODES, :]

        def _prob(type_idx: int) -> torch.Tensor:
            if type_idx >= self.dvae_diff.nvt:
                return torch.zeros(
                    op_probs.shape[:2], device=op_probs.device,
                    dtype=op_probs.dtype)
            return op_probs[:, :, type_idx]

        if self.binary_masks:
            op_type = op_probs.argmax(dim=-1)
            gat = (op_type == GAT_TYPE).float()
            sage = (op_type == SAGE_TYPE).float()
            gin = (op_type == GIN_TYPE).float()
        else:
            gat = _prob(GAT_TYPE)
            sage = _prob(SAGE_TYPE)
            gin = _prob(GIN_TYPE)

        if self.detach_masks:
            gat = gat.detach()
            sage = sage.detach()
            gin = gin.detach()

        cond = torch.cat([gat, sage, gin], dim=-1)
        cond = cond.to(device=flat.device, dtype=flat.dtype) * self.active_weight

        masks = torch.ones(
            flat.shape[0], self.hp_dim, device=flat.device, dtype=flat.dtype)
        if self.hp_dim >= HP_DIM_LAYERWISE:
            masks[:, 2:2 + 3 * MAX_OP_NODES] = cond[:, :3 * MAX_OP_NODES]
        elif self.hp_dim >= 4:
            masks[:, 2] = gat.amax(dim=-1).to(flat.device, flat.dtype) * self.active_weight
            masks[:, 3] = sage.amax(dim=-1).to(flat.device, flat.dtype) * self.active_weight

        return masks.reshape(*leading_shape[:-1], leading_shape[-1], self.hp_dim)

    def forward(
        self,
        X: torch.Tensor,
        X2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("ConditionalARDKernel does not support last_dim_is_batch")
        if X2 is None:
            X2 = X

        ls = self.lengthscale.to(device=X.device, dtype=X.dtype)
        ls = ls.view(*([1] * (X.dim())), -1).clamp(min=1e-8)
        ls_arch = ls[..., :self.arch_nz]
        ls_hp = ls[..., self.arch_nz:self.arch_nz + self.hp_dim]

        X_arch = X[..., :, :self.arch_nz]
        X2_arch = X2[..., :, :self.arch_nz]
        diff_arch = X_arch.unsqueeze(-2) - X2_arch.unsqueeze(-3)
        scaled_arch = diff_arch / ls_arch[..., :self.arch_nz]

        X_hp = X[..., :, self.arch_nz:self.arch_nz + self.hp_dim]
        X2_hp = X2[..., :, self.arch_nz:self.arch_nz + self.hp_dim]
        masks1 = self._op_masks(X)
        masks2 = self._op_masks(X2)
        w_eff = torch.sqrt(
            (masks1.unsqueeze(-2) * masks2.unsqueeze(-3)).clamp(min=0.0))
        diff_hp = (X_hp.unsqueeze(-2) - X2_hp.unsqueeze(-3)) * w_eff
        scaled_hp = diff_hp / ls_hp

        r2 = (scaled_arch ** 2).sum(dim=-1) + (scaled_hp ** 2).sum(dim=-1)
        r = torch.sqrt(r2.clamp(min=1e-12))
        sqrt5r = np.sqrt(5.0) * r
        K = (1.0 + sqrt5r + (5.0 / 3.0) * r2) * torch.exp(-sqrt5r)

        if diag:
            return K.diagonal(dim1=-2, dim2=-1)
        return K


class ConditionalSingleTaskGP(SingleTaskGP):
    """SingleTaskGP using ConditionalARDKernel as its covariance module."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        dvae_diff,
        x_min: torch.Tensor,
        x_range: torch.Tensor,
        arch_nz: int,
        hp_dim: int,
        tau: float = 0.3,
        active_weight: float = 1.0,
        binary_masks: bool = True,
        detach_masks: bool = False,
        **kwargs,
    ):
        super().__init__(train_X, train_Y, **kwargs)
        base_kernel = ConditionalARDKernel(
            dvae_diff=dvae_diff,
            x_min=x_min,
            x_range=x_range,
            arch_nz=arch_nz,
            hp_dim=hp_dim,
            tau=tau,
            active_weight=active_weight,
            binary_masks=binary_masks,
            detach_masks=detach_masks,
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )


# ======================================================================
# 组件 7：条件特征映射（保留为兼容旧实验，不作为默认条件核路径）
# ======================================================================

class ConditionalFeatureMap:
    """
    可微条件特征映射 phi_cond(x)。

    BoTorch 的 GP 仍使用标准核函数，但输入被替换为 phi_cond(x)，
    因此等价于在原始搜索空间上使用条件核：

        k_cond(x, x') = k_base(phi_cond(x), phi_cond(x'))

    hp_dim=17:
      phi_cond = [
          z_arch_norm,
          global_lr_norm, global_dropout_norm,
          gat_hp_i * P(op_i=GAT),
          sage_hp_i * P(op_i=SAGE),
          gin_hp_i * P(op_i=GIN),
          mask_weight * P(op_i=GAT/SAGE/GIN)
      ]

    hp_dim=4:
      使用 type-wise GAT/SAGE 激活概率。
    """

    def __init__(
        self,
        dvae_diff,
        x_min: torch.Tensor,
        x_range: torch.Tensor,
        arch_nz: int = ARCH_NZ,
        hp_dim: int = 4,
        tau: float = 0.3,
        mask_weight: float = 0.5,
        detach_masks: bool = False,
    ):
        self.dvae_diff = dvae_diff
        self.x_min = x_min.float()
        self.x_range = x_range.float().clamp(min=1e-8)
        self.arch_nz = arch_nz
        self.hp_dim = hp_dim
        self.tau = tau
        self.mask_weight = mask_weight
        self.detach_masks = detach_masks

    def _op_probabilities(self, z_arch_orig: torch.Tensor) -> torch.Tensor:
        device = self.dvae_diff.get_device()
        z_arch_model = z_arch_orig.to(device)
        p = self.dvae_diff.soft_decode(
            z_arch_model, tau=self.tau, deterministic=True)

        type_part = p[:, :self.dvae_diff.type_dim]
        type_probs = type_part.reshape(
            -1, self.dvae_diff.max_n - 1, self.dvae_diff.nvt)
        op_probs = type_probs[:, :MAX_OP_NODES, :]

        def _prob(type_idx: int) -> torch.Tensor:
            if type_idx >= self.dvae_diff.nvt:
                return torch.zeros(
                    op_probs.shape[:2], device=op_probs.device,
                    dtype=op_probs.dtype)
            return op_probs[:, :, type_idx]

        masks = torch.cat([
            _prob(GAT_TYPE),
            _prob(SAGE_TYPE),
            _prob(GIN_TYPE),
        ], dim=-1)

        if self.detach_masks:
            masks = masks.detach()
        return masks

    def __call__(self, z_norm: torch.Tensor) -> torch.Tensor:
        orig_dtype = z_norm.dtype
        orig_device = z_norm.device
        leading_shape = z_norm.shape[:-1]
        flat = z_norm.reshape(-1, z_norm.shape[-1]).float()

        x_min = self.x_min.to(flat.device)
        x_range = self.x_range.to(flat.device)
        z_orig = flat * x_range + x_min

        arch_feat = flat[:, :self.arch_nz]
        global_hp = flat[:, self.arch_nz:self.arch_nz + 2]
        hp_norm = z_orig[:, self.arch_nz:self.arch_nz + self.hp_dim].clamp(0.0, 1.0)

        masks = self._op_probabilities(z_orig[:, :self.arch_nz])
        masks = masks.to(device=flat.device, dtype=flat.dtype)

        if self.hp_dim >= HP_DIM_LAYERWISE and hp_norm.shape[1] >= HP_DIM_LAYERWISE:
            gat_hp = hp_norm[:, 2:2 + MAX_OP_NODES]
            sage_hp = hp_norm[:, 2 + MAX_OP_NODES:2 + 2 * MAX_OP_NODES]
            gin_hp = hp_norm[:, 2 + 2 * MAX_OP_NODES:2 + 3 * MAX_OP_NODES]
            cond_hp = torch.cat([
                gat_hp * masks[:, :MAX_OP_NODES],
                sage_hp * masks[:, MAX_OP_NODES:2 * MAX_OP_NODES],
                gin_hp * masks[:, 2 * MAX_OP_NODES:3 * MAX_OP_NODES],
            ], dim=-1)
            cond_mask = masks
        else:
            gat_mask = masks[:, :MAX_OP_NODES].amax(dim=-1, keepdim=True)
            sage_mask = masks[:, MAX_OP_NODES:2 * MAX_OP_NODES].amax(
                dim=-1, keepdim=True)

            gat_hp = hp_norm[:, 2:3] if hp_norm.shape[1] > 2 else torch.zeros_like(gat_mask)
            sage_hp = hp_norm[:, 3:4] if hp_norm.shape[1] > 3 else torch.zeros_like(sage_mask)
            cond_hp = torch.cat([gat_hp * gat_mask, sage_hp * sage_mask], dim=-1)
            cond_mask = torch.cat([gat_mask, sage_mask], dim=-1)

        if self.mask_weight > 0:
            cond_mask = cond_mask * float(self.mask_weight)
            features = torch.cat([arch_feat, global_hp, cond_hp, cond_mask], dim=-1)
        else:
            features = torch.cat([arch_feat, global_hp, cond_hp], dim=-1)

        return features.reshape(*leading_shape, features.shape[-1]).to(
            device=orig_device, dtype=orig_dtype)


class ConditionalLogEIWrapper(AcquisitionFunction):
    """在原始 z_norm 空间中优化，但用条件特征映射后的 GP 计算 LogEI。"""

    def __init__(self, logei: qLogExpectedImprovement,
                 feature_map: ConditionalFeatureMap):
        super().__init__(model=logei.model)
        self.logei = logei
        self.feature_map = feature_map

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.logei(self.feature_map(X.float()).double())


# ======================================================================
# GeometricLogEI
# ======================================================================

class GeometricLogEI:
    def __init__(self, gp_model, best_f: float,
                 gpnd_novelty: GPNDNASNovelty,
                 X_obs: torch.Tensor,
                 X_min: torch.Tensor, X_range: torch.Tensor,
                 novelty_weight: float = 0.15,
                 feature_map: ConditionalFeatureMap = None):
        self.gp      = gp_model
        self.best_f  = best_f
        self.gpnd    = gpnd_novelty
        self.X_obs   = X_obs
        self.X_min   = X_min.float()
        self.X_range = X_range.float()
        self.lam     = novelty_weight
        self.feature_map = feature_map
        self._logei  = qLogExpectedImprovement(gp_model, best_f=best_f)
        self._opt_logei = (
            ConditionalLogEIWrapper(self._logei, feature_map)
            if feature_map is not None else self._logei
        )

    def _logei_score(self, z_norm_batch: torch.Tensor) -> torch.Tensor:
        X_raw = z_norm_batch.float().unsqueeze(1)
        X = self.feature_map(X_raw).double() if self.feature_map is not None \
            else X_raw.double()
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
                    self._opt_logei, bounds=bounds, q=1,
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
    conditional_kernel: bool = False,
    dvae_diff = None,
    arch_nz: int = ARCH_NZ,
    hp_dim: int = 4,
    cond_tau: float = 0.3,
    cond_mask_weight: float = 1.0,
    detach_cond_mask: bool = False,
) -> torch.Tensor:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X_min   = X_obs.min(0).values
    X_range = (X_obs.max(0).values - X_min).clamp(min=1e-8)
    X_norm  = (X_obs - X_min) / X_range
    Y_norm  = (Y_obs - Y_obs.mean()) / Y_obs.std().clamp(min=1e-6)

    feature_map = None
    if conditional_kernel:
        if dvae_diff is None:
            raise ValueError("conditional_kernel=True requires dvae_diff")
        gp = ConditionalSingleTaskGP(
            X_norm.double(),
            Y_norm.double(),
            dvae_diff=dvae_diff,
            x_min=X_min,
            x_range=X_range,
            arch_nz=arch_nz,
            hp_dim=hp_dim,
            tau=cond_tau,
            active_weight=cond_mask_weight,
            binary_masks=True,
            detach_masks=detach_cond_mask,
        )
    else:
        gp = SingleTaskGP(X_norm.double(), Y_norm.double())
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
        feature_map    = feature_map,
    )

    return acqf.optimize(
        search_dim   = search_dim,
        num_restarts = num_restarts,
        raw_samples  = raw_samples,
        n_extra      = n_extra,
    )


def _z_key(z: torch.Tensor) -> str:
    return str(z.cpu().numpy().round(3).tobytes())
