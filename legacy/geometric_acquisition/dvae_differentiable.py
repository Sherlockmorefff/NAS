"""
dvae_differentiable.py  v4  (Search Space v3 — generalised max_n/nvt)
=======================================================================
v4 变更：
  - 支持 max_n=7, nvt=8 (Search Space v3)
  - _forbidden_mask 改为在 __init__ 中按 max_n 动态构建，完全泛化，
    同时向后兼容 max_n=5 的旧模型（用于 Phase 2/3）
  - soft_decode 边约束逻辑泛化：
      * idx=1 (第一个算子节点)：仅从 START(vi=0) 强制连入
      * idx=max_n-1 (END 节点)：仅从 Node_{max_n-2} 强制连入，其余强制断开
      * 中间算子节点 (idx=2..max_n-2)：
          vi=0            → 强制断开（START 不直接连中间节点）
          vi=idx-1        → 强制连入（主链前驱）
          其余 vi         → 可学习跳跃边
  - p_dim 维度 (max_n=7, nvt=8):
      type_dim = (7-1)*8 = 48
      edge_dim = (7-1)*(7-1) = 36
      p_dim    = 84
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_TYPE_INF = 1e9

# ── 内置类型索引常量（与 op_mapping 保持一致）──────────────
_START_TYPE    = 0
_END_TYPE      = 1
_IDENTITY_TYPE = 6


class DVAEDifferentiable(nn.Module):
    def __init__(self, dvae: nn.Module):
        super().__init__()
        self.dvae  = dvae
        self.max_n = dvae.max_n
        self.nvt   = dvae.nvt
        self.hs    = dvae.hs

        # p_dim 由 max_n 和 nvt 全自动推导
        self.type_dim = (self.max_n - 1) * self.nvt
        self.edge_dim = (self.max_n - 1) * (self.max_n - 1)
        self.p_dim    = self.type_dim + self.edge_dim

        # ── 动态构建 forbidden mask ───────────────────────────
        # 规则：
        #   所有算子节点 (idx=1..max_n-2)：禁止 START(0) 和 END(1)
        #   第一个算子节点 (idx=1)：还禁止 Identity(6)
        #   最后一个算子节点 (idx=max_n-2)：还禁止 Identity(6)
        #   END 节点 (idx=max_n-1)：soft_decode 直接强制，无需 mask
        forbidden = torch.zeros(self.max_n, self.nvt, dtype=torch.bool)
        for idx in range(1, self.max_n - 1):   # 算子节点
            if _START_TYPE < self.nvt:
                forbidden[idx, _START_TYPE] = True
            if _END_TYPE < self.nvt:
                forbidden[idx, _END_TYPE] = True
            # 首尾算子节点额外禁止 Identity
            if idx == 1 or idx == self.max_n - 2:
                if _IDENTITY_TYPE < self.nvt:
                    forbidden[idx, _IDENTITY_TYPE] = True

        self.register_buffer('_forbidden_mask', forbidden)

        # 保存供 verify_topology_mask 打印用（字典形式）
        self._forbidden_types_dict = {
            idx: [t for t in range(self.nvt) if forbidden[idx, t].item()]
            for idx in range(1, self.max_n - 1)
        }

    def get_device(self) -> torch.device:
        return next(self.dvae.parameters()).device

    def soft_decode(self, z: torch.Tensor, tau: float = 0.5,
                    deterministic: bool = True) -> torch.Tensor:
        """
        可微软解码。

        输出维度（以 max_n=7, nvt=8 为例）：
          type_dim = 48,  edge_dim = 36,  p_dim = 84
        """
        device = self.get_device()
        z = z.to(device)
        B = z.shape[0]

        H0 = torch.tanh(self.dvae.fc3(z))
        Hv = [H0]
        type_probs_all = []
        edge_probs_all = []

        for idx in range(1, self.max_n):
            h_cur = Hv[-1]

            # ── 1. 节点类型预测 ───────────────────────────────
            if idx == self.max_n - 1:
                # END 节点：强制输出 END_TYPE
                type_soft = torch.zeros(B, self.nvt, device=device)
                type_soft[:, self.dvae.END_TYPE] = 1.0
            else:
                type_logits = self.dvae.add_vertex(h_cur)
                # 应用拓扑掩码（被禁止的类型概率趋近于 0）
                mask = self._forbidden_mask[idx].to(device)
                type_logits = type_logits - mask.float() * _TYPE_INF

                if deterministic:
                    type_soft = F.softmax(type_logits / tau, dim=-1)
                else:
                    type_soft = F.gumbel_softmax(
                        type_logits, tau=tau, hard=False, dim=-1)

            type_probs_all.append(type_soft)

            # ── 2. 边概率预测（泛化版）───────────────────────────
            # 边约束规则（适用于任意 max_n）：
            #   idx=1（第一个算子节点）：仅 vi=0(START) 强制连入
            #   idx=max_n-1（END 节点）：仅 vi=max_n-2 强制连入，其余断开
            #   中间算子节点（idx=2..max_n-2）：
            #       vi=0        → 强制断开（START 不直连中间节点）
            #       vi=idx-1    → 强制连入（主链）
            #       其余        → 可学习跳跃边
            edge_row = []
            for vi in range(self.max_n - 1):
                if vi >= idx:
                    # 上三角之外：不可能存在的边（因果约束）
                    edge_row.append(torch.zeros(B, device=device))
                    continue

                forced_zero = False
                forced_one  = False

                if idx == 1:
                    # 第一个算子节点，唯一可能的前驱就是 START(vi=0)
                    forced_one = (vi == 0)
                    # vi < idx=1 意味着 vi 只可能是 0，不会到达 else

                elif idx == self.max_n - 1:
                    # END 节点：只接受来自最后一个算子节点的边
                    if vi == self.max_n - 2:
                        forced_one = True
                    else:
                        forced_zero = True

                else:
                    # 中间算子节点（idx ∈ [2, max_n-2]）
                    if vi == 0:
                        # START 不直接连到中间节点
                        forced_zero = True
                    elif vi == idx - 1:
                        # 主链前驱：强制连入
                        forced_one = True
                    # else: 其他前驱节点 → 可学习跳跃边（不设 forced）

                if forced_zero:
                    edge_row.append(torch.zeros(B, device=device))
                elif forced_one:
                    edge_row.append(torch.ones(B, device=device))
                else:
                    pair   = torch.cat([Hv[vi], h_cur], dim=-1)
                    e_prob = torch.sigmoid(
                        self.dvae.add_edge(pair)).squeeze(-1)
                    edge_row.append(e_prob)

            edge_probs_all.append(torch.stack(edge_row, dim=-1))
            Hv.append(self.dvae.grud(type_soft, h_cur))

        type_tensor = torch.cat(type_probs_all, dim=-1)   # (B, type_dim)
        edge_tensor = torch.cat(edge_probs_all, dim=-1)   # (B, edge_dim)
        return torch.cat([type_tensor, edge_tensor], dim=-1)   # (B, p_dim)

    def forward(self, z: torch.Tensor, tau: float = 0.5,
                deterministic: bool = True) -> torch.Tensor:
        return self.soft_decode(z, tau=tau, deterministic=deterministic)

    def check_gradient_flow(self, nz: int = None) -> bool:
        device = self.get_device()
        if nz is None:
            nz = self.dvae.nz
        z = torch.randn(1, nz, device=device, requires_grad=True)
        p = self.soft_decode(z, tau=0.5, deterministic=True)
        p.sum().backward()
        ok = z.grad is not None and not torch.isnan(z.grad).any()
        print(f"  [GradCheck] device={device}  nz={nz}  "
              f"p_dim={self.p_dim}  "
              f"grad_norm={z.grad.norm().item():.4f}  OK={ok}")
        return ok

    def verify_topology_mask(self, n_test: int = 100) -> bool:
        """验证拓扑掩码：被禁止的节点类型概率必须 < 1e-6。"""
        device = self.get_device()
        nz     = self.dvae.nz
        z = torch.randn(n_test, nz, device=device)
        with torch.no_grad():
            p = self.soft_decode(z, tau=0.3, deterministic=True)

        nvt     = self.nvt
        all_ok  = True
        for idx, forbidden_types in self._forbidden_types_dict.items():
            start       = (idx - 1) * nvt
            type_probs  = p[:, start: start + nvt]
            for t in forbidden_types:
                if t < nvt:
                    max_prob = type_probs[:, t].max().item()
                    if max_prob > 1e-6:
                        print(f"  ❌ Mask FAILED: Node{idx} type={t} "
                              f"max_prob={max_prob:.4e}")
                        all_ok = False
        if all_ok:
            print(f"  ✅ 全部类型掩码验证通过 ({n_test} 随机样本, "
                  f"max_n={self.max_n}, nvt={self.nvt})")
        return all_ok


def build_differentiable_dvae(joint_vae) -> DVAEDifferentiable:
    wrapper = DVAEDifferentiable(joint_vae.arch_vae)
    wrapper.eval()
    return wrapper


if __name__ == '__main__':
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from nas_space import JointSpaceVAE

    class ArchArgs:
        max_n=7; num_vertex_type=8; START_TYPE=0; END_TYPE=1
        hs=501; nz=12; bidirectional=True

    vae = JointSpaceVAE(ArchArgs(), hp_latent_dim=4).eval()
    d   = build_differentiable_dvae(vae)

    print(f"max_n={d.max_n}  nvt={d.nvt}  nz={d.dvae.nz}")
    print(f"type_dim={d.type_dim}  edge_dim={d.edge_dim}  p_dim={d.p_dim}")
    print(f"Expected: type_dim=48  edge_dim=36  p_dim=84")
    assert d.type_dim == 48, f"Got {d.type_dim}"
    assert d.edge_dim == 36, f"Got {d.edge_dim}"
    assert d.p_dim    == 84, f"Got {d.p_dim}"

    # 形状验证
    z = torch.randn(4, 12)
    with torch.no_grad():
        p = d.soft_decode(z, deterministic=True)
    assert p.shape == (4, 84), f"Shape wrong: {p.shape}"

    # 类型概率归一化验证
    type_part = p[:, :d.type_dim].reshape(-1, d.nvt)
    assert (type_part.sum(-1) - 1.0).abs().max() < 1e-4

    d.check_gradient_flow(nz=12)
    d.verify_topology_mask(n_test=50)

    # 向后兼容测试 (max_n=5, nvt=7, nz=8)
    print("\n[Backward Compat] max_n=5, nvt=7, nz=8")

    class ArchArgsV2:
        max_n=5; num_vertex_type=7; START_TYPE=0; END_TYPE=1
        hs=501; nz=8; bidirectional=True

    from nas_space import JointSpaceVAE as JSVAE2
    # Note: JointSpaceVAE now forces max_n=7/nvt=8 in __init__.
    # For backward compat testing of DVAEDifferentiable alone, we instantiate DVAE directly.
    from models import DVAE
    dvae_v2 = DVAE(max_n=5, nvt=7, START_TYPE=0, END_TYPE=1, hs=501, nz=8, bidirectional=True)
    d2 = DVAEDifferentiable(dvae_v2)
    assert d2.type_dim == 28
    assert d2.edge_dim == 16
    assert d2.p_dim    == 44
    z2 = torch.randn(2, 8)
    with torch.no_grad():
        p2 = d2.soft_decode(z2)
    assert p2.shape == (2, 44), f"v2 shape wrong: {p2.shape}"
    print("  ✅ Backward compat OK")

    print("\n✅ dvae_differentiable v4 ALL TESTS PASSED")
