"""
nas_space.py  v4  (Search Space v3 + HP VAE 4维升级)
======================================================
v4 变更（相对 v3）：
  - HP_VAE input_dim: 2 → 4
      [0] log_lr   (归一化)
      [1] dropout  (归一化)
      [2] hidden_dim_norm  (归一化, 映射至 {16,32,64,128,256,512})
      [3] l2_norm          (归一化, 映射至 {1e-4..5e-4})
  - decode_from_joint_latent：
      * 新增 return_full=False 参数（向后兼容）
      * return_full=True 时额外返回 hidden_dims, l2_regs
  - JointSpaceVAE: HP_VAE(input_dim=4)

⚠️  HP VAE input_dim 变化导致 v3 及更早权重与 HP 解码器不兼容，
    必须重新运行 generate_mini_data.py（v4）和 train_joint.py。
"""

import torch
import torch.nn as nn
from models import DVAE


OP_MAPPING = {
    2: "GCNConv",
    3: "GATConv",
    4: "SAGEConv",
    5: "GINConv",
    6: "Identity",
    7: "GCNII",        # v3 新增
}


class HP_VAE(nn.Module):
    """超参数变分自编码器。

    input_dim=4 时编码 [log_lr, dropout, hidden_norm, l2_norm]（v4 默认）。
    input_dim=2 时仅编码 [log_lr, dropout]（向后兼容）。
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, latent_dim: int = 4):
        super(HP_VAE, self).__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)


class JointSpaceVAE(nn.Module):
    START_TYPE      = 0
    END_TYPE        = 1
    NUM_VERTEX_TYPE = 8    # v3: 7 → 8

    def __init__(self, arch_args, hp_latent_dim: int = 4):
        super(JointSpaceVAE, self).__init__()

        # Force v3 dimensions onto arch_args
        arch_args.max_n           = 7
        arch_args.num_vertex_type = self.NUM_VERTEX_TYPE
        arch_args.START_TYPE      = self.START_TYPE
        arch_args.END_TYPE        = self.END_TYPE

        self.op_mapping = OP_MAPPING

        self.arch_vae = DVAE(
            max_n         = arch_args.max_n,
            nvt           = arch_args.num_vertex_type,
            START_TYPE    = arch_args.START_TYPE,
            END_TYPE      = arch_args.END_TYPE,
            hs            = arch_args.hs,
            nz            = arch_args.nz,
            bidirectional = arch_args.bidirectional,
        )

        # v4: HP VAE 升级到 input_dim=4，完整编码 4 个超参
        self.hp_vae = HP_VAE(input_dim=4, latent_dim=hp_latent_dim)

        self.arch_nz  = arch_args.nz
        self.hp_nz    = hp_latent_dim
        self.total_nz = self.arch_nz + self.hp_nz

    def forward(self, graph_input, hp_input):
        mu_arch, logvar_arch = self.arch_vae.encode(graph_input)
        z_arch               = self.arch_vae.reparameterize(mu_arch, logvar_arch)
        arch_decode_output   = self.arch_vae.decode(z_arch)

        mu_hp, logvar_hp = self.hp_vae.encode(hp_input)
        z_hp             = self.hp_vae.reparameterize(mu_hp, logvar_hp)
        hp_decode_output = self.hp_vae.decode(z_hp)

        return (arch_decode_output, mu_arch, logvar_arch), \
               (hp_decode_output,   mu_hp,   logvar_hp)

    def encode_to_joint_latent(self, graph_input, hp_input):
        mu_arch, _ = self.arch_vae.encode(graph_input)
        mu_hp,   _ = self.hp_vae.encode(hp_input)
        return torch.cat([mu_arch, mu_hp], dim=-1)

    def decode_from_joint_latent(self, z_total, return_full: bool = False):
        """
        从联合隐向量解码架构配置和超参数。

        Parameters
        ----------
        z_total     : Tensor, shape [B, arch_nz + hp_nz]
        return_full : bool
            False（默认）→ 返回 (gnn_configs, learning_rates, dropouts)，
                           向后兼容所有旧调用方。
            True          → 返回 (gnn_configs, learning_rates, dropouts,
                                  hidden_dims, l2_regs)，支持 v4 完整解码。

        Returns
        -------
        gnn_configs    : list of dict
        learning_rates : list of float
        dropouts       : list of float
        hidden_dims    : list of float  (仅 return_full=True)
        l2_regs        : list of float  (仅 return_full=True)
        """
        z_arch = z_total[:, :self.arch_nz]
        z_hp   = z_total[:, self.arch_nz:]

        raw_hps        = self.hp_vae.decode(z_hp)
        learning_rates = torch.pow(10, raw_hps[:, 0]).tolist()
        dropouts       = torch.clamp(raw_hps[:, 1], 0.0, 0.6).tolist()

        # v4: 提取 hidden_dim 和 l2 (当 HP_VAE input_dim >= 3/4 时有效)
        if raw_hps.shape[1] >= 3:
            hidden_dims = raw_hps[:, 2].tolist()
        else:
            # 向后兼容：input_dim=2 的旧模型，返回默认归一化值 0.5 → 64
            hidden_dims = [0.5] * raw_hps.shape[0]

        if raw_hps.shape[1] >= 4:
            l2_regs = raw_hps[:, 3].tolist()
        else:
            # 向后兼容：返回默认归一化值 0.8 → 5e-4 (最后一档)
            l2_regs = [0.8] * raw_hps.shape[0]

        graphs = self.arch_vae.decode(z_arch)
        gnn_configs = []
        for g in graphs:
            actual_ops = []
            for i in range(1, g.vcount() - 1):
                op_idx  = g.vs[i]['type']
                op_name = self.op_mapping.get(op_idx, "Identity")
                actual_ops.append(op_name)
            edges            = g.get_edgelist()
            effective_layers = sum(1 for op in actual_ops if op != "Identity")
            gnn_configs.append({
                "effective_layers": effective_layers,
                "operations":       actual_ops,
                "edges":            edges,
            })

        if return_full:
            return gnn_configs, learning_rates, dropouts, hidden_dims, l2_regs
        return gnn_configs, learning_rates, dropouts