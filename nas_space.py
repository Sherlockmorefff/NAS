"""
nas_space.py  v3  (Search Space v3)
=====================================
变更：
  - max_n   : 5 → 7  (START + 5个算子节点 + END)
  - nvt     : 7 → 8  (新增 type 7 = GCNII)
  - nz      : 8 → 12 (适应更深的拓扑复杂度)
  - op_mapping: 加入 7: 'GCNII'

⚠️  维度变化导致 VAE 权重不兼容，必须重新运行 train_joint.py。
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
    7: "GCNII",        # ← v3 新增
}


class HP_VAE(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, latent_dim: int = 4):
        super(HP_VAE, self).__init__()
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
    NUM_VERTEX_TYPE = 8    # ← v3: 7 → 8

    def __init__(self, arch_args, hp_latent_dim: int = 4):
        super(JointSpaceVAE, self).__init__()

        # Force v3 dimensions onto arch_args so callers need not remember
        arch_args.max_n           = 7                   # ← v3: 5 → 7
        arch_args.num_vertex_type = self.NUM_VERTEX_TYPE
        arch_args.START_TYPE      = self.START_TYPE
        arch_args.END_TYPE        = self.END_TYPE

        self.op_mapping = OP_MAPPING

        self.arch_vae = DVAE(
            max_n         = arch_args.max_n,            # 7
            nvt           = arch_args.num_vertex_type,  # 8
            START_TYPE    = arch_args.START_TYPE,
            END_TYPE      = arch_args.END_TYPE,
            hs            = arch_args.hs,               # 501
            nz            = arch_args.nz,               # 12
            bidirectional = arch_args.bidirectional,
        )

        self.hp_vae = HP_VAE(input_dim=2, latent_dim=hp_latent_dim)

        self.arch_nz  = arch_args.nz        # 12
        self.hp_nz    = hp_latent_dim        # 4
        self.total_nz = self.arch_nz + self.hp_nz   # 16

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

    def decode_from_joint_latent(self, z_total):
        z_arch = z_total[:, :self.arch_nz]
        z_hp   = z_total[:, self.arch_nz:]

        raw_hps        = self.hp_vae.decode(z_hp)
        learning_rates = torch.pow(10, raw_hps[:, 0]).tolist()
        dropouts       = torch.clamp(raw_hps[:, 1], 0.0, 0.6).tolist()

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

        return gnn_configs, learning_rates, dropouts