"""
nas_space.py  v5  (Search Space v3 + 条件参数 4 维对齐)
======================================================
v5 变更（相对 v4）：
  - HP_VAE 4 维语义统一为条件参数向量：
      [0] log_lr_norm       (全局生效)
      [1] dropout_norm      (全局生效)
      [2] gat_heads_norm    (仅当架构含 GATConv 时生效)
      [3] sage_aggr_norm    (仅当架构含 SAGEConv 时生效)
  - decode_from_joint_latent(return_full=True) 返回 gat_heads/sage_aggr，
    并根据解码架构自动 mask 非活跃条件参数。
  - JointSpaceVAE: HP_VAE(input_dim=4)

⚠️  HP VAE 条件参数语义变化导致 v4 及更早数据/权重不兼容，
    必须重新运行 generate_mini_data.py（v5）和 train_joint.py。
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

LOG_LR_MIN = -4.0
LOG_LR_MAX = -1.5
DROPOUT_MIN = 0.1
DROPOUT_MAX = 0.6
GAT_HEAD_OPTIONS = [1, 2, 4, 8]
SAGE_AGGR_OPTIONS = ["mean", "max", "add"]


def _norm_index(x: float, n: int) -> int:
    x = max(0.0, min(float(x), 1.0 - 1e-8))
    return min(int(x * n), n - 1)


def norm_to_gat_heads(x: float) -> int:
    return GAT_HEAD_OPTIONS[_norm_index(x, len(GAT_HEAD_OPTIONS))]


def norm_to_sage_aggr(x: float) -> str:
    return SAGE_AGGR_OPTIONS[_norm_index(x, len(SAGE_AGGR_OPTIONS))]


def condition_mask_from_types(types) -> list:
    type_set = {int(t) for t in types}
    return [
        1.0,
        1.0,
        1.0 if 3 in type_set else 0.0,
        1.0 if 4 in type_set else 0.0,
    ]


def condition_mask_from_graph(graph) -> list:
    return condition_mask_from_types(graph.vs["type"])


def condition_masks_from_graphs(graphs, device=None) -> torch.Tensor:
    masks = [condition_mask_from_graph(g) for g in graphs]
    return torch.tensor(masks, dtype=torch.float32, device=device)


class HP_VAE(nn.Module):
    """超参数变分自编码器。

    input_dim=4 时编码 [log_lr_norm, dropout_norm,
    gat_heads_norm, sage_aggr_norm]（v5 默认）。
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

        # v5: HP VAE 编码 4 维条件参数，非活跃条件位由训练侧 mask
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
                                  gat_heads, sage_aggrs)，支持条件参数解码。

        Returns
        -------
        gnn_configs    : list of dict
        learning_rates : list of float
        dropouts       : list of float
        gat_heads      : list of int  (仅 return_full=True)
        sage_aggrs     : list of str  (仅 return_full=True)
        """
        z_arch = z_total[:, :self.arch_nz]
        z_hp   = z_total[:, self.arch_nz:]

        raw_hps        = self.hp_vae.decode(z_hp)
        lr_norm        = torch.clamp(raw_hps[:, 0], 0.0, 1.0)
        drop_norm      = torch.clamp(raw_hps[:, 1], 0.0, 1.0)
        log_lrs        = lr_norm * (LOG_LR_MAX - LOG_LR_MIN) + LOG_LR_MIN
        learning_rates = torch.pow(10, log_lrs).tolist()
        dropouts       = (drop_norm * (DROPOUT_MAX - DROPOUT_MIN) + DROPOUT_MIN).tolist()

        gat_norms = raw_hps[:, 2].tolist() if raw_hps.shape[1] >= 3 \
            else [0.0] * raw_hps.shape[0]
        sage_norms = raw_hps[:, 3].tolist() if raw_hps.shape[1] >= 4 \
            else [0.0] * raw_hps.shape[0]

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
            gat_heads = []
            sage_aggrs = []
            for cfg, gat_n, sage_n in zip(gnn_configs, gat_norms, sage_norms):
                ops = cfg["operations"]
                gat_heads.append(norm_to_gat_heads(gat_n) if "GATConv" in ops else 1)
                sage_aggrs.append(
                    norm_to_sage_aggr(sage_n) if "SAGEConv" in ops else "mean")
            return gnn_configs, learning_rates, dropouts, gat_heads, sage_aggrs
        return gnn_configs, learning_rates, dropouts
