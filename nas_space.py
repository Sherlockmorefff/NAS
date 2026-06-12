"""Joint architecture and hyperparameter search space.

HP semantics are centralized in hp_modes.py. JointSpaceVAE now takes hp_mode
as the primary control variable while preserving explicit hp_input_dim and
hp_latent_dim compatibility for older checkpoints and scripts.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hp_modes import (
    MAX_OP_NODES,
    condition_mask_dict_from_ops,
    condition_mask_vector_from_ops,
    hp_dim_from_mode,
    norm_to_gat_heads,
    norm_to_gin_eps,
    norm_to_hidden,
    norm_to_l2,
    norm_to_sage_aggr,
    validate_hp_mode,
)
from models import DVAE


OP_MAPPING = {
    2: "GCNConv",
    3: "GATConv",
    4: "SAGEConv",
    5: "GINConv",
    6: "Identity",
    7: "GCNII",
}

LOG_LR_MIN = -4.0
LOG_LR_MAX = -1.5
DROPOUT_MIN = 0.1
DROPOUT_MAX = 0.6

HP_DIM_GLOBAL = hp_dim_from_mode("global4")
HP_DIM_HYBRID = hp_dim_from_mode("hybrid_cond7")
HP_DIM_LAYERWISE = hp_dim_from_mode("layer_cond19")
HP_DIM_TYPEWISE = HP_DIM_HYBRID
HP_DIM_LAYERWISE_LEGACY = 2 + 3 * MAX_OP_NODES


def op_types_from_types(types) -> list[int]:
    op_types = [int(t) for t in list(types)[1:1 + MAX_OP_NODES]]
    return op_types + [6] * (MAX_OP_NODES - len(op_types))


def ops_from_types(types) -> list[str]:
    return [OP_MAPPING.get(t, "Identity") for t in op_types_from_types(types)]


def _mode_from_hp_dim(hp_dim: int | None) -> str:
    if hp_dim == HP_DIM_GLOBAL:
        return "global4"
    if hp_dim == HP_DIM_HYBRID:
        return "hybrid_cond7"
    if hp_dim == HP_DIM_LAYERWISE:
        return "layer_cond19"
    raise ValueError(
        f"Unsupported hp_dim={hp_dim}. Only 4, 7, and 19 map to main hp_mode values."
    )


def condition_mask_from_types(
    types,
    hp_mode: str | None = None,
    hp_dim: int | None = None,
) -> list[float]:
    """Return active HP mask for a graph type vector.

    New code should pass hp_mode. If only hp_dim is provided, the function maps
    4->global4, 7->hybrid_cond7, and 19->layer_cond19. The legacy 17-dim
    layer-wise format is still recognized for reading older datasets.
    """

    ops = ops_from_types(types)
    if hp_mode is not None:
        return condition_mask_vector_from_ops(ops, validate_hp_mode(hp_mode))

    if hp_dim == HP_DIM_LAYERWISE_LEGACY:
        return (
            [1.0, 1.0]
            + [1.0 if op == "GATConv" else 0.0 for op in ops]
            + [1.0 if op == "SAGEConv" else 0.0 for op in ops]
            + [1.0 if op == "GINConv" else 0.0 for op in ops]
        )

    legacy_mode = "global4" if hp_dim is None else _mode_from_hp_dim(hp_dim)
    return condition_mask_vector_from_ops(ops, legacy_mode)


def condition_mask_from_graph(
    graph,
    hp_mode: str | None = None,
    hp_dim: int | None = None,
) -> list[float]:
    return condition_mask_from_types(graph.vs["type"], hp_mode=hp_mode, hp_dim=hp_dim)


def condition_masks_from_graphs(
    graphs,
    device=None,
    hp_mode: str | None = None,
    hp_dim: int | None = None,
) -> torch.Tensor:
    masks = [
        condition_mask_from_graph(g, hp_mode=hp_mode, hp_dim=hp_dim)
        for g in graphs
    ]
    return torch.tensor(masks, dtype=torch.float32, device=device)


class HP_VAE(nn.Module):
    """Variational autoencoder for normalized HP vectors."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, latent_dim: int = 4):
        super(HP_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
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
    START_TYPE = 0
    END_TYPE = 1
    NUM_VERTEX_TYPE = 8

    def __init__(
        self,
        arch_args,
        hp_mode: str | int = "global4",
        hp_latent_dim: int | None = None,
        hp_input_dim: int | None = None,
    ):
        super(JointSpaceVAE, self).__init__()

        if isinstance(hp_mode, int) and not isinstance(hp_mode, bool):
            hp_mode = _mode_from_hp_dim(hp_mode)
        elif not isinstance(hp_mode, str):
            raise TypeError(
                f"hp_mode must be str or int, got {type(hp_mode).__name__}"
            )

        self.hp_mode = validate_hp_mode(hp_mode)
        self.hp_dim = hp_dim_from_mode(self.hp_mode)

        if hp_input_dim is None:
            hp_input_dim = self.hp_dim
        elif hp_input_dim != self.hp_dim:
            print(
                "[JointSpaceVAE][WARNING] "
                f"hp_input_dim={hp_input_dim} differs from hp_mode={self.hp_mode} "
                f"hp_dim={self.hp_dim}. This is allowed only for legacy checkpoint "
                "compatibility; new code should keep hp_input_dim == hp_dim."
            )
        if hp_latent_dim is None:
            hp_latent_dim = hp_input_dim

        arch_args.max_n = 7
        arch_args.num_vertex_type = self.NUM_VERTEX_TYPE
        arch_args.START_TYPE = self.START_TYPE
        arch_args.END_TYPE = self.END_TYPE

        self.op_mapping = OP_MAPPING

        self.arch_vae = DVAE(
            max_n=arch_args.max_n,
            nvt=arch_args.num_vertex_type,
            START_TYPE=arch_args.START_TYPE,
            END_TYPE=arch_args.END_TYPE,
            hs=arch_args.hs,
            nz=arch_args.nz,
            bidirectional=arch_args.bidirectional,
        )

        self.hp_input_dim = int(hp_input_dim)
        self.hp_vae = HP_VAE(input_dim=self.hp_input_dim, latent_dim=int(hp_latent_dim))

        self.arch_nz = arch_args.nz
        self.hp_nz = int(hp_latent_dim)
        self.total_nz = self.arch_nz + self.hp_nz

        print(
            "[JointSpaceVAE] "
            f"hp_mode={self.hp_mode} "
            f"mode_hp_dim={self.hp_dim} "
            f"hp_input_dim={self.hp_input_dim} "
            f"hp_latent_dim={self.hp_nz} "
            f"total_nz={self.total_nz}"
        )

    def forward(self, graph_input, hp_input):
        mu_arch, logvar_arch = self.arch_vae.encode(graph_input)
        z_arch = self.arch_vae.reparameterize(mu_arch, logvar_arch)
        arch_decode_output = self.arch_vae.decode(z_arch)

        mu_hp, logvar_hp = self.hp_vae.encode(hp_input)
        z_hp = self.hp_vae.reparameterize(mu_hp, logvar_hp)
        hp_decode_output = self.hp_vae.decode(z_hp)

        return (arch_decode_output, mu_arch, logvar_arch), (
            hp_decode_output,
            mu_hp,
            logvar_hp,
        )

    def encode_to_joint_latent(self, graph_input, hp_input):
        mu_arch, _ = self.arch_vae.encode(graph_input)
        mu_hp, _ = self.hp_vae.encode(hp_input)
        return torch.cat([mu_arch, mu_hp], dim=-1)

    def _decode_graphs_to_configs(self, z_arch: torch.Tensor) -> list[dict[str, Any]]:
        graphs = self.arch_vae.decode(z_arch)
        gnn_configs = []

        for graph in graphs:
            actual_ops = []
            for i in range(1, graph.vcount() - 1):
                op_idx = graph.vs[i]["type"]
                op_name = self.op_mapping.get(op_idx, "Identity")
                actual_ops.append(op_name)
            edges = graph.get_edgelist()
            effective_layers = sum(1 for op in actual_ops if op != "Identity")
            gnn_configs.append(
                {
                    "effective_layers": effective_layers,
                    "operations": actual_ops,
                    "edges": edges,
                }
            )

        return gnn_configs

    def _prepare_hp_matrix(self, raw_hps: torch.Tensor) -> torch.Tensor:
        raw_hps = torch.clamp(raw_hps, 0.0, 1.0)
        if raw_hps.shape[1] == self.hp_dim:
            return raw_hps
        if raw_hps.shape[1] > self.hp_dim:
            return raw_hps[:, :self.hp_dim]

        pad = torch.zeros(
            raw_hps.shape[0],
            self.hp_dim - raw_hps.shape[1],
            dtype=raw_hps.dtype,
            device=raw_hps.device,
        )
        return torch.cat([raw_hps, pad], dim=1)

    def _decode_hp_row(self, hp_row: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
        hp = hp_row.detach().float().cpu().tolist()
        lr_norm = float(hp[0])
        dropout_norm = float(hp[1])
        hidden_norm = float(hp[2])
        l2_norm = float(hp[3])

        log_lr = lr_norm * (LOG_LR_MAX - LOG_LR_MIN) + LOG_LR_MIN
        learning_rate = float(10 ** log_lr)
        dropout = float(dropout_norm * (DROPOUT_MAX - DROPOUT_MIN) + DROPOUT_MIN)
        hidden_dim = norm_to_hidden(hidden_norm)
        weight_decay = norm_to_l2(l2_norm)

        ops = list(config.get("operations", []))[:MAX_OP_NODES]
        if len(ops) < MAX_OP_NODES:
            ops.extend(["Identity"] * (MAX_OP_NODES - len(ops)))

        gat_heads = 1
        sage_aggr = "mean"
        gin_eps = 0.0
        gat_heads_by_layer = None
        sage_aggr_by_layer = None
        gin_eps_by_layer = None

        if self.hp_mode == "hybrid_cond7":
            if "GATConv" in ops:
                gat_heads = norm_to_gat_heads(hp[4])
            if "SAGEConv" in ops:
                sage_aggr = norm_to_sage_aggr(hp[5])
            if "GINConv" in ops:
                gin_eps = norm_to_gin_eps(hp[6])

        elif self.hp_mode == "layer_cond19":
            gat_norms = hp[4:4 + MAX_OP_NODES]
            sage_norms = hp[4 + MAX_OP_NODES:4 + 2 * MAX_OP_NODES]
            gin_norms = hp[4 + 2 * MAX_OP_NODES:4 + 3 * MAX_OP_NODES]
            gat_heads_by_layer = []
            sage_aggr_by_layer = []
            gin_eps_by_layer = []

            for i, op in enumerate(ops):
                gat_heads_by_layer.append(
                    norm_to_gat_heads(gat_norms[i]) if op == "GATConv" else 1
                )
                sage_aggr_by_layer.append(
                    norm_to_sage_aggr(sage_norms[i]) if op == "SAGEConv" else "mean"
                )
                gin_eps_by_layer.append(
                    norm_to_gin_eps(gin_norms[i]) if op == "GINConv" else 0.0
                )

            for i, op in enumerate(ops):
                if op == "GATConv":
                    gat_heads = gat_heads_by_layer[i]
                    break
            for i, op in enumerate(ops):
                if op == "SAGEConv":
                    sage_aggr = sage_aggr_by_layer[i]
                    break
            for i, op in enumerate(ops):
                if op == "GINConv":
                    gin_eps = gin_eps_by_layer[i]
                    break

        return {
            "hp_mode": self.hp_mode,
            "lr": learning_rate,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "l2": weight_decay,
            "gat_heads": gat_heads,
            "sage_aggr": sage_aggr,
            "gin_eps": gin_eps,
            "gat_heads_by_layer": gat_heads_by_layer,
            "sage_aggr_by_layer": sage_aggr_by_layer,
            "gin_eps_by_layer": gin_eps_by_layer,
            "condition_mask": condition_mask_dict_from_ops(ops, self.hp_mode),
            "condition_mask_vector": condition_mask_vector_from_ops(ops, self.hp_mode),
            "hp_norm": hp,
        }

    def decode_from_joint_latent(
        self,
        z_total,
        return_full: bool = False,
        return_hp_dicts: bool = False,
    ):
        """Decode joint latent vectors into architecture configs and HP values.

        Default return stays backward-compatible:
            (gnn_configs, learning_rates, dropouts)

        return_hp_dicts=True returns:
            (gnn_configs, hp_dicts)

        return_full=True returns complete HP fields in a mode-specific tuple.
        """

        z_arch = z_total[:, :self.arch_nz]
        z_hp = z_total[:, self.arch_nz:self.arch_nz + self.hp_nz]

        raw_hps = self.hp_vae.decode(z_hp)
        hps = self._prepare_hp_matrix(raw_hps)
        gnn_configs = self._decode_graphs_to_configs(z_arch)
        hp_dicts = [
            self._decode_hp_row(hps[i], cfg)
            for i, cfg in enumerate(gnn_configs)
        ]

        learning_rates = [hp["lr"] for hp in hp_dicts]
        dropouts = [hp["dropout"] for hp in hp_dicts]

        if return_hp_dicts:
            return gnn_configs, hp_dicts

        if not return_full:
            return gnn_configs, learning_rates, dropouts

        hidden_dims = [hp["hidden_dim"] for hp in hp_dicts]
        weight_decays = [hp["weight_decay"] for hp in hp_dicts]

        if self.hp_mode == "global4":
            return (
                gnn_configs,
                learning_rates,
                dropouts,
                hidden_dims,
                weight_decays,
            )

        if self.hp_mode == "hybrid_cond7":
            gat_heads = [hp["gat_heads"] for hp in hp_dicts]
            sage_aggrs = [hp["sage_aggr"] for hp in hp_dicts]
            gin_eps = [hp["gin_eps"] for hp in hp_dicts]
            return (
                gnn_configs,
                learning_rates,
                dropouts,
                hidden_dims,
                weight_decays,
                gat_heads,
                sage_aggrs,
                gin_eps,
            )

        gat_heads_by_layer = [hp["gat_heads_by_layer"] for hp in hp_dicts]
        sage_aggr_by_layer = [hp["sage_aggr_by_layer"] for hp in hp_dicts]
        gin_eps_by_layer = [hp["gin_eps_by_layer"] for hp in hp_dicts]
        return (
            gnn_configs,
            learning_rates,
            dropouts,
            hidden_dims,
            weight_decays,
            gat_heads_by_layer,
            sage_aggr_by_layer,
            gin_eps_by_layer,
        )
