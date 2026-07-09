"""Evaluation utilities for decoded NAS architectures.

This module evaluates a joint search vector

    z_search = [z_arch, hp]

where HP semantics are controlled by hp_mode instead of inferred from hp_dim.
The supported main modes are defined in hp_modes.py:

    global4, hybrid_cond7, layer_cond19
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv, SAGEConv

from hp_modes import (
    GAT_HEAD_OPTIONS,
    GIN_EPS_MAX,
    GIN_EPS_MIN,
    HIDDEN_DIM_OPTIONS,
    L2_OPTIONS,
    MAX_OP_NODES,
    SAGE_AGGR_OPTIONS,
    condition_mask_dict_from_ops,
    hp_dim_from_mode,
    norm_to_gat_heads,
    norm_to_gin_eps,
    norm_to_hidden,
    norm_to_l2,
    norm_to_sage_aggr,
    validate_hp_mode,
)


MAX_EPOCHS = 150
PATIENCE = 20
HIDDEN = 64

HP_DIM_GLOBAL = hp_dim_from_mode("global4")
HP_DIM_HYBRID = hp_dim_from_mode("hybrid_cond7")
HP_DIM_LAYERWISE = hp_dim_from_mode("layer_cond19")
HP_DIM_TYPEWISE = HP_DIM_HYBRID
HP_DIM_TYPEWISE_LEGACY = 4
HP_DIM_LAYERWISE_LEGACY = 2 + 3 * MAX_OP_NODES

DEFAULT_GAT_HEADS = 1
DEFAULT_SAGE_AGGR = "mean"
DEFAULT_GIN_EPS = 0.0

OP_REG = {
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "SAGEConv": SAGEConv,
    "GINConv": GINConv,
}


def _match_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Return ``x`` with feature width matched to ``target_dim``.

    The decoded search space can produce skip/identity wiring where the tensor
    reaching a layer does not have the static width used when that layer was
    constructed.  Truncating or zero-padding preserves the original device,
    dtype, and autograd path for the existing feature columns.
    """

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if x.shape[1] == target_dim:
        return x
    if x.shape[1] > target_dim:
        return x[:, :target_dim]
    pad = x.new_zeros(x.shape[0], target_dim - x.shape[1])
    return torch.cat([x, pad], dim=1)


def _invalid_eval_result(track_test: bool) -> tuple:
    return (0.0, False, 0.0) if track_test else (0.0, False)


def _empty_cuda_cache_if_oom(exc: BaseException) -> None:
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _warn_dynamic_gnn_failure(stage: str, exc: BaseException) -> None:
    warnings.warn(
        f"[DynamicGNN {stage} failed] {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


def _clip01(x: float) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        value = 0.0
    if not np.isfinite(value):
        value = 0.0
    return float(np.clip(value, 0.0, 1.0))


def _tensor_to_1d_cpu(z: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
    if isinstance(z, torch.Tensor):
        return z.detach().float().flatten().cpu()
    return torch.as_tensor(z, dtype=torch.float32).flatten().cpu()


def _ops_from_config(config: dict | None) -> list[str]:
    if not config:
        return []
    return [str(op) for op in list(config.get("operations", []))]


def _ops5_from_config(config: dict | None) -> list[str]:
    ops = _ops_from_config(config)[:MAX_OP_NODES]
    if len(ops) < MAX_OP_NODES:
        ops.extend(["Identity"] * (MAX_OP_NODES - len(ops)))
    return ops


def _legacy_condition_mask_from_config(config: dict | None) -> dict[str, Any]:
    ops5 = _ops5_from_config(config)
    return {
        "gat_heads": any(op == "GATConv" for op in ops5),
        "sage_aggr": any(op == "SAGEConv" for op in ops5),
        "gin_eps": any(op == "GINConv" for op in ops5),
        "gat_by_layer": [op == "GATConv" for op in ops5],
        "sage_by_layer": [op == "SAGEConv" for op in ops5],
        "gin_by_layer": [op == "GINConv" for op in ops5],
    }


def condition_mask_from_config(config: dict | None, hp_mode: str | None = None) -> dict[str, Any]:
    """Return condition-mask metadata for a decoded architecture.

    With hp_mode=None, this preserves the historical dictionary shape used by
    older BO scripts. With a mode, it delegates to hp_modes.py and includes the
    full mode-specific HP mask vector and by-name mapping.
    """

    if hp_mode is None:
        return _legacy_condition_mask_from_config(config)
    return condition_mask_dict_from_ops(_ops_from_config(config), hp_mode)


def _layer_values(values, default, n: int = MAX_OP_NODES) -> list:
    if values is None:
        return [default for _ in range(n)]
    if isinstance(values, (list, tuple)):
        out = list(values[:n])
        return out + [default for _ in range(n - len(out))]
    return [values for _ in range(n)]


def decode_layerwise_conditional_hp(
    z_search: torch.Tensor,
    arch_nz: int,
    config: dict | None,
) -> tuple[list[int], list[str], list[float]]:
    """Compatibility helper for layer-wise conditional HP decoding.

    New layer_cond19 vectors use offsets:
        [lr, dropout, hidden_dim, l2, gat_0..4, sage_0..4, gin_0..4]

    Legacy layer_cond17 vectors used offsets:
        [lr, dropout, gat_0..4, sage_0..4, gin_0..4]
    """

    z_flat = _tensor_to_1d_cpu(z_search)
    hp_dim = int(z_flat.numel() - arch_nz)
    if hp_dim >= HP_DIM_LAYERWISE:
        start = arch_nz + 4
    elif hp_dim >= HP_DIM_LAYERWISE_LEGACY:
        start = arch_nz + 2
    else:
        return (
            [DEFAULT_GAT_HEADS] * MAX_OP_NODES,
            [DEFAULT_SAGE_AGGR] * MAX_OP_NODES,
            [DEFAULT_GIN_EPS] * MAX_OP_NODES,
        )

    mask = _legacy_condition_mask_from_config(config)
    gat_norms = z_flat[start:start + MAX_OP_NODES]
    sage_norms = z_flat[start + MAX_OP_NODES:start + 2 * MAX_OP_NODES]
    gin_norms = z_flat[start + 2 * MAX_OP_NODES:start + 3 * MAX_OP_NODES]

    gat_heads = [DEFAULT_GAT_HEADS] * MAX_OP_NODES
    sage_aggr = [DEFAULT_SAGE_AGGR] * MAX_OP_NODES
    gin_eps = [DEFAULT_GIN_EPS] * MAX_OP_NODES

    for i in range(MAX_OP_NODES):
        if mask["gat_by_layer"][i]:
            gat_heads[i] = norm_to_gat_heads(float(gat_norms[i]))
        if mask["sage_by_layer"][i]:
            sage_aggr[i] = norm_to_sage_aggr(float(sage_norms[i]))
        if mask["gin_by_layer"][i]:
            gin_eps[i] = norm_to_gin_eps(float(gin_norms[i]))

    return gat_heads, sage_aggr, gin_eps


def decode_hp_by_mode(
    z_search: torch.Tensor,
    arch_nz: int,
    config: dict,
    hp_mode: str,
    log_lr_min: float,
    log_lr_max: float,
    dropout_min: float,
    dropout_max: float,
    default_hidden_dim: int = 64,
    default_weight_decay: float = 5e-4,
) -> dict[str, Any]:
    """Decode the HP slice of z_search according to hp_mode.

    Returns a dictionary containing:
        lr, dropout, hidden_dim, weight_decay, gat_heads, sage_aggr, gin_eps,
        gat_heads_by_layer, sage_aggr_by_layer, gin_eps_by_layer,
        condition_mask
    """

    hp_mode = validate_hp_mode(hp_mode)
    z_flat = _tensor_to_1d_cpu(z_search)
    hp_dim = hp_dim_from_mode(hp_mode)
    expected = arch_nz + hp_dim
    if z_flat.numel() < expected:
        raise ValueError(
            f"z_search length {z_flat.numel()} is too short for hp_mode={hp_mode}; "
            f"expected at least {expected}"
        )

    hp = z_flat[arch_nz:expected]
    lr_norm = _clip01(float(hp[0]))
    dropout_norm = _clip01(float(hp[1]))
    hidden_norm = _clip01(float(hp[2]))
    l2_norm = _clip01(float(hp[3]))

    log_lr = lr_norm * (log_lr_max - log_lr_min) + log_lr_min
    lr = float(np.clip(10 ** log_lr, 1e-6, 1.0))
    dropout = float(np.clip(
        dropout_norm * (dropout_max - dropout_min) + dropout_min,
        0.0,
        0.9,
    ))
    hidden_dim = norm_to_hidden(hidden_norm)
    weight_decay = norm_to_l2(l2_norm)

    ops = _ops_from_config(config)
    mask = condition_mask_dict_from_ops(ops, hp_mode)

    gat_heads = DEFAULT_GAT_HEADS
    sage_aggr = DEFAULT_SAGE_AGGR
    gin_eps = DEFAULT_GIN_EPS
    gat_heads_by_layer = None
    sage_aggr_by_layer = None
    gin_eps_by_layer = None

    if hp_mode == "hybrid_cond7":
        gat_heads_norm = _clip01(float(hp[4]))
        sage_aggr_norm = _clip01(float(hp[5]))
        gin_eps_norm = _clip01(float(hp[6]))

        if mask["gat_heads"]:
            gat_heads = norm_to_gat_heads(gat_heads_norm)
        if mask["sage_aggr"]:
            sage_aggr = norm_to_sage_aggr(sage_aggr_norm)
        if mask["gin_eps"]:
            gin_eps = norm_to_gin_eps(gin_eps_norm)

    elif hp_mode == "layer_cond19":
        ops5 = _ops5_from_config(config)
        gat_norms = hp[4:4 + MAX_OP_NODES]
        sage_norms = hp[4 + MAX_OP_NODES:4 + 2 * MAX_OP_NODES]
        gin_norms = hp[4 + 2 * MAX_OP_NODES:4 + 3 * MAX_OP_NODES]

        gat_heads_by_layer = []
        sage_aggr_by_layer = []
        gin_eps_by_layer = []

        for i, op in enumerate(ops5):
            if op == "GATConv":
                gat_heads_by_layer.append(norm_to_gat_heads(float(gat_norms[i])))
            else:
                gat_heads_by_layer.append(DEFAULT_GAT_HEADS)

            if op == "SAGEConv":
                sage_aggr_by_layer.append(norm_to_sage_aggr(float(sage_norms[i])))
            else:
                sage_aggr_by_layer.append(DEFAULT_SAGE_AGGR)

            if op == "GINConv":
                gin_eps_by_layer.append(norm_to_gin_eps(float(gin_norms[i])))
            else:
                gin_eps_by_layer.append(DEFAULT_GIN_EPS)

        gat_heads = next(
            (
                gat_heads_by_layer[i]
                for i, is_active in enumerate(mask["gat_by_layer"])
                if is_active
            ),
            DEFAULT_GAT_HEADS,
        )
        sage_aggr = next(
            (
                sage_aggr_by_layer[i]
                for i, is_active in enumerate(mask["sage_by_layer"])
                if is_active
            ),
            DEFAULT_SAGE_AGGR,
        )
        gin_eps = next(
            (
                gin_eps_by_layer[i]
                for i, is_active in enumerate(mask["gin_by_layer"])
                if is_active
            ),
            DEFAULT_GIN_EPS,
        )

    return {
        "lr": lr,
        "dropout": dropout,
        "hidden_dim": int(hidden_dim),
        "weight_decay": float(weight_decay),
        "l2": float(weight_decay),
        "gat_heads": int(gat_heads),
        "sage_aggr": str(sage_aggr),
        "gin_eps": float(gin_eps),
        "gat_heads_by_layer": gat_heads_by_layer,
        "sage_aggr_by_layer": sage_aggr_by_layer,
        "gin_eps_by_layer": gin_eps_by_layer,
        "condition_mask": mask,
        "condition_mask_vector": mask["vector"],
        "hp_norm": [float(_clip01(v)) for v in hp.tolist()],
        "hp_mode": hp_mode,
        "hp_dim": hp_dim,
    }


class DynamicGNN(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.5,
        hidden_dim: int = HIDDEN,
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        gat_heads: int = DEFAULT_GAT_HEADS,
        sage_aggr: str = DEFAULT_SAGE_AGGR,
        gin_eps: float = DEFAULT_GIN_EPS,
        gat_heads_by_layer=None,
        sage_aggr_by_layer=None,
        gin_eps_by_layer=None,
    ):
        super().__init__()
        self.ops = config["operations"]
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.gat_heads = max(1, int(gat_heads))
        self.sage_aggr = sage_aggr if sage_aggr in SAGE_AGGR_OPTIONS else DEFAULT_SAGE_AGGR
        self.gin_eps = float(gin_eps)
        self.gat_heads_by_layer = [
            max(1, int(v)) for v in _layer_values(gat_heads_by_layer, self.gat_heads)
        ]
        self.sage_aggr_by_layer = [
            v if v in SAGE_AGGR_OPTIONS else DEFAULT_SAGE_AGGR
            for v in _layer_values(sage_aggr_by_layer, self.sage_aggr)
        ]
        self.gin_eps_by_layer = [
            float(v) for v in _layer_values(gin_eps_by_layer, self.gin_eps)
        ]

        self.edges = list(config.get("edges", []))
        n_ops = len(self.ops)
        if not self.edges:
            self.edges = [(i, i + 1) for i in range(n_ops + 1)]

        self.n_nodes = n_ops + 2
        self._predecessors = defaultdict(list)
        for src, dst in self.edges:
            self._predecessors[dst].append(src)

        self.has_gcnii = "GCNII" in self.ops
        if self.has_gcnii:
            self.input_proj = Linear(in_ch, hidden_dim)
        else:
            self.input_proj = None

        self.layers = ModuleList()
        self.layer_input_dims: list[int | None] = []
        gcnii_layer_idx = 1
        node_out_dim = {0: in_ch}

        for i, op in enumerate(self.ops):
            node_idx = i + 1
            preds = self._predecessors.get(node_idx, [node_idx - 1])
            in_dim = node_out_dim.get(preds[0], hidden_dim) if preds else in_ch

            if op == "Identity":
                self.layers.append(None)
                self.layer_input_dims.append(None)
                node_out_dim[node_idx] = in_dim

            elif op == "GINConv":
                self.layer_input_dims.append(in_dim)
                mlp = Sequential(
                    Linear(in_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                self.layers.append(
                    GINConv(
                        mlp,
                        eps=self.gin_eps_by_layer[i],
                        train_eps=False,
                    )
                )
                node_out_dim[node_idx] = hidden_dim

            elif op == "GCNII":
                self.layer_input_dims.append(hidden_dim)
                self.layers.append(
                    GCN2Conv(
                        channels=hidden_dim,
                        alpha=gcnii_alpha,
                        theta=gcnii_theta,
                        layer=gcnii_layer_idx,
                        shared_weights=True,
                    )
                )
                gcnii_layer_idx += 1
                node_out_dim[node_idx] = hidden_dim

            elif op == "GATConv":
                self.layer_input_dims.append(in_dim)
                self.layers.append(
                    GATConv(
                        in_dim,
                        hidden_dim,
                        heads=self.gat_heads_by_layer[i],
                        concat=False,
                    )
                )
                node_out_dim[node_idx] = hidden_dim

            elif op == "SAGEConv":
                self.layer_input_dims.append(in_dim)
                self.layers.append(
                    SAGEConv(
                        in_dim,
                        hidden_dim,
                        aggr=self.sage_aggr_by_layer[i],
                    )
                )
                node_out_dim[node_idx] = hidden_dim

            else:
                self.layer_input_dims.append(in_dim)
                self.layers.append(OP_REG[op](in_dim, hidden_dim))
                node_out_dim[node_idx] = hidden_dim

        end_node_idx = self.n_nodes - 1
        end_preds = self._predecessors.get(end_node_idx, [self.n_nodes - 2])
        end_in_dim = node_out_dim.get(end_preds[0], hidden_dim) if end_preds else hidden_dim
        self.clf = Linear(end_in_dim, out_ch)
        self._node_out_dim = node_out_dim

    def forward(self, x, edge_index):
        if self.has_gcnii and self.input_proj is not None:
            x_0 = F.relu(self.input_proj(x))
        else:
            x_0 = None

        node_states: dict[int, torch.Tensor] = {0: x}

        for i, (layer, op) in enumerate(zip(self.layers, self.ops)):
            node_idx = i + 1
            preds = self._predecessors.get(node_idx, [node_idx - 1])

            agg = None
            for pred in preds:
                pred_feat = node_states.get(pred, x)
                if agg is None:
                    agg = pred_feat
                else:
                    if agg.shape[1] != pred_feat.shape[1]:
                        min_dim = min(agg.shape[1], pred_feat.shape[1])
                        agg = agg[:, :min_dim]
                        pred_feat = pred_feat[:, :min_dim]
                    agg = agg + pred_feat

            if agg is None:
                agg = x

            if op == "Identity" or layer is None:
                out = agg
            elif op == "GCNII":
                expected_input_dim = int(self.layer_input_dims[i] or self.hidden_dim)
                if agg.shape[1] == self.in_ch and self.input_proj is not None:
                    agg = F.relu(self.input_proj(agg))
                agg = _match_feature_dim(agg, expected_input_dim)
                out = F.dropout(
                    F.relu(layer(agg, x_0, edge_index)),
                    self.dropout,
                    self.training,
                )
            else:
                expected_input_dim = self.layer_input_dims[i]
                if expected_input_dim is None:
                    raise RuntimeError(f"missing input dimension for layer {i} ({op})")
                agg = _match_feature_dim(agg, expected_input_dim)
                out = F.dropout(
                    F.relu(layer(agg, edge_index)),
                    self.dropout,
                    self.training,
                )

            node_states[node_idx] = out

        end_node_idx = self.n_nodes - 1
        end_preds = self._predecessors.get(end_node_idx, [self.n_nodes - 2])
        agg_end = None
        for pred in end_preds:
            feat = node_states.get(pred, list(node_states.values())[-1])
            if agg_end is None:
                agg_end = feat
            else:
                if agg_end.shape[1] != feat.shape[1]:
                    min_dim = min(agg_end.shape[1], feat.shape[1])
                    agg_end = agg_end[:, :min_dim]
                    feat = feat[:, :min_dim]
                agg_end = agg_end + feat

        if agg_end is None:
            agg_end = list(node_states.values())[-1]

        agg_end = _match_feature_dim(agg_end, self.clf.in_features)

        return self.clf(agg_end)


def train_and_eval_arch(
    config: dict,
    data,
    in_ch: int,
    out_ch: int,
    lr: float = 1e-3,
    dropout: float = 0.5,
    device=None,
    hidden_dim: int = HIDDEN,
    weight_decay: float = 5e-4,
    gcnii_alpha: float = 0.1,
    gcnii_theta: float = 0.5,
    gat_heads: int = DEFAULT_GAT_HEADS,
    sage_aggr: str = DEFAULT_SAGE_AGGR,
    gin_eps: float = DEFAULT_GIN_EPS,
    gat_heads_by_layer=None,
    sage_aggr_by_layer=None,
    gin_eps_by_layer=None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seed: int = None,
    track_test: bool = False,
) -> tuple:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config is None or config.get("effective_layers", 0) == 0:
        return (0.0, False, 0.0) if track_test else (0.0, False)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        gnn = DynamicGNN(
            config,
            in_ch,
            out_ch,
            dropout=dropout,
            hidden_dim=hidden_dim,
            gcnii_alpha=gcnii_alpha,
            gcnii_theta=gcnii_theta,
            gat_heads=gat_heads,
            sage_aggr=sage_aggr,
            gin_eps=gin_eps,
            gat_heads_by_layer=gat_heads_by_layer,
            sage_aggr_by_layer=sage_aggr_by_layer,
            gin_eps_by_layer=gin_eps_by_layer,
        ).to(device)
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        _empty_cuda_cache_if_oom(exc)
        _warn_dynamic_gnn_failure("build", exc)
        return _invalid_eval_result(track_test)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=lr * 0.01,
    )

    best_val = 0.0
    best_test = 0.0
    no_improve = 0

    try:
        for _epoch in range(max_epochs):
            gnn.train()
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index)
            if not bool(torch.isfinite(out).all().detach().cpu().item()):
                raise FloatingPointError("non-finite DynamicGNN training output")
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError("non-finite DynamicGNN training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()

            gnn.eval()
            with torch.no_grad():
                eval_out = gnn(data.x, data.edge_index)
                if not bool(torch.isfinite(eval_out).all().detach().cpu().item()):
                    raise FloatingPointError("non-finite DynamicGNN validation output")
                pred = eval_out.argmax(dim=1)
                val_count = int(data.val_mask.sum().item())
                if val_count <= 0:
                    raise ValueError("validation mask is empty")
                val_acc = (
                    pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
                    / val_count
                )
                if track_test:
                    test_count = int(data.test_mask.sum().item())
                    if test_count <= 0:
                        raise ValueError("test mask is empty")
                    test_acc = (
                        pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
                        / test_count
                    )

            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
                if track_test:
                    best_test = test_acc
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        _empty_cuda_cache_if_oom(exc)
        _warn_dynamic_gnn_failure("eval", exc)
        return _invalid_eval_result(track_test)

    if track_test:
        return best_val, True, best_test
    return best_val, True


def eval_z_search(
    vae,
    z_search: torch.Tensor,
    data,
    in_ch: int,
    out_ch: int,
    arch_nz: int,
    log_lr_min: float,
    log_lr_max: float,
    dropout_min: float,
    dropout_max: float,
    device,
    has_hidden: bool = False,
    has_l2: bool = False,
    use_conditional_params: bool = False,
    default_hidden_dim: int = HIDDEN,
    default_weight_decay: float = 5e-4,
    gcnii_alpha: float = 0.1,
    gcnii_theta: float = 0.5,
    n_trials: int = 3,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    return_layerwise: bool = False,
    hp_mode: str = "global4",
    return_hp: bool = False,
) -> tuple | dict[str, Any]:
    """Decode z_search, train the architecture, and return validation score.

    Main HP decoding is driven only by hp_mode. The legacy flags has_hidden,
    has_l2, and use_conditional_params are kept for return-shape compatibility
    with existing callers; they no longer decide HP semantics.
    """

    hp_mode = validate_hp_mode(hp_mode)
    z_flat = _tensor_to_1d_cpu(z_search)
    z_arch = z_search[:arch_nz]
    config = _decode_arch(vae, z_arch, device, n_trials)

    try:
        hp = decode_hp_by_mode(
            z_search=z_search,
            arch_nz=arch_nz,
            config=config,
            hp_mode=hp_mode,
            log_lr_min=log_lr_min,
            log_lr_max=log_lr_max,
            dropout_min=dropout_min,
            dropout_max=dropout_max,
            default_hidden_dim=default_hidden_dim,
            default_weight_decay=default_weight_decay,
        )
    except Exception as exc:
        print(f"  [HP decode failed] {exc}")
        hp = {
            "lr": 1e-3,
            "dropout": 0.5,
            "hidden_dim": int(default_hidden_dim),
            "weight_decay": float(default_weight_decay),
            "l2": float(default_weight_decay),
            "gat_heads": DEFAULT_GAT_HEADS,
            "sage_aggr": DEFAULT_SAGE_AGGR,
            "gin_eps": DEFAULT_GIN_EPS,
            "gat_heads_by_layer": None,
            "sage_aggr_by_layer": None,
            "gin_eps_by_layer": None,
            "condition_mask": condition_mask_from_config(config, hp_mode=hp_mode),
            "condition_mask_vector": condition_mask_from_config(config, hp_mode=hp_mode)["vector"],
            "hp_norm": [],
            "hp_mode": hp_mode,
            "hp_dim": hp_dim_from_mode(hp_mode),
        }
        val_acc, is_valid = 0.0, False
        if return_hp:
            return {
                "val_acc": val_acc,
                "valid": is_valid,
                "config": config,
                "z_arch": z_flat[:arch_nz].tolist(),
                "z_search": z_flat.tolist(),
                "hp": hp,
                "condition_mask_vector": hp["condition_mask_vector"],
            }
        if return_layerwise:
            return (
                val_acc,
                hp["lr"],
                hp["dropout"],
                hp["gat_heads_by_layer"],
                hp["sage_aggr_by_layer"],
                hp["gin_eps_by_layer"],
                is_valid,
            )
        if use_conditional_params:
            return (
                val_acc,
                hp["lr"],
                hp["dropout"],
                hp["gat_heads"],
                hp["sage_aggr"],
                is_valid,
            )
        return (
            val_acc,
            hp["lr"],
            hp["dropout"],
            hp["hidden_dim"],
            hp["weight_decay"],
            is_valid,
        )

    val_acc, is_valid = train_and_eval_arch(
        config,
        data,
        in_ch,
        out_ch,
        lr=hp["lr"],
        dropout=hp["dropout"],
        hidden_dim=hp["hidden_dim"],
        weight_decay=hp["weight_decay"],
        gcnii_alpha=gcnii_alpha,
        gcnii_theta=gcnii_theta,
        gat_heads=hp["gat_heads"],
        sage_aggr=hp["sage_aggr"],
        gin_eps=hp["gin_eps"],
        gat_heads_by_layer=hp["gat_heads_by_layer"],
        sage_aggr_by_layer=hp["sage_aggr_by_layer"],
        gin_eps_by_layer=hp["gin_eps_by_layer"],
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        track_test=False,
    )

    if return_hp:
        return {
            "val_acc": float(val_acc),
            "valid": bool(is_valid),
            "config": config,
            "operations": [] if config is None else list(config.get("operations", [])),
            "edges": [] if config is None else list(config.get("edges", [])),
            "z_arch": z_flat[:arch_nz].tolist(),
            "z_search": z_flat.tolist(),
            "hp": hp,
            "lr": hp["lr"],
            "dropout": hp["dropout"],
            "hidden_dim": hp["hidden_dim"],
            "l2": hp["weight_decay"],
            "gat_heads": hp["gat_heads"],
            "sage_aggr": hp["sage_aggr"],
            "gin_eps": hp["gin_eps"],
            "gat_heads_by_layer": hp["gat_heads_by_layer"],
            "sage_aggr_by_layer": hp["sage_aggr_by_layer"],
            "gin_eps_by_layer": hp["gin_eps_by_layer"],
            "condition_mask": hp["condition_mask"],
            "condition_mask_vector": hp["condition_mask_vector"],
            "hp_mode": hp_mode,
            "search_dim": int(arch_nz + hp_dim_from_mode(hp_mode)),
        }

    if return_layerwise:
        return (
            val_acc,
            hp["lr"],
            hp["dropout"],
            hp["gat_heads_by_layer"],
            hp["sage_aggr_by_layer"],
            hp["gin_eps_by_layer"],
            is_valid,
        )

    if use_conditional_params:
        return (
            val_acc,
            hp["lr"],
            hp["dropout"],
            hp["gat_heads"],
            hp["sage_aggr"],
            is_valid,
        )

    return (
        val_acc,
        hp["lr"],
        hp["dropout"],
        hp["hidden_dim"],
        hp["weight_decay"],
        is_valid,
    )


def _decode_arch(vae, z_arch: torch.Tensor, device, n_trials: int = 3):
    from collections import Counter

    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(device))
            g = graphs[0]
            ops = [
                vae.op_mapping.get(g.vs[i]["type"], "Identity")
                for i in range(1, g.vcount() - 1)
            ]
            edges = g.get_edgelist()
            n_eff = sum(1 for op in ops if op != "Identity")
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append(
                    (
                        key,
                        {
                            "effective_layers": n_eff,
                            "operations": ops,
                            "edges": edges,
                        },
                    )
                )
    if not results:
        return None
    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(config for key, config in results if key == best_key)


if __name__ == "__main__":
    print("eval_utils self-test")
    config = {
        "operations": ["GCNConv", "GATConv", "SAGEConv", "GINConv", "Identity"],
        "effective_layers": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    }
    z = torch.rand(12 + HP_DIM_LAYERWISE)
    hp = decode_hp_by_mode(
        z_search=z,
        arch_nz=12,
        config=config,
        hp_mode="layer_cond19",
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
    )
    assert hp["hidden_dim"] in HIDDEN_DIM_OPTIONS
    assert hp["weight_decay"] in L2_OPTIONS
    assert hp["gat_heads_by_layer"] is not None
    assert hp["sage_aggr_by_layer"] is not None
    assert hp["gin_eps_by_layer"] is not None
    assert GIN_EPS_MIN <= hp["gin_eps"] <= GIN_EPS_MAX
    assert hp["gat_heads"] in GAT_HEAD_OPTIONS
    print("eval_utils OK")
