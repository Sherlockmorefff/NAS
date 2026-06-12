"""Central definitions for NAS hyperparameter search modes.

The three main modes are:

global4:
    hp = [lr, dropout, hidden_dim, l2]

hybrid_cond7:
    hp = [lr, dropout, hidden_dim, l2, gat_heads, sage_aggr, gin_eps]

layer_cond19:
    hp = [lr, dropout, hidden_dim, l2,
          gat_heads_0..4, sage_aggr_0..4, gin_eps_0..4]

All values in the search vector are normalized to [0, 1]. This module owns
the mode names, dimensions, parameter names, normalization helpers, and
architecture-conditioned masks used by data generation, joint VAE training,
BO/TPE search, and final evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


MAX_OP_NODES = 5

HIDDEN_DIM_OPTIONS = [16, 32, 64, 128, 256, 512]
L2_OPTIONS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
GAT_HEAD_OPTIONS = [1, 2, 4, 8]
SAGE_AGGR_OPTIONS = ["mean", "max", "add"]
GIN_EPS_MIN = -0.5
GIN_EPS_MAX = 1.0

HP_MODE_SPECS: dict[str, dict[str, Any]] = {
    "global4": {
        "hp_dim": 4,
        "names": ["lr", "dropout", "hidden_dim", "l2"],
    },
    "hybrid_cond7": {
        "hp_dim": 7,
        "names": [
            "lr",
            "dropout",
            "hidden_dim",
            "l2",
            "gat_heads",
            "sage_aggr",
            "gin_eps",
        ],
    },
    "layer_cond19": {
        "hp_dim": 19,
        "names": [
            "lr",
            "dropout",
            "hidden_dim",
            "l2",
            "gat_heads_0",
            "gat_heads_1",
            "gat_heads_2",
            "gat_heads_3",
            "gat_heads_4",
            "sage_aggr_0",
            "sage_aggr_1",
            "sage_aggr_2",
            "sage_aggr_3",
            "sage_aggr_4",
            "gin_eps_0",
            "gin_eps_1",
            "gin_eps_2",
            "gin_eps_3",
            "gin_eps_4",
        ],
    },
}


def validate_hp_mode(hp_mode: str) -> str:
    """Return a normalized hp_mode or raise ValueError."""

    if not isinstance(hp_mode, str):
        raise TypeError(f"hp_mode must be str, got {type(hp_mode).__name__}")
    hp_mode = hp_mode.strip()
    if hp_mode not in HP_MODE_SPECS:
        valid = ", ".join(sorted(HP_MODE_SPECS))
        raise ValueError(f"Unsupported hp_mode={hp_mode!r}. Valid modes: {valid}")
    return hp_mode


def hp_dim_from_mode(hp_mode: str) -> int:
    """Return the HP vector dimension for a mode."""

    return int(HP_MODE_SPECS[validate_hp_mode(hp_mode)]["hp_dim"])


def hp_names_from_mode(hp_mode: str) -> list[str]:
    """Return HP parameter names in z_search order."""

    return list(HP_MODE_SPECS[validate_hp_mode(hp_mode)]["names"])


def hp_slice_from_mode(hp_mode: str) -> dict[str, int | slice]:
    """Return useful index metadata for the mode's HP vector.

    Individual parameter names map to integer positions. Group aliases map to
    slices and are provided for callers that need contiguous blocks.
    """

    hp_mode = validate_hp_mode(hp_mode)
    names = hp_names_from_mode(hp_mode)
    out: dict[str, int | slice] = {name: idx for idx, name in enumerate(names)}
    out["global"] = slice(0, 4)

    if hp_mode == "hybrid_cond7":
        out["conditional"] = slice(4, 7)
    elif hp_mode == "layer_cond19":
        out["gat_heads_by_layer"] = slice(4, 4 + MAX_OP_NODES)
        out["sage_aggr_by_layer"] = slice(4 + MAX_OP_NODES, 4 + 2 * MAX_OP_NODES)
        out["gin_eps_by_layer"] = slice(4 + 2 * MAX_OP_NODES, 4 + 3 * MAX_OP_NODES)

    return out


def _finite_float(x: float, default: float = 0.0) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _clip01(x: float) -> float:
    value = _finite_float(x, default=0.0)
    return max(0.0, min(1.0, value))


def _norm_index(x: float, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    value = min(_clip01(x), 1.0 - 1e-12)
    return min(int(value * n), n - 1)


def norm_to_hidden(x: float) -> int:
    """Map normalized scalar to the hidden_dim option grid."""

    return HIDDEN_DIM_OPTIONS[_norm_index(x, len(HIDDEN_DIM_OPTIONS))]


def norm_to_l2(x: float) -> float:
    """Map normalized scalar to the L2/weight_decay option grid."""

    return L2_OPTIONS[_norm_index(x, len(L2_OPTIONS))]


def norm_to_gat_heads(x: float) -> int:
    """Map normalized scalar to the GAT heads option grid."""

    return GAT_HEAD_OPTIONS[_norm_index(x, len(GAT_HEAD_OPTIONS))]


def norm_to_sage_aggr(x: float) -> str:
    """Map normalized scalar to the SAGE aggregation option grid."""

    return SAGE_AGGR_OPTIONS[_norm_index(x, len(SAGE_AGGR_OPTIONS))]


def norm_to_gin_eps(x: float) -> float:
    """Map normalized scalar to a continuous GIN epsilon value."""

    value = _clip01(x)
    return value * (GIN_EPS_MAX - GIN_EPS_MIN) + GIN_EPS_MIN


def _ops5(ops: Sequence[str] | None) -> list[str]:
    if ops is None:
        ops_list: list[str] = []
    else:
        ops_list = [str(op) for op in list(ops)[:MAX_OP_NODES]]
    if len(ops_list) < MAX_OP_NODES:
        ops_list.extend(["Identity"] * (MAX_OP_NODES - len(ops_list)))
    return ops_list


def condition_mask_vector_from_ops(ops: list[str], hp_mode: str) -> list[float]:
    """Return the active-mask vector for HP dimensions in z_search order."""

    hp_mode = validate_hp_mode(hp_mode)
    ops5 = _ops5(ops)

    if hp_mode == "global4":
        return [1.0, 1.0, 1.0, 1.0]

    has_gat = any(op == "GATConv" for op in ops5)
    has_sage = any(op == "SAGEConv" for op in ops5)
    has_gin = any(op == "GINConv" for op in ops5)

    if hp_mode == "hybrid_cond7":
        return [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0 if has_gat else 0.0,
            1.0 if has_sage else 0.0,
            1.0 if has_gin else 0.0,
        ]

    return (
        [1.0, 1.0, 1.0, 1.0]
        + [1.0 if op == "GATConv" else 0.0 for op in ops5]
        + [1.0 if op == "SAGEConv" else 0.0 for op in ops5]
        + [1.0 if op == "GINConv" else 0.0 for op in ops5]
    )


def condition_mask_dict_from_ops(ops: list[str], hp_mode: str) -> dict[str, Any]:
    """Return a structured active-mask dictionary for decoded operations."""

    hp_mode = validate_hp_mode(hp_mode)
    ops5 = _ops5(ops)
    vector = condition_mask_vector_from_ops(ops5, hp_mode)
    names = hp_names_from_mode(hp_mode)

    gat_by_layer = [op == "GATConv" for op in ops5]
    sage_by_layer = [op == "SAGEConv" for op in ops5]
    gin_by_layer = [op == "GINConv" for op in ops5]

    return {
        "hp_mode": hp_mode,
        "hp_dim": hp_dim_from_mode(hp_mode),
        "names": names,
        "vector": vector,
        "mask": vector,
        "by_name": {name: vector[idx] for idx, name in enumerate(names)},
        "lr": True,
        "dropout": True,
        "hidden_dim": True,
        "l2": True,
        "gat_heads": any(gat_by_layer),
        "sage_aggr": any(sage_by_layer),
        "gin_eps": any(gin_by_layer),
        "gat_by_layer": gat_by_layer,
        "sage_by_layer": sage_by_layer,
        "gin_by_layer": gin_by_layer,
    }


if __name__ == "__main__":
    assert hp_dim_from_mode("global4") == 4
    assert hp_dim_from_mode("hybrid_cond7") == 7
    assert hp_dim_from_mode("layer_cond19") == 19
    assert condition_mask_vector_from_ops(["GATConv"], "global4") == [1.0, 1.0, 1.0, 1.0]
    assert condition_mask_vector_from_ops(["GATConv"], "hybrid_cond7") == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    print("hp_modes OK")
