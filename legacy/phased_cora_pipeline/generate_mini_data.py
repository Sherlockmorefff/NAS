"""Generate mini NAS datasets for hp_mode-driven joint VAE training.

Search space v3:
    - max_n = 7
    - 5 operator nodes between START and END
    - operator types include GCNConv, GATConv, SAGEConv, GINConv, GCNII,
      and Identity on the middle optional nodes.

Main HP modes:
    global4:
        [lr, dropout, hidden_dim, l2]
    hybrid_cond7:
        [lr, dropout, hidden_dim, l2, gat_heads, sage_aggr, gin_eps]
    layer_cond19:
        [lr, dropout, hidden_dim, l2,
         gat_heads_0..4, sage_aggr_0..4, gin_eps_0..4]
"""

from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from hp_modes import (
    HP_MODE_SPECS,
    MAX_OP_NODES,
    condition_mask_vector_from_ops,
    hp_dim_from_mode,
    hp_names_from_mode,
    validate_hp_mode,
)


START_TYPE = 0
END_TYPE = 1

REAL_OP_TYPES = [2, 3, 4, 5, 7]
ALL_OP_TYPES = [2, 3, 4, 5, 6, 7]

TYPE_TO_OP = {
    2: "GCNConv",
    3: "GATConv",
    4: "SAGEConv",
    5: "GINConv",
    6: "Identity",
    7: "GCNII",
}

DEFAULT_OUTPUT_BY_MODE = {
    "global4": "data/mini_gnn_dataset_global4.pkl",
    "hybrid_cond7": "data/mini_gnn_dataset_hybrid_cond7.pkl",
    "layer_cond19": "data/mini_gnn_dataset_layer_cond19.pkl",
}


def ops_from_types(types: list[int]) -> list[str]:
    op_types = list(types)[1:1 + MAX_OP_NODES]
    if len(op_types) < MAX_OP_NODES:
        op_types.extend([6] * (MAX_OP_NODES - len(op_types)))
    return [TYPE_TO_OP.get(int(t), "Identity") for t in op_types]


def sample_hp_by_mode(types: list[int], hp_mode: str) -> list[float]:
    hp_mode = validate_hp_mode(hp_mode)
    ops = ops_from_types(types)
    mask = condition_mask_vector_from_ops(ops, hp_mode)
    return [random.random() if active > 0.0 else 0.0 for active in mask]


def assert_valid_v3(types: list[int], adj: np.ndarray) -> None:
    for i in range(6):
        assert adj[i, i + 1] == 1, f"main-chain edge {i}->{i + 1} is missing"

    for j in range(2, 7):
        assert adj[0, j] == 0, f"illegal edge 0->{j}"

    for i in range(5):
        assert adj[i, 6] == 0, f"illegal edge {i}->6"

    for i in range(7):
        for j in range(i + 1):
            if i != j:
                assert adj[i, j] == 0, f"backward edge {i}->{j}"

    assert types[1] != 6, "Node1 cannot be Identity"
    assert types[5] != 6, "Node5 cannot be Identity"
    assert types[0] == START_TYPE
    assert types[6] == END_TYPE


def generate_one_graph(skip_prob: float) -> tuple[list[int], np.ndarray]:
    types = [
        START_TYPE,
        random.choice(REAL_OP_TYPES),
        random.choice(ALL_OP_TYPES),
        random.choice(ALL_OP_TYPES),
        random.choice(ALL_OP_TYPES),
        random.choice(REAL_OP_TYPES),
        END_TYPE,
    ]

    adj = np.zeros((7, 7), dtype=np.int8)
    for i in range(6):
        adj[i, i + 1] = 1

    for i in range(1, 5):
        for j in range(i + 2, 6):
            if random.random() < skip_prob:
                adj[i, j] = 1

    assert_valid_v3(types, adj)
    return types, adj


def generate_mini_dataset(
    output_file: str,
    num_samples: int = 3000,
    skip_prob: float = 0.35,
    hp_mode: str = "global4",
    hp_dim: int | None = None,
) -> str:
    hp_mode = validate_hp_mode(hp_mode)
    expected_hp_dim = hp_dim_from_mode(hp_mode)
    if hp_dim is not None and hp_dim != expected_hp_dim:
        raise ValueError(
            f"--hp_dim={hp_dim} does not match hp_mode={hp_mode} "
            f"(expected {expected_hp_dim})"
        )

    dataset = []
    print(
        f"Generating {num_samples} v3 DAG samples "
        f"(max_n=7, nvt=8, hp_mode={hp_mode}, hp_dim={expected_hp_dim})"
    )
    print(f"  HP names: {hp_names_from_mode(hp_mode)}")
    print("  Mask rule:")
    if hp_mode == "global4":
        print("    [lr, dropout, hidden_dim, l2] are always active")
    elif hp_mode == "hybrid_cond7":
        print("    global dims always active; gat_heads/sage_aggr/gin_eps are type-conditional")
    else:
        print("    global dims always active; GAT/SAGE/GIN params are layer-conditional")

    for _ in tqdm(range(num_samples)):
        types, adj = generate_one_graph(skip_prob)
        hp = sample_hp_by_mode(types, hp_mode)
        dataset.append((types, adj, hp))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nSaved dataset: {output_file} ({len(dataset)} samples)")
    print(f"  hp_mode={hp_mode}")
    print(f"  hp_dim={expected_hp_dim}")
    print(f"  HP_MODE_SPECS[{hp_mode!r}]={HP_MODE_SPECS[hp_mode]}")

    print("\n[Self-check] First 3 samples:")
    for i, (types, adj, hp) in enumerate(dataset[:3]):
        edges = [(r, c) for r in range(7) for c in range(7) if int(adj[r, c]) == 1]
        ops = ops_from_types(types)
        mask = condition_mask_vector_from_ops(ops, hp_mode)
        print(f"  [{i}] types={types}")
        print(f"      ops={ops}")
        print(f"      edges={edges}")
        print(f"      hp={np.round(hp, 3).tolist()}")
        print(f"      mask={mask}")

    n_skip = sum(
        1
        for _, adj, _ in dataset
        if any(adj[i, j] for i in range(1, 5) for j in range(i + 2, 6))
    )
    n_gcnii = sum(1 for types, _, _ in dataset if 7 in types)
    print(f"\n  skip-edge sample ratio: {n_skip}/{len(dataset)} = {n_skip / len(dataset):.1%}")
    print(f"  GCNII sample ratio: {n_gcnii}/{len(dataset)} = {n_gcnii / len(dataset):.1%}")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate_mini_data hp_mode")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--skip_prob", type=float, default=0.35)
    parser.add_argument(
        "--hp_mode",
        type=str,
        default="global4",
        choices=["global4", "hybrid_cond7", "layer_cond19"],
    )
    parser.add_argument(
        "--hp_dim",
        type=int,
        default=None,
        help="Deprecated compatibility guard. If provided, must match hp_mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    hp_mode = validate_hp_mode(args.hp_mode)
    output = args.output or DEFAULT_OUTPUT_BY_MODE[hp_mode]
    generate_mini_dataset(
        output_file=output,
        num_samples=args.num_samples,
        skip_prob=args.skip_prob,
        hp_mode=hp_mode,
        hp_dim=args.hp_dim,
    )
