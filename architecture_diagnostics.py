"""Shared architecture helpers for read-only NAS diagnostics.

Node positions are intentionally preserved.  Fingerprints canonicalize only
the order in which edges are serialized; they do not perform graph-isomorphism
or node-renumbering canonicalization.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import igraph
import numpy as np
import torch

from hp_modes import MAX_OP_NODES, hp_dim_from_mode, validate_hp_mode
from nas_space import JointSpaceVAE, OP_MAPPING


DECODER_POLICIES = ("historical_modal3", "deterministic_greedy")
ARCH_NZ = 12
MAX_NODES = MAX_OP_NODES + 2
OP_NAMES = tuple(OP_MAPPING[index] for index in sorted(OP_MAPPING))


class ArchitectureArgs:
    """Minimal argument object required by :class:`JointSpaceVAE`."""

    def __init__(self, arch_nz: int = ARCH_NZ) -> None:
        self.max_n = MAX_NODES
        self.num_vertex_type = max(OP_MAPPING) + 1
        self.START_TYPE = 0
        self.END_TYPE = 1
        self.hs = 501
        self.nz = int(arch_nz)
        self.bidirectional = True


def validate_decoder_policy(decoder_policy: str) -> str:
    if decoder_policy not in DECODER_POLICIES:
        raise ValueError(
            f"unsupported decoder_policy={decoder_policy!r}; "
            f"expected one of {DECODER_POLICIES}"
        )
    return decoder_policy


def normalize_architecture_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("architecture config must be a mapping")
    operations = [str(value) for value in config.get("operations", [])]
    edges = [tuple(map(int, edge)) for edge in config.get("edges", [])]
    if not operations:
        raise ValueError("architecture config has no operations")
    if len(operations) > MAX_OP_NODES:
        raise ValueError(
            f"architecture has {len(operations)} operations; maximum is {MAX_OP_NODES}"
        )
    unknown = sorted(set(operations) - set(OP_NAMES))
    if unknown:
        raise ValueError(f"unknown architecture operations: {unknown}")
    node_count = len(operations) + 2
    for source, destination in edges:
        if not (0 <= source < destination < node_count):
            raise ValueError(
                f"edge {(source, destination)} is outside the ordered DAG with "
                f"{node_count} nodes"
            )
    return {
        "effective_layers": sum(op != "Identity" for op in operations),
        "operations": operations,
        "edges": edges,
    }


def architecture_fingerprint(
    config_or_operations: Mapping[str, Any] | Sequence[str],
    edges: Sequence[Sequence[int]] | None = None,
) -> str:
    """Return a stable position-sensitive architecture fingerprint."""

    if isinstance(config_or_operations, Mapping):
        operations = [str(value) for value in config_or_operations.get("operations", [])]
        raw_edges = config_or_operations.get("edges", [])
    else:
        operations = [str(value) for value in config_or_operations]
        raw_edges = [] if edges is None else edges
    payload = {
        "operations": operations,
        "edges": sorted([list(map(int, edge)) for edge in raw_edges]),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_to_igraph(config: Mapping[str, Any]) -> igraph.Graph:
    """Convert a recorded architecture config into the DVAE igraph format."""

    normalized = normalize_architecture_config(config)
    reverse_mapping = {name: index for index, name in OP_MAPPING.items()}
    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(normalized["operations"]) + 2)
    graph.vs["type"] = [
        0,
        *[reverse_mapping[operation] for operation in normalized["operations"]],
        1,
    ]
    graph.add_edges(normalized["edges"])
    return graph


def config_from_igraph(model: Any, graph: igraph.Graph) -> dict[str, Any]:
    operations = [
        model.op_mapping.get(graph.vs[index]["type"], "Identity")
        for index in range(1, graph.vcount() - 1)
    ]
    return normalize_architecture_config(
        {"operations": operations, "edges": graph.get_edgelist()}
    )


def encode_architecture_mu(model: Any, config: Mapping[str, Any]) -> list[float]:
    graph = config_to_igraph(config)
    with torch.no_grad():
        mu, _logvar = model.arch_vae.encode([graph])
    return torch.as_tensor(mu[0]).detach().cpu().to(torch.float32).tolist()


def cycle_distances(
    z_arch: Sequence[float], mu_arch: Sequence[float]
) -> tuple[float, float]:
    z = np.asarray(z_arch, dtype=np.float64)
    mu = np.asarray(mu_arch, dtype=np.float64)
    if z.shape != mu.shape or z.ndim != 1 or z.size == 0:
        raise ValueError(
            f"z_arch and mu_arch must be matching non-empty vectors, got {z.shape} and {mu.shape}"
        )
    distance = float(np.linalg.norm(z - mu))
    return distance, distance / math.sqrt(z.size)


def structure_feature_vector(
    config: Mapping[str, Any], condition_mask_vector: Sequence[float]
) -> tuple[np.ndarray, list[str]]:
    """Encode position-wise operations, upper-triangle edges, and HP mask."""

    normalized = normalize_architecture_config(config)
    operations = list(normalized["operations"])
    padded = operations + ["<PAD>"] * (MAX_OP_NODES - len(operations))
    values: list[float] = []
    names: list[str] = []
    for position, observed in enumerate(padded):
        for operation in OP_NAMES:
            names.append(f"op_{position}_{operation}")
            values.append(float(observed == operation))
    edge_set = set(normalized["edges"])
    for source in range(MAX_NODES):
        for destination in range(source + 1, MAX_NODES):
            names.append(f"edge_{source}_{destination}")
            values.append(float((source, destination) in edge_set))
    for index, value in enumerate(condition_mask_vector):
        names.append(f"condition_mask_{index}")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64), names


def load_joint_vae(
    checkpoint: str | Path,
    hp_mode: str,
    *,
    arch_nz: int = ARCH_NZ,
    device: torch.device | str = "cpu",
) -> JointSpaceVAE:
    hp_mode = validate_hp_mode(hp_mode)
    hp_dim = hp_dim_from_mode(hp_mode)
    model = JointSpaceVAE(
        ArchitectureArgs(arch_nz),
        hp_mode=hp_mode,
        hp_latent_dim=hp_dim,
        hp_input_dim=hp_dim,
    ).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
