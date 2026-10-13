"""Frozen, label-free directed operator-WL features and training distances."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from typing import Mapping

import numpy as np


FEATURE_VERSION = "directed_operator_wl_h2_A_only_v1"
OPS = {"GCNConv", "GATConv", "SAGEConv", "GINConv", "GCNII", "Identity"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def build_operator_graph(config: Mapping) -> dict:
    """Represent recorded operator edges, including input/output and Identity.

    Global HP never enters a node label. GCNII occurrence is a semantic layer
    attribute, not a mutable vertex number; it travels with a relabelled node.
    No missing connection is invented from the model's execution fallback.
    """
    operations = list(config.get("operations", []))
    if not operations or any(op not in OPS for op in operations):
        raise ValueError("BLOCKED: missing or unsupported actual architecture operations")
    raw_edges = config.get("edges")
    if not isinstance(raw_edges, (list, tuple)) or not raw_edges:
        raise ValueError("BLOCKED: missing actual architecture connections")
    n = len(operations) + 2
    edges = []
    for edge in raw_edges:
        if len(edge) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in edge):
            raise ValueError("BLOCKED: invalid recorded edge")
        u, v = edge
        if not 0 <= u < v < n:
            raise ValueError("BLOCKED: operator graph must be a recorded ordered DAG")
        edges.append([u, v])
    if len({tuple(e) for e in edges}) != len(edges):
        raise ValueError("BLOCKED: duplicate recorded operator edge")
    for name in ("gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer"):
        if config.get(name) is not None:
            raise ValueError("BLOCKED: variable per-layer attributes require a new protocol")
    nodes = [{"role": "input", "operation": "none", "activation": "none"}]
    gcnii_occurrence = 0
    for op in operations:
        node = {"role": "operator", "operation": op,
                "activation": "none" if op == "Identity" else "relu", "merge": "sum"}
        if op == "GATConv":
            node.update(heads=int(config.get("gat_heads", 1)), concat=False)
        elif op == "SAGEConv":
            node.update(aggr=config.get("sage_aggr", "mean"))
        elif op == "GINConv":
            node.update(eps=float(config.get("gin_eps", 0.0)), train_eps=False)
        elif op == "GCNII":
            gcnii_occurrence += 1
            node.update(alpha=float(config.get("gcnii_alpha", 0.1)),
                        theta=float(config.get("gcnii_theta", 0.5)),
                        layer=gcnii_occurrence, shared_weights=True)
        nodes.append(node)
    nodes.append({"role": "output", "operation": "linear", "merge": "sum"})
    return {"nodes": nodes, "edges": sorted(edges)}


def graph_hash(graph: Mapping) -> str:
    return _digest({"nodes": graph["nodes"], "edges": sorted(graph["edges"])})


def isomorphic_operator_graph(left: Mapping, right: Mapping) -> bool:
    """Exact typed directed isomorphism for these at-most-seven-node graphs."""
    if len(left["nodes"]) != len(right["nodes"]) or len(left["edges"]) != len(right["edges"]):
        return False
    groups_left, groups_right = {}, {}
    for i, node in enumerate(left["nodes"]):
        groups_left.setdefault(_canonical(node), []).append(i)
    for i, node in enumerate(right["nodes"]):
        groups_right.setdefault(_canonical(node), []).append(i)
    if {k: len(v) for k, v in groups_left.items()} != {k: len(v) for k, v in groups_right.items()}:
        return False
    keys = sorted(groups_left)
    edge_set = {tuple(e) for e in right["edges"]}
    for alternatives in itertools.product(*(itertools.permutations(groups_right[k]) for k in keys)):
        mapping = {u: v for k, nodes in zip(keys, alternatives) for u, v in zip(groups_left[k], nodes)}
        if {(mapping[u], mapping[v]) for u, v in left["edges"]} == edge_set:
            return True
    return False


def directed_wl(graphs: Mapping[str, Mapping]) -> tuple[dict[str, list[float]], dict]:
    """H=2 counts with separate depth dictionaries, normalized then augmented.

    ``graphs`` is architecture_id -> {nodes:[attribute mappings], edges:[[u,v]]}.
    The dictionary uses all and only the frozen A pool's unlabelled structures.
    Returns architecture_id -> psi and a complete reproducible feature manifest.
    """
    if not graphs:
        raise ValueError("BLOCKED: empty unlabelled operator graph pool")
    ids = sorted(graphs)
    labels, incoming, outgoing = {}, {}, {}
    for aid in ids:
        graph = graphs[aid]
        count = len(graph["nodes"])
        if count == 0:
            raise ValueError("BLOCKED: operator graph without nodes")
        labels[aid] = [_canonical(node) for node in graph["nodes"]]
        incoming[aid] = [[] for _ in range(count)]
        outgoing[aid] = [[] for _ in range(count)]
        for u, v in graph["edges"]:
            if not 0 <= u < count or not 0 <= v < count or u == v:
                raise ValueError("BLOCKED: invalid WL graph edge")
            incoming[aid][v].append(u)
            outgoing[aid][u].append(v)
    dictionaries, counts = [], {aid: [] for aid in ids}
    for h in range(3):
        dictionary = sorted({label for aid in ids for label in labels[aid]})
        index = {label: i for i, label in enumerate(dictionary)}
        dictionaries.append({"h": h, "size": len(dictionary), "labels": dictionary})
        encoded = {}
        for aid in ids:
            encoded[aid] = [index[label] for label in labels[aid]]
            histogram = Counter(encoded[aid])
            counts[aid].extend(histogram.get(i, 0) for i in range(len(dictionary)))
        if h < 2:
            labels = {
                aid: [_canonical([encoded[aid][v],
                                  sorted(encoded[aid][u] for u in incoming[aid][v]),
                                  sorted(encoded[aid][w] for w in outgoing[aid][v])])
                      for v in range(len(encoded[aid]))]
                for aid in ids
            }
    n_max = max(len(graphs[aid]["nodes"]) for aid in ids)
    e_max = max(len(graphs[aid]["edges"]) for aid in ids)
    features = {}
    collisions = {}
    for aid in ids:
        phi = np.asarray(counts[aid], dtype=np.float64)
        norm = float(np.linalg.norm(phi))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("BLOCKED: invalid WL count norm")
        features[aid] = np.r_[phi / norm, len(graphs[aid]["nodes"]) / n_max,
                              len(graphs[aid]["edges"]) / e_max if e_max else 0.0].tolist()
        collisions.setdefault(_digest(counts[aid]), []).append(aid)
    collision_groups = [group for group in collisions.values() if len(group) > 1]
    manifest = {
        "version": FEATURE_VERSION, "H": 2, "pool": "frozen_A_only_no_performance",
        "architecture_ids": ids, "architecture_count": len(ids),
        "graph_hashes": {aid: graph_hash(graphs[aid]) for aid in ids},
        "graph_hash_semantics": "ordered node-attribute/edge hash, not an isomorphism canonical hash",
        "dictionaries": dictionaries, "dictionary_sizes": [d["size"] for d in dictionaries],
        "phi_dimension": sum(d["size"] for d in dictionaries),
        "psi_dimension": len(features[ids[0]]), "n_max": n_max, "e_max": e_max,
        "normalization": "psi=[phi/L2norm(phi),n_nodes/n_max,n_edges/e_max]; no column scaling",
        "unique_count_vectors": len(collisions), "collision_groups": collision_groups,
        "collision_architecture_fraction": sum(map(len, collision_groups)) / len(ids),
        "all_structures_identical": len(collisions) == 1,
        "features_sha256": _digest(features), "performance_labels_used": False,
        "not_NAS_BOWL_exact_reproduction": True,
    }
    manifest["identity"] = _digest(manifest)
    return features, manifest


def positive_distance_median(x) -> dict:
    """Strictly positive, unordered training-pair Euclidean distances only."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Non-finite or malformed distance inputs")
    # Direct differences preserve zero for exact duplicates (no cancellation).
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    pairs = d[np.triu_indices(len(x), k=1)]
    positive = pairs[pairs > 0]
    return {"scale": float(np.median(positive)) if len(positive) else 1.0,
            "degenerate": len(positive) == 0, "positive_pairs": int(len(positive)),
            "zero_pairs": int(np.count_nonzero(pairs == 0)),
            "sample_count": len(x)}


def fit_distance_scales(x, wl=None) -> dict:
    """Take only T's already-standardized [latent12,HP4]; WL stays unscaled."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 16 or not np.isfinite(x).all():
        raise ValueError("Expected finite T-standardized latent12 + HP4")
    result = {"latent": positive_distance_median(x[:, :12]),
              "hp": positive_distance_median(x[:, 12:]),
              "raw16": positive_distance_median(x)}
    if wl is not None:
        wl = np.asarray(wl, dtype=np.float64)
        if wl.ndim != 2 or len(wl) != len(x):
            raise ValueError("WL training rows must match T")
        result["wl"] = positive_distance_median(wl)
    return result
