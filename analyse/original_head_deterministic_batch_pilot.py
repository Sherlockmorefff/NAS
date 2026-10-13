#!/usr/bin/env python3
"""Freeze a model-blind deterministic batch NAS+HPO pilot.

This module only generates candidates, evaluates frozen Original Head members,
partitions candidates with an explicit structure/HP distance, and freezes three
selection policies.  It has no GNN label, test-metric, BO, GP, acquisition, or
model-training path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import architecture_fingerprint, load_joint_vae
from analyse.performance_aware_vae_strict_regression import config_from_igraph
from analyse.structure_native_surrogate_pilot import realize_global4_hp
from analyse.original_head_uncertainty_selective_risk_audit import head_member_predictions


FORMAT_VERSION = "original_head_deterministic_batch_pilot_global4_v1"
OUTPUT_RELATIVE = "results/posthoc/original-head-deterministic-batch-pilot-global4-v1"
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
MASTER_SEED = 20260824
POOL_COUNT = 8
ARCHITECTURES_PER_POOL = 64
HP_REGIMES = {
    "hp_low": (0.05, 0.20),
    "hp_mid_low": (0.30, 0.45),
    "hp_mid_high": (0.55, 0.70),
    "hp_high": (0.80, 0.95),
}
INITIAL_DRAWS = 4096
MAX_DRAWS = 65536
DECODE_BATCH = 256
K_MEDOIDS_K = 4
K_MEDOIDS_MAX_ITERATIONS = 100
TRAINING_SEEDS = (0, 1, 2)
MAX_SYSTEM_RETRIES = 2
OPERATION_TYPES = {"GCNConv", "GATConv", "SAGEConv", "GINConv", "GCNII", "Identity"}
HIDDEN_DIM_OPTIONS = (16, 32, 64, 128, 256, 512)
FORBIDDEN_TEST_TOKENS = ("test_acc", "test_accuracy", "test_metric", "test_loss")
FORBIDDEN_DISTANCE_PREFIXES = (
    "mu", "latent", "prediction", "head_prediction", "h_prediction", "refit",
    "label", "validation", "val_acc", "test",
)
CLASSIFICATIONS = (
    "HEAD_DIVERSITY_PROMOTED",
    "HEAD_ONLY_PROMOTED",
    "RANDOM_DIVERSITY_RETAINED",
    "PILOT_INCONCLUSIVE",
    "INSUFFICIENT_FRESH_CANDIDATES",
    "EXECUTION_OR_PROTOCOL_FAILURE",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_seed(*parts: Any) -> int:
    digest = stable_hash([MASTER_SEED, *parts])
    return int(digest[:8], 16) & 0x7FFFFFFF


def hp_fingerprint(values: Sequence[float]) -> str:
    return stable_hash([float(value) for value in values])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(row.get(key), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple)) else row.get(key)
                    for key in fields
                })
    temporary.replace(path)


def run_text(args: Sequence[str]) -> str:
    return subprocess.run(
        list(args), cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def contains_test_field(value: Any, path: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lower = str(key).lower()
            if any(token in lower for token in FORBIDDEN_TEST_TOKENS) or lower == "test":
                hits.append(f"{path}.{key}")
            hits.extend(contains_test_field(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            hits.extend(contains_test_field(nested, f"{path}[{index}]") )
    return hits


def validate_static_config(config: Mapping[str, Any]) -> tuple[bool, str | None]:
    operations = list(config.get("operations", []))
    edges = [tuple(map(int, edge)) for edge in config.get("edges", [])]
    if not 1 <= len(operations) <= 5:
        return False, "operation_count"
    if not set(operations) <= OPERATION_TYPES:
        return False, "operation_type"
    node_count = len(operations) + 2
    if not edges or len(edges) != len(set(edges)):
        return False, "edges_empty_or_duplicate"
    if any(left < 0 or right >= node_count or left >= right for left, right in edges):
        return False, "edge_range_or_dag_order"
    if not any(left == 0 for left, _ in edges):
        return False, "missing_input_edge"
    if not any(right == node_count - 1 for _, right in edges):
        return False, "missing_output_edge"
    return True, None


def _fingerprints_from_text(path: Path) -> set[str]:
    """Extract only fingerprint strings; never deserialize label/test values."""
    text = path.read_text(encoding="utf-8", errors="strict")
    patterns = (
        r'"architecture_fingerprint"\s*:\s*"([0-9a-f]{64})"',
        r'"architecture_group_id"\s*:\s*"([0-9a-f]{64})"',
        r'"(?:train|validation)_architecture_fingerprints"\s*:\s*\[(.*?)\]',
    )
    found = set(re.findall(patterns[0], text)) | set(re.findall(patterns[1], text))
    for block in re.findall(patterns[2], text, flags=re.DOTALL):
        found.update(re.findall(r'"([0-9a-f]{64})"', block))
    return found


def _config_fingerprint_pairs_from_text(path: Path) -> list[tuple[str, str]]:
    """Extract config/fingerprint identity pairs without decoding unrelated fields."""
    text = path.read_text(encoding="utf-8", errors="strict")
    pairs = []
    forward = re.compile(
        r'"architecture_fingerprint"\s*:\s*"([0-9a-f]{64})"'
        r'(?:(?!\n\s*}\s*,?).){0,20000}?'
        r'"config_id"\s*:\s*"([^"]+)"',
        flags=re.DOTALL,
    )
    reverse = re.compile(
        r'"config_id"\s*:\s*"([^"]+)"'
        r'(?:(?!\n\s*}\s*,?).){0,20000}?'
        r'"architecture_fingerprint"\s*:\s*"([0-9a-f]{64})"',
        flags=re.DOTALL,
    )
    pairs.extend((config_id, fingerprint) for fingerprint, config_id in forward.findall(text))
    pairs.extend((config_id, fingerprint) for config_id, fingerprint in reverse.findall(text))
    return sorted(set(pairs))


def _reference_config_ids_from_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="strict")
    ids = []
    for block in re.findall(r'"config_ids"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL):
        ids.extend(re.findall(r'"([^"]+)"', block))
    return sorted(set(ids))


def exposure_sources() -> list[Path]:
    required = [
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/phase_a/exposure_registry.json",
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/phase_a/frozen_selection_48.json",
        REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1/unified_training_data_no_test.json",
        REPO_ROOT / "results/posthoc/performance-aware-vae-strict-regression-global4-v2/fixed_split_manifest.json",
        REPO_ROOT / "results/posthoc/h-targeted-reliable-pair-validation-global4-v1/phase_b/targeted_selection.json",
        REPO_ROOT / "results/posthoc/h-targeted-reliable-pair-validation-global4-v1/phase_b/adaptive_selection.json",
        REPO_ROOT / "results/posthoc/structure-native-surrogate-pilot-global4-v1/development/development_80_manifest.json",
        REPO_ROOT / "results/posthoc/structure-native-surrogate-pilot-global4-v1/phase_4/frozen_selection_24.json",
        REPO_ROOT / "results/posthoc/structure-native-surrogate-pilot-global4-v1/phase_5/adaptive_selection.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing exposure sources: {missing}")
    posthoc = REPO_ROOT / "results/posthoc"
    patterns = (
        "**/candidate_pool.json", "**/unbiased_candidate_pool.json",
        "**/frozen_selection*.json", "**/targeted_selection.json",
        "**/adaptive_selection.json", "**/development_80_manifest.json",
        "**/unified_training_data_no_test.json",
    )
    discovered = set(required)
    target = (REPO_ROOT / OUTPUT_RELATIVE).resolve()
    for pattern in patterns:
        for path in posthoc.glob(pattern):
            if path.is_file() and target not in path.resolve().parents:
                discovered.add(path)
    return sorted(discovered)


def build_exposure_registry() -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
    sources = exposure_sources()
    direct_by_path = {path: _fingerprints_from_text(path) for path in sources}
    pairs_by_path = {path: _config_fingerprint_pairs_from_text(path) for path in sources}
    references_by_path = {path: _reference_config_ids_from_text(path) for path in sources}
    config_lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path, pairs in pairs_by_path.items():
        for config_id, fingerprint in pairs:
            config_lookup[config_id].append({
                "architecture_fingerprint": fingerprint,
                "source_path": str(path.resolve()),
                "source_sha256": sha256_file(path),
            })
    registry: set[str] = set().union(*direct_by_path.values())
    inventory = []
    unresolved_references = []
    ambiguous_references = []
    for path in sources:
        resolved = []
        for config_id in references_by_path[path]:
            matches = config_lookup.get(config_id, [])
            fingerprints = sorted({row["architecture_fingerprint"] for row in matches})
            if not matches:
                unresolved_references.append({"manifest": str(path.resolve()), "config_id": config_id})
                continue
            if len(fingerprints) != 1:
                ambiguous_references.append({
                    "manifest": str(path.resolve()), "config_id": config_id,
                    "fingerprints": fingerprints, "matches": matches,
                })
                continue
            fingerprint = fingerprints[0]
            registry.add(fingerprint)
            resolved.append({
                "config_id": config_id,
                "architecture_fingerprint": fingerprint,
                "resolved_by": sorted({row["source_path"] for row in matches}),
                "already_covered_by_direct_fingerprint_source": any(
                    fingerprint in values for values in direct_by_path.values()
                ),
            })
        inventory.append({
            **file_identity(path),
            "direct_fingerprint_count": len(direct_by_path[path]),
            "reference_config_id_count": len(references_by_path[path]),
            "resolved_reference_count": len(resolved),
            "resolved_references": resolved,
        })
    if not registry:
        raise RuntimeError("exposure registry is empty")
    rows = [{"architecture_fingerprint": fingerprint} for fingerprint in sorted(registry)]
    audit = {
        "passed": True,
        "union_count": len(registry),
        "source_count": len(inventory),
        "reference_resolution_passed": not unresolved_references and not ambiguous_references,
        "unresolved_references": unresolved_references,
        "ambiguous_references": ambiguous_references,
        "fingerprint_only_text_extraction": True,
        "label_values_deserialized": False,
        "test_values_deserialized": False,
        "all_hp_excluded_by_architecture_fingerprint": True,
    }
    if unresolved_references or ambiguous_references:
        raise RuntimeError(
            "exposure config_id resolution failed: "
            f"unresolved={unresolved_references[:3]}, ambiguous={ambiguous_references[:3]}"
        )
    return registry, rows, {"inventory": inventory, "audit": audit}


def _decode_architectures(
    model: Any, pool_id: int, excluded: set[str], target: int = ARCHITECTURES_PER_POOL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = derived_seed("candidate_pool", pool_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unique: dict[str, dict[str, Any]] = {}
    invalid = Counter()
    exposure_rejects = 0
    duplicate_rejects = 0
    draw_target = INITIAL_DRAWS
    processed = 0
    schedule = []
    while True:
        while processed < draw_target:
            size = min(DECODE_BATCH, draw_target - processed)
            z = torch.randn(size, 12, generator=generator)
            with torch.no_grad():
                graphs = model.arch_vae.decode(z, stochastic=False)
            for offset, graph in enumerate(graphs):
                try:
                    config = config_from_igraph(model, graph)
                    fingerprint = architecture_fingerprint(config)
                except (TypeError, ValueError, KeyError, IndexError) as error:
                    invalid[type(error).__name__] += 1
                    continue
                valid, reason = validate_static_config(config)
                if not valid:
                    invalid[str(reason)] += 1
                    continue
                if fingerprint in excluded:
                    exposure_rejects += 1
                    continue
                if fingerprint in unique:
                    duplicate_rejects += 1
                    continue
                unique[fingerprint] = {
                    "pool_id": pool_id,
                    "architecture_fingerprint": fingerprint,
                    "operations": list(config["operations"]),
                    "edges": [list(map(int, edge)) for edge in config["edges"]],
                    "effective_layers": int(config["effective_layers"]),
                    "generation_index": processed + offset,
                    "architecture_source": "frozen_vae_standard_normal_deterministic_greedy",
                }
            processed += size
        schedule.append({"cumulative_draws": processed, "fresh_unique_count": len(unique)})
        if len(unique) >= target or draw_target >= MAX_DRAWS:
            break
        draw_target = min(MAX_DRAWS, draw_target * 2)
    selected = sorted(unique.values(), key=lambda row: (int(row["generation_index"]), row["architecture_fingerprint"]))[:target]
    audit = {
        "pool_id": pool_id,
        "pool_seed": seed,
        "initial_draws": INITIAL_DRAWS,
        "maximum_draws": MAX_DRAWS,
        "draw_schedule": schedule,
        "draws_used": processed,
        "valid_unique_fresh_before_truncation": len(unique),
        "selected_architecture_count": len(selected),
        "exposure_or_prior_pool_reject_count": exposure_rejects,
        "within_pool_duplicate_reject_count": duplicate_rejects,
        "invalid_counts": dict(sorted(invalid.items())),
        "legal_rate": (processed - sum(invalid.values())) / processed,
        "fresh_unique_rate": len(unique) / processed,
        "passed": len(selected) == target,
    }
    return selected, audit


def generate_candidate_pools(exposure: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = load_joint_vae(BASE_CHECKPOINT, "global4", device="cpu")
    model.arch_vae.eval()
    excluded = set(exposure)
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pool_id in range(POOL_COUNT):
        architectures, audit = _decode_architectures(model, pool_id, excluded)
        audits.append(audit)
        if len(architectures) != ARCHITECTURES_PER_POOL:
            break
        hp_rng = np.random.default_rng(derived_seed("hp_regimes", pool_id))
        for architecture in architectures:
            excluded.add(str(architecture["architecture_fingerprint"]))
            for regime, (low, high) in HP_REGIMES.items():
                raw_hp = hp_rng.uniform(low, high, size=4).astype(np.float64).tolist()
                row = {
                    **architecture,
                    "hp_regime": regime,
                    "raw_hp_vector": raw_hp,
                    "hp_fingerprint": hp_fingerprint(raw_hp),
                }
                row.update(realize_global4_hp(row))
                row["config_id"] = (
                    f"pool{pool_id}_{row['architecture_fingerprint'][:12]}_"
                    f"{regime}_{row['hp_fingerprint'][:8]}"
                )
                candidates.append(row)
    return candidates, audits


def _assert_distance_schema(row: Mapping[str, Any]) -> None:
    for key in row:
        lower = str(key).lower()
        if lower.startswith(FORBIDDEN_DISTANCE_PREFIXES) or "_prediction" in lower:
            raise ValueError(f"forbidden distance field: {key}")


def explicit_feature_blocks(row: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    _assert_distance_schema(row)
    operations = tuple(list(row["operations"])[:5] + ["<PAD>"] * (5 - len(row["operations"])))
    edge_set = {tuple(map(int, edge)) for edge in row["edges"]}
    edge_universe = [(left, right) for left in range(7) for right in range(left + 1, 7)]
    edge_bits = np.asarray([float(edge in edge_set) for edge in edge_universe], dtype=np.float64)
    hidden = int(row["hidden_dim"])
    if hidden not in HIDDEN_DIM_OPTIONS:
        raise ValueError(f"hidden dimension outside fixed search space: {hidden}")
    hp = np.asarray([
        (math.log10(float(row["lr"])) + 4.0) / 2.5,
        (float(row["dropout"]) - 0.1) / 0.5,
        HIDDEN_DIM_OPTIONS.index(hidden) / (len(HIDDEN_DIM_OPTIONS) - 1),
        (math.log10(float(row["l2"])) + 5.0) / 2.0,
    ], dtype=np.float64)
    if not np.all(np.isfinite(hp)) or np.any(hp < -1e-8) or np.any(hp > 1.0 + 1e-8):
        raise ValueError("HP outside fixed global4 bounds")
    return operations, edge_bits, np.clip(hp, 0.0, 1.0)


def mixed_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    left_ops, left_edges, left_hp = explicit_feature_blocks(left)
    right_ops, right_edges, right_hp = explicit_feature_blocks(right)
    blocks = {
        "operation_hamming": float(np.mean([a != b for a, b in zip(left_ops, right_ops)])),
        "edge_hamming": float(np.mean(left_edges != right_edges)),
        "hp_gower_l1": float(np.mean(np.abs(left_hp - right_hp))),
    }
    return float(np.mean(list(blocks.values()))), blocks


def distance_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    size = len(rows)
    matrix = np.zeros((size, size), dtype=np.float64)
    summaries = []
    for left in range(size):
        for right in range(left + 1, size):
            value, blocks = mixed_distance(rows[left], rows[right])
            matrix[left, right] = matrix[right, left] = value
            summaries.append({"left": left, "right": right, "distance": value, **blocks})
    return matrix, summaries


def deterministic_kmedoids(
    distances: np.ndarray, config_ids: Sequence[str], k: int, seed: int,
    max_iterations: int = K_MEDOIDS_MAX_ITERATIONS,
) -> tuple[np.ndarray, list[int], int]:
    distances = np.asarray(distances, dtype=np.float64)
    size = len(config_ids)
    if distances.shape != (size, size) or size < k or k < 1:
        raise ValueError("invalid k-medoids input")
    if not np.allclose(distances, distances.T) or not np.allclose(np.diag(distances), 0.0):
        raise ValueError("distance matrix must be symmetric with zero diagonal")
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(0, size))]
    while len(medoids) < k:
        minimum = np.min(distances[:, medoids], axis=1)
        minimum[medoids] = -1.0
        best_value = float(np.max(minimum))
        choices = [index for index, value in enumerate(minimum) if abs(value - best_value) <= 1e-12]
        medoids.append(min(choices, key=lambda index: config_ids[index]))
    labels = np.full(size, -1, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        labels.fill(-1)
        for cluster, medoid in enumerate(medoids):
            labels[medoid] = cluster
        for index in range(size):
            if labels[index] >= 0:
                continue
            nearest = min(
                range(k), key=lambda cluster: (distances[index, medoids[cluster]], config_ids[medoids[cluster]])
            )
            labels[index] = nearest
        if set(labels.tolist()) != set(range(k)):
            raise RuntimeError("k-medoids produced an empty cluster")
        updated = []
        for cluster in range(k):
            members = np.where(labels == cluster)[0].tolist()
            costs = [(float(np.sum(distances[index, members])), config_ids[index], index) for index in members]
            updated.append(min(costs)[2])
        if updated == medoids:
            break
        medoids = updated
    else:
        iteration = max_iterations
    order = sorted(range(k), key=lambda cluster: config_ids[medoids[cluster]])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int64)
    medoids = [medoids[old] for old in order]
    return labels, medoids, iteration


def silhouette_score(distances: np.ndarray, labels: Sequence[int]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    values = []
    for index, own in enumerate(labels):
        same = np.where(labels == own)[0]
        same = same[same != index]
        a = float(np.mean(distances[index, same])) if len(same) else 0.0
        other = [
            float(np.mean(distances[index, np.where(labels == cluster)[0]]))
            for cluster in sorted(set(labels.tolist())) if cluster != own
        ]
        b = min(other)
        values.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(values))


def partition_pool(rows: Sequence[Mapping[str, Any]], pool_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix, pairs = distance_matrix(rows)
    config_ids = [str(row["config_id"]) for row in rows]
    labels, medoids, iterations = deterministic_kmedoids(
        matrix, config_ids, K_MEDOIDS_K, derived_seed("k_medoids", pool_id),
    )
    assignments = [
        {
            "pool_id": pool_id,
            "config_id": config_ids[index],
            "cluster_id": int(labels[index]),
            "is_medoid": index in medoids,
            "medoid_config_id": config_ids[medoids[int(labels[index])]],
        }
        for index in range(len(rows))
    ]
    audit = {
        "pool_id": pool_id,
        "distance_definition": {
            "operation_block": "mean positional categorical Hamming over five padded operation positions",
            "edge_block": "mean binary Hamming over the fixed 7-node upper-triangular edge universe",
            "hp_block": "mean Gower/L1 after fixed-bound transforms: log10(lr), dropout, hidden option index, log10(weight_decay)",
            "total": "equal arithmetic mean of operation, edge, and HP blocks",
            "uses_mu_or_latent_distance": False,
            "uses_prediction_or_label": False,
        },
        "seed": derived_seed("k_medoids", pool_id),
        "iterations": iterations,
        "medoid_config_ids": [config_ids[index] for index in medoids],
        "cluster_sizes": dict(Counter(map(int, labels))),
        "silhouette": silhouette_score(matrix, labels),
        "pair_count": len(pairs),
        "distance_min_nonzero": float(np.min(matrix[matrix > 0])) if np.any(matrix > 0) else 0.0,
        "distance_mean": float(np.mean(matrix[np.triu_indices(len(rows), 1)])),
        "passed": len(set(labels.tolist())) == K_MEDOIDS_K and all(Counter(labels).values()),
    }
    return assignments, audit


def _head_rank(ids: Sequence[str], means: Mapping[str, float]) -> list[str]:
    return sorted(ids, key=lambda config_id: (-float(means[config_id]), config_id))


def head_partition_selection(
    assignments: Sequence[Mapping[str, Any]], means: Mapping[str, float],
) -> list[str]:
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for row in assignments:
        by_cluster[int(row["cluster_id"])].append(str(row["config_id"]))
    selected = [_head_rank(ids, means)[0] for _, ids in sorted(by_cluster.items())]
    return _head_rank(selected, means)


def head_only_selection(config_ids: Sequence[str], means: Mapping[str, float]) -> list[str]:
    return _head_rank(config_ids, means)[:4]


def random_partition_selection(rows: Sequence[Mapping[str, Any]], seed: int) -> list[str]:
    allowed = {"pool_id", "config_id", "architecture_fingerprint", "hp_fingerprint", "cluster_id"}
    if any(set(row) - allowed for row in rows):
        raise ValueError("Random selector received fields outside its label/Head-blind schema")
    rng = np.random.default_rng(seed)
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(str(row["config_id"]))
    selected = []
    for cluster, ids in sorted(by_cluster.items()):
        choices = sorted(ids)
        selected.append(choices[int(rng.integers(0, len(choices)))])
    return [selected[index] for index in rng.permutation(len(selected))]


def strategy_selections(
    candidates: Sequence[Mapping[str, Any]], assignments: Sequence[Mapping[str, Any]],
    head_means: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment_by_pool: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_pool: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_id = {str(row["config_id"]): dict(row) for row in candidates}
    for row in assignments:
        assignment_by_pool[int(row["pool_id"])].append(dict(row))
    for row in candidates:
        candidates_by_pool[int(row["pool_id"])].append(dict(row))
    output = []
    overlap_audits = []
    for pool_id in range(POOL_COUNT):
        pool_assignments = assignment_by_pool[pool_id]
        ids = [str(row["config_id"]) for row in candidates_by_pool[pool_id]]
        policies = {
            "HEAD_PARTITION_DIVERSITY": head_partition_selection(pool_assignments, head_means),
            "HEAD_ONLY": head_only_selection(ids, head_means),
            "RANDOM_PARTITION_DIVERSITY": random_partition_selection([
                {
                    "pool_id": pool_id,
                    "config_id": row["config_id"],
                    "architecture_fingerprint": candidate_by_id[str(row["config_id"])]["architecture_fingerprint"],
                    "hp_fingerprint": candidate_by_id[str(row["config_id"])]["hp_fingerprint"],
                    "cluster_id": row["cluster_id"],
                }
                for row in pool_assignments
            ], derived_seed("random_partition", pool_id)),
        }
        for policy, selected in policies.items():
            if len(selected) != 4 or len(set(selected)) != 4:
                raise RuntimeError(f"{policy} did not select four unique configurations in pool {pool_id}")
            for order, config_id in enumerate(selected, start=1):
                candidate = candidate_by_id[config_id]
                output.append({
                    "pool_id": pool_id,
                    "strategy": policy,
                    "budget_order": order,
                    "config_id": config_id,
                    "architecture_fingerprint": candidate["architecture_fingerprint"],
                    "hp_fingerprint": candidate["hp_fingerprint"],
                    "cluster_id": next(int(row["cluster_id"]) for row in pool_assignments if row["config_id"] == config_id),
                    "head_prediction_mean": float(head_means[config_id]) if policy != "RANDOM_PARTITION_DIVERSITY" else None,
                })
        sets = {policy: set(selected) for policy, selected in policies.items()}
        overlap_audits.append({
            "pool_id": pool_id,
            "A_B_overlap": len(sets["HEAD_PARTITION_DIVERSITY"] & sets["HEAD_ONLY"]),
            "A_C_overlap": len(sets["HEAD_PARTITION_DIVERSITY"] & sets["RANDOM_PARTITION_DIVERSITY"]),
            "B_C_overlap": len(sets["HEAD_ONLY"] & sets["RANDOM_PARTITION_DIVERSITY"]),
            "union_count": len(set().union(*sets.values())),
        })
    audit = {
        "passed": len(output) == POOL_COUNT * 3 * 4,
        "strategy_counts": dict(Counter(row["strategy"] for row in output)),
        "pool_strategy_counts": {
            f"{pool}:{strategy}": count for (pool, strategy), count in sorted(
                Counter((row["pool_id"], row["strategy"]) for row in output).items()
            )
        },
        "overlaps": overlap_audits,
        "random_selector_received_head_fields": False,
        "random_selector_received_label_fields": False,
        "head_std_used_by_any_selector": False,
    }
    return output, audit


def make_union_tasks(
    candidates: Sequence[Mapping[str, Any]], selections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["config_id"]): dict(row) for row in candidates}
    selected_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        selected_by[str(row["config_id"])].append({
            "strategy": row["strategy"], "budget_order": int(row["budget_order"]),
        })
    union = []
    tasks = []
    for config_id in sorted(selected_by):
        row = dict(by_id[config_id])
        row["selected_by"] = sorted(selected_by[config_id], key=lambda item: (item["strategy"], item["budget_order"]))
        row["assigned_gpu"] = int(stable_hash([row["pool_id"], config_id])[:8], 16) % 2
        union.append(row)
        for seed in TRAINING_SEEDS:
            tasks.append({
                **row,
                "seed": seed,
                "task_key": f"{row['pool_id']}::{config_id}::{row['architecture_fingerprint']}::{row['hp_fingerprint']}::{seed}",
            })
    return union, tasks


def _classification_files() -> list[Path]:
    roots = (
        "structure-native-surrogate-pilot-global4-v1",
        "direct-head-chronological-offline-replay-global4-v2",
        "h-vs-head-model-blind-prospective-global4-v1",
        "original-head-uncertainty-selective-risk-audit-global4-v1",
    )
    paths = []
    for name in roots:
        root = REPO_ROOT / "results/posthoc" / name
        for candidate in (root / "final_classification.json", root / "final_decision.json"):
            if candidate.is_file():
                paths.append(candidate)
                break
    return paths


def protected_paths() -> list[Path]:
    return [
        REPO_ROOT / "bo_phase4.py",
        REPO_ROOT / "eval_utils.py",
        BASE_CHECKPOINT,
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/gnn_run_manifest.json",
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/phase_b/seed_results_merged.json",
        REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1/unified_training_data_no_test.json",
        *_classification_files(),
    ]


def source_identity() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout
    untracked = run_text(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    status = run_text(["git", "status", "--short"]).splitlines()
    gpu_query = run_text([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader",
    ])
    return {
        "created_at": utc_now(),
        "branch": run_text(["git", "branch", "--show-current"]),
        "head": run_text(["git", "rev-parse", "HEAD"]),
        "status_short": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_size_bytes": len(diff),
        "untracked_files": untracked,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_query": gpu_query.splitlines(),
        "working_tree_was_not_cleaned": True,
        "experiment_identity": "HEAD + tracked_diff_sha256 + experiment_added_file_hashes",
    }


def write_failure(output: Path, classification: str, evidence: Mapping[str, Any]) -> None:
    if classification not in CLASSIFICATIONS:
        raise ValueError(classification)
    dump_json(output / "final_classification.json", {
        "format_version": FORMAT_VERSION,
        "classification": classification,
        "created_at": utc_now(),
        "evidence": dict(evidence),
        "training_started": False,
    })


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "source_identity.json", source_identity())
    before = {str(path.resolve()): file_identity(path) for path in protected_paths()}
    dump_json(output / "protected_hashes_before.json", before)

    exposure, exposure_rows, exposure_metadata = build_exposure_registry()
    dump_json(output / "exposure_registry.json", exposure_rows)
    dump_json(output / "exposure_source_inventory.json", exposure_metadata)

    strict_root = REPO_ROOT / STRICT_RELATIVE
    head_paths = sorted((strict_root / "checkpoints/full/group_repeated").glob("group_*/seed_*/frozen_head/checkpoint.pth"))
    if len(head_paths) != 150:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "expected 150 Original Head members", "found": len(head_paths)})
        raise RuntimeError("expected exactly 150 Original Head members")
    input_manifest = {
        "format_version": FORMAT_VERSION,
        "base_vae": file_identity(BASE_CHECKPOINT),
        "strict_task_plan": file_identity(strict_root / "task_plan.json"),
        "strict_split_manifest": file_identity(strict_root / "fixed_split_manifest.json"),
        "head_members": [file_identity(path) for path in head_paths],
        "exposure_sources": exposure_metadata["inventory"],
        "prior_classifications": [file_identity(path) for path in _classification_files()],
        "new48_protocol": file_identity(REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/protocol.json"),
    }
    dump_json(output / "input_manifest.json", input_manifest)

    candidates, generation_audits = generate_candidate_pools(exposure)
    dump_json(output / "candidate_generation_audit.json", generation_audits)
    if len(candidates) != POOL_COUNT * ARCHITECTURES_PER_POOL * len(HP_REGIMES):
        write_failure(output, "INSUFFICIENT_FRESH_CANDIDATES", {
            "expected_candidate_configurations": POOL_COUNT * ARCHITECTURES_PER_POOL * len(HP_REGIMES),
            "actual_candidate_configurations": len(candidates),
            "pool_audits": generation_audits,
        })
        return
    fingerprints_by_pool = {
        pool_id: {row["architecture_fingerprint"] for row in candidates if row["pool_id"] == pool_id}
        for pool_id in range(POOL_COUNT)
    }
    cross_overlaps = [
        {"left_pool": left, "right_pool": right, "overlap_count": len(fingerprints_by_pool[left] & fingerprints_by_pool[right])}
        for left in range(POOL_COUNT) for right in range(left + 1, POOL_COUNT)
    ]
    pool_audit = {
        "passed": (
            len({row["config_id"] for row in candidates}) == len(candidates)
            and len({row["architecture_fingerprint"] for row in candidates}) == POOL_COUNT * ARCHITECTURES_PER_POOL
            and not ({row["architecture_fingerprint"] for row in candidates} & exposure)
            and all(row["overlap_count"] == 0 for row in cross_overlaps)
        ),
        "configuration_count": len(candidates),
        "architecture_count": len({row["architecture_fingerprint"] for row in candidates}),
        "configurations_per_pool": dict(Counter(row["pool_id"] for row in candidates)),
        "architectures_per_pool": {pool: len(values) for pool, values in fingerprints_by_pool.items()},
        "exposure_overlap_count": len({row["architecture_fingerprint"] for row in candidates} & exposure),
        "cross_pool_overlaps": cross_overlaps,
        "contains_test_fields": bool(contains_test_field(candidates)),
    }
    dump_json(output / "candidate_pool_audit.json", pool_audit)
    if not pool_audit["passed"] or pool_audit["contains_test_fields"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "candidate pool audit failed", "audit": pool_audit})
        return

    dump_json(output / "frozen_candidate_pools.json", candidates)
    dump_csv(output / "frozen_candidate_pools.csv", candidates)

    member_matrix, members, checkpoints = head_member_predictions(candidates, strict_root)
    if member_matrix.shape != (150, len(candidates)):
        raise RuntimeError(f"unexpected Head prediction shape: {member_matrix.shape}")
    prediction_rows = []
    for index, candidate in enumerate(candidates):
        values = member_matrix[:, index].astype(float).tolist()
        prediction_rows.append({
            "pool_id": candidate["pool_id"],
            "config_id": candidate["config_id"],
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "hp_fingerprint": candidate["hp_fingerprint"],
            "head_prediction_mean": float(np.mean(values)),
            "head_prediction_std_descriptive_only": float(np.std(values, ddof=1)),
            "head_member_predictions": values,
            "member_count": len(values),
        })
    dump_json(output / "frozen_head_predictions.json", prediction_rows)
    dump_csv(output / "frozen_head_predictions.csv", prediction_rows)
    dump_json(output / "frozen_head_member_identity.json", {"members": members, "checkpoints": checkpoints})
    means = {row["config_id"]: row["head_prediction_mean"] for row in prediction_rows}

    assignments = []
    partition_audits = []
    for pool_id in range(POOL_COUNT):
        pool_rows = [row for row in candidates if row["pool_id"] == pool_id]
        pool_assignments, audit = partition_pool(pool_rows, pool_id)
        assignments.extend(pool_assignments)
        partition_audits.append(audit)
    dump_json(output / "frozen_partitions.json", assignments)
    dump_csv(output / "frozen_partitions.csv", assignments)
    dump_json(output / "partition_audit.json", partition_audits)
    if not all(row["passed"] for row in partition_audits):
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "partition audit failed"})
        return

    selections, selection_audit = strategy_selections(candidates, assignments, means)
    dump_json(output / "frozen_strategy_selections.json", selections)
    dump_csv(output / "frozen_strategy_selections.csv", selections)
    dump_json(output / "strategy_selection_audit.json", selection_audit)
    union, tasks = make_union_tasks(candidates, selections)
    dump_json(output / "selected_union_configs.json", union)
    dump_csv(output / "selected_union_configs.csv", union)
    dump_json(output / "selected_union_tasks.json", tasks)
    dump_csv(output / "selected_union_tasks.csv", tasks)

    new48_protocol = load_json(
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/protocol.json"
    )
    gpu_rows = [
        [piece.strip() for piece in line.split(",", 2)] for line in run_text([
            "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
        ]).splitlines()
    ]
    gpu_map = {str(int(row[0])): row[1] for row in gpu_rows if len(row) == 3}
    if set(gpu_map) != {"0", "1"}:
        raise RuntimeError(f"expected exactly GPU 0 and GPU 1, found {gpu_map}")
    preregistration = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "scientific_name": "Original Head deterministic batch NAS+HPO pilot, global4 v1",
        "method_name": "predictor-guided deterministic batch NAS+HPO search",
        "method_is_bo": False,
        "pool_count": POOL_COUNT,
        "architectures_per_pool": ARCHITECTURES_PER_POOL,
        "hp_regimes": HP_REGIMES,
        "configurations_per_pool": ARCHITECTURES_PER_POOL * len(HP_REGIMES),
        "strategy_budget": 4,
        "master_seed": MASTER_SEED,
        "candidate_draw_rule": {"initial": INITIAL_DRAWS, "doubling": True, "maximum": MAX_DRAWS},
        "candidate_decode": "z ~ N(0,I), stochastic=False, one deterministic greedy decode",
        "distance": partition_audits[0]["distance_definition"],
        "partition": {"method": "deterministic k-medoids", "k": 4, "max_iterations": K_MEDOIDS_MAX_ITERATIONS},
        "strategies": {
            "HEAD_PARTITION_DIVERSITY": "one maximum Original Head mean per partition; order by mean descending",
            "HEAD_ONLY": "global maximum four Original Head means; order by mean descending",
            "RANDOM_PARTITION_DIVERSITY": "one fixed random selection per partition and a fixed random order; blind schema",
        },
        "training": {
            "dataset": "Cora", "split": "public", "hp_mode": "global4", "max_epochs": 300,
            "patience": 80, "seeds": list(TRAINING_SEEDS), "track_test": False,
            "maximum_system_retries": MAX_SYSTEM_RETRIES,
            "dataset_content_fingerprint": new48_protocol["dataset_content_fingerprint"],
            "split_fingerprint": new48_protocol["split_fingerprint"],
            "shard_to_gpu_uuid": gpu_map,
            "configuration_stable_sharding": True,
        },
        "statistics": {
            "primary_unit": "pool", "bootstrap_replicates": 20000,
            "bootstrap_seed": derived_seed("paired_pool_bootstrap"),
            "comparisons": ["A-C", "A-B", "B-C"], "budgets": [1, 2, 4],
            "regret": "observed-union regret only",
            "random_diversity_retained_definition": (
                "mean(A-C)<=0 and wins(A-C)<=4 and mean(B-C)<=0 and wins(B-C)<=4"
            ),
        },
        "success_gates": {
            "A": ["mean_A_minus_C >= 0.006", "bootstrap_lower_A_minus_C > 0", "A_wins_C >= 6/8", "bootstrap_lower_A_minus_B > -0.006"],
            "B_if_A_fails": ["mean_B_minus_C >= 0.006", "bootstrap_lower_B_minus_C > 0", "B_wins_C >= 6/8"],
        },
        "classifications": CLASSIFICATIONS,
        "forbidden": [
            "model or scaler fitting", "VAE or Head training", "GP/BO/acquisition", "Head std selection",
            "H or Refit Head selection", "test metric", "latent Euclidean diversity", "post-label adaptation",
        ],
        "code_hashes": {
            str(path.resolve()): sha256_file(path) for path in [
                Path(__file__),
                REPO_ROOT / "analyse/run_original_head_batch_pilot_labels.py",
                REPO_ROOT / "analyse/analyse_original_head_batch_pilot.py",
                REPO_ROOT / "tests/test_original_head_deterministic_batch_pilot.py",
            ]
        },
    }
    dump_json(output / "preregistration.json", preregistration)

    expected_task_count = len(union) * len(TRAINING_SEEDS)
    firewall = {
        "passed": (
            pool_audit["passed"] and selection_audit["passed"]
            and len(tasks) == expected_task_count
            and not contains_test_field(candidates)
            and not contains_test_field(selections)
            and all(row["seed"] in TRAINING_SEEDS for row in tasks)
        ),
        "candidate_and_selection_validation_or_test_values_accessed": False,
        "exposure_scan_deserialized_label_or_test_values": False,
        "candidate_exposure_overlap_count": pool_audit["exposure_overlap_count"],
        "cross_pool_overlap_count": sum(row["overlap_count"] for row in cross_overlaps),
        "strategy_budget_counts": selection_audit["pool_strategy_counts"],
        "random_strategy_received_head": False,
        "random_strategy_received_labels": False,
        "distance_used_mu_or_latent_euclidean": False,
        "distance_used_predictions_or_labels": False,
        "head_std_used_for_selection": False,
        "union_configuration_count": len(union),
        "expected_task_count": expected_task_count,
        "actual_task_count": len(tasks),
        "task_keys_unique": len({row["task_key"] for row in tasks}) == len(tasks),
        "three_fixed_seeds_per_configuration": all(
            {row["seed"] for row in tasks if row["config_id"] == config["config_id"]} == set(TRAINING_SEEDS)
            for config in union
        ),
        "test_tracking": False,
        "test_fields_found": contains_test_field({"candidates": candidates, "selections": selections, "tasks": tasks}),
    }
    dump_json(output / "pre_label_firewall_audit.json", firewall)
    if not firewall["passed"] or firewall["test_fields_found"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "pre-label firewall failed", "firewall": firewall})
        return

    frozen_names = [
        "frozen_candidate_pools.json", "frozen_candidate_pools.csv",
        "frozen_head_predictions.json", "frozen_head_predictions.csv",
        "frozen_partitions.json", "frozen_partitions.csv",
        "frozen_strategy_selections.json", "frozen_strategy_selections.csv",
        "selected_union_configs.json", "selected_union_configs.csv",
        "selected_union_tasks.json", "selected_union_tasks.csv",
        "preregistration.json", "pre_label_firewall_audit.json",
    ]
    entries = [{**file_identity(output / name), "relative_path": name} for name in frozen_names]
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in entries])
    dump_json(output / "pre_label_immutable_manifest.json", {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "files": entries,
        "manifest_root_hash": root_hash,
        "immutable_after_new_label_connection": True,
    })
    print(json.dumps({
        "status": "PRE_LABEL_FREEZE_COMPLETE",
        "candidate_count": len(candidates),
        "architecture_count": len({row["architecture_fingerprint"] for row in candidates}),
        "union_config_count": len(union),
        "task_count": len(tasks),
        "manifest_root_hash": root_hash,
    }, sort_keys=True))


def verify_freeze(output: Path) -> dict[str, Any]:
    manifest = load_json(output / "pre_label_immutable_manifest.json")
    mismatches = []
    for row in manifest["files"]:
        path = output / row["relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"path": str(path), "expected": row["sha256"], "actual": actual})
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in manifest["files"]])
    audit = {
        "passed": not mismatches and root == manifest["manifest_root_hash"],
        "mismatches": mismatches,
        "stored_root_hash": manifest["manifest_root_hash"],
        "recomputed_root_hash": root,
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify-freeze"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        prepare(output)
    else:
        audit = verify_freeze(output)
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
