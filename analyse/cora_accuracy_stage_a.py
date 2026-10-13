"""Read-only Cora Stage-A accuracy diagnostics.

The analysis consumes the frozen 15-run Cora S0/G100/G150 history family.  It
never imports or calls the GNN training evaluator, and it rejects test fields
from all primary diagnostic records.  Historical inputs are read in place;
new outputs are written only to a caller-selected, previously absent directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import igraph
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_utils import load_dataset_from_request, resolve_dataset_request
from nas_space import JointSpaceVAE, OP_MAPPING


ANALYSIS_SEED = 20260816
ARCH_DIM = 12
HP_DIM = 4
Z_BOUND = 2.5
EXPECTED_FULL_RECORDS = 4500
EXPECTED_DEDUPLICATED = 3582
EXPECTED_PREQUENTIAL = 3750
EXPECTED_DIRECT_PAIRS = {"G100": 500, "G150": 750}
EXPECTED_CHECKPOINT_SHA256 = (
    "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4"
)
SEED_SCHEME = "sha256_v1_candidate_evaluation_search_seed_fingerprint_fidelity"
SEED_DERIVATION = "sha256_v1"
FORBIDDEN_PRIMARY_FIELDS = {
    "test_acc",
    "test_accuracy",
    "test_mean",
    "mean_test_accuracy",
    "selected_candidate_test_mean",
}
RUN_RE = re.compile(
    r"^formal_seedfair_full300_(?P<kind>schur|gmm_ted_lowfid_gmm_fit_pool_exp(?P<exp>100|150))"
    r"_global4_seed(?P<seed>[0-4])_pool768$"
)


class IntegrityFailure(RuntimeError):
    """Raised before any formal Stage-A output is created."""


class ArchArgs:
    def __init__(self) -> None:
        self.max_n = 7
        self.num_vertex_type = 8
        self.START_TYPE = 0
        self.END_TYPE = 1
        self.hs = 501
        self.nz = ARCH_DIM
        self.bidirectional = True


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_fingerprint(z_search: Sequence[float]) -> str:
    array = np.ascontiguousarray(
        np.asarray(z_search, dtype=np.float32).reshape(1, -1)
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def architecture_fingerprint(record: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "operations": [str(value) for value in record["operations"]],
            "edges": sorted(
                [list(map(int, edge)) for edge in record["edges"]]
            ),
        }
    )


def hp_fingerprint(record: Mapping[str, Any]) -> str:
    source = record.get("hp")
    if not isinstance(source, Mapping):
        source = record
    payload = {
        "lr": float(source["lr"]),
        "dropout": float(source["dropout"]),
        "hidden_dim": int(source["hidden_dim"]),
        "weight_decay": float(source.get("l2", source.get("weight_decay"))),
        "gat_heads": int(source.get("gat_heads", 1)),
        "sage_aggr": str(source.get("sage_aggr", "mean")),
        "gin_eps": float(source.get("gin_eps", 0.0)),
        "gat_heads_by_layer": source.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": source.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": source.get("gin_eps_by_layer"),
        "condition_mask_vector": source.get("condition_mask_vector"),
    }
    return canonical_json_sha256(payload)


def _seed_component_bytes(component: Any) -> tuple[bytes, bytes]:
    if isinstance(component, str):
        return b"str", component.encode("utf-8")
    if isinstance(component, (int, np.integer)):
        return b"int", str(int(component)).encode("ascii")
    array = np.ascontiguousarray(component, dtype=np.float32)
    return b"bytes", array.tobytes(order="C")


def stable_seed(base_seed: int, namespace: str, *components: Any) -> int:
    digest = hashlib.sha256()
    tagged = (
        (b"version", SEED_DERIVATION.encode("ascii")),
        (b"base_seed", str(int(base_seed)).encode("ascii")),
        (b"namespace", namespace.encode("utf-8")),
    )
    for tag, payload in tagged:
        digest.update(len(tag).to_bytes(4, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    for component in components:
        tag, payload = _seed_component_bytes(component)
        digest.update(len(tag).to_bytes(4, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def candidate_evaluation_seed(
    search_seed: int, fingerprint: str, fidelity: str
) -> int:
    return stable_seed(
        search_seed, "candidate_evaluation", fingerprint, fidelity
    )


def decoder_seed(search_seed: int, z_arch: Sequence[float]) -> int:
    return stable_seed(search_seed, "decoder", np.asarray(z_arch, np.float32))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def method_from_root(path: Path) -> tuple[str, int]:
    match = RUN_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a frozen Cora Stage-A run: {path}")
    method = "S0" if match.group("kind") == "schur" else f"G{match.group('exp')}"
    return method, int(match.group("seed"))


def discover_runs(artifact_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(artifact_root.iterdir()):
        if not path.is_dir() or RUN_RE.match(path.name) is None:
            continue
        method, seed = method_from_root(path)
        runs.append({"method": method, "search_seed": seed, "path": path})
    expected = {(method, seed) for method in ("S0", "G100", "G150") for seed in range(5)}
    observed = {(row["method"], row["search_seed"]) for row in runs}
    if observed != expected:
        raise IntegrityFailure(
            f"formal run matrix mismatch: expected {sorted(expected)}, got {sorted(observed)}"
        )
    return runs


def _assert_no_test_fields(record: Mapping[str, Any], context: str) -> None:
    present = sorted(FORBIDDEN_PRIMARY_FIELDS.intersection(record))
    if present:
        raise IntegrityFailure(
            f"validation/test leakage in {context}: forbidden fields {present}"
        )


def _validate_record(
    record: Mapping[str, Any], *, method: str, search_seed_value: int,
    fidelity: str, context: str,
) -> None:
    _assert_no_test_fields(record, context)
    if record.get("evaluation_fidelity") != fidelity:
        raise IntegrityFailure(
            f"{context}: fidelity mismatch {record.get('evaluation_fidelity')!r}"
        )
    if int(record.get("search_seed", -1)) != search_seed_value:
        raise IntegrityFailure(f"{context}: search seed mismatch")
    z_search = record.get("z_search")
    if not isinstance(z_search, list) or len(z_search) != ARCH_DIM + HP_DIM:
        raise IntegrityFailure(f"{context}: invalid z_search")
    fingerprint = candidate_fingerprint(z_search)
    if record.get("candidate_fingerprint") != fingerprint:
        raise IntegrityFailure(f"{context}: candidate fingerprint mismatch")
    if architecture_fingerprint(record) != record.get("architecture_fingerprint"):
        raise IntegrityFailure(f"{context}: architecture fingerprint mismatch")
    if hp_fingerprint(record) != record.get("hp_fingerprint"):
        raise IntegrityFailure(f"{context}: HP fingerprint mismatch")
    if record.get("seed_derivation") != SEED_DERIVATION:
        raise IntegrityFailure(f"{context}: seed derivation mismatch")
    if record.get("candidate_evaluation_seed_scheme") != SEED_SCHEME:
        raise IntegrityFailure(f"{context}: evaluation seed scheme mismatch")
    expected_eval_seed = candidate_evaluation_seed(
        search_seed_value, fingerprint, fidelity
    )
    if record.get("evaluation_seed") != expected_eval_seed:
        raise IntegrityFailure(f"{context}: evaluation seed mismatch")
    if record.get("candidate_eval_seed") != expected_eval_seed:
        raise IntegrityFailure(f"{context}: candidate_eval_seed mismatch")
    expected_decoder_seed = decoder_seed(search_seed_value, z_search[:ARCH_DIM])
    if record.get("decoder_seed") != expected_decoder_seed:
        raise IntegrityFailure(f"{context}: decoder seed mismatch")
    if fidelity == "full":
        for field in ("best_epoch", "stopped_epoch", "epochs_ran"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IntegrityFailure(f"{context}: invalid {field}")
        if int(record["best_epoch"]) > int(record["stopped_epoch"]):
            raise IntegrityFailure(f"{context}: best_epoch after stop")
        if int(record["stopped_epoch"]) != int(record["epochs_ran"]):
            raise IntegrityFailure(f"{context}: stopped_epoch/epochs_ran mismatch")


def _final_eval_root(artifact_root: Path) -> Path:
    return artifact_root / "final_eval_topk10_r30_seedfair_base20260724" / "runs"


def _final_label(method: str, seed: int) -> str:
    prefix = {"S0": "schur", "G100": "gmm_exp100", "G150": "gmm_exp150"}[method]
    return f"{prefix}_seed{seed}"


def integrity_gate(
    repo_root: Path, artifact_root: Path, checkpoint: Path, cora_root: Path,
    initial_git_status_file: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    runs = discover_runs(artifact_root)
    full_records: list[dict[str, Any]] = []
    prequential_count = 0
    history_hash_matches = 0
    per_run: list[dict[str, Any]] = []
    low_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        path = run["path"]
        history_path = path / "history_final.json"
        raw = load_json(history_path)
        if not isinstance(raw, list) or len(raw) != 300:
            raise IntegrityFailure(f"{history_path}: expected 300 records")
        run_gp = 0
        for position, record in enumerate(raw):
            _validate_record(
                record, method=run["method"],
                search_seed_value=run["search_seed"], fidelity="full",
                context=f"{path.name}/history[{position}]",
            )
            row = dict(record)
            row["_method"] = run["method"]
            row["_run_seed"] = run["search_seed"]
            row["_run_name"] = path.name
            full_records.append(row)
            if record.get("gp_pred_mean") is not None:
                run_gp += 1
        if run_gp != 250:
            raise IntegrityFailure(f"{path.name}: expected 250 prequential records")
        prequential_count += run_gp

        final_config = _final_eval_root(artifact_root) / _final_label(
            run["method"], run["search_seed"]
        ) / "final_eval_config.json"
        config = load_json(final_config)
        if config.get("history_sha256") != sha256_file(history_path):
            raise IntegrityFailure(f"{path.name}: final-eval history hash mismatch")
        if config.get("sort_by") != "val_acc" or not bool(config.get("deduplicate")):
            raise IntegrityFailure(f"{path.name}: final selection semantics mismatch")
        history_hash_matches += 1

        if run["method"] != "S0":
            low_path = path / "low_fidelity_history.json"
            low = load_json(low_path)
            if not isinstance(low, list):
                raise IntegrityFailure(f"{low_path}: expected JSON list")
            for position, record in enumerate(low):
                _validate_record(
                    record, method=run["method"],
                    search_seed_value=run["search_seed"], fidelity="low",
                    context=f"{path.name}/low_fidelity[{position}]",
                )
                row = dict(record)
                row["_method"] = run["method"]
                row["_run_seed"] = run["search_seed"]
                row["_run_name"] = path.name
                low_by_method[run["method"]].append(row)
        per_run.append(
            {
                "method": run["method"], "search_seed": run["search_seed"],
                "full_records": len(raw), "prequential_records": run_gp,
                "history_sha256": sha256_file(history_path),
                "final_eval_history_hash_match": True,
            }
        )

    if len(full_records) != EXPECTED_FULL_RECORDS:
        raise IntegrityFailure("full-record population mismatch")
    unique = {row["candidate_fingerprint"] for row in full_records}
    if len(unique) != EXPECTED_DEDUPLICATED:
        raise IntegrityFailure(
            f"deduplicated population mismatch: {len(unique)}"
        )
    if prequential_count != EXPECTED_PREQUENTIAL:
        raise IntegrityFailure("prequential population mismatch")

    direct_counts: dict[str, int] = {}
    for method in ("G100", "G150"):
        low_keys = {
            (int(row["search_seed"]), row["candidate_fingerprint"])
            for row in low_by_method[method]
        }
        full_keys = {
            (int(row["search_seed"]), row["candidate_fingerprint"])
            for row in full_records if row["_method"] == method
        }
        direct_counts[method] = len(low_keys.intersection(full_keys))
        if direct_counts[method] != EXPECTED_DIRECT_PAIRS[method]:
            raise IntegrityFailure(
                f"{method} direct-pair mismatch: {direct_counts[method]}"
            )

    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise IntegrityFailure("VAE checkpoint fingerprint mismatch")
    metadata_hashes = set()
    for run in runs:
        metadata_path = run["path"] / "candidate_pool_metadata.json"
        if metadata_path.is_file():
            metadata_hashes.add(load_json(metadata_path).get("vae_checkpoint_sha256"))
    if metadata_hashes != {checkpoint_hash}:
        raise IntegrityFailure(
            f"candidate-pool checkpoint provenance mismatch: {sorted(metadata_hashes)}"
        )

    request = resolve_dataset_request(
        "cora", None, cora_root=str(cora_root)
    )
    bundle = load_dataset_from_request(request)
    if bundle.split_protocol != "planetoid_public":
        raise IntegrityFailure("Cora split protocol mismatch")

    initial_status = None
    if initial_git_status_file is not None:
        initial_status = initial_git_status_file.read_text(encoding="utf-8")
    gate = {
        "status": "PASS",
        "repository": {
            "branch": _git(repo_root, "branch", "--show-current"),
            "head_commit": _git(repo_root, "rev-parse", "HEAD"),
            "initial_git_status_porcelain": initial_status,
            "current_git_status_porcelain": _git(repo_root, "status", "--porcelain=v1"),
        },
        "population": {
            "formal_full_records": len(full_records),
            "deduplicated_candidates": len(unique),
            "stored_true_prequential_records": prequential_count,
            "direct_lf_hf_pairs": direct_counts,
            "run_matrix": "S0/G100/G150 x search seeds 0-4",
        },
        "identity": {
            "candidate_fingerprints_recomputed": len(full_records),
            "architecture_fingerprints_recomputed": len(full_records),
            "hp_fingerprints_recomputed": len(full_records),
            "evaluation_and_decoder_seeds_recomputed": len(full_records)
            + sum(len(rows) for rows in low_by_method.values()),
            "history_hashes_linked_to_final_eval": history_hash_matches,
            "vae_checkpoint": str(checkpoint),
            "vae_checkpoint_sha256": checkpoint_hash,
        },
        "cora_dataset": {
            "split_protocol": bundle.split_protocol,
            "current_reference_split_fingerprint": bundle.split_fingerprint,
            "current_reference_dataset_content_fingerprint": bundle.dataset_content_fingerprint,
            "nodes": bundle.num_nodes,
            "edges": bundle.num_edges,
            "features": bundle.num_features,
            "classes": bundle.num_classes,
            "historical_split_fingerprint": "legacy_unrecorded",
            "historical_split_assessment": (
                "The historical artifacts name the Planetoid public split but predate "
                "split-hash recording. No mismatch is observed; exact historical hash "
                "identity cannot be independently proven."
            ),
        },
        "semantics": {
            "primary_metric": "validation accuracy only",
            "validation_selected_epoch": True,
            "best_epoch_is_1_based": True,
            "search_records_contain_test_fields": False,
            "final_test_selection": "candidate selected by final-validation mean; test reported after selection",
            "formal_history_git_commit": "legacy_unrecorded",
            "formal_history_protocol_id": "legacy_unrecorded",
            "provenance_assessment": (
                "Artifact identities, seeds, history hashes, protocol settings, and checkpoint "
                "are internally consistent. Historical run Git commit/protocol ID were not recorded "
                "and current HEAD is not substituted."
            ),
        },
        "per_run": per_run,
        "caveats": [
            "Historical Cora search split hash is unavailable; the recoverable protocol is planetoid_public.",
            "Historical Cora search Git commit and formal protocol ID are unavailable.",
        ],
    }
    return gate, full_records, [row for rows in low_by_method.values() for row in rows]


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(map(float, x))) < 2 or len(set(map(float, y))) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def _safe_kendall(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(map(float, x))) < 2 or len(set(map(float, y))) < 2:
        return None
    value = float(kendalltau(x, y).statistic)
    return value if math.isfinite(value) else None


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int = 2000,
) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"estimate": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n": int(array.size),
    }


def bootstrap_rho_difference(
    pairs: Sequence[Mapping[str, float]], *, seed: int, replicates: int = 1000,
) -> dict[str, Any]:
    if len(pairs) < 20:
        return {"estimate": None, "ci95": [None, None], "n_pairs": len(pairs)}
    latent = np.asarray([row["latent_distance"] for row in pairs], dtype=float)
    discrete = np.asarray([row["combined_distance"] for row in pairs], dtype=float)
    delta = np.asarray([row["performance_delta"] for row in pairs], dtype=float)
    estimate = float(spearmanr(discrete, delta).statistic - spearmanr(latent, delta).statistic)
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(replicates):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        left = float(spearmanr(discrete[indices], delta[indices]).statistic)
        right = float(spearmanr(latent[indices], delta[indices]).statistic)
        if math.isfinite(left) and math.isfinite(right):
            effects.append(left - right)
    return {
        "estimate": estimate,
        "ci95": [float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))],
        "n_pairs": len(pairs),
        "definition": "Spearman(discrete distance, |delta validation|) - Spearman(latent distance, |delta validation|)",
    }


def op_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = list(left["operations"])
    b = list(right["operations"])
    width = max(len(a), len(b), 1)
    a += ["<PAD>"] * (width - len(a))
    b += ["<PAD>"] * (width - len(b))
    return sum(x != y for x, y in zip(a, b)) / width


def edge_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = {tuple(map(int, edge)) for edge in left["edges"]}
    b = {tuple(map(int, edge)) for edge in right["edges"]}
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def _path_signatures(record: Mapping[str, Any]) -> set[tuple[str, ...]]:
    operations = list(record["operations"])
    end = len(operations) + 1
    adjacency: dict[int, list[int]] = defaultdict(list)
    for source, target in record["edges"]:
        adjacency[int(source)].append(int(target))
    signatures: set[tuple[str, ...]] = set()

    def visit(node: int, path: list[int]) -> None:
        if node == end:
            signatures.add(tuple(operations[index - 1] for index in path if 1 <= index <= len(operations)))
            return
        for target in sorted(adjacency.get(node, [])):
            if target not in path:
                visit(target, [*path, target])

    visit(0, [0])
    return signatures or {tuple()}


def path_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = _path_signatures(left)
    b = _path_signatures(right)
    return 1.0 - len(a & b) / len(a | b)


def pair_distances(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    z_left = np.asarray(left["z_search"][:ARCH_DIM], dtype=float)
    z_right = np.asarray(right["z_search"][:ARCH_DIM], dtype=float)
    op = op_distance(left, right)
    edge = edge_distance(left, right)
    return {
        "latent_distance": float(np.linalg.norm(z_left - z_right) / (2 * Z_BOUND * math.sqrt(ARCH_DIM))),
        "operation_distance": op,
        "edge_distance": edge,
        "combined_distance": (op + edge) / 2.0,
        "path_distance": path_distance(left, right),
        "performance_delta": abs(float(left["val_acc"]) - float(right["val_acc"])),
    }


def _random_pair_indices(n: int, count: int, seed: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    maximum = n * (n - 1) // 2
    if maximum <= count:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        values = rng.integers(0, n, size=(min(count, 8192), 2))
        for left, right in values:
            if left == right:
                continue
            pair = (int(min(left, right)), int(max(left, right)))
            pairs.add(pair)
            if len(pairs) == count:
                break
    return sorted(pairs)


def _nearest_pairs(records: Sequence[Mapping[str, Any]], columns: slice, k: int = 6) -> list[tuple[int, int]]:
    if len(records) < 2:
        return []
    matrix = np.asarray([row["z_search"][columns] for row in records], dtype=float)
    tree = cKDTree(matrix)
    _distances, neighbors = tree.query(matrix, k=min(k + 1, len(records)))
    pairs = {
        (min(i, int(j)), max(i, int(j)))
        for i, row in enumerate(neighbors)
        for j in np.atleast_1d(row)[1:]
        if i != int(j)
    }
    return sorted(pairs)


def _pair_rows(records: Sequence[Mapping[str, Any]], indices: Iterable[tuple[int, int]]) -> list[dict[str, float]]:
    return [pair_distances(records[left], records[right]) for left, right in indices]


def _geometry_metrics(pairs: Sequence[Mapping[str, float]], seed: int) -> dict[str, Any]:
    delta = [row["performance_delta"] for row in pairs]
    metrics = {
        "n_pairs": len(pairs),
        "latent_rho": _safe_spearman([row["latent_distance"] for row in pairs], delta),
        "operation_rho": _safe_spearman([row["operation_distance"] for row in pairs], delta),
        "edge_rho": _safe_spearman([row["edge_distance"] for row in pairs], delta),
        "combined_rho": _safe_spearman([row["combined_distance"] for row in pairs], delta),
        "path_rho": _safe_spearman([row["path_distance"] for row in pairs], delta),
        "median_abs_validation_delta": float(np.median(delta)) if delta else None,
    }
    metrics["discrete_minus_latent_effect"] = bootstrap_rho_difference(
        pairs, seed=seed
    )
    return metrics


@contextmanager
def isolated_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _config_from_graph(model: JointSpaceVAE, graph: igraph.Graph) -> dict[str, Any]:
    operations = [
        model.op_mapping.get(graph.vs[index]["type"], "Identity")
        for index in range(1, graph.vcount() - 1)
    ]
    return {"operations": operations, "edges": [list(edge) for edge in graph.get_edgelist()]}


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "operations": list(config["operations"]),
            "edges": sorted([list(map(int, edge)) for edge in config["edges"]]),
        }
    )


def _graph_from_config(config: Mapping[str, Any]) -> igraph.Graph:
    reverse = {name: index for index, name in OP_MAPPING.items()}
    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(config["operations"]) + 2)
    graph.vs["type"] = [0, *[reverse[name] for name in config["operations"]], 1]
    graph.add_edges([tuple(map(int, edge)) for edge in config["edges"]])
    return graph


def load_vae(checkpoint: Path) -> JointSpaceVAE:
    model = JointSpaceVAE(
        ArchArgs(), hp_mode="global4", hp_latent_dim=HP_DIM,
        hp_input_dim=HP_DIM,
    ).cpu()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def select_decoder_representatives(records: Sequence[Mapping[str, Any]], count: int = 128) -> list[Mapping[str, Any]]:
    ordered = sorted(records, key=lambda row: (float(row["val_acc"]), row["candidate_fingerprint"]))
    top_count = count // 2
    top_start = max(0, math.floor(0.8 * len(ordered)))
    top = ordered[top_start:]
    top_indices = np.linspace(0, len(top) - 1, num=top_count, dtype=int)
    rest_indices = np.linspace(0, max(top_start - 1, 0), num=count - top_count, dtype=int)
    selected = [top[index] for index in top_indices] + [ordered[index] for index in rest_indices]
    unique: dict[str, Mapping[str, Any]] = {}
    for row in selected:
        unique[row["candidate_fingerprint"]] = row
    for row in reversed(ordered):
        if len(unique) >= count:
            break
        unique.setdefault(row["candidate_fingerprint"], row)
    return sorted(unique.values(), key=lambda row: row["candidate_fingerprint"])[:count]


def deduplicate_candidate_population(
    full_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in full_records:
        grouped[row["candidate_fingerprint"]].append(row)
    deduplicated = []
    architecture_disagreements = 0
    validation_ranges = []
    same_seed_repeated_candidates = 0
    for fingerprint, raw_rows in grouped.items():
        seed_counts = Counter(int(row["search_seed"]) for row in raw_rows)
        same_seed_repeated_candidates += int(any(count > 1 for count in seed_counts.values()))
        architecture_counts = Counter(
            row["architecture_fingerprint"] for row in raw_rows
        )
        modal_architecture = sorted(
            architecture_counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        representative = next(
            row for row in raw_rows
            if row["architecture_fingerprint"] == modal_architecture
        )
        values = [float(row["val_acc"]) for row in raw_rows]
        aggregated = dict(representative)
        aggregated["val_acc"] = float(np.mean(values))
        aggregated["value"] = aggregated["val_acc"]
        aggregated["_formal_observations"] = len(raw_rows)
        aggregated["_distinct_search_seeds"] = len(seed_counts)
        aggregated["_modal_architecture_support"] = (
            architecture_counts[modal_architecture] / len(raw_rows)
        )
        deduplicated.append(aggregated)
        validation_ranges.append(max(values) - min(values))
        architecture_disagreements += int(len(architecture_counts) > 1)
    return (
        sorted(deduplicated, key=lambda row: row["candidate_fingerprint"]),
        {
            "raw_records": len(full_records),
            "deduplicated_candidates": len(deduplicated),
            "candidates_with_multiple_formal_observations": sum(
                row["_formal_observations"] > 1 for row in deduplicated
            ),
            "candidates_repeated_within_the_same_search_seed_across_methods": same_seed_repeated_candidates,
            "candidates_with_cross_seed_architecture_disagreement": architecture_disagreements,
            "mean_cross_seed_validation_range": float(np.mean(validation_ranges)),
            "max_cross_seed_validation_range": float(np.max(validation_ranges)),
            "aggregation": (
                "mean validation across all formal observations for each candidate fingerprint; "
                "modal decoded architecture with deterministic fingerprint tie-break"
            ),
        },
    )


def decoder_audit(records: Sequence[Mapping[str, Any]], checkpoint: Path) -> dict[str, Any]:
    model = load_vae(checkpoint)
    representatives = select_decoder_representatives(records, 128)
    rows: list[dict[str, Any]] = []
    modal_fingerprints: list[str] = []
    for record in representatives:
        z = torch.tensor(record["z_search"][:ARCH_DIM], dtype=torch.float32)
        decoded: list[dict[str, Any]] = []
        seeds: list[int] = []
        for trial in range(10):
            seed = stable_seed(
                ANALYSIS_SEED, "stage_a_decoder", record["candidate_fingerprint"], trial
            )
            seeds.append(seed)
            with isolated_rng(seed), torch.no_grad():
                graph = model.arch_vae.decode(z.unsqueeze(0), stochastic=True)[0]
            decoded.append(_config_from_graph(model, graph))
        fingerprints = [_config_fingerprint(config) for config in decoded]
        counts = Counter(fingerprints)
        modal, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        modal_config = decoded[fingerprints.index(modal)]
        probabilities = np.asarray(list(counts.values()), dtype=float) / 10.0
        entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(10.0))
        graph = _graph_from_config(modal_config)
        with torch.no_grad():
            mu, _logvar = model.arch_vae.encode([graph])
        roundtrip = float(
            torch.linalg.vector_norm(mu[0] - z).item()
            / (2 * Z_BOUND * math.sqrt(ARCH_DIM))
        )
        with isolated_rng(seeds[0]), torch.no_grad():
            repeated = model.arch_vae.decode(z.unsqueeze(0), stochastic=True)[0]
        reproducible = _config_fingerprint(_config_from_graph(model, repeated)) == fingerprints[0]
        modal_fingerprints.append(modal)
        rows.append(
            {
                "candidate_fingerprint": record["candidate_fingerprint"],
                "validation_accuracy": float(record["val_acc"]),
                "unique_architectures": len(counts),
                "modal_consistency": modal_count / 10.0,
                "normalized_architecture_entropy": entropy,
                "modal_architecture_fingerprint": modal,
                "normalized_roundtrip_l2": roundtrip,
                "fixed_seed_reproducible": reproducible,
            }
        )
    return {
        "representative_latents": len(rows),
        "decoder_seeds_per_latent": 10,
        "mean_modal_consistency": float(np.mean([row["modal_consistency"] for row in rows])),
        "mean_normalized_architecture_entropy": float(np.mean([row["normalized_architecture_entropy"] for row in rows])),
        "one_z_many_architecture_rate": float(np.mean([row["unique_architectures"] > 1 for row in rows])),
        "many_z_one_modal_architecture_collision_rate": 1.0 - len(set(modal_fingerprints)) / len(modal_fingerprints),
        "median_normalized_roundtrip_l2": float(np.median([row["normalized_roundtrip_l2"] for row in rows])),
        "p90_normalized_roundtrip_l2": float(np.quantile([row["normalized_roundtrip_l2"] for row in rows], 0.9)),
        "fixed_seed_reproducibility": float(np.mean([row["fixed_seed_reproducible"] for row in rows])),
        "rows": rows,
    }


def a1_analysis(full_records: Sequence[Mapping[str, Any]], checkpoint: Path) -> dict[str, Any]:
    records, deduplication = deduplicate_candidate_population(full_records)
    global_pairs = _pair_rows(records, _random_pair_indices(len(records), 100000, ANALYSIS_SEED))
    top_threshold = float(np.quantile([row["val_acc"] for row in records], 0.8))
    top = [row for row in records if float(row["val_acc"]) >= top_threshold]
    top_pairs = _pair_rows(top, _random_pair_indices(len(top), 50000, ANALYSIS_SEED + 1))
    hp_near_pairs = _pair_rows(records, _nearest_pairs(records, slice(ARCH_DIM, ARCH_DIM + HP_DIM), 6))
    latent_local_pairs = _pair_rows(records, _nearest_pairs(records, slice(0, ARCH_DIM), 6))

    hp_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        hp_groups[row["hp_fingerprint"]].append(index)
    exact_indices = [
        (left, right)
        for indices in hp_groups.values() if len(indices) > 1
        for offset, left in enumerate(indices)
        for right in indices[offset + 1:]
        if records[left]["architecture_fingerprint"] != records[right]["architecture_fingerprint"]
    ]
    exact_pairs = _pair_rows(records, exact_indices)

    seed_effects = []
    for method in ("S0", "G100", "G150"):
        for seed in range(5):
            run = [row for row in full_records if row["_method"] == method and row["_run_seed"] == seed]
            pairs = _pair_rows(run, _random_pair_indices(len(run), 10000, ANALYSIS_SEED + seed + 100 * (1 + ("S0", "G100", "G150").index(method))))
            effect = _geometry_metrics(pairs, ANALYSIS_SEED)["discrete_minus_latent_effect"]["estimate"]
            seed_effects.append({"method": method, "search_seed": seed, "effect": effect})

    decoder = decoder_audit(records, checkpoint)
    regions = {
        "global": _geometry_metrics(global_pairs, ANALYSIS_SEED + 10),
        "top_region": _geometry_metrics(top_pairs, ANALYSIS_SEED + 11),
        "hp_near": _geometry_metrics(hp_near_pairs, ANALYSIS_SEED + 12),
        "latent_local": _geometry_metrics(latent_local_pairs, ANALYSIS_SEED + 13),
        "exact_hp_architecture_comparison": _geometry_metrics(exact_pairs, ANALYSIS_SEED + 14),
    }
    decision_effects = [
        regions[name]["discrete_minus_latent_effect"]
        for name in ("top_region", "hp_near", "latent_local")
    ]
    stable_discrete_advantage = sum(
        effect["ci95"][0] is not None and effect["ci95"][0] > 0.02
        for effect in decision_effects
    ) >= 2
    decoder_unstable = (
        decoder["fixed_seed_reproducibility"] < 1.0
        or decoder["mean_modal_consistency"] < 0.8
        or decoder["one_z_many_architecture_rate"] > 0.5
        or decoder["many_z_one_modal_architecture_collision_rate"] > 0.2
    )
    if decoder_unstable:
        classification = "A1-FAIL-DECODER"
    elif stable_discrete_advantage:
        classification = "A1-FAIL-GEOMETRY"
    elif all(
        effect["estimate"] is not None and abs(effect["estimate"]) <= 0.03
        for effect in decision_effects
    ):
        classification = "A1-PASS"
    else:
        classification = "A1-INCONCLUSIVE"
    return {
        "classification": classification,
        "population": {
            "deduplicated_candidates": len(records),
            "top_region_threshold": top_threshold,
            "top_region_candidates": len(top),
            "exact_hp_architecture_pairs": len(exact_pairs),
            "deduplication": deduplication,
        },
        "regions": regions,
        "seed_level_effects": seed_effects,
        "seed_level_discrete_advantage_fraction": float(np.mean([row["effect"] > 0 for row in seed_effects if row["effect"] is not None])),
        "decoder": decoder,
        "interpretation": (
            "Decoder instability/collision is the upstream geometry blocker."
            if classification == "A1-FAIL-DECODER" else
            "Discrete architecture distance consistently out-explains latent distance in decision-relevant regions."
            if classification == "A1-FAIL-GEOMETRY" else
            "Latent and discrete geometry are practically similar and the decoder is stable."
            if classification == "A1-PASS" else
            "Decision-region geometry effects are mixed or exact-HP support is insufficient."
        ),
    }


def ndcg_score(actual: Sequence[float], predicted: Sequence[float], k: int = 10) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    relevance = actual_array - actual_array.min()
    order = np.argsort(-predicted_array, kind="stable")[:k]
    ideal = np.argsort(-actual_array, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(relevance[order] * discounts))
    idcg = float(np.sum(relevance[ideal] * discounts))
    return dcg / idcg if idcg > 0 else 1.0


def pairwise_accuracy(records: Sequence[Mapping[str, Any]], pairs: Iterable[tuple[int, int]]) -> tuple[float | None, int]:
    correct = 0.0
    count = 0
    for left, right in pairs:
        actual_delta = float(records[left]["val_acc"]) - float(records[right]["val_acc"])
        predicted_delta = float(records[left]["gp_pred_mean"]) - float(records[right]["gp_pred_mean"])
        if actual_delta == 0 or predicted_delta == 0:
            continue
        correct += float((actual_delta > 0) == (predicted_delta > 0))
        count += 1
    return (correct / count if count else None), count


def _architecture_local_pairs(records: Sequence[Mapping[str, Any]], neighbors: int = 6) -> list[tuple[int, int]]:
    signatures: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        signatures[row["architecture_fingerprint"]].append(index)
    exact = {
        (left, right)
        for indices in signatures.values()
        for offset, left in enumerate(indices)
        for right in indices[offset + 1:]
    }
    all_pairs = _random_pair_indices(len(records), min(20000, len(records) * (len(records) - 1) // 2), ANALYSIS_SEED)
    closest = sorted(
        all_pairs,
        key=lambda pair: (
            (op_distance(records[pair[0]], records[pair[1]]) + edge_distance(records[pair[0]], records[pair[1]])) / 2.0,
            pair,
        ),
    )[: neighbors * len(records)]
    return sorted(exact.union(closest))


def _run_surrogate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["full_evaluation_index"]))
    actual = np.asarray([row["val_acc"] for row in ordered], dtype=float)
    predicted = np.asarray([row["gp_pred_mean"] for row in ordered], dtype=float)
    std = np.asarray([row["gp_pred_std"] for row in ordered], dtype=float)
    threshold = float(np.quantile(actual, 0.8))
    top = actual >= threshold
    latent_pairs = _nearest_pairs(ordered, slice(0, ARCH_DIM), 6)
    arch_pairs = _architecture_local_pairs(ordered, 6)
    latent_acc, latent_count = pairwise_accuracy(ordered, latent_pairs)
    arch_acc, arch_count = pairwise_accuracy(ordered, arch_pairs)
    top_k = min(10, len(ordered))
    predicted_top = np.argsort(-predicted, kind="stable")[:top_k]
    late_actual = actual[-50:]
    late_predicted = predicted[-50:]
    standardized = (actual - predicted) / std
    return {
        "n": len(ordered),
        "global_spearman": _safe_spearman(predicted, actual),
        "global_kendall": _safe_kendall(predicted, actual),
        "top20_spearman": _safe_spearman(predicted[top], actual[top]),
        "top20_kendall": _safe_kendall(predicted[top], actual[top]),
        "latent_local_pairwise_accuracy": latent_acc,
        "latent_local_pairs": latent_count,
        "architecture_local_pairwise_accuracy": arch_acc,
        "architecture_local_pairs": arch_count,
        "ndcg_at_10": ndcg_score(actual, predicted, top_k),
        "top10_regret": float(actual.max() - actual[predicted_top].max()),
        "coverage_95": float(np.mean(np.abs(standardized) <= 1.959963984540054)),
        "standardized_residual_rmse": float(np.sqrt(np.mean(standardized**2))),
        "prediction_bias": float(np.mean(predicted - actual)),
        "late50_spearman": _safe_spearman(late_predicted, late_actual),
        "late50_kendall": _safe_kendall(late_predicted, late_actual),
        "late50_top20_spearman": _safe_spearman(
            late_predicted[late_actual >= np.quantile(late_actual, 0.8)],
            late_actual[late_actual >= np.quantile(late_actual, 0.8)],
        ),
        "protocol": "stored true prequential predictions; each row was predicted before its label was appended",
    }


def assert_prefix_only(records: Sequence[Mapping[str, Any]], context: str) -> None:
    for row in records:
        _assert_no_test_fields(row, context)
        before = int(row["gp_train_size_before"])
        after = int(row["gp_train_size_after"])
        index = int(row["full_evaluation_index"])
        stage = row.get("evaluation_stage")
        online_valid = stage == "online_bo" and before == index and after == index + 1
        frozen_block_valid = (
            stage == "initial_expand"
            and before == 50
            and after == 50
            and index >= 50
            and not bool(row.get("gp_update_performed"))
        )
        if not (online_valid or frozen_block_valid):
            raise IntegrityFailure(
                f"{context}/index{index}: future-label leakage invariant failed"
            )


def a2_analysis(full_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runs = []
    for method in ("S0", "G100", "G150"):
        for seed in range(5):
            rows = [
                row for row in full_records
                if row["_method"] == method and row["_run_seed"] == seed
                and row.get("gp_pred_mean") is not None
            ]
            if len(rows) != 250:
                raise IntegrityFailure(f"{method}/seed{seed}: prequential count changed")
            assert_prefix_only(rows, f"{method}/seed{seed}")
            metrics = _run_surrogate_metrics(rows)
            metrics.update({"method": method, "search_seed": seed})
            runs.append(metrics)

    aggregate_fields = (
        "global_spearman", "global_kendall", "top20_spearman", "top20_kendall",
        "latent_local_pairwise_accuracy", "architecture_local_pairwise_accuracy",
        "ndcg_at_10", "top10_regret", "coverage_95",
        "standardized_residual_rmse", "prediction_bias", "late50_spearman",
        "late50_kendall", "late50_top20_spearman",
    )
    aggregate = {
        field: bootstrap_mean_ci(
            [row[field] for row in runs if row[field] is not None],
            seed=ANALYSIS_SEED + position,
        )
        for position, field in enumerate(aggregate_fields)
    }
    stored_online_count = sum(
        1 for row in full_records
        if row.get("gp_pred_mean") is not None and row.get("evaluation_stage") == "online_bo"
    )
    stored_frozen_block_count = sum(
        1 for row in full_records
        if row.get("gp_pred_mean") is not None and row.get("evaluation_stage") == "initial_expand"
    )
    global_reliable = (
        aggregate["global_spearman"]["estimate"] >= 0.3
        and aggregate["global_kendall"]["estimate"] >= 0.2
    )
    local_reliable = (
        aggregate["top20_spearman"]["estimate"] >= 0.3
        and aggregate["top20_kendall"]["estimate"] >= 0.2
        and aggregate["latent_local_pairwise_accuracy"]["estimate"] >= 0.6
        and aggregate["architecture_local_pairwise_accuracy"]["estimate"] >= 0.6
    )
    calibration_reliable = (
        0.90 <= aggregate["coverage_95"]["estimate"] <= 0.99
        and 0.7 <= aggregate["standardized_residual_rmse"]["estimate"] <= 1.3
    )
    if not global_reliable:
        classification = "A2-GLOBAL-FAIL"
    elif not local_reliable:
        classification = "A2-LOCAL-FAIL"
    elif not calibration_reliable:
        classification = "A2-CALIBRATION-FAIL"
    else:
        classification = "A2-PASS"
    modification = "NO" if classification == "A2-PASS" else "CONDITIONAL"
    return {
        "classification": classification,
        "surrogate_modification_may_improve_cora_accuracy": modification,
        "aggregate_across_15_runs": aggregate,
        "seed_level": runs,
        "seed_consistency": {
            "positive_global_spearman": sum(row["global_spearman"] > 0 for row in runs),
            "positive_top20_spearman": sum((row["top20_spearman"] or 0) > 0 for row in runs),
            "local_pairwise_above_chance": sum(
                row["latent_local_pairwise_accuracy"] > 0.5
                and row["architecture_local_pairwise_accuracy"] > 0.5
                for row in runs
            ),
            "runs": len(runs),
        },
        "temporal_replay": {
            "mode": "stored predictions with online-prequential and frozen-initial-expansion strata",
            "stored_online_prequential_records": stored_online_count,
            "stored_frozen_block_prediction_records": stored_frozen_block_count,
            "frozen_block_replay_performed": False,
            "exact_posterior_replay_claimed": False,
            "no_future_label_invariant": (
                "online_bo: gp_train_size_before == full_evaluation_index and current label is "
                "appended after prediction; initial_expand: all block predictions use frozen GP50 "
                "with gp_train_size_before == gp_train_size_after == 50"
            ),
        },
    }


def _direct_pairs(
    full_records: Sequence[Mapping[str, Any]], low_records: Sequence[Mapping[str, Any]],
    method: str, seed: int,
) -> list[dict[str, Any]]:
    full = {
        row["candidate_fingerprint"]: row for row in full_records
        if row["_method"] == method and row["_run_seed"] == seed
    }
    low = {
        row["candidate_fingerprint"]: row for row in low_records
        if row["_method"] == method and row["_run_seed"] == seed
    }
    rows = []
    for fingerprint in sorted(full.keys() & low.keys()):
        lf = low[fingerprint]
        hf = full[fingerprint]
        if not bool(lf.get("selected_for_full_expansion")):
            raise IntegrityFailure(f"{method}/seed{seed}: direct LF/HF pair was not promoted")
        if lf["evaluation_seed"] == hf["evaluation_seed"]:
            raise IntegrityFailure(f"{method}/seed{seed}: LF/HF seeds did not separate fidelity")
        rows.append(
            {
                "candidate_fingerprint": fingerprint,
                "lf_val": float(lf["val_acc"]),
                "hf_val": float(hf["val_acc"]),
                "combined_score": float(lf["combined_score"]),
                "lf_rank": float(lf["low_fidelity_rank"]),
            }
        )
    return rows


def _run_fidelity_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lf = np.asarray([row["lf_val"] for row in rows])
    hf = np.asarray([row["hf_val"] for row in rows])
    combined = np.asarray([row["combined_score"] for row in rows])
    n = len(rows)
    top_count = max(1, math.ceil(0.2 * n))
    hf_top = set(np.argsort(-hf, kind="stable")[:top_count].tolist())
    lf_top = set(np.argsort(-lf, kind="stable")[:top_count].tolist())
    top_hf_mask = hf >= np.quantile(hf, 0.8)
    k = min(10, n)
    lf_selected = np.argsort(-lf, kind="stable")[:k]
    combined_selected = np.argsort(-combined, kind="stable")[:k]
    q25, q75 = np.quantile(lf, [0.25, 0.75])
    return {
        "n_pairs": n,
        "spearman": _safe_spearman(lf, hf),
        "kendall": _safe_kendall(lf, hf),
        "top_region_spearman": _safe_spearman(lf[top_hf_mask], hf[top_hf_mask]),
        "top_region_kendall": _safe_kendall(lf[top_hf_mask], hf[top_hf_mask]),
        "top20_overlap": len(hf_top & lf_top) / top_count,
        "chance_top20_overlap": top_count / n,
        "lf_top10_regret": float(hf.max() - hf[lf_selected].max()),
        "combined_top10_regret": float(hf.max() - hf[combined_selected].max()),
        "lf_top_quartile_hf_mean": float(hf[lf >= q75].mean()),
        "lf_bottom_quartile_hf_mean": float(hf[lf <= q25].mean()),
        "cutoff_hf_mean_effect": float(hf[lf >= q75].mean() - hf[lf <= q25].mean()),
        "combined_score_spearman": _safe_spearman(combined, hf),
    }


def a3_analysis(
    full_records: Sequence[Mapping[str, Any]], low_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runs = []
    expected_per_seed = {"G100": 100, "G150": 150}
    for method in ("G100", "G150"):
        for seed in range(5):
            pairs = _direct_pairs(full_records, low_records, method, seed)
            if len(pairs) != expected_per_seed[method]:
                raise IntegrityFailure(
                    f"{method}/seed{seed}: expected {expected_per_seed[method]} direct pairs"
                )
            metrics = _run_fidelity_metrics(pairs)
            metrics.update({"method": method, "search_seed": seed})
            runs.append(metrics)
    aggregate_fields = (
        "spearman", "kendall", "top_region_spearman", "top_region_kendall",
        "top20_overlap", "lf_top10_regret", "combined_top10_regret",
        "cutoff_hf_mean_effect", "combined_score_spearman",
    )
    aggregate = {
        field: bootstrap_mean_ci(
            [row[field] for row in runs if row[field] is not None],
            seed=ANALYSIS_SEED + 300 + position,
        )
        for position, field in enumerate(aggregate_fields)
    }
    rho = aggregate["spearman"]
    overlap = aggregate["top20_overlap"]
    internal_failure = (
        (rho["ci95"][1] is not None and rho["ci95"][1] < 0.2)
        or (overlap["estimate"] <= 0.25 and overlap["ci95"][1] < 0.35)
    )
    internal_support = (
        rho["ci95"][0] is not None and rho["ci95"][0] >= 0.3
        and aggregate["top_region_spearman"]["estimate"] >= 0.2
        and overlap["estimate"] >= 0.4
    )
    if internal_failure:
        classification = "A3-DROP"
    elif internal_support:
        classification = "A3-RETAIN"
    else:
        classification = "A3-UNCERTAIN"
    return {
        "classification": classification,
        "aggregate_across_10_method_seed_runs": aggregate,
        "seed_level": runs,
        "seed_consistency": {
            "positive_spearman": sum(row["spearman"] > 0 for row in runs),
            "positive_top_region_spearman": sum((row["top_region_spearman"] or 0) > 0 for row in runs),
            "runs": len(runs),
        },
        "selection_bias_caveat": (
            "HF labels exist only for promoted candidates. True shortlist recall is not identifiable; "
            "unlabeled candidates are not treated as negatives. Cross-run accidental matches are excluded."
        ),
        "default_for_later_accuracy_experiments": (
            "LF20" if classification == "A3-RETAIN" else "S0/full fidelity"
        ),
    }


def joint_diagnosis(a1: Mapping[str, Any], a2: Mapping[str, Any], a3: Mapping[str, Any]) -> dict[str, Any]:
    a1_class = a1["classification"]
    a2_class = a2["classification"]
    if a1_class == "A1-FAIL-DECODER":
        interpretation = "Decoder consistency is an upstream blocker; do not begin B0 until it is resolved."
        next_direction = "FIX_UPSTREAM_FIRST"
    elif a1_class == "A1-FAIL-GEOMETRY" and a2_class == "A2-LOCAL-FAIL":
        interpretation = "Structure-aware surrogate work is a high-value conditional route, but remains unapproved."
        next_direction = "MORE_DIAGNOSTIC_NEEDED"
    elif a1_class == "A1-PASS" and a2_class == "A2-LOCAL-FAIL":
        interpretation = "The GP kernel/prior/noise is more suspect than the VAE representation."
        next_direction = "PROCEED_TO_B0"
    elif a1_class == "A1-PASS" and a2_class == "A2-PASS":
        interpretation = "Representation and GP are not the leading accuracy bottleneck; prioritize B0."
        next_direction = "PROCEED_TO_B0"
    elif a2_class == "A2-PASS":
        interpretation = "Do not modify the surrogate merely for theoretical sophistication."
        next_direction = "PROCEED_TO_B0"
    else:
        interpretation = "Stage-A identifies a surrogate/geometry limitation but not an approved accuracy intervention."
        next_direction = "MORE_DIAGNOSTIC_NEEDED"
    evidence_ranking = []
    if a1_class == "A1-FAIL-DECODER":
        evidence_ranking.append("decoder instability")
    if a2_class != "A2-PASS":
        evidence_ranking.append("surrogate ranking/calibration")
    if a1_class == "A1-FAIL-GEOMETRY":
        evidence_ranking.append("latent representation geometry")
    if a3["classification"] != "A3-RETAIN":
        evidence_ranking.append("LF20 promotion signal")
    evidence_ranking.append("unisolated GNN component choices (requires B0 to test)")
    return {
        "a1_x_a2_interpretation": interpretation,
        "cora_accuracy_bottleneck_evidence_order": evidence_ranking,
        "recommended_next_step": next_direction,
    }


def b0_impact(a1: Mapping[str, Any], a2: Mapping[str, Any]) -> dict[str, Any]:
    if a1["classification"] == "A1-FAIL-DECODER":
        return {
            "execute_now": False,
            "first_round_variants_after_upstream_fix": [
                "Baseline", "Projected-combine", "BatchNorm",
                "Projected residual", "PostMLP1",
            ],
            "deferred_until_after_first_round": ["PReLU"],
            "priority_answers": {
                "Projected-combine": "P0 after upstream fix",
                "BatchNorm": "P0 after upstream fix",
                "Projected residual": "P0 after upstream fix",
                "PostMLP1": "RETAIN FIRST ROUND after upstream fix",
                "PReLU": "DEFER until after first round",
            },
            "reason": (
                "Decoder consistency must be fixed before B0. After that fix, retain the four "
                "distinct combine/normalization/residual/post-processing probes and defer PReLU, "
                "which has the lowest structure-specific information gain."
            ),
        }
    retained = ["Baseline", "Projected-combine", "BatchNorm", "Projected residual", "PostMLP1"]
    return {
        "execute_now": False,
        "first_round_variants": retained,
        "deferred": ["PReLU"],
        "priority_answers": {
            "Projected-combine": "P0",
            "BatchNorm": "P0",
            "Projected residual": "P0",
            "PostMLP1": "RETAIN FIRST ROUND",
            "PReLU": "DEFER; lowest structure-specific information gain",
        },
        "reason": (
            "Stage A supplies no evidence that activation choice is a leading Cora bottleneck; "
            "the other four variants probe distinct combine/normalization/residual/post-processing mechanisms."
        ),
    }


def write_outputs(
    output: Path, gate: Mapping[str, Any], a1: Mapping[str, Any],
    a2: Mapping[str, Any], a3: Mapping[str, Any], repo_root: Path,
) -> None:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    directories = {
        "integrity": output / "integrity",
        "latent_geometry": output / "latent_geometry",
        "surrogate_audit": output / "surrogate_audit",
        "fidelity_audit": output / "fidelity_audit",
    }
    for directory in directories.values():
        directory.mkdir(parents=True)
    dump_json(directories["integrity"] / "integrity_gate.json", gate)
    dump_json(directories["latent_geometry"] / "a1_result.json", a1)
    dump_json(directories["surrogate_audit"] / "a2_result.json", a2)
    dump_json(directories["fidelity_audit"] / "a3_result.json", a3)
    diagnosis = joint_diagnosis(a1, a2, a3)
    b0 = b0_impact(a1, a2)
    summary = {
        "integrity_gate": gate["status"],
        "a1": a1["classification"],
        "a2": a2["classification"],
        "a3": a3["classification"],
        "joint_diagnosis": diagnosis,
        "impact_on_b0": b0,
        "stage_a_stop": True,
        "no_training_performed": True,
        "test_data_used_for_primary_diagnostics": False,
    }
    dump_json(output / "stage_a_summary.json", summary)
    manifest_files = sorted(path for path in output.rglob("*.json"))
    manifest = {
        "format_version": 1,
        "analysis_seed": ANALYSIS_SEED,
        "analysis_script": "analyse/cora_accuracy_stage_a.py",
        "analysis_script_sha256": sha256_file(repo_root / "analyse/cora_accuracy_stage_a.py"),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_files
        ],
    }
    dump_json(output / "manifest.json", manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cora Stage-A zero-cost diagnostics")
    parser.add_argument(
        "--artifact-root",
        default="legacy_artifacts/pre_20260813/results",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/joint_model_pipeline_global4_best.pth"
    )
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument(
        "--output", default="results/posthoc/cora-accuracy-stage-a-v1"
    )
    parser.add_argument("--initial-git-status-file", default=None)
    parser.add_argument("--integrity-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = (repo_root / args.artifact_root).resolve()
    checkpoint = (repo_root / args.checkpoint).resolve()
    cora_root = (repo_root / args.cora_root).resolve()
    status_file = (
        None if args.initial_git_status_file is None
        else Path(args.initial_git_status_file).resolve()
    )
    try:
        gate, full_records, low_records = integrity_gate(
            repo_root, artifact_root, checkpoint, cora_root, status_file
        )
    except IntegrityFailure as exc:
        print(f"STOP_STAGE_A_INTEGRITY_FAILURE: {exc}")
        return 2
    if args.integrity_only:
        print(json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    a1 = a1_analysis(full_records, checkpoint)
    a2 = a2_analysis(full_records)
    a3 = a3_analysis(full_records, low_records)
    output = (repo_root / args.output).resolve()
    write_outputs(output, gate, a1, a2, a3, repo_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
