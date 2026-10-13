"""Offline discrete-predictor replay on frozen Cora search histories.

This analysis never imports the GNN evaluator.  It uses validation accuracy
only, keeps each search history independent, predicts before observing the
current label, and removes repeated architecture/HP configurations after their
first occurrence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a


ANALYSIS_SEED = 20260816
OP_NAMES = ("GCNConv", "GATConv", "SAGEConv", "GINConv", "Identity", "GCNII")
MAX_OPS = 5
MAX_NODES = MAX_OPS + 2
FORBIDDEN_FEATURE_FIELDS = frozenset(
    {"test_acc", "test_accuracy", "test_mean", "mean_test_accuracy"}
)
METHODS = (
    "latent_gp",
    "boosted_tree",
    "predictor_ensemble",
    "top_weighted",
    "pairwise_ranking",
    "ranking_ensemble",
)


class ReplayError(RuntimeError):
    """Raised when a history cannot support a leakage-free replay."""


def _dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _stable_seed(namespace: str, *parts: object) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for part in parts:
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def configuration_key(record: Mapping[str, Any]) -> str:
    """Group equivalent architecture/HP configurations.

    Candidate fingerprints are latent-vector identities.  Architecture plus HP
    is the stricter unit needed to keep repeated configurations from appearing
    in both a training prefix and its later replay evaluation.
    """

    return f"{record['architecture_fingerprint']}:{record['hp_fingerprint']}"


def _paths(record: Mapping[str, Any]) -> list[tuple[str, ...]]:
    operations = [str(value) for value in record["operations"]]
    end = len(operations) + 1
    adjacency: dict[int, list[int]] = defaultdict(list)
    for source, destination in record["edges"]:
        adjacency[int(source)].append(int(destination))
    paths: list[tuple[str, ...]] = []

    def visit(node: int, internal: tuple[int, ...]) -> None:
        if node == end:
            paths.append(tuple(operations[index - 1] for index in internal))
            return
        for destination in sorted(adjacency.get(node, ())):
            if destination == end:
                visit(destination, internal)
            elif 1 <= destination <= len(operations) and destination not in internal:
                visit(destination, (*internal, destination))

    visit(0, ())
    return paths or [tuple()]


def _static_width_mismatch(record: Mapping[str, Any], input_width: int = 1433) -> tuple[int, int]:
    operations = [str(value) for value in record["operations"]]
    predecessors: dict[int, list[int]] = defaultdict(list)
    for source, destination in record["edges"]:
        predecessors[int(destination)].append(int(source))
    hidden = int(record["hidden_dim"])
    widths = {0: int(input_width)}
    mismatch_count = 0
    discarded_columns = 0
    for node in range(1, len(operations) + 2):
        pred_widths = [widths.get(index, hidden) for index in predecessors.get(node, [node - 1])]
        combined = pred_widths[0]
        for width in pred_widths[1:]:
            if width != combined:
                mismatch_count += 1
                discarded_columns += abs(width - combined)
                combined = min(width, combined)
        if node <= len(operations):
            widths[node] = combined if operations[node - 1] == "Identity" else hidden
    return mismatch_count, discarded_columns


def architecture_features(
    record: Mapping[str, Any], *, prior_architecture_count: int = 0
) -> tuple[np.ndarray, list[str]]:
    """Return fixed, explainable discrete architecture features."""

    operations = [str(value) for value in record["operations"]]
    if len(operations) > MAX_OPS or any(operation not in OP_NAMES for operation in operations):
        raise ReplayError(f"unsupported operations: {operations}")
    edges = sorted({(int(source), int(destination)) for source, destination in record["edges"]})
    if any(source < 0 or destination >= MAX_NODES or source >= destination for source, destination in edges):
        raise ReplayError(f"invalid DAG edges: {edges}")

    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float | int | bool) -> None:
        names.append(name)
        values.append(float(value))

    padded = operations + ["<PAD>"] * (MAX_OPS - len(operations))
    for position, observed in enumerate(padded):
        for operation in OP_NAMES:
            add(f"op_position_{position}_{operation}", observed == operation)
    counts = Counter(operations)
    for operation in OP_NAMES:
        add(f"op_count_{operation}", counts[operation])

    edge_set = set(edges)
    for source in range(MAX_NODES):
        for destination in range(source + 1, MAX_NODES):
            add(f"edge_{source}_{destination}", (source, destination) in edge_set)

    predecessor_counts = Counter(destination for _source, destination in edges)
    successor_counts = Counter(source for source, _destination in edges)
    for node in range(1, MAX_NODES):
        add(f"predecessor_count_{node}", predecessor_counts[node])
    add("multi_predecessor_nodes", sum(predecessor_counts[node] > 1 for node in range(1, MAX_NODES)))
    add("max_predecessors", max(predecessor_counts.values(), default=0))
    add("max_successors", max(successor_counts.values(), default=0))

    paths = _paths(record)
    path_lengths = np.asarray([len(path) for path in paths], dtype=float)
    add("input_output_path_count", len(paths))
    add("unique_path_operation_sequences", len(set(paths)))
    add("path_length_min", float(path_lengths.min()))
    add("path_length_mean", float(path_lengths.mean()))
    add("path_length_max", float(path_lengths.max()))
    add("direct_input_output_path", tuple() in paths)
    for operation in OP_NAMES:
        add(
            f"path_operation_occurrences_{operation}",
            sum(path.count(operation) for path in paths),
        )
    for left in OP_NAMES:
        for right in OP_NAMES:
            add(
                f"path_transition_{left}_{right}",
                sum(
                    sum(a == left and b == right for a, b in zip(path, path[1:]))
                    for path in paths
                ),
            )

    effective_depth = sum(operation != "Identity" for operation in operations)
    add("operation_depth", len(operations))
    add("effective_depth", effective_depth)
    add("has_identity", counts["Identity"] > 0)
    add("has_gcnii", counts["GCNII"] > 0)
    add("edge_count", len(edges))
    possible_edges = MAX_NODES * (MAX_NODES - 1) / 2
    add("edge_density", len(edges) / possible_edges)
    add("unique_operation_types", len(counts))
    probabilities = np.asarray(list(counts.values()), dtype=float) / max(len(operations), 1)
    add("operation_entropy", float(-(probabilities * np.log2(probabilities)).sum()))
    mismatch_count, discarded_columns = _static_width_mismatch(record)
    add("mixed_width_predecessor_events", mismatch_count)
    add("mixed_width_discarded_columns", discarded_columns)

    fusion = str(record.get("fusion_type", record.get("combine_type", "sum"))).lower()
    fusion_categories = ("sum", "mean", "concat", "attention")
    for category in fusion_categories:
        add(f"fusion_{category}", fusion == category)
    add("fusion_other", fusion not in fusion_categories)
    residual = record.get("residual", record.get("use_residual", False))
    add("has_residual", bool(residual))
    add("architecture_prior_observations", int(prior_architecture_count))

    return np.asarray(values, dtype=np.float64), names


def hp_features(record: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Return HP features separately from architecture features."""

    lr = float(record["lr"])
    dropout = float(record["dropout"])
    hidden = int(record["hidden_dim"])
    weight_decay = float(record.get("l2", record.get("weight_decay")))
    if lr <= 0 or hidden <= 0 or weight_decay <= 0:
        raise ReplayError("learning rate, hidden dimension, and weight decay must be positive")
    return (
        np.asarray([math.log10(lr), dropout, math.log2(hidden), math.log10(weight_decay)]),
        ["hp_log10_learning_rate", "hp_dropout", "hp_log2_hidden_dimension", "hp_log10_weight_decay"],
    )


def history_feature_matrix(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, int, list[str]]:
    architecture_counts: Counter[str] = Counter()
    rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    architecture_dim = 0
    for record in records:
        arch, arch_names = architecture_features(
            record,
            prior_architecture_count=architecture_counts[str(record["architecture_fingerprint"])],
        )
        hp, hp_names = hp_features(record)
        names = [*arch_names, *hp_names]
        if feature_names is None:
            feature_names = names
            architecture_dim = len(arch_names)
        elif names != feature_names:
            raise ReplayError("feature schema changed within history")
        rows.append(np.concatenate((arch, hp)))
        architecture_counts[str(record["architecture_fingerprint"])] += 1
    return np.stack(rows), architecture_dim, feature_names or []


def first_configuration_indices(records: Sequence[Mapping[str, Any]]) -> list[int]:
    seen: set[str] = set()
    indices: list[int] = []
    for index, record in enumerate(records):
        key = configuration_key(record)
        if key not in seen:
            indices.append(index)
            seen.add(key)
    return indices


def replay_train_cutoff(record: Mapping[str, Any], refit_interval: int) -> int:
    index = int(record["full_evaluation_index"])
    stage = record.get("evaluation_stage")
    if stage == "initial_expand":
        return 50
    if stage == "online_bo":
        return max(50, (index // refit_interval) * refit_interval)
    raise ReplayError(f"record {index} is not a replay prediction row: {stage!r}")


def _fit_tree(x: np.ndarray, y: np.ndarray, seed: int, sample_weight: np.ndarray | None = None):
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=0.01,
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _fit_mlp(x: np.ndarray, y: np.ndarray, seed: int):
    y_mean = float(np.mean(y))
    y_scale = max(float(np.std(y)), 1e-3)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=2e-3, max_iter=250,
            early_stopping=False, random_state=int(seed),
        ),
    )
    model.fit(x, (y - y_mean) / y_scale)
    return model, y_mean, y_scale


def _pair_indices(
    records: Sequence[Mapping[str, Any]], x_arch: np.ndarray, y: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray]:
    n = len(records)
    pairs: dict[tuple[int, int], float] = {}

    def add(left: int, right: int, weight: float) -> None:
        if left == right or y[left] == y[right]:
            return
        pair = (min(left, right), max(left, right))
        pairs[pair] = max(pairs.get(pair, 0.0), float(weight))

    top_threshold = float(np.quantile(y, 0.8))
    top = np.flatnonzero(y >= top_threshold)
    for offset, left in enumerate(top):
        for right in top[offset + 1 :]:
            add(int(left), int(right), 4.0)

    for left in range(n):
        for right in range(left + 1, n):
            if abs(float(y[left] - y[right])) <= 0.02:
                add(left, right, 3.0)

    by_architecture: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_architecture[str(record["architecture_fingerprint"])].append(index)
    for indices in by_architecture.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                add(left, right, 4.0)

    scaled = StandardScaler().fit_transform(x_arch)
    squared = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    for left in range(n):
        neighbors = np.argsort(squared[left], kind="stable")[1:5]
        for right in neighbors:
            add(left, int(right), 2.0)

    ordered = sorted(pairs)
    if len(ordered) > 6000:
        ranked = sorted(ordered, key=lambda pair: (-pairs[pair], pair))[:6000]
        ordered = sorted(ranked)
    weights = np.asarray([pairs[pair] for pair in ordered], dtype=float)
    return ordered, weights


def _fit_pairwise_ranker(
    x: np.ndarray, x_arch: np.ndarray, y: np.ndarray,
    records: Sequence[Mapping[str, Any]], seed: int,
):
    pairs, pair_weights = _pair_indices(records, x_arch, y)
    if len(pairs) < 10:
        return None
    differences: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for (left, right), weight in zip(pairs, pair_weights):
        label = int(y[left] > y[right])
        differences.extend((x[left] - x[right], x[right] - x[left]))
        labels.extend((label, 1 - label))
        weights.extend((float(weight), float(weight)))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=int(seed)),
    )
    model.fit(np.stack(differences), np.asarray(labels), logisticregression__sample_weight=np.asarray(weights))
    return model


def _pairwise_scores(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x_test), float(np.mean(y_train)))
    order = np.argsort(y_train, kind="stable")
    reference_indices = order[np.linspace(0, len(order) - 1, num=min(64, len(order)), dtype=int)]
    references = x_train[reference_indices]
    scores = []
    for row in x_test:
        win_rate = float(model.predict_proba(row[None, :] - references)[:, 1].mean())
        scores.append(float(np.quantile(y_train, np.clip(win_rate, 0.0, 1.0))))
    return np.asarray(scores)


def _predict_snapshot(
    records: Sequence[Mapping[str, Any]], x: np.ndarray, architecture_dim: int,
    train_indices: Sequence[int], test_indices: Sequence[int], *, run_seed: int,
    cutoff: int, mlp_members: int,
) -> dict[str, np.ndarray]:
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    x_train = x[train]
    x_test = x[test]
    y_train = np.asarray([float(records[index]["val_acc"]) for index in train])
    train_records = [records[index] for index in train]

    tree = _fit_tree(x_train, y_train, _stable_seed("tree", run_seed, cutoff))
    tree_prediction = tree.predict(x_test)

    mlp_predictions = []
    for member in range(mlp_members):
        model, mean, scale = _fit_mlp(
            x_train, y_train, _stable_seed("mlp", run_seed, cutoff, member)
        )
        mlp_predictions.append(mean + scale * model.predict(x_test))
    mlp_ensemble = np.mean(np.stack(mlp_predictions), axis=0)
    ensemble = 0.5 * tree_prediction + 0.5 * mlp_ensemble

    top_threshold = float(np.quantile(y_train, 0.8))
    weights = np.where(y_train >= top_threshold, 4.0, 1.0)
    top_tree = _fit_tree(
        x_train, y_train, _stable_seed("top_tree", run_seed, cutoff), weights
    )
    top_prediction = top_tree.predict(x_test)

    ranker = _fit_pairwise_ranker(
        x_train, x_train[:, :architecture_dim], y_train, train_records,
        _stable_seed("pairwise", run_seed, cutoff),
    )
    pairwise_prediction = _pairwise_scores(ranker, x_train, y_train, x_test)
    ranking_ensemble = 0.4 * ensemble + 0.3 * top_prediction + 0.3 * pairwise_prediction
    return {
        "boosted_tree": tree_prediction,
        "predictor_ensemble": ensemble,
        "top_weighted": top_prediction,
        "pairwise_ranking": pairwise_prediction,
        "ranking_ensemble": ranking_ensemble,
    }


def replay_run(
    records: Sequence[Mapping[str, Any]], *, method: str, search_seed: int,
    refit_interval: int, mlp_members: int,
) -> dict[str, Any]:
    if len(records) != 300:
        raise ReplayError(f"{method}/seed{search_seed}: expected 300 records")
    if any(FORBIDDEN_FEATURE_FIELDS.intersection(record) for record in records):
        raise ReplayError(f"{method}/seed{search_seed}: test field found in history")
    for index, record in enumerate(records):
        if int(record["full_evaluation_index"]) != index:
            raise ReplayError(f"{method}/seed{search_seed}: non-monotonic history index")

    x, architecture_dim, feature_names = history_feature_matrix(records)
    first_indices = first_configuration_indices(records)
    first_set = set(first_indices)
    evaluation_indices = [
        index for index in first_indices
        if records[index].get("gp_pred_mean") is not None
    ]
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in evaluation_indices:
        cutoff = replay_train_cutoff(records[index], refit_interval)
        if cutoff > index:
            raise ReplayError(f"{method}/seed{search_seed}: future cutoff at {index}")
        grouped[cutoff].append(index)

    predictions = {name: np.full(len(records), np.nan) for name in METHODS}
    for index in evaluation_indices:
        predictions["latent_gp"][index] = float(records[index]["gp_pred_mean"])

    for cutoff, test_indices in sorted(grouped.items()):
        train_indices = [index for index in first_indices if index < cutoff]
        train_keys = {configuration_key(records[index]) for index in train_indices}
        test_keys = {configuration_key(records[index]) for index in test_indices}
        if train_keys.intersection(test_keys):
            raise ReplayError(f"{method}/seed{search_seed}: configuration leakage at cutoff {cutoff}")
        snapshot = _predict_snapshot(
            records, x, architecture_dim, train_indices, test_indices,
            run_seed=search_seed + 100 * ("S0", "G100", "G150").index(method),
            cutoff=cutoff, mlp_members=mlp_members,
        )
        for name, values in snapshot.items():
            predictions[name][test_indices] = values

    actual = np.asarray([float(records[index]["val_acc"]) for index in evaluation_indices])
    run_metrics = {}
    for name in METHODS:
        predicted = predictions[name][evaluation_indices]
        if not np.isfinite(predicted).all():
            raise ReplayError(f"{method}/seed{search_seed}: non-finite {name} prediction")
        run_metrics[name] = ranking_metrics(actual, predicted)

    return {
        "method": method,
        "search_seed": search_seed,
        "history_records": len(records),
        "unique_configuration_records": len(first_indices),
        "excluded_repeated_configurations": len(records) - len(first_indices),
        "replay_evaluation_records": len(evaluation_indices),
        "refit_snapshots": len(grouped),
        "architecture_feature_dim": architecture_dim,
        "hp_feature_dim": len(feature_names) - architecture_dim,
        "feature_names": feature_names,
        "metrics": run_metrics,
    }


def _safe_kendall(actual: np.ndarray, predicted: np.ndarray) -> float:
    value = float(kendalltau(actual, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def _pairwise_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    correct = 0
    count = 0
    for left in range(len(actual)):
        actual_delta = actual[left] - actual[left + 1 :]
        predicted_delta = predicted[left] - predicted[left + 1 :]
        usable = (actual_delta != 0) & (predicted_delta != 0)
        correct += int(np.sum((actual_delta[usable] > 0) == (predicted_delta[usable] > 0)))
        count += int(np.sum(usable))
    return correct / count if count else 0.5


def _ndcg(actual: np.ndarray, predicted: np.ndarray, k: int = 10) -> float:
    k = min(k, len(actual))
    relevance = actual - actual.min()
    predicted_order = np.argsort(-predicted, kind="stable")[:k]
    ideal_order = np.argsort(-actual, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted_order] * discounts))
    idcg = float(np.sum(relevance[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else 1.0


def ranking_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | int]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if len(actual_array) != len(predicted_array) or len(actual_array) < 50:
        raise ReplayError("ranking metrics require matching arrays with at least 50 rows")
    top_count = max(10, math.ceil(0.2 * len(actual_array)))
    actual_order = np.argsort(-actual_array, kind="stable")
    predicted_order = np.argsort(-predicted_array, kind="stable")
    top_region = actual_order[:top_count]
    actual_top10 = set(actual_order[:10].tolist())
    predicted_top10 = set(predicted_order[:10].tolist())
    late = np.arange(max(0, len(actual_array) - 50), len(actual_array))
    pearson = float(pearsonr(actual_array, predicted_array).statistic)
    return {
        "n": len(actual_array),
        "top20_kendall": _safe_kendall(actual_array[top_region], predicted_array[top_region]),
        "top_region_pairwise_accuracy": _pairwise_accuracy(
            actual_array[top_region], predicted_array[top_region]
        ),
        "top10_recall": len(actual_top10.intersection(predicted_top10)) / 10.0,
        "ndcg_at_10": _ndcg(actual_array, predicted_array, 10),
        "best_of_10_regret": float(actual_array.max() - actual_array[predicted_order[:10]].max()),
        "late50_kendall": _safe_kendall(actual_array[late], predicted_array[late]),
        "rmse": float(np.sqrt(np.mean((actual_array - predicted_array) ** 2))),
        "pearson": pearson if math.isfinite(pearson) else 0.0,
    }


def _bootstrap(values: Sequence[float], seed: int, replicates: int = 4000) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_runs": len(array),
    }


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(runs[0]["metrics"]["latent_gp"])
    aggregate: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        aggregate[method] = {}
        for metric_index, metric in enumerate(metric_names):
            values = [float(run["metrics"][method][metric]) for run in runs]
            aggregate[method][metric] = _bootstrap(
                values, ANALYSIS_SEED + 100 * method_index + metric_index
            )
    return aggregate


def compare_methods(
    runs: Sequence[Mapping[str, Any]], baseline: str, challenger: str
) -> dict[str, Any]:
    higher_is_better = (
        "top20_kendall", "top_region_pairwise_accuracy", "top10_recall",
        "ndcg_at_10", "late50_kendall",
    )
    lower_is_better = ("best_of_10_regret",)
    comparison = {}
    for metric in (*higher_is_better, *lower_is_better):
        raw = [
            float(run["metrics"][challenger][metric])
            - float(run["metrics"][baseline][metric])
            for run in runs
        ]
        oriented = raw if metric in higher_is_better else [-value for value in raw]
        summary = _bootstrap(oriented, _stable_seed("bootstrap", baseline, challenger, metric))
        comparison[metric] = {
            "oriented_improvement": summary,
            "run_wins": sum(value > 0 for value in oriented),
            "run_ties": sum(value == 0 for value in oriented),
            "direction": "higher_is_better" if metric in higher_is_better else "lower_is_better",
        }
    return comparison


def decision_gate(
    discrete: Mapping[str, Any], ranking: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    key_metrics = ("top20_kendall", "top_region_pairwise_accuracy")
    support_metrics = ("top10_recall", "ndcg_at_10", "best_of_10_regret")

    def stable(comparison: Mapping[str, Any], metric: str) -> bool:
        row = comparison[metric]
        return (
            row["oriented_improvement"]["mean"] > 0
            and (
                row["oriented_improvement"]["ci95"][0] > 0
                or row["run_wins"] >= 10
            )
        )

    discrete_key = sum(stable(discrete, metric) for metric in key_metrics)
    discrete_support = sum(
        discrete[metric]["oriented_improvement"]["mean"] > 0
        for metric in support_metrics
    )
    discrete_late = stable(discrete, "late50_kendall")
    discrete_pass = discrete_key == 2 and discrete_support >= 2 and discrete_late

    ranking_key = sum(stable(ranking, metric) for metric in key_metrics)
    ranking_support = sum(
        ranking[metric]["oriented_improvement"]["mean"] > 0
        for metric in support_metrics
    )
    ranking_pass = discrete_pass and ranking_key == 2 and ranking_support >= 2

    mixed_discrete = any(
        6 <= discrete[metric]["run_wins"] <= 9 for metric in key_metrics
    )
    if ranking_pass:
        verdict = "PASS-RANKING"
    elif discrete_pass:
        verdict = "PASS-DISCRETE"
    elif mixed_discrete:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO-GAIN"
    return verdict, {
        "preregistered_rule": (
            "Discrete passes when both top Kendall and top pairwise have positive mean and "
            "either CI>0 or >=10/15 run wins, at least two of recall/nDCG/regret improve, "
            "and late50 Kendall is stable. Ranking passes when the same key/support rule "
            "also holds against the base ensemble. A 6-9/15 split on a key metric is inconclusive."
        ),
        "discrete_pass": discrete_pass,
        "ranking_pass": ranking_pass,
        "mixed_discrete_key_direction": mixed_discrete,
    }


def discover_histories(artifact_root: Path) -> list[tuple[str, int, Path]]:
    rows = []
    for run in stage_a.discover_runs(artifact_root):
        rows.append((run["method"], int(run["search_seed"]), run["path"] / "history_final.json"))
    return rows


def run_experiment(
    artifact_root: Path, *, refit_interval: int = 10, mlp_members: int = 3
) -> dict[str, Any]:
    if refit_interval <= 0 or mlp_members <= 0:
        raise ValueError("refit_interval and mlp_members must be positive")
    runs = []
    source_files = []
    for method, search_seed, path in discover_histories(artifact_root):
        records = stage_a.load_json(path)
        run = replay_run(
            records, method=method, search_seed=search_seed,
            refit_interval=refit_interval, mlp_members=mlp_members,
        )
        runs.append(run)
        source_files.append(
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": stage_a.sha256_file(path)}
        )
        print(f"completed {method}/seed{search_seed}: {run['replay_evaluation_records']} replay rows")

    aggregate = aggregate_runs(runs)
    discrete_comparison = compare_methods(runs, "latent_gp", "predictor_ensemble")
    tree_comparison = compare_methods(runs, "latent_gp", "boosted_tree")
    ranking_comparison = compare_methods(runs, "predictor_ensemble", "ranking_ensemble")
    ranking_vs_gp = compare_methods(runs, "latent_gp", "ranking_ensemble")
    verdict, rule_result = decision_gate(discrete_comparison, ranking_comparison)
    return {
        "format_version": 1,
        "analysis_seed": ANALYSIS_SEED,
        "no_gnn_training_performed": True,
        "training_scope": "offline predictor fitting only; no GNN, NAS, mutation, VAE, or GP fitting",
        "label": "validation accuracy only",
        "test_accuracy_used": False,
        "history_isolation": "S0/G100/G150 x search seeds 0-4 replayed independently",
        "replay_protocol": {
            "initial_seed": "first 50 labels are the first training prefix",
            "initial_expand": "all predictions use the frozen first-50-label model",
            "online_bo": f"prefix-only refit every {refit_interval} observed records",
            "repeat_unit": "architecture_fingerprint + hp_fingerprint; later repeats excluded",
            "future_labels_used": False,
        },
        "models": {
            "boosted_tree": "HistGradientBoostingRegressor, fixed hyperparameters",
            "predictor_ensemble": f"equal mean of boosted tree and {mlp_members}-member small MLP ensemble",
            "top_weighted": "boosted tree with 4x weight for prefix top 20%",
            "pairwise_ranking": "logistic pairwise model emphasizing top20, <=0.02 accuracy gaps, and architecture-local pairs",
            "ranking_ensemble": "0.4 predictor ensemble + 0.3 top-weighted + 0.3 pairwise score",
        },
        "source_files": source_files,
        "population": {
            "histories": len(runs),
            "history_records": sum(int(run["history_records"]) for run in runs),
            "replay_evaluation_records": sum(int(run["replay_evaluation_records"]) for run in runs),
            "excluded_repeated_configurations": sum(int(run["excluded_repeated_configurations"]) for run in runs),
            "architecture_feature_dim": runs[0]["architecture_feature_dim"],
            "hp_feature_dim": runs[0]["hp_feature_dim"],
        },
        "aggregate_across_15_histories": aggregate,
        "comparisons": {
            "boosted_tree_minus_latent_gp": tree_comparison,
            "predictor_ensemble_minus_latent_gp": discrete_comparison,
            "ranking_ensemble_minus_predictor_ensemble": ranking_comparison,
            "ranking_ensemble_minus_latent_gp": ranking_vs_gp,
        },
        "decision_gate": {"verdict": verdict, **rule_result},
        "seed_level": runs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=REPO_ROOT / "legacy_artifacts/pre_20260813/results",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refit-interval", type=int, default=10)
    parser.add_argument("--mlp-members", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    result = run_experiment(
        artifact_root, refit_interval=args.refit_interval, mlp_members=args.mlp_members
    )
    output.mkdir(parents=True)
    _dump_json(output / "offline_predictor_result.json", result)
    _dump_json(
        output / "manifest.json",
        {
            "analysis_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "analysis_script_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "result_sha256": stage_a.sha256_file(output / "offline_predictor_result.json"),
        },
    )
    print(f"verdict: {result['decision_gate']['verdict']}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
