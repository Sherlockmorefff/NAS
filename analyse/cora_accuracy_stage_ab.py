"""Cora decoder-policy triage and conditional offline surrogate replay.

This module performs no GNN training.  It reuses the frozen Stage-A population,
the frozen 128 representative latents, and historical validation labels only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a


ANALYSIS_SEED = 20260816
POLICY_COUNTS = (3, 5, 9, 15, 25)


def _valid_graph(model, graph) -> bool:
    if not graph.is_dag() or graph.vcount() < 2:
        return False
    types = list(graph.vs["type"])
    if types.count(0) != 1 or types.count(1) != 1:
        return False
    start = types.index(0)
    end = types.index(1)
    if end not in set(graph.subcomponent(start, mode="out")):
        return False
    config = stage_a._config_from_graph(model, graph)
    return any(operation != "Identity" for operation in config["operations"])


def _decode_batch(model, z_matrix: torch.Tensor, seed: int, *, stochastic: bool) -> list[dict[str, Any] | None]:
    with stage_a.isolated_rng(seed), torch.no_grad():
        graphs = model.arch_vae.decode(z_matrix, stochastic=stochastic)
    return [
        stage_a._config_from_graph(model, graph) if _valid_graph(model, graph) else None
        for graph in graphs
    ]


def _identity(config: Mapping[str, Any] | None) -> str:
    return "INVALID" if config is None else stage_a._config_fingerprint(config)


def _mode(configs: Sequence[Mapping[str, Any] | None]) -> tuple[str, Mapping[str, Any] | None, float, float]:
    identities = [_identity(config) for config in configs]
    counts = Counter(identities)
    modal_identity, modal_count = sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    modal_config = next(
        (config for config in configs if _identity(config) == modal_identity),
        None,
    )
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(configs)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized_entropy = entropy / math.log(len(configs)) if len(configs) > 1 else 0.0
    return modal_identity, modal_config, modal_count / len(configs), normalized_entropy


def _roundtrip_metrics(
    model,
    representatives: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any] | None],
) -> tuple[list[float], list[float]]:
    latent_distances = []
    architecture_discrepancies = []
    for record, config in zip(representatives, configs):
        if config is None:
            latent_distances.append(1.0)
            architecture_discrepancies.append(1.0)
            continue
        graph = stage_a._graph_from_config(config)
        z = torch.tensor(record["z_search"][: stage_a.ARCH_DIM], dtype=torch.float32)
        with torch.no_grad():
            mu, _logvar = model.arch_vae.encode([graph])
        latent_distances.append(
            float(
                torch.linalg.vector_norm(mu[0] - z).item()
                / (2 * stage_a.Z_BOUND * math.sqrt(stage_a.ARCH_DIM))
            )
        )
        stabilized = {**record, "operations": list(config["operations"]), "edges": list(config["edges"])}
        architecture_discrepancies.append(
            (
                stage_a.op_distance(record, stabilized)
                + stage_a.edge_distance(record, stabilized)
            )
            / 2.0
        )
    return latent_distances, architecture_discrepancies


def _summarize_policy(
    name: str,
    model,
    representatives: Sequence[Mapping[str, Any]],
    first_modes: Sequence[Mapping[str, Any] | None],
    first_probabilities: Sequence[float],
    entropies: Sequence[float],
    half_agreements: Sequence[bool],
    invalid_rates: Sequence[float],
    reproducible: Sequence[bool],
    *,
    sample_count: int | str,
) -> dict[str, Any]:
    roundtrip_latent, architecture_discrepancy = _roundtrip_metrics(
        model, representatives, first_modes
    )
    fingerprints = [_identity(config) for config in first_modes]
    valid_fingerprints = [value for value in fingerprints if value != "INVALID"]
    collision = (
        1.0 - len(set(valid_fingerprints)) / len(valid_fingerprints)
        if valid_fingerprints else 1.0
    )
    return {
        "policy": name,
        "sample_count": sample_count,
        "modal_consistency": float(np.mean(first_probabilities)),
        "normalized_architecture_entropy": float(np.mean(entropies)),
        "independent_half_mode_agreement": float(np.mean(half_agreements)),
        "architecture_fingerprint_reproducibility": float(np.mean(reproducible)),
        "invalid_decode_rate": float(np.mean(invalid_rates)),
        "median_roundtrip_architecture_discrepancy": float(np.median(architecture_discrepancy)),
        "median_roundtrip_latent_distance": float(np.median(roundtrip_latent)),
        "many_z_one_architecture_collision": collision,
        "modal_configs": list(first_modes),
        "modal_fingerprints": fingerprints,
    }


def decoder_policy_comparison(
    records: Sequence[Mapping[str, Any]], checkpoint: Path,
) -> dict[str, Any]:
    model = stage_a.load_vae(checkpoint)
    representatives = stage_a.select_decoder_representatives(records, 128)
    z_matrix = torch.tensor(
        [row["z_search"][: stage_a.ARCH_DIM] for row in representatives],
        dtype=torch.float32,
    )
    sample_pool: list[list[dict[str, Any] | None]] = [
        [] for _ in representatives
    ]
    repeated_pool: list[list[dict[str, Any] | None]] = [
        [] for _ in representatives
    ]
    for draw in range(2 * max(POLICY_COUNTS)):
        seed = stage_a.stable_seed(ANALYSIS_SEED, "decoder_policy_draw", draw)
        decoded = _decode_batch(model, z_matrix, seed, stochastic=True)
        for index, config in enumerate(decoded):
            sample_pool[index].append(config)
        if draw < max(POLICY_COUNTS):
            repeated = _decode_batch(model, z_matrix, seed, stochastic=True)
            for index, config in enumerate(repeated):
                repeated_pool[index].append(config)

    policies: list[dict[str, Any]] = []
    policy_configs: dict[str, list[Mapping[str, Any] | None]] = {}
    for count in POLICY_COUNTS:
        first_modes = []
        probabilities = []
        entropies = []
        agreements = []
        invalid_rates = []
        reproducible = []
        for pool, repeat in zip(sample_pool, repeated_pool):
            first_identity, first_config, probability, entropy = _mode(pool[:count])
            second_identity, _second_config, _probability, _entropy = _mode(
                pool[count : 2 * count]
            )
            repeated_identity, _repeat_config, _repeat_probability, _repeat_entropy = _mode(
                repeat[:count]
            )
            first_modes.append(first_config)
            probabilities.append(probability)
            entropies.append(entropy)
            agreements.append(first_identity == second_identity)
            invalid_rates.append(
                sum(config is None for config in pool[: 2 * count]) / (2 * count)
            )
            reproducible.append(first_identity == repeated_identity)
        name = "current_modal_3" if count == 3 else f"canonical_modal_{count}"
        summary = _summarize_policy(
            name,
            model,
            representatives,
            first_modes,
            probabilities,
            entropies,
            agreements,
            invalid_rates,
            reproducible,
            sample_count=count,
        )
        policies.append(summary)
        policy_configs[name] = first_modes

    greedy_seed = stage_a.stable_seed(ANALYSIS_SEED, "decoder_policy_greedy")
    greedy = _decode_batch(model, z_matrix, greedy_seed, stochastic=False)
    greedy_repeat = _decode_batch(model, z_matrix, greedy_seed, stochastic=False)
    greedy_summary = _summarize_policy(
        "deterministic_greedy",
        model,
        representatives,
        greedy,
        [1.0] * len(greedy),
        [0.0] * len(greedy),
        [True] * len(greedy),
        [float(config is None) for config in greedy],
        [_identity(left) == _identity(right) for left, right in zip(greedy, greedy_repeat)],
        sample_count="deterministic",
    )
    policies.append(greedy_summary)
    policy_configs["deterministic_greedy"] = greedy

    by_count = {int(row["sample_count"]): row for row in policies if isinstance(row["sample_count"], int)}
    agreement_gain_5_9 = (
        by_count[9]["independent_half_mode_agreement"]
        - by_count[5]["independent_half_mode_agreement"]
    )
    agreement_gain_9_15 = (
        by_count[15]["independent_half_mode_agreement"]
        - by_count[9]["independent_half_mode_agreement"]
    )
    entropy_drop_5_9 = (
        by_count[5]["normalized_architecture_entropy"]
        - by_count[9]["normalized_architecture_entropy"]
    )
    entropy_drop_9_15 = (
        by_count[9]["normalized_architecture_entropy"]
        - by_count[15]["normalized_architecture_entropy"]
    )
    include_25 = (
        agreement_gain_9_15 > max(0.0, 0.5 * agreement_gain_5_9)
        or entropy_drop_9_15 > max(0.0, 0.5 * entropy_drop_5_9)
    )
    if not include_25:
        policies = [row for row in policies if row["sample_count"] != 25]

    def selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        return (
            float(row["invalid_decode_rate"]),
            -float(row["independent_half_mode_agreement"]),
            float(row["normalized_architecture_entropy"]),
            float(row["many_z_one_architecture_collision"]),
            float(row["median_roundtrip_architecture_discrepancy"]),
            float(row["median_roundtrip_latent_distance"]),
        )

    best = min(policies, key=selection_key)
    serializable_policies = []
    for row in policies:
        serializable_policies.append(
            {key: value for key, value in row.items() if key != "modal_configs"}
        )
    return {
        "representative_latents": len(representatives),
        "representative_fingerprints": [
            row["candidate_fingerprint"] for row in representatives
        ],
        "policies": serializable_policies,
        "sample_25_triggered": include_25,
        "convergence": {
            "agreement_gain_5_to_9": agreement_gain_5_9,
            "agreement_gain_9_to_15": agreement_gain_9_15,
            "entropy_drop_5_to_9": entropy_drop_5_9,
            "entropy_drop_9_to_15": entropy_drop_9_15,
        },
        "best_policy": best["policy"],
        "best_policy_configs": policy_configs[best["policy"]],
        "representatives": representatives,
        "current_policy_identity_semantics": (
            "Current _decode_arch already aggregates valid decodes by final operations+edges "
            "architecture identity; canonicalized_modal_3 is therefore equivalent to current_modal_3."
        ),
        "greedy_api_available": True,
    }


def decode_policy_for_population(
    records: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    policy: str,
    *,
    batch_size: int = 256,
) -> list[Mapping[str, Any] | None]:
    """Apply the selected decoder policy to every deduplicated candidate."""

    model = stage_a.load_vae(checkpoint)
    sample_count = None
    stochastic = policy != "deterministic_greedy"
    if stochastic:
        if not policy.startswith("canonical_modal_") and policy != "current_modal_3":
            raise ValueError(f"unsupported decoder policy: {policy}")
        sample_count = 3 if policy == "current_modal_3" else int(policy.rsplit("_", 1)[1])

    output: list[Mapping[str, Any] | None] = []
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch = records[start : start + batch_size]
        z_matrix = torch.tensor(
            [row["z_search"][: stage_a.ARCH_DIM] for row in batch],
            dtype=torch.float32,
        )
        if not stochastic:
            seed = stage_a.stable_seed(
                ANALYSIS_SEED, "population_decoder_greedy", batch_index
            )
            output.extend(_decode_batch(model, z_matrix, seed, stochastic=False))
            continue

        pools: list[list[Mapping[str, Any] | None]] = [[] for _ in batch]
        assert sample_count is not None
        for draw in range(sample_count):
            seed = stage_a.stable_seed(
                ANALYSIS_SEED, "population_decoder_modal", policy, batch_index, draw
            )
            decoded = _decode_batch(model, z_matrix, seed, stochastic=True)
            for row_index, config in enumerate(decoded):
                pools[row_index].append(config)
        output.extend(_mode(pool)[1] for pool in pools)

    if len(output) != len(records):
        raise RuntimeError(
            f"population decode cardinality mismatch: {len(output)} != {len(records)}"
        )
    return output


def _stabilized_records(
    records: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    if len(records) != len(configs):
        raise ValueError(
            f"record/config cardinality mismatch: {len(records)} != {len(configs)}"
        )
    stabilized = []
    for row, config in zip(records, configs):
        if config is None:
            stabilized.append(
                {
                    **row,
                    "operations": ["<INVALID>"],
                    "edges": [],
                    "architecture_fingerprint": "INVALID",
                }
            )
        else:
            stabilized.append(
                {
                    **row,
                    "operations": list(config["operations"]),
                    "edges": [list(edge) for edge in config["edges"]],
                    "architecture_fingerprint": stage_a._config_fingerprint(config),
                }
            )
    return stabilized


def geometry_after_stabilization(
    deduplicated: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any] | None],
    stage_a_result: Mapping[str, Any],
) -> dict[str, Any]:
    records = _stabilized_records(deduplicated, configs)
    top_threshold = float(np.quantile([row["val_acc"] for row in records], 0.8))
    top = [row for row in records if float(row["val_acc"]) >= top_threshold]
    regions = {
        "hp_near": stage_a._geometry_metrics(
            stage_a._pair_rows(
                records,
                stage_a._nearest_pairs(
                    records, slice(stage_a.ARCH_DIM, stage_a.ARCH_DIM + stage_a.HP_DIM), 6
                ),
            ),
            ANALYSIS_SEED + 1,
        ),
        "latent_local": stage_a._geometry_metrics(
            stage_a._pair_rows(
                records,
                stage_a._nearest_pairs(records, slice(0, stage_a.ARCH_DIM), 6),
            ),
            ANALYSIS_SEED + 2,
        ),
        "top_region": stage_a._geometry_metrics(
            stage_a._pair_rows(
                top,
                stage_a._random_pair_indices(len(top), 50000, ANALYSIS_SEED + 3),
            ),
            ANALYSIS_SEED + 3,
        ),
    }
    comparisons = {}
    shrinkages = []
    for region, metrics in regions.items():
        old = stage_a_result["regions"][region]["discrete_minus_latent_effect"]["estimate"]
        new = metrics["discrete_minus_latent_effect"]["estimate"]
        shrinkage = None if old == 0 else (abs(old) - abs(new)) / abs(old)
        comparisons[region] = {
            "stage_a_effect": old,
            "stabilized_effect": new,
            "relative_absolute_gap_shrinkage": shrinkage,
        }
        if shrinkage is not None:
            shrinkages.append(shrinkage)
    median_shrinkage = float(np.median(shrinkages))
    case = "CASE_A" if median_shrinkage >= 0.5 else "CASE_B"
    verdict = (
        "DECODE_POLICY_FIX_SUFFICIENT"
        if case == "CASE_A"
        else "LATENT_GEOMETRY_STILL_FAILS"
    )
    return {
        "best_policy_geometry": regions,
        "comparison_to_stage_a": comparisons,
        "median_relative_absolute_gap_shrinkage": median_shrinkage,
        "case": case,
        "representation_verdict": verdict,
        "label_attribution_caveat": (
            "Historical validation labels were produced by the historical current-policy "
            "architecture. Alternative-policy architecture distances are a zero-training "
            "counterfactual geometry diagnostic, not relabeled GNN performance."
        ),
    }


def _kernel_predictions(
    history: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    predictions = {"architecture_kernel": [], "architecture_hp_additive": [], "architecture_hp_product": []}
    for target in targets:
        index = int(target["full_evaluation_index"])
        if target.get("evaluation_stage") == "initial_expand":
            training = [row for row in history if int(row["full_evaluation_index"]) < 50]
        else:
            training = [row for row in history if int(row["full_evaluation_index"]) < index]
        labels = np.asarray([row["val_acc"] for row in training], dtype=float)
        arch_dist = np.asarray(
            [
                (stage_a.op_distance(target, row) + stage_a.edge_distance(target, row)) / 2.0
                for row in training
            ],
            dtype=float,
        )
        hp_target = np.asarray(target["z_search"][stage_a.ARCH_DIM :], dtype=float)
        hp_dist = np.asarray(
            [
                np.linalg.norm(
                    hp_target - np.asarray(row["z_search"][stage_a.ARCH_DIM :], dtype=float)
                ) / math.sqrt(stage_a.HP_DIM)
                for row in training
            ],
            dtype=float,
        )

        def kernel(distances: np.ndarray) -> np.ndarray:
            positive = distances[distances > 0]
            bandwidth = float(np.median(positive)) if positive.size else 1.0
            bandwidth = max(bandwidth, 1e-6)
            return np.exp(-0.5 * (distances / bandwidth) ** 2)

        architecture_weight = kernel(arch_dist)
        hp_weight = kernel(hp_dist)
        weights = {
            "architecture_kernel": architecture_weight,
            "architecture_hp_additive": 0.5 * (architecture_weight + hp_weight),
            "architecture_hp_product": architecture_weight * hp_weight,
        }
        for name, value in weights.items():
            denominator = float(value.sum())
            predictions[name].append(
                float(np.dot(value, labels) / denominator)
                if denominator > 0
                else float(labels.mean())
            )
    return predictions


def _decision_metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]
) -> dict[str, Any]:
    actual = np.asarray([row["val_acc"] for row in rows], dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    top = actual >= np.quantile(actual, 0.8)
    local_pairs = stage_a._architecture_local_pairs(rows, 6)
    synthetic = [dict(row, gp_pred_mean=float(value)) for row, value in zip(rows, predicted)]
    local_accuracy, local_count = stage_a.pairwise_accuracy(synthetic, local_pairs)
    selected = np.argsort(-predicted, kind="stable")[:10]
    late_actual = actual[-50:]
    late_prediction = predicted[-50:]
    return {
        "top_region_kendall": stage_a._safe_kendall(predicted[top], actual[top]),
        "architecture_local_pairwise_accuracy": local_accuracy,
        "architecture_local_pairs": local_count,
        "ndcg_at_10": stage_a.ndcg_score(actual, predicted, 10),
        "top10_regret": float(actual.max() - actual[selected].max()),
        "late50_kendall": stage_a._safe_kendall(late_prediction, late_actual),
    }


def offline_surrogate_replay(full_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    run_results = []
    model_names = (
        "current_latent_gp",
        "architecture_kernel",
        "architecture_hp_additive",
        "architecture_hp_product",
    )
    for method in ("S0", "G100", "G150"):
        for seed in range(5):
            history = sorted(
                [
                    row for row in full_records
                    if row["_method"] == method and row["_run_seed"] == seed
                ],
                key=lambda row: int(row["full_evaluation_index"]),
            )
            targets = [row for row in history if row.get("gp_pred_mean") is not None]
            stage_a.assert_prefix_only(targets, f"offline/{method}/seed{seed}")
            predictions = _kernel_predictions(history, targets)
            predictions["current_latent_gp"] = [float(row["gp_pred_mean"]) for row in targets]
            for model_name in model_names:
                metrics = _decision_metrics(targets, predictions[model_name])
                run_results.append(
                    {
                        "method": method,
                        "search_seed": seed,
                        "model": model_name,
                        **metrics,
                    }
                )
    fields = (
        "top_region_kendall",
        "architecture_local_pairwise_accuracy",
        "ndcg_at_10",
        "top10_regret",
        "late50_kendall",
    )
    aggregate = {}
    for model_index, model_name in enumerate(model_names):
        model_rows = [row for row in run_results if row["model"] == model_name]
        aggregate[model_name] = {
            field: stage_a.bootstrap_mean_ci(
                [row[field] for row in model_rows if row[field] is not None],
                seed=ANALYSIS_SEED + 100 * model_index + field_index,
            )
            for field_index, field in enumerate(fields)
        }
    deltas = {}
    current = [row for row in run_results if row["model"] == "current_latent_gp"]
    for model_index, model_name in enumerate(model_names[1:], start=1):
        other = [row for row in run_results if row["model"] == model_name]
        deltas[model_name] = {}
        for field_index, field in enumerate(fields):
            values = [
                float(candidate[field]) - float(baseline[field])
                for baseline, candidate in zip(current, other)
                if baseline[field] is not None and candidate[field] is not None
            ]
            if field == "top10_regret":
                values = [-value for value in values]
            deltas[model_name][field] = stage_a.bootstrap_mean_ci(
                values,
                seed=ANALYSIS_SEED + 1000 + 100 * model_index + field_index,
            )
    qualifying = []
    for model_name, model_delta in deltas.items():
        ranking_ci = model_delta["top_region_kendall"]["ci95"]
        local_ci = model_delta["architecture_local_pairwise_accuracy"]["ci95"]
        regret_estimate = model_delta["top10_regret"]["estimate"]
        if (
            ranking_ci[0] is not None
            and local_ci[0] is not None
            and ranking_ci[0] > 0
            and local_ci[0] > 0
            and regret_estimate >= 0
        ):
            qualifying.append(model_name)
    return {
        "status": "COMPLETED_CASE_B",
        "protocol": (
            "strict historical prefix-only offline replay; initial_expand uses frozen first-50 "
            "labels and online_bo uses labels with lower full_evaluation_index"
        ),
        "aggregate": aggregate,
        "paired_delta_vs_current_latent_gp": deltas,
        "structure_aware_candidates_for_next_stage": qualifying,
        "decision_relevant_improvement": bool(qualifying),
        "seed_level": run_results,
        "formal_gp_modified": False,
        "new_nas_run": False,
    }


def dependency_audit() -> dict[str, Any]:
    return {
        "B0_INDEPENDENT_OF_DECODER": True,
        "evidence": [
            "final_eval.evaluate_candidate constructs DynamicGNN config directly from frozen operations and edges",
            "train_and_eval_arch receives frozen lr/dropout/hidden_dim/l2 and the already loaded frozen Cora split",
            "the training call has no latent z, VAE, stochastic decoder, GP, BO, or LF dependency",
        ],
        "direct_call_chain": "frozen candidate manifest -> config/HP -> train_and_eval_arch -> DynamicGNN",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cora Stage-AB decoder and surrogate triage")
    parser.add_argument(
        "--artifact-root", default="legacy_artifacts/pre_20260813/results"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/joint_model_pipeline_global4_best.pth"
    )
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument(
        "--stage-a-root", default="results/posthoc/cora-accuracy-stage-a-v1"
    )
    parser.add_argument(
        "--output", default="results/posthoc/cora-accuracy-decoder-b0-v1/track_a"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_root = (REPO_ROOT / args.artifact_root).resolve()
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    cora_root = (REPO_ROOT / args.cora_root).resolve()
    stage_a_root = (REPO_ROOT / args.stage_a_root).resolve()
    output = (REPO_ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    gate, full_records, _low_records = stage_a.integrity_gate(
        REPO_ROOT, artifact_root, checkpoint, cora_root, None
    )
    deduplicated, deduplication = stage_a.deduplicate_candidate_population(full_records)
    comparison = decoder_policy_comparison(deduplicated, checkpoint)
    population_configs = decode_policy_for_population(
        deduplicated, checkpoint, comparison["best_policy"]
    )
    stage_a_result = stage_a.load_json(
        stage_a_root / "latent_geometry/a1_result.json"
    )
    geometry = geometry_after_stabilization(
        deduplicated,
        population_configs,
        stage_a_result,
    )
    surrogate = (
        offline_surrogate_replay(full_records)
        if geometry["case"] == "CASE_B"
        else {"status": "NOT_TRIGGERED"}
    )
    output.mkdir(parents=True)
    stage_a.dump_json(output / "dependency_audit.json", dependency_audit())
    policy_output = {
        key: value
        for key, value in comparison.items()
        if key not in ("best_policy_configs", "representatives")
    }
    stage_a.dump_json(output / "decoder_policy_results.json", _json_safe(policy_output))
    stage_a.dump_json(output / "representation_verdict.json", _json_safe(geometry))
    stage_a.dump_json(output / "offline_surrogate_result.json", _json_safe(surrogate))
    summary = {
        "integrity": gate["status"],
        "dependency_audit": dependency_audit(),
        "deduplication": deduplication,
        "best_decoder_policy": comparison["best_policy"],
        "representation_verdict": geometry["representation_verdict"],
        "offline_surrogate_status": surrogate["status"],
        "no_gnn_training": True,
    }
    stage_a.dump_json(output / "track_a_summary.json", _json_safe(summary))
    files = sorted(output.glob("*.json"))
    manifest = {
        "format_version": 1,
        "analysis_seed": ANALYSIS_SEED,
        "script_sha256": stage_a.sha256_file(Path(__file__).resolve()),
        "files": [
            {
                "path": path.name,
                "sha256": stage_a.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    stage_a.dump_json(output / "manifest.json", manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
