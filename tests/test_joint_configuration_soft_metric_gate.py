"""Focused CPU-safe contracts for the soft joint-configuration metric gate.

The tests are deliberately synthetic.  They never call ``execute_formal``,
load the historical validation shards, or train a candidate GNN.  Optional
PyTorch/GPyTorch checks use tiny in-memory inputs only.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyse import joint_configuration_soft_metric_gate as soft


CONFIG = REPO_ROOT / "configs/joint_configuration_soft_metric_gate_v2.yaml"
PROGRAM = REPO_ROOT / "analyse/joint_configuration_soft_metric_gate.py"
FROZEN_SPLIT_SHA256 = "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec"


@pytest.fixture
def frozen_config() -> dict:
    pytest.importorskip("yaml")
    return soft.load_config(CONFIG)


def _configuration(
    index: int,
    *,
    architecture_index: int | None = None,
    hp_index: int | None = None,
    pool_id: int | None = None,
    seeds: tuple[float, float, float] | None = None,
    fold: int | None = None,
) -> dict:
    architecture_index = index if architecture_index is None else architecture_index
    hp_index = index if hp_index is None else hp_index
    observed = seeds or (
        0.700 + index * 0.001,
        0.701 + index * 0.001,
        0.702 + index * 0.001,
    )
    summary = soft.aggregate_three_seed_validation(observed)
    row = {
        "joint_fingerprint": f"joint-{index:04d}",
        "architecture_fingerprint": f"arch-{architecture_index:04d}",
        "hp_fingerprint": f"hp-{hp_index:04d}",
        "operations": [
            soft.joint_v1.OPERATION_ORDER[
                architecture_index % len(soft.joint_v1.OPERATION_ORDER)
            ]
        ],
        "edges": [[0, 1], [1, 2]],
        "lr": 10.0 ** (-4.0 + (hp_index % 4) * 0.5),
        "dropout": 0.05 * (hp_index % 10),
        "hidden_dim": 16 * (2 ** (hp_index % 4)),
        "l2": 10.0 ** (-6.0 + (hp_index % 4) * 0.5),
        "pool_id": int(architecture_index if pool_id is None else pool_id),
        "batch_id": int(index // 4),
        "slot_id": int(index % 4),
        "validation_seed_0": observed[0],
        "validation_seed_1": observed[1],
        "validation_seed_2": observed[2],
        "validation_mean": summary["mean"],
        "validation_sd": summary["sd"],
        "validation_se": summary["se"],
    }
    row["raw_hp"] = {
        "lr": row["lr"], "dropout": row["dropout"],
        "hidden_dim": row["hidden_dim"], "l2": row["l2"],
    }
    if fold is not None:
        row["fold"] = int(fold)
    return row


def _exact_inventory() -> dict:
    return {
        "raw_row_count": 1008,
        "successful_required_seed_row_count": 1008,
        "unique_joint_configurations": 336,
        "unique_architectures": 192,
        "unique_pools": 24,
        "complete_training_seeds": [0, 1, 2],
        "target": "raw_validation_accuracy_seed_mean",
        "hp_residualization_used": False,
    }


def test_exact_main_input_identity_contract_is_336_192_24_and_1008(
    frozen_config: dict,
) -> None:
    """Requirement 1: the gate accepts only the frozen main-data identity."""
    soft.assert_input_identity(_exact_inventory(), frozen_config)
    mutations = {
        "raw_row_count": 1007,
        "successful_required_seed_row_count": 1007,
        "unique_joint_configurations": 335,
        "unique_architectures": 191,
        "unique_pools": 23,
        "complete_training_seeds": [0, 1, 2, 3],
    }
    for field, wrong in mutations.items():
        inventory = _exact_inventory()
        inventory[field] = wrong
        with pytest.raises(soft.InputIdentityMismatch):
            soft.assert_input_identity(inventory, frozen_config)


def test_split_sha_is_exactly_the_joint_v1_frozen_identity(frozen_config: dict) -> None:
    """Requirement 2: v2 points at, rather than regenerates, the v1 split."""
    assert frozen_config["split"]["expected_sha256"] == FROZEN_SPLIT_SHA256
    assert frozen_config["authoritative_inputs"]["joint_metric_v1"]["split_sha256"] == (
        FROZEN_SPLIT_SHA256
    )
    assert frozen_config["split"]["source_manifest"] == (
        "results/posthoc/joint-configuration-metric-gate-global4-v1/split_manifest.json"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"test_accuracy": 0.9},
        {"metrics": {"test": 0.9}},
        {"test_predictions": [1, 2]},
    ],
)
def test_test_fields_are_rejected_at_the_loader_layer(tmp_path: Path, payload: dict) -> None:
    """Requirement 3: rejection happens before any formal aggregation."""
    shard = tmp_path / "seed_results.jsonl"
    shard.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(soft.joint_v1.ProtocolError, match="test field rejected"):
        soft.joint_v1._read_result_jsonl(shard)


def test_hp_residualization_is_forbidden_and_has_no_soft_target_path(
    frozen_config: dict,
) -> None:
    """Requirement 4: only raw validation targets are admitted."""
    assert frozen_config["data"]["hp_residualization"] == "forbidden"
    assert frozen_config["data"]["aggregate_target"] == (
        "raw_validation_accuracy_seed_mean"
    )
    modified = copy.deepcopy(frozen_config)
    modified["data"]["hp_residualization"] = "enabled"
    with pytest.raises(soft.ProtocolError, match="frozen protocol constants changed"):
        soft.validate_frozen_protocol(modified)


def test_raw_validation_mean_sd_and_se_are_exact() -> None:
    """Requirement 5: target is the arithmetic three-seed validation mean."""
    summary = soft.aggregate_three_seed_validation([0.70, 0.71, 0.72])
    assert summary["mean"] == pytest.approx(0.71)
    assert summary["sd"] == pytest.approx(0.01)
    assert summary["se"] == pytest.approx(0.01 / math.sqrt(3.0))


def test_matched_seed_paired_se_is_sample_sd_over_sqrt_three() -> None:
    """Requirement 6: uncertainty uses matched differences, not marginal SEs."""
    left = _configuration(0, seeds=(0.72, 0.75, 0.76))
    right = _configuration(1, seeds=(0.70, 0.71, 0.72))
    statistics = soft.paired_soft_statistics(left, right)
    differences = [0.02, 0.04, 0.04]
    mean = sum(differences) / 3.0
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in differences) / 2.0)
    assert statistics["matched_seed_differences"] == pytest.approx(differences)
    assert statistics["paired_se"] == pytest.approx(sample_sd / math.sqrt(3.0))


def test_soft_similarity_is_monotone_decreasing_in_performance_gap() -> None:
    """Requirement 7: closer raw performance means greater target similarity."""
    gaps = [0.0, 0.006, 0.012, 0.024]
    values = [soft.soft_similarity(gap) for gap in gaps]
    assert all(left > right for left, right in zip(values, values[1:]))
    assert values[0] == pytest.approx(1.0)
    assert values[2] == pytest.approx(math.exp(-0.5))


def test_uncertainty_weight_is_monotone_decreasing_in_paired_se() -> None:
    """Requirement 8: noisy pairs are downweighted without becoming hard labels."""
    values = [soft.uncertainty_weight(se) for se in (0.0, 0.003, 0.012, 0.12)]
    assert all(left > right for left, right in zip(values, values[1:]))


def test_uncertainty_weight_is_always_in_the_frozen_closed_interval() -> None:
    """Requirement 9: every finite pair retains weight in [0.20,1.00]."""
    weights = [soft.uncertainty_weight(se) for se in (0.0, 1e-9, 0.012, 1.0e6)]
    assert all(0.20 <= value <= 1.00 for value in weights)
    assert weights[0] == 1.0
    with pytest.raises(soft.ProtocolError):
        soft.uncertainty_weight(float("inf"))


def test_outer_heldout_cannot_enter_soft_pair_loss() -> None:
    """Requirement 10: the pair constructor enforces an exact train scope."""
    train = [_configuration(0), _configuration(1)]
    heldout = _configuration(99, seeds=(0.99, 0.99, 0.99))
    allowed = {row["joint_fingerprint"] for row in train}
    with pytest.raises(soft.ProtocolError, match="declared training scope"):
        soft.construct_soft_pairs([*train, heldout], allowed_joint_fingerprints=allowed)


def test_inner_validation_never_enters_inner_train_pair_construction() -> None:
    """Requirement 11: train/validation loss pair endpoints stay disjoint."""
    inner_train = [_configuration(index) for index in range(4)]
    inner_validation = [_configuration(index) for index in range(4, 7)]
    fit = soft.construct_soft_pairs(
        inner_train,
        allowed_joint_fingerprints={row["joint_fingerprint"] for row in inner_train},
    )
    validation = soft.construct_soft_pairs(
        inner_validation,
        allowed_joint_fingerprints={row["joint_fingerprint"] for row in inner_validation},
    )
    fit_endpoints = {pair[key] for pair in fit["pairs"] for key in ("left", "right")}
    validation_endpoints = {
        pair[key] for pair in validation["pairs"] for key in ("left", "right")
    }
    assert fit_endpoints.isdisjoint(validation_endpoints)


def test_best_epoch_is_refit_from_scratch_on_complete_outer_train(
    frozen_config: dict,
) -> None:
    """Requirement 12: early-stop weights are never used for final OOF encoding."""
    assert frozen_config["split"]["final_refit_initialization"] == "fresh"
    assert frozen_config["split"]["final_refit_data"] == "full_outer_train"
    early_model = object()
    rows = [_configuration(index) for index in range(5)]
    calls: list[tuple[object, tuple[str, ...], int]] = []

    def factory() -> object:
        return object()

    def epoch(model: object, fit_rows: list[dict], number: int) -> None:
        calls.append((model, tuple(row["joint_fingerprint"] for row in fit_rows), number))

    final_model, audit = soft.joint_v1.fresh_full_outer_refit(
        early_model, rows, 3, factory, epoch,
    )
    assert final_model is not early_model
    assert [call[2] for call in calls] == [1, 2, 3]
    assert all(call[0] is final_model for call in calls)
    assert all(len(call[1]) == len(rows) for call in calls)
    assert audit["fresh_model_for_full_outer_refit"] is True


def test_joint_embedding_contract_contains_architecture_hp_and_interaction() -> None:
    """Requirement 13: fusion is [u_a,v_h,u_a*v_h] and outputs normalized 8D."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    architecture = np.arange(32, dtype=np.float64).reshape(2, 16)
    hp = np.linspace(0.5, 2.0, 32, dtype=np.float64).reshape(2, 16)
    probe = soft.fusion_contract_probe(architecture, hp)
    assert probe.shape == (2, 48)
    assert np.array_equal(probe[:, :16], architecture)
    assert np.array_equal(probe[:, 16:32], hp)
    assert np.array_equal(probe[:, 32:], architecture * hp)

    torch.manual_seed(0)
    model_class, _regressor_class = soft.joint_v1._joint_model_classes(torch)
    model = model_class().eval()
    with torch.no_grad():
        embedding = soft._fuse_components(
            model,
            torch.tensor(architecture, dtype=torch.float32),
            torch.tensor(hp, dtype=torch.float32),
            torch,
        )
    assert tuple(embedding.shape) == (2, 8)
    assert np.allclose(np.linalg.norm(embedding.numpy(), axis=1), 1.0, atol=1e-5)


def test_m1_uses_raw_embedding_euclidean_distance(frozen_config: dict) -> None:
    """Requirement 14: M1 does not standardize the learned embedding."""
    np = pytest.importorskip("numpy")
    assert frozen_config["metric_methods"]["M1"]["distance"] == (
        "raw_embedding_Euclidean"
    )
    assert frozen_config["metric_methods"]["M1"]["per_dimension_standardization"] is False
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    train_y = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    query = np.asarray([[0.25]], dtype=np.float64)
    prediction = soft.joint_v1.weighted_knn_predict(train_x, train_y, query, k=5)
    distances = np.abs(train_x[:, 0] - 0.25)
    weights = 1.0 / (distances + 1e-8)
    assert prediction[0] == pytest.approx(float(np.sum(weights * train_y) / np.sum(weights)))


def test_m2_is_unstandardized_and_isotropic(
    frozen_config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirements 15-16: M2 preserves coordinates and has one lengthscale."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    method = frozen_config["metric_methods"]["M2"]
    assert method["per_dimension_standardization"] is False
    assert method["ard"] is False
    assert method["lengthscale_count"] == 1
    monkeypatch.setitem(soft.joint_v1.GP_NUMERICAL_PROTOCOL, "optimization_steps", 0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("M2 attempted per-dimension standardization")

    monkeypatch.setattr(soft.joint_v1, "standardize_train_query", forbidden)
    train_x = np.eye(8, dtype=np.float64)[:6]
    prediction, audit = soft.joint_v1.exact_matern_gp_predict(
        train_x,
        np.linspace(0.70, 0.80, len(train_x)),
        np.full(len(train_x), 0.002),
        train_x[:2],
        standardize_inputs=False,
        ard=False,
        protocol_name="M2",
    )
    assert prediction.shape == (2,)
    assert audit["input_standardized"] is False
    assert audit["input_exactly_preserved"] is True
    assert audit["ard"] is False
    assert audit["ard_num_dims"] is None
    assert audit["lengthscale_count"] == 1


def test_three_metric_seeds_are_arithmetic_mean_and_never_selected() -> None:
    """Requirement 19: no best-seed selection is possible through this API."""
    np = pytest.importorskip("numpy")
    averaged = soft.average_metric_seed_predictions({
        0: np.asarray([0.0, 3.0]),
        1: np.asarray([3.0, 6.0]),
        2: np.asarray([6.0, 9.0]),
    })
    assert averaged.tolist() == [3.0, 6.0]
    with pytest.raises(soft.ProtocolError, match="exactly metric seeds"):
        soft.average_metric_seed_predictions({0: np.asarray([0.0]), 2: np.asarray([1.0])})


def test_all_models_share_one_best_of_four_batch_manifest(frozen_config: dict) -> None:
    """Requirement 20: batch membership is generated once and shared by models."""
    rows = []
    for fold in range(5):
        for index in range(5):
            row = _configuration(fold * 10 + index, fold=fold)
            row["raw_validation_mean"] = row["validation_mean"]
            row["B0_prediction"] = float(index)
            row["M1_prediction"] = float(4 - index)
            rows.append(row)
    manifest, results = soft.joint_v1.best_of_four_evaluation(
        rows, frozen_config, ["B0", "M1"],
    )
    assert manifest["shared_by_all_models"] is True
    batch_counts = {
        model: sum(row["model"] == model for row in results["rows"])
        for model in ("B0", "M1")
    }
    assert batch_counts["B0"] == batch_counts["M1"]
    memberships = {
        (int(fold["fold"]), index): tuple(batch)
        for fold in manifest["folds"]
        for index, batch in enumerate(fold["batches"])
    }
    assert memberships
    assert all(
        (int(row["fold"]), int(row["batch_index"])) in memberships
        for row in results["rows"]
    )


@pytest.mark.parametrize(
    ("path", "wrong"),
    [
        (("encoder", "output_dim"), 16),
        (("training", "metric_model_seeds"), [0, 1]),
        (("soft_metric", "delta"), 0.010),
        (("soft_metric", "minimum_pair_weight"), 0.10),
        (("metric_methods", "M1", "k"), 7),
        (("metric_methods", "M3", "neighbors"), 16),
        (("training", "max_epochs"), 201),
        (("training", "patience"), 21),
        (("training", "learning_rate"), 2e-3),
        (("training", "weight_decay"), 2e-4),
    ],
)
def test_all_frozen_thresholds_reject_mutation(
    frozen_config: dict, path: tuple[str, ...], wrong: object,
) -> None:
    """Requirement 22: no scientific hyperparameter can be tuned in-place."""
    modified = copy.deepcopy(frozen_config)
    cursor = modified
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = wrong
    with pytest.raises(soft.ProtocolError, match="frozen protocol constants changed"):
        soft.validate_frozen_protocol(modified)


def test_pair_sampling_is_unordered_deterministic_and_label_independent() -> None:
    """Additional core contract: the cap is a uniform seed-0 index sample."""
    rows = [_configuration(index) for index in range(8)]
    allowed = {row["joint_fingerprint"] for row in rows}
    first = soft.construct_soft_pairs(
        rows, allowed_joint_fingerprints=allowed, pair_cap=7, sampling_seed=0,
    )
    altered = [dict(row, validation_seed_0=0.1, validation_seed_1=0.9) for row in rows]
    second = soft.construct_soft_pairs(
        altered, allowed_joint_fingerprints=allowed, pair_cap=7, sampling_seed=0,
    )
    endpoints = [(pair["left_index"], pair["right_index"]) for pair in first["pairs"]]
    assert endpoints == [
        (pair["left_index"], pair["right_index"]) for pair in second["pairs"]
    ]
    assert len(endpoints) == len(set(endpoints)) == 7
    assert all(left < right for left, right in endpoints)
    assert first["performance_aware_sampling_used"] is False


def test_weighted_soft_loss_matches_the_frozen_cosine_smooth_l1_formula() -> None:
    """Additional core contract: weights normalize SmoothL1 of cosine similarity."""
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    inventory = {
        "pairs": [
            {"left_index": 0, "right_index": 1, "soft_similarity": 0.5, "weight": 1.0},
            {"left_index": 0, "right_index": 2, "soft_similarity": 1.0, "weight": 0.5},
        ]
    }
    actual = soft.weighted_soft_metric_loss(embeddings, inventory, torch)
    predicted = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.5, 1.0])
    per_pair = torch.nn.functional.smooth_l1_loss(
        predicted, target, reduction="none", beta=1.0,
    )
    expected = (per_pair[0] + 0.5 * per_pair[1]) / 1.5
    assert float(actual) == pytest.approx(float(expected))


def _synthetic_decision_inputs() -> tuple[dict, dict, dict]:
    geometry = {
        "folds": [
            {
                "fold": fold,
                "seed_distance_gap_spearman": {
                    "0": 0.30, "1": 0.30, "2": 0.30,
                },
            }
            for fold in range(5)
        ],
        "cross_factor_locality_ratio": 0.70,
        "local_gap5_relative_reduction": 0.10,
    }
    predictions = {
        model: {"mae": 1.00, "top20_precision": 0.50}
        for model in soft.MODEL_NAMES
    }
    predictions.update({
        "B0": {"mae": 1.10, "top20_precision": 0.45},
        "B1": {"mae": 0.98, "top20_precision": 0.50},
        "B2": {"mae": 0.97, "top20_precision": 0.50},
        "B3": {"mae": 0.95, "top20_precision": 0.50},
        "B4": {"mae": 0.99, "top20_precision": 0.50},
        "EJ_KNN": {"mae": 1.00, "top20_precision": 0.50},
        "M1": {"mae": 0.90, "top20_precision": 0.56},
        "M2": {"mae": 0.88, "top20_precision": 0.56},
        "M4": {"mae": 0.99, "top20_precision": 0.50},
        "ABL_ARCH_M1": {"mae": 0.98, "top20_precision": 0.50},
        "ABL_HP_M1": {"mae": 0.98, "top20_precision": 0.50},
        "ABL_ARCH_M2": {"mae": 0.95, "top20_precision": 0.50},
        "ABL_HP_M2": {"mae": 0.95, "top20_precision": 0.50},
        "M2_architecture_shuffled": {"mae": 0.92, "top20_precision": 0.52},
        "M2_hp_shuffled": {"mae": 0.92, "top20_precision": 0.52},
    })
    regrets = {
        model: 1.00 for model in soft.MODEL_NAMES
    }
    regrets.update({
        "B0": 1.10, "B1": 0.98, "B2": 0.98, "B3": 1.00,
        "B4": 0.99, "EJ_KNN": 1.00, "M1": 0.85, "M2": 0.85,
        "M4": 0.99, "ABL_ARCH_M1": 0.98, "ABL_HP_M1": 0.98,
        "ABL_ARCH_M2": 0.95, "ABL_HP_M2": 0.95,
        "M2_architecture_shuffled": 0.92, "M2_hp_shuffled": 0.92,
    })
    best_of_four = {
        "models": {
            model: {
                "mean_simple_regret": regret,
                "per_fold_mean_simple_regret": {
                    str(fold): regret for fold in range(5)
                },
            }
            for model, regret in regrets.items()
        }
    }
    return geometry, predictions, best_of_four


def _make_primary_metric_fail(predictions: dict, best_of_four: dict, model: str) -> None:
    predictions[model] = {"mae": 0.99, "top20_precision": 0.51}
    best_of_four["models"][model]["mean_simple_regret"] = 0.99
    best_of_four["models"][model]["per_fold_mean_simple_regret"] = {
        str(fold): 0.99 for fold in range(5)
    }


def test_m4_cannot_replace_m2_to_form_gp_ready(frozen_config: dict) -> None:
    """Requirement 17: an ARD-only improvement is feature-only evidence."""
    geometry, predictions, best = _synthetic_decision_inputs()
    _make_primary_metric_fail(predictions, best, "M1")
    _make_primary_metric_fail(predictions, best, "M2")
    predictions["M4"] = {"mae": 0.88, "top20_precision": 0.56}
    best["models"]["M4"] = {
        "mean_simple_regret": 0.85,
        "per_fold_mean_simple_regret": {str(fold): 0.85 for fold in range(5)},
    }
    decision = soft.classify_soft_joint_metric_gate(
        geometry, predictions, best, frozen_config["decision"],
    )
    assert decision["classification"] == "SOFT_JOINT_METRIC_FEATURE_ONLY"
    assert decision["M4_may_replace_M2"] is False
    assert decision["M2_gate"]["passed"] is False


def test_representation_shuffle_is_seeded_fold_local_and_never_retrained(
    frozen_config: dict,
) -> None:
    """Requirement 18: controls permute only one heldout-fold representation."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    assert frozen_config["controls"]["shuffle_seed"] == 0
    assert frozen_config["controls"]["shuffle_scope"] == (
        "heldout_representations_within_outer_fold_only"
    )
    rows = [_configuration(index, fold=2) for index in range(4)]
    standardizer = soft.joint_v1.fit_hp_standardizer(rows)
    torch.manual_seed(0)
    model_class, _regressor_class = soft.joint_v1._joint_model_classes(torch)
    model = model_class().eval()
    for mode in ("architecture_shuffled", "hp_shuffled"):
        first, first_audit = soft.control_embeddings_from_model(
            model, rows, standardizer, mode=mode, fold=2, seed=0,
        )
        second, second_audit = soft.control_embeddings_from_model(
            model, rows, standardizer, mode=mode, fold=2, seed=0,
        )
        assert np.array_equal(first, second)
        assert first_audit["permutation"] == second_audit["permutation"]
        assert sorted(first_audit["permutation"]) == list(range(len(rows)))
        assert first_audit["scope"] == "within_outer_fold_heldout_representations"
        assert first_audit["reference_embeddings_shuffled"] is False
        assert first_audit["retrained"] is False


def test_30seed_sensitivity_cannot_enter_training_or_change_classification(
    frozen_config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 21: stabilization data is post-classification and descriptive."""
    assert frozen_config["data"]["stabilization_seeds_in_training"] == "forbidden"
    assert frozen_config["sensitivity_30seed"]["may_train_or_select_parameters"] is False
    assert frozen_config["sensitivity_30seed"]["may_change_classification"] is False
    assert "sensitivity" not in inspect.signature(soft.train_soft_metric_encoder).parameters
    assert "sensitivity" not in inspect.signature(soft.classify_soft_joint_metric_gate).parameters

    def unavailable(*_args, **_kwargs):
        raise soft.pair_audit.CalibrationUnavailable("synthetic unavailable")

    monkeypatch.setattr(soft.pair_audit, "load_calibration_subset", unavailable)
    frozen_decision = {"classification": "SOFT_JOINT_METRIC_GEOMETRY_ONLY"}
    before = copy.deepcopy(frozen_decision)
    result = soft.sensitivity_30seed(frozen_config, [], frozen_decision)
    assert frozen_decision == before
    assert result["classification_changed"] is False
    assert result["classification_before"] == result["classification_after"]
    assert result["used_for_training"] is False


def test_frozen_classification_logic_and_underdefined_state(frozen_config: dict) -> None:
    """Requirement 23: state precedence and the uncovered M2-only state are closed."""
    expected = {
        (False, False, False, False, False): "SOFT_JOINT_METRIC_REJECTED",
        (True, False, False, True, False): "SOFT_JOINT_METRIC_GEOMETRY_ONLY",
        (True, True, False, True, False): "SOFT_JOINT_METRIC_LOCALITY_ONLY",
        (True, False, False, True, True): "SOFT_JOINT_METRIC_FEATURE_ONLY",
        (True, True, True, False, False): "SOFT_JOINT_METRIC_NOT_JOINT",
        (True, True, True, True, False): "SOFT_JOINT_METRIC_GP_READY",
    }
    for values, classification in expected.items():
        assert soft.classify_gate_states(
            geometry_passed=values[0], m1_passed=values[1],
            m2_passed=values[2], joint_contribution_passed=values[3],
            feature_diagnostic_passed=values[4],
        ) == classification
    with pytest.raises(soft.SpecUnderdefinedError, match="SPEC_UNDERDEFINED"):
        soft.classify_gate_states(
            geometry_passed=True, m1_passed=False, m2_passed=True,
            joint_contribution_passed=True, feature_diagnostic_passed=False,
        )
    assert frozen_config["decision"]["underdefined_state"]["action"] == (
        "SPEC_UNDERDEFINED_fail_closed"
    )


def test_decision_threshold_boundaries_are_frozen_and_strict(frozen_config: dict) -> None:
    """Requirement 23: strict G1/G3 boundaries cannot pass by equality."""
    geometry, predictions, best = _synthetic_decision_inputs()
    for fold in geometry["folds"]:
        fold["seed_distance_gap_spearman"] = {"0": 0.20, "1": 0.20, "2": 0.20}
    decision = soft.classify_soft_joint_metric_gate(
        geometry, predictions, best, frozen_config["decision"],
    )
    assert decision["classification"] == "SOFT_JOINT_METRIC_REJECTED"
    assert decision["geometry_gate"]["checks"]["G1_median_spearman"] is False

    geometry, predictions, best = _synthetic_decision_inputs()
    geometry["cross_factor_locality_ratio"] = 0.85
    decision = soft.classify_soft_joint_metric_gate(
        geometry, predictions, best, frozen_config["decision"],
    )
    assert decision["classification"] == "SOFT_JOINT_METRIC_REJECTED"
    assert decision["geometry_gate"]["checks"]["G3_cross_factor_locality"] is False


def test_source_snapshot_contains_every_runtime_local_module_and_recomputable_hash(
    tmp_path: Path, frozen_config: dict,
) -> None:
    """Requirement 24: source bytes, not only filenames/hashes, are archived."""
    output = tmp_path / "result"
    manifest = soft.snapshot_runtime_sources(
        output, frozen_config["source_snapshot"]["mandatory_files"],
    )
    verification = soft.pair_audit.verify_source_snapshot(output)
    entries = {row["repository_relative_path"]: row for row in manifest["entries"]}
    assert verification["manifest_root_sha256"] == manifest["manifest_root_sha256"]
    assert {
        "analyse/joint_configuration_soft_metric_gate.py",
        "analyse/joint_configuration_metric_gate.py",
        "analyse/performance_metric_quick_gate.py",
        "tests/test_joint_configuration_soft_metric_gate.py",
    } <= set(entries)
    imported = json.loads(
        (output / "source_snapshot/imported_local_modules.json").read_text(encoding="utf-8")
    )
    imported_paths = {
        row["repository_relative_path"] for row in imported["modules"]
    }
    for module in list(sys.modules.values()):
        source = soft.pair_audit._source_path_for_module(module)
        if source is not None:
            assert source.relative_to(REPO_ROOT.resolve()).as_posix() in imported_paths
    for relative, row in entries.items():
        copied = output / row["snapshot_relative_path"]
        assert copied.read_bytes() == (REPO_ROOT / relative).read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == row["sha256"]
    assert soft.stable_hash(manifest["entries"]) == manifest["manifest_root_sha256"]


def test_final_report_contains_all_seventeen_sections_and_nonempty_tables(
    frozen_config: dict,
) -> None:
    """Requirement 25: the report remains understandable without opening JSON."""
    report = soft.render_final_report({
        "decision": {"classification": "SOFT_JOINT_METRIC_GEOMETRY_ONLY"},
        "input_integrity": {"status": "SYNTHETIC"},
        "data_inventory": {
            "unique_joint_configurations": 336,
            "unique_architectures": 192,
            "unique_pools": 24,
            "metric_training_seeds": [0, 1, 2],
        },
        "split_manifest": {"split_sha256": FROZEN_SPLIT_SHA256},
        "pair_weight_summary": {"folds": [{"fold": 0, "summary": {"selected_pair_count": 1}}]},
        "metric_training_audit": {"runs": [{"fold": 0, "seed": 0, "best_epoch": 1}]},
        "geometry": {"folds": [{"fold": 0}]},
        "knn_locality": {}, "prediction_metric_rows": [],
        "best_of_four": {"models": {}},
        "branch_ablation": {"models": []}, "shuffle_controls": {"models": []},
        "sensitivity": {"records": []}, "environment": {"branch": "synthetic"},
    })
    soft.validate_final_report(report, frozen_config["reporting"]["required_sections"])
    for index, title in enumerate(frozen_config["reporting"]["required_sections"], 1):
        assert f"## {index}. {title}" in report


def test_nonempty_formal_result_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    """Requirement 26: existing bytes are preserved fail-closed."""
    output = tmp_path / "formal"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(soft.ProtocolError, match="non-empty"):
        soft.ensure_output_policy(output)
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_dry_run_does_not_create_formal_result_directory(tmp_path: Path) -> None:
    """Requirement 27: dry-run cannot read formal data or start model training."""
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable, str(PROGRAM), "--config", str(CONFIG),
            "--output", str(output), "--dry-run",
        ],
        cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["formal_data_read"] is False
    assert payload["formal_result_root_created"] is False
    assert payload["model_training_started"] is False
    assert payload["candidate_gnn_training_started"] is False
    assert payload["test_data_accessed"] is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("rejected", "SOFT_JOINT_METRIC_REJECTED"),
        ("geometry_only", "SOFT_JOINT_METRIC_GEOMETRY_ONLY"),
        ("locality_only", "SOFT_JOINT_METRIC_LOCALITY_ONLY"),
        ("feature_only", "SOFT_JOINT_METRIC_FEATURE_ONLY"),
        ("not_joint", "SOFT_JOINT_METRIC_NOT_JOINT"),
        ("gp_ready", "SOFT_JOINT_METRIC_GP_READY"),
    ],
)
def test_all_six_synthetic_classifications(
    frozen_config: dict, case: str, expected: str,
) -> None:
    """Requirement 28: every complete frozen classification has a smoke fixture."""
    geometry, predictions, best = _synthetic_decision_inputs()
    if case == "rejected":
        geometry["cross_factor_locality_ratio"] = 0.90
    elif case == "geometry_only":
        _make_primary_metric_fail(predictions, best, "M1")
        _make_primary_metric_fail(predictions, best, "M2")
    elif case == "locality_only":
        _make_primary_metric_fail(predictions, best, "M2")
    elif case == "feature_only":
        _make_primary_metric_fail(predictions, best, "M1")
        _make_primary_metric_fail(predictions, best, "M2")
        predictions["M4"] = {"mae": 0.88, "top20_precision": 0.56}
        best["models"]["M4"] = {
            "mean_simple_regret": 0.85,
            "per_fold_mean_simple_regret": {str(fold): 0.85 for fold in range(5)},
        }
    elif case == "not_joint":
        predictions["M2_architecture_shuffled"] = dict(predictions["M2"])
        best["models"]["M2_architecture_shuffled"] = copy.deepcopy(best["models"]["M2"])
    decision = soft.classify_soft_joint_metric_gate(
        geometry, predictions, best, frozen_config["decision"],
    )
    assert decision["classification"] == expected


def test_synthetic_smoke_reports_all_six_frozen_cases(
    frozen_config: dict, capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 28: the one-shot preflight exposes every synthetic case."""
    assert soft.synthetic_smoke(CONFIG) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    for case in (
        "rejected", "geometry_only", "locality_only",
        "feature_only", "not_joint", "gp_ready",
    ):
        assert payload[f"synthetic_{case}_case"] is True
