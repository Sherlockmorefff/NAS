"""Stage 2 CPU checks using only synthetic data, never historical labels.

The module-scoped numerical fixture performs exactly four tiny mapping fits and
four tiny GP fits (two steps each). Production fit ledgers are never touched.
"""
import copy
import json
import math
from collections import Counter

import numpy as np
import pytest
import torch

from analyse import fuzzy_metric_offline as offline
from analyse import fuzzy_metric_offline_core as core
from analyse.fuzzy_metric_replay import file_hash, tensor_hash
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel


def label(mean, offset=0.003):
    values = [mean - offset, mean, mean + offset]
    return {"validation_mean": float(np.mean(values)),
            "validation_se": float(np.std(values, ddof=1) / math.sqrt(3)),
            "validation_seeds": values, "training_seeds": [0, 1, 2]}


def candidate(cid, architecture, index=0):
    return {"candidate_id": cid, "joint_fingerprint": cid,
            "architecture_fingerprint": architecture,
            "raw_hp": {"lr": 0.001 * (index + 1), "dropout": 0.1 + 0.02 * index,
                       "hidden_dim": 32 if index % 2 == 0 else 64, "l2": 0.0001 * (index + 1)}}


@pytest.fixture(scope="module")
def partition():
    candidates = [candidate(f"c{arch:03d}-{hp}", f"arch{arch:03d}", hp)
                  for arch in range(120) for hp in range(6)]
    config = {"split_seed": 20260910, "a_architecture_fraction": 0.5,
              "outer_folds": 5, "inner_train_fraction": 0.8,
              "inner_split_seeds": list(range(20260911, 20260917)),
              "future_initial_seeds": list(range(20260920, 20260925)),
              "future_initial_size": 50, "future_query_budget": 100, "future_batch_size": 5}
    frozen = offline.build_partition(candidates, ["c000-0", "c001-2"], config)
    return candidates, config, frozen


def test_architecture_partition_inner_folds_and_p2_capacity(partition):
    candidates, config, frozen = partition
    by_id = {row["candidate_id"]: row for row in candidates}
    architectures = lambda ids: {by_id[cid]["architecture_fingerprint"] for cid in ids}
    assert not set(frozen["A_ids"]) & set(frozen["B_ids"])
    assert set(frozen["A_ids"]) | set(frozen["B_ids"]) == set(by_id)
    assert not architectures(frozen["A_ids"]) & architectures(frozen["B_ids"])
    assert {f"c{arch:03d}-{hp}" for arch in (0, 1) for hp in range(6)} <= set(frozen["A_ids"])
    held_out = []
    for fold in frozen["folds"]:
        held_out.extend(fold["query_ids"])
        assert not architectures(fold["train_ids"]) & architectures(fold["query_ids"])
        assert set(fold["train_ids"]) | set(fold["query_ids"]) == set(frozen["A_ids"])
        inner = fold["inner"]
        assert not architectures(inner["train_ids"]) & architectures(inner["query_ids"])
        assert set(inner["train_ids"]) | set(inner["query_ids"]) == set(fold["train_ids"])
    assert Counter(held_out) == Counter(frozen["A_ids"])
    assert len(frozen["p2_checks"]) == 5
    for check in frozen["p2_checks"]:
        assert check["feasible"] and len(check["initial_ids"]) == 50
        queried = [cid for batch in check["schedule"] for cid in batch]
        assert len(queried) == len(set(queried)) == 100
        assert not set(queried) & set(check["initial_ids"])
        assert max(Counter(by_id[cid]["architecture_fingerprint"] for cid in queried).values()) <= 2
        assert all(len(batch) == len(architectures(batch)) == 5 for batch in check["schedule"])
    assert offline.build_partition(list(reversed(candidates)), ["c001-2", "c000-0"], config) == frozen


def test_grid_matrix_576_unique_and_deterministic(partition):
    grid = offline.configuration_grid()
    assert len(grid) == 28 and len({item["config_id"] for item in grid}) == 28
    assert Counter(item["method"] for item in grid) == {"regression": 1, "N0": 9, "N1": 9, "N2": 9}
    for method in ("N0", "N1", "N2"):
        assert {(g["epsilon"], g["lambda_f"]) for g in grid if g["method"] == method} == {
            (epsilon, weight) for epsilon in (0.005, 0.01, 0.02) for weight in (0.01, 0.1, 1.0)}
    matrix = offline.build_task_matrix(partition[2])
    assert len(matrix) == len({task["task_id"] for task in matrix}) == 576
    assert Counter(task["stage"] for task in matrix) == {"inner": 504, "refit": 72}
    assert Counter(task["seed"] for task in matrix) == {0: 192, 1: 192, 2: 192}
    assert offline.build_task_matrix(copy.deepcopy(partition[2])) == matrix


def selection_rows():
    return [{"setting": setting, "seed": seed,
             "metrics": {"gp_mae": 0.01 if setting["method"] == "regression" else 0.02,
                         "main_neighborhood_metric": 0.02}}
            for setting in offline.configuration_grid() for seed in (0, 1, 2)]


def set_metrics(rows, config_id, maes, primary):
    for row in rows:
        if row["setting"]["config_id"] == config_id:
            row["metrics"] = {"gp_mae": maes[row["seed"]], "main_neighborhood_metric": primary}


def test_selection_uses_three_seed_means_shared_n1_n2_and_frozen_ties():
    rows = selection_rows()
    set_metrics(rows, "N0-e0-w0", [0.009, 0.012, 0.009], 0.003)
    set_metrics(rows, "N0-e0-w1", [0.010] * 3, 0.003)
    # An individually excellent N1 cannot select its cell if N2 fails the gate.
    set_metrics(rows, "N1-e0-w0", [0.010] * 3, 0.001)
    set_metrics(rows, "N1-e1-w1", [0.010] * 3, 0.006)
    set_metrics(rows, "N2-e1-w1", [0.010] * 3, 0.008)
    selected = offline.select_settings(rows)
    assert selected["settings"]["N0"]["config_id"] == "N0-e0-w0"
    assert selected["settings"]["N1"]["config_id"] == "N1-e1-w1"
    assert selected["settings"]["N2"]["config_id"] == "N2-e1-w1"
    assert selected["N0_prediction_gate_satisfied"]
    assert selected["N1_N2_joint_prediction_gate_satisfied"]
    assert selected == offline.select_settings(list(reversed(rows)))


def test_selection_no_eligible_pair_minimizes_total_excess_before_neighbor_metric():
    rows = selection_rows()
    set_metrics(rows, "N0-e1-w0", [0.012] * 3, 0.03)
    set_metrics(rows, "N0-e1-w1", [0.012] * 3, 0.02)
    set_metrics(rows, "N1-e2-w1", [0.012] * 3, 0.030)
    set_metrics(rows, "N2-e2-w1", [0.012] * 3, 0.030)
    set_metrics(rows, "N1-e2-w2", [0.011] * 3, 0.001)
    set_metrics(rows, "N2-e2-w2", [0.020] * 3, 0.001)
    selected = offline.select_settings(rows)
    assert not selected["N0_prediction_gate_satisfied"]
    assert not selected["N1_N2_joint_prediction_gate_satisfied"]
    assert selected["settings"]["N0"]["config_id"] == "N0-e1-w1"
    assert selected["settings"]["N1"]["config_id"] == "N1-e2-w1"
    assert selected["settings"]["N2"]["config_id"] == "N2-e2-w1"
    with pytest.raises(ValueError):
        offline.select_settings(rows[:-1])


def test_allowlisted_label_reader_skips_unselected_invalid_json(tmp_path):
    expected = {"A1": label(0.62), "A2": label(0.63)}
    payload = {**expected, "B_HIDDEN": {"validation_mean": "DO_NOT_PARSE", "validation_se": "DO_NOT_PARSE"}}
    text = json.dumps(payload, indent=2, sort_keys=True).replace('"DO_NOT_PARSE"', 'INVALID_JSON_DO_NOT_PARSE')
    path = tmp_path / "synthetic_labels.json"
    path.write_text(text)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert offline.read_selected_labels(path, {"A1", "A2"}) == expected
    with pytest.raises((ValueError, RuntimeError)):
        offline.read_selected_labels(path, {"MISSING"})
    with pytest.raises(json.JSONDecodeError):
        offline.read_selected_labels(path, {"B_HIDDEN"})


@pytest.fixture(scope="module")
def numerical():
    torch.set_num_threads(1)
    train = [candidate(f"train{i:02d}", f"train_arch{i}", max(0, i - 1)) for i in range(10)]
    query = [candidate(f"query{i:02d}", f"query_arch{i}", i) for i in range(6)]
    rng = np.random.default_rng(18)
    base = {row["architecture_fingerprint"]: rng.normal(size=12).tolist() for row in train + query}
    base["train_arch1"] = base["train_arch0"][:]
    base["query_arch0"] = base["train_arch0"][:]
    base["query_arch5"] = [1000.0] * 12  # Must not enter a training standardizer.
    train_labels = {row["candidate_id"]: label(0.60 + 0.004 * i, 0.001 * (i + 1)) for i, row in enumerate(train)}
    query_labels = {row["candidate_id"]: label(0.62 + 0.003 * i) for i, row in enumerate(query)}
    common = {"representation_steps": 2, "gp_steps": 2, "learning_rate": 0.001,
              "gp_learning_rate": 0.05, "lambda_e": 0.001, "gp_seed": 0}
    grid = {setting["config_id"]: setting for setting in offline.configuration_grid()}
    counts = []
    outputs = {}
    for method in ("N1", "N2"):
        outputs[method] = core.fit_predict(train, query, base, train_labels, grid[f"{method}-e1-w1"],
                                          0, common, on_fit_start=counts.append)
    outputs["repeat"] = core.fit_predict(list(reversed(train)), list(reversed(query)), base, train_labels,
                                         grid["N2-e1-w1"], 0, common, on_fit_start=counts.append)
    outputs["archive"] = core.fit_predict(train, [], base, train_labels, grid["regression"],
                                          0, common, on_fit_start=counts.append)
    assert Counter(counts) == {"mapping": 4, "gp": 4}
    return train, query, base, train_labels, query_labels, outputs


def test_fold_training_standardization_initialization_and_finite_gradients(numerical):
    train, query, base, train_labels, _, outputs = numerical
    first, state1 = outputs["N1"]
    second, state2 = outputs["N2"]
    assert first["initialization_hash"] == second["initialization_hash"]
    assert tensor_hash(state1["model_initial_state"]) == tensor_hash(state2["model_initial_state"])
    assert state1["optimizer"] is not state2["optimizer"]
    scaler = first["standardization"]
    z = torch.tensor([base[c["architecture_fingerprint"]] for c in train], dtype=torch.float32)
    hp = torch.tensor([hp_feature_vector(c) for c in train], dtype=torch.float32)
    np.testing.assert_allclose(scaler["z_mean"], z.mean(0).numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(scaler["z_scale"], z.std(0, correction=0).clamp_min(1e-6).numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(scaler["hp_mean"], hp.mean(0).numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(scaler["hp_scale"], hp.std(0, correction=0).clamp_min(1e-6).numpy(), rtol=0, atol=0)
    assert scaler["fit_ids"] == sorted(train_labels)
    assert scaler["y_mean"] == pytest.approx(np.mean([r["validation_mean"] for r in train_labels.values()]))
    assert not set(scaler["fit_ids"]) & {row["candidate_id"] for row in query}
    for bundle, state in outputs.values():
        assert bundle["diagnostics"]["mapping_gradients_finite"]
        assert bundle["diagnostics"]["gp_gradients_finite"]
        assert torch.equal(state["gp_train_inputs"], state["train_embeddings"].double())
        assert all(torch.isfinite(t).all() for t in state["model"].values())


def test_observation_noise_is_train_se_squared_plus_fitted_noise_once(numerical):
    _, _, _, train_labels, _, outputs = numerical
    bundle, state = outputs["N2"]
    pooled = np.mean([row["validation_se"] ** 2 for row in train_labels.values()])
    normalized = [(train_labels[cid]["validation_se"] / bundle["standardization"]["y_scale"]) ** 2
                  for cid in bundle["train_ids"]]
    np.testing.assert_allclose(state["gp_fixed_noise_normalized"].numpy(), normalized, rtol=1e-12)
    for prediction in bundle["predictions"]:
        assert prediction["pooled_observation_noise_variance"] == pytest.approx(pooled)
        expected = prediction["latent_variance"] + pooled + prediction["learned_noise_variance"]
        assert prediction["observation_variance"] == pytest.approx(expected)
        radius = 1.959963984540054 * math.sqrt(expected)
        assert prediction["observation_interval95"] == pytest.approx([prediction["mean"] - radius, prediction["mean"] + radius])


def test_five_nearest_distances_and_candidate_id_ties(numerical):
    train, query, base, _, _, outputs = numerical
    bundle, state = outputs["N2"]
    model = JointMetricModel(z_dim=12, seed=0)
    model.load_state_dict(state["model"])
    scaler = bundle["standardization"]
    z = torch.tensor([base[c["architecture_fingerprint"]] for c in query], dtype=torch.float32)
    hp = torch.tensor([hp_feature_vector(c) for c in query], dtype=torch.float32)
    with torch.no_grad():
        query_e, _ = model((z - torch.tensor(scaler["z_mean"])) / torch.tensor(scaler["z_scale"]),
                           (hp - torch.tensor(scaler["hp_mean"])) / torch.tensor(scaler["hp_scale"]))
    distances = torch.cdist(query_e.double(), state["train_embeddings"].double()).numpy()
    for i, prediction in enumerate(bundle["predictions"]):
        indices = sorted(range(len(train)), key=lambda j: (distances[i, j], train[j]["candidate_id"]))[:5]
        assert prediction["neighbor_ids"] == [train[j]["candidate_id"] for j in indices]
        np.testing.assert_allclose(prediction["neighbor_distances"], distances[i, indices], atol=1e-12)
    assert bundle["predictions"][0]["neighbor_ids"][:2] == ["train00", "train01"]


def test_query_mean_and_se_perturbation_changes_only_evaluation(numerical):
    _, _, _, train_labels, query_labels, outputs = numerical
    bundle, state = outputs["N2"]
    saved = copy.deepcopy(bundle)
    model_hash, gp_hash = tensor_hash(state["model"]), tensor_hash(state["gp"])
    original = core.evaluate_predictions(bundle, train_labels, query_labels)
    changed_labels = {cid: label(0.85 - 0.001 * i, 0.05) for i, cid in enumerate(query_labels)}
    changed = core.evaluate_predictions(bundle, train_labels, changed_labels)
    assert original["metrics"]["gp_mae"] != changed["metrics"]["gp_mae"]
    assert bundle == saved
    assert tensor_hash(state["model"]) == model_hash and tensor_hash(state["gp"]) == gp_hash
    repeat, repeat_state = outputs["repeat"]
    assert repeat["predictions"] == bundle["predictions"]
    assert repeat["standardization"] == bundle["standardization"]
    assert tensor_hash(repeat_state["model"]) == model_hash and tensor_hash(repeat_state["gp"]) == gp_hash
    with pytest.raises(ValueError, match="exactly"):
        core.evaluate_predictions(bundle, {**train_labels, "HIDDEN": label(0.7)}, query_labels)
    assert not bundle["query_labels_used"] and not bundle["test_labels_used"]


def test_final_archive_and_candidate_once_aggregation(numerical):
    _, _, _, train_labels, query_labels, outputs = numerical
    archive, state = outputs["archive"]
    assert archive["predictions"] == [] and archive["query_ids"] == []
    assert core.evaluate_predictions(archive, train_labels, {})["metrics"]["query_count"] == 0
    assert state["model"] and state["gp"] and state["optimizer"] and state["gp_optimizer"]
    evaluated = core.evaluate_predictions(outputs["N2"][0], train_labels, query_labels)
    left, right = copy.deepcopy(evaluated), copy.deepcopy(evaluated)
    left["per_query"], right["per_query"] = evaluated["per_query"][:1], evaluated["per_query"][1:]
    left["per_query"][0]["neighbor_mean_abs_difference"] = 0.5
    for row in right["per_query"]:
        row["neighbor_mean_abs_difference"] = 0.01
    aggregate = core.aggregate_evaluations([left, right])
    assert aggregate["metrics"]["main_neighborhood_metric"] == pytest.approx(0.01)
    assert aggregate["metrics"]["main_neighborhood_metric"] != pytest.approx((0.5 + 0.01) / 2)
    assert len(aggregate["subgroups"]["posthoc_top10_percent"]["candidate_ids"]) == 1
    with pytest.raises(ValueError):
        core.aggregate_evaluations([left, left])


def test_resume_artifact_identity_and_hard_fit_limit(tmp_path):
    task_dir = tmp_path / "task"
    attempt = task_dir / "attempt01"
    attempt.mkdir(parents=True)
    artifact = attempt / "predictions.json"
    artifact.write_text('{"synthetic": true}\n')
    result = {"identity": "synthetic_identity", "artifact_hashes": {"predictions.json": file_hash(artifact)}}
    (attempt / "completed.json").write_text(json.dumps(result))
    assert offline.completed_result(task_dir, "synthetic_identity") == result
    assert offline.completed_result(task_dir, "synthetic_identity") == result
    with pytest.raises(RuntimeError):
        offline.completed_result(task_dir, "changed_identity")
    artifact.write_text('{"synthetic": false}\n')
    with pytest.raises(RuntimeError):
        offline.completed_result(task_dir, "synthetic_identity")
    # These entries are mock starts in a temporary ledger, not executed fits.
    offline.record_fit_start(tmp_path, "synthetic", 1, "mapping", 2)
    offline.record_fit_start(tmp_path, "synthetic", 2, "mapping", 2)
    with pytest.raises(RuntimeError):
        offline.record_fit_start(tmp_path, "synthetic", 3, "mapping", 2)
    offline.record_fit_start(tmp_path, "synthetic", 1, "gp", 2)
    ledger = [json.loads(line) for line in (tmp_path / "fit_attempts.jsonl").read_text().splitlines()]
    assert Counter(row["kind"] for row in ledger) == {"mapping": 2, "gp": 1}


def test_label_export_resumes_partial_freeze_and_rejects_changed_source(tmp_path, monkeypatch):
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    prepared.mkdir()
    output.mkdir()
    oracle = prepared / "oracle_validation.json"
    expected = {"A1": label(0.61), "A2": label(0.64)}
    payload = {**expected, "B_HIDDEN": {"validation_mean": "MUST_STAY_UNPARSED"}}
    oracle.write_text(json.dumps(payload, indent=2, sort_keys=True).replace(
        '"MUST_STAY_UNPARSED"', 'INVALID_JSON_HIDDEN_B'))
    manifest = {"oracle_sha256": file_hash(oracle)}
    partition = {"A_ids": ["A1", "A2"], "B_ids": ["B_HIDDEN"]}
    offline.finish_label_export(output, prepared, partition, manifest)
    exported = output / "A_validation.json"
    boundary = output / "label_boundary.json"
    exported_hash = file_hash(exported)
    saved_boundary = json.loads(boundary.read_text())
    assert json.loads(exported.read_text()) == expected
    assert saved_boundary["B_parsed_records"] == 0
    assert saved_boundary["exported_ids"] == ["A1", "A2"]
    # Simulate interruption after A export and before boundary persistence.
    boundary.unlink()
    offline.finish_label_export(output, prepared, partition, manifest)
    assert file_hash(exported) == exported_hash
    assert json.loads(boundary.read_text()) == saved_boundary
    # A changed oracle must be rejected before even the allowlisted parser runs.
    oracle.write_text(oracle.read_text() + "\n")
    def forbidden_read(*args, **kwargs):
        pytest.fail("changed source was parsed before its identity was checked")
    monkeypatch.setattr(offline, "read_selected_labels", forbidden_read)
    with pytest.raises(RuntimeError, match="oracle bytes"):
        offline.finish_label_export(output, prepared, partition, manifest)
    assert file_hash(exported) == exported_hash
