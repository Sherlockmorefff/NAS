from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyse import joint_configuration_metric_gate as gate


CONFIG = REPO_ROOT / "configs/joint_configuration_metric_gate_v1.yaml"
PROGRAM = REPO_ROOT / "analyse/joint_configuration_metric_gate.py"


@pytest.fixture
def frozen_config() -> dict:
    pytest.importorskip("yaml")
    return gate.load_config(CONFIG)


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
    pool_id = architecture_index if pool_id is None else pool_id
    observed = seeds or (
        0.700 + index * 0.001,
        0.701 + index * 0.001,
        0.702 + index * 0.001,
    )
    mean = sum(observed) / 3.0
    variance = sum((value - mean) ** 2 for value in observed) / 2.0
    row = {
        "joint_fingerprint": f"joint-{index:04d}",
        "architecture_fingerprint": f"arch-{architecture_index:04d}",
        "hp_fingerprint": f"hp-{hp_index:04d}",
        "operations": [gate.OPERATION_ORDER[architecture_index % len(gate.OPERATION_ORDER)]],
        "edges": [[0, 1], [1, 2]],
        "lr": 10.0 ** (-4.0 + (hp_index % 4) * 0.5),
        "dropout": 0.05 * (hp_index % 10),
        "hidden_dim": 16 * (2 ** (hp_index % 4)),
        "l2": 10.0 ** (-6.0 + (hp_index % 4) * 0.5),
        "pool_id": int(pool_id),
        "validation_seed_0": observed[0],
        "validation_seed_1": observed[1],
        "validation_seed_2": observed[2],
        "validation_mean": mean,
        "validation_sd": math.sqrt(max(variance, 0.0)),
        "validation_se": math.sqrt(max(variance, 0.0)) / math.sqrt(3.0),
    }
    row["raw_hp"] = {
        "lr": row["lr"],
        "dropout": row["dropout"],
        "hidden_dim": row["hidden_dim"],
        "l2": row["l2"],
    }
    if fold is not None:
        row["fold"] = int(fold)
    return row


def _pair_label_lookup(inventory: dict) -> dict[frozenset[str], str]:
    return {
        frozenset((str(pair["left"]), str(pair["right"]))): str(pair["label"])
        for pair in inventory["pairs"]
    }


def test_test_field_is_rejected_at_loader_layer(tmp_path: Path):
    shard = tmp_path / "seed_results.jsonl"
    shard.write_text(json.dumps({"test_accuracy": 0.91}) + "\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="test field rejected at loader layer"):
        gate._read_result_jsonl(shard)


def test_same_architecture_and_all_hp_stay_in_one_outer_fold():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rows = [
        _configuration(
            architecture * 3 + hp,
            architecture_index=architecture,
            hp_index=hp,
            pool_id=(1 if architecture == 0 and hp == 2 else architecture),
        )
        for architecture in range(10)
        for hp in range(3)
    ]
    manifest = gate.build_split_manifest(rows, fold_count=5)
    assert manifest["split_kind"] == "architecture_pool_connected_component"
    gate.assert_split_integrity(rows, manifest)
    for architecture in {row["architecture_fingerprint"] for row in rows}:
        heldout_folds = {
            int(fold["fold"])
            for fold in manifest["folds"]
            if architecture in fold["heldout_architecture_fingerprints"]
        }
        assert len(heldout_folds) == 1
        architecture_rows = [row for row in rows if row["architecture_fingerprint"] == architecture]
        assert all(
            row["joint_fingerprint"]
            in manifest["folds"][next(iter(heldout_folds))]["heldout_joint_fingerprints"]
            for row in architecture_rows
        )


def test_outer_heldout_raw_validation_cannot_enter_triplet_pairs():
    pytest.importorskip("numpy")
    outer_train = [_configuration(index) for index in range(4)]
    heldout = _configuration(99, seeds=(0.99, 0.99, 0.99))
    with pytest.raises(gate.ProtocolError, match="outer-held-out label"):
        gate.construct_statistical_pairs(
            [*outer_train, heldout],
            allowed_joint_fingerprints={row["joint_fingerprint"] for row in outer_train},
        )


def test_pair_labels_use_only_outer_train_matched_seed_differences():
    pytest.importorskip("numpy")
    anchor = _configuration(0, seeds=(0.801, 0.811, 0.821))
    positive = _configuration(1, seeds=(0.800, 0.810, 0.820))
    negative = _configuration(2, seeds=(0.780, 0.790, 0.800))
    ambiguous = _configuration(3, seeds=(0.793, 0.803, 0.813))
    # Deliberately contradictory point summaries prove that pair labels read
    # the three matched training seeds, not validation_mean.
    anchor["validation_mean"] = 0.10
    positive["validation_mean"] = 0.90
    rows = [anchor, positive, negative, ambiguous]
    inventory = gate.construct_statistical_pairs(
        rows,
        allowed_joint_fingerprints={row["joint_fingerprint"] for row in rows},
    )
    labels = _pair_label_lookup(inventory)
    assert labels[frozenset((anchor["joint_fingerprint"], positive["joint_fingerprint"]))] == "positive"
    assert labels[frozenset((anchor["joint_fingerprint"], negative["joint_fingerprint"]))] == "negative"
    assert labels[frozenset((anchor["joint_fingerprint"], ambiguous["joint_fingerprint"]))] == "ambiguous"
    assert inventory["outer_heldout_labels_used"] is False


def test_ambiguous_pairs_never_enter_sampled_triplet_loss():
    inventory = {
        "eligible_anchors": ["anchor"],
        "cross_factor_anchors": ["anchor"],
        "positive_by_anchor": {"anchor": ["positive"]},
        "negative_by_anchor": {"anchor": ["negative"]},
        "cross_factor_by_anchor": {"anchor": ["positive"]},
        "ambiguous_pairs": [{"left": "anchor", "right": "ambiguous"}],
    }
    triplets = gate.sample_triplets(inventory, batch_anchors=9, seed=0)
    assert all(row["positive"] == "positive" and row["negative"] == "negative" for row in triplets)
    assert all("ambiguous" not in row.values() for row in triplets)
    assert sum(bool(row["cross_factor_positive"]) for row in triplets) >= math.ceil(len(triplets) / 2)


def test_raw_target_forbids_hp_residualization_and_has_no_residual_path(frozen_config: dict):
    source = Path(inspect.getsourcefile(gate) or gate.__file__).read_text(encoding="utf-8")
    assert frozen_config["data"]["hp_residualization"] == "forbidden"
    assert frozen_config["data"]["aggregate_target"] == "raw_validation_accuracy_seed_mean"
    assert "fold_local_hp_residuals(" not in source
    assert "choose_hp_control_route(" not in source
    assert "residual_target" not in source


def test_inner_validation_does_not_fit_inner_train_pair_threshold():
    pytest.importorskip("numpy")
    rows = [_configuration(index) for index in range(8)]
    inner_train, inner_validation = gate.inner_architecture_split(rows, seed=0, train_fraction=0.80)
    standardizer = gate.fit_hp_standardizer(rows)
    validation_ids = {row["joint_fingerprint"] for row in inner_validation}
    inventory = gate.construct_statistical_pairs(
        rows,
        allowed_joint_fingerprints={row["joint_fingerprint"] for row in rows},
        hp_standardizer=standardizer,
        hp_median_fit_rows=inner_train,
        anchor_joint_fingerprints=validation_ids,
    )
    fitted_ids = set(inventory["hp_median_fit_joint_fingerprints"])
    assert fitted_ids == {row["joint_fingerprint"] for row in inner_train}
    assert fitted_ids.isdisjoint(validation_ids)


def test_best_epoch_requires_fresh_complete_outer_train_refit(frozen_config: dict):
    assert frozen_config["split"]["refit_on_full_outer_train"] is True
    assert frozen_config["training"]["final_refit_initialization"] == "fresh"
    assert frozen_config["training"]["final_refit_data"] == "full_outer_train"
    assert frozen_config["training"]["final_refit_epochs"] == "best_epoch"
    early_model = object()
    outer_train = [_configuration(index) for index in range(5)]
    calls: list[tuple[object, list[str], int]] = []

    def model_factory():
        return object()

    def train_epoch(model, rows, epoch):
        calls.append((model, [row["joint_fingerprint"] for row in rows], epoch))

    final_model, audit = gate.fresh_full_outer_refit(
        early_model, outer_train, 3, model_factory, train_epoch,
    )
    assert final_model is not early_model
    assert len(calls) == 3
    assert [call[2] for call in calls] == [1, 2, 3]
    assert all(call[0] is final_model for call in calls)
    assert all(call[1] == [row["joint_fingerprint"] for row in outer_train] for call in calls)
    assert audit["fresh_model_for_full_outer_refit"] is True
    assert audit["final_refit_joint_fingerprints"] == sorted(
        row["joint_fingerprint"] for row in outer_train
    )
    assert audit["final_refit_epochs"] == 3


def test_joint_embedding_contains_architecture_hp_and_interaction_components():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    architecture = np.arange(16, dtype=np.float64)[None, :]
    hp = np.full((1, 16), 2.0, dtype=np.float64)
    fused = gate.fusion_contract_probe(architecture, hp)
    assert fused.shape == (1, 48)
    assert np.array_equal(fused[:, :16], architecture)
    assert np.array_equal(fused[:, 16:32], hp)
    assert np.array_equal(fused[:, 32:], architecture * hp)

    rows = [_configuration(0), _configuration(1)]
    standardizer = gate.fit_hp_standardizer(rows)
    metric_class, _regressor_class = gate._joint_model_classes(torch)
    model = metric_class().eval()
    with torch.no_grad():
        embedding, components = model(
            gate._graph_tensors(rows, torch),
            torch.tensor(gate.transform_hp(rows, standardizer), dtype=torch.float32),
            "full_joint",
            True,
        )
    assert embedding.shape == (2, 8)
    assert all(tuple(component.shape) == (2, 16) for component in components)
    assert torch.allclose(components[2], components[0] * components[1])
    assert np.allclose(np.linalg.norm(embedding.numpy(), axis=1), 1.0, atol=1e-5)


def test_m2_gp_is_unstandardized_and_isotropic(
    frozen_config: dict, monkeypatch: pytest.MonkeyPatch,
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    assert frozen_config["metric_methods"]["M2"]["standardize_input_dimensions"] is False
    assert frozen_config["metric_methods"]["M2"]["ard"] is False
    assert frozen_config["metric_methods"]["M2"]["lengthscale_count"] == 1
    monkeypatch.setitem(gate.GP_NUMERICAL_PROTOCOL, "optimization_steps", 0)

    def forbidden_standardization(*_args, **_kwargs):
        raise AssertionError("M2 attempted per-dimension input standardization")

    monkeypatch.setattr(gate, "standardize_train_query", forbidden_standardization)
    train_x = np.eye(8, dtype=np.float64)[:6]
    prediction, audit = gate.exact_matern_gp_predict(
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
    assert audit["input_transform"]["fit_scope"] == "none"
    assert audit["ard"] is False
    assert audit["ard_num_dims"] is None
    assert audit["lengthscale_count"] == 1


def test_m4_cannot_substitute_for_failed_m2(frozen_config: dict):
    predictions, best_of_four = gate._synthetic_predictions(feature_only=True)
    decision = gate.classify_joint_metric_gate(
        336,
        192,
        True,
        gate._synthetic_geometry(True),
        predictions,
        best_of_four,
        frozen_config["decision"],
    )
    assert decision["classification"] == "JOINT_METRIC_FEATURE_ONLY"
    assert decision["feature_only_diagnostics"]["M4"]["passed"] is True
    assert decision["M2_gate"]["passed"] is False
    assert decision["M4_substituted_for_M2"] is False


def test_metric_seed_predictions_are_averaged_and_never_selected():
    np = pytest.importorskip("numpy")
    averaged = gate.average_metric_seed_predictions({
        0: np.asarray([0.0, 3.0]),
        1: np.asarray([3.0, 6.0]),
        2: np.asarray([6.0, 9.0]),
    })
    assert averaged.tolist() == [3.0, 6.0]
    with pytest.raises(gate.ProtocolError, match="exactly metric seeds"):
        gate.average_metric_seed_predictions({0: np.asarray([0.0]), 2: np.asarray([6.0])})


def test_m1_and_m2_gates_use_one_baseline_each_without_metric_splicing(frozen_config: dict):
    predictions, best_of_four = gate._synthetic_predictions()
    decision = gate.classify_joint_metric_gate(
        336,
        192,
        True,
        gate._synthetic_geometry(True),
        predictions,
        best_of_four,
        frozen_config["decision"],
    )
    assert decision["M1_gate"]["single_baseline_model"] == "EJ_KNN"
    assert decision["M2_gate"]["single_baseline_model"] == decision["strongest_non_metric_gp"]
    assert decision["M1_gate"]["cross_model_metric_splicing_used"] is False
    assert decision["M2_gate"]["cross_model_metric_splicing_used"] is False
    assert decision["cross_model_metric_splicing_used"] is False

    # M1 supplies only the MAE threshold while M2 supplies only precision and
    # regret.  Combining those methods would manufacture a pass, so both
    # frozen method-specific gates must remain false.
    spliced_predictions, spliced_best = gate._synthetic_predictions()
    spliced_predictions["M1"] = {"mae": 0.80, "top20_precision": 0.51}
    spliced_best["models"]["M1"]["mean_simple_regret"] = 0.99
    spliced_best["models"]["M1"]["per_fold_mean_simple_regret"] = {
        str(fold): 0.99 for fold in range(5)
    }
    spliced_predictions["M2"] = {"mae": 0.94, "top20_precision": 0.60}
    spliced_decision = gate.classify_joint_metric_gate(
        336, 192, True, gate._synthetic_geometry(True),
        spliced_predictions, spliced_best, frozen_config["decision"],
    )
    assert spliced_decision["M1_gate"]["passed"] is False
    assert spliced_decision["M2_gate"]["passed"] is False
    assert spliced_decision["classification"] not in {
        "JOINT_METRIC_GP_READY", "JOINT_METRIC_LOCALITY_ONLY",
    }


@pytest.mark.parametrize("branch", ["architecture", "hp"])
def test_branch_shuffle_is_deterministic_and_confined_to_each_fold(branch: str):
    rows = [
        _configuration(0, fold=0),
        _configuration(1, fold=0),
        _configuration(2, fold=1),
        _configuration(3, fold=1),
    ]
    shuffled = gate.shuffle_rows_within_fold(rows, branch=branch, seed=17)
    assert shuffled == gate.shuffle_rows_within_fold(rows, branch=branch, seed=17)
    for fold in (0, 1):
        before = [row for row in rows if row["fold"] == fold]
        after = [row for row in shuffled if row["fold"] == fold]
        if branch == "architecture":
            assert sorted(row["architecture_fingerprint"] for row in after) == sorted(
                row["architecture_fingerprint"] for row in before
            )
            assert [row["hp_fingerprint"] for row in after] == [row["hp_fingerprint"] for row in before]
        else:
            assert sorted(row["hp_fingerprint"] for row in after) == sorted(
                row["hp_fingerprint"] for row in before
            )
            assert [row["architecture_fingerprint"] for row in after] == [
                row["architecture_fingerprint"] for row in before
            ]
        assert [row["joint_fingerprint"] for row in after] == [row["joint_fingerprint"] for row in before]
        assert [row["validation_mean"] for row in after] == [row["validation_mean"] for row in before]


def test_nonempty_formal_directory_cannot_be_overwritten(tmp_path: Path):
    output = tmp_path / "formal"
    output.mkdir()
    (output / "occupied.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="non-empty"):
        gate.ensure_output_policy(output)


def test_dry_run_does_not_create_formal_result_directory(tmp_path: Path):
    output = tmp_path / "formal"
    result = subprocess.run(
        [
            sys.executable,
            str(PROGRAM),
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    report = json.loads(result.stdout)
    assert report["formal_result_root_created"] is False
    assert report["formal_data_read"] is False
    assert report["test_data_accessed"] is False


def test_synthetic_pass_reject_underpowered_and_not_joint_classifications(frozen_config: dict):
    predictions, best_of_four = gate._synthetic_predictions()
    passed = gate.classify_joint_metric_gate(
        336, 192, True, gate._synthetic_geometry(True), predictions, best_of_four,
        frozen_config["decision"],
    )
    rejected = gate.classify_joint_metric_gate(
        336, 192, True, gate._synthetic_geometry(False), predictions, best_of_four,
        frozen_config["decision"],
    )
    underpowered = gate.classify_joint_metric_gate(
        336, 192, False, None, None, None, frozen_config["decision"],
    )
    not_joint_predictions, not_joint_best = gate._synthetic_predictions(not_joint=True)
    not_joint = gate.classify_joint_metric_gate(
        336, 192, True, gate._synthetic_geometry(True), not_joint_predictions,
        not_joint_best, frozen_config["decision"],
    )
    insufficient = gate.classify_joint_metric_gate(
        199, 192, None, None, None, None, frozen_config["decision"],
    )
    assert passed["classification"] == "JOINT_METRIC_GP_READY"
    assert rejected["classification"] == "JOINT_METRIC_REJECTED"
    assert underpowered["classification"] == "JOINT_METRIC_UNDERPOWERED"
    assert not_joint["classification"] == "JOINT_METRIC_NOT_JOINT"
    assert insufficient["classification"] == "INSUFFICIENT_JOINT_CONFIG_DATA"
    assert gate.data_gate_decision(336, 192, frozen_config["decision"]) is None
    assert gate.data_gate_decision(199, 192, frozen_config["decision"])["classification"] == (
        "INSUFFICIENT_JOINT_CONFIG_DATA"
    )


def test_launcher_failure_helpers_are_defined_outside_fallback_python():
    launcher = (REPO_ROOT / "scripts/run_joint_configuration_metric_gate.sh").read_text(encoding="utf-8")
    fallback_start = launcher.index("fallback_failure_bundle()")
    heredoc_start = launcher.index("<<'PY'", fallback_start)
    heredoc_end = launcher.index("\nPY\n", heredoc_start)
    quarantine_definition = launcher.index("quarantine_interrupted_package_files()")
    failure_handler = launcher.index("write_failure_and_exit()")
    assert quarantine_definition > heredoc_end
    assert quarantine_definition < failure_handler
    assert "--restart-after-source-fix" in launcher
    assert '[[ "$old_split_sha256" != "null" ]]' in launcher
    assert 'mv "$OUTPUT" "$preserved_output"' in launcher


def test_synthetic_smoke_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    pytest.importorskip("yaml")
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    monkeypatch.setitem(gate.GP_NUMERICAL_PROTOCOL, "optimization_steps", 0)
    assert gate.synthetic_smoke(CONFIG) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["synthetic_pass_case"] is True
    assert report["synthetic_reject_case"] is True
    assert report["synthetic_underpowered_case"] is True
    assert report["synthetic_not_joint_case"] is True
