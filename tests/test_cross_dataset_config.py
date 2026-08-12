from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("botorch")

import bo_phase4
import cross_dataset_cost_estimate
import cross_dataset_runner
import dataset_utils
import final_eval
import method_path_smoke
from initialization_wgmm_ted import candidate_fingerprint, fingerprint_array


def test_all_formal_methods_request_exactly_300_unique_full_evaluations():
    config = cross_dataset_runner.load_and_validate_method_config()

    for method in config["methods"].values():
        assert (
            method["initial_seed_evals"]
            + method["initial_expand_evals"]
            + method["n_iter"]
            == 300
        )
        assert method["planned_unique_full_evals"] == 300
        assert method["max_total_full_evals"] == 300


def test_g100_g150_are_one_method_family_with_same_low_fidelity_semantics():
    methods = cross_dataset_runner.load_and_validate_method_config()["methods"]
    g100 = methods["G100"]
    g150 = methods["G150"]

    assert g100["initial_selection_strategy"] == "wgmm_ted_lowfid"
    assert g150["initial_selection_strategy"] == "wgmm_ted_lowfid"
    for section in ("gmm", "ted", "low_fidelity", "promotion"):
        assert g100[section] == g150[section]
    assert g100["initial_expand_evals"] == 100
    assert g150["initial_expand_evals"] == 150


def test_method_path_smoke_reduces_only_recorded_smoke_budgets(tmp_path):
    command, metadata = method_path_smoke.build_smoke_command(
        python_executable="/env/python",
        checkpoint="/checkpoints/joint.pt",
        data_root="/datasets",
        dataset="citeseer",
        method="G100",
        search_seed=0,
        expected_dataset_manifest=str(tmp_path / "manifest.json"),
        output=tmp_path / "output",
        log_dir=tmp_path / "logs",
        method_config=str(cross_dataset_runner.DEFAULT_METHOD_CONFIG),
    )

    def option(name: str) -> str:
        return command[command.index(f"--{name}") + 1]

    assert metadata["formal_full_budget_unchanged"] == {
        "initial_seed_evals": 50,
        "initial_expand_evals": 100,
        "n_iter": 150,
        "max_total_full_evals": 300,
    }
    assert option("n_lhs_candidates") == "64"
    assert option("initial_seed_evals") == "4"
    assert option("initial_expand_evals") == "1"
    assert option("initial_shortlist_evals") == "8"
    assert option("n_iter") == "1"
    assert option("max_total_full_evals") == "6"
    assert option("low_fidelity_epochs") == "1"
    assert option("surrogate_type") == "exact_gp"
    assert option("online_candidate_strategy") == "qlogei"


def test_training_mode_decision_blocks_unfrozen_dataset(tmp_path):
    path = tmp_path / "training_mode_decisions.json"
    path.write_text(
        json.dumps(
            {
                "datasets": {
                    "flickr": {
                        "status": "frozen",
                        "training_mode": "full_batch",
                        "validated_search_seeds": [0],
                    },
                    "ogbn-arxiv": {
                        "status": "blocked_resource_preflight_oom",
                        "training_mode": None,
                        "reason": "worst candidate OOM",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    accepted = cross_dataset_runner.validate_training_mode_decision(
        path, "flickr", search_seed=0
    )
    assert accepted["training_mode"] == "full_batch"
    with pytest.raises(RuntimeError, match="not frozen"):
        cross_dataset_runner.validate_training_mode_decision(
            path, "ogbn-arxiv"
        )
    with pytest.raises(RuntimeError, match="search_seed=1"):
        cross_dataset_runner.validate_training_mode_decision(
            path, "flickr", search_seed=1
        )


def test_cross_dataset_runner_enforces_isolated_paths_and_real_cli(tmp_path):
    config = cross_dataset_runner.load_and_validate_method_config()
    command, output, log_dir = cross_dataset_runner.build_search_command(
        python_executable="/env/python",
        checkpoint="/checkpoints/joint.pt",
        data_root="/datasets",
        run_tag="cross_v1",
        dataset="CitationFull-DBLP",
        method_key="G100",
        search_seed=5,
        config=config,
        expected_dataset_manifest=(
            tmp_path / "manifests" / "dblp" / "dataset_manifest.json"
        ),
        repo_root=tmp_path,
    )

    expected = (
        Path("cross_v1") / "dblp" / "gmm_exp100" / "search_seed5"
    )
    assert output == tmp_path / "results" / expected
    assert log_dir == tmp_path / "logs" / expected
    assert "--dataset" in command
    assert command[command.index("--dataset") + 1] == "dblp"
    assert command[command.index("--split_seed") + 1] == "0"
    assert command[command.index("--initial_selection_strategy") + 1] == (
        "wgmm_ted_lowfid"
    )
    assert command[command.index("--initial_expand_evals") + 1] == "100"
    assert command[command.index("--n_iter") + 1] == "150"
    assert command[command.index("--max_online_proposal_attempts") + 1] == "32"
    assert "--require_cuda" in command
    assert command[command.index("--expected_dataset_manifest") + 1] == str(
        tmp_path / "manifests" / "dblp" / "dataset_manifest.json"
    )


def test_all_60_formal_commands_parse_and_have_unique_output_paths(
    tmp_path, monkeypatch
):
    config = cross_dataset_runner.load_and_validate_method_config()
    outputs = set()
    logs = set()
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        for method in cross_dataset_runner.METHOD_KEYS:
            for search_seed in config["common"]["formal_search_seeds"]:
                command, output, log_dir = (
                    cross_dataset_runner.build_search_command(
                        python_executable="/env/python",
                        checkpoint="/checkpoints/joint.pt",
                        data_root="/datasets",
                        run_tag="cross_v1",
                        dataset=dataset,
                        method_key=method,
                        search_seed=search_seed,
                        config=config,
                        repo_root=tmp_path,
                    )
                )
                monkeypatch.setattr(
                    sys, "argv", [str(tmp_path / "bo_phase4.py"), *command[2:]]
                )
                parsed = bo_phase4.parse_args()
                bo_phase4.validate_frozen_init_configuration(parsed)
                bo_phase4.validate_two_stage_initialization_config(parsed)
                assert parsed.dataset == dataset
                assert parsed.require_cuda is True
                assert parsed.max_total_full_evals == 300
                assert parsed.max_online_proposal_attempts == 32
                outputs.add(output)
                logs.add(log_dir)

    assert len(outputs) == 60
    assert len(logs) == 60


def test_formal_and_development_search_seed_semantics_are_disjoint():
    config = cross_dataset_runner.load_and_validate_method_config()

    assert config["common"]["search_seeds"] == [0, 1, 2, 3, 4]
    assert config["common"]["development_search_seeds"] == [0, 1, 2, 3, 4]
    assert config["common"]["formal_search_seeds"] == [5, 6, 7, 8, 9]
    with pytest.raises(ValueError, match="formal search_seed"):
        cross_dataset_runner.build_search_command(
            python_executable="/env/python",
            checkpoint="/checkpoints/joint.pt",
            data_root="/datasets",
            run_tag="cross_v1",
            dataset="citeseer",
            method_key="S0",
            search_seed=0,
            config=config,
            repo_root=Path("/tmp/formal-seed-rejection"),
        )


@pytest.mark.parametrize("dataset", ["cora", "ogbn-arxiv"])
def test_formal_runner_rejects_dataset_outside_frozen_matrix(
    dataset, tmp_path
):
    config = cross_dataset_runner.load_and_validate_method_config()
    with pytest.raises(ValueError, match="excluded from the formal matrix"):
        cross_dataset_runner.build_search_command(
            python_executable="/env/python",
            checkpoint="/checkpoints/joint.pt",
            data_root="/datasets",
            run_tag="cross_v1",
            dataset=dataset,
            method_key="S0",
            search_seed=0,
            config=config,
            repo_root=tmp_path,
        )


def test_two_gpu_cost_scheduler_is_deterministic_and_conservative():
    durations = [9.0, 8.0, 7.0, 6.0, 5.0]
    makespan = cross_dataset_cost_estimate._two_gpu_makespan_seconds(
        durations
    )

    assert makespan == 20.0
    assert makespan >= sum(durations) / 2.0
    assert cross_dataset_cost_estimate.TARGET_DATASETS == (
        "citeseer",
        "pubmed",
        "dblp",
        "flickr",
    )


def _pool_args(dataset: str):
    return SimpleNamespace(
        dataset=dataset,
        hp_mode="global4",
        n_lhs_candidates=64,
        n_init=50,
        seed=4,
        sigma_arch=0.8,
        z_bound=2.5,
    )


def test_dataset_does_not_change_latent_candidate_pool_or_candidate_identity():
    cora = torch.stack(bo_phase4.make_lhs_pool(_pool_args("cora")))
    pubmed = torch.stack(bo_phase4.make_lhs_pool(_pool_args("pubmed")))

    assert torch.equal(cora, pubmed)
    assert fingerprint_array(cora.numpy()) == fingerprint_array(pubmed.numpy())
    assert candidate_fingerprint(cora[0].numpy()) == candidate_fingerprint(
        pubmed[0].numpy()
    )


def test_evaluation_seed_is_dataset_and_method_independent():
    fingerprint = "f" * 64

    assert bo_phase4.candidate_evaluation_seed(
        2, fingerprint, "full"
    ) == bo_phase4.candidate_evaluation_seed(2, fingerprint, "full")
    assert bo_phase4.candidate_evaluation_seed(
        2, fingerprint, "low"
    ) != bo_phase4.candidate_evaluation_seed(2, fingerprint, "full")


def _history_record() -> dict:
    return {
        "step": 0,
        "valid": True,
        "val_acc": 0.8,
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 64,
        "l2": 5e-4,
        "candidate_fingerprint": "a" * 64,
        "search_seed": 0,
        "evaluation_fidelity": "full",
    }


def _pubmed_context() -> dict:
    context = dataset_utils.legacy_cora_context()
    context.update(
        {
            "canonical_name": "pubmed",
            "dataset_content_fingerprint": "content-pubmed",
            "split_fingerprint": "split-pubmed",
        }
    )
    return context


def test_final_eval_rejects_explicit_dataset_mismatching_history(tmp_path):
    history = tmp_path / "history_final.json"
    history.write_text(json.dumps([_history_record()]), encoding="utf-8")
    (tmp_path / "history_metadata.json").write_text(
        json.dumps({"dataset_context": _pubmed_context()}),
        encoding="utf-8",
    )
    args = final_eval.parse_args(
        [
            "--history_path",
            str(history),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            "cora",
            "--method_label",
            "schur",
            "--search_seed",
            "0",
            "--final_base_seed",
            "7",
            "--top_k",
            "1",
            "--n_replicates",
            "1",
            "--dry_run",
        ]
    )

    with pytest.raises(ValueError, match="dataset mismatch"):
        final_eval.run_final_evaluation(args, logging.getLogger("mismatch"))


def test_final_eval_infers_dataset_from_history_when_cli_is_implicit(tmp_path):
    history = tmp_path / "history_final.json"
    history.write_text(json.dumps([_history_record()]), encoding="utf-8")
    (tmp_path / "history_metadata.json").write_text(
        json.dumps({"dataset_context": _pubmed_context()}),
        encoding="utf-8",
    )
    args = final_eval.parse_args(
        [
            "--history_path",
            str(history),
            "--output",
            str(tmp_path / "out"),
            "--method_label",
            "schur",
            "--search_seed",
            "0",
            "--final_base_seed",
            "7",
            "--top_k",
            "1",
            "--n_replicates",
            "1",
            "--dry_run",
        ]
    )

    summary = final_eval.run_final_evaluation(
        args, logging.getLogger("infer-dataset")
    )

    assert args.dataset == "pubmed"
    assert summary["dataset_context"]["canonical_name"] == "pubmed"


def test_final_eval_rejects_method_mismatching_history_metadata(tmp_path):
    history = tmp_path / "history_final.json"
    history.write_text(json.dumps([_history_record()]), encoding="utf-8")
    (tmp_path / "history_metadata.json").write_text(
        json.dumps(
            {
                "dataset_context": _pubmed_context(),
                "method_label": "gmm_exp100",
                "search_seed": 0,
            }
        ),
        encoding="utf-8",
    )
    args = final_eval.parse_args(
        [
            "--history_path",
            str(history),
            "--output",
            str(tmp_path / "out"),
            "--method_label",
            "global_schur",
            "--search_seed",
            "0",
            "--final_base_seed",
            "7",
            "--top_k",
            "1",
            "--n_replicates",
            "1",
            "--dry_run",
        ]
    )

    with pytest.raises(ValueError, match="method_label mismatch"):
        final_eval.run_final_evaluation(
            args, logging.getLogger("method-mismatch")
        )
