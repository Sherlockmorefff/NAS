from __future__ import annotations

import sys
from pathlib import Path

import cross_dataset_runner
import deterministic_three_strategy_pipeline as pipeline
import formal_matrix
from source_freeze import SOURCE_GATE_DISCOVERY_RULES


def test_formal_search_command_uses_isolated_artifact_root_and_real_cli(tmp_path: Path) -> None:
    config = cross_dataset_runner.load_and_validate_method_config()
    command, output, log_dir = cross_dataset_runner.build_search_command(
        python_executable=sys.executable,
        checkpoint="/tmp/checkpoint.pth",
        data_root="/tmp/data",
        run_tag="deterministic_test",
        dataset="citeseer",
        method_key="G100",
        search_seed=5,
        config=config,
        repo_root=cross_dataset_runner.ROOT,
        artifact_root=tmp_path,
        formal_source_id="a" * 64,
        formal_config_fingerprint="b" * 64,
    )
    assert output == tmp_path / "results" / "deterministic_test" / "citeseer" / "gmm_exp100" / "search_seed5"
    assert log_dir == tmp_path / "logs" / "deterministic_test" / "citeseer" / "gmm_exp100" / "search_seed5"
    parsed = cross_dataset_runner.validate_phase4_command(command)
    assert parsed.formal_source_id == "a" * 64
    assert parsed.formal_config_fingerprint == "b" * 64
    assert parsed.initial_seed_evals == 50
    assert parsed.initial_expand_evals == 100
    assert parsed.n_iter == 150
    assert parsed.n_lhs_candidates == 768
    assert parsed.max_online_proposal_attempts == 32


def test_synthetic_formal_matrix_audit_is_exactly_60_and_18000() -> None:
    tasks = []
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        for method in cross_dataset_runner.METHOD_KEYS:
            for seed in cross_dataset_runner.FORMAL_SEARCH_SEEDS:
                task_id = f"{dataset}__{method}__search_seed{seed}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "method": method,
                        "search_seed": seed,
                        "output_directory": f"/new/results/{task_id}",
                        "log_directory": f"/new/logs/{task_id}",
                        "candidate_budget": 300,
                        "n_lhs_candidates": 768,
                        "exact_argv": ["python", "cross_dataset_runner.py", "--execute"],
                        "status": "not_started",
                    }
                )
    audit = formal_matrix.audit_formal_manifest({"tasks": tasks})
    assert audit["task_count"] == 60
    assert audit["total_requested_full_evaluations"] == 18_000
    assert audit["by_dataset_method"] == {
        f"{dataset}__{method}": 5
        for dataset in cross_dataset_runner.FORMAL_DATASETS
        for method in cross_dataset_runner.METHOD_KEYS
    }


def test_builder_generates_matrix_from_self_contained_fixtures(
    tmp_path: Path,
) -> None:
    identity = pipeline.current_source_identity()
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fixture checkpoint\n")
    checkpoint = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": pipeline.sha256_file(checkpoint_path),
    }
    source_manifest = tmp_path / "source_manifest.json"
    pipeline.atomic_json(
        source_manifest,
        {
            "format_version": 1,
            "source_id": identity["source_id"],
            "source_gate": {"discovery_rules": SOURCE_GATE_DISCOVERY_RULES},
            "checkpoint": checkpoint,
            "files": identity["formal_source_rows"],
        },
    )
    manifest_root = tmp_path / "dataset_manifests"
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        pipeline.atomic_json(
            manifest_root / dataset / "dataset_manifest.json",
            {"dataset": dataset, "fixture": True},
        )
    decisions_path = tmp_path / "training_mode_decisions_formal.json"
    pipeline.atomic_json(
        decisions_path,
        {
            "format_version": 3,
            "formal_matrix": {
                "search_seeds": list(cross_dataset_runner.FORMAL_SEARCH_SEEDS)
            },
            "datasets": {
                dataset: {
                    "status": "frozen",
                    "training_mode": "full_batch",
                    **(
                        {
                            "continuous_extreme_resource_preflight": {
                                "status": "completed"
                            }
                        }
                        if dataset == "flickr"
                        else {}
                    ),
                }
                for dataset in cross_dataset_runner.FORMAL_DATASETS
            },
        },
    )
    artifact_root = tmp_path / "pipeline"
    manifest, audit = formal_matrix.build_formal_manifest(
        repo_root=pipeline.REPO_ROOT,
        source_manifest_path=source_manifest,
        frozen_source_id=identity["source_id"],
        checkpoint=checkpoint_path,
        data_root=tmp_path / "data",
        manifest_root=manifest_root,
        training_mode_decisions=decisions_path,
        method_config=pipeline.METHOD_CONFIG,
        python_executable=sys.executable,
        run_tag="test_deterministic_matrix",
        artifact_root=artifact_root,
    )
    assert manifest["task_count"] == 60
    assert audit["total_requested_full_evaluations"] == 18_000
    assert all(
        Path(task["output_directory"]).is_relative_to(artifact_root / "results")
        for task in manifest["tasks"]
    )
    assert all("--artifact-root" in task["exact_argv"] for task in manifest["tasks"])
    assert all(task["exact_argv"][-1] == "--execute" for task in manifest["tasks"])
