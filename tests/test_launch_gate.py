from __future__ import annotations

import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("botorch")

import bo_phase4
import cross_dataset_runner
import flickr_continuous_extreme_preflight as extreme
import formal_matrix
import source_verification


def _write_source_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "formal.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    checkpoint = repo / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    rows = [
        {
            "path": "formal.py",
            "tracked_status": "untracked",
            "size_bytes": source.stat().st_size,
            "sha256": source_verification.sha256_file(source),
            "formal_run_required": True,
            "functional_category": "formal_runtime",
        }
    ]
    source_id = source_verification.source_id_from_rows(rows)
    manifest = {
        "format_version": 1,
        "source_id": source_id,
        "source_gate": {
            "discovery_rules": [
                {"root": ".", "recursive": False, "suffixes": [".py"]}
            ]
        },
        "files": rows,
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": source_verification.sha256_file(checkpoint),
        },
    }
    manifest_path = repo / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return repo, source, checkpoint, manifest_path, source_id


def test_source_gate_accepts_exact_source_and_checkpoint(tmp_path):
    repo, _source, checkpoint, manifest, source_id = _write_source_fixture(
        tmp_path
    )

    audit = source_verification.verify_frozen_source(
        manifest,
        repo_root=repo,
        expected_source_id=source_id,
        checkpoint_path=checkpoint,
    )

    assert audit["status"] == "completed"
    assert audit["source_id"] == source_id
    assert audit["verified_source_files"] == ["formal.py"]


def test_source_gate_rejects_content_mismatch(tmp_path):
    repo, source, checkpoint, manifest, source_id = _write_source_fixture(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(
        source_verification.SourceVerificationError, match="SHA-256 mismatch"
    ):
        source_verification.verify_frozen_source(
            manifest,
            repo_root=repo,
            expected_source_id=source_id,
            checkpoint_path=checkpoint,
        )


def test_source_gate_rejects_missing_source(tmp_path):
    repo, source, checkpoint, manifest, source_id = _write_source_fixture(tmp_path)
    source.unlink()

    with pytest.raises(
        source_verification.SourceVerificationError,
        match="missing frozen source file",
    ):
        source_verification.verify_frozen_source(
            manifest,
            repo_root=repo,
            expected_source_id=source_id,
            checkpoint_path=checkpoint,
        )


def test_source_gate_rejects_unexpected_source(tmp_path):
    repo, _source, checkpoint, manifest, source_id = _write_source_fixture(
        tmp_path
    )
    (repo / "new_formal.py").write_text("NEW = True\n", encoding="utf-8")

    with pytest.raises(
        source_verification.SourceVerificationError,
        match="unexpected source file",
    ):
        source_verification.verify_frozen_source(
            manifest,
            repo_root=repo,
            expected_source_id=source_id,
            checkpoint_path=checkpoint,
        )


def test_source_gate_rejects_checkpoint_content_mismatch(tmp_path):
    repo, _source, checkpoint, manifest, source_id = _write_source_fixture(
        tmp_path
    )
    checkpoint.write_bytes(b"checkpoint-v2")

    with pytest.raises(
        source_verification.SourceVerificationError,
        match="checkpoint SHA-256 mismatch",
    ):
        source_verification.verify_frozen_source(
            manifest,
            repo_root=repo,
            expected_source_id=source_id,
            checkpoint_path=checkpoint,
        )


def test_runner_dry_run_executes_source_gate_and_never_starts_training(
    tmp_path, monkeypatch, capsys
):
    repo = tmp_path / "repo"
    source_dir = repo / "source"
    source_dir.mkdir(parents=True)
    guarded = source_dir / "runtime.py"
    guarded.write_text("VALUE = 1\n", encoding="utf-8")
    checkpoint = repo / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    rows = [
        {
            "path": "source/runtime.py",
            "size_bytes": guarded.stat().st_size,
            "sha256": source_verification.sha256_file(guarded),
            "formal_run_required": True,
        }
    ]
    source_id = source_verification.source_id_from_rows(rows)
    source_manifest = repo / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_id": source_id,
                "source_gate": {
                    "discovery_rules": [
                        {
                            "root": "source",
                            "recursive": True,
                            "suffixes": [".py"],
                        }
                    ]
                },
                "files": rows,
                "checkpoint": {
                    "path": str(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": source_verification.sha256_file(checkpoint),
                },
            }
        ),
        encoding="utf-8",
    )
    manifests = repo / "manifests" / "citeseer"
    manifests.mkdir(parents=True)
    (manifests / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    decisions = repo / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "format_version": 3,
                "formal_matrix": {"search_seeds": [5, 6, 7, 8, 9]},
                "datasets": {
                    "citeseer": {
                        "status": "frozen",
                        "training_mode": "full_batch",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cross_dataset_runner, "ROOT", repo)
    monkeypatch.setattr(
        cross_dataset_runner,
        "validate_phase4_command",
        lambda _command: type(
            "Parsed",
            (),
            {"dataset": "citeseer", "seed": 5, "max_total_full_evals": 300},
        )(),
    )
    monkeypatch.setattr(
        cross_dataset_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("dry-run started a subprocess"),
    )

    exit_code = cross_dataset_runner.main(
        [
            "--run-tag",
            "dry_gate",
            "--dataset",
            "citeseer",
            "--method",
            "S0",
            "--search-seed",
            "5",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(repo / "data"),
            "--manifest-root",
            str(repo / "manifests"),
            "--training-mode-decisions",
            str(decisions),
            "--method-config",
            str(cross_dataset_runner.DEFAULT_METHOD_CONFIG),
            "--expected-source-manifest",
            str(source_manifest),
            "--frozen-source-id",
            source_id,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["executed"] is False
    assert payload["source_verification"]["status"] == "completed"


def test_training_mode_artifact_preserves_development_evidence(tmp_path):
    accepted = {
        "format_version": 2,
        "datasets": {
            dataset: {
                "status": "frozen",
                "training_mode": "full_batch",
                "validated_search_seeds": [0, 1, 2, 3, 4],
            }
            for dataset in cross_dataset_runner.FORMAL_DATASETS
        },
    }
    accepted["datasets"]["ogbn-arxiv"] = {
        "status": "blocked_resource_preflight_oom",
        "training_mode": None,
        "validated_search_seeds": [],
    }
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(accepted), encoding="utf-8")
    preflight = tmp_path / "flickr_continuous.json"
    preflight.write_text(
        json.dumps({"status": "completed", "dataset": "flickr", "oom": False}),
        encoding="utf-8",
    )

    output = formal_matrix.build_formal_training_mode_decisions(
        path, flickr_continuous_preflight_path=preflight
    )

    assert output["development_search_seeds"] == [0, 1, 2, 3, 4]
    assert output["formal_matrix"]["search_seeds"] == [5, 6, 7, 8, 9]
    assert output["datasets"]["flickr"]["validated_search_seeds"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    decisions = tmp_path / "formal.json"
    decisions.write_text(json.dumps(output), encoding="utf-8")
    row = cross_dataset_runner.validate_training_mode_decision(
        decisions, "flickr", search_seed=5
    )
    assert row["training_mode"] == "full_batch"


def _minimal_manifest_tasks():
    tasks = []
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        for method in cross_dataset_runner.METHOD_KEYS:
            for seed in cross_dataset_runner.FORMAL_SEARCH_SEEDS:
                tasks.append(
                    {
                        "task_id": f"{dataset}__{method}__{seed}",
                        "dataset": dataset,
                        "method": method,
                        "search_seed": seed,
                        "output_directory": f"/results/{dataset}/{method}/{seed}",
                        "log_directory": f"/logs/{dataset}/{method}/{seed}",
                        "status": "not_started",
                    }
                )
    return tasks


def test_formal_manifest_audit_requires_exact_60_and_seed_counts():
    audit = formal_matrix.audit_formal_manifest(
        {"tasks": _minimal_manifest_tasks()}
    )

    assert audit["task_count"] == 60
    assert audit["by_dataset"] == {
        "citeseer": 15,
        "dblp": 15,
        "flickr": 15,
        "pubmed": 15,
    }
    assert audit["by_method"] == {"G100": 20, "G150": 20, "S0": 20}
    assert audit["by_seed"] == {str(seed): 12 for seed in range(5, 10)}


def test_formal_manifest_audit_rejects_development_seed_and_duplicate_task():
    development = _minimal_manifest_tasks()
    development[0]["search_seed"] = 0
    with pytest.raises(ValueError, match="seeds 5--9"):
        formal_matrix.audit_formal_manifest({"tasks": development})

    duplicate = _minimal_manifest_tasks()
    duplicate[1]["task_id"] = duplicate[0]["task_id"]
    with pytest.raises(ValueError, match="task_id"):
        formal_matrix.audit_formal_manifest({"tasks": duplicate})


def test_decode_search_has_every_architecture_corner():
    rows = extreme.deterministic_arch_search_vectors(
        arch_nz=3, z_bound=2.5, sobol_count=0
    )

    assert len(rows) == 8
    assert {tuple(row.tolist()) for row in rows} == {
        (a, b, c)
        for a in (-2.5, 2.5)
        for b in (-2.5, 2.5)
        for c in (-2.5, 2.5)
    }


def test_continuous_extreme_candidate_is_canonical_out_of_pool_and_joint(
    monkeypatch,
):
    monkeypatch.setattr(
        extreme,
        "deterministic_arch_search_vectors",
        lambda **_kwargs: [torch.zeros(bo_phase4.ARCH_NZ)],
    )
    monkeypatch.setattr(
        extreme,
        "_formal_pool_identity",
        lambda: (set(), {str(seed): f"pool-{seed}" for seed in range(5, 10)}),
    )
    monkeypatch.setattr(
        bo_phase4,
        "decode_arch",
        lambda *_args, **_kwargs: {
            "operations": ["GATConv"] * 5,
            "effective_layers": 5,
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        },
    )
    monkeypatch.setattr(extreme, "_parameter_count", lambda *_args, **_kwargs: 123)

    candidates, pools = extreme.decode_continuous_extremes(
        object(),
        device=torch.device("cpu"),
        search_seed=5,
        in_ch=500,
        out_ch=7,
        num_nodes=89250,
        num_edges=899756,
        sobol_count=0,
    )

    assert len(candidates) == 1
    row = candidates[0]
    assert row["candidate_pool_index"] is None
    assert row["source"] == "continuous_search_space_extreme"
    assert row["formal_pool_member"] is False
    assert row["hidden_dimension"] == 512
    assert row["effective_layers"] == 5
    assert row["gat_layer_count"] == 5
    assert row["heads"] == 1
    assert len(row["canonical_z_search"]) == 16
    assert pools == {str(seed): f"pool-{seed}" for seed in range(5, 10)}
