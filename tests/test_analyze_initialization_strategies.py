from __future__ import annotations

import csv
import json
from pathlib import Path

from analyse import analyze_initialization_strategies as analysis


def test_documented_formal_comparison_invariants_are_fixed() -> None:
    documentation = (
        Path(__file__).parents[1] / "docs" / "wgmm_clustered_ted.md"
    ).read_text(encoding="utf-8")
    assert "--n_lhs_candidates 768 --eval_epochs 150 --patience 40" in documentation
    assert "<strategy>__checkpoint" in documentation
    assert "<strategy>__gmm_fit_pool" in documentation
    assert "768 candidates to 1000 defines a new experiment matrix" in documentation


def test_analysis_uses_explicit_own_history_and_marks_counterfactual_limit(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for index in range(4):
        actual = 0.70 + 0.01 * index
        rows.append(
            {
                "step": index,
                "evaluation_stage": "initial_seed" if index < 2 else "online_bo",
                "evaluation_fidelity": "full",
                "valid": True,
                "val_acc": actual,
                "gp_pred_mean": None if index < 2 else actual - 0.005,
                "gp_pred_std": None if index < 2 else 0.02,
                "z_search": [float(index + column) / 20.0 for column in range(16)],
                "operations": ["GCNConv", "GATConv"],
                "edges": [[0, 1], [1, 2]],
                "cluster_id": index % 2,
            }
        )
    (run / "history_final.json").write_text(json.dumps(rows), encoding="utf-8")
    (run / "budget_summary.json").write_text(
        json.dumps({"low_fidelity_candidate_count": 6, "low_fidelity_actual_epochs": 120}),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    assert analysis.main(
        [
            "--run", f"toy={run}",
            "--output", str(output),
            "--accuracy_threshold", "0.72",
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["counterfactual_claim_allowed"] is False
    assert "own full-fidelity" in summary["runs"][0]["label_evidence_scope"]
    assert "no counterfactual superiority" in summary["interpretation"]["offline_limit"]
    with open(output / "per_run_metrics.csv", "r", encoding="utf-8", newline="") as handle:
        metric = next(csv.DictReader(handle))
    assert metric["full_evaluation_count"] == "4"
    assert metric["full_evals_to_val_0.72"] == "3"
    assert metric["low_fidelity_candidate_count"] == "6"
    assert metric["test_metric_status"] == "unavailable:no_explicit_final_results"
    assert metric["final_test_mean_best"] == ""


def test_analysis_reads_test_only_from_explicit_final_results(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history_final.json").write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "evaluation_fidelity": "full",
                    "valid": True,
                    "val_acc": 0.99,
                    "z_search": [0.0] * 16,
                    "operations": ["GCNConv"],
                    "edges": [[0, 1]],
                }
            ]
        ),
        encoding="utf-8",
    )
    final_results = tmp_path / "final_results_global4.json"
    final_results.write_text(
        json.dumps(
            [
                {
                    "candidate_rank": 1,
                    "source": "history",
                    "search_step": 0,
                    "test_mean": 0.81,
                    "test_std": 0.01,
                    "n_valid": 5,
                },
                {
                    "candidate_rank": 2,
                    "source": "baseline",
                    "search_step": None,
                    "test_mean": 0.95,
                    "test_std": 0.02,
                    "n_valid": 5,
                },
                {
                    "candidate_rank": 3,
                    "source": "search_validation",
                    "search_step": 0,
                    "test_mean": 0.99,
                    "test_std": 0.0,
                    "n_valid": 1,
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    assert analysis.main(
        [
            "--run", f"toy={run}",
            "--final_results", f"toy={final_results}",
            "--output", str(output),
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metric = summary["runs"][0]
    assert metric["best_val_acc"] == 0.99
    assert metric["test_metric_status"] == "available:explicit_final_results"
    assert metric["final_test_mean_best"] == 0.81
    assert metric["final_test_mean_best"] != metric["best_val_acc"]
    assert metric["final_test_candidate_rank"] == 1
    assert len(metric["final_results_sha256"]) == 64


def test_analysis_refuses_nonempty_output(tmp_path) -> None:
    run = tmp_path / "history.json"
    run.write_text("[]", encoding="utf-8")
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    try:
        analysis.main(["--run", f"toy={run}", "--output", str(output)])
    except FileExistsError:
        pass
    else:
        raise AssertionError("analysis must refuse a nonempty output directory")
