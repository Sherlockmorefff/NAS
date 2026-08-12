from pathlib import Path

from analyse.collect_experiment_results import parse_args, scan_result_dirs


def test_legacy_root_is_an_explicit_read_option() -> None:
    args = parse_args(
        [
            "--legacy-root",
            "legacy_artifacts/pre_20260813",
            "--output",
            "/tmp/report",
        ]
    )
    assert args.legacy_root == "legacy_artifacts/pre_20260813"
    assert args.results_root is None


def test_collector_discovers_structured_search_and_final_runs(tmp_path: Path) -> None:
    search = tmp_path / "citeseer/s0/search_seed5"
    final = tmp_path / "final/citeseer/s0"
    search.mkdir(parents=True)
    final.mkdir(parents=True)
    (search / "history_final.json").write_text("[]", encoding="utf-8")
    (final / "final_results_global4.json").write_text("[]", encoding="utf-8")
    result_dirs, final_dirs = scan_result_dirs(tmp_path)
    assert search.resolve() in result_dirs
    assert final.resolve() in final_dirs
