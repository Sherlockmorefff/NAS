from pathlib import Path

import pytest

from experiment_paths import (
    final_eval_dir,
    log_dir,
    posthoc_dir,
    search_dir,
    validate_protocol_id,
)


PROTOCOL = "kg-k6lf_v1_20260813_64392499"


def test_validate_protocol_id_accepts_fixed_format() -> None:
    assert validate_protocol_id(PROTOCOL) == PROTOCOL


@pytest.mark.parametrize(
    "value",
    ["", "/tmp/run", "../run", "run", "run_v0_20260813_64392499", "run_v1_20260230_64392499", "RUN_v1_20260813_64392499"],
)
def test_validate_protocol_id_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_protocol_id(value)


def test_structured_paths_are_repository_relative(tmp_path: Path) -> None:
    assert search_dir(tmp_path, PROTOCOL, "citeseer", "G100", 5) == (
        tmp_path / "results/search" / PROTOCOL / "citeseer/g100/search_seed5"
    )
    assert final_eval_dir(tmp_path, PROTOCOL, "pubmed", "S0") == (
        tmp_path / "results/final_eval" / PROTOCOL / "pubmed/s0"
    )
    assert posthoc_dir(tmp_path, PROTOCOL, "summary") == (
        tmp_path / "results/posthoc" / PROTOCOL / "summary"
    )
    assert log_dir(tmp_path, PROTOCOL, "search") == (
        tmp_path / "logs" / PROTOCOL / "search"
    )
    assert log_dir(
        tmp_path,
        PROTOCOL,
        "search",
        dataset="citeseer",
        method="g100",
        search_seed=5,
    ) == tmp_path / "logs" / PROTOCOL / "search/citeseer/g100/search_seed5"


def test_explicit_output_wins_and_traversal_is_rejected(tmp_path: Path) -> None:
    assert final_eval_dir(
        tmp_path, PROTOCOL, "cora", "s0", output="custom/final"
    ) == tmp_path / "custom/final"
    with pytest.raises(ValueError):
        posthoc_dir(tmp_path, PROTOCOL, "summary", output="../outside")
    with pytest.raises(ValueError):
        posthoc_dir(tmp_path, PROTOCOL, "../outside")


def test_nonempty_directory_requires_explicit_resume(tmp_path: Path) -> None:
    target = tmp_path / "results/search" / PROTOCOL / "cora/s0/search_seed5"
    target.mkdir(parents=True)
    (target / "history.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        search_dir(tmp_path, PROTOCOL, "cora", "s0", 5)
    assert search_dir(tmp_path, PROTOCOL, "cora", "s0", 5, resume=True) == target
