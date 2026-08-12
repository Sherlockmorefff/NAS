from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_score_script_matches_packaged_reference() -> None:
    root_script = (REPO_ROOT / "compute_score.R").read_text(encoding="utf-8")
    packaged_script = (
        REPO_ROOT / "bayesian_optimization" / "compute_score.R"
    ).read_text(encoding="utf-8")

    assert root_script.rstrip() == packaged_script.rstrip()
    assert "prrint" not in root_script
