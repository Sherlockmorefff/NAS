from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

from source_freeze import SOURCE_GATE_DISCOVERY_RULES
from source_verification import discover_source_gate_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "validate_dataset_loading.py",
    "gpu_dataset_smoke.py",
    "resource_preflight.py",
    "flickr_continuous_extreme_preflight.py",
    "flickr_failed_candidate_resource_smoke.py",
    "method_path_smoke.py",
    "cross_dataset_cost_estimate.py",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("canonical", (False, True))
def test_validation_help_works_outside_repository(
    entrypoint: str, canonical: bool
) -> None:
    path = (
        REPO_ROOT / "scripts" / "validation" / entrypoint
        if canonical
        else REPO_ROOT / entrypoint
    )

    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_root_validation_import_is_canonical_module(entrypoint: str) -> None:
    module_name = entrypoint.removesuffix(".py")
    compatibility_module = importlib.import_module(module_name)
    implementation_module = importlib.import_module(
        f"scripts.validation.{module_name}"
    )

    assert compatibility_module is implementation_module


def test_source_freeze_covers_new_runtime_layout() -> None:
    paths = set(discover_source_gate_paths(REPO_ROOT, SOURCE_GATE_DISCOVERY_RULES))

    assert "scripts/validation/resource_preflight.py" in paths
    assert "scripts/analysis/run_collect_results.sh" in paths
    assert "resource_preflight.py" in paths
    assert "legacy/phased_cora_pipeline/bo_phase2.py" in paths
    assert not any(
        path.startswith("legacy/geometric_acquisition/") for path in paths
    )
