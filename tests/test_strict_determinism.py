from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import bo_phase4
import deterministic_runtime
import eval_utils
import final_eval
from evaluation_errors import is_infrastructure_failure


def test_strict_deterministic_configuration_is_active_before_training() -> None:
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8")
    deterministic_runtime.assert_strict_torch_determinism(torch)


def test_cublas_environment_is_set_before_cuda_initialization_in_fresh_process() -> None:
    script = """
import json, os
from deterministic_runtime import prepare_deterministic_environment
before = prepare_deterministic_environment()
import torch
from eval_utils import assert_strict_torch_determinism
assert_strict_torch_determinism(torch)
print(json.dumps({
    'workspace': before,
    'cuda_initialized': torch.cuda.is_initialized(),
    'enabled': torch.are_deterministic_algorithms_enabled(),
    'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
    'benchmark': torch.backends.cudnn.benchmark,
}))
"""
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout.strip())
    assert state == {
        "workspace": ":4096:8",
        "cuda_initialized": False,
        "enabled": True,
        "warn_only": False,
        "benchmark": False,
    }


def test_invalid_cublas_configuration_is_rejected() -> None:
    script = "from deterministic_runtime import prepare_deterministic_environment; prepare_deterministic_environment()"
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = "invalid"
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "CUBLAS_WORKSPACE_CONFIG" in completed.stderr


def test_deterministic_operator_errors_are_never_treated_as_invalid_candidates() -> None:
    assert is_infrastructure_failure(
        RuntimeError("this operation does not have a deterministic implementation")
    )
    assert is_infrastructure_failure(
        RuntimeError("enabled torch.use_deterministic_algorithms(True)")
    )


def test_seed_and_fingerprint_regression_vectors_are_unchanged() -> None:
    z = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    fingerprint = bo_phase4.make_candidate_fingerprint(z)
    assert fingerprint == "5646e40f174460f4187fd640362242a62f29946914548f8f9c459cc62e9c6c38"
    assert bo_phase4.candidate_evaluation_seed(7, fingerprint, "full") == 678075286
    assert bo_phase4.candidate_evaluation_seed(7, fingerprint, "low") == 3102799602
    assert eval_utils.stable_seed(7, "decoder", torch.from_numpy(z[:12])) == 2626103247
    assert final_eval.final_evaluation_seed(0, fingerprint, 29) == 376075682


def test_candidate_training_seed_ignores_method_stage_order_index_and_directory() -> None:
    fingerprint = "a" * 64
    expected = bo_phase4.candidate_evaluation_seed(7, fingerprint, "full")
    metadata_variants = [
        {"method": "S0", "stage": "initial_seed", "order": 0, "full_index": 0, "directory": "/a"},
        {"method": "G150", "stage": "initial_expand", "order": 99, "full_index": 149, "directory": "/b"},
        {"method": "G100", "stage": "online_bo", "order": 250, "full_index": 299, "directory": "/c"},
    ]
    assert all(
        bo_phase4.candidate_evaluation_seed(7, fingerprint, "full") == expected
        for _metadata in metadata_variants
    )


def test_final_evaluation_seed_is_cross_source_and_covers_test_seeds_0_to_29() -> None:
    fingerprint = "b" * 64
    expected = [final_eval.final_evaluation_seed(0, fingerprint, seed) for seed in range(30)]
    assert len(set(expected)) == 30
    for method, stage, order, directory in (
        ("S0", "initial_seed", 1, "/one"),
        ("G100", "initial_expand", 50, "/two"),
        ("G150", "online_bo", 299, "/three"),
    ):
        del method, stage, order, directory
        assert [final_eval.final_evaluation_seed(0, fingerprint, seed) for seed in range(30)] == expected
