from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "bo_phase4.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name):
    return next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_surrogate_cli_defaults_keep_exact_gp_and_dkl_defaults_are_explicit():
    calls = {}
    for node in ast.walk(_function("parse_args")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                calls[node.args[0].value] = node
    defaults = {
        keyword.arg: keyword.value.value
        for keyword in calls["--surrogate_type"].keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert defaults["default"] == "exact_gp"
    expected = {
        "--dkl_hidden_dim": 32,
        "--dkl_feature_dim": 8,
        "--dkl_activation": "silu",
        "--dkl_lr": 0.01,
        "--dkl_weight_decay": 1e-4,
        "--dkl_grad_clip": 5.0,
        "--dkl_init_steps": 200,
        "--dkl_refit_steps": 50,
        "--dkl_early_stopping_patience": 25,
        "--dkl_min_delta": 1e-5,
    }
    for argument, value in expected.items():
        default = next(keyword.value.value for keyword in calls[argument].keywords if keyword.arg == "default")
        assert default == value


def test_exact_and_dkl_fit_update_and_checkpoint_branches_are_separate():
    scratch = ast.get_source_segment(SOURCE, _function("_fit_scratch_predictor"))
    append = ast.get_source_segment(SOURCE, _function("_append_eval"))
    main = ast.get_source_segment(SOURCE, _function("main"))
    assert "AccuracyGPPredictor.fit_offline" in scratch
    assert "fit_steps=int(args.gp_refit_steps)" in scratch
    assert "DKLAccuracyGPPredictor.fit_offline" in scratch
    assert "fit_steps=int(args.dkl_init_steps)" in scratch
    assert "steps=int(args.gp_refit_steps) if should_optimize else None" in append
    assert "steps=int(args.dkl_refit_steps) if should_optimize else None" in append
    assert "AccuracyGPPredictor.load" in main
    assert "DKLAccuracyGPPredictor.load" in main
    assert "stable_seed(" in scratch and '"dkl_initial_fit"' in scratch
    assert '"dkl_refit"' in append
