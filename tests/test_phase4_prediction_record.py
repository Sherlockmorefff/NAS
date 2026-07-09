from __future__ import annotations

import math
from pathlib import Path
import ast

from surrogate.metrics import prediction_record_fields


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_error_fields_and_train_sizes():
    fields = prediction_record_fields(
        {"mean": 0.7, "std": 0.05, "lower_95": 0.602, "upper_95": 0.798},
        actual=0.75, train_size_before=10, train_size_after=11,
    )
    assert fields["gp_train_size_before"] == 10
    assert fields["gp_train_size_after"] == 11
    assert math.isclose(fields["gp_residual"], 0.05)
    assert math.isclose(fields["gp_abs_error"], 0.05)
    assert math.isclose(fields["gp_squared_error"], 0.0025)
    assert math.isclose(fields["gp_standardized_residual"], 1.0)
    assert fields["gp_covered_by_95"] is True


def _function_node(name: str) -> ast.FunctionDef:
    source = (ROOT / "bo_phase4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_lines(function: ast.FunctionDef, call_name: str) -> list[int]:
    lines = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and _name(node.func) == call_name:
            lines.append(node.lineno)
    return sorted(lines)


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _has_update_guard(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(node.test)
        if text == "update_online and safe_for_gp_training":
            return True
    return False


def test_phase4_predicts_before_evaluation_and_append():
    function = _function_node("_append_eval")
    predict = _call_lines(function, "predictor.predict")
    evaluate = _call_lines(function, "eval_candidate")
    persist = _call_lines(function, "_save_prediction_csv")
    append = _call_lines(function, "predictor.append_observation")
    assert predict and evaluate and persist and append
    assert max(predict) < min(evaluate)
    assert max(evaluate) < min(append)
    assert any(min(evaluate) < line < min(append) for line in persist)
    assert _has_update_guard(function)
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert any(isinstance(value, ast.Tuple) and len(value.elts) == 4 for value in returns)


def test_phase4_candidate_ranking_is_pure_logei():
    function = _function_node("optimize_acq")
    names = {
        child.id.lower()
        for child in ast.walk(function)
        if isinstance(child, ast.Name)
    }
    attributes = {
        child.attr.lower()
        for child in ast.walk(function)
        if isinstance(child, ast.Attribute)
    }
    assert not any("novelty" in name for name in names | attributes)
    assert "batch_score" not in attributes

    argmax_assignments = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "best_idx" for target in node.targets
        ):
            continue
        calls = [child for child in ast.walk(node.value) if isinstance(child, ast.Call)]
        argmax_assignments.extend(
            call for call in calls
            if _name(call.func) == "torch.argmax"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "logei_scores"
        )
    assert len(argmax_assignments) == 1
    assert all(
        not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Add, ast.Sub))
            and _contains_name(node, "logei_scores")
        )
        for node in ast.walk(function)
    )
