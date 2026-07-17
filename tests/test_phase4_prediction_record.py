from __future__ import annotations

import math
from pathlib import Path
import ast
import copy
from types import SimpleNamespace

from surrogate.metrics import prediction_record_fields


ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "bo_phase4.py").read_text(encoding="utf-8")


def _module_tree() -> ast.Module:
    return ast.parse(_source())


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
    tree = _module_tree()
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


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _parser_argument_calls() -> dict[str, ast.Call]:
    function = _function_node("parse_args")
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and _name(node.func) == "parser.add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            calls[node.args[0].value] = node
    return calls


def _function_source(name: str) -> str:
    return ast.get_source_segment(_source(), _function_node(name)) or ""


def _has_call(node: ast.AST, call_name: str) -> bool:
    return any(
        isinstance(child, ast.Call) and _name(child.func) == call_name
        for child in ast.walk(node)
    )


def test_gp_init_mode_cli_keeps_checkpoint_backwards_compatible():
    calls = _parser_argument_calls()
    gp_init = calls["--gp_init_mode"]
    choices = _keyword(gp_init, "choices")
    default = _keyword(gp_init, "default")
    assert isinstance(choices, ast.Tuple)
    assert [item.value for item in choices.elts if isinstance(item, ast.Constant)] == [
        "checkpoint",
        "scratch",
    ]
    assert isinstance(default, ast.Constant)
    assert default.value == "checkpoint"

    gp_checkpoint = calls["--gp_checkpoint"]
    assert not any(
        keyword.arg == "required"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in gp_checkpoint.keywords
    )

    scratch_min = calls["--scratch_gp_min_points"]
    scratch_default = _keyword(scratch_min, "default")
    assert isinstance(scratch_default, ast.Constant)
    assert scratch_default.value == 3


def test_main_requires_checkpoint_only_in_checkpoint_mode():
    function = _function_node("main")
    source = _function_source("main")
    assert 'args.gp_init_mode == "checkpoint" and not args.gp_checkpoint' in source
    assert "--gp_checkpoint is required when --gp_init_mode checkpoint" in source
    assert 'args.gp_init_mode == "scratch" and args.gp_checkpoint' in source
    assert "--gp_checkpoint is ignored when --gp_init_mode scratch" in source

    load_branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and "args.gp_init_mode" in ast.unparse(node.test)
        and "checkpoint" in ast.unparse(node.test)
        and any(_has_call(stmt, "AccuracyGPPredictor.load") for stmt in node.body)
    ]
    assert len(load_branches) == 1


def test_phase4_predicts_before_evaluation_and_append():
    function = _function_node("_append_eval")
    predict = _call_lines(function, "predictor.predict")
    evaluate = _call_lines(function, "eval_candidate")
    persist = _call_lines(function, "_save_prediction_csv")
    metrics = _call_lines(function, "convergence_monitor.observe_bo_result")
    append = _call_lines(function, "predictor.append_observation")
    assert predict and evaluate and persist and metrics and append
    assert max(predict) < min(persist) < min(evaluate)
    assert max(evaluate) < min(metrics) < min(append)
    assert any(min(evaluate) < line < min(append) for line in persist)
    assert _has_update_guard(function)
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert any(isinstance(value, ast.Tuple) and len(value.elts) == 4 for value in returns)


def test_append_eval_persists_pre_update_prediction_and_metrics_before_gp_update():
    events = []

    class FakePredictor:
        train_size = 10
        use_conditional_kernel = False

        def predict(self, z_search, condition_mask=None):
            events.append("predict")
            return {"mean": 0.74, "std": 0.02, "lower_95": 0.70, "upper_95": 0.78}

        def append_observation(self, z_search, actual, condition_mask=None):
            events.append("append")
            self.train_size += 1

        def refit(self, optimize, steps):
            events.append("refit")

    class FakeMonitor:
        def observe_bo_result(self, actual, prediction):
            events.append("metrics")
            assert prediction["gp_train_size_before"] == 10
            assert prediction["gp_train_size_after"] == 10
            return {"prequential_spearman": 1.0}

    class FakeLogger:
        def info(self, *args, **kwargs):
            return None

    def null_fields(before, after):
        return {
            "gp_pred_mean": None,
            "gp_pred_std": None,
            "gp_pred_95_low": None,
            "gp_pred_95_high": None,
            "gp_residual": None,
            "gp_abs_error": None,
            "gp_squared_error": None,
            "gp_standardized_residual": None,
            "gp_covered_by_95": None,
            "gp_train_size_before": before,
            "gp_train_size_after": after,
        }

    def save_prediction_csv(records, output):
        events.append(("save", copy.deepcopy(records[-1])))

    def evaluate(*args, **kwargs):
        events.append("evaluate")
        return "z", {
            "val_acc": 0.75,
            "valid": True,
            "config": {"operations": ["GCNConv"], "edges": [[0, 1], [1, 2]]},
            "hp": {},
        }

    def make_history(step, record_type, z_search, result, args, best_acc=None, gp_record=None):
        return {"step": step, "val_acc": result["val_acc"], **dict(gp_record or {})}

    namespace = {
        "time": SimpleNamespace(monotonic=lambda: 1.0),
        "_null_gp_record_fields": null_fields,
        "_save_prediction_csv": save_prediction_csv,
        "eval_candidate": evaluate,
        "_holdout_leakage_reason": lambda *args: "",
        "prediction_record_fields": prediction_record_fields,
        "history_record": make_history,
    }
    exec("from __future__ import annotations\n" + _function_source("_append_eval"), namespace)
    args = SimpleNamespace(
        output="unused",
        gp_update_mode="warm_refit",
        gp_checkpoint=None,
        gp_init_mode="scratch",
        gp_refit_every=1,
        gp_refit_steps=2,
    )
    predictor = FakePredictor()
    prediction_records = []

    namespace["_append_eval"](
        object(),
        "z",
        object(),
        8,
        3,
        args,
        "cpu",
        [],
        [],
        [],
        4,
        "bo",
        FakeLogger(),
        predictor,
        prediction_records,
        "online_bo",
        0.1,
        True,
        0,
        [1.0],
        best_acc=0.70,
        convergence_monitor=FakeMonitor(),
    )

    event_names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert event_names == [
        "predict",
        "save",
        "evaluate",
        "metrics",
        "save",
        "append",
        "refit",
        "save",
    ]
    snapshots = [event[1] for event in events if isinstance(event, tuple)]
    assert snapshots[0]["val_acc"] is None
    assert snapshots[0]["gp_pred_mean"] == 0.74
    assert snapshots[0]["gp_train_size_after"] == 10
    assert snapshots[1]["val_acc"] == 0.75
    assert snapshots[1]["prequential_spearman"] == 1.0
    assert snapshots[1]["gp_update_performed"] is False
    assert snapshots[2]["gp_train_size_after"] == 11
    assert snapshots[2]["gp_update_performed"] is True


def test_scratch_init_records_null_gp_fields_without_checkpoint_source():
    source = _function_source("_append_eval")
    assert "_null_gp_record_fields" in source
    assert "prediction is None" in source
    assert "online GP update requested before a GP predictor exists" in source
    assert '"gp_checkpoint_source": args.gp_checkpoint if args.gp_init_mode == "checkpoint" else None' in source

    null_source = _function_source("_null_gp_record_fields")
    for field in (
        "gp_pred_mean",
        "gp_pred_std",
        "gp_pred_95_low",
        "gp_pred_95_high",
        "gp_residual",
        "gp_abs_error",
        "gp_squared_error",
        "gp_standardized_residual",
        "gp_covered_by_95",
    ):
        assert f'"{field}": None' in null_source


def test_scratch_initial_gp_is_fit_from_current_valid_points_only():
    function = _function_node("_fit_scratch_predictor")
    source = _function_source("_fit_scratch_predictor")
    assert _call_lines(function, "AccuracyGPPredictor.fit_offline")
    assert '"gp_init_mode": "scratch"' in source
    assert '"offline_train_size": 0' in source
    assert '"used_previous_history": False' in source
    assert '"used_offline_checkpoint": False' in source
    assert "predictor.offline_train_size = 0" in source
    assert "accuracy_gp_scratch_initial.pt" in source
    assert "init_X = torch.stack([item[0] for item in init_valid])" in source
    assert "init_y = [item[1] for item in init_valid]" in source


def test_scratch_mode_does_not_load_gmm_history():
    source = _function_source("run_gmm_init")
    assert 'if args.gp_init_mode == "scratch"' in source
    assert "avoid old history" in source
    assert source.index('if args.gp_init_mode == "scratch"') < source.index("load_history_vectors")


def test_run_bo_scratch_flow_trains_gp_before_formal_bo():
    function = _function_node("run_bo")
    source = _function_source("run_bo")
    assert '"scratch_init_no_gp"' in source
    assert "_ensure_scratch_min_points" in source
    assert "_fit_scratch_predictor" in source
    assert '"online_bo"' in source
    assert '"used_previous_history": bool(args.gp_init_mode == "checkpoint")' in source
    assert '"used_offline_checkpoint": bool(args.gp_init_mode == "checkpoint")' in source
    assert "online_bo_samples" in source
    assert min(_call_lines(function, "_fit_scratch_predictor")) < min(_call_lines(function, "optimize_acq"))


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


def test_seed_defaults_remain_unchanged():
    bo_calls = _parser_argument_calls()
    bo_seed_default = _keyword(bo_calls["--seed"], "default")
    assert isinstance(bo_seed_default, ast.Constant)
    assert bo_seed_default.value == 42

    final_source = (ROOT / "final_eval.py").read_text(encoding="utf-8")
    final_tree = ast.parse(final_source)
    final_parse = next(
        node
        for node in final_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    final_calls = {
        node.args[0].value: node
        for node in ast.walk(final_parse)
        if isinstance(node, ast.Call)
        and _name(node.func) == "parser.add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    seed_start_default = _keyword(final_calls["--seed_start"], "default")
    n_seeds_default = _keyword(final_calls["--n_seeds"], "default")
    assert isinstance(seed_start_default, ast.Constant)
    assert seed_start_default.value == 0
    assert isinstance(n_seeds_default, ast.Constant)
    assert n_seeds_default.value == 10


def test_prequential_metrics_are_persisted_in_step_csv_and_run_summary():
    csv_source = _function_source("_save_prediction_csv")
    run_source = _function_source("run_bo")

    assert "*PREQUENTIAL_METRIC_FIELDS" in csv_source
    assert "**monitor.current_prequential_metrics()" in run_source
    assert '"valid_convergence_checks"' in run_source
    assert '"stagnation_deferred_events"' in run_source
