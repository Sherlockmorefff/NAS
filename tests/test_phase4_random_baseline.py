from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "bo_phase4.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _function(name: str) -> ast.FunctionDef:
    return next(
        node for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _source(name: str) -> str:
    return ast.get_source_segment(SOURCE, _function(name)) or ""


def _parser_calls() -> dict[str, ast.Call]:
    return {
        node.args[0].value: node
        for node in ast.walk(_function("parse_args"))
        if isinstance(node, ast.Call)
        and _name(node.func) == "parser.add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _strategy_branch(value: str) -> ast.If:
    run_bo = _function("run_bo")
    for node in ast.walk(run_bo):
        if not isinstance(node, ast.If):
            continue
        comparison = ast.unparse(node.test)
        if f"args.online_candidate_strategy == '{value}'" in comparison:
            return node
    raise AssertionError(f"missing {value} strategy branch")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call) and _name(child.func) == name
    ]


def test_default_online_candidate_strategy_is_qlogei():
    call = _parser_calls()["--online_candidate_strategy"]
    default = _keyword(call, "default")
    choices = _keyword(call, "choices")
    assert isinstance(default, ast.Constant) and default.value == "qlogei"
    assert isinstance(choices, ast.Tuple)
    assert [item.value for item in choices.elts] == ["qlogei", "random"]


def test_frozen_history_cli_defaults_to_none():
    default = _keyword(_parser_calls()["--frozen_init_history"], "default")
    assert isinstance(default, ast.Constant) and default.value is None


def test_random_branch_never_calls_optimize_acq_and_sets_logei_none():
    branch = _strategy_branch("random")
    assert _calls(branch, "sample_random_online_candidate")
    assert not _calls(branch, "optimize_acq")
    assert "logei_value = None" in ast.unparse(branch)


def test_qlogei_branch_still_calls_original_optimize_acq():
    branch = _strategy_branch("qlogei")
    assert len(_calls(branch, "optimize_acq")) == 1
    text = ast.unparse(branch)
    assert "stable_seed(int(args.seed), 'acquisition', int(step))" in text
    assert "candidate_selection_seed = acquisition_seed" in text
    assert "with isolated_rng(acquisition_seed, device)" in text


def test_online_strategies_share_prediction_evaluation_and_gp_update_path():
    run_source = _source("run_bo")
    qlogei_position = run_source.index('args.online_candidate_strategy == "qlogei"')
    random_position = run_source.index('args.online_candidate_strategy == "random"')
    append_position = run_source.index("_append_eval(", random_position)
    assert qlogei_position < random_position < append_position
    append_source = _source("_append_eval")
    assert "predictor.predict(" in append_source
    assert "eval_candidate(" in append_source
    assert "predictor.append_observation(" in append_source
    assert '"online_bo"' in run_source


def test_random_helper_uses_stable_seed_local_generator_and_existing_transforms():
    source = _source("sample_random_online_candidate")
    assert "stable_seed(" in source
    assert 'int(args.seed), "random_acquisition", int(step)' in source
    assert 'torch.Generator(device="cpu")' in source
    assert "denormalize_search_vector(" in source
    assert "clip_z_search_by_mode(" in source
    assert "_mask_for_candidate(" in source
    for forbidden in ("optimize_acq", "predictor", "time.time", "os.getpid", "hash("):
        assert forbidden not in source


def test_frozen_branch_bypasses_initial_generation_gmm_and_extra_evaluation():
    source = _source("run_bo")
    frozen_start = source.index("if args.frozen_init_history is not None:")
    normal_start = source.index("else:", frozen_start)
    frozen_block = source[frozen_start:normal_start]
    assert "replay_frozen_initialization(args)" in frozen_block
    assert "initial_points(" not in frozen_block
    assert "_append_eval(" not in frozen_block
    assert "eval_candidate(" not in frozen_block
    assert "step = int(args.n_init)" in frozen_block
    assert "if args.frozen_init_history is None:" in source
    assert "run_gmm_init(" in source


def test_audit_fields_are_present_in_history_csv_and_summary():
    history_source = _source("history_record")
    csv_source = _source("_save_prediction_csv")
    run_source = _source("run_bo")
    for field in (
        "online_candidate_strategy",
        "candidate_selection_seed",
        "initial_record_replayed",
        "initial_history_source",
    ):
        assert field in history_source
        assert field in csv_source
    for field in (
        "online_candidate_strategy",
        "initialization_source",
        "frozen_init_history",
        "replayed_initial_samples",
        "newly_evaluated_online_samples",
    ):
        assert field in run_source


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "torch", "torch_geometric", "botorch", "igraph")
)
runtime_only = pytest.mark.skipif(
    not RUNTIME_AVAILABLE,
    reason="Phase4 runtime dependencies (NumPy/Torch/PyG/BoTorch) are unavailable",
)


def _module():
    import bo_phase4

    return bo_phase4


def _args(path: str | None = None, *, seed: int = 17, n_init: int = 3):
    return SimpleNamespace(
        seed=seed,
        n_init=n_init,
        hp_mode="global4",
        z_bound=2.5,
        frozen_init_history=path,
        gp_init_mode="scratch",
        warm_start="",
        gmm_init_history=None,
        gmm_init_trials=0,
    )


def _valid_records(module, args, *, include_bo: bool = True):
    hp_dim = module.hp_dim_from_mode(args.hp_mode)
    search_dim = module.ARCH_NZ + hp_dim
    records = []
    for step in range(args.n_init):
        z = [(-0.5 + 0.1 * step)] * module.ARCH_NZ + [0.5] * hp_dim
        fingerprint = module.make_candidate_fingerprint(
            module.np.asarray(z, dtype=module.np.float32)
        )
        evaluation_seed = module.candidate_evaluation_seed(
            args.seed, fingerprint, "full",
        )
        records.append(
            {
                "step": step,
                "type": "lhs_init",
                "hp_mode": args.hp_mode,
                "search_dim": search_dim,
                "search_seed": args.seed,
                "z_search": z,
                "z_arch": z[: module.ARCH_NZ],
                "val_acc": 0.70 + 0.01 * step,
                "valid": True,
                "operations": ["GCNConv", "Identity"],
                "edges": [[0, 1], [1, 2], [2, 3]],
                "lr": 0.01,
                "dropout": 0.3,
                "hidden_dim": 64,
                "l2": 5e-4,
                "condition_mask_vector": [1.0] * module.hp_dim_from_mode(args.hp_mode),
                "seed_derivation": module.SEED_DERIVATION,
                "candidate_fingerprint": fingerprint,
                "evaluation_stage": "initial_seed",
                "evaluation_fidelity": "full",
                "evaluation_seed": evaluation_seed,
                "candidate_eval_seed": evaluation_seed,
                "candidate_evaluation_seed_scheme": (
                    module.CANDIDATE_EVALUATION_SEED_SCHEME
                ),
                "initialization_strategy": "schur",
                "full_evaluation_index": step,
                "decoder_seed": module.stable_seed(
                    args.seed, "decoder", module.np.asarray(z[: module.ARCH_NZ], dtype=module.np.float32),
                ),
            }
        )
    if include_bo:
        records.append(
            {
                **copy.deepcopy(records[-1]),
                "step": args.n_init,
                "type": "bo",
                "val_acc": 0.99,
            }
        )
    return records


def _write_history(tmp_path: Path, records) -> str:
    path = tmp_path / "history_final.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


@runtime_only
def test_random_candidate_is_repeatable_seed_sensitive_bounded_and_rng_isolated(monkeypatch):
    module = _module()
    args = _args(seed=23)
    args.frozen_init_history = None
    monkeypatch.setattr(
        module,
        "_mask_for_candidate",
        lambda vae, z, args, device, logger: [1.0] * z.numel(),
    )
    module.random.seed(91) if hasattr(module, "random") else random.seed(91)
    module.np.random.seed(91)
    module.torch.manual_seed(91)
    python_before = random.getstate()
    numpy_before = module.np.random.get_state()
    torch_before = module.torch.random.get_rng_state().clone()

    first, _, first_seed = module.sample_random_online_candidate(
        object(), args, module.torch.device("cpu"), None, step=5,
    )
    second, _, second_seed = module.sample_random_online_candidate(
        object(), args, module.torch.device("cpu"), None, step=5,
    )
    changed_step, _, _ = module.sample_random_online_candidate(
        object(), args, module.torch.device("cpu"), None, step=6,
    )
    changed_args = copy.copy(args)
    changed_args.seed += 1
    changed_seed, _, _ = module.sample_random_online_candidate(
        object(), changed_args, module.torch.device("cpu"), None, step=5,
    )

    assert module.torch.equal(first, second)
    assert first_seed == second_seed == module.stable_seed(23, "random_acquisition", 5)
    assert not module.torch.equal(first, changed_step)
    assert not module.torch.equal(first, changed_seed)
    assert module.torch.all(first[: module.ARCH_NZ].abs() <= args.z_bound)
    assert module.torch.all((first[module.ARCH_NZ :] >= 0) & (first[module.ARCH_NZ :] <= 1))
    assert random.getstate() == python_before
    after_numpy = module.np.random.get_state()
    assert numpy_before[0] == after_numpy[0]
    assert module.np.array_equal(numpy_before[1], after_numpy[1])
    assert numpy_before[2:] == after_numpy[2:]
    assert module.torch.equal(torch_before, module.torch.random.get_rng_state())


@runtime_only
def test_frozen_initialization_replays_exact_lhs_and_ignores_online_records(tmp_path, monkeypatch):
    module = _module()
    args = _args()
    source_records = _valid_records(module, args, include_bo=True)
    args.frozen_init_history = _write_history(tmp_path, source_records)
    monkeypatch.setattr(
        module,
        "eval_candidate",
        lambda *args, **kwargs: pytest.fail("frozen initialization must not evaluate candidates"),
    )

    history, X_obs, Y_obs, init_valid, predictions = module.replay_frozen_initialization(args)

    assert len(history) == args.n_init
    assert [record["step"] for record in history] == list(range(args.n_init))
    for tensor, record in zip(X_obs, source_records[: args.n_init]):
        assert tensor.tolist() == pytest.approx(record["z_search"])
    assert Y_obs == pytest.approx([record["val_acc"] for record in source_records[: args.n_init]])
    assert len(init_valid) == args.n_init
    for (_, _, mask), record in zip(init_valid, source_records[: args.n_init]):
        hp_mask = record["condition_mask_vector"]
        assert len(hp_mask) == 4
        assert len(mask) == module.ARCH_NZ + 4
        assert mask[: module.ARCH_NZ] == [1.0] * module.ARCH_NZ
        assert mask[module.ARCH_NZ :] == pytest.approx(hp_mask)
    assert len(predictions) == args.n_init
    assert all(record["initial_record_replayed"] is True for record in history)
    assert all(record["initial_history_source"] == args.frozen_init_history for record in history)
    assert all(record["candidate_selection_seed"] is None for record in history)


@runtime_only
def test_global4_four_dimensional_condition_mask_loads_successfully(tmp_path):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    args.frozen_init_history = _write_history(tmp_path, records)

    loaded = module.load_frozen_init_records(args)

    assert len(loaded[0]["condition_mask_vector"]) == 4


@runtime_only
def test_global4_search_dim_condition_mask_is_rejected_with_expected_length(tmp_path):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    records[0]["condition_mask_vector"] = [1.0] * (module.ARCH_NZ + 4)
    args.frozen_init_history = _write_history(tmp_path, records)

    with pytest.raises(
        ValueError,
        match=r"condition_mask_vector.*expected length 4",
    ):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize(
    ("hp_mode", "expected_mask_dim"),
    [("global4", 4), ("hybrid_cond7", 7), ("layer_cond19", 19)],
)
def test_frozen_condition_mask_dimension_matches_hp_mode(
    tmp_path, hp_mode, expected_mask_dim,
):
    module = _module()
    args = _args()
    args.hp_mode = hp_mode
    records = _valid_records(module, args)
    args.frozen_init_history = _write_history(tmp_path, records)

    history, _, _, init_valid, _ = module.replay_frozen_initialization(args)

    assert len(history[0]["condition_mask_vector"]) == expected_mask_dim
    replay_mask = init_valid[0][2]
    assert len(replay_mask) == module.ARCH_NZ + expected_mask_dim
    assert replay_mask[: module.ARCH_NZ] == [1.0] * module.ARCH_NZ
    assert replay_mask[module.ARCH_NZ :] == pytest.approx(
        history[0]["condition_mask_vector"]
    )


@runtime_only
@pytest.mark.parametrize("field", ["search_seed", "hp_mode", "search_dim"])
def test_frozen_history_rejects_mismatched_identity_fields(tmp_path, field):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    records[1][field] = {"search_seed": 99, "hp_mode": "layer_cond19", "search_dim": 999}[field]
    args.frozen_init_history = _write_history(tmp_path, records)
    with pytest.raises(ValueError, match=field):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_frozen_history_rejects_missing_or_duplicate_steps(tmp_path, case):
    module = _module()
    args = _args()
    records = _valid_records(module, args, include_bo=False)
    if case == "missing":
        records[1].pop("step")
    else:
        records[1]["step"] = records[0]["step"]
    args.frozen_init_history = _write_history(tmp_path, records)
    with pytest.raises(ValueError, match="step"):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize("field", ["z_search", "val_acc"])
def test_frozen_history_rejects_non_finite_values(tmp_path, field):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    records[0][field] = [math.nan] * 16 if field == "z_search" else math.inf
    args.frozen_init_history = _write_history(tmp_path, records)
    with pytest.raises(ValueError, match=field):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize("field", ["decoder_seed", "candidate_eval_seed", "evaluation_seed"])
def test_frozen_history_rejects_mismatched_derived_seeds(tmp_path, field):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    records[2][field] += 1
    args.frozen_init_history = _write_history(tmp_path, records)
    with pytest.raises(ValueError, match=field):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_evaluation_seed_scheme", "legacy_stage_seed_v0"),
        ("candidate_evaluation_seed_scheme", None),
        ("candidate_fingerprint", "f" * 64),
    ],
)
def test_frozen_history_rejects_legacy_scheme_or_tampered_fingerprint(
    tmp_path, field, value,
):
    module = _module()
    args = _args()
    records = _valid_records(module, args)
    if value is None:
        records[0].pop(field)
    else:
        records[0][field] = value
    args.frozen_init_history = _write_history(tmp_path, records)
    with pytest.raises(ValueError, match=field):
        module.load_frozen_init_records(args)


@runtime_only
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gp_init_mode": "checkpoint"}, "gp_init_mode"),
        ({"warm_start": "warm.pt"}, "warm_start"),
        ({"gmm_init_history": ["old.json"]}, "gmm_init_history"),
        ({"gmm_init_trials": 1}, "gmm_init_trials"),
    ],
)
def test_frozen_history_rejects_conflicting_initialization_modes(changes, message):
    module = _module()
    args = _args("history.json")
    for key, value in changes.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        module.validate_frozen_init_configuration(args)
