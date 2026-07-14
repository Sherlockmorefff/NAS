from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
import statistics
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


class _Array(list):
    @property
    def size(self) -> int:
        return len(self)


def _load_final_eval_with_dependency_stubs():
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.float64 = float
    numpy_stub.asarray = lambda values, dtype=None: _Array(values)
    numpy_stub.mean = statistics.fmean
    numpy_stub.std = statistics.pstdev
    numpy_stub.min = min
    numpy_stub.max = max
    numpy_stub.random = SimpleNamespace(seed=lambda seed: None)

    torch_stub = types.ModuleType("torch")
    torch_stub.manual_seed = lambda seed: None
    torch_stub.device = lambda value: value
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)

    transforms_stub = types.ModuleType("torch_geometric.transforms")
    transforms_stub.NormalizeFeatures = lambda: object()
    datasets_stub = types.ModuleType("torch_geometric.datasets")
    datasets_stub.Planetoid = object
    torch_geometric_stub = types.ModuleType("torch_geometric")
    torch_geometric_stub.transforms = transforms_stub
    torch_geometric_stub.datasets = datasets_stub

    eval_utils_stub = types.ModuleType("eval_utils")
    eval_utils_stub.DEFAULT_GAT_HEADS = 1
    eval_utils_stub.DEFAULT_SAGE_AGGR = "mean"
    eval_utils_stub.DEFAULT_GIN_EPS = 0.0
    eval_utils_stub.train_and_eval_arch = lambda **kwargs: (0.0, False, 0.0)

    hp_modes_stub = types.ModuleType("hp_modes")
    hp_modes_stub.validate_hp_mode = lambda hp_mode: hp_mode

    stubs = {
        "numpy": numpy_stub,
        "torch": torch_stub,
        "torch_geometric": torch_geometric_stub,
        "torch_geometric.transforms": transforms_stub,
        "torch_geometric.datasets": datasets_stub,
        "eval_utils": eval_utils_stub,
        "hp_modes": hp_modes_stub,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_final_eval_history_test_module", ROOT / "final_eval.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


final_eval = _load_final_eval_with_dependency_stubs()


def _record(
    step: int,
    val_acc: float,
    *,
    valid: bool = True,
    operations: list[str] | None = None,
    lr: float = 0.01,
    use_weight_decay: bool = False,
) -> dict:
    record = {
        "step": step,
        "valid": valid,
        "val_acc": val_acc,
        "operations": operations or ["GCNConv", "Identity", "Identity"],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "lr": lr,
        "dropout": 0.5,
        "hidden_dim": 64,
    }
    if use_weight_decay:
        record["weight_decay"] = 5e-4
    else:
        record["l2"] = 5e-4
    return record


def _write_history(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "history_final.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_load_history_records_normalizes_defaults_and_weight_decay(tmp_path):
    path = _write_history(
        tmp_path,
        [
            {
                **_record(3, 0.8, use_weight_decay=True),
                "z_search": [0.1, 0.2],
                "gp_pred_mean": 0.79,
                "gp_stage": "online_bo",
            }
        ],
    )

    records = final_eval.load_history_records(str(path))

    assert len(records) == 1
    assert records[0]["l2"] == pytest.approx(5e-4)
    assert records[0]["gat_heads"] == final_eval.DEFAULT_GAT_HEADS
    assert records[0]["sage_aggr"] == final_eval.DEFAULT_SAGE_AGGR
    assert records[0]["gin_eps"] == final_eval.DEFAULT_GIN_EPS
    assert records[0]["z_search"] == [0.1, 0.2]
    assert records[0]["gp_pred_mean"] == pytest.approx(0.79)
    assert records[0]["gp_stage"] == "online_bo"


def test_select_history_records_sorts_top_k_and_skips_invalid(tmp_path):
    path = _write_history(
        tmp_path,
        [
            _record(1, 0.75),
            _record(2, 0.99, valid=False),
            _record(3, 0.82, operations=["GATConv", "Identity", "Identity"]),
            _record(4, 0.78, operations=["SAGEConv", "Identity", "Identity"]),
        ],
    )

    selected = final_eval.select_history_records(
        final_eval.load_history_records(str(path)), top_k=2
    )

    assert [record["search_step"] for record in selected] == [3, 4]
    assert [record["search_rank"] for record in selected] == [1, 2]
    assert [record["search_val_acc"] for record in selected] == [0.82, 0.78]
    assert all(record["source"] == "history" for record in selected)


def test_select_history_records_uses_requested_step_order(tmp_path):
    path = _write_history(
        tmp_path,
        [
            _record(78, 0.82, operations=["GCNConv", "Identity", "Identity"]),
            _record(88, 0.81, operations=["GATConv", "Identity", "Identity"]),
            _record(113, 0.83, operations=["SAGEConv", "Identity", "Identity"]),
        ],
    )

    selected = final_eval.select_history_records(
        final_eval.load_history_records(str(path)),
        top_k=3,
        history_steps=[88, 78, 113],
    )

    assert [record["search_step"] for record in selected] == [88, 78, 113]
    assert [record["search_rank"] for record in selected] == [1, 2, 3]


def test_select_history_records_deduplicates_architecture_and_hp(tmp_path):
    duplicate_low = _record(4, 0.70)
    duplicate_high = _record(7, 0.85)
    distinct_hp = _record(9, 0.80, lr=0.005)
    path = _write_history(tmp_path, [duplicate_low, distinct_hp, duplicate_high])
    records = final_eval.load_history_records(str(path))

    deduplicated = final_eval.select_history_records(records, top_k=3, deduplicate=True)
    retained = final_eval.select_history_records(records, top_k=3, deduplicate=False)

    assert [record["search_step"] for record in deduplicated] == [7, 9]
    assert [record["search_step"] for record in retained] == [7, 9, 4]


def test_cli_requires_history_and_has_no_decode_options():
    with pytest.raises(SystemExit):
        final_eval.parse_args([])

    args = final_eval.parse_args(["--history_path", "history_final.json"])
    assert args.history_path == "history_final.json"
    assert args.top_k == 10
    assert args.deduplicate is True
    assert args.sort_by == "val_acc"
    assert args.n_seeds == 10
    assert args.seed_start == 0
    assert args.eval_epochs == 300
    assert args.patience == 80
    for removed_option in (
        "checkpoint",
        "auto_best_z",
        "disable_auto_best",
        "auto_name",
        "auto_decode_trials",
    ):
        assert not hasattr(args, removed_option)


def test_module_contains_no_vae_decode_path():
    source = Path(final_eval.__file__).read_text(encoding="utf-8")
    for removed_symbol in (
        "load_vae",
        "decode_arch_from_z",
        "build_auto_candidates",
        "JointSpaceVAE",
        "decode_hp_by_mode",
        "auto_best_z",
    ):
        assert removed_symbol not in source


def test_evaluate_candidate_aggregates_valid_seeds_and_keeps_invalid(monkeypatch):
    calls: list[dict] = []
    outcomes = iter(
        [
            (0.80, True, 0.70),
            (0.00, False, 0.00),
            (0.90, True, 0.80),
        ]
    )

    def fake_train_and_eval_arch(**kwargs):
        calls.append(kwargs)
        return next(outcomes)

    monkeypatch.setattr(final_eval, "train_and_eval_arch", fake_train_and_eval_arch)
    candidate = {
        "candidate_rank": 1,
        "source": "history",
        "name": "history_rank_001_step_78",
        "search_step": 78,
        "search_val_acc": 0.82,
        "operations": ["GATConv", "SAGEConv", "Identity"],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "lr": 0.01,
        "dropout": 0.4,
        "hidden_dim": 32,
        "l2": 1e-4,
        "gat_heads": 4,
        "sage_aggr": "max",
        "gin_eps": 0.1,
        "gat_heads_by_layer": [4, 1, 1],
        "sage_aggr_by_layer": ["mean", "max", "mean"],
        "gin_eps_by_layer": [0.0, 0.0, 0.0],
    }
    args = SimpleNamespace(
        n_seeds=3,
        seed_start=5,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=300,
        patience=80,
    )
    logger = logging.getLogger("test_final_eval_history_records")

    aggregate, per_seed = final_eval.evaluate_candidate(
        candidate,
        data=object(),
        in_ch=8,
        out_ch=3,
        args=args,
        device="cpu",
        logger=logger,
    )

    assert [row["seed"] for row in per_seed] == [5, 6, 7]
    assert [row["valid"] for row in per_seed] == [True, False, True]
    assert aggregate["n_valid"] == 2
    assert aggregate["n_invalid"] == 1
    assert aggregate["n_seeds"] == 3
    assert aggregate["val_mean"] == pytest.approx(0.85)
    assert aggregate["val_std"] == pytest.approx(0.05)
    assert aggregate["test_mean"] == pytest.approx(0.75)
    assert aggregate["test_std"] == pytest.approx(0.05)
    assert math.isclose(aggregate["val_min"], 0.8)
    assert math.isclose(aggregate["val_max"], 0.9)
    assert set(final_eval.PER_SEED_FIELDS).issubset(per_seed[0])
    assert all(call["track_test"] is True for call in calls)
    assert [call["seed"] for call in calls] == [5, 6, 7]
    assert calls[0]["config"] == {
        "operations": ["GATConv", "SAGEConv", "Identity"],
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "effective_layers": 2,
    }
    assert calls[0]["gat_heads_by_layer"] == [4, 1, 1]
    assert calls[0]["sage_aggr_by_layer"] == ["mean", "max", "mean"]
    assert calls[0]["gin_eps_by_layer"] == [0.0, 0.0, 0.0]
