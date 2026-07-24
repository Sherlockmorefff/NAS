from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import pytest

from test_final_eval_history_records import final_eval


def _test_stable_seed(base_seed: int, namespace: str, *components) -> int:
    payload = json.dumps(
        [int(base_seed), namespace, *components],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


@pytest.fixture(autouse=True)
def _stable_seed_implementation(monkeypatch):
    monkeypatch.setattr(final_eval, "stable_seed", _test_stable_seed)


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record(
    label: str,
    step: int,
    val_acc: float,
    *,
    search_seed: int = 0,
    valid: bool = True,
    z_search=None,
) -> dict:
    record = {
        "step": step,
        "valid": valid,
        "val_acc": val_acc,
        "operations": [f"GCNConv_{label}", "Identity", "Identity"],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 64,
        "l2": 5e-4,
        "search_seed": search_seed,
        "candidate_fingerprint": _fingerprint(label),
    }
    if z_search is not None:
        record["z_search"] = z_search
    return record


def _write_history(tmp_path: Path, records: list[dict], name: str = "history.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _formal_args(
    history_path: Path,
    output: Path,
    *,
    final_base_seed: int = 20260724,
    search_seed: int = 0,
    n_replicates: int = 2,
    top_k: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    patience: int = 80,
):
    argv = [
        "--history_path",
        str(history_path),
        "--output",
        str(output),
        "--method_label",
        "schur",
        "--search_seed",
        str(search_seed),
        "--final_base_seed",
        str(final_base_seed),
        "--n_replicates",
        str(n_replicates),
        "--top_k",
        str(top_k),
        "--patience",
        str(patience),
    ]
    if dry_run:
        argv.append("--dry_run")
    if resume:
        argv.append("--resume")
    return final_eval.parse_args(argv)


def _candidate(label: str = "candidate") -> dict:
    return {
        "candidate_rank": 1,
        "source": "history",
        "name": "history_rank_001_step_7",
        "search_step": 7,
        "search_val_acc": 0.8,
        "candidate_fingerprint": _fingerprint(label),
        "operations": ["GCNConv", "Identity", "Identity"],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 64,
        "l2": 5e-4,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
    }


def test_final_seed_is_deterministic_and_changes_with_each_variable_input():
    fingerprint = _fingerprint("same")
    seed = final_eval.final_evaluation_seed(123, fingerprint, 0)

    assert seed == final_eval.final_evaluation_seed(123, fingerprint, 0)
    assert seed != final_eval.final_evaluation_seed(124, fingerprint, 0)
    assert seed != final_eval.final_evaluation_seed(123, _fingerprint("other"), 0)
    assert seed != final_eval.final_evaluation_seed(123, fingerprint, 1)


def test_method_search_context_stage_rank_path_and_order_do_not_affect_seed():
    fingerprint = _fingerprint("shared")
    first_args = final_eval.parse_args(
        [
            "--history_path",
            "/a/history.json",
            "--method_label",
            "schur",
            "--search_seed",
            "0",
            "--final_base_seed",
            "55",
        ]
    )
    second_args = final_eval.parse_args(
        [
            "--history_path",
            "/different/history.json",
            "--method_label",
            "gmm_exp150",
            "--search_seed",
            "4",
            "--final_base_seed",
            "55",
        ]
    )
    first_candidate = {
        **_candidate("shared"),
        "candidate_rank": 1,
        "search_step": 1,
    }
    second_candidate = {
        **_candidate("shared"),
        "candidate_rank": 10,
        "search_step": 299,
        "search_gp_stage": "different_stage",
    }

    seeds = [
        final_eval._seed_for_replicate(first_args, first_candidate, 3),
        final_eval._seed_for_replicate(second_args, second_candidate, 3),
    ]

    assert seeds[0] == seeds[1]
    assert seeds[0] == final_eval.final_evaluation_seed(55, fingerprint, 3)
    assert final_eval.FINAL_EVALUATION_SEED_SCHEME == (
        "final_base_candidate_fingerprint_replicate_v1"
    )


def test_formal_history_rejects_missing_invalid_and_mismatched_fingerprint(
    tmp_path, monkeypatch
):
    missing = _record("missing", 1, 0.8)
    missing.pop("candidate_fingerprint")
    records = final_eval.load_history_records(
        str(_write_history(tmp_path, [missing], "missing.json"))
    )
    with pytest.raises(ValueError, match="candidate_fingerprint"):
        final_eval.validate_formal_history(records, 0)

    invalid = _record("invalid", 1, 0.8)
    invalid["candidate_fingerprint"] = "not-sha256"
    records = final_eval.load_history_records(
        str(_write_history(tmp_path, [invalid], "invalid.json"))
    )
    with pytest.raises(ValueError, match="candidate_fingerprint"):
        final_eval.validate_formal_history(records, 0)

    mismatch = _record("mismatch", 1, 0.8, z_search=[0.1, 0.2])
    records = final_eval.load_history_records(
        str(_write_history(tmp_path, [mismatch], "mismatch.json"))
    )
    monkeypatch.setattr(
        final_eval, "_existing_candidate_fingerprint", lambda value: _fingerprint("z")
    )
    with pytest.raises(ValueError, match="does not match z_search"):
        final_eval.validate_formal_history(records, 0)


def test_top_k_is_selected_independently_per_history_with_deterministic_ties(tmp_path):
    first = final_eval.load_history_records(
        str(
            _write_history(
                tmp_path,
                [
                    _record("a", 9, 0.8),
                    _record("b", 2, 0.8),
                    _record("c", 3, 0.7),
                ],
                "first.json",
            )
        )
    )
    second = final_eval.load_history_records(
        str(
            _write_history(
                tmp_path,
                [_record("x", 1, 0.95), _record("y", 2, 0.6)],
                "second.json",
            )
        )
    )

    first_selected = final_eval.select_history_records(first, top_k=1)
    second_selected = final_eval.select_history_records(second, top_k=1)

    assert first_selected[0]["candidate_fingerprint"] == _fingerprint("b")
    assert second_selected[0]["candidate_fingerprint"] == _fingerprint("x")


def test_replicate_seed_and_metadata_are_written_to_json_and_csv(
    tmp_path, monkeypatch
):
    history = _write_history(tmp_path, [_record("candidate", 7, 0.8)])
    args = _formal_args(history, tmp_path / "out", n_replicates=1)
    calls: list[dict] = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return 0.81, True, 0.73, {
            "best_epoch": 12,
            "stopped_epoch": 20,
            "epochs_ran": 20,
        }

    monkeypatch.setattr(final_eval, "train_and_eval_arch", fake_train)
    aggregate, rows = final_eval.evaluate_candidate(
        _candidate(),
        data=object(),
        in_ch=8,
        out_ch=3,
        args=args,
        device="cpu",
        logger=logging.getLogger("seed-fields"),
    )
    json_path = tmp_path / "rows.json"
    final_eval.atomic_json_dump(rows, json_path)
    csv_path = tmp_path / "rows.csv"
    final_eval.write_csv(csv_path, rows, final_eval.PER_REPLICATE_FIELDS)
    csv_row = next(csv.DictReader(csv_path.open(encoding="utf-8")))

    expected = final_eval.final_evaluation_seed(
        args.final_base_seed, _fingerprint("candidate"), 0
    )
    assert calls[0]["seed"] == expected
    assert calls[0]["return_metadata"] is True
    assert rows[0]["replicate_id"] == 0
    assert rows[0]["final_eval_seed"] == expected
    assert rows[0]["best_epoch"] == 12
    assert rows[0]["epochs_ran"] == 20
    assert aggregate["n_valid"] == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))[0][
        "final_eval_seed"
    ] == expected
    assert int(csv_row["replicate_id"]) == 0
    assert int(csv_row["final_eval_seed"]) == expected


def test_final_candidate_selection_uses_val_mean_not_test_mean():
    results = [
        {
            "candidate_rank": 1,
            "candidate_fingerprint": _fingerprint("high-val"),
            "val_mean": 0.9,
            "test_mean": 0.1,
            "n_valid": 3,
        },
        {
            "candidate_rank": 2,
            "candidate_fingerprint": _fingerprint("high-test"),
            "val_mean": 0.8,
            "test_mean": 0.99,
            "n_valid": 3,
        },
    ]

    selected = final_eval.select_by_final_validation(results, 3)

    assert selected["candidate_rank"] == 1
    assert final_eval.select_by_final_validation(
        [{**results[0], "n_valid": 2}], 3
    ) is None


def test_invalid_replicate_is_kept_and_never_replaced_with_a_new_seed(
    tmp_path, monkeypatch
):
    history = _write_history(tmp_path, [_record("candidate", 7, 0.8)])
    args = _formal_args(history, tmp_path / "out", n_replicates=2)
    calls: list[int] = []

    def fake_train(**kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 1:
            return 0.0, False, 0.0, {
                "best_epoch": None,
                "stopped_epoch": 2,
                "epochs_ran": 2,
            }
        return 0.8, True, 0.7, {
            "best_epoch": 3,
            "stopped_epoch": 4,
            "epochs_ran": 4,
        }

    monkeypatch.setattr(final_eval, "train_and_eval_arch", fake_train)
    aggregate, rows = final_eval.evaluate_candidate(
        _candidate(),
        object(),
        8,
        3,
        args,
        "cpu",
        logging.getLogger("invalid-seed"),
    )

    assert len(calls) == 2
    assert calls == [row["final_eval_seed"] for row in rows]
    assert [row["replicate_id"] for row in rows] == [0, 1]
    assert rows[0]["valid"] is False
    assert aggregate["n_invalid"] == 1


def test_resume_skips_completed_replicates_and_rejects_config_mismatch(
    tmp_path, monkeypatch
):
    history = _write_history(tmp_path, [_record("candidate", 7, 0.8)])
    output = tmp_path / "formal"
    calls: list[int] = []
    monkeypatch.setattr(final_eval, "load_cora", lambda root, device: (object(), 8, 3))

    def fake_train(**kwargs):
        calls.append(kwargs["seed"])
        return 0.8, True, 0.7, {
            "best_epoch": 2,
            "stopped_epoch": 3,
            "epochs_ran": 3,
        }

    monkeypatch.setattr(final_eval, "train_and_eval_arch", fake_train)
    logger = logging.getLogger("resume")
    final_eval.run_final_evaluation(
        _formal_args(history, output, n_replicates=2), logger
    )
    assert len(calls) == 2

    final_eval.run_final_evaluation(
        _formal_args(history, output, n_replicates=2, resume=True), logger
    )
    assert len(calls) == 2

    with pytest.raises(ValueError, match="resume configuration mismatch"):
        final_eval.run_final_evaluation(
            _formal_args(
                history,
                output,
                n_replicates=2,
                resume=True,
                patience=79,
            ),
            logger,
        )


def test_dry_run_generates_manifest_without_loading_dataset_or_training(
    tmp_path, monkeypatch
):
    history = _write_history(tmp_path, [_record("candidate", 7, 0.8)])
    output = tmp_path / "preflight"
    monkeypatch.setattr(
        final_eval,
        "load_cora",
        lambda *args, **kwargs: pytest.fail("dry-run loaded Cora"),
    )
    monkeypatch.setattr(
        final_eval,
        "train_and_eval_arch",
        lambda **kwargs: pytest.fail("dry-run started training"),
    )

    summary = final_eval.run_final_evaluation(
        _formal_args(history, output, n_replicates=2, dry_run=True),
        logging.getLogger("dry-run"),
    )
    manifest = json.loads(
        (output / "seed_manifest_global4.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "dry_run_preflight_passed"
    assert len(manifest) == 2
    assert [row["replicate_id"] for row in manifest] == [0, 1]
    assert all(row["candidate_fingerprint"] == _fingerprint("candidate") for row in manifest)


def test_legacy_seed_arguments_require_explicit_legacy_mode():
    with pytest.raises(SystemExit):
        final_eval.parse_args(
            ["--history_path", "history.json", "--seed_start", "12"]
        )

    args = final_eval.parse_args(
        [
            "--history_path",
            "history.json",
            "--legacy_sequential_seeds",
            "--seed_start",
            "12",
        ]
    )
    assert args.legacy_sequential_seeds is True
    assert args.seed_start == 12
