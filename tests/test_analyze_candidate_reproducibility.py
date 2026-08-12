from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyse import analyze_candidate_reproducibility as repro


def _fingerprints() -> list[str]:
    ordinary = [f"{index:064x}" for index in range(1, 15)]
    return [
        repro.MAX_DIFFERENCE_FINGERPRINT,
        repro.SHARED_BEST_FINGERPRINT,
        *ordinary,
    ]


def _record(
    fingerprint: str,
    index: int,
    *,
    evaluation_seed: int | None = None,
    val_acc: float = 0.8,
) -> dict:
    operations = ["GCNConv", "Identity"]
    edges = [[0, 1], [1, 2], [2, 3]]
    hp = {
        "lr": 0.01,
        "dropout": 0.25,
        "hidden_dim": 64,
        "weight_decay": 1e-4,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "condition_mask_vector": [1.0, 1.0, 1.0, 1.0],
    }
    return {
        "evaluation_fidelity": "full",
        "candidate_fingerprint": fingerprint,
        "z_search": [float(index), 0.5],
        "operations": operations,
        "edges": edges,
        "lr": hp["lr"],
        "dropout": hp["dropout"],
        "hidden_dim": hp["hidden_dim"],
        "l2": hp["weight_decay"],
        "gat_heads": hp["gat_heads"],
        "sage_aggr": hp["sage_aggr"],
        "gin_eps": hp["gin_eps"],
        "gat_heads_by_layer": hp["gat_heads_by_layer"],
        "sage_aggr_by_layer": hp["sage_aggr_by_layer"],
        "gin_eps_by_layer": hp["gin_eps_by_layer"],
        "condition_mask_vector": hp["condition_mask_vector"],
        "architecture_fingerprint": repro._architecture_fingerprint(
            operations, edges
        ),
        "hp_fingerprint": repro._hp_fingerprint(hp),
        "decoder_seed": 1000 + index,
        "evaluation_seed": (
            2000 + index if evaluation_seed is None else evaluation_seed
        ),
        "search_seed": 0,
        "candidate_evaluation_seed_scheme": "candidate-seed-v1",
        "seed_derivation": "sha256_v1",
        "full_evaluation_index": index,
        "val_acc": val_acc,
        "best_epoch": 10,
        "stopped_epoch": 20,
        "epochs_ran": 20,
    }


def _write_histories(
    tmp_path: Path,
    *,
    mismatch_fingerprint: str | None = None,
) -> tuple[Path, Path, dict[int, str]]:
    mapping: dict[int, str] = {}
    t1 = []
    c1 = []
    for index, fingerprint in enumerate(_fingerprints()):
        mapping[index] = fingerprint
        t1.append(_record(fingerprint, index, val_acc=0.75 + index * 1e-4))
        c1_seed = None
        if fingerprint == mismatch_fingerprint:
            c1_seed = 999999
        c1.append(
            _record(
                fingerprint,
                index,
                evaluation_seed=c1_seed,
                val_acc=0.76 + index * 1e-4,
            )
        )
    t1_path = tmp_path / "t1.json"
    c1_path = tmp_path / "c1.json"
    t1_path.write_text(json.dumps(t1), encoding="utf-8")
    c1_path.write_text(json.dumps(c1), encoding="utf-8")
    return t1_path, c1_path, mapping


def test_manifest_selection_is_fixed_equidistant_and_identity_checked(
    tmp_path,
    monkeypatch,
) -> None:
    t1_path, c1_path, mapping = _write_histories(tmp_path)
    monkeypatch.setattr(
        repro,
        "_candidate_fingerprint",
        lambda z_search: mapping[int(z_search[0])],
    )
    first = repro.build_candidate_manifest(t1_path, c1_path)
    second = repro.build_candidate_manifest(t1_path, c1_path)
    selected = [
        row["candidate_fingerprint"] for row in first["candidates"]
    ]

    assert first == second
    assert first["candidate_count"] == 10
    assert selected[:2] == list(repro.ANCHOR_FINGERPRINTS)
    assert len(set(selected)) == 10
    assert first["equidistant_positions"] == repro._equidistant_positions(
        first["equidistant_source_count"],
        8,
    )
    assert len(first["manifest_sha256"]) == 64
    assert all(
        row["historical"]["absolute_difference"] == pytest.approx(0.01)
        for row in first["candidates"]
    )


def test_manifest_rejects_cross_history_seed_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    fingerprint = repro.MAX_DIFFERENCE_FINGERPRINT
    t1_path, c1_path, mapping = _write_histories(
        tmp_path,
        mismatch_fingerprint=fingerprint,
    )
    monkeypatch.setattr(
        repro,
        "_candidate_fingerprint",
        lambda z_search: mapping[int(z_search[0])],
    )
    with pytest.raises(ValueError, match="identity or seed mismatch"):
        repro.build_candidate_manifest(t1_path, c1_path)


def test_validation_history_rejects_test_fields(tmp_path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps([{"candidate_fingerprint": "a" * 64, "test_acc": 0.9}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden test fields"):
        repro.load_search_validation_history(path)


def test_repeat_statistics_use_fresh_repeat_against_first_baseline() -> None:
    rows = []
    for repeat_id, value in enumerate((0.80, 0.80, 0.82)):
        rows.append(
            {
                "candidate_fingerprint": "a" * 64,
                "repeat_id": repeat_id,
                "return_code": 0,
                "valid": True,
                "validation_accuracy": value,
                "best_epoch": 10 + (repeat_id == 2),
                "epochs_ran": 20 + (repeat_id == 2),
                "wall_time_seconds": 2.0,
            }
        )
    for repeat_id, value in enumerate((0.70, 0.70, 0.70)):
        rows.append(
            {
                "candidate_fingerprint": "b" * 64,
                "repeat_id": repeat_id,
                "return_code": 0,
                "valid": True,
                "validation_accuracy": value,
                "best_epoch": 8,
                "epochs_ran": 18,
                "wall_time_seconds": 4.0,
            }
        )
    stats = repro.repeat_statistics(rows)
    assert stats["comparison_count"] == 4
    assert stats["exact_validation_comparison_ratio"] == pytest.approx(0.75)
    assert stats["validation_mae"] == pytest.approx(0.005)
    assert stats["validation_max_absolute_difference"] == pytest.approx(0.02)
    assert stats["validation_p90_absolute_difference"] == pytest.approx(0.014)
    assert stats["validation_signed_mean_difference"] == pytest.approx(0.005)
    assert stats["best_epoch_match_rate"] == pytest.approx(0.5)
    assert stats["epochs_ran_match_rate"] == pytest.approx(0.5)
    assert stats["mean_wall_time_seconds"] == pytest.approx(3.0)


def test_strict_pilot_requires_exact_validation_and_epoch_metadata() -> None:
    rows = []
    for fingerprint in repro.ANCHOR_FINGERPRINTS:
        for repeat_id in range(2):
            rows.append(
                {
                    "candidate_fingerprint": fingerprint,
                    "repeat_id": repeat_id,
                    "return_code": 0,
                    "valid": True,
                    "validation_accuracy": 0.8,
                    "best_epoch": 10,
                    "stopped_epoch": 20,
                    "epochs_ran": 20,
                }
            )
    succeeded, reason = repro._strict_pilot_succeeded(rows)
    assert succeeded is True
    assert reason == ""

    rows[-1]["epochs_ran"] = 21
    succeeded, reason = repro._strict_pilot_succeeded(rows)
    assert succeeded is False
    assert "not exactly repeatable" in reason


def test_diagnostic_has_no_vae_decode_or_test_output_path() -> None:
    source = Path(repro.__file__).read_text(encoding="utf-8")
    for forbidden_symbol in (
        "load_vae(",
        "eval_candidate(",
        "decode_arch(",
        "track_test=True",
    ):
        assert forbidden_symbol not in source
    assert all("test" not in field for field in repro.CSV_FIELDS)
