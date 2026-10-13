"""Parse Cora None/LayerNorm pilot histories without running training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from layernorm_search_space import (
    NORMALIZATION_LAYER_NORM,
    NORMALIZATION_NONE,
    top_normalization_statistics,
)
from surrogate.checkpoint_io import atomic_json_dump


TRAJECTORY_CHECKPOINTS = (50, 75, 100, 125, 150)
VALIDATION_LEVELS = (0.80, 0.81, 0.82)
CONFIRMATION_METHODS = ("original", "expanded", "archive_best")


def _best_so_far(records: Sequence[Mapping[str, Any]]) -> list[float]:
    values = []
    best = float("-inf")
    for row in records:
        best = max(best, float(row["val_acc"]))
        values.append(best)
    return values


def summarize_history(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = [
        dict(row)
        for row in records
        if row.get("evaluation_fidelity", "full") == "full"
    ]
    if not full:
        raise ValueError("history has no full-fidelity records")
    required = {
        "val_acc",
        "normalization",
        "base_candidate_fingerprint",
        "full_candidate_fingerprint",
    }
    return _summarize_history_impl(full, required)


def summarize_confirmation(
    rows: Sequence[Mapping[str, Any]],
    common_seeds: Sequence[int],
    *,
    practical_validation_delta: float = 0.004,
    historical_observed_ceiling: float = 0.828,
) -> dict[str, Any]:
    """Answer the frozen final questions using validation-first comparisons."""

    seeds = [int(seed) for seed in common_seeds]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("confirmation requires exactly five unique common seeds")
    if len(rows) != 7:
        raise ValueError("confirmation requires six NAS winners and one archive best")
    grouped: dict[str, list[dict[str, Any]]] = {
        method: [] for method in CONFIRMATION_METHODS
    }
    for raw in rows:
        row = dict(raw)
        method = str(row.get("method"))
        if method not in grouped:
            raise ValueError(f"unknown confirmation method: {method}")
        per_seed = row.get("per_seed")
        if not isinstance(per_seed, list) or len(per_seed) != 5:
            raise ValueError("each confirmation candidate requires five seed results")
        if [int(item["seed"]) for item in per_seed] != seeds:
            raise ValueError("all confirmation candidates must use identical ordered seeds")
        for item in per_seed:
            for field in ("validation_accuracy", "test_accuracy"):
                value = float(item[field])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"invalid {field}: {value}")
        row["validation_mean"] = sum(
            float(item["validation_accuracy"]) for item in per_seed
        ) / 5.0
        row["test_mean"] = sum(
            float(item["test_accuracy"]) for item in per_seed
        ) / 5.0
        grouped[method].append(row)
    if [len(grouped[name]) for name in CONFIRMATION_METHODS] != [3, 3, 1]:
        raise ValueError("confirmation must contain 3 original, 3 expanded, 1 archive")
    original = {int(row["search_seed"]): row for row in grouped["original"]}
    expanded = {int(row["search_seed"]): row for row in grouped["expanded"]}
    if set(original) != set(expanded) or len(original) != 3:
        raise ValueError("original and expanded winners must share three search seeds")

    pair_rows = []
    layernorm_responder = False
    for search_seed in sorted(original):
        left = original[search_seed]
        right = expanded[search_seed]
        validation_deltas = [
            float(r["validation_accuracy"]) - float(l["validation_accuracy"])
            for l, r in zip(left["per_seed"], right["per_seed"])
        ]
        test_deltas = [
            float(r["test_accuracy"]) - float(l["test_accuracy"])
            for l, r in zip(left["per_seed"], right["per_seed"])
        ]
        validation_delta = sum(validation_deltas) / 5.0
        is_layernorm_responder = (
            right.get("normalization") == NORMALIZATION_LAYER_NORM
            and validation_delta >= practical_validation_delta
            and sum(value > 0 for value in validation_deltas) >= 3
        )
        layernorm_responder = layernorm_responder or is_layernorm_responder
        pair_rows.append(
            {
                "search_seed": search_seed,
                "expanded_normalization": right.get("normalization"),
                "validation_delta": validation_delta,
                "positive_validation_seeds": sum(value > 0 for value in validation_deltas),
                "test_delta": sum(test_deltas) / 5.0,
                "positive_test_seeds": sum(value > 0 for value in test_deltas),
            }
        )
    overall_original_delta = sum(row["validation_delta"] for row in pair_rows) / 3.0
    beats_original = (
        overall_original_delta >= practical_validation_delta
        and sum(row["validation_delta"] > 0 for row in pair_rows) >= 2
    )

    # This selection is validation-only. Test is read only for the frozen candidate.
    best_expanded = sorted(
        grouped["expanded"],
        key=lambda row: (-float(row["validation_mean"]), str(row["candidate_id"])),
    )[0]
    archive = grouped["archive_best"][0]
    archive_validation_deltas = [
        float(new["validation_accuracy"]) - float(old["validation_accuracy"])
        for new, old in zip(best_expanded["per_seed"], archive["per_seed"])
    ]
    archive_test_deltas = [
        float(new["test_accuracy"]) - float(old["test_accuracy"])
        for new, old in zip(best_expanded["per_seed"], archive["per_seed"])
    ]
    archive_validation_delta = sum(archive_validation_deltas) / 5.0
    archive_test_delta = sum(archive_test_deltas) / 5.0
    beats_archive = (
        archive_validation_delta >= practical_validation_delta
        and sum(value > 0 for value in archive_validation_deltas) >= 3
    )
    breaks_ceiling = (
        beats_archive
        and float(best_expanded["validation_mean"]) > historical_observed_ceiling
        and archive_test_delta >= 0.0
        and sum(value > 0 for value in archive_test_deltas) >= 3
    )
    return {
        "selection_endpoint": "confirmation validation mean only",
        "selected_expanded_candidate_id": best_expanded["candidate_id"],
        "selected_expanded_normalization": best_expanded.get("normalization"),
        "layernorm_effective": bool(layernorm_responder and beats_original),
        "beats_original": bool(beats_original),
        "beats_historical_archive_best": bool(beats_archive),
        "breaks_historical_ceiling": bool(breaks_ceiling),
        "historical_observed_ceiling": float(historical_observed_ceiling),
        "expanded_minus_original_validation": overall_original_delta,
        "expanded_minus_archive_validation": archive_validation_delta,
        "expanded_minus_archive_test": archive_test_delta,
        "search_seed_pairs": pair_rows,
        "test_used_for_selection": False,
    }


def _summarize_history_impl(
    full: Sequence[Mapping[str, Any]], required: set[str]
) -> dict[str, Any]:
    for index, row in enumerate(full):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"history record {index} is missing {missing}")
    trajectory = _best_so_far(full)
    best_index = max(
        range(len(full)),
        key=lambda index: (
            float(full[index]["val_acc"]),
            -index,
            str(full[index]["full_candidate_fingerprint"]),
        ),
    )
    online = [row for row in full if row.get("type") == "bo"]
    late = online[-50:]
    diagnostics = [
        attempt["normalization_diagnostics"]
        for row in online
        for attempt in row.get("proposal_attempts", [])
        if attempt.get("normalization_diagnostics") is not None
        and not attempt.get("duplicate", False)
    ]
    posterior_differences = [
        float(row["posterior_mean_difference"]) for row in diagnostics
    ]
    lengthscales = [
        float(row["normalization_lengthscale_after"])
        for row in online
        if row.get("normalization_lengthscale_after") is not None
    ]
    return {
        "evaluation_count": len(full),
        "best_validation": float(full[best_index]["val_acc"]),
        "best_found_evaluation": best_index + 1,
        "best_normalization": full[best_index]["normalization"],
        "trajectory": {
            str(checkpoint): trajectory[checkpoint - 1]
            for checkpoint in TRAJECTORY_CHECKPOINTS
            if checkpoint <= len(trajectory)
        },
        "best_so_far_auc": sum(trajectory) / len(trajectory),
        "validation_level_first_evaluation": {
            f"{level:.2f}": next(
                (index + 1 for index, value in enumerate(trajectory) if value >= level),
                None,
            )
            for level in VALIDATION_LEVELS
        },
        "overall_normalization": top_normalization_statistics(full, top_k=10),
        "online_normalization": top_normalization_statistics(online, top_k=10),
        "late_online_normalization": top_normalization_statistics(late, top_k=10),
        "interaction_diagnostics": {
            "paired_acquisition_comparisons": len(diagnostics),
            "mean_abs_posterior_difference": (
                None
                if not posterior_differences
                else sum(abs(value) for value in posterior_differences)
                / len(posterior_differences)
            ),
            "last_normalization_lengthscale": (
                None if not lengthscales else lengthscales[-1]
            ),
            "min_normalization_lengthscale": (
                None if not lengthscales else min(lengthscales)
            ),
            "max_normalization_lengthscale": (
                None if not lengthscales else max(lengthscales)
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history")
    source.add_argument("--confirmation")
    parser.add_argument("--confirmation-seed-manifest")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.history is not None:
        summary = summarize_history(
            json.loads(Path(args.history).read_text(encoding="utf-8"))
        )
    else:
        if args.confirmation_seed_manifest is None:
            raise ValueError(
                "--confirmation requires --confirmation-seed-manifest"
            )
        seed_manifest = json.loads(
            Path(args.confirmation_seed_manifest).read_text(encoding="utf-8")
        )
        summary = summarize_confirmation(
            json.loads(Path(args.confirmation).read_text(encoding="utf-8")),
            seed_manifest["seeds"],
        )
    atomic_json_dump(summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
