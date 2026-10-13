#!/usr/bin/env python3
"""Fixed paired analysis for a completed frozen-checkpoint benchmark reveal.

This analysis never loads a dataset or a checkpoint.  It accepts only the
one-time worker's 30 paired scalar scores.  The already established method-level
state is copied unchanged; the classification concerns the two frozen winner
configurations only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


METHOD_LEVEL_STATE = "VALIDATION_CONFIRMED_HIGH_REGIME_PRIOR"
EXPECTED_SEEDS = tuple(range(100000, 100030))
N_PAIRS = 30
T_CRITICAL_DF29_975 = 2.045229642132703
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20260827
PRACTICAL_MARGIN = 0.006

P_ADVANTAGE = "FROZEN_CANDIDATE_P_ADVANTAGE_ON_CORA_BENCHMARK"
P_NONINFERIOR = "FROZEN_CANDIDATE_P_NONINFERIOR_ON_CORA_BENCHMARK"
Q_ADVANTAGE = "FROZEN_CANDIDATE_Q_ADVANTAGE_ON_CORA_BENCHMARK"
INCONCLUSIVE = "FROZEN_CANDIDATE_BENCHMARK_INCONCLUSIVE"
IDENTICAL = "FINAL_CANDIDATES_IDENTICAL"
PROTOCOL_FAILURE = "EXECUTION_OR_PROTOCOL_FAILURE"


class AnalysisProtocolError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisProtocolError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisProtocolError(f"{label} is not finite")
    return result


def _load_reveal_worker_module() -> Any:
    name = "_frozen_reveal_worker_for_analysis"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("reveal_frozen_checkpoints_once.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AnalysisProtocolError("cannot load frozen reveal worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_worker_result_hash(document: Mapping[str, Any]) -> None:
    recorded = document.get("result_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise AnalysisProtocolError("worker result SHA-256 is missing or malformed")
    payload = dict(document)
    payload.pop("result_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual != recorded:
        raise AnalysisProtocolError("worker result SHA-256 does not match its content")


def _sample_summary(values: Sequence[float]) -> Mapping[str, Any]:
    if len(values) != N_PAIRS:
        raise AnalysisProtocolError("every summary requires exactly 30 values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half_width = T_CRITICAL_DF29_975 * sd / math.sqrt(N_PAIRS)
    return {
        "n": N_PAIRS,
        "mean": mean,
        "sd": sd,
        "ci95_t": [mean - half_width, mean + half_width],
        "df": 29,
        "t_critical_0_975": T_CRITICAL_DF29_975,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisProtocolError("cannot take percentile of empty values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Mapping[str, Any]:
    if len(deltas) != N_PAIRS:
        raise AnalysisProtocolError("paired bootstrap requires exactly 30 deltas")
    if replicates != BOOTSTRAP_REPLICATES:
        raise AnalysisProtocolError("bootstrap replicate count is frozen at 50,000")
    rng = random.Random(seed)
    samples: list[float] = []
    index_digest = hashlib.sha256()
    for _ in range(replicates):
        total = 0.0
        for _ in range(N_PAIRS):
            index = rng.randrange(N_PAIRS)
            index_digest.update(struct.pack("B", index))
            total += deltas[index]
        samples.append(total / N_PAIRS)
    samples.sort()
    return {
        "replicates": replicates,
        "seed": seed,
        "mean": statistics.fmean(samples),
        "ci95_percentile": [
            _percentile(samples, 0.025),
            _percentile(samples, 0.975),
        ],
        "paired_index_sha256": index_digest.hexdigest(),
        "role": "sensitivity_only_not_classification",
    }


def classify_frozen_candidates(
    *,
    mean_delta: float,
    ci95: Sequence[float],
    candidates_identical: bool,
) -> str:
    if candidates_identical:
        return IDENTICAL
    lower, upper = map(float, ci95)
    p_advantage = mean_delta >= PRACTICAL_MARGIN and lower > 0.0
    q_advantage = mean_delta <= -PRACTICAL_MARGIN and upper < 0.0
    if p_advantage:
        return P_ADVANTAGE
    if not p_advantage and not q_advantage and lower > -PRACTICAL_MARGIN:
        return P_NONINFERIOR
    if q_advantage:
        return Q_ADVANTAGE
    return INCONCLUSIVE


def _validate_and_extract(
    document: Mapping[str, Any], ready_bundle: Any
) -> tuple[list[float], list[float], bool]:
    if document.get("result_schema") != "one-time-frozen-checkpoint-reveal-result-v1":
        raise AnalysisProtocolError("unsupported reveal result schema")
    if document.get("status") != "REVEAL_COMPLETE":
        raise AnalysisProtocolError("reveal did not complete")
    if document.get("method_level_state") != METHOD_LEVEL_STATE:
        raise AnalysisProtocolError("method-level state changed")
    if document.get("immutable_ready_root") != ready_bundle.immutable_ready_root:
        raise AnalysisProtocolError("reveal result is not bound to the expected READY root")
    if document.get("all_physical_forward_counts_one") is not True:
        raise AnalysisProtocolError("not every checkpoint had exactly one forward")
    _verify_worker_result_hash(document)

    reveal_worker = _load_reveal_worker_module()
    claim_path = ready_bundle.root_dir / reveal_worker.CLAIM_RELATIVE
    failure_path = ready_bundle.root_dir / reveal_worker.FAILURE_RELATIVE
    if failure_path.exists():
        raise AnalysisProtocolError("terminal reveal failure record exists")
    if not claim_path.is_file():
        raise AnalysisProtocolError("fixed one-time claim is missing")
    with claim_path.open("r", encoding="utf-8") as handle:
        on_disk_claim = json.load(handle)
    embedded_claim = document.get("claim")
    if embedded_claim != on_disk_claim:
        raise AnalysisProtocolError("embedded claim differs from fixed on-disk claim")
    if (
        not isinstance(embedded_claim, dict)
        or embedded_claim.get("claim_schema")
        != "one-time-frozen-checkpoint-reveal-claim-v1"
        or embedded_claim.get("ready_immutable_root")
        != ready_bundle.immutable_ready_root
        or embedded_claim.get("irreversible") is not True
        or embedded_claim.get("fixed_claim_relative_path")
        != reveal_worker.CLAIM_RELATIVE
    ):
        raise AnalysisProtocolError("one-time claim identity is invalid")

    candidates = document.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {"P", "Q"}:
        raise AnalysisProtocolError("candidate identity is missing")
    if not isinstance(candidates["P"], dict) or not isinstance(candidates["Q"], dict):
        raise AnalysisProtocolError("candidate identity rows are malformed")
    p_fingerprint = candidates["P"].get("joint_fingerprint")
    q_fingerprint = candidates["Q"].get("joint_fingerprint")
    if p_fingerprint != ready_bundle.ready.get("P_joint_config_fingerprint"):
        raise AnalysisProtocolError("P fingerprint differs from READY")
    if q_fingerprint != ready_bundle.ready.get("Q_joint_config_fingerprint"):
        raise AnalysisProtocolError("Q fingerprint differs from READY")
    inferred_identical = p_fingerprint == q_fingerprint
    declared_identical = bool(document.get("final_candidates_identical", False))
    if inferred_identical != declared_identical:
        raise AnalysisProtocolError("identical-candidate declaration is inconsistent")

    raw_scores = document.get("checkpoint_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != 2 * N_PAIRS:
        raise AnalysisProtocolError("worker result must contain 60 logical checkpoint scores")
    score_by_key: dict[tuple[str, int], float] = {}
    score_sha_by_key: dict[tuple[str, int], str] = {}
    ready_by_key = {
        (row["role"], int(row["seed"])): row for row in ready_bundle.logical_records
    }
    fingerprint_by_role = {"P": p_fingerprint, "Q": q_fingerprint}
    for row in raw_scores:
        if not isinstance(row, dict):
            raise AnalysisProtocolError("checkpoint score row is malformed")
        role = row.get("method_role")
        seed = row.get("seed")
        if role not in {"P", "Q"} or not isinstance(seed, int):
            raise AnalysisProtocolError("checkpoint score role/seed is malformed")
        key = (role, seed)
        if key in score_by_key:
            raise AnalysisProtocolError("duplicate logical checkpoint score")
        if row.get("joint_fingerprint") != fingerprint_by_role[role]:
            raise AnalysisProtocolError("checkpoint score fingerprint is inconsistent")
        if row.get("physical_forward_count") != 1:
            raise AnalysisProtocolError("checkpoint score did not come from one forward")
        ready_row = ready_by_key.get(key)
        if ready_row is None:
            raise AnalysisProtocolError("checkpoint score has no READY inventory row")
        if row.get("checkpoint_sha256") != ready_row.get("checkpoint_file_sha256"):
            raise AnalysisProtocolError("checkpoint score SHA differs from READY")
        score = _finite_number(row.get("score"), "checkpoint score")
        if not 0.0 <= score <= 1.0:
            raise AnalysisProtocolError("checkpoint score is not a proportion in [0,1]")
        score_by_key[key] = score
        score_sha_by_key[key] = str(row["checkpoint_sha256"])
    expected_logical_keys = {
        (role, seed) for role in ("P", "Q") for seed in EXPECTED_SEEDS
    }
    if set(score_by_key) != expected_logical_keys:
        raise AnalysisProtocolError("checkpoint score role/seed grid is incomplete")
    expected_physical_count = N_PAIRS if declared_identical else 2 * N_PAIRS
    if document.get("physical_checkpoint_count") != expected_physical_count:
        raise AnalysisProtocolError("physical checkpoint count is inconsistent")
    if document.get("logical_checkpoint_count") != 2 * N_PAIRS:
        raise AnalysisProtocolError("logical checkpoint count is inconsistent")
    if declared_identical and any(
        score_sha_by_key[("P", seed)] != score_sha_by_key[("Q", seed)]
        for seed in EXPECTED_SEEDS
    ):
        raise AnalysisProtocolError("identical P/Q roles did not share checkpoint SHA")

    raw_pairs = document.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != N_PAIRS:
        raise AnalysisProtocolError("reveal must contain exactly 30 pairs")
    by_seed: dict[int, tuple[float, float]] = {}
    for row in raw_pairs:
        if not isinstance(row, dict):
            raise AnalysisProtocolError("pair row is malformed")
        seed = row.get("seed")
        if not isinstance(seed, int) or seed in by_seed:
            raise AnalysisProtocolError("pair seed is invalid or duplicated")
        p_score = _finite_number(row.get("P_score"), "P score")
        q_score = _finite_number(row.get("Q_score"), "Q score")
        recorded_delta = _finite_number(row.get("delta"), "paired delta")
        if not math.isclose(recorded_delta, p_score - q_score, rel_tol=0.0, abs_tol=1e-15):
            raise AnalysisProtocolError("recorded paired delta is inconsistent")
        if not math.isclose(
            p_score,
            score_by_key[("P", seed)],
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            q_score,
            score_by_key[("Q", seed)],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise AnalysisProtocolError("paired row disagrees with checkpoint scores")
        by_seed[seed] = (p_score, q_score)
    if sorted(by_seed) != list(EXPECTED_SEEDS):
        raise AnalysisProtocolError("paired seed set is not exactly 100000..100029")
    p_values = [by_seed[seed][0] for seed in EXPECTED_SEEDS]
    q_values = [by_seed[seed][1] for seed in EXPECTED_SEEDS]
    if declared_identical and any(p != q for p, q in zip(p_values, q_values)):
        raise AnalysisProtocolError("identical candidates produced different paired scores")
    return p_values, q_values, declared_identical


def analyse_reveal_result(
    document: Mapping[str, Any], *, ready_bundle: Any | None = None
) -> Mapping[str, Any]:
    """Return the frozen analysis, or a protocol-failure classification."""

    try:
        if ready_bundle is None:
            raise AnalysisProtocolError("externally verified READY bundle is required")
        p_values, q_values, candidates_identical = _validate_and_extract(
            document, ready_bundle
        )
        deltas = [p - q for p, q in zip(p_values, q_values)]
        p_summary = _sample_summary(p_values)
        q_summary = _sample_summary(q_values)
        delta_summary = _sample_summary(deltas)
        delta_sd = float(delta_summary["sd"])
        mean_delta = float(delta_summary["mean"])
        if delta_sd == 0.0:
            paired_dz: float | None = 0.0 if mean_delta == 0.0 else None
            paired_dz_note = (
                "zero_delta_and_zero_sd"
                if mean_delta == 0.0
                else "undefined_because_delta_sd_is_zero"
            )
        else:
            paired_dz = mean_delta / delta_sd
            paired_dz_note = None
        wins = sum(p > q for p, q in zip(p_values, q_values))
        losses = sum(p < q for p, q in zip(p_values, q_values))
        ties = N_PAIRS - wins - losses
        classification = classify_frozen_candidates(
            mean_delta=mean_delta,
            ci95=delta_summary["ci95_t"],
            candidates_identical=candidates_identical,
        )
        analysis: MutableMapping[str, Any] = {
            "analysis_schema": "frozen-checkpoint-paired-analysis-v1",
            "status": "COMPLETE",
            "classification": classification,
            "classification_scope": "two_frozen_winner_configurations_only",
            "evidence_name": "prospective-after-freeze benchmark reveal",
            "project_level_test_was_untouched": False,
            "method_level_state_before": METHOD_LEVEL_STATE,
            "method_level_state_after": METHOD_LEVEL_STATE,
            "method_level_state_changed": False,
            "final_candidates_identical": candidates_identical,
            "P": p_summary,
            "Q": q_summary,
            "paired_delta": {
                **delta_summary,
                "median": statistics.median(deltas),
                "P_wins": wins,
                "Q_wins": losses,
                "ties": ties,
                "paired_dz": paired_dz,
                "paired_dz_note": paired_dz_note,
                "primary_interval": "paired_t_df29",
            },
            "paired_bootstrap": paired_bootstrap(deltas),
            "classification_thresholds": {
                "practical_margin": PRACTICAL_MARGIN,
                "P_advantage": "mean_delta >= 0.006 and paired_t_lower > 0",
                "P_noninferior": (
                    "neither advantage rule holds and paired_t_lower > -0.006"
                ),
                "Q_advantage": "mean_delta <= -0.006 and paired_t_upper < 0",
            },
        }
        analysis["analysis_sha256"] = hashlib.sha256(
            _canonical_json_bytes(analysis)
        ).hexdigest()
        return analysis
    except (AnalysisProtocolError, KeyError, TypeError, ValueError) as exc:
        failure = {
            "analysis_schema": "frozen-checkpoint-paired-analysis-v1",
            "status": "PROTOCOL_FAILURE",
            "classification": PROTOCOL_FAILURE,
            "classification_scope": "two_frozen_winner_configurations_only",
            "method_level_state_before": METHOD_LEVEL_STATE,
            "method_level_state_after": METHOD_LEVEL_STATE,
            "method_level_state_changed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure["analysis_sha256"] = hashlib.sha256(
            _canonical_json_bytes(failure)
        ).hexdigest()
        return failure


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = _canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--expected-ready-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reveal_worker = _load_reveal_worker_module()
    ready_bundle = reveal_worker.validate_ready_bundle(
        args.ready, expected_ready_root=args.expected_ready_root
    )
    expected_input = (ready_bundle.root_dir / reveal_worker.RESULT_RELATIVE).resolve()
    if args.input.resolve() != expected_input:
        raise SystemExit("analysis input must be the fixed one-time result path")
    with args.input.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    analysis = analyse_reveal_result(document, ready_bundle=ready_bundle)
    _atomic_json_write(args.output, analysis)
    return 0 if analysis["classification"] != PROTOCOL_FAILURE else 2


if __name__ == "__main__":
    raise SystemExit(main())
