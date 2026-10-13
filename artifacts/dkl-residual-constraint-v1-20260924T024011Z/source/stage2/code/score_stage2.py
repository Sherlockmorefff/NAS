#!/usr/bin/env python3
"""Independent, training-free scorer for the frozen Stage 2 prediction matrix.

Only this executable opens scoring labels. All prediction files and the seal are
verified before labels are read. It uses only Python's standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("FROZEN", "REG", "DKL", "RAW16")
SAMPLE_SEEDS = (20260922, 20260923)
MODEL_SEEDS = (0, 1, 2)
NS = (20, 50)
TIE_TOLERANCE_PP = 1e-9
LATENT_VARIANCE_TOLERANCE = 1e-8
A_ID_HASH = "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"
METRICS = (
    "regret5_pp", "top1_regret_pp", "mae_pp", "rmse_pp", "spearman",
    "coverage90_fraction", "coverage95_fraction", "interval90_width_pp",
    "interval95_width_pp", "nlpd_accuracy_units", "mean_latent_sd_pp",
    "mean_observed_sd_pp",
)
FORBIDDEN_PREDICTION_KEYS = {
    "labels", "query_labels", "query_se", "query_SE", "validation_mean",
    "validation_se", "validation_seeds", "true_y", "y_true", "target",
    "targets", "ground_truth", "query_y", "observed_y", "champion_id",
    "test_accuracy", "test_acc",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject_nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject_nonfinite)


def a_id_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Nonfinite or nonnumeric " + name)
    return float(value)


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def spread(values):
    values = [v for v in values if v is not None]
    return {
        "defined_count": len(values), "mean": mean(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def ranks(values):
    """Average ranks for exact ties, starting at 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    answer = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for i in order[start:stop]:
            answer[i] = rank
        start = stop
    return answer


def spearman(predicted, observed):
    if len(predicted) != len(observed) or len(predicted) < 2:
        return None, "fewer_than_two_query_candidates"
    if len(set(predicted)) < 2:
        return None, "constant_gp_prediction"
    if len(set(observed)) < 2:
        return None, "constant_query_validation_mean"
    x, y = ranks(predicted), ranks(observed)
    xc, yc = mean(x), mean(y)
    numerator = math.fsum((a - xc) * (b - yc) for a, b in zip(x, y))
    denominator = math.sqrt(math.fsum((a - xc) ** 2 for a in x) *
                            math.fsum((b - yc) ** 2 for b in y))
    return max(-1.0, min(1.0, numerator / denominator)), None


def exact_random5(observed, k=5):
    """Exact uniform-without-replacement order-statistic expectation."""
    y = sorted(finite(v, "random_reference_y") for v in observed)
    if len(y) < k or k < 1:
        raise ValueError("Random reference requires at least k query candidates")
    denominator = math.comb(len(y), k)
    expected_best = math.fsum(y[i - 1] * math.comb(i - 1, k - 1) / denominator
                              for i in range(k, len(y) + 1))
    regret = max(y) - expected_best
    # Only remove an impossible roundoff-sized negative expected regret.
    if regret < 0 and abs(regret) <= 1e-15:
        regret = 0.0
    return {
        "query_size": len(y), "k": k, "combination_count": denominator,
        "query_best_raw": max(y), "expected_best5_raw": expected_best,
        "expected_regret5_raw": regret, "expected_regret5_pp": 100.0 * regret,
        "definition": "uniform k candidates without replacement within this query fold",
    }


def observed_variance(latent_standardized, extra_variance_standardized, y_scale, query_se):
    latent = finite(latent_standardized, "latent_variance_standardized")
    extra = finite(extra_variance_standardized, "extra_noise_variance_standardized")
    scale = finite(y_scale, "y_scale")
    se = finite(query_se, "query_SE")
    if latent < -LATENT_VARIANCE_TOLERANCE:
        raise ValueError("Numerical failure: standardized latent variance < -1e-8")
    if extra < 1e-6 - 1e-14 or scale < 1e-8 or se < 0:
        raise ValueError("Invalid noise variance, y scale, or query SE")
    # Extra noise is already a variance. SE is already a mean standard error.
    # Retain the raw latent value here; clipping is only used for sqrt(latent).
    variance = scale * scale * (latent + extra) + se * se
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Observed variance must be finite and positive")
    return variance


def forbid_query_truth(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PREDICTION_KEYS:
                raise ValueError("Query truth field in sealed prediction: " + key)
            forbid_query_truth(child)
    elif isinstance(value, list):
        for child in value:
            forbid_query_truth(child)


def identity(unit):
    return (unit["fold"], unit["sample_seed"], unit["n"], unit["model_seed"], unit["method"])


def expected_matrix():
    return {(fold, sample_seed, n, model_seed, method)
            for fold in range(5) for sample_seed in SAMPLE_SEEDS for n in NS
            for method in METHODS
            for model_seed in ([None] if method == "RAW16" else MODEL_SEEDS)}


def validate_prediction(unit):
    forbid_query_truth(unit)
    required = {
        "unit_id", "context_id", "subset_id", "fold", "sample_seed", "n", "model_seed", "method",
        "candidate_ids", "architecture_fingerprints", "mean_standardized", "mean_raw",
        "latent_variance_standardized", "latent_variance_raw", "y_center", "y_scale",
        "extra_noise_variance_standardized", "endpoint_state_hash", "training_ids_sha256", "cost",
        "state_path", "state_file_sha256",
    }
    missing = required - set(unit)
    if missing:
        raise ValueError("Missing prediction fields: " + str(sorted(missing)))
    if identity(unit) not in expected_matrix():
        raise ValueError("Prediction identity outside fixed Stage 2 matrix")
    ids = unit["candidate_ids"]
    if len(ids) < 5 or len(set(ids)) != len(ids) or not all(isinstance(c, str) for c in ids):
        raise ValueError("Query candidate IDs must be unique strings, at least five")
    vector_fields = ("architecture_fingerprints", "mean_standardized", "mean_raw",
                     "latent_variance_standardized", "latent_variance_raw")
    if any(len(unit[field]) != len(ids) for field in vector_fields):
        raise ValueError("Prediction array length mismatch")
    center, scale = finite(unit["y_center"], "y_center"), finite(unit["y_scale"], "y_scale")
    extra = finite(unit["extra_noise_variance_standardized"], "extra_noise_variance_standardized")
    if scale < 1e-8 or extra < 1e-6 - 1e-14:
        raise ValueError("Invalid frozen scale or learned extra-noise bound")
    if not all(isinstance(a, str) and a for a in unit["architecture_fingerprints"]):
        raise ValueError("Missing query architecture identity")
    for field in ("endpoint_state_hash", "training_ids_sha256", "state_file_sha256"):
        digest = unit[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid SHA-256 field: " + field)
    for j in range(len(ids)):
        standardized = finite(unit["mean_standardized"][j], "mean_standardized")
        raw = finite(unit["mean_raw"][j], "mean_raw")
        latent = finite(unit["latent_variance_standardized"][j], "latent_variance_standardized")
        latent_raw = finite(unit["latent_variance_raw"][j], "latent_variance_raw")
        if latent < -LATENT_VARIANCE_TOLERANCE:
            raise ValueError("Numerical failure: standardized latent variance < -1e-8")
        if not math.isclose(raw, center + scale * standardized, abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError("Mean de-standardization mismatch")
        if not math.isclose(latent_raw, scale * scale * latent, abs_tol=1e-16, rel_tol=1e-10):
            raise ValueError("Latent variance de-standardization mismatch")
    return unit


def safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest file escapes artifact directory")
    return path


def load_sealed_predictions(root, manifest_path):
    """No label file is opened by this function."""
    manifest = read_json(manifest_path)
    if manifest.get("sealed") is not True:
        raise ValueError("Predictions must be uniformly sealed before scoring")
    if manifest.get("phase2_status") in ("INVALID", "BLOCKED"):
        raise ValueError("Cannot score an INVALID or BLOCKED experiment as a valid partial run")
    if manifest.get("phase2_fit_status") not in ("COMPLETE", "PARTIAL"):
        raise ValueError("Unknown fit status")
    if manifest.get("expected_endpoint_count") != 200:
        raise ValueError("Expected endpoint count must remain 200")
    for hash_key, path_key, default in (
        ("protocol_sha256", "protocol_path", "protocol.json"),
        ("code_manifest_sha256", "code_manifest_path", "code_manifest.json"),
    ):
        path = safe_child(root, manifest.get(path_key, default))
        if sha256(path) != manifest[hash_key]:
            raise ValueError("Frozen identity mismatch: " + path.name)
    units, seen, unit_ids, prediction_paths = [], set(), set(), set()
    fold_queries, fold_arches, subset_hashes, subset_ids = {}, {}, {}, {}
    for entry in manifest["files"]:
        path = safe_child(root, entry["path"])
        if path in prediction_paths or sha256(path) != entry["sha256"]:
            raise ValueError("Duplicate prediction path or prediction hash mismatch")
        prediction_paths.add(path)
        unit = validate_prediction(read_json(path))
        if entry.get("state_path") != unit["state_path"] or entry.get("state_sha256") != unit["state_file_sha256"]:
            raise ValueError("Manifest and prediction endpoint-state identity mismatch")
        state_path = safe_child(root, unit["state_path"])
        if sha256(state_path) != unit["state_file_sha256"]:
            raise ValueError("Sealed endpoint state file hash mismatch")
        key = identity(unit)
        if key in seen or unit["unit_id"] in unit_ids:
            raise ValueError("Duplicate endpoint or unit ID")
        if entry["unit_id"] != unit["unit_id"]:
            raise ValueError("Manifest unit ID mismatch")
        for field in ("fold", "sample_seed", "n", "model_seed", "method", "endpoint_state_hash"):
            if field in entry and entry[field] != unit[field]:
                raise ValueError("Manifest identity mismatch: " + field)
        seen.add(key)
        unit_ids.add(unit["unit_id"])
        fold = unit["fold"]
        for target, value in ((fold_queries, unit["candidate_ids"]),
                              (fold_arches, unit["architecture_fingerprints"])):
            if fold in target and target[fold] != value:
                raise ValueError("Query identity or order differs across matched units")
            target[fold] = value
        subset_key = (fold, unit["sample_seed"], unit["n"])
        for target, value in ((subset_hashes, unit["training_ids_sha256"]),
                              (subset_ids, unit["subset_id"])):
            if subset_key in target and target[subset_key] != value:
                raise ValueError("Training subset differs across methods or model seeds")
            target[subset_key] = value
        unit["prediction_file_sha256"] = entry["sha256"]
        units.append(unit)
    id_folds, architecture_folds = {}, {}
    for fold in fold_queries:
        for cid, architecture in zip(fold_queries[fold], fold_arches[fold]):
            if cid in id_folds and id_folds[cid] != fold:
                raise ValueError("Candidate appears in two query folds")
            if architecture in architecture_folds and architecture_folds[architecture] != fold:
                raise ValueError("Architecture overlaps query folds")
            id_folds[cid], architecture_folds[architecture] = fold, fold
    complete = seen == expected_matrix() and manifest["phase2_fit_status"] == "COMPLETE"
    if manifest["phase2_fit_status"] == "COMPLETE" and seen != expected_matrix():
        raise ValueError("COMPLETE seal does not contain the fixed 200-endpoint matrix")
    if complete and (len(id_folds) != 179 or len(architecture_folds) != 96):
        raise ValueError("Complete matrix does not cover 179 A candidates / 96 architectures")
    if "expected_unit_ids" in manifest and not unit_ids.issubset(set(manifest["expected_unit_ids"])):
        raise ValueError("Unexpected unit ID against frozen manifest")
    return manifest, units, complete, sorted(expected_matrix() - seen, key=str)


def checked_labels(path, required_ids):
    """Called only after ALL sealed prediction files have passed validation."""
    payload = read_json(path)
    if set(payload) - {"labels", "metadata"}:
        raise ValueError("Unexpected top-level fields in A-only scoring snapshot")
    labels = payload["labels"]
    if len(labels) != 179 or a_id_digest(labels) != A_ID_HASH:
        raise ValueError("Scoring snapshot must contain exactly the 179 pinned legal A IDs")
    if not required_ids.issubset(labels):
        raise ValueError("Scoring snapshot omits required query candidates")
    allowed = {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}
    for cid, row in labels.items():
        if set(row) - allowed:
            raise ValueError("Unauthorized scoring label fields for " + cid)
        y, se = finite(row["validation_mean"], "validation_mean"), finite(row["validation_se"], "validation_se")
        if y < 0 or y > 1 or se < 0:
            raise ValueError("Expected accuracy in [0,1] and nonnegative SE")
        if "validation_seeds" in row:
            seeds = [finite(v, "validation_seed") for v in row["validation_seeds"]]
            if len(seeds) != 3 or not math.isclose(y, mean(seeds), abs_tol=1e-12):
                raise ValueError("Three-seed validation mean mismatch")
            if not math.isclose(se, statistics.stdev(seeds) / math.sqrt(3), abs_tol=1e-12):
                raise ValueError("Validation SE must use ddof=1 divided by sqrt(3)")
        if "training_seeds" in row and row["training_seeds"] != [0, 1, 2]:
            raise ValueError("Unexpected GNN evaluation seed identities")
    return labels


def score_unit(unit, labels):
    ids, predicted = unit["candidate_ids"], unit["mean_raw"]
    observed = [labels[cid]["validation_mean"] for cid in ids]
    order = sorted(range(len(ids)), key=lambda i: (-predicted[i], ids[i]))
    chosen = order[:5]
    best_index = min(chosen, key=lambda i: (-observed[i], ids[i]))
    query_best_index = min(range(len(ids)), key=lambda i: (-observed[i], ids[i]))
    errors = [p - y for p, y in zip(predicted, observed)]
    correlation, reason = spearman(predicted, observed)
    latent = unit["latent_variance_standardized"]
    observed_vars = [observed_variance(v, unit["extra_noise_variance_standardized"],
                                       unit["y_scale"], labels[cid]["validation_se"])
                     for v, cid in zip(latent, ids)]
    standard_deviations = [math.sqrt(v) for v in observed_vars]
    result = {key: unit[key] for key in (
        "unit_id", "context_id", "subset_id", "fold", "sample_seed", "n", "model_seed", "method",
        "endpoint_state_hash", "training_ids_sha256",
    )}
    result.update({
        "query_size": len(ids), "selected_top5_ids": [ids[i] for i in chosen],
        "selected_best_id": ids[best_index], "selected_best_validation_mean_raw": observed[best_index],
        "selected_top1_id": ids[order[0]], "selected_top1_validation_mean_raw": observed[order[0]],
        "query_best_id": ids[query_best_index], "query_best_validation_mean_raw": observed[query_best_index],
        "regret5_raw": observed[query_best_index] - observed[best_index],
        "top1_regret_raw": observed[query_best_index] - observed[order[0]],
        "mae_raw": mean([abs(e) for e in errors]),
        "rmse_raw": math.sqrt(mean([e * e for e in errors])),
        "spearman": correlation, "spearman_undefined_reason": reason,
        "negative_latent_roundoff_count": sum(v < 0 for v in latent),
        "minimum_latent_variance_standardized_raw": min(latent),
        "mean_latent_variance_raw": mean(unit["latent_variance_raw"]),
        "mean_observed_variance_raw": mean(observed_vars),
        "mean_latent_sd_pp": 100 * unit["y_scale"] * mean([math.sqrt(max(v, 0.0)) for v in latent]),
        "mean_observed_sd_pp": 100 * mean(standard_deviations),
        "nlpd_accuracy_units": mean([0.5 * (math.log(2 * math.pi * v) + e * e / v)
                                      for e, v in zip(errors, observed_vars)]),
        "prediction_file_sha256": unit.get("prediction_file_sha256"),
    })
    for name in ("regret5", "top1_regret", "mae", "rmse"):
        result[name + "_pp"] = 100.0 * result[name + "_raw"]
    for level, z in ((90, 1.6448536269514722), (95, 1.959963984540054)):
        result[f"coverage{level}_fraction"] = mean([float(abs(e) <= z * sd)
                                                   for e, sd in zip(errors, standard_deviations)])
        result[f"interval{level}_width_pp"] = 100 * mean([2 * z * sd for sd in standard_deviations])
    return result


def classification(reg_improvements, dkl_regret, raw_regret, random_regret, complete):
    if not complete:
        return "不完整，暂不分类"
    if len(reg_improvements) != 5:
        raise ValueError("Classification requires all five matched folds")
    positive = sum(v > TIE_TOLERANCE_PP for v in reg_improvements)
    if (mean(reg_improvements) > TIE_TOLERANCE_PP and positive >= 4
            and raw_regret - dkl_regret > TIE_TOLERANCE_PP
            and random_regret - dkl_regret > TIE_TOLERANCE_PP):
        return "较一致的离线选优信号"
    elif positive > 0:
        return "局部或混合信号"
    else:
        return "未见相对继续回归的收益"


def summarize(rows, random_references, complete):
    """Model seeds -> sample seeds -> equally weighted outer folds."""
    sample_groups = defaultdict(list)
    for row in rows:
        sample_groups[(row["n"], row["method"], row["fold"], row["sample_seed"])].append(row)
    sample_rows = []
    for (n, method, fold, sample_seed), group in sorted(sample_groups.items()):
        row = {"n": n, "method": method, "fold": fold, "sample_seed": sample_seed,
               "endpoint_count": len(group), "expected_endpoint_count": 1 if method == "RAW16" else 3}
        for metric in METRICS:
            detail = spread([r[metric] for r in group])
            row[metric] = detail["mean"]
            row[metric + "_model_seed_sd"] = detail["sample_sd"]
            row[metric + "_defined_count"] = detail["defined_count"]
            row[metric + "_model_seed_min"] = detail["min"]
            row[metric + "_model_seed_max"] = detail["max"]
        sample_rows.append(row)
    fold_groups = defaultdict(list)
    for row in sample_rows:
        fold_groups[(row["n"], row["method"], row["fold"])].append(row)
    fold_rows = []
    for (n, method, fold), group in sorted(fold_groups.items()):
        row = {"n": n, "method": method, "fold": fold, "sample_seed_count": len(group),
               "endpoint_count": sum(r["endpoint_count"] for r in group)}
        for metric in METRICS:
            detail = spread([r[metric] for r in group])
            row[metric] = detail["mean"]
            row[metric + "_sample_seed_sd"] = detail["sample_sd"]
            row[metric + "_sample_seed_min"] = detail["min"]
            row[metric + "_sample_seed_max"] = detail["max"]
            row[metric + "_defined_count"] = detail["defined_count"]
        fold_rows.append(row)
    lookup = {identity(r): r for r in rows}
    paired_rows = []
    for dkl in (r for r in rows if r["method"] == "DKL"):
        for comparator in ("REG", "FROZEN", "RAW16"):
            key = (dkl["fold"], dkl["sample_seed"], dkl["n"],
                   None if comparator == "RAW16" else dkl["model_seed"], comparator)
            reference = lookup.get(key)
            if reference is None:
                continue
            row = {"n": dkl["n"], "fold": dkl["fold"], "sample_seed": dkl["sample_seed"],
                   "model_seed": dkl["model_seed"], "comparator": comparator,
                   "dkl_unit_id": dkl["unit_id"], "reference_unit_id": reference["unit_id"],
                   "raw_reference_reused_for_pairing_only": comparator == "RAW16"}
            for metric in METRICS:
                # All differences are comparator - DKL; only loss-like metrics
                # should be interpreted as positive-implies-better.
                a, b = reference[metric], dkl[metric]
                row[metric + "_comparator_minus_DKL"] = a - b if a is not None and b is not None else None
            row["improvement_REG_direction"] = "comparator regret5 minus DKL regret5; positive favors DKL"
            paired_rows.append(row)
    pair_groups = defaultdict(list)
    for row in paired_rows:
        pair_groups[(row["n"], row["comparator"], row["fold"], row["sample_seed"])].append(row)
    pair_sample_means = []
    for (n, comparator, fold, sample_seed), group in sorted(pair_groups.items()):
        row = {"n": n, "comparator": comparator, "fold": fold, "sample_seed": sample_seed,
               "matched_model_seed_count": len(group)}
        for metric in METRICS:
            field = metric + "_comparator_minus_DKL"
            row[field] = mean([r[field] for r in group])
        pair_sample_means.append(row)
    pair_fold_groups = defaultdict(list)
    for row in pair_sample_means:
        pair_fold_groups[(row["n"], row["comparator"], row["fold"])].append(row)
    fold_pairs = []
    for (n, comparator, fold), group in sorted(pair_fold_groups.items()):
        row = {"n": n, "comparator": comparator, "fold": fold,
               "matched_sample_seed_count": len(group),
               "matched_model_seed_pairs": sum(r["matched_model_seed_count"] for r in group)}
        for metric in METRICS:
            field = metric + "_comparator_minus_DKL"
            row[field] = mean([r[field] for r in group])
        row["regret5_improvement_pp"] = row["regret5_pp_comparator_minus_DKL"]
        value = row["regret5_improvement_pp"]
        row["direction"] = "improved" if value > TIE_TOLERANCE_PP else "worse" if value < -TIE_TOLERANCE_PP else "tie"
        fold_pairs.append(row)
    result = {
        "phase2_status": "COMPLETE_VALID" if complete else "PARTIAL",
        "next_stage_authorized": False,
        "endpoint_count": len(rows), "expected_endpoint_count": 200,
        "descriptive_only": True, "no_independent_confirmation_or_significance_claim": True,
        "aggregation": "mean model seeds within (fold,sample_seed,n,method), then mean sample seeds within fold, then five folds equally; RAW16 once per subset",
        "partial_aggregation": "available matched data only; incomplete results are not assigned full-protocol classifications",
        "tie_tolerance_pp": TIE_TOLERANCE_PP,
        "metric_units": {"suffix_pp": "percentage points (100 x raw accuracy difference)",
                         "coverage": "fraction", "nlpd": "Gaussian density in raw accuracy units",
                         "spearman": "dimensionless; undefined constant vectors are null, not zero"},
        "per_n": {},
    }
    for n in NS:
        methods = {}
        for method in METHODS:
            method_folds = [r for r in fold_rows if r["n"] == n and r["method"] == method]
            methods[method] = {
                "endpoint_count": sum(r["n"] == n and r["method"] == method for r in rows),
                "fold_count": len(method_folds),
                "metrics": {m: spread([r[m] for r in method_folds]) for m in METRICS},
            }
        paired = {}
        for comparator in ("REG", "FROZEN", "RAW16"):
            pairs = [r for r in fold_pairs if r["n"] == n and r["comparator"] == comparator]
            values = [r["regret5_improvement_pp"] for r in pairs]
            paired[comparator] = {**spread(values), "improved_folds": sum(v > TIE_TOLERANCE_PP for v in values),
                                  "tied_folds": sum(abs(v) <= TIE_TOLERANCE_PP for v in values),
                                  "worse_folds": sum(v < -TIE_TOLERANCE_PP for v in values),
                                  "fold_values_pp": {str(r["fold"]): r["regret5_improvement_pp"] for r in pairs}}
        dkl_fold_ids = {r["fold"] for r in fold_rows if r["n"] == n and r["method"] == "DKL"}
        # For PARTIAL matrices, match the random reference to the available DKL
        # folds before displaying a DKL-versus-random difference.
        matched_random = [r for r in random_references if r["fold"] in dkl_fold_ids]
        random_value = mean([r["expected_regret5_pp"] for r in matched_random])
        reg_values = list(paired["REG"]["fold_values_pp"].values())
        dkl_value = methods["DKL"]["metrics"]["regret5_pp"]["mean"]
        raw_value = methods["RAW16"]["metrics"]["regret5_pp"]["mean"]
        signal = classification(reg_values, dkl_value, raw_value, random_value, complete)
        result["scientific_signal_n" + str(n)] = signal
        result["per_n"][str(n)] = {
            "scientific_signal": signal, "methods": methods,
            "paired_regret5_improvement_pp": paired,
            "random5_expected_regret5_pp": random_value,
            "random5_minus_DKL_regret5_pp": random_value - dkl_value if dkl_value is not None and random_value is not None else None,
            "random_reference_fold_count": len(matched_random),
        }
    return result, sample_rows, fold_rows, paired_rows, fold_pairs


def cost_outputs(units):
    rows = []
    expected = ("common_training_seconds", "branch_training_seconds", "prediction_seconds", "standalone_fit_predict_seconds")
    for unit in units:
        cost = unit["cost"]
        if any(k not in cost for k in expected):
            raise ValueError("Missing method independent-use cost fields")
        values = {k: finite(cost[k], k) for k in expected}
        if any(v < 0 for v in values.values()):
            raise ValueError("Negative resource cost")
        if not math.isclose(values["standalone_fit_predict_seconds"], sum(values[k] for k in expected[:3]), abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("Standalone method cost must include common training, own branch, and prediction")
        rows.append({**{k: unit[k] for k in ("unit_id", "context_id", "fold", "sample_seed", "n", "model_seed", "method")}, **values})
    common_by_context = {}
    for row in rows:
        if row["method"] != "RAW16":
            key = (row["fold"], row["sample_seed"], row["n"], row["model_seed"])
            if key in common_by_context and not math.isclose(common_by_context[key], row["common_training_seconds"], abs_tol=1e-9):
                raise ValueError("Shared common-state cost differs across mapped branches")
            common_by_context[key] = row["common_training_seconds"]
    completed_cost = (sum(common_by_context.values()) + sum(r["branch_training_seconds"] + r["prediction_seconds"] for r in rows)
                      + sum(r["common_training_seconds"] for r in rows if r["method"] == "RAW16"))
    summary = {
        "units": "seconds", "completed_endpoint_fit_predict_seconds_with_shared_common_cost_once": completed_cost,
        "resource_ledger_remains_authoritative_for_incomplete_blocks_validation_and_reruns": True,
        "per_n_method_standalone_cost": {
            str(n): {m: spread([r["standalone_fit_predict_seconds"] for r in rows if r["n"] == n and r["method"] == m])
                     for m in METHODS} for n in NS},
    }
    return rows, summary


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run(artifact_dir, manifest_path=None, labels_path=None, output_dir=None):
    root = Path(artifact_dir).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "predictions_manifest.json"
    labels_path = Path(labels_path) if labels_path else root / "scoring" / "labels_A.json"
    output = Path(output_dir) if output_dir else root / "scoring"
    output.mkdir(parents=True, exist_ok=True)
    manifest, units, complete, missing = load_sealed_predictions(root, manifest_path)
    # This line is the only scoring-label release point and follows the full seal check.
    labels = checked_labels(labels_path, {c for unit in units for c in unit["candidate_ids"]})
    rows = [score_unit(unit, labels) for unit in units]
    fold_queries = {unit["fold"]: unit["candidate_ids"] for unit in units}
    random_refs = [{"fold": fold, **exact_random5([labels[c]["validation_mean"] for c in ids])}
                   for fold, ids in sorted(fold_queries.items())]
    summary, sample_rows, fold_rows, paired_rows, fold_pairs = summarize(rows, random_refs, complete)
    cost_rows, cost_summary = cost_outputs(units)
    summary["missing_endpoint_identities"] = missing
    summary["negative_latent_roundoff_count"] = sum(row["negative_latent_roundoff_count"] for row in rows)
    summary["score_provenance"] = {
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_manifest_sha256": sha256(manifest_path), "labels_A_sha256": sha256(labels_path),
        "scorer_sha256": sha256(__file__), "sealed_at_utc": manifest.get("sealed_at_utc"),
        "all_prediction_hashes_verified_before_label_read": True,
        "query_SE_used_only_in_postseal_calibration": True,
        "no_training_imported_or_triggered": True,
    }
    for filename, content in (("unit_metrics.csv", rows), ("sample_seed_metrics.csv", sample_rows),
                              ("fold_metrics.csv", fold_rows), ("paired_unit_metrics.csv", paired_rows),
                              ("fold_pairs.csv", fold_pairs), ("cost_table.csv", cost_rows)):
        write_csv(output / filename, content)
    calibration_fields = ("unit_id", "fold", "sample_seed", "n", "model_seed", "method", "query_size",
                          "coverage90_fraction", "coverage95_fraction", "interval90_width_pp", "interval95_width_pp",
                          "nlpd_accuracy_units", "mean_latent_variance_raw", "mean_observed_variance_raw",
                          "mean_latent_sd_pp", "mean_observed_sd_pp", "negative_latent_roundoff_count")
    write_csv(output / "calibration.csv", [{k: row[k] for k in calibration_fields} for row in rows])
    write_json(output / "summary.json", summary)
    write_json(output / "random_reference.json", {"references_by_fold": random_refs,
               "equal_fold_mean_expected_regret5_pp": mean([r["expected_regret5_pp"] for r in random_refs]),
               "not_independent_repetitions_across_training_subsets_or_model_seeds": True})
    write_json(output / "cost_summary.json", cost_summary)
    write_json(output / "score_provenance.json", summary["score_provenance"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--labels")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run(args.artifact_dir, args.manifest, args.labels, args.output_dir)
    print(json.dumps({k: summary[k] for k in ("phase2_status", "endpoint_count", "scientific_signal_n20",
                                            "scientific_signal_n50", "next_stage_authorized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
