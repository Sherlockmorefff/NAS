"""Independent, frozen-prediction scoring; never imported by the GP fitter."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist, mean, median

METHODS = ("L-A", "L-I", "W-A", "W-I", "RAW16")
DRAWS = (20260922, 20260923)
Z90 = NormalDist().inv_cdf(0.95)
IDENTITY_KEYS = ("setting_id", "method", "n", "fold", "draw")
METRIC_KEYS = (
    "mae_pp", "nlpd", "coverage90", "width90_pp", "is90_pp", "spearman",
    "q_top10_bias_pp", "predicted_top5_bias_pp", "regret5_pp",
    "latent_coverage90", "latent_width90_pp", "latent_is90_pp",
    "retrospective_nlpd", "retrospective_coverage90", "retrospective_width90_pp",
    "retrospective_is90_pp",
)


def _finite(value, field):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field}")
    return value


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = mean(rx), mean(ry)
    covariance = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    variance = sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    return covariance / math.sqrt(variance) if variance > 0 else None


def _interval_metrics(y, mu, variance, *, density):
    widths, scores, covered, nlpds = [], [], [], []
    for observed, prediction, var in zip(y, mu, variance):
        if not math.isfinite(var) or var < 0 or (density and var == 0):
            raise ValueError("Invalid predictive variance; no clipping is permitted")
        half_width = Z90 * math.sqrt(var)
        lower, upper = prediction - half_width, prediction + half_width
        width = upper - lower
        interval_score = width + 20.0 * max(lower - observed, 0.0) + 20.0 * max(observed - upper, 0.0)
        widths.append(width * 100.0)
        scores.append(interval_score * 100.0)
        covered.append(float(lower <= observed <= upper))
        if density:
            nlpds.append(0.5 * (math.log(2 * math.pi) + math.log(var) + (observed - prediction) ** 2 / var))
    result = {"coverage90": mean(covered), "width90_pp": mean(widths), "is90_pp": mean(scores)}
    if density:
        result["nlpd"] = mean(nlpds)
    return result


def score_prediction(unit, labels):
    """Score one immutable prediction. Labels are A-only {id: {mean,se,seeds}}.

    NLPD is a density on raw accuracy (0--1), hence may legitimately be negative.
    Only retrospective metrics use Q SE. Optimizer-start selection is upstream.
    """
    ids = list(unit["ids"])
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(cid, str) for cid in ids):
        raise ValueError("Prediction IDs must be distinct nonempty string IDs")
    mu = [_finite(v, "mean_raw") for v in unit["mean_raw"]]
    latent_std = [_finite(v, "latent_variance_standardized") for v in unit["latent_variance_standardized"]]
    if len(mu) != len(ids) or len(latent_std) != len(ids):
        raise ValueError("Prediction arrays do not match query IDs")
    scale = _finite(unit["y_scale"], "y_scale")
    extra = _finite(unit["extra_noise_variance_standardized"], "extra_noise_variance_standardized")
    median_known = _finite(unit["median_known_var_standardized"], "median_known_var_standardized")
    if scale <= 0 or extra < 0 or median_known < 0 or any(v < -1e-8 for v in latent_std):
        raise ValueError("Invalid frozen scale or variance")
    y = [_finite(labels[cid]["mean"], "A mean") for cid in ids]
    se = [_finite(labels[cid]["se"], "A SE") for cid in ids]
    if any(v < 0 for v in se) or any(v < 0 or v > 1 for v in y):
        raise ValueError("Labels must use raw accuracy 0--1 and nonnegative SE")
    variance_scale = scale * scale
    latent = [v * variance_scale for v in latent_std]
    prospective = [(v + extra + median_known) * variance_scale for v in latent_std]
    retrospective = [(v + extra) * variance_scale + s * s for v, s in zip(latent_std, se)]
    result = {key: unit[key] for key in IDENTITY_KEYS}
    result["model_hash"] = unit["model_hash"]
    result["query_count"] = len(ids)
    result["mae_pp"] = mean(abs(a - b) for a, b in zip(y, mu)) * 100.0
    result.update(_interval_metrics(y, mu, prospective, density=True))
    if any(v < 0 for v in latent):
        result.update(latent_coverage90=None, latent_width90_pp=None, latent_is90_pp=None,
                      latent_interval_status="MISSING_NEGATIVE_VARIANCE_ROUNDOFF_NO_CLIPPING",
                      latent_negative_variance_count=sum(v < 0 for v in latent))
    else:
        result.update({"latent_" + k: v for k, v in _interval_metrics(y, mu, latent, density=False).items()})
        result.update(latent_interval_status="COMPLETE", latent_negative_variance_count=0)
    result.update({"retrospective_" + k: v for k, v in _interval_metrics(y, mu, retrospective, density=True).items()})
    result["spearman"] = _spearman(y, mu)
    true_top = sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(0.1 * len(ids))]
    predicted_top = sorted(range(len(ids)), key=lambda i: (-mu[i], ids[i]))[:5]
    result["q_top10_bias_pp"] = mean(mu[i] - y[i] for i in true_top) * 100.0
    result["predicted_top5_bias_pp"] = mean(mu[i] - y[i] for i in predicted_top) * 100.0
    result["regret5_pp"] = (max(y) - max(y[i] for i in predicted_top)) * 100.0
    if any(v is not None and not math.isfinite(v) for k, v in result.items() if k in METRIC_KEYS):
        raise ValueError("Nonfinite score")
    return result


def _metric_mean(rows, key):
    values = [row[key] for row in rows]
    # Undefined correlations remain explicitly missing; do not omit them silently.
    return None if any(value is None for value in values) else mean(values)


def aggregate_metrics(contexts):
    """Average draws within a fold, then five folds equally; no missing-cell means."""
    indexed = {}
    for row in contexts:
        key = (row["method"], int(row["n"]), int(row["fold"]), int(row["draw"]))
        if key in indexed:
            raise ValueError(f"Duplicate scored context {key}")
        indexed[key] = row
    folds, summary, missing = [], [], []
    for method in METHODS:
        for n in (20, 50):
            method_folds = []
            for fold in range(5):
                required = [(method, n, fold, draw) for draw in DRAWS]
                absent = [key for key in required if key not in indexed]
                if absent:
                    missing.extend([dict(zip(("method", "n", "fold", "draw"), key)) for key in absent])
                    continue
                selected = [indexed[key] for key in required]
                row = {"method": method, "n": n, "fold": fold, "draw_count": 2}
                row.update({key: _metric_mean(selected, key) for key in METRIC_KEYS})
                folds.append(row)
                method_folds.append(row)
            if len(method_folds) == 5:
                row = {"method": method, "n": n, "fold_count": 5, "context_count": 10}
                row.update({key: _metric_mean(method_folds, key) for key in METRIC_KEYS})
                summary.append(row)
    return folds, summary, missing


def engineering_gate(summary, folds, *, complete, failures=None):
    """Apply only frozen engineering thresholds; never claim statistical significance."""
    summary_index = {(row["method"], row["n"]): row for row in summary}
    fold_index = {(row["method"], row["n"], row["fold"]): row for row in folds}
    comparisons = []
    pairs = [("L-I", "L-A"), ("W-A", "L-A"), ("W-I", "W-A")]
    pairs += [(method, "RAW16") for method in METHODS if method != "RAW16"]
    for target, baseline in pairs:
        for n in (20, 50):
            target_row, base_row = summary_index.get((target, n)), summary_index.get((baseline, n))
            if target_row is None or base_row is None:
                comparisons.append({"target": target, "baseline": baseline, "n": n, "status": "INCOMPLETE"})
                continue
            comparisons.append({"target": target, "baseline": baseline, "n": n, "status": "COMPLETE",
                                "mae_improvement_pp": base_row["mae_pp"] - target_row["mae_pp"],
                                "nlpd_change_nat": target_row["nlpd"] - base_row["nlpd"],
                                "is90_change_pp": target_row["is90_pp"] - base_row["is90_pp"],
                                "coverage90_change": target_row["coverage90"] - base_row["coverage90"],
                                "width90_change_pp": target_row["width90_pp"] - base_row["width90_pp"]})
    candidates = []
    for method in METHODS[:-1]:
        required = [(method, 20), (method, 50), ("RAW16", 20), ("RAW16", 50)]
        if any(key not in summary_index for key in required):
            candidates.append({"method": method, "passed": False, "status": "INCOMPLETE"})
            continue
        m20, m50 = summary_index[(method, 20)], summary_index[(method, 50)]
        b20, b50 = summary_index[("RAW16", 20)], summary_index[("RAW16", 50)]
        fold_deltas = [fold_index[("RAW16", 50, f)]["mae_pp"] - fold_index[(method, 50, f)]["mae_pp"] for f in range(5)]
        checks = {
            "all_100_settings_complete": bool(complete),
            "n50_mae_improvement_at_least_0_3_pp": b50["mae_pp"] - m50["mae_pp"] >= 0.3,
            "n50_at_least_4_positive_folds": sum(delta > 0 for delta in fold_deltas) >= 4,
            "n50_nlpd_deterioration_at_most_0_1_nat": m50["nlpd"] - b50["nlpd"] <= 0.1,
            "n50_is90_deterioration_at_most_5_percent": m50["is90_pp"] <= b50["is90_pp"] * 1.05,
            "n20_mae_deterioration_at_most_0_3_pp": m20["mae_pp"] - b20["mae_pp"] <= 0.3,
        }
        candidates.append({"method": method, "passed": all(checks.values()), "status": "COMPLETE" if complete else "INCOMPLETE",
                           "checks": checks, "n50_mae_improvement_pp": b50["mae_pp"] - m50["mae_pp"],
                           "n50_positive_fold_count": sum(delta > 0 for delta in fold_deltas), "n50_fold_improvements_pp": fold_deltas})
    passing = [row["method"] for row in candidates if row["passed"]]
    selected = None
    tie_steps = []
    if passing:
        best_mae = min(summary_index[(method, 50)]["mae_pp"] for method in passing)
        tied = [method for method in passing if summary_index[(method, 50)]["mae_pp"] - best_mae < 0.05]
        tie_steps.append({"stage": "mae_within_0_05_pp_of_minimum", "methods": tied})
        best_nlpd = min(summary_index[(method, 50)]["nlpd"] for method in tied)
        tied = [method for method in tied if summary_index[(method, 50)]["nlpd"] - best_nlpd < 0.01]
        tie_steps.append({"stage": "nlpd_within_0_01_nat_of_minimum", "methods": tied})
        selected = sorted(tied, key=lambda method: (not method.endswith("-A"), not method.startswith("L-"), method))[0]
    return {"status": "INCOMPLETE" if not complete else ("AWAITING_SECOND_ROUND_CONFIRMATION" if selected else "NO_STABLE_GAIN"),
            "all_settings_complete": bool(complete), "selected_method": selected, "passing_methods": passing,
            "candidates": candidates, "comparisons": comparisons, "tie_steps": tie_steps,
            "failures": failures or [], "inference": "Engineering thresholds on exploratory A; no B/test validation or significance claim."}


def score_frozen(run_dir, repo, allow_partial=True):
    """Check sealed fits before A-label access. Partial runs never pass the gate."""
    from .io import read_json, write_json, sha256, digest, write_csv, now
    from .data import read_A_labels

    root, repo = Path(run_dir).resolve(), Path(repo).resolve()

    def contained(base, relative):
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Artifact path escapes its root")
        return path

    def verify_identity(value, name):
        expected = digest({k: v for k, v in value.items() if k != "identity"})
        if value.get("identity") != expected:
            raise ValueError(f"{name} identity mismatch")

    freeze = read_json(root / "frozen_manifest.json")
    seal = read_json(root / "prediction_seal.json")
    status = read_json(root / "status.json")
    verify_identity(freeze, "frozen_manifest")
    verify_identity(seal, "prediction_seal")
    if digest(freeze["code_files"]) != freeze["code_hash"] or seal["code_hash"] != freeze["code_hash"]:
        raise ValueError("Frozen code identity mismatch")
    if sha256(root / "protocol.json") != freeze["protocol_sha256"] or seal["protocol_sha256"] != freeze["protocol_sha256"]:
        raise ValueError("Frozen protocol identity mismatch")
    immutable = {}
    for relative, expected in freeze["code_files"].items():
        path = contained(repo, relative)
        if sha256(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
        immutable[str(path)] = expected
    for relative, expected in freeze["input_files"].items():
        path = contained(root, relative)
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
        immutable[str(path)] = expected
    for path in (root / "protocol.json", root / "frozen_manifest.json", root / "prediction_seal.json"):
        immutable[str(path)] = sha256(path)
    if "settings.json" not in freeze["input_files"]:
        raise ValueError("Missing frozen scientific worklist")
    expected_settings = {row["setting_id"]: row for row in read_json(root / "settings.json")}
    verified, failures = [], []
    for setting_id, hashes in seal["settings"].items():
        entry = status["settings"].get(setting_id)
        if entry is None:
            raise ValueError(f"Sealed setting absent from worklist: {setting_id}")
        try:
            directory = contained(root / "settings", setting_id)
            required = {"prediction_sha256": "prediction.json", "model_sha256": "model.pt"}
            for start in range(3):
                required[f"start{start}_json_sha256"] = f"start{start}.json"
                required[f"start{start}_model_sha256"] = f"start{start}.pt"
                required[f"start{start}_log_sha256"] = f"start{start}.jsonl"
            for hash_key, filename in required.items():
                path = directory / filename
                actual = sha256(path)
                if hashes.get(hash_key) != actual:
                    raise ValueError(f"Frozen {filename} mismatch")
                immutable[str(path)] = actual
            unit = read_json(directory / "prediction.json")
            setting = expected_settings[setting_id]
            training_name = f"training/{setting['subset_id']}.json"
            query_name = f"queries/fold{setting['fold']}.json"
            if training_name not in freeze["input_files"] or query_name not in freeze["input_files"]:
                raise ValueError("Missing frozen T/Q bundles")
            training, query = read_json(root / training_name), read_json(root / query_name)
            expected_identity = digest({"frozen_identity": freeze["identity"], "setting": setting,
                                        "T_ids": training["ids"], "Q_ids": query["ids"],
                                        "scaler": training["scaler"], "distance_scales": unit["distance_scales"]})
            if unit["identity"] != expected_identity:
                raise ValueError("Prediction identity cannot be reconstructed from frozen T/Q/scalers")
            if unit["ids"] != query["ids"] or unit["T_ids"] != training["ids"] or set(unit["ids"]) & set(unit["T_ids"]):
                raise ValueError("Prediction T/Q identities mismatch or overlap")
            if unit["y_scale"] != training["scaler"]["y_scale"] or unit["y_center"] != training["scaler"]["y_mean"]:
                raise ValueError("Prediction target scaler differs from frozen T scaler")
            if unit["median_known_var_standardized"] != median(training["known_variance_standardized"]):
                raise ValueError("Prospective variance is not frozen median T known variance")
            if unit["setting_id"] != setting_id or unit["identity"] != entry["identity"]:
                raise ValueError("Prediction setting identity mismatch")
            if any(unit[key] != entry[key] for key in ("method", "n", "fold", "draw")):
                raise ValueError("Prediction context differs from frozen worklist")
            if unit["model_hash"] != hashes["model_sha256"]:
                raise ValueError("Prediction is associated with a different model")
            starts = [read_json(directory / f"start{start}.json") for start in range(3)]
            if any(result["identity"] != entry["identity"] or result["start_id"] != start for start, result in enumerate(starts)):
                raise ValueError("Optimizer-start identity mismatch")
            valid = [result for result in starts if result.get("success") is True and math.isfinite(float(result["final_nll"]))]
            if not valid:
                raise ValueError("No finite optimizer start")
            expected_start = min(valid, key=lambda result: (result["final_nll"], result["start_id"]))["start_id"]
            if unit["selected_start"] != expected_start:
                raise ValueError("Selected start does not minimize final training NLL")
            if hashes["model_sha256"] != hashes[f"start{expected_start}_model_sha256"]:
                raise ValueError("Chosen model differs from training-NLL-selected checkpoint")
            success = len(valid) == 3 and entry["state"] not in ("FAILED", "BLOCKED")
            verified.append((setting_id, unit, success))
            if not success:
                failures.append({"setting_id": setting_id, "reason": "Not all three starts succeeded or setting previously failed/blocked"})
        except (KeyError, ValueError, TypeError, OSError) as exc:
            entry.update(state="FAILED", phase="SCORING_FAILED", scorer_error=f"{type(exc).__name__}: {exc}")
            failures.append({"setting_id": setting_id, "reason": entry["scorer_error"]})

    # The sole label read occurs after all available fit states were sealed and checked.
    labels = read_A_labels(repo) if verified else {}
    contexts, prediction_rows = [], []
    for setting_id, unit, success in verified:
        entry = status["settings"][setting_id]
        try:
            metrics = score_prediction(unit, labels)
            metrics.update(complete_setting=success, excluded_from_aggregate=not success,
                           prediction_hash=seal["settings"][setting_id]["prediction_sha256"])
            write_json(root / "settings" / setting_id / "metrics.json", metrics)
            contexts.append(metrics)
            if success:
                entry.update(state="COMPLETE", phase="SCORED", scored_at=now())
            for index, cid in enumerate(unit["ids"]):
                variance_scale = unit["y_scale"] ** 2
                latent_std = unit["latent_variance_standardized"][index]
                prediction_rows.append({**{key: unit[key] for key in IDENTITY_KEYS},
                    "candidate_id": cid, "model_hash": unit["model_hash"], "mean_raw": unit["mean_raw"][index],
                    "y_raw": labels[cid]["mean"], "retrospective_se_raw": labels[cid]["se"],
                    "latent_variance_raw": latent_std * variance_scale,
                    "prospective_variance_raw": (latent_std + unit["extra_noise_variance_standardized"] + unit["median_known_var_standardized"]) * variance_scale,
                    "retrospective_variance_raw": (latent_std + unit["extra_noise_variance_standardized"]) * variance_scale + labels[cid]["se"] ** 2,
                    "complete_setting": success})
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            entry.update(state="FAILED", phase="SCORING_FAILED", scorer_error=f"{type(exc).__name__}: {exc}")
            failures.append({"setting_id": setting_id, "reason": entry["scorer_error"]})

    for path, expected in immutable.items():
        if sha256(path) != expected:
            raise ValueError(f"Scorer changed frozen artifact: {path}")
    complete_contexts = [row for row in contexts if row["complete_setting"]]
    folds, summary, missing = aggregate_metrics(complete_contexts)
    expected_cells = {(m, n, f, d) for m in METHODS for n in (20, 50) for f in range(5) for d in DRAWS}
    actual_cells = {(e["method"], e["n"], e["fold"], e["draw"]) for e in status["settings"].values()}
    complete = (len(status["settings"]) == 100 and actual_cells == expected_cells and len(complete_contexts) == 100
                and all(e["state"] == "COMPLETE" for e in status["settings"].values()))
    for setting_id, entry in status["settings"].items():
        if entry["state"] in ("FAILED", "BLOCKED") and not any(r["setting_id"] == setting_id for r in failures):
            failures.append({"setting_id": setting_id, "reason": entry.get("error", entry["state"])})
    gate = engineering_gate(summary, folds, complete=complete, failures=failures)
    gate.update(expected_settings=100, scored_settings=len(contexts), complete_settings=len(complete_contexts),
                missing_contexts=missing, label_scope="A-only", scorer_verified_frozen_state_unchanged=True)
    write_csv(root / "metrics_by_context.csv", contexts, list(IDENTITY_KEYS) + ["model_hash", "query_count"] + list(METRIC_KEYS)
              + ["complete_setting", "excluded_from_aggregate", "prediction_hash", "latent_interval_status", "latent_negative_variance_count"])
    write_csv(root / "metrics_by_fold.csv", folds, ["method", "n", "fold", "draw_count"] + list(METRIC_KEYS))
    write_csv(root / "metrics_summary.csv", summary, ["method", "n", "fold_count", "context_count"] + list(METRIC_KEYS))
    write_json(root / "metrics_summary.json", summary)
    write_csv(root / "predictions.csv", prediction_rows)
    write_json(root / "engineering_gate.json", gate)
    write_json(root / "paired_comparisons.json", gate["comparisons"])
    write_csv(root / "paired_comparisons.csv", gate["comparisons"])
    status["updated_at"] = now()
    status["counts"] = {state: sum(e["state"] == state for e in status["settings"].values())
                        for state in ("PENDING", "RUNNING", "COMPLETE", "FAILED", "BLOCKED")}
    write_json(root / "status.json", status)
    write_json(root / "scorer_integrity.json", {"checked_at": now(), "prediction_seal_identity": seal["identity"],
               "immutable_files_checked": len(immutable), "frozen_states_unchanged": True,
               "Q_labels_used_by": "independent_scorer_only", "B_label_files_opened": 0, "node_test_files_opened": 0})
    if not allow_partial and not complete:
        raise RuntimeError("Scoring is incomplete; individual artifacts and blocking reasons were saved")
    return gate
