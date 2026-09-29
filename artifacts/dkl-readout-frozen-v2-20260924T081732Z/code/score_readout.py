#!/usr/bin/env python3
"""Post-seal, legal-A-only readout scoring. Standard library; no model imports.

All point losses are raw accuracy converted to percentage points. NLPD uses
natural logs in raw accuracy units; coverage is a fraction. HEAD is deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import score_stage2 as base

NEW_METHODS = ("HEAD", "GP_CONST", "GP_TREND", "GP_SEMI")
HISTORICAL_METHODS = ("RAW16", "R_FROZEN", "R_REG", "R_DKL_C")
METHODS = NEW_METHODS + HISTORICAL_METHODS
TOL_PP = 1e-9
TOL_RAW = TOL_PP / 100
Z90 = 1.6448536269514722
IDENTITY = ("unit_id", "context_id", "subset_id", "fold", "sample_seed", "n", "model_seed", "method")
METRICS = (
    "regret5_pp", "gain5_pp", "headroom_pp", "simple_regret_pp", "break_best_rate",
    "has_headroom_fraction", "top1_regret_pp", "champion_top5_hit_fraction",
    "champion_prediction_rank", "mae_pp", "rmse_pp", "spearman", "bias_pp",
    "nlpd_accuracy_units", "coverage90_fraction", "interval90_width_pp", "IS90_pp",
    "champion_mae_pp", "champion_rmse_pp", "champion_bias_pp", "champion_coverage90_fraction",
    "top20_mae_pp", "top20_rmse_pp", "top20_bias_pp", "top20_coverage90_fraction",
    "improver_mean_prediction_rank", "improver_best_prediction_rank",
    "random_regret5_pp", "random_gain5_pp", "random_simple_regret_pp", "random_break_best_probability",
)
PAIR_METRICS = ("regret5_improvement_pp", "gain5_improvement_pp", "simple_regret_improvement_pp",
                "nlpd_increase_nats", "IS90_increase_pp", "coverage90_change_fraction",
                "champion_coverage90_change_fraction")
FLAGS = ("PRIMARY_SELECTION_SIGNAL", "HEAD_SIGNAL_RETAINED", "ADDITIONAL_OVER_HEAD_SIGNAL",
         "INCUMBENT_GAIN_SIGNAL", "PROBABILITY_NO_LARGE_DEGRADATION", "OPTIMIZATION_READY",
         "READOUT_CANDIDATE_FOR_NEXT_PROTOCOL_REVIEW")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checked_file(root, relative, sha):
    path = base.safe_child(root, relative)
    require(base.sha256(path) == sha, "Frozen SHA256 mismatch: " + str(relative))
    return path


def identity(row):
    return tuple(row[k] for k in ("n", "fold", "sample_seed", "model_seed", "method"))


def expected(methods):
    return {(n, f, s, m, method) for n in base.NS for f in range(5) for s in base.SAMPLE_SEEDS
            for method in methods for m in ([None] if method == "RAW16" else base.MODEL_SEEDS)}


def validate_prediction(p, historical=False):
    base.forbid_query_truth(p)
    method = p["method"]
    require(method in (HISTORICAL_METHODS if historical else NEW_METHODS), "Unprescribed method")
    require(identity(p) in expected((method,)), "Unprescribed matrix identity")
    if not historical:
        require(p["actual_updates"] == (0 if method == "HEAD" else 1500), "Only prescribed completed endpoints may be scored")
    ids = p["candidate_ids"]
    require(len(ids) >= 5 and len(set(ids)) == len(ids) and all(isinstance(x, str) for x in ids), "Invalid query IDs")
    require(len(p["architecture_fingerprints"]) == len(ids), "Missing architecture identities")
    center, scale = base.finite(p["y_center"], "center"), base.finite(p["y_scale"], "scale")
    require(scale >= 1e-8, "Invalid frozen label scale")
    for field in ("mean_raw", "mean_standardized"):
        require(len(p[field]) == len(ids), "Prediction vector length differs")
    for z, y in zip(p["mean_standardized"], p["mean_raw"]):
        require(math.isclose(base.finite(y, "mean_raw"), center + scale * base.finite(z, "mean_standardized"),
                             abs_tol=1e-12, rel_tol=1e-10), "Mean unit conversion differs")
    for field in ("endpoint_state_hash", "training_ids_sha256"):
        v = p[field]
        require(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v), "Missing identity hash: " + field)
    if method == "HEAD":
        require(p.get("latent_variance_standardized") is None and p.get("latent_variance_raw") is None
                and p.get("extra_noise_variance_standardized") is None, "HEAD must not invent uncertainty")
        return
    extra = base.finite(p["extra_noise_variance_standardized"], "extra_variance")
    require(extra >= 1e-6 - 1e-14, "Extra noise violates inherited lower bound")
    require(len(p["latent_variance_standardized"]) == len(p["latent_variance_raw"]) == len(ids), "Variance vector length differs")
    for z, v in zip(p["latent_variance_standardized"], p["latent_variance_raw"]):
        require(base.finite(z, "latent_variance_standardized") >= -1e-8, "Invalid latent variance")
        require(math.isclose(base.finite(v, "latent_variance_raw"), scale * scale * z, abs_tol=1e-16, rel_tol=1e-10),
                "Variance unit conversion differs")
    cov = p.get("latent_covariance_standardized")
    if cov is not None:
        require(len(cov) == len(ids) and all(len(r) == len(ids) for r in cov), "Full latent covariance shape differs")
        for i, row in enumerate(cov):
            require(math.isclose(row[i], p["latent_variance_standardized"][i], abs_tol=1e-10, rel_tol=1e-9), "Covariance diagonal differs")
            for j, v in enumerate(row):
                require(math.isfinite(v) and abs(v - cov[j][i]) <= 1e-10, "Covariance nonfinite/asymmetric")
    if method == "GP_SEMI":
        d = p["semi_decomposition"]
        require(len(d["trend_mean_standardized"]) == len(d["matern_residual_mean_standardized"]) == len(ids), "SEMI decomposition shape")
        for mu, tr, gp in zip(p["mean_standardized"], d["trend_mean_standardized"], d["matern_residual_mean_standardized"]):
            require(math.isclose(mu, tr + gp, rel_tol=1e-7, abs_tol=1e-7), "SEMI mean decomposition mismatch")
        bm, bc = d["beta_posterior_mean"], d["beta_posterior_covariance"]
        require(len(bm) == 2 and len(bc) == 2 and all(len(r) == 2 for r in bc), "Beta posterior shape")
        require(all(math.isfinite(x) for x in bm + bc[0] + bc[1]) and min(bc[0][0], bc[1][1]) >= -1e-8, "Invalid beta posterior")


def random5(y, incumbent):
    values = sorted(y)
    denominator = math.comb(len(values), 5)
    weighted = [(v, math.comb(i - 1, 4) / denominator) for i, v in enumerate(values, 1) if i >= 5]
    best = math.fsum(v * w for v, w in weighted)
    gain = math.fsum(max(0, v - incumbent) * w for v, w in weighted)
    return {"random_regret5_pp": 100 * (max(values) - best), "random_gain5_pp": 100 * gain,
            "random_simple_regret_pp": 100 * (max(0, max(values) - incumbent) - gain),
            "random_break_best_probability": math.fsum(w for v, w in weighted if v - incumbent > TOL_RAW)}


def score_unit(p, training, labels):
    """No disk access; synthetic tests can supply labels and selected training."""
    ids = p["candidate_ids"]
    train_ids = training["ids"]
    require(len(train_ids) == p["n"] and len(set(train_ids)) == len(train_ids), "Selected training size differs")
    require(digest(train_ids) == p["training_ids_sha256"] and not set(train_ids) & set(ids), "Training identity or leakage")
    if "scaler" in training:
        require(training["scaler"]["fit_ids"] == train_ids and training["scaler"]["y_mean"] == p["y_center"]
                and training["scaler"]["y_scale"] == p["y_scale"], "Prediction uses different selected-training label scaler")
    for cid, y in zip(train_ids, training["y"]):
        require(math.isclose(y, labels[cid]["validation_mean"], abs_tol=1e-12, rel_tol=0), "Selected training labels differ from frozen scoring source")
    require(len(training["y"]) == len(train_ids), "Training labels length differs")
    y = [labels[c]["validation_mean"] for c in ids]
    mu = p["mean_raw"]
    order = sorted(range(len(ids)), key=lambda i: (-mu[i], ids[i]))
    ranks = {i: j + 1 for j, i in enumerate(order)}
    chosen = order[:5]
    incumbent = max(training["y"])
    best = max(y)
    champions = [i for i, v in enumerate(y) if best - v <= TOL_RAW]
    top20 = sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(.2 * len(ids))]
    improvers = [i for i, v in enumerate(y) if v - incumbent > TOL_RAW]
    selected_best = max(y[i] for i in chosen)
    gain, headroom = max(0, selected_best - incumbent), max(0, best - incumbent)
    errors = [a - b for a, b in zip(mu, y)]
    is_head = p["method"] == "HEAD"
    if is_head:
        var = [None] * len(ids)
        latent_sd = [None] * len(ids)
        sd = [None] * len(ids)
        coverage = [None] * len(ids)
        nlpd = [None] * len(ids)
        width = [None] * len(ids)
        is90 = [None] * len(ids)
    else:
        # Only the latent diagonal, fitted extra variance, and query SE enter.
        # Jitter and integrated beta uncertainty must never be added again.
        var = [v + p["y_scale"] ** 2 * p["extra_noise_variance_standardized"] + labels[c]["validation_se"] ** 2
               for v, c in zip(p["latent_variance_raw"], ids)]
        require(all(math.isfinite(v) and v > 0 for v in var), "Observed variance nonpositive")
        latent_sd = [math.sqrt(max(0, v)) for v in p["latent_variance_raw"]]
        sd = [math.sqrt(v) for v in var]
        coverage = [float(abs(e) <= Z90 * s) for e, s in zip(errors, sd)]
        nlpd = [.5 * (math.log(2 * math.pi * v) + e * e / v) for e, v in zip(errors, var)]
        width = [200 * Z90 * s for s in sd]
        is90 = [100 * (2 * Z90 * s + 20 * max(abs(e) - Z90 * s, 0)) for e, s in zip(errors, sd)]
    r = {k: p[k] for k in IDENTITY}
    correlation, undefined_reason = base.spearman(mu, y)
    r.update(query_size=len(ids), training_best_raw=incumbent, query_best_raw=best,
             selected_top5_ids=[ids[i] for i in chosen], selected_top5_truth_raw=[y[i] for i in chosen],
             selected_top5_prediction_raw=[mu[i] for i in chosen], champion_ids=[ids[i] for i in champions],
             champion_count=len(champions), top20_ids=[ids[i] for i in top20], top20_count=len(top20),
             improver_ids=[ids[i] for i in improvers], improver_count=len(improvers),
             regret5_pp=100 * (best - selected_best), gain5_pp=100 * gain, headroom_pp=100 * headroom,
             simple_regret_pp=100 * (headroom - gain), break_best_rate=float(selected_best - incumbent > TOL_RAW),
             has_headroom_fraction=float(headroom > TOL_RAW), top1_regret_pp=100 * (best - y[order[0]]),
             champion_top5_hit_fraction=float(bool(set(champions) & set(chosen))),
             champion_prediction_rank=min(ranks[i] for i in champions),
             mae_pp=100 * base.mean([abs(e) for e in errors]), rmse_pp=100 * math.sqrt(base.mean([e * e for e in errors])),
             bias_pp=100 * base.mean(errors), spearman=correlation, spearman_undefined_reason=undefined_reason,
             nlpd_accuracy_units=base.mean(nlpd), coverage90_fraction=base.mean(coverage),
             interval90_width_pp=base.mean(width), IS90_pp=base.mean(is90),
             improver_mean_prediction_rank=base.mean([ranks[i] for i in improvers]),
             improver_best_prediction_rank=min([ranks[i] for i in improvers], default=None),
             historical_reference=p["method"] in HISTORICAL_METHODS,
             prediction_file_sha256=p.get("prediction_file_sha256"), endpoint_state_hash=p["endpoint_state_hash"])
    r.update(random5(y, incumbent))
    for name, ix in (("champion", champions), ("top20", top20)):
        r[name + "_mae_pp"] = 100 * base.mean([abs(errors[i]) for i in ix])
        r[name + "_rmse_pp"] = 100 * math.sqrt(base.mean([errors[i] ** 2 for i in ix]))
        r[name + "_bias_pp"] = 100 * base.mean([errors[i] for i in ix])
        r[name + "_coverage90_fraction"] = base.mean([coverage[i] for i in ix])
    candidates = []
    for i, cid in enumerate(ids):
        c = {k: p[k] for k in IDENTITY}
        c.update(candidate_id=cid, truth_raw=y[i], query_se_raw=labels[cid]["validation_se"], mean_raw=mu[i],
                 prediction_rank=ranks[i], selected_top5=i in chosen, true_champion=i in champions,
                 true_top20=i in top20, can_break_incumbent=i in improvers, signed_error_pp=100 * errors[i],
                 latent_sd_raw=latent_sd[i], observed_sd_raw=sd[i], observed_variance_raw=var[i],
                 coverage90=coverage[i], nlpd_accuracy_units=nlpd[i], interval90_width_pp=width[i], IS90_pp=is90[i])
        if p["method"] == "GP_SEMI":
            d = p["semi_decomposition"]
            c.update(semi_trend_standardized=d["trend_mean_standardized"][i],
                     semi_matern_residual_standardized=d["matern_residual_mean_standardized"][i],
                     semi_trend_raw_including_center=p["y_center"] + p["y_scale"] * d["trend_mean_standardized"][i],
                     semi_matern_residual_raw=p["y_scale"] * d["matern_residual_mean_standardized"][i])
        candidates.append(c)
    return r, candidates


def aggregate(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for values, group in sorted(groups.items(), key=lambda x: str(x[0])):
        row = dict(zip(fields, values)); row["child_count"] = len(group)
        for metric in metrics:
            values = [g.get(metric) for g in group]
            row[metric] = base.mean(values)
            row[metric + "_defined_count"] = sum(v is not None for v in values)
        result.append(row)
    return result


def hierarchy(rows, metrics, method="method"):
    subset = aggregate(rows, ("n", method, "fold", "sample_seed"), metrics)
    folds = aggregate(subset, ("n", method, "fold"), metrics)
    samples = aggregate(subset, ("n", method, "sample_seed"), metrics)
    model_folds = aggregate(rows, ("n", method, "model_seed", "fold"), metrics)
    models = aggregate(model_folds, ("n", method, "model_seed"), metrics)
    return subset, folds, samples, models


def primary_rule(fold_values, seed_values):
    return bool(len(fold_values) == 5 and len(seed_values) == 2 and
                base.mean(fold_values) >= .20 and sum(v > TOL_PP for v in fold_values) >= 4 and
                all(v > TOL_PP for v in seed_values))


def directions(values):
    return {"positive": sum(v > TOL_PP for v in values), "tied": sum(abs(v) <= TOL_PP for v in values),
            "negative": sum(v < -TOL_PP for v in values)}


def make_pairs(rows):
    lookup = {identity(r): r for r in rows}
    pairs = []
    for r in rows:
        if r["method"] not in ("GP_SEMI", "GP_TREND"):
            continue
        references = ("GP_CONST", "GP_TREND", "HEAD", "RAW16", "R_FROZEN", "R_REG", "R_DKL_C") if r["method"] == "GP_SEMI" else ("GP_CONST",)
        for method in references:
            ref = lookup.get((r["n"], r["fold"], r["sample_seed"], None if method == "RAW16" else r["model_seed"], method))
            if ref is None:
                continue
            p = {k: r[k] for k in ("context_id", "n", "fold", "sample_seed", "model_seed")}
            p.update(comparison=r["method"] + "_vs_" + method, target_method=r["method"], reference_method=method,
                     target_unit_id=r["unit_id"], reference_unit_id=ref["unit_id"],
                     RAW16_shared_reference_not_independent=method == "RAW16",
                     historical_unequal_steps=method in HISTORICAL_METHODS)
            for name, metric, sign in (("regret5_improvement_pp", "regret5_pp", -1), ("gain5_improvement_pp", "gain5_pp", 1),
                                       ("simple_regret_improvement_pp", "simple_regret_pp", -1), ("nlpd_increase_nats", "nlpd_accuracy_units", 1),
                                       ("IS90_increase_pp", "IS90_pp", 1), ("coverage90_change_fraction", "coverage90_fraction", 1),
                                       ("champion_coverage90_change_fraction", "champion_coverage90_fraction", 1)):
                a, b = r[metric], ref[metric]
                p[name] = sign * (a - b) if a is not None and b is not None else None
            pairs.append(p)
    return pairs


def conclusions(n, means, pair_subsets, pair_folds, pair_samples, stability, complete, unit_rows):
    if not complete:
        return {"flags": {name: None for name in FLAGS}, "secondary_GP_TREND_selection_signal": None,
                "status": "INCOMPLETE_NO_SCIENTIFIC_CLASSIFICATION"}
    required = set(NEW_METHODS)
    require(required <= set(means), "Missing full-matrix method means")
    def values(rows, comp, field):
        return [r[field] for r in rows if r["n"] == n and r["comparison"] == comp]
    def signal(comp):
        return primary_rule(values(pair_folds, comp, "regret5_improvement_pp"), values(pair_samples, comp, "regret5_improvement_pp"))
    semi, const, head = (means[m] for m in ("GP_SEMI", "GP_CONST", "HEAD"))
    supporting = [r for r in pair_subsets if r["n"] == n and r["comparison"] == "GP_SEMI_vs_GP_CONST" and r["gain5_improvement_pp"] > 0]
    supporter_details = []
    for r in supporting:
        units = [u for u in unit_rows if u["n"] == n and u["fold"] == r["fold"] and u["sample_seed"] == r["sample_seed"] and u["method"] == "GP_SEMI"]
        supporter_details.append({"fold": r["fold"], "sample_seed": r["sample_seed"], "gain5_improvement_pp": r["gain5_improvement_pp"],
                                  "champion_ids": sorted({c for u in units for c in u["champion_ids"]})})
    sharing = []
    for i, a in enumerate(supporter_details):
        for b in supporter_details[i + 1:]:
            sharing.append({"a": [a["fold"], a["sample_seed"]], "b": [b["fold"], b["sample_seed"]],
                            "shared_champion_ids": sorted(set(a["champion_ids"]) & set(b["champion_ids"]))})
    probability_deltas = {
        "nlpd_increase_nats_per_candidate": semi["nlpd_accuracy_units"] - const["nlpd_accuracy_units"],
        "IS90_increase_pp": semi["IS90_pp"] - const["IS90_pp"],
        "coverage90_change_percentage_points": 100 * (semi["coverage90_fraction"] - const["coverage90_fraction"]),
        "champion_coverage90_change_percentage_points": 100 * (semi["champion_coverage90_fraction"] - const["champion_coverage90_fraction"]),
    }
    counts = {m: sum(r["numeric_predictive_stable"] for r in stability if r["n"] == n and r["method"] == m)
              for m in NEW_METHODS if m != "HEAD"}
    flags = {
        "PRIMARY_SELECTION_SIGNAL": signal("GP_SEMI_vs_GP_CONST"),
        "HEAD_SIGNAL_RETAINED": semi["regret5_pp"] <= head["regret5_pp"] + .10 and semi["gain5_pp"] >= head["gain5_pp"] - .02,
        "ADDITIONAL_OVER_HEAD_SIGNAL": signal("GP_SEMI_vs_HEAD") and semi["gain5_pp"] >= head["gain5_pp"],
        "INCUMBENT_GAIN_SIGNAL": semi["gain5_pp"] > const["gain5_pp"] and len(supporting) >= 2,
        "PROBABILITY_NO_LARGE_DEGRADATION": probability_deltas["nlpd_increase_nats_per_candidate"] <= .10 and
            probability_deltas["IS90_increase_pp"] <= 1.0 and probability_deltas["coverage90_change_percentage_points"] >= -5 and
            probability_deltas["champion_coverage90_change_percentage_points"] >= -5,
        "OPTIMIZATION_READY": counts["GP_CONST"] >= 24 and counts["GP_SEMI"] >= 24,
    }
    flags["READOUT_CANDIDATE_FOR_NEXT_PROTOCOL_REVIEW"] = all(flags[k] for k in (
        "PRIMARY_SELECTION_SIGNAL", "HEAD_SIGNAL_RETAINED", "INCUMBENT_GAIN_SIGNAL", "PROBABILITY_NO_LARGE_DEGRADATION", "OPTIMIZATION_READY"))
    return {"flags": flags, "secondary_GP_TREND_selection_signal": signal("GP_TREND_vs_GP_CONST"),
            "numeric_predictive_stable_counts_of_30": counts, "probability_deltas": probability_deltas,
            "gain_supporting_subsets": supporter_details, "gain_subset_champion_sharing": sharing,
            "main_question": n == 50, "further_execution_authorized": False,
            "scope": "developmental frozen-readout evidence on repeatedly used A; neither joint-DKL nor actual-search evidence"}


def stability_rows(predictions):
    result = []
    for p in predictions:
        if p["method"] not in NEW_METHODS or p["method"] == "HEAD":
            continue
        s = p["stability"]
        require(isinstance(s["numeric_predictive_stable"], bool), "Missing three-window stability boolean")
        windows = s["windows"]
        require(len(windows) == 3, "Stability requires all three prescribed windows")
        require([(w["from_step"], w["to_step"]) for w in windows] == [(1200, 1300), (1300, 1400), (1400, 1500)],
                "Stability window endpoints differ from frozen rule")
        rechecked = []
        for w in windows:
            criteria = {"mll_stable": w["mll_relative_change"] <= 1e-4,
                        "mean_stable": w["query_mean_max_abs_change_raw"] <= .001,
                        "latent_sd_stable": w["query_latent_sd_max_abs_change_raw"] <= .001,
                        "kernels_stable": w["matern_training_kernel_relative_change"] <= .01 and w["full_training_kernel_relative_change"] <= .01}
            for field in ("mll_relative_change", "query_mean_max_abs_change_raw", "query_latent_sd_max_abs_change_raw",
                          "matern_training_kernel_relative_change", "full_training_kernel_relative_change"):
                require(math.isfinite(w[field]) and w[field] >= 0, "Stability change must be finite nonnegative")
            require(criteria == w["criteria"] and all(criteria.values()) == w["numeric_predictive_stable"], "Stability threshold result mismatch")
            rechecked.append(all(criteria.values()))
        require(all(rechecked) == s["numeric_predictive_stable"], "Three-window conjunction differs")
        require(s["top5_stable"] == all(w["top5_stable"] for w in windows), "Top5 stability conjunction differs")
        result.append({**{k: p[k] for k in IDENTITY}, **s})
    return result


def summarize(rows, stability, complete):
    subset, folds, samples, models = hierarchy(rows, METRICS)
    pairs = make_pairs(rows)
    psubset, pfolds, psamples, pmodels = hierarchy(pairs, PAIR_METRICS, "comparison")
    summary = {"execution_status": "COMPLETE_VALID" if complete else "PARTIAL", "complete_matrix": complete,
               "new_endpoint_count": sum(r["method"] in NEW_METHODS for r in rows),
               "historical_reference_count": sum(r["method"] in HISTORICAL_METHODS for r in rows),
               "RAW16_actual_historical_fits": sum(r["method"] == "RAW16" for r in rows),
               "aggregation": "paired model seeds -> two sample subsets per fold -> five equally weighted folds",
               "probability_units": {"NLPD": "natural log, raw accuracy proportion; nat/candidate", "coverage": "fraction", "IS90_and_width": "pp"},
               "HEAD_probability": "N/A; deterministic head has no supplied uncertainty", "per_n": {},
               "historical_reference_scope": "old source and unequal fitting budgets; no new fitting or equal-budget claim",
               "A_repeated_development_not_independent_confirmation": True, "next_stage_authorized": False}
    for n in base.NS:
        means = {m: {metric: base.mean([r[metric] for r in folds if r["n"] == n and r["method"] == m]) for metric in METRICS}
                 for m in METHODS if any(r["n"] == n and r["method"] == m for r in rows)}
        comparisons = {}
        for comp in sorted({r["comparison"] for r in pairs if r["n"] == n}):
            f = [r for r in pfolds if r["n"] == n and r["comparison"] == comp]
            s = [r for r in psamples if r["n"] == n and r["comparison"] == comp]
            comparisons[comp] = {"fold_deltas": {str(r["fold"]): r["regret5_improvement_pp"] for r in f},
                                 "sample_seed_deltas": {str(r["sample_seed"]): r["regret5_improvement_pp"] for r in s},
                                 "mean_deltas": {metric: base.mean([r[metric] for r in f]) for metric in PAIR_METRICS},
                                 "fold_directions": directions([r["regret5_improvement_pp"] for r in f])}
        summary["per_n"][str(n)] = {"method_means": means, "comparisons": comparisons,
            **conclusions(n, means, psubset, pfolds, psamples, stability, complete, rows)}
    tables = {"unit_metrics": rows, "subset_metrics": subset, "fold_metrics": folds, "sample_seed_metrics": samples,
              "model_seed_metrics": models, "paired_unit": pairs, "paired_subset": psubset, "paired_fold": pfolds,
              "paired_sample_seed": psamples, "paired_model_seed": pmodels}
    conditioned = [r for r in rows if r["has_headroom_fraction"] == 1]
    cs, cf, css, cm = hierarchy(conditioned, METRICS)
    tables.update(headroom_conditioned_units=conditioned, headroom_conditioned_subsets=cs,
                  headroom_conditioned_folds=cf, headroom_conditioned_sample_seeds=css)
    summary["headroom_conditioned_descriptive_only"] = {
        str(n): {m: {"eligible_trajectories": sum(r["n"] == n and r["method"] == m for r in conditioned),
                     "eligible_subsets": sum(r["n"] == n and r["method"] == m for r in cs),
                     "gain5_pp_equal_eligible_folds": base.mean([r["gain5_pp"] for r in cf if r["n"] == n and r["method"] == m])}
                 for m in METHODS} for n in base.NS}
    summary["READOUT_CANDIDATE_FOR_NEXT_PROTOCOL_REVIEW"] = summary["per_n"]["50"]["flags"]["READOUT_CANDIDATE_FOR_NEXT_PROTOCOL_REVIEW"]
    return summary, tables


def read_sealed(root):
    """Complete file verification precedes ANY label-source access."""
    manifest = base.read_json(root / "predictions_manifest.json")
    require(manifest.get("sealed") is True, "Uniform prediction seal required before scoring")
    require(manifest.get("execution_status") == "COMPLETE_VALID", "Invalid or blocked run cannot be scored")
    require(manifest["expected_endpoint_count"] == 240, "Endpoint target changed")
    for name in ("protocol", "code_manifest", "source_manifest"):
        checked_file(root, name + ".json", manifest[name + "_sha256"])
    for name in ("frozen_manifest", "source_manifest", "code_manifest"):
        doc = base.read_json(root / (name + ".json"))
        for path, sha in doc["files"].items():
            checked_file(root, path, sha)
    source = base.read_json(root / "source_manifest.json")
    rows = []
    seen_ids, seen = set(), set()
    for entry in manifest["files"]:
        p = base.read_json(checked_file(root, entry["path"], entry["sha256"]))
        validate_prediction(p)
        require(p["unit_id"] == entry["unit_id"] and p["unit_id"] not in seen_ids and identity(p) not in seen, "Duplicate/mismatched endpoint")
        seen_ids.add(p["unit_id"]); seen.add(identity(p))
        state_hash = entry.get("state_file_sha256", entry.get("state_sha256"))
        require(entry.get("state_path") and state_hash, "Missing sealed source/endpoint state identity")
        checked_file(root, entry["state_path"], state_hash)
        require(p.get("state_path") == entry["state_path"] and p.get("state_file_sha256") == state_hash, "Prediction and seal state differ")
        require(entry.get("endpoint_state_hash", p["endpoint_state_hash"]) == p["endpoint_state_hash"], "Model state fingerprint differs")
        p["prediction_file_sha256"] = entry["sha256"]
        rows.append(p)
    complete = manifest["execution_status"] == "COMPLETE_VALID" and seen == expected(NEW_METHODS)
    require(complete and len(rows) == 240, "Complete status misses prescribed matrix")
    require(sum(p["actual_updates"] for p in rows) == 270000, "Complete endpoint update count differs")
    protocol = base.read_json(root / "protocol.json")
    budget = base.read_json(root / "budget.json")
    require(budget["limit"] == protocol["budget"]["model_seconds"] == 1500, "Frozen V2 budget differs")
    require(budget["formal_updates"] == budget["formal_optimizer_calls"] == 270000, "Formal accounting incomplete")
    require(manifest["resources"]["formal_updates"] == manifest["resources"]["formal_optimizer_calls"] == 270000, "Seal accounting incomplete")
    require(budget["seconds"] <= budget["limit"] and budget["synthetic_optimizer_calls"] <= 300, "Model budget exceeded")
    historical = []
    historical_entries = source.get("historical_predictions")
    if historical_entries is None:
        historical_entries = [{"path": path, "sha256": sha, "unit_id": Path(path).stem}
                              for path, sha in source["files"].items() if path.startswith("source/historical_predictions/") and path.endswith(".json")]
    for entry in historical_entries:
        p = base.read_json(checked_file(root, entry["path"], entry["sha256"]))
        validate_prediction(p, historical=True)
        require(p["unit_id"] == entry["unit_id"], "Historical prediction identity mismatch")
        p["prediction_file_sha256"] = entry["sha256"]
        historical.append(p)
    require({identity(r) for r in historical} == expected(HISTORICAL_METHODS) and len(historical) == 200,
            "Historical reference matrix must contain 20 RAW and 60 of each fixed reference")
    allrows = rows + historical
    fold_ids, fold_arches, subset_hashes = {}, {}, {}
    for p in allrows:
        f, s = p["fold"], p["subset_id"]
        if f in fold_ids:
            require(fold_ids[f] == p["candidate_ids"] and fold_arches[f] == p["architecture_fingerprints"], "Historical/new query IDs or architecture order differ; incomparable")
        else:
            fold_ids[f], fold_arches[f] = p["candidate_ids"], p["architecture_fingerprints"]
        if s in subset_hashes:
            require(subset_hashes[s] == p["training_ids_sha256"], "Training identity differs across paired methods")
        subset_hashes[s] = p["training_ids_sha256"]
    allids = [cid for f in sorted(fold_ids) for cid in fold_ids[f]]
    require(len(allids) == len(set(allids)) == 179 and base.a_id_digest(allids) == base.A_ID_HASH, "Query folds differ from fixed legal A")
    return manifest, source, allrows, complete


def load_labels(root, source, labels_from=None):
    destination = root / "scoring" / "labels_A.json"
    if not destination.exists():
        src = Path(labels_from or source.get("label_source_path") or source["source_label_path"])
        require(base.sha256(src) == source["scoring_labels_expected_sha256"], "Legal A label source hash differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
    require(base.sha256(destination) == source["scoring_labels_expected_sha256"], "Scoring labels differ from pinned source")
    base.write_json(root / "scoring" / "release.json", {"released_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_manifest_sha256": base.sha256(root / "predictions_manifest.json"),
        "labels_A_sha256": base.sha256(destination), "all_prediction_hashes_verified_before_label_access": True,
        "training_may_not_resume_after_this_release": True})
    return base.checked_labels(destination, set())


def candidate_explanations(candidates):
    contexts = defaultdict(dict)
    for c in candidates:
        if c["method"] in NEW_METHODS:
            contexts[(c["context_id"], c["candidate_id"])][c["method"]] = c
    result = []
    for (ctx, cid), methods in sorted(contexts.items()):
        example = next(iter(methods.values()))
        if not (example["true_champion"] or example["can_break_incumbent"]):
            continue
        row = {k: example[k] for k in ("context_id", "subset_id", "n", "fold", "sample_seed", "model_seed",
                                       "candidate_id", "truth_raw", "query_se_raw", "true_champion", "can_break_incumbent")}
        for method in NEW_METHODS:
            value = methods.get(method)
            for field in ("mean_raw", "latent_sd_raw", "observed_sd_raw", "prediction_rank", "selected_top5"):
                row[method + "_" + field] = value[field] if value else None
        for field in ("semi_trend_standardized", "semi_matern_residual_standardized", "semi_trend_raw_including_center", "semi_matern_residual_raw"):
            row[field] = methods.get("GP_SEMI", {}).get(field)
        row["all_four_new_methods_present"] = set(methods) == set(NEW_METHODS)
        result.append(row)
    return result


def score(root, labels_from=None):
    start = time.perf_counter()
    seal, source, predictions, complete = read_sealed(root)
    stability = stability_rows(predictions)
    labels = load_labels(root, source, labels_from)
    rows, candidates, beta = [], [], []
    train_cache = {}
    for p in predictions:
        sid = p["subset_id"]
        if sid not in train_cache:
            train_cache[sid] = base.read_json(root / "training" / (sid + ".json"))
        r, cc = score_unit(p, train_cache[sid], labels)
        rows.append(r); candidates.extend(cc)
        if p["method"] == "GP_SEMI":
            d = p["semi_decomposition"]
            beta.append({**{k: p[k] for k in IDENTITY}, "intercept_posterior_mean_standardized": d["beta_posterior_mean"][0],
                "slope_posterior_mean": d["beta_posterior_mean"][1],
                "intercept_posterior_sd_standardized": math.sqrt(max(0, d["beta_posterior_covariance"][0][0])),
                "slope_posterior_sd": math.sqrt(max(0, d["beta_posterior_covariance"][1][1])),
                "intercept_slope_posterior_covariance": d["beta_posterior_covariance"][0][1]})
    summary, tables = summarize(rows, stability, complete)
    tables.update(candidate_metrics=candidates, champion_and_improver_details=candidate_explanations(candidates), beta_posterior=beta)
    output = root / "scoring"
    for name, table in tables.items():
        base.write_csv(output / (name + ".csv"), table)
    base.write_json(output / "stability.json", {"trajectory_count": len(stability), "trajectories": stability,
        "rule": "all four numeric/predictive conditions in all three fixed windows; top5 stability reported separately"})
    random = {}
    for r in rows:
        random.setdefault(r["subset_id"], {k: r[k] for k in ("subset_id", "n", "fold", "sample_seed", "headroom_pp",
                                                           "training_best_raw", "query_best_raw", "random_regret5_pp", "random_gain5_pp",
                                                           "random_simple_regret_pp", "random_break_best_probability")})
    base.write_json(output / "random_reference.json", random)
    ledger = base.read_json(root / "budget.json")
    fit_costs = base.read_json(root / "fit_costs.json")
    costs = {"new_model_budget_ledger": ledger, "old_model_cost_not_charged_to_new_budget": True,
             "historical_fit_count_not_added_to_current_180_GP_fits": True,
             "completed_GP_branch_costs": fit_costs,
             "cost_scope": "budget.json includes all new verification, HEAD, GP training and diagnostics; branch model totals do not sum to whole-run total",
             "new_scientific_endpoint_optimizer_updates": sum(p["actual_updates"] for p in predictions if p["method"] in NEW_METHODS),
             "historical_model_seconds_reference": base.read_json(root / "protocol.json")["budget"]["old_model_seconds"]}
    base.write_json(output / "costs.json", costs)
    summary.update(predictions_manifest_sha256=base.sha256(root / "predictions_manifest.json"),
                   labels_A_sha256=base.sha256(output / "labels_A.json"), scorer_sha256=base.sha256(__file__),
                   statistical_scoring_seconds=time.perf_counter() - start, scoring_performed_at_utc=datetime.now(timezone.utc).isoformat())
    base.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels-from")
    args = parser.parse_args()
    result = score(Path(args.root), args.labels_from)
    print({"execution_status": result["execution_status"], "new_endpoint_count": result["new_endpoint_count"],
           "n50_flags": result["per_n"]["50"]["flags"]})


if __name__ == "__main__":
    main()
