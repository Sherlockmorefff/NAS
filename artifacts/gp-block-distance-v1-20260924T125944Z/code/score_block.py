#!/usr/bin/env python3
"""Frozen block-distance experiment: statistics only, complete-matrix release only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import score_stage2 as base
import score_oof as metric_math

NEW = ("RAW16_ISO", "HP4_ISO", "RAW16_BLOCK2")
HISTORY = ("HEAD", "H_CONST", "H_TREND")
METHODS = NEW + HISTORY
PARENT = {"HEAD": "HEAD", "H_CONST": "GP_CONST", "H_TREND": "GP_TREND"}
REGIONS = ("all_Q", "HEAD_pred20", "true_top20", "true_bottom20", "champions")
SEEDS = (20260922, 20260923)
META = ("unit_id", "subset_id", "fold", "sample_seed", "model_seed", "method")
TOL = 1e-9
Z = metric_math.Z
SELECTION = ("regret5_pp", "gain5_pp", "headroom_pp", "simple_regret_pp", "break_best_rate", "top1_regret_pp",
             "champion_prediction_rank", "champion_top5_hit_fraction")
REGION_METRICS = ("mae_pp", "rmse_pp", "bias_pp", "nlpd_accuracy_units", "IS90_pp") + tuple(
    f"{name}{level}{suffix}" for level in Z for name, suffix in (("coverage", "_fraction"), ("interval", "_width_pp"))) + tuple(
    f"coverage{level}_target_distance_fraction" for level in Z)
MAIN = SELECTION + ("mae_pp", "rmse_pp", "bias_pp", "spearman", "nlpd_accuracy_units", "IS90_pp") + tuple(
    f"{name}{level}{suffix}" for level in Z for name, suffix in (("coverage", "_fraction"), ("interval", "_width_pp")))
METRICS = MAIN + tuple(r + "__" + m for r in REGIONS for m in REGION_METRICS)
COMPARISONS = ((NEW[2], NEW[0]), (NEW[2], NEW[1]), (NEW[1], NEW[0])) + tuple((a, b) for a in NEW for b in HISTORY)


def require(ok, message):
    if not ok: raise ValueError(message)


def close(a, b, message, atol=1e-12, rtol=1e-10):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol), message)


def checked(root, path, sha):
    p = base.safe_child(root, path)
    require(base.sha256(p) == sha, "Hash mismatch: " + str(path))
    return p


def hash_files(root, manifest):
    files = manifest["files"]
    for path, sha in files.items(): checked(root, path, sha)


def count(record, fields, expected):
    values = [record[k] for k in fields if k in record]
    require(values and all(x == expected for x in values), "Incomplete count: " + "/".join(fields))


def stable(p):
    s = p["stability"]
    windows = s["windows"]
    require([(w["from_step"], w["to_step"]) for w in windows] == [(1200, 1300), (1300, 1400), (1400, 1500)], "Wrong stability windows")
    reasons = defaultdict(int)
    for w in windows:
        criteria = {"mll": w["mll_relative_change"] <= 1e-4,
                    "query_mean": w["query_mean_max_abs_change_raw"] <= .001,
                    "query_latent_sd": w["query_latent_sd_max_abs_change_raw"] <= .001,
                    "training_kernel": w.get("full_training_kernel_relative_change", w.get("training_kernel_relative_change", float("inf"))) <= .01}
        require(all(math.isfinite(w[k]) and w[k] >= 0 for k in ("mll_relative_change", "query_mean_max_abs_change_raw", "query_latent_sd_max_abs_change_raw")), "Invalid stability statistic")
        for k, v in criteria.items(): reasons[k] += int(not v)
    result = not any(reasons.values())
    require(result == s["numeric_predictive_stable"], "Stability boolean differs from fixed criteria")
    return result, dict(reasons)


def model_variance(p, method):
    if method == "HEAD": return None
    latent = p["latent_variance_standardized"]
    require(len(latent) == len(p["candidate_ids"]), "Latent diagonal shape")
    scale, extra = p["y_scale"], p["extra_noise_variance_standardized"]
    require(math.isfinite(scale) and scale >= 1e-8 and math.isfinite(extra) and extra >= 1e-6 - 1e-14, "Invalid variance scale or extra noise")
    require(all(math.isfinite(x) and x >= -1e-8 for x in latent), "Invalid latent variance")
    values = [scale * scale * (max(v, 0) + extra) for v in latent]
    require(all(v > 0 and math.isfinite(v) for v in values), "Nonpositive model variance")
    if "model_variance_raw" in p:
        require(len(p["model_variance_raw"]) == len(values), "Model variance vector shape")
        for a, b in zip(values, p["model_variance_raw"]): close(a, b, "Model variance must contain extra noise once, no SE/tau")
    require(p.get("tau_squared_raw", 0) in (0, None), "Tau forbidden in this experiment")
    return values


def endpoint_nll(p):
    value = p.get("endpoint_nll_after_update", p.get("negative_mll"))
    require(isinstance(value, (int, float)) and math.isfinite(value), "Missing finite endpoint per-observation NLL")
    return value


def sealed_inputs(root):
    """Only label-free predictions/masks plus frozen-byte hashes are read here."""
    seal = base.read_json(root / "predictions_manifest.json")
    require(seal.get("sealed") is True and seal.get("execution_status") == "COMPLETE_VALID", "Complete valid seal required before Q labels")
    for fields, value in ((("new_endpoint_count", "completed_new_endpoints"), 30), (("historical_reference_count",), 90),
                          (("raw_unit_count", "completed_endpoints"), 120), (("mask_count",), 30), (("formal_updates",), 54000)):
        count(seal, fields, value)
    for name in ("protocol", "source_manifest", "code_manifest", "worklist", "masks_manifest", "frozen_manifest", "data_manifest"):
        checked(root, name + ".json", seal[name + "_sha256"])
    for name in ("source_manifest", "code_manifest", "frozen_manifest", "masks_manifest"):
        manifest = base.read_json(root / (name + ".json"))
        hash_files(root, manifest)
    hash_files(root, seal)
    protocol = base.read_json(root / "protocol.json"); budget = base.read_json(root / "budget.json")
    lim = protocol["budget"].get("model_seconds", protocol["budget"].get("limit"))
    require(lim == budget["limit"] == 600 and budget["seconds"] <= 600, "Wrong/exceeded new model budget")
    require(budget["formal_updates"] == budget["formal_optimizer_calls"] == 54000 and budget["synthetic_optimizer_calls"] <= 300, "Wrong optimizer accounting")
    refs = base.read_json(root / "historical_refs.json")["records"]
    require(len(refs) == 90, "History must have 90 frozen references")
    reflut = {r["unit_id"]: r for r in refs}
    require(len(reflut) == 90, "Duplicate historical references")
    records = seal["records"]
    require(len(records) == 120 and len({r["unit_id"] for r in records}) == 120, "Expected exactly 120 unique raw units")
    inputs = []; seen = set(); queries = {}; subset_ids = {}
    for r in records:
        p = base.read_json(checked(root, r["path"], r["sha256"]))
        require(seal["files"].get(r["path"]) == r["sha256"], "Record not in sealed files")
        base.forbid_query_truth(p)
        method = r["method"]; require(method in METHODS, "Unknown method")
        key = (r["fold"], r["sample_seed"], r["model_seed"], method)
        require(key not in seen, "Duplicate scientific unit"); seen.add(key)
        require(r["fold"] in range(5) and r["sample_seed"] in SEEDS, "Wrong fold/sample")
        for k in ("subset_id", "fold", "sample_seed", "model_seed"):
            require(p[k] == r[k], "Prediction metadata mismatch: " + k)
        sid = r["subset_id"]; skey = (r["fold"], r["sample_seed"])
        require(skey not in subset_ids or subset_ids[skey] == sid, "Subset identity mismatch"); subset_ids[skey] = sid
        ids = p["candidate_ids"]
        require(len(ids) == (38, 36, 36, 34, 35)[r["fold"]] and len(set(ids)) == len(ids), "Wrong query identity count")
        require(len(p["mean_raw"]) == len(p["architecture_fingerprints"]) == len(ids) and all(math.isfinite(v) for v in p["mean_raw"]), "Invalid prediction shape/means")
        if r["fold"] in queries:
            require(queries[r["fold"]] == (ids, p["architecture_fingerprints"]), "Query ordering differs")
        queries[r["fold"]] = (ids, p["architecture_fingerprints"])
        if method in NEW:
            require(r["model_seed"] is None and p["method"] == method and p.get("rng_seed", p.get("execution_rng_seed")) == 0, "New method must be deterministic/null with RNG 0")
            require(p.get("actual_updates", p.get("total_steps")) == 1800, "Fixed 1800 endpoint required")
            checked(root, p["state_path"], p["state_file_sha256"])
            state = p["stage_states"]["FIT_GP"]
            require((state["path"], state["sha256"], state["fingerprint"]) == (p["state_path"], p["state_file_sha256"], p["endpoint_state_hash"]), "Endpoint state identity differs")
            require(p["latent_covariance_symmetry_max_abs"] <= 1e-10 and p["latent_covariance_min_eigenvalue"] >= -1e-8, "Endpoint covariance numerical check failed")
            cov, rawcov = p["latent_covariance_standardized"], p["latent_covariance_raw"]
            require(len(cov) == len(rawcov) == len(ids) and all(len(row) == len(ids) for row in cov + rawcov), "Full latent covariance missing")
            for i in range(len(ids)):
                close(p["mean_raw"][i], p["y_center"] + p["y_scale"] * p["mean_standardized"][i], "Raw mean transform differs")
                close(cov[i][i], p["latent_variance_standardized"][i], "Latent diagonal differs")
                for j in range(len(ids)):
                    close(rawcov[i][j], p["y_scale"] ** 2 * cov[i][j], "Latent covariance raw transform differs")
                    require(abs(cov[i][j] - cov[j][i]) <= 1e-10, "Full standardized covariance asymmetric")
            stable(p)
            endpoint_nll(p)
        else:
            require(r["model_seed"] in range(3) and p["method"] == PARENT[method], "Historical method/seed mismatch")
            old = reflut[r["unit_id"]]
            require(all(r[k] == old[k] for k in ("path", "sha256", "method", "subset_id", "fold", "sample_seed", "model_seed")), "Historical reference changed")
        values = model_variance(p, method)
        inputs.append({"record": r, "prediction": p, "model_variance": values})
    expected = {(f, s, None, m) for f in range(5) for s in SEEDS for m in NEW} | {(f, s, k, m) for f in range(5) for s in SEEDS for k in range(3) for m in HISTORY}
    require(seen == expected, "Full prescribed scientific matrix required")
    masks = base.read_json(root / "masks_manifest.json")["records"]
    require(len(masks) == 30 and len({(m["subset_id"], m["model_seed"]) for m in masks}) == 30, "30 unique pre-frozen HEAD masks required")
    heads = {(x["record"]["subset_id"], x["record"]["model_seed"]): x for x in inputs if x["record"]["method"] == "HEAD"}
    for m in masks:
        h = heads[m["subset_id"], m["model_seed"]]; p = h["prediction"]; r = h["record"]
        require(m["query_ids"] == p["candidate_ids"], "Mask query ordering differs")
        require(m["source_prediction_path"] == r["path"] and m["source_prediction_sha256"] == r["sha256"], "Mask source differs")
        wanted = sorted(range(len(p["candidate_ids"])), key=lambda i: (-p["mean_raw"][i], p["candidate_ids"][i]))[:math.ceil(.2 * len(p["candidate_ids"]))]
        require(m["candidate_ids"] == [p["candidate_ids"][i] for i in wanted], "Mask must be exact individual HEAD-seed top20")
        saved = base.read_json(checked(root, m["path"], m["sha256"]))
        require(all(saved[k] == m[k] for k in ("subset_id", "model_seed", "candidate_ids", "query_ids", "source_prediction_path", "source_prediction_sha256")), "Mask record/file differ")
    return seal, inputs, masks, budget


def release_labels(root, inputs, labels_from=None):
    source = base.read_json(root / "source_manifest.json")
    expected = source.get("legal_A_scoring_truth_sha256", source.get("expected_labels_sha256", source.get("scoring_labels_expected_sha256")))
    require(expected, "Missing source label hash")
    out = root / "scoring"; out.mkdir(exist_ok=True)
    target = out / "labels_A.json"
    path = Path(labels_from or source.get("legal_A_scoring_truth_path", source.get("source_label_path", source.get("label_source_path", target))))
    require(base.sha256(path) == expected, "A label source hash differs")
    if path.resolve() != target.resolve(): shutil.copyfile(path, target)
    labels = base.checked_labels(target, {i for x in inputs for i in x["prediction"]["candidate_ids"]})
    base.write_json(out / "release.json", {"released_at_utc": datetime.now(timezone.utc).isoformat(), "predictions_manifest_sha256": base.sha256(root / "predictions_manifest.json"),
                    "labels_sha256": base.sha256(target), "new_endpoints": 30, "historical_references": 90, "head_masks": 30, "formal_updates": 54000,
                    "further_model_training_authorized": False})
    return labels


def region_metrics(ids, mu, truth, variance, chosen):
    require(chosen and len(set(chosen)) == len(chosen), "Region cannot be empty/duplicated")
    error = [mu[i] - truth[i] for i in chosen]
    result = {"mae_pp": 100 * base.mean([abs(e) for e in error]), "rmse_pp": 100 * math.sqrt(base.mean([e * e for e in error])), "bias_pp": 100 * base.mean(error)}
    if variance is None:
        return {**result, **{k: None for k in REGION_METRICS if k not in result}}
    sd = [math.sqrt(variance[i]) for i in chosen]
    result["nlpd_accuracy_units"] = base.mean([.5 * (math.log(2 * math.pi * variance[i]) + (mu[i] - truth[i]) ** 2 / variance[i]) for i in chosen])
    result["IS90_pp"] = 100 * base.mean([2 * Z[90] * s + 20 * max(abs(e) - Z[90] * s, 0) for e, s in zip(error, sd)])
    for level, z in Z.items():
        coverage = base.mean([float(abs(e) <= z * s) for e, s in zip(error, sd)])
        result[f"coverage{level}_fraction"] = coverage
        result[f"interval{level}_width_pp"] = 200 * z * base.mean(sd)
        result[f"coverage{level}_target_distance_fraction"] = abs(coverage - level / 100)
    return result


def random_reference(y, best_t):
    result = base.exact_random5(y); values = sorted(y); denom = math.comb(len(y), 5)
    gain = math.fsum(max(0, values[i - 1] - best_t) * math.comb(i - 1, 4) / denom for i in range(5, len(y) + 1))
    prob_break = math.fsum(math.comb(i - 1, 4) / denom for i in range(5, len(y) + 1) if values[i - 1] - best_t > TOL / 100)
    champions = sum(max(y) - v <= TOL / 100 for v in y)
    no_hit = math.comb(len(y) - champions, 5) / denom if len(y) - champions >= 5 else 0
    return {**result, "expected_gain5_pp": 100 * gain, "headroom_pp": 100 * max(0, max(y) - best_t),
            "expected_simple_regret_pp": 100 * (max(0, max(y) - best_t) - gain), "break_best_probability": prob_break,
            "champion_hit_probability": 1 - no_hit}


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, key)), "child_count": len(group), **{m: base.mean([r.get(m) for r in group]) for m in metrics}}
            for key, group in sorted(groups.items(), key=lambda x: str(x[0]))]


def hierarchy(units):
    subsets = aggregate(units, ("method", "subset_id", "fold", "sample_seed"), METRICS)
    require(len(subsets) == 60 and all(r["child_count"] == (1 if r["method"] in NEW else 3) for r in subsets), "Historical seeds must be scored then averaged")
    folds = aggregate(subsets, ("method", "fold"), METRICS)
    samples = aggregate(subsets, ("method", "sample_seed"), METRICS)
    means = {m: {k: base.mean([r[k] for r in folds if r["method"] == m]) for k in METRICS} for m in METHODS}
    return subsets, folds, samples, means


def paired(subsets):
    lookup = {(r["subset_id"], r["method"]): r for r in subsets}; rows = []
    for target, reference in COMPARISONS:
        for c in [r for r in subsets if r["method"] == target]:
            b = lookup[c["subset_id"], reference]
            row = {k: c[k] for k in ("subset_id", "fold", "sample_seed")}
            row.update(comparison=target + "_vs_" + reference, target=target, reference=reference)
            for key in METRICS:
                if c[key] is None or b[key] is None: value = None
                elif "coverage" in key and "target_distance" not in key or key in ("gain5_pp", "break_best_rate", "champion_top5_hit_fraction", "spearman"):
                    value = c[key] - b[key]
                else: value = b[key] - c[key]
                row[key] = value
            rows.append(row)
    folds = aggregate(rows, ("comparison", "fold"), METRICS)
    samples = aggregate(rows, ("comparison", "sample_seed"), METRICS)
    means = {name: {k: base.mean([r[k] for r in folds if r["comparison"] == name]) for k in METRICS} for name in sorted({r["comparison"] for r in rows})}
    return rows, folds, samples, means


def flags(subsets, means, pairs, pair_folds, pair_samples, optimization):
    def selection(reference):
        name = NEW[2] + "_vs_" + reference; p = pairs[name]
        f = {str(r["fold"]): r["regret5_pp"] for r in pair_folds if r["comparison"] == name}
        s = {str(r["sample_seed"]): r["regret5_pp"] for r in pair_samples if r["comparison"] == name}
        conditions = {"mean_regret_improvement_at_least_0_20_pp": p["regret5_pp"] + TOL >= .20,
                      "at_least_four_positive_folds": sum(v > TOL for v in f.values()) >= 4,
                      "both_samples_positive": all(v > TOL for v in s.values()), "gain_within_0_02_pp_tolerance": p["gain5_pp"] + TOL >= -.02}
        return {"pass": all(conditions.values()), "conditions": conditions, "delta_regret_pp": p["regret5_pp"], "delta_gain_pp": p["gain5_pp"],
                "delta_simple_regret_improvement_pp": p["simple_regret_pp"], "gain_direction": "positive" if p["gain5_pp"] > TOL else "negative" if p["gain5_pp"] < -TOL else "tie",
                "fold_regret_improvements_pp": f, "sample_regret_improvements_pp": s}
    main = selection(NEW[0]); architecture = selection(NEW[1]); retention = {}
    for reference in ("HEAD", "H_TREND"):
        p = pairs[NEW[2] + "_vs_" + reference]
        retention[reference] = {"pass": p["regret5_pp"] + TOL >= -.10 and p["gain5_pp"] + TOL >= -.02,
                                "regret_increase_pp": -p["regret5_pp"], "gain_change_pp": p["gain5_pp"]}
    p = pairs[NEW[2] + "_vs_" + NEW[0]]
    risk_conditions = {"nlpd_increase_within_0_10_nat": -p["nlpd_accuracy_units"] <= .10 + TOL,
                       "IS90_increase_within_1_pp": -p["IS90_pp"] <= 1 + TOL,
                       "all_Q_coverage90_drop_within_5_points": p["coverage90_fraction"] >= -.05 - TOL / 100,
                       "champion_coverage90_drop_within_5_points": p["champions__coverage90_fraction"] >= -.05 - TOL / 100}
    stable_counts = {m: sum(r["method"] == m and r["numeric_predictive_stable"] for r in optimization) for m in NEW}
    ready = stable_counts[NEW[0]] >= 8 and stable_counts[NEW[2]] >= 8
    retained = all(r["pass"] for r in retention.values())
    result = {"BLOCK_SCALE_SELECTION_SIGNAL": main["pass"], "HISTORICAL_SELECTION_RETAINED_VS_HEAD": retention["HEAD"]["pass"],
              "HISTORICAL_SELECTION_RETAINED_VS_H_TREND": retention["H_TREND"]["pass"], "HISTORICAL_SELECTION_RETAINED": retained,
              "ARCHITECTURE_INCREMENT_SIGNAL": architecture["pass"], "PROBABILITY_NO_LARGE_DEGRADATION_C_VS_A": all(risk_conditions.values()),
              "OPTIMIZATION_READY": ready, "BLOCK_DISTANCE_CANDIDATE_FOR_PROTOCOL_REVIEW": main["pass"] and retained and ready}
    return {"flags": result, "main_selection_detail": main, "architecture_increment_detail": architecture,
            "historical_retention_details": retention, "probability_risk_conditions": risk_conditions,
            "probability_risk_actual": {"nlpd_increase_nats": -p["nlpd_accuracy_units"], "IS90_increase_pp": -p["IS90_pp"],
                "coverage90_change_percentage_points": 100 * p["coverage90_fraction"], "champion_coverage90_change_percentage_points": 100 * p["champions__coverage90_fraction"]},
            "stability_counts_out_of10": stable_counts}


def run(root, labels_from=None):
    start = time.perf_counter(); seal, inputs, masks, budget = sealed_inputs(root)
    labels = release_labels(root, inputs, labels_from)
    snapshots = {}; mask_lookup = {(m["subset_id"], m["model_seed"]): m for m in masks}
    units, regions, mask_rows, candidates, optimization, random_rows = [], [], [], [], [], []
    for x in inputs:
        r, p = x["record"], x["prediction"]; sid = r["subset_id"]; method = r["method"]
        if sid not in snapshots:
            t = base.read_json(root / "training" / (sid + ".json"))
            require(len(t["ids"]) == len(t["y"]) == 50 and not set(t["ids"]) & set(p["candidate_ids"]), "Current T/Q invalid")
            snapshots[sid] = t
            ry = [labels[i]["validation_mean"] for i in p["candidate_ids"]]
            random_rows.append({"subset_id": sid, "fold": r["fold"], "sample_seed": r["sample_seed"], **random_reference(ry, max(t["y"]))})
        t = snapshots[sid]; ids = p["candidate_ids"]; mu = p["mean_raw"]; y = [labels[i]["validation_mean"] for i in ids]
        variance = None if x["model_variance"] is None else [v + labels[i]["validation_se"] ** 2 for v, i in zip(x["model_variance"], ids)]
        met, points = metric_math.evaluate(ids, mu, y, variance, max(t["y"])); meta = {k: r[k] for k in META}
        row = {**meta, **{k: met[k] for k in MAIN}, "ranking_ids": met["ranking_ids"], "selected_top5_ids": met["selected_top5_ids"],
               "champion_ids": met["champion_ids"], "best_T_raw": max(t["y"]), "best_Q_raw": max(y)}
        indices = {"all_Q": list(range(len(ids))), "true_top20": sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(.2 * len(ids))],
                   "true_bottom20": sorted(range(len(ids)), key=lambda i: (y[i], ids[i]))[:math.ceil(.2 * len(ids))],
                   "champions": [i for i in range(len(ids)) if max(y) - y[i] <= TOL / 100]}
        for region, ix in indices.items():
            rm = region_metrics(ids, mu, y, variance, ix)
            regions.append({**meta, "region": region, "candidate_ids": [ids[i] for i in ix], "candidate_count": len(ix), **rm})
            row.update({region + "__" + k: v for k, v in rm.items()})
        unit_masks = []
        for seed in range(3) if method in NEW else [r["model_seed"]]:
            mask = mask_lookup[sid, seed]; ix = [ids.index(i) for i in mask["candidate_ids"]]
            rm = region_metrics(ids, mu, y, variance, ix); unit_masks.append(rm)
            mask_rows.append({**meta, "head_mask_seed": seed, "candidate_ids": mask["candidate_ids"], "candidate_count": len(ix), **rm})
        rm = {k: base.mean([m[k] for m in unit_masks]) for k in REGION_METRICS}
        regions.append({**meta, "region": "HEAD_pred20", "mask_count": len(unit_masks), "aggregation": "mean of separately scored fixed HEAD masks", **rm})
        row.update({"HEAD_pred20__" + k: v for k, v in rm.items()}); units.append(row)
        for q in points: candidates.append({**meta, **q})
        if method in NEW:
            ok, reasons = stable(p)
            optimization.append({**meta, "numeric_predictive_stable": ok, "failure_window_counts": reasons,
                                 "top5_stable": p["stability"]["top5_stable"], "top5_order_stable": p["stability"].get("top5_order_stable"), "negative_mll": endpoint_nll(p)})
    subset, folds, samples, means = hierarchy(units)
    pair_rows, pair_folds, pair_samples, pair_means = paired(subset)
    summary = flags(subset, means, pair_means, pair_folds, pair_samples, optimization)
    olut = {(r["subset_id"], r["method"]): r for r in optimization}; ambiguity = []
    for c in [r for r in optimization if r["method"] == NEW[2]]:
        delta = c["negative_mll"] - olut[c["subset_id"], NEW[0]]["negative_mll"]
        ambiguity.append({"subset_id": c["subset_id"], "fold": c["fold"], "sample_seed": c["sample_seed"], "C_minus_A_negative_mll_nats_per_observation": delta,
                          "EXPANDED_MODEL_OPTIMIZATION_AMBIGUITY": delta > 1e-6})
    headroom_subsets = sorted({r["subset_id"] for r in subset if r["headroom_pp"] > TOL})
    headroom_summary = {m: {k: base.mean([r[k] for r in subset if r["method"] == m and r["subset_id"] in headroom_subsets]) for k in METRICS} for m in METHODS}
    summary.update(engineering_status="COMPLETE_VALID", new_fit_count=30, historical_reference_count=90, raw_scoring_units=120, formal_updates=54000,
                   fixed_head_mask_count=30, method_means=means, paired_means=pair_means,
                   expanded_model_optimization_ambiguity_count=sum(r["EXPANDED_MODEL_OPTIMIZATION_AMBIGUITY"] for r in ambiguity),
                   expanded_model_optimization_ambiguity=ambiguity, headroom_subset_count=len(headroom_subsets), headroom_subset_ids=headroom_subsets,
                   headroom_conditional_means=headroom_summary, headroom_condition="headroom_pp > 1e-9; descriptive subset means, all-subset main table retained",
                   no_new_neural_training=True, new_method_model_seed=None, execution_rng_seed=0, historical_seed_aggregation="score each seed, then average metrics within subset",
                   mask_aggregation="new: three separate fixed HEAD-mask metrics then average; historical: each seed its matching mask then average metrics",
                   coverage_target_distances_of_method_means={m: {r: {str(level): abs(means[m][r + f"__coverage{level}_fraction"] - level / 100) if means[m][r + f"__coverage{level}_fraction"] is not None else None for level in Z} for r in REGIONS} for m in METHODS},
                   break_best_raw_unit_counts={m: {"count": sum(r["break_best_rate"] for r in units if r["method"] == m), "raw_units": 10 if m in NEW else 30} for m in METHODS},
                   A_repeated_development=True, next_stage_authorized=False, probability_flag_is_calibration_guarantee=False,
                   NLPD_units="nat/candidate, natural log and original accuracy proportion", error_width_IS90_units="percentage points", coverage_units="fraction",
                   budget=budget, scorer_sha256=base.sha256(__file__), predictions_manifest_sha256=base.sha256(root / "predictions_manifest.json"),
                   labels_A_sha256=base.sha256(root / "scoring/labels_A.json"), statistical_seconds=time.perf_counter() - start)
    out = root / "scoring"
    tables = {"unit_metrics": units, "subset_metrics": subset, "fold_metrics": folds, "sample_seed_metrics": samples,
              "candidate_metrics": candidates, "region_unit_metrics": regions, "head_mask_metrics": mask_rows,
              "region_subset_metrics": aggregate(regions, ("method", "region", "subset_id", "fold", "sample_seed"), REGION_METRICS),
              "paired_subset": pair_rows, "paired_fold": pair_folds, "paired_sample_seed": pair_samples,
              "optimization": optimization, "expanded_model_optimization": ambiguity, "random5_reference": random_rows,
              "headroom_subset_metrics": [r for r in subset if r["subset_id"] in headroom_subsets]}
    region_folds = aggregate(tables["region_subset_metrics"], ("method", "region", "fold"), REGION_METRICS)
    tables["region_fold_metrics"] = region_folds
    tables["region_sample_seed_metrics"] = aggregate(tables["region_subset_metrics"], ("method", "region", "sample_seed"), REGION_METRICS)
    for name, rows in tables.items(): base.write_csv(out / (name + ".csv"), rows)
    base.write_json(out / "summary.json", summary)
    return summary


def synthetic_checks():
    import audit_block as independent
    start = time.perf_counter(); checks = []
    ids = list("abcdefg"); y = [.1, .2, .3, .4, .5, .7, .7]; mu = [.9, .8, .7, .6, .5, .4, .3]
    m, _ = metric_math.evaluate(ids, mu, y, [.01] * 7, .55)
    require(m["selected_top5_ids"] == list("abcde") and math.isclose(m["regret5_pp"], 20) and m["gain5_pp"] == 0 and math.isclose(m["headroom_pp"], 15), "Selection/control formulas")
    require(m["champion_ids"] == ["f", "g"] and m["champion_prediction_rank"] == 6, "All champion ties")
    rr = random_reference(y, .55); choices = list(itertools.combinations(range(7), 5))
    close(rr["expected_gain5_pp"], base.mean([100 * max(0, max(y[i] for i in c) - .55) for c in choices]), "Random exact gain enumeration")
    close(rr["expected_regret5_pp"], base.mean([100 * (max(y) - max(y[i] for i in c)) for c in choices]), "Random exact regret enumeration")
    checks.append("selection_gain_headroom_simple_regret_champion_ties_exact_random5")
    rm = region_metrics(ids, mu, y, [.01 + .0025] * 7, [0, 6]); require(rm["nlpd_accuracy_units"] is not None, "Variance SE once")
    h = region_metrics(ids, mu, y, None, [0, 6]); require(h["IS90_pp"] is None, "HEAD probability NA")
    a = region_metrics(ids, mu, y, [.01] * 7, [0, 1]); b = region_metrics(ids, mu, y, [.01] * 7, [5, 6])
    require(not math.isclose((a["rmse_pp"] + b["rmse_pp"]) / 2, region_metrics(ids, mu, y, [.01] * 7, [0, 1, 5, 6])["rmse_pp"]), "Mask mean RMSE must not pool")
    checks.append("region_variance_HEAD_NA_and_nonpooled_mask_RMSE")
    for variance in (None, [.01 + .0025] * 7):
        for ix in ([0, 1, 6], [5, 6], [0, 2, 3, 4, 6]):
            actual = region_metrics(ids, mu, y, variance, ix); other = independent.region(mu, y, variance, ix)
            for key in actual:
                if actual[key] is None: require(other[key] is None, "Independent HEAD region must be NA")
                else: close(actual[key], other[key], "Independent region formula: " + key)
    independent_selected = independent.selection(ids, mu, y, .55)
    for key in SELECTION: close(m[key], independent_selected[key], "Independent selection formula: " + key)
    history_y = [0, .1, .2, .3, .4, 1.]
    history_mu = [[1., .9, .8, .7, .6, 0.], [1., .9, .8, .7, .6, 0.], [0., .1, .2, .3, .4, 10.]]
    historical_regrets = [independent.selection(list("abcdef"), v, history_y, .5)["regret5_pp"] for v in history_mu]
    ensemble_regret = independent.selection(list("abcdef"), [base.mean(v) for v in zip(*history_mu)], history_y, .5)["regret5_pp"]
    require(math.isclose(base.mean(historical_regrets), 40.) and ensemble_regret == 0, "History scores must not be ensemble scores")
    checks.append("independent_probability_region_selection_and_historical_nonensemble_math")
    rows = []
    for f in range(5):
        for s in SEEDS:
            for method in METHODS:
                for seed in ([None] if method in NEW else range(3)):
                    r = {"method": method, "subset_id": f"{s}_{f}", "fold": f, "sample_seed": s, "model_seed": seed, **{k: 1. for k in METRICS}}
                    r["regret5_pp"] = (1. + seed if method in HISTORY else 1.) if method != NEW[2] else .8
                    r["gain5_pp"] = .2 if method != NEW[2] else .18
                    rows.append(r)
    subs, fs, ss, means = hierarchy(rows); ps, pf, pseed, pm = paired(subs)
    opt = [{"method": m, "numeric_predictive_stable": k < 8} for m in NEW for k in range(10)]
    result = flags(subs, means, pm, pf, pseed, opt)
    require(len(subs) == 60 and len(fs) == 30 and len(ss) == 12 and len(ps) == 120, "Hierarchy and fixed pair counts")
    require(all(r["regret5_pp"] == 2. for r in subs if r["method"] in HISTORY), "Three nonidentical historical metrics average before folds")
    require(all(result["flags"].values()), "Inclusive threshold tolerance and all independent flags")
    pm[NEW[2] + "_vs_" + NEW[0]]["regret5_pp"] = .20 - 2 * TOL
    require(not flags(subs, means, pm, pf, pseed, opt)["flags"]["BLOCK_SCALE_SELECTION_SIGNAL"], "Outside border tolerance must fail")
    checks.append("historical_metric_average_hierarchy_all_flags_and_boundary_tolerance")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); base.write_json(root / "predictions_manifest.json", {"sealed": True, "execution_status": "PARTIAL"})
        try: sealed_inputs(root)
        except ValueError as e: require("Complete valid seal" in str(e), "Partial must reject before labels")
        else: raise AssertionError("Partial accepted")
    checks.append("partial_seal_rejects_before_label_read")
    return {"pass": True, "checks": checks, "statistical_seconds": time.perf_counter() - start, "model_seconds": 0, "optimizer_updates": 0, "real_Q_labels_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path); parser.add_argument("--labels-from", type=Path); parser.add_argument("--synthetic-checks", action="store_true")
    args = parser.parse_args()
    if args.synthetic_checks: result = synthetic_checks()
    else:
        require(args.root is not None, "--root is required")
        result = run(args.root.resolve(), args.labels_from)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
