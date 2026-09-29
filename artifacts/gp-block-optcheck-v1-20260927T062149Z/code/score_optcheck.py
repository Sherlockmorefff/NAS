#!/usr/bin/env python3
"""Nested bounded GP optimization experiment: sealed-array statistics only."""
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

NEW=("A_REFIT","B_REFIT","C_REFIT")
OLD=("A_OLD","B_OLD","C_OLD")
HISTORY=("HEAD","H_CONST","H_TREND")
METHODS=NEW+OLD+HISTORY
PARENT={"A_OLD":"RAW16_ISO","B_OLD":"HP4_ISO","C_OLD":"RAW16_BLOCK2","HEAD":"HEAD","H_CONST":"GP_CONST","H_TREND":"GP_TREND"}
REGIONS=("all_Q","HEAD_pred20","true_top20","true_bottom20","champions")
SEEDS=(20260922,20260923)
META=("unit_id","subset_id","fold","sample_seed","model_seed","method")
TOL=1e-9
COMPARISONS=(("C_REFIT","B_REFIT"),("C_REFIT","A_REFIT"),("A_REFIT","A_OLD"),("B_REFIT","B_OLD"),("C_REFIT","C_OLD"),("C_REFIT","HEAD"),("C_REFIT","H_CONST"),("C_REFIT","H_TREND"),("C_REFIT","RANDOM5"),("C_REFIT","A_OLD"))


Z = metric_math.Z


SELECTION = ("regret5_pp", "gain5_pp", "headroom_pp", "simple_regret_pp", "break_best_rate", "top1_regret_pp",
             "champion_prediction_rank", "champion_top5_hit_fraction")


REGION_METRICS = ("mae_pp", "rmse_pp", "bias_pp", "nlpd_accuracy_units", "IS90_pp") + tuple(
    f"{name}{level}{suffix}" for level in Z for name, suffix in (("coverage", "_fraction"), ("interval", "_width_pp"))) + tuple(
    f"coverage{level}_target_distance_fraction" for level in Z)


MAIN = SELECTION + ("mae_pp", "rmse_pp", "bias_pp", "spearman", "nlpd_accuracy_units", "IS90_pp") + tuple(
    f"{name}{level}{suffix}" for level in Z for name, suffix in (("coverage", "_fraction"), ("interval", "_width_pp")))


METRICS = MAIN + tuple(r + "__" + m for r in REGIONS for m in REGION_METRICS)


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


def budget_check(budget, protocol):
    limit = protocol["budget"].get("model_seconds", protocol["budget"].get("limit"))
    require(limit == budget["limit"] == 600, "New experiment must retain its own 600-second limit")
    seconds = budget.get("seconds", budget.get("model_seconds"))
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and 0 <= seconds <= 600, "Model budget invalid/exceeded")
    def calls(names, maximum):
        values = [budget[k] for k in names if k in budget]
        require(values and all(isinstance(v, int) and 0 <= v <= maximum for v in values), "Missing/invalid request accounting: " + "/".join(names))
        require(len(set(values)) == 1, "Request-count aliases differ")
        return values[0]
    return {"model_seconds": seconds,
            "optimization_objective_calls": calls(("optimization_nll_gradient_requests", "optimization_objective_calls", "optimization_requests", "formal_objective_calls"), 18000),
            "synthetic_objective_calls": calls(("synthetic_nll_gradient_requests", "synthetic_objective_calls", "synthetic_objective_requests", "synthetic_requests"), 300)}


def projected_gradient(coordinates, gradient, bounds):
    require(len(coordinates) == len(gradient) == len(bounds), "Gradient/coordinate/bound shape differs")
    result = []
    for x, g, (low, high) in zip(coordinates, gradient, bounds):
        require(math.isfinite(x) and math.isfinite(g), "Nonfinite independent gradient")
        if low is not None:
            tol = 1e-12 * max(1, abs(low)); require(x >= low - tol, "Selected coordinate below lower bound")
            if abs(x - low) <= tol and g > 0: g = 0.
        if high is not None:
            tol = 1e-12 * max(1, abs(high)); require(x <= high + tol, "Selected coordinate above upper bound")
            if abs(x - high) <= tol and g < 0: g = 0.
        result.append(g)
    return result


def prescribed_bounds(method):
    common = [[None, None], [math.log(1e-8), None], [math.log(1e-6), None]]
    return common + ([[0., math.log1p(1e8)], [math.log(1e-4), None]] if method == "C_REFIT" else [[math.log(1e-4), None]])


def optimization_decision(d, method):
    x, gradient, bounds, loss = d["coordinates"], d["independent_gradient"], d["bounds"], d["selected_nll"]
    require(len(x) == (5 if method == "C_REFIT" else 4) and bounds == prescribed_bounds(method), "Wrong optimization coordinates or bounds")
    require(math.isfinite(loss), "Invalid selected NLL")
    pg = projected_gradient(x, gradient, bounds); norm = max(abs(g) for g in pg)
    known = d["known_feasible_nlls"]
    required = {"parent_C", "A_REFIT", "B_REFIT"} if method == "C_REFIT" else {"parent"}
    require({v["source"] for v in known} == required and len(known) == len(required), "Known feasible pool incomplete")
    require(all(math.isfinite(v["nll"]) for v in known), "Nonfinite known feasible objective")
    differences = {v["source"]: loss - v["nll"] for v in known}
    expected_probes = {}
    for index in range(len(x)):
        for direction in (-1, 1):
            probe = list(x); probe[index] += direction * .001
            low, high = bounds[index]
            if low is not None: probe[index] = max(low, probe[index])
            if high is not None: probe[index] = min(high, probe[index])
            if probe[index] != x[index]: expected_probes[index, direction] = probe
    probes = d["probes"]; active = [p for p in probes if not p.get("skipped_no_displacement", False)]
    require(len(active) == len(expected_probes) and {(p["coordinate"], p["direction"]) for p in active} == set(expected_probes), "Prescribed local probes incomplete/duplicated")
    drops = []
    for p in active:
        require(math.isfinite(p["nll"]), "Nonfinite local probe")
        coordinates = expected_probes[p["coordinate"], p["direction"]]
        require(len(p["probe_coordinates"]) == len(x), "Probe coordinate count")
        for actual, expected in zip(p["probe_coordinates"], coordinates): close(actual, expected, "Probe differs from fixed feasible +-0.001")
        drops.append(loss - p["nll"])
    return {"KNOWN_FEASIBLE_DOMINANCE_PASS": all(v <= 1e-6 for v in differences.values()),
            "LOCAL_STATIONARITY_PASS": norm <= 1e-5, "LOCAL_PROBE_PASS": max(drops, default=0) <= 1e-6,
            "joint_local_pass": norm <= 1e-5 and max(drops, default=0) <= 1e-6,
            "projected_gradient": pg, "projected_gradient_inf": norm, "independent_gradient": gradient,
            "known_feasible_nll_differences": differences, "max_probe_nll_improvement": max(drops, default=0), "probe_count": len(active),
            "selected_nll": loss, "coordinates": x, "architecture_disabled": x[3] == 0 if method == "C_REFIT" else None,
            "selected_start": d.get("selected_start_index", d.get("selected_start")), "selected_iteration": d.get("accepted_iteration", d.get("selected_iteration")),
            "selection_numerical_minimum_gap": d.get("selected_minus_numerical_minimum", d.get("selection_numerical_minimum_gap"))}


def variance_values(p, method):
    if method == "HEAD": return None
    latent, scale, extra = p["latent_variance_standardized"], p["y_scale"], p["extra_noise_variance_standardized"]
    require(len(latent) == len(p["candidate_ids"]) and scale >= 1e-8 and extra >= 1e-6 - 1e-14, "Invalid variance scale/shape/noise")
    require(all(math.isfinite(v) and v >= -1e-8 for v in latent), "Numerically invalid latent variance")
    corrected = [scale * scale * (max(v, 0) + extra) for v in latent]
    require(all(math.isfinite(v) and v > 0 for v in corrected), "Nonpositive predictive variance")
    if "model_variance_raw" in p:
        require(len(p["model_variance_raw"]) == len(latent), "Model variance shape")
        for given, raw, fixed in zip(p["model_variance_raw"], [scale * scale * (v + extra) for v in latent], corrected):
            require(math.isclose(given, raw, rel_tol=1e-10, abs_tol=1e-12) or math.isclose(given, fixed, rel_tol=1e-10, abs_tol=1e-12), "Model variance contains unprescribed SE/tau/other noise")
    require(p.get("tau_squared_raw", 0) in (0, None), "No tau is authorized")
    return corrected


def sealed_inputs(root):
    seal = base.read_json(root / "seal.json"); manifest = base.read_json(root / "predictions_manifest.json")
    for record in (seal, manifest):
        require(record.get("sealed") is True and record.get("execution_status") == "COMPLETE_VALID", "Complete valid seals required before Q labels")
    checked(root, "predictions_manifest.json", seal["predictions_manifest_sha256"])
    for names, expected in ((("new_endpoint_count",), 30), (("parent_reference_count",), 30), (("historical_reference_count",), 90),
                            (("raw_unit_count",), 150), (("mask_count",), 30), (("completed_trajectories", "trajectory_count"), 70),
                            (("new_candidate_rows",), 1074), (("all_candidate_rows", "scoring_candidate_rows"), 5370),
                            (("independent_restoration_count",), 30), (("local_diagnostic_count",), 30)):
        count(manifest, names, expected)
    for name in ("protocol", "source_manifest", "code_manifest", "worklist", "masks_manifest", "frozen_manifest", "data_manifest", "case_audit_manifest"):
        checked(root, name + ".json", manifest[name + "_sha256"])
    for name in ("source_manifest", "code_manifest", "frozen_manifest", "masks_manifest", "data_manifest"):
        hash_files(root, base.read_json(root / (name + ".json")))
    hash_files(root, manifest)
    budget = base.read_json(root / "budget.json"); budget_check(budget, base.read_json(root / "protocol.json"))
    refs = base.read_json(root / "historical_refs.json")["records"]
    require(len(refs) == 120 and len({r["unit_id"] for r in refs}) == 120, "120 old references required")
    reflut = {r["unit_id"]: r for r in refs}
    records = manifest["records"]; require(len(records) == 150 and len({r["unit_id"] for r in records}) == 150, "150 unique scoring units required")
    queries, inputs, seen = {}, [], set()
    for r in records:
        require(manifest["files"].get(r["path"]) == r["sha256"], "Prediction not bound to uniform seal")
        p = base.read_json(checked(root, r["path"], r["sha256"])); base.forbid_query_truth(p)
        method = r["method"]; require(method in METHODS, "Unprescribed method")
        key = (r["fold"], r["sample_seed"], r["model_seed"], method); require(key not in seen, "Duplicate scoring identity"); seen.add(key)
        for k in ("subset_id", "fold", "sample_seed", "model_seed"): require(r[k] == p[k], "Prediction/manifest identity differs")
        require(r["fold"] in range(5) and r["sample_seed"] in SEEDS, "Wrong fold/sample")
        ids = p["candidate_ids"]; require(len(ids) == len(set(ids)) == (38, 36, 36, 34, 35)[r["fold"]], "Wrong query size")
        require(len(p["mean_raw"]) == len(p["architecture_fingerprints"]) == len(ids) and all(math.isfinite(v) for v in p["mean_raw"]), "Invalid prediction shape/mean")
        q = (ids, p["architecture_fingerprints"])
        require(r["fold"] not in queries or queries[r["fold"]] == q, "Query ordering/architecture changed"); queries[r["fold"]] = q
        decision, d = None, None
        if method in NEW:
            require(p["method"] == method and r["model_seed"] is None and p.get("rng_seed", p.get("execution_rng_seed")) == 0, "New methods have no model-seed replicates")
            state = base.read_json(checked(root, p["state_path"], p["state_file_sha256"]))
            d = base.read_json(checked(root, p["diagnostics_path"], p["diagnostics_sha256"]))
            require(d["unit_id"] == r["unit_id"] and d["method"] == method and d["subset_id"] == r["subset_id"], "Diagnostic identity mismatch")
            require(d["coordinates"] == p["selected"]["coordinates"] == state["coordinates"] and state["method"] == method, "Selected state/diagnostic coordinate mismatch")
            close(d["selected_nll"], p["selected"]["nll"], "Independent objective differs from selected prediction", atol=1e-10, rtol=1e-8)
            decision = optimization_decision(d, method)
            cov, rawcov = p["latent_covariance_standardized"], p["latent_covariance_raw"]
            require(len(cov) == len(rawcov) == len(ids) and all(len(v) == len(ids) for v in cov + rawcov), "Complete covariance missing")
            require(p["latent_covariance_symmetry_max_abs"] <= 1e-10 and p["latent_covariance_min_eigenvalue"] >= -1e-8, "Covariance check failed")
            for i in range(len(ids)):
                close(p["mean_raw"][i], p["y_center"] + p["y_scale"] * p["mean_standardized"][i], "Mean inverse transform")
                close(cov[i][i], p["latent_variance_standardized"][i], "Latent diagonal differs")
                for j in range(len(ids)):
                    close(rawcov[i][j], p["y_scale"] ** 2 * cov[i][j], "Covariance inverse transform")
                    require(abs(cov[i][j] - cov[j][i]) <= 1e-10, "Covariance not symmetric")
        else:
            old = reflut[r["unit_id"]]
            require(p["method"] == PARENT[method] and all(r[k] == old[k] for k in ("path", "sha256", "method", "subset_id", "fold", "sample_seed", "model_seed")), "Old reference bytes/identity changed")
            require(r["model_seed"] is None if method in OLD else r["model_seed"] in range(3), "Old deterministic/history seed distinction")
        inputs.append({"record": r, "prediction": p, "variance": variance_values(p, method), "optimization": decision, "diagnostic": d})
    expected = {(f, s, None, m) for f in range(5) for s in SEEDS for m in NEW + OLD} | {(f, s, k, m) for f in range(5) for s in SEEDS for k in range(3) for m in HISTORY}
    require(seen == expected and sum(len(x["prediction"]["candidate_ids"]) for x in inputs) == 5370, "Full prescribed matrix required")
    unit_lut = {(x["record"]["subset_id"], x["record"]["method"]): x for x in inputs if x["record"]["method"] not in HISTORY}
    for x in inputs:
        method, sid = x["record"]["method"], x["record"]["subset_id"]
        if method not in NEW: continue
        for known in x["diagnostic"]["known_feasible_nlls"]:
            ref = {"parent": method.replace("REFIT", "OLD"), "parent_C": "C_OLD"}.get(known["source"], known["source"])
            reference = unit_lut[sid, ref]
            value = reference["diagnostic"]["selected_nll"] if ref in NEW else reference["prediction"].get("endpoint_nll_after_update", reference["prediction"].get("negative_mll"))
            close(known["nll"], value, "Known-feasible objective source differs", atol=1e-10, rtol=1e-8)
    masks = base.read_json(root / "masks_manifest.json")["records"]
    require(len(masks) == len({(m["subset_id"], m["model_seed"]) for m in masks}) == 30, "30 distinct frozen HEAD masks required")
    heads = {(x["record"]["subset_id"], x["record"]["model_seed"]): x for x in inputs if x["record"]["method"] == "HEAD"}
    for m in masks:
        x = heads[m["subset_id"], m["model_seed"]]; p, r = x["prediction"], x["record"]
        require(m["query_ids"] == p["candidate_ids"] and m["source_prediction_sha256"] == r["sha256"], "Frozen mask source differs")
        order = sorted(range(len(p["candidate_ids"])), key=lambda i: (-p["mean_raw"][i], p["candidate_ids"][i]))
        require(m["candidate_ids"] == [p["candidate_ids"][i] for i in order[:math.ceil(.2 * len(order))]], "Mask not individual frozen HEAD top20")
        saved = base.read_json(checked(root, m["path"], m["sha256"]))
        require(all(saved[k] == m[k] for k in ("subset_id", "model_seed", "candidate_ids", "query_ids", "source_prediction_sha256")), "Mask file differs")
    case_summary = base.read_json(checked(root, "case_audit/summary.json", manifest["files"]["case_audit/summary.json"]))
    require(case_summary["confirmed_data_inconsistency"] is False, "Confirmed source data inconsistency blocks scoring")
    require(case_summary["case_manifest_sha256"] == base.sha256(root / "case_audit_manifest.json"), "Case audit identity differs")
    return manifest, inputs, masks, budget, case_summary


def release_labels(root, inputs, labels_from=None):
    source = base.read_json(root / "source_manifest.json")
    expected = source.get("legal_A_scoring_truth_sha256", source.get("expected_labels_sha256"))
    require(expected, "Missing pinned A-label hash")
    out = root / "scoring"; out.mkdir(exist_ok=True); target = out / "labels_A.json"
    path = Path(labels_from or source.get("legal_A_scoring_truth_path", source.get("source_label_path", target)))
    require(base.sha256(path) == expected, "Scoring label identity differs")
    if path.resolve() != target.resolve(): shutil.copyfile(path, target)
    labels = base.checked_labels(target, {i for x in inputs for i in x["prediction"]["candidate_ids"]})
    base.write_json(out / "release.json", {"released_at_utc": datetime.now(timezone.utc).isoformat(), "predictions_manifest_sha256": base.sha256(root / "predictions_manifest.json"),
        "seal_sha256": base.sha256(root / "seal.json"), "labels_sha256": base.sha256(target), "new_selected_models": 30, "old_references": 120,
        "prescribed_trajectories": 70, "historical_case_audit_read_A_labels_before_formal": True, "further_model_training_authorized": False})
    return labels


def hierarchy(units):
    subsets = aggregate(units, ("method", "subset_id", "fold", "sample_seed"), METRICS)
    require(len(subsets) == 90 and all(r["child_count"] == (3 if r["method"] in HISTORY else 1) for r in subsets), "Must score history seeds before averaging")
    folds = aggregate(subsets, ("method", "fold"), METRICS); samples = aggregate(subsets, ("method", "sample_seed"), METRICS)
    means = {m: {k: base.mean([r[k] for r in folds if r["method"] == m]) for k in METRICS} for m in METHODS}
    return subsets, folds, samples, means


def paired(subsets, random_rows):
    lookup = {(r["subset_id"], r["method"]): r for r in subsets}
    for r in random_rows:
        lookup[r["subset_id"], "RANDOM5"] = {**{k: None for k in METRICS}, "regret5_pp": r["expected_regret5_pp"], "gain5_pp": r["expected_gain5_pp"],
            "simple_regret_pp": r["expected_simple_regret_pp"], "headroom_pp": r["headroom_pp"], "break_best_rate": r["break_best_probability"], "champion_top5_hit_fraction": r["champion_hit_probability"]}
    rows = []
    for target, reference in COMPARISONS:
        for c in [r for r in subsets if r["method"] == target]:
            b = lookup[c["subset_id"], reference]; row = {k: c[k] for k in ("subset_id", "fold", "sample_seed")}
            row.update(comparison=target + "_vs_" + reference, target=target, reference=reference)
            for k in METRICS:
                if c[k] is None or b[k] is None: value = None
                elif ("coverage" in k and "target_distance" not in k) or k in ("gain5_pp", "break_best_rate", "champion_top5_hit_fraction", "spearman"): value = c[k] - b[k]
                else: value = b[k] - c[k]
                row[k] = value
            rows.append(row)
    folds = aggregate(rows, ("comparison", "fold"), METRICS); samples = aggregate(rows, ("comparison", "sample_seed"), METRICS)
    means = {name: {k: base.mean([r[k] for r in folds if r["comparison"] == name]) for k in METRICS} for name in sorted({r["comparison"] for r in rows})}
    return rows, folds, samples, means


def scientific_flags(means, pair_means, pair_folds, pair_samples, optimization, source_consistent):
    def selection(reference):
        name = "C_REFIT_vs_" + reference; p = pair_means[name]
        f = {str(r["fold"]): r["regret5_pp"] for r in pair_folds if r["comparison"] == name}
        s = {str(r["sample_seed"]): r["regret5_pp"] for r in pair_samples if r["comparison"] == name}
        conditions = {"mean_regret_at_least_0_20_pp": p["regret5_pp"] + TOL >= .2, "four_positive_folds": sum(v > TOL for v in f.values()) >= 4,
                      "both_samples_positive": all(v > TOL for v in s.values()), "gain_within_0_02_pp": p["gain5_pp"] + TOL >= -.02}
        return {"pass": all(conditions.values()), "conditions": conditions, "delta_regret_pp": p["regret5_pp"], "delta_gain_pp": p["gain5_pp"],
                "fold_regret_improvements_pp": f, "sample_regret_improvements_pp": s, "simple_regret_improvement_pp": p["simple_regret_pp"],
                "gain_direction": "positive" if p["gain5_pp"] > TOL else "negative" if p["gain5_pp"] < -TOL else "tie"}
    architecture, block = selection("B_REFIT"), selection("A_REFIT")
    retention = {ref: {"regret_increase_pp": means["C_REFIT"]["regret5_pp"] - means[ref]["regret5_pp"], "gain_change_pp": means["C_REFIT"]["gain5_pp"] - means[ref]["gain5_pp"]} for ref in ("HEAD", "H_TREND")}
    for r in retention.values(): r["pass"] = r["regret_increase_pp"] <= .1 + TOL and r["gain_change_pp"] >= -.02 - TOL
    risk = {}
    for ref in ("A_REFIT", "A_OLD"):
        c, b = means["C_REFIT"], means[ref]
        actual = {"nlpd_increase_nats": c["nlpd_accuracy_units"] - b["nlpd_accuracy_units"], "IS90_increase_pp": c["IS90_pp"] - b["IS90_pp"],
                  "coverage90_change_percentage_points": 100 * (c["coverage90_fraction"] - b["coverage90_fraction"]),
                  "champion_coverage90_change_percentage_points": 100 * (c["champions__coverage90_fraction"] - b["champions__coverage90_fraction"])}
        conditions = {"nlpd": actual["nlpd_increase_nats"] <= .1 + TOL, "IS90": actual["IS90_increase_pp"] <= 1 + TOL,
                      "all_coverage": actual["coverage90_change_percentage_points"] >= -5 - TOL, "champion_coverage": actual["champion_coverage90_change_percentage_points"] >= -5 - TOL}
        risk[ref] = {"pass": all(conditions.values()), "conditions": conditions, **actual}
    counts = {m: {k: sum(bool(r[k]) for r in optimization if r["method"] == m) for k in ("KNOWN_FEASIBLE_DOMINANCE_PASS", "LOCAL_STATIONARITY_PASS", "LOCAL_PROBE_PASS", "joint_local_pass")} for m in NEW}
    known = all(r["KNOWN_FEASIBLE_DOMINANCE_PASS"] for r in optimization)
    adequate = known and all(counts[m]["joint_local_pass"] >= 8 for m in NEW)
    retained, probability = all(r["pass"] for r in retention.values()), all(r["pass"] for r in risk.values())
    flags = {"KNOWN_FEASIBLE_DOMINANCE_PASS": known, "OPTIMIZATION_ADEQUATE": adequate, "ARCHITECTURE_INCREMENT_SIGNAL": architecture["pass"],
             "BLOCK_SCALE_SIGNAL": block["pass"], "HISTORICAL_SELECTION_RETAINED_VS_HEAD": retention["HEAD"]["pass"], "HISTORICAL_SELECTION_RETAINED_VS_H_TREND": retention["H_TREND"]["pass"],
             "HISTORICAL_SELECTION_RETAINED": retained, "PROBABILITY_NO_LARGE_DEGRADATION_VS_A_REFIT": risk["A_REFIT"]["pass"],
             "PROBABILITY_NO_LARGE_DEGRADATION_VS_A_OLD": risk["A_OLD"]["pass"], "PROBABILITY_NO_LARGE_DEGRADATION": probability,
             "DEVELOPMENT_CANDIDATE_FOR_CONFIRMATION_DESIGN": bool(source_consistent and adequate and architecture["pass"] and block["pass"] and retained and probability)}
    return {"flags": flags, "architecture_increment_detail": architecture, "block_scale_detail": block, "historical_retention_details": retention,
            "probability_risk_details": risk, "optimization_counts_out_of10": counts, "source_consistent": source_consistent,
            "C_exact_boundary_count": sum(r["method"] == "C_REFIT" and r["architecture_disabled"] for r in optimization)}


def run(root, labels_from=None):
    start = time.perf_counter(); manifest, inputs, masks, budget, case_summary = sealed_inputs(root)
    labels = release_labels(root, inputs, labels_from)
    snapshots, mask_lut = {}, {(m["subset_id"], m["model_seed"]): m for m in masks}
    units, regions, mask_rows, candidates, optimization, random_rows = [], [], [], [], [], []
    case_ids = base.read_json(root / "case_audit_manifest.json")["candidate_ids"]
    require(case_ids == case_summary["candidate_ids"] and len(set(case_ids)) == len(case_ids), "Case IDs changed after frozen rule")
    for x in inputs:
        r, p = x["record"], x["prediction"]; method, sid = r["method"], r["subset_id"]
        ids, mu = p["candidate_ids"], p["mean_raw"]; y = [labels[i]["validation_mean"] for i in ids]
        if sid not in snapshots:
            t = base.read_json(root / "training" / (sid + ".json")); require(len(t["ids"]) == len(t["y"]) == 50 and not set(t["ids"]) & set(ids), "Selected T identity invalid")
            snapshots[sid] = t; random_rows.append({"subset_id": sid, "fold": r["fold"], "sample_seed": r["sample_seed"], **random_reference(y, max(t["y"]))})
        incumbent = max(snapshots[sid]["y"])
        obs = None if x["variance"] is None else [v + labels[i]["validation_se"] ** 2 for v, i in zip(x["variance"], ids)]
        met, points = metric_math.evaluate(ids, mu, y, obs, incumbent); meta = {k: r[k] for k in META}
        row = {**meta, **{k: met[k] for k in MAIN}, "ranking_ids": met["ranking_ids"], "selected_top5_ids": met["selected_top5_ids"],
               "selected_top5_truth_raw": met["selected_top5_truth_raw"], "champion_ids": met["champion_ids"], "best_T_raw": incumbent, "best_Q_raw": max(y)}
        ix = {"all_Q": list(range(len(ids))), "true_top20": sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(.2 * len(ids))],
              "true_bottom20": sorted(range(len(ids)), key=lambda i: (y[i], ids[i]))[:math.ceil(.2 * len(ids))], "champions": [i for i in range(len(ids)) if max(y) - y[i] <= TOL / 100]}
        for region_name, indexes in ix.items():
            values = region_metrics(ids, mu, y, obs, indexes)
            regions.append({**meta, "region": region_name, "candidate_ids": [ids[i] for i in indexes], "candidate_count": len(indexes), **values})
            row.update({region_name + "__" + k: v for k, v in values.items()})
        local_masks = []
        for seed in [r["model_seed"]] if method in HISTORY else range(3):
            mask = mask_lut[sid, seed]; indexes = [ids.index(i) for i in mask["candidate_ids"]]
            values = region_metrics(ids, mu, y, obs, indexes); local_masks.append(values)
            mask_rows.append({**meta, "head_mask_seed": seed, "candidate_ids": mask["candidate_ids"], **values})
        values = {k: base.mean([v[k] for v in local_masks]) for k in REGION_METRICS}
        regions.append({**meta, "region": "HEAD_pred20", "mask_count": len(local_masks), **values}); row.update({"HEAD_pred20__" + k: v for k, v in values.items()}); units.append(row)
        weight = 1 / (10 * (3 if method in HISTORY else 1) * len(ids))
        for i, q in enumerate(points):
            q.update(meta); q["measurement_variance_raw"] = labels[ids[i]]["validation_se"] ** 2
            q["latent_variance_raw"] = None if obs is None else p["y_scale"] ** 2 * p["latent_variance_standardized"][i]
            q["extra_noise_variance_raw"] = None if obs is None else p["y_scale"] ** 2 * p["extra_noise_variance_standardized"]
            q["model_variance_raw_for_scoring"] = None if obs is None else x["variance"][i]
            q["raw_unit_hierarchical_candidate_weight"] = weight; q["global_NLPD_contribution"] = None if obs is None else weight * q["nlpd_accuracy_units"]
            for level, z in Z.items():
                q[f"interval{level}_lower_raw"] = None if obs is None else mu[i] - z * math.sqrt(obs[i])
                q[f"interval{level}_upper_raw"] = None if obs is None else mu[i] + z * math.sqrt(obs[i])
            candidates.append(q)
        if x["optimization"] is not None: optimization.append({**meta, **x["optimization"]})
    subset, folds, samples, means = hierarchy(units)
    pairs, pair_folds, pair_samples, pair_means = paired(subset, random_rows)
    summary = scientific_flags(means, pair_means, pair_folds, pair_samples, optimization, not case_summary["confirmed_data_inconsistency"])
    rooms = sorted({r["subset_id"] for r in subset if r["headroom_pp"] > TOL})
    headroom = {m: {k: base.mean([r[k] for r in subset if r["method"] == m and r["subset_id"] in rooms]) for k in METRICS} for m in METHODS}
    case_rows = [r for r in candidates if r["candidate_id"] in case_ids]; contributions = []
    by_case = {(m, cid): base.mean([r["nlpd_accuracy_units"] for r in case_rows if r["method"] == m and r["candidate_id"] == cid]) for m in METHODS for cid in case_ids}
    weighted = {(m, cid): math.fsum(r["global_NLPD_contribution"] for r in case_rows if r["method"] == m and r["candidate_id"] == cid and r["global_NLPD_contribution"] is not None) if m != "HEAD" else None for m in METHODS for cid in case_ids}
    for target, reference in COMPARISONS:
        if reference in ("HEAD", "RANDOM5"): continue
        for cid in case_ids:
            contributions.append({"comparison": target + "_vs_" + reference, "candidate_id": cid, "target_mean_case_nlpd": by_case[target, cid],
                "reference_mean_case_nlpd": by_case[reference, cid], "global_nlpd_improvement_contribution": weighted[reference, cid] - weighted[target, cid]})
    case_decomposition = {}
    for name in {r["comparison"] for r in contributions}:
        csum = math.fsum(r["global_nlpd_improvement_contribution"] for r in contributions if r["comparison"] == name)
        total = pair_means[name]["nlpd_accuracy_units"]
        case_decomposition[name] = {"total_nlpd_improvement": total, "frozen_case_set_contribution": csum, "outside_case_contribution_posthoc_only": total - csum}
    summary.update(engineering_status="COMPLETE_VALID", new_selected_model_count=30, prescribed_trajectory_count=70, parent_reference_count=30,
        historical_reference_count=90, raw_scoring_units=150, new_candidate_rows=1074, all_candidate_rows=len(candidates), fixed_head_mask_count=30,
        method_means=means, paired_means=pair_means, headroom_subset_count=len(rooms), headroom_subset_ids=rooms, headroom_conditional_means=headroom,
        headroom_condition="headroom_pp > 1e-9; descriptive eligible-subset means, full ten-subset main results retained",
        break_best_raw_unit_counts={m: {"count": sum(r["break_best_rate"] for r in units if r["method"] == m), "raw_units": 30 if m in HISTORY else 10} for m in METHODS},
        headroom_break_best_raw_unit_counts={m: {"count": sum(r["break_best_rate"] for r in units if r["method"] == m and r["subset_id"] in rooms), "raw_units": len(rooms) * (3 if m in HISTORY else 1)} for m in METHODS},
        C_selected_start_counts={str(k): sum(r["method"] == "C_REFIT" and r["selected_start"] == k for r in optimization) for k in {r["selected_start"] for r in optimization if r["method"] == "C_REFIT"}},
        coverage_target_distances_of_method_means={m: {r: {str(level): abs(means[m][r + f"__coverage{level}_fraction"] - level / 100) if means[m][r + f"__coverage{level}_fraction"] is not None else None for level in Z} for r in REGIONS} for m in METHODS},
        case_trace_status=case_summary["trace_status"], case_trace_unresolved=case_summary["trace_status"] == "TRACE_UNRESOLVED", case_ids=case_ids,
        case_nlpd_decomposition=case_decomposition, A_repeated_development=True, prior_historical_Q_access_disclosed=True, next_stage_authorized=False,
        no_neural_training=True, no_global_optimality_claim=True, no_absolute_probability_guarantee=True, model_seed=None, execution_rng_seed=0,
        budget=budget, NLPD_units="natural log, original accuracy proportion, nat/candidate", error_width_IS90_units="percentage points", coverage_units="fraction",
        predictions_manifest_sha256=base.sha256(root / "predictions_manifest.json"), seal_sha256=base.sha256(root / "seal.json"), labels_A_sha256=base.sha256(root / "scoring/labels_A.json"),
        scorer_sha256=base.sha256(__file__), statistical_seconds=time.perf_counter() - start)
    require(len(units) == 150 and len(candidates) == 5370 and len(optimization) == 30, "Final scoring count mismatch")
    tables = {"unit_metrics": units, "subset_metrics": subset, "fold_metrics": folds, "sample_seed_metrics": samples, "candidate_metrics": candidates,
              "region_unit_metrics": regions, "head_mask_metrics": mask_rows, "region_subset_metrics": aggregate(regions, ("method", "region", "subset_id", "fold", "sample_seed"), REGION_METRICS),
              "paired_subset": pairs, "paired_fold": pair_folds, "paired_sample_seed": pair_samples, "optimization": optimization, "random5_reference": random_rows,
              "headroom_subset_metrics": [r for r in subset if r["subset_id"] in rooms], "case_details": case_rows, "case_nlpd_contributions": contributions}
    tables["region_fold_metrics"] = aggregate(tables["region_subset_metrics"], ("method", "region", "fold"), REGION_METRICS)
    tables["region_sample_seed_metrics"] = aggregate(tables["region_subset_metrics"], ("method", "region", "sample_seed"), REGION_METRICS)
    for name, rows in tables.items(): base.write_csv(root / "scoring" / (name + ".csv"), rows)
    base.write_json(root / "scoring/summary.json", summary)
    return summary


def synthetic_diagnostic(method, a=0., architecture_gradient=.1):
    x = [0., 0., math.log(.01)] + ([a, 0.] if method == "C_REFIT" else [0.])
    bounds = prescribed_bounds(method); gradient = [0.] * len(x)
    if method == "C_REFIT": gradient[3] = architecture_gradient
    probes = []
    for i in range(len(x)):
        for direction in (-1, 1):
            point = list(x); point[i] += direction * .001
            low, high = bounds[i]
            if low is not None: point[i] = max(low, point[i])
            if high is not None: point[i] = min(high, point[i])
            probe = {"coordinate": i, "direction": direction, "probe_coordinates": point, "skipped_no_displacement": point[i] == x[i]}
            if not probe["skipped_no_displacement"]: probe["nll"] = 1.
            probes.append(probe)
    sources = ("parent_C", "A_REFIT", "B_REFIT") if method == "C_REFIT" else ("parent",)
    return {"method": method, "coordinates": x, "bounds": bounds, "independent_gradient": gradient, "selected_nll": 1.,
            "known_feasible_nlls": [{"source": source, "nll": 1.} for source in sources], "probes": probes,
            "selected_start_index": 2 if method == "C_REFIT" else 0, "selected_iteration": 0, "selected_minus_numerical_minimum": 0.}


def synthetic_checks():
    import audit_optcheck as independent
    start = time.perf_counter(); checks = []
    ids, y, mu = list("abcdefg"), [.1, .2, .3, .4, .5, .7, .7], [.9, .8, .7, .6, .5, .4, .3]
    result, _ = metric_math.evaluate(ids, mu, y, [.0125] * 7, .55)
    require(result["selected_top5_ids"] == list("abcde") and result["champion_ids"] == ["f", "g"] and result["champion_prediction_rank"] == 6, "Selection/tie identities")
    independent_selection = independent.selection(ids, mu, y, .55)
    for k in SELECTION: close(result[k], independent_selection[k], "Independent selection " + k)
    rr = random_reference(y, .55); combinations = list(itertools.combinations(range(7), 5))
    close(rr["expected_regret5_pp"], base.mean([100 * (max(y) - max(y[i] for i in c)) for c in combinations]), "Exact random regret")
    close(rr["expected_gain5_pp"], base.mean([100 * max(0, max(y[i] for i in c) - .55) for c in combinations]), "Exact random gain")
    for variance in (None, [.0125] * 7):
        for ix in ([0, 2, 6], [5, 6]):
            a, b = region_metrics(ids, mu, y, variance, ix), independent.region(mu, y, variance, ix)
            for k in a:
                if a[k] is None: require(b[k] is None, "HEAD probabilities must be NA")
                else: close(a[k], b[k], "Independent probability/region " + k)
    checks.append("selection_gain_headroom_champion_ties_random5_exact_and_independent_regions")
    for method in NEW:
        d = synthetic_diagnostic(method); a = optimization_decision(d, method); b = independent.optimization(d)
        for k in ("KNOWN_FEASIBLE_DOMINANCE_PASS", "LOCAL_STATIONARITY_PASS", "LOCAL_PROBE_PASS", "joint_local_pass", "architecture_disabled"):
            require(a[k] == b[k], "Independent optimization decision differs")
        require(a["joint_local_pass"], "Valid local fixture failed")
    wrong = synthetic_diagnostic("C_REFIT", architecture_gradient=-2e-5)
    require(not optimization_decision(wrong, "C_REFIT")["LOCAL_STATIONARITY_PASS"], "Wrong negative boundary gradient accepted")
    near = synthetic_diagnostic("C_REFIT", a=1e-11)
    require(not optimization_decision(near, "C_REFIT")["LOCAL_STATIONARITY_PASS"] and not optimization_decision(near, "C_REFIT")["architecture_disabled"], "Small positive a must not be declared zero")
    probe = synthetic_diagnostic("A_REFIT"); probe["probes"][0]["nll"] = 1 - 2e-6
    require(not optimization_decision(probe, "A_REFIT")["LOCAL_PROBE_PASS"], "Improving probe ignored")
    known = synthetic_diagnostic("B_REFIT"); known["known_feasible_nlls"][0]["nll"] = 1 - 2e-6
    require(not optimization_decision(known, "B_REFIT")["KNOWN_FEASIBLE_DOMINANCE_PASS"], "Known feasible violation ignored")
    checks.append("closed_boundary_projection_near_positive_a_probe_and_known_feasible_failures")
    rows, random_rows, optimization = [], [], []
    for fold in range(5):
        for sample in SEEDS:
            sid = f"{sample}_{fold}"
            random_rows.append({"subset_id": sid, "fold": fold, "sample_seed": sample, **random_reference(y, .55)})
            for method in METHODS:
                for seed in range(3) if method in HISTORY else [None]:
                    r = {"method": method, "subset_id": sid, "fold": fold, "sample_seed": sample, "model_seed": seed, **{k: 1. for k in METRICS}}
                    r["regret5_pp"] = .8 if method == "C_REFIT" else 1. + (seed or 0) if method in HISTORY else 1.
                    r["gain5_pp"] = .18 if method == "C_REFIT" else .2
                    rows.append(r)
                if method in NEW: optimization.append({"method": method, "subset_id": sid, **optimization_decision(synthetic_diagnostic(method), method)})
    subs, folds, samples, means = hierarchy(rows); pairs, pf, ps, pm = paired(subs, random_rows)
    require(len(rows) == 150 and len(subs) == 90 and len(folds) == 45 and len(samples) == 18 and len(pairs) == 100, "Fixed scientific aggregation counts")
    require(all(r["regret5_pp"] == 2. for r in subs if r["method"] in HISTORY), "History 3seed metric averaging")
    flags = scientific_flags(means, pm, pf, ps, optimization, True)["flags"]
    require(all(flags.values()), "Positive threshold fixture must pass all eleven flags")
    require(flags == independent.derive_flags(means, pm, pf, ps, optimization, True), "Independent complete scientific rules")
    # C exactly selects the B boundary: preserving that feasible state cannot by itself establish architecture gain.
    pm["C_REFIT_vs_B_REFIT"]["regret5_pp"] = 0
    same = scientific_flags(means, pm, pf, ps, optimization, True)["flags"]
    require(not same["ARCHITECTURE_INCREMENT_SIGNAL"] and not same["DEVELOPMENT_CANDIDATE_FOR_CONFIRMATION_DESIGN"], "B boundary mistaken for architecture success")
    pm["C_REFIT_vs_B_REFIT"]["regret5_pp"] = .2 - 2 * TOL
    require(not scientific_flags(means, pm, pf, ps, optimization, True)["flags"]["ARCHITECTURE_INCREMENT_SIGNAL"], "Outside numerical border tolerance accepted")
    pm["C_REFIT_vs_B_REFIT"]["regret5_pp"] = .2
    means["A_OLD"]["nlpd_accuracy_units"] = means["C_REFIT"]["nlpd_accuracy_units"] - .11
    risk = scientific_flags(means, pm, pf, ps, optimization, True)["flags"]
    require(risk["PROBABILITY_NO_LARGE_DEGRADATION_VS_A_REFIT"] and not risk["PROBABILITY_NO_LARGE_DEGRADATION_VS_A_OLD"] and not risk["PROBABILITY_NO_LARGE_DEGRADATION"], "Parent-A risk cannot be masked by relative refit comparison")
    checks.append("150_units_history_metrics_ten_pairs_all_flags_boundary_is_not_architecture_success_double_risk")
    one = region_metrics(ids, mu, y, [.0125] * 7, [0, 1]); two = region_metrics(ids, mu, y, [.0125] * 7, [5, 6])
    require(not math.isclose((one["rmse_pp"] + two["rmse_pp"]) / 2, region_metrics(ids, mu, y, [.0125] * 7, [0, 1, 5, 6])["rmse_pp"]), "Masks must score nonlinear metrics separately")
    checks.append("fixed_HEAD_masks_keep_separate_nonlinear_metrics")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("seal.json", "predictions_manifest.json"): base.write_json(root / name, {"sealed": True, "execution_status": "PARTIAL"})
        for fn, error in ((sealed_inputs, ValueError), (independent.gate, AssertionError)):
            try: fn(root)
            except error: pass
            else: raise AssertionError("Partial matrix accepted before label read")
    checks.append("both_full_seal_gates_reject_partial_before_Q_label_read")
    return {"pass": True, "checks": checks, "statistical_seconds": time.perf_counter() - start, "model_seconds": 0, "nll_gradient_requests": 0,
            "real_Q_labels_accessed": False, "models_imported_or_called": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path); parser.add_argument("--labels-from", type=Path); parser.add_argument("--synthetic-checks", action="store_true")
    args = parser.parse_args()
    if args.synthetic_checks: output = synthetic_checks()
    else:
        require(args.root is not None, "--root is required")
        output = run(args.root.resolve(), args.labels_from)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, allow_nan=False))
