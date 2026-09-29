#!/usr/bin/env python3
"""Independent optcheck audit: standard library only, raw sealed arrays, no official-score imports."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
NEW=("A_REFIT","B_REFIT","C_REFIT")
PARENT=("A_OLD","B_OLD","C_OLD")
HISTORY=("HEAD","H_CONST","H_TREND")
ALL=NEW+PARENT+HISTORY
SEEDS=(20260922,20260923)
Z={50:.6744897501960817,80:1.2815515655446004,90:1.6448536269514722,95:1.959963984540054}
EPS=1e-9
REGIONS=("all_Q","HEAD_pred20","true_top20","true_bottom20","champions")
A_IDS_SHA256="d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"
COMPARISONS=(("C_REFIT","B_REFIT"),("C_REFIT","A_REFIT"),("A_REFIT","A_OLD"),("B_REFIT","B_OLD"),("C_REFIT","C_OLD"),("C_REFIT","HEAD"),("C_REFIT","H_CONST"),("C_REFIT","H_TREND"),("C_REFIT","RANDOM5"),("C_REFIT","A_OLD"))


def ensure(ok, text):
    if not ok: raise AssertionError(text)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    v = [x for x in values if x is not None]
    return math.fsum(v) / len(v) if v else None


def checked(root, path, sha):
    child = (root / path).resolve()
    ensure(child.is_relative_to(root.resolve()), "Escaping path")
    ensure(digest(child) == sha, "Frozen bytes changed: " + path)
    return child


def within(actual, wanted, name):
    if wanted is None: ensure(actual is None, "Expected N/A: " + name); return 0.
    ensure(isinstance(actual, (int, float)) and math.isfinite(actual), "Nonfinite/missing " + name)
    delta = abs(actual - wanted)
    ensure(math.isclose(actual, wanted, abs_tol=1e-8, rel_tol=1e-10), "Independent mismatch: " + name + f" ({actual} != {wanted})")
    return delta


def midranks(values):
    order = sorted(range(len(values)), key=values.__getitem__); result = [0.] * len(values); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        for k in range(i, j): result[order[k]] = (i + j + 1) / 2
        i = j
    return result


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2: return None
    a, b = midranks(x), midranks(y); ac, bc = mean(a), mean(b)
    cov = math.fsum((i - ac) * (j - bc) for i, j in zip(a, b))
    denom = math.sqrt(math.fsum((i - ac) ** 2 for i in a) * math.fsum((j - bc) ** 2 for j in b))
    return max(-1., min(1., cov / denom))


def region(pred, truth, variance, ix):
    err = [pred[i] - truth[i] for i in ix]
    out = {"mae_pp": mean([abs(v) for v in err]) * 100, "rmse_pp": math.sqrt(mean([v * v for v in err])) * 100, "bias_pp": mean(err) * 100}
    if variance is None:
        out["nlpd_accuracy_units"] = out["IS90_pp"] = None
        for level in Z:
            out[f"coverage{level}_fraction"] = out[f"interval{level}_width_pp"] = out[f"coverage{level}_target_distance_fraction"] = None
        return out
    sigmas = [math.sqrt(variance[i]) for i in ix]
    out["nlpd_accuracy_units"] = mean([math.log(math.sqrt(2 * math.pi * variance[i])) + (truth[i] - pred[i]) ** 2 / (2 * variance[i]) for i in ix])
    scores = []
    for i, sigma in zip(ix, sigmas):
        lower, upper = pred[i] - Z[90] * sigma, pred[i] + Z[90] * sigma
        scores.append((upper - lower + 20 * max(lower - truth[i], 0) + 20 * max(truth[i] - upper, 0)) * 100)
    out["IS90_pp"] = mean(scores)
    for level, z in Z.items():
        cv = mean([float(abs(e) <= z * sigma) for e, sigma in zip(err, sigmas)])
        out[f"coverage{level}_fraction"] = cv
        out[f"interval{level}_width_pp"] = mean([2 * z * sigma * 100 for sigma in sigmas])
        out[f"coverage{level}_target_distance_fraction"] = abs(cv - level / 100)
    return out


def selection(ids, pred, truth, incumbent):
    order = sorted(range(len(ids)), key=lambda k: (-pred[k], ids[k])); top = order[:5]; best = max(truth)
    champ = [k for k, value in enumerate(truth) if abs(best - value) <= EPS / 100]
    chosen_best = max(truth[k] for k in top); gain = max(0, chosen_best - incumbent) * 100; room = max(0, best - incumbent) * 100
    return {"regret5_pp": (best - chosen_best) * 100, "gain5_pp": gain, "headroom_pp": room, "simple_regret_pp": room - gain,
            "break_best_rate": float(chosen_best - incumbent > EPS / 100), "top1_regret_pp": (best - truth[order[0]]) * 100,
            "champion_prediction_rank": min(order.index(k) + 1 for k in champ), "champion_top5_hit_fraction": float(any(k in top for k in champ)),
            "spearman": spearman(pred, truth), "ranking_ids": [ids[k] for k in order], "selected_top5_ids": [ids[k] for k in top], "champion_ids": [ids[k] for k in champ]}


def average_group(rows, keys, fields):
    groups = defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, k)), **{f: mean([r[f] for r in values]) for f in fields}} for k, values in groups.items()]


def read_csv(path):
    out = []
    for row in csv.DictReader(Path(path).open()):
        parsed = {}
        for k, v in row.items():
            if v == "": parsed[k] = None
            elif v in ("True", "False"): parsed[k] = v == "True"
            else:
                try: parsed[k] = json.loads(v)
                except (ValueError, TypeError): parsed[k] = v
        out.append(parsed)
    return out


def compare_table(ours, path, keys, fields):
    theirs = read_csv(path); lookup = {tuple(r[k] for k in keys): r for r in theirs}; maximum = 0.
    ensure(len(theirs) == len(lookup) == len(ours), "Table row count mismatch " + str(path))
    for row in ours:
        other = lookup[tuple(row[k] for k in keys)]
        for k in fields: maximum = max(maximum, within(other[k], row[k], str(path.name) + ":" + k))
    return maximum


def optimization(d):
    method = d["method"]; x, g = d["coordinates"], d["independent_gradient"]
    expected = [[None, None], [math.log(1e-8), None], [math.log(1e-6), None]]
    expected += [[0, math.log1p(1e8)], [math.log(1e-4), None]] if method == "C_REFIT" else [[math.log(1e-4), None]]
    ensure(d["bounds"] == expected and len(x) == len(g) == len(expected), "Coordinate domain changed")
    projection = []
    for i, (lo, hi) in enumerate(expected):
        ensure(math.isfinite(x[i]) and math.isfinite(g[i]), "Nonfinite independent stationarity diagnostic")
        val = g[i]
        if lo is not None:
            tolerance = 1e-12 * max(1, abs(lo)); ensure(x[i] >= lo - tolerance, "Lower bound violated")
            if abs(x[i] - lo) <= tolerance: val = min(0, val)
        if hi is not None:
            tolerance = 1e-12 * max(1, abs(hi)); ensure(x[i] <= hi + tolerance, "Upper bound violated")
            if abs(x[i] - hi) <= tolerance: val = max(0, val)
        projection.append(val)
    norm = max(abs(v) for v in projection); wanted = {}
    for i, (lo, hi) in enumerate(expected):
        for direction in [-1, 1]:
            point = list(x); point[i] += direction / 1000
            if lo is not None and point[i] < lo: point[i] = lo
            if hi is not None and point[i] > hi: point[i] = hi
            if point[i] != x[i]: wanted[i, direction] = point
    active = [p for p in d["probes"] if not p["skipped_no_displacement"]]
    ensure(len(active) == len(wanted) and {(p["coordinate"], p["direction"]) for p in active} == set(wanted), "Local probe directions not complete")
    improvements = []
    for p in active:
        point = wanted[p["coordinate"], p["direction"]]
        ensure(len(point) == len(p["probe_coordinates"]) and all(math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-10) for a, b in zip(point, p["probe_coordinates"])), "Wrong fixed probe coordinate")
        ensure(math.isfinite(p["nll"]), "Nonfinite probe objective")
        improvements.append(d["selected_nll"] - p["nll"])
    sources = {v["source"] for v in d["known_feasible_nlls"]}
    ensure(sources == ({"parent_C", "A_REFIT", "B_REFIT"} if method == "C_REFIT" else {"parent"}), "Incomplete known-feasible pool")
    differences = {v["source"]: d["selected_nll"] - v["nll"] for v in d["known_feasible_nlls"]}
    ensure(all(math.isfinite(v) for v in differences.values()), "Known feasible objective invalid")
    probe_ok = max(improvements, default=0.) <= 1e-6
    return {"KNOWN_FEASIBLE_DOMINANCE_PASS": all(v <= 1e-6 for v in differences.values()), "LOCAL_STATIONARITY_PASS": norm <= 1e-5,
            "LOCAL_PROBE_PASS": probe_ok, "joint_local_pass": norm <= 1e-5 and probe_ok, "projected_gradient_inf": norm,
            "max_probe_nll_improvement": max(improvements, default=0.), "selected_nll": d["selected_nll"],
            "architecture_disabled": x[3] == 0 if method == "C_REFIT" else None}


def gate(root):
    seal, manifest = read(root / "seal.json"), read(root / "predictions_manifest.json")
    ensure(all(v.get("sealed") is True and v.get("execution_status") == "COMPLETE_VALID" for v in (seal, manifest)), "Incomplete/invalid matrix cannot release labels")
    checked(root, "predictions_manifest.json", seal["predictions_manifest_sha256"])
    for fields, value in ((("new_endpoint_count",), 30), (("parent_reference_count",), 30), (("historical_reference_count",), 90),
                           (("raw_unit_count",), 150), (("mask_count",), 30), (("completed_trajectories", "trajectory_count"), 70),
                           (("new_candidate_rows",), 1074), (("all_candidate_rows", "scoring_candidate_rows"), 5370),
                           (("independent_restoration_count",), 30), (("local_diagnostic_count",), 30)):
        values = [manifest[k] for k in fields if k in manifest]; ensure(values and all(v == value for v in values), "Full-seal accounting missing")
    for name in ("protocol", "source_manifest", "code_manifest", "worklist", "masks_manifest", "frozen_manifest", "data_manifest", "case_audit_manifest"):
        checked(root, name + ".json", manifest[name + "_sha256"])
    for m in [manifest] + [read(root / (n + ".json")) for n in ("source_manifest", "code_manifest", "frozen_manifest", "masks_manifest", "data_manifest")]:
        for path, sha in m["files"].items(): checked(root, path, sha)
    budget = read(root / "budget.json"); protocol = read(root / "protocol.json")
    ensure(budget["limit"] == protocol["budget"].get("model_seconds", protocol["budget"].get("limit")) == 600 and budget["seconds"] <= 600, "Invalid model budget")
    ensure(0 <= budget["optimization_nll_gradient_requests"] <= 18000 and 0 <= budget["synthetic_nll_gradient_requests"] <= 300, "Invalid objective-call budgets")
    ensure(budget["optimization_nll_gradient_requests"] == budget["counts"]["optimization"]["nll_gradient_requests"] and budget["synthetic_nll_gradient_requests"] == budget["counts"]["synthetic"]["nll_gradient_requests"], "Objective accounting aliases differ")
    records = manifest["records"]
    ensure(len(records) == len({r["unit_id"] for r in records}) == 150, "150 distinct raw units required")
    expected = {(f, s, None, m) for f in range(5) for s in SEEDS for m in NEW + PARENT} | {(f, s, k, m) for f in range(5) for s in SEEDS for k in range(3) for m in HISTORY}
    ensure({(r["fold"], r["sample_seed"], r["model_seed"], r["method"]) for r in records} == expected, "Scientific matrix differs")
    ref_rows = read(root / "historical_refs.json")["records"]; refs = {r["unit_id"]: r for r in ref_rows}
    ensure(len(ref_rows) == len(refs) == 120, "120 byte-preserved references required")
    original_methods = {"A_OLD": "RAW16_ISO", "B_OLD": "HP4_ISO", "C_OLD": "RAW16_BLOCK2", "HEAD": "HEAD", "H_CONST": "GP_CONST", "H_TREND": "GP_TREND"}
    items = []; old_heads = {}; diagnostics = []
    for r in records:
        ensure(manifest["files"].get(r["path"]) == r["sha256"], "Unsealed prediction")
        p = read(checked(root, r["path"], r["sha256"]))
        ensure(all(p[k] == r[k] for k in ("subset_id", "fold", "sample_seed", "model_seed")), "Prediction identity differs")
        if r["method"] in NEW:
            ensure(p["method"] == r["method"] and p.get("rng_seed", p.get("execution_rng_seed")) == 0, "New method identity differs")
            state = read(checked(root, p["state_path"], p["state_file_sha256"]))
            d = read(checked(root, p["diagnostics_path"], p["diagnostics_sha256"]))
            ensure(d["unit_id"] == r["unit_id"] and d["method"] == r["method"] and d["subset_id"] == r["subset_id"], "Independent optimization diagnostic identity")
            ensure(state["coordinates"] == d["coordinates"] == p["selected"]["coordinates"] and state["method"] == r["method"], "Selected state coordinates differ")
            ensure(math.isclose(d["selected_nll"], p["selected"]["nll"], abs_tol=1e-10, rel_tol=1e-8), "Selected objective identity differs")
            diagnostics.append({**{k: r[k] for k in ("unit_id", "method", "subset_id", "fold", "sample_seed")}, **optimization(d)})
        else:
            ensure(p["method"] == original_methods[r["method"]] and all(refs[r["unit_id"]][k] == r[k] for k in ("path", "sha256", "method", "subset_id", "fold", "sample_seed", "model_seed")), "Old prediction/reference changed")
        if r["method"] == "HEAD": old_heads[r["subset_id"], r["model_seed"]] = (r, p)
        items.append((r, p))
    masks = read(root / "masks_manifest.json")["records"]
    ensure(len(masks) == len({(m["subset_id"], m["model_seed"]) for m in masks}) == 30, "Masks incomplete")
    for mask in masks:
        r, p = old_heads[mask["subset_id"], mask["model_seed"]]
        order = sorted(zip(p["candidate_ids"], p["mean_raw"]), key=lambda v: (-v[1], v[0]))
        ensure(mask["candidate_ids"] == [i for i, _ in order[:math.ceil(len(order) / 5)]] and mask["query_ids"] == p["candidate_ids"], "HEAD mask definition changed")
        ensure(mask["source_prediction_sha256"] == r["sha256"], "Mask source hash changed")
        saved = read(checked(root, mask["path"], mask["sha256"]))
        ensure(saved["candidate_ids"] == mask["candidate_ids"] and saved["query_ids"] == mask["query_ids"], "Mask file identity differs")
    case_summary = read(checked(root, "case_audit/summary.json", manifest["files"]["case_audit/summary.json"]))
    ensure(case_summary["confirmed_data_inconsistency"] is False and case_summary["case_manifest_sha256"] == digest(root / "case_audit_manifest.json"), "Known source inconsistency or case identity error")
    release = read(root / "scoring/release.json")
    ensure(release["predictions_manifest_sha256"] == digest(root / "predictions_manifest.json") and release["seal_sha256"] == digest(root / "seal.json"), "Label release preceded valid full seal")
    source = read(root / "source_manifest.json")
    ensure(digest(root / "scoring/labels_A.json") == source.get("legal_A_scoring_truth_sha256", source.get("expected_labels_sha256")) == release["labels_sha256"], "Released A labels differ")
    return items, masks, budget, diagnostics, case_summary


def derive_flags(means, pair_means, folds, samples, diagnostics, source_ok):
    def improvement(reference):
        key = "C_REFIT_vs_" + reference; values = pair_means[key]
        return (values["regret5_pp"] + EPS >= .2 and values["gain5_pp"] + EPS >= -.02
                and sum(r["regret5_pp"] > EPS for r in folds if r["comparison"] == key) >= 4
                and all(r["regret5_pp"] > EPS for r in samples if r["comparison"] == key))
    retains = {m: means["C_REFIT"]["regret5_pp"] - means[m]["regret5_pp"] <= .1 + EPS and means["C_REFIT"]["gain5_pp"] - means[m]["gain5_pp"] >= -.02 - EPS for m in ("HEAD", "H_TREND")}
    risks = {}
    for m in ("A_REFIT", "A_OLD"):
        a, b = means["C_REFIT"], means[m]
        risks[m] = (a["nlpd_accuracy_units"] - b["nlpd_accuracy_units"] <= .1 + EPS and a["IS90_pp"] - b["IS90_pp"] <= 1 + EPS
                    and (a["coverage90_fraction"] - b["coverage90_fraction"]) * 100 >= -5 - EPS
                    and (a["champions__coverage90_fraction"] - b["champions__coverage90_fraction"]) * 100 >= -5 - EPS)
    known = all(d["KNOWN_FEASIBLE_DOMINANCE_PASS"] for d in diagnostics)
    adequate = known and all(sum(d["joint_local_pass"] for d in diagnostics if d["method"] == m) >= 8 for m in NEW)
    flags = {"KNOWN_FEASIBLE_DOMINANCE_PASS": known, "OPTIMIZATION_ADEQUATE": adequate, "ARCHITECTURE_INCREMENT_SIGNAL": improvement("B_REFIT"),
             "BLOCK_SCALE_SIGNAL": improvement("A_REFIT"), "HISTORICAL_SELECTION_RETAINED_VS_HEAD": retains["HEAD"], "HISTORICAL_SELECTION_RETAINED_VS_H_TREND": retains["H_TREND"],
             "HISTORICAL_SELECTION_RETAINED": all(retains.values()), "PROBABILITY_NO_LARGE_DEGRADATION_VS_A_REFIT": risks["A_REFIT"],
             "PROBABILITY_NO_LARGE_DEGRADATION_VS_A_OLD": risks["A_OLD"], "PROBABILITY_NO_LARGE_DEGRADATION": all(risks.values())}
    flags["DEVELOPMENT_CANDIDATE_FOR_CONFIRMATION_DESIGN"] = bool(source_ok and adequate and improvement("B_REFIT") and improvement("A_REFIT") and all(retains.values()) and all(risks.values()))
    return flags


def run(root):
    start = time.perf_counter(); items, masks, budget, diagnostics, case_summary = gate(root)
    labels = read(root / "scoring/labels_A.json")["labels"]
    labels_identity = hashlib.sha256(json.dumps(sorted(labels), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    ensure(len(labels) == 179 and labels_identity == A_IDS_SHA256, "Legal A label pool differs")
    case_ids = read(root / "case_audit_manifest.json")["candidate_ids"]
    ensure(case_ids == case_summary["candidate_ids"] and len(set(case_ids)) == len(case_ids), "Case set changed")
    mask_lut = {(m["subset_id"], m["model_seed"]): m["candidate_ids"] for m in masks}
    units, regions, mask_rows, random_rows, snapshots, case_parts = [], [], [], [], {}, []
    candidate_count = 0; new_count = 0
    for rec, pred in items:
        method, sid = rec["method"], rec["subset_id"]; ids = pred["candidate_ids"]; mu = pred["mean_raw"]
        ensure(len(ids) == len(set(ids)) == (38, 36, 36, 34, 35)[rec["fold"]], "Query count/order invalid")
        candidate_count += len(ids); new_count += len(ids) if method in NEW else 0
        y = [labels[i]["validation_mean"] for i in ids]; ses = [labels[i]["validation_se"] for i in ids]
        if sid not in snapshots:
            t = read(root / "training" / (sid + ".json")); snapshots[sid] = t
            ensure(len(t["ids"]) == len(t["y"]) == 50 and not set(t["ids"]) & set(ids), "Training/query identity invalid")
            incumbent = max(t["y"]); ys = sorted(y); denominator = math.comb(len(y), 5)
            distribution = [(value, math.comb(j, 4) / denominator) for j, value in enumerate(ys) if j >= 4]
            room = 100 * max(0, max(y) - incumbent); egain = math.fsum(100 * max(0, v - incumbent) * p for v, p in distribution)
            nchamp = sum(max(y) - value <= EPS / 100 for value in y)
            random_rows.append({"subset_id": sid, "expected_regret5_pp": math.fsum((max(y) - v) * p * 100 for v, p in distribution),
                "expected_gain5_pp": egain, "headroom_pp": room, "expected_simple_regret_pp": room - egain,
                "break_best_probability": math.fsum(p for v, p in distribution if v - incumbent > EPS / 100),
                "champion_hit_probability": 1 - (math.comb(len(y) - nchamp, 5) / denominator if len(y) - nchamp >= 5 else 0)})
        if method == "HEAD": obs = None
        else:
            latent = pred["latent_variance_standardized"]; scale2 = pred["y_scale"] ** 2; extra = pred["extra_noise_variance_standardized"]
            ensure(len(latent) == len(ids) and all(v >= -1e-8 and math.isfinite(v) for v in latent), "Invalid latent diagonal")
            vm = [scale2 * max(v, 0) + scale2 * extra for v in latent]
            if "model_variance_raw" in pred:
                for saved, original, corrected in zip(pred["model_variance_raw"], [scale2 * (v + extra) for v in latent], vm):
                    ensure(math.isclose(saved, original, abs_tol=1e-12, rel_tol=1e-10) or math.isclose(saved, corrected, abs_tol=1e-12, rel_tol=1e-10), "Variance components incorrect")
            ensure(pred.get("tau_squared_raw", 0) in (0, None), "Unexpected calibration variance")
            obs = [v + se * se for v, se in zip(vm, ses)]
        meta = {k: rec[k] for k in ("unit_id", "method", "subset_id", "fold", "sample_seed", "model_seed")}
        row = {**meta, **selection(ids, mu, y, max(snapshots[sid]["y"]))}
        groups = {"all_Q": list(range(len(ids))), "true_top20": sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(len(ids) / 5)],
                  "true_bottom20": sorted(range(len(ids)), key=lambda i: (y[i], ids[i]))[:math.ceil(len(ids) / 5)],
                  "champions": [i for i, value in enumerate(y) if max(y) - value <= EPS / 100]}
        for name, ix in groups.items():
            stats = region(mu, y, obs, ix); regions.append({**meta, "region": name, **stats})
            row.update({name + "__" + k: v for k, v in stats.items()})
            if name == "all_Q": row.update({k: v for k, v in stats.items() if "target_distance" not in k})
        local_masks = []
        for seed in [rec["model_seed"]] if method in HISTORY else range(3):
            ix = [ids.index(i) for i in mask_lut[sid, seed]]; stats = region(mu, y, obs, ix); local_masks.append(stats)
            mask_rows.append({**meta, "head_mask_seed": seed, **stats})
        stats = {k: mean([m[k] for m in local_masks]) for k in local_masks[0]}
        regions.append({**meta, "region": "HEAD_pred20", **stats}); row.update({"HEAD_pred20__" + k: v for k, v in stats.items()}); units.append(row)
        for i, cid in enumerate(ids):
            if cid not in case_ids: continue
            nlpd = None if obs is None else math.log(math.sqrt(2 * math.pi * obs[i])) + (mu[i] - y[i]) ** 2 / (2 * obs[i])
            weight = 1 / (10 * (3 if method in HISTORY else 1) * len(ids))
            case_parts.append({**meta, "candidate_id": cid, "nlpd_accuracy_units": nlpd, "global_NLPD_contribution": None if nlpd is None else nlpd * weight,
                               "observed_variance_raw": None if obs is None else obs[i], "mean_raw": mu[i], "prediction_rank": row["ranking_ids"].index(cid) + 1})
    fields = [k for k in units[0] if k not in ("unit_id", "method", "subset_id", "fold", "sample_seed", "model_seed", "ranking_ids", "selected_top5_ids", "champion_ids")]
    subsets = average_group(units, ("method", "subset_id", "fold", "sample_seed"), fields)
    folds = average_group(subsets, ("method", "fold"), fields); samples = average_group(subsets, ("method", "sample_seed"), fields)
    means = {m: {k: mean([r[k] for r in folds if r["method"] == m]) for k in fields} for m in ALL}
    pairs = []; lut = {(r["subset_id"], r["method"]): r for r in subsets}
    for r in random_rows:
        lut[r["subset_id"], "RANDOM5"] = {**{k: None for k in fields}, "regret5_pp": r["expected_regret5_pp"], "gain5_pp": r["expected_gain5_pp"],
            "headroom_pp": r["headroom_pp"], "simple_regret_pp": r["expected_simple_regret_pp"], "break_best_rate": r["break_best_probability"], "champion_top5_hit_fraction": r["champion_hit_probability"]}
    for target, reference in COMPARISONS:
        for row in [r for r in subsets if r["method"] == target]:
            baseline = lut[row["subset_id"], reference]; p = {k: row[k] for k in ("subset_id", "fold", "sample_seed")}; p["comparison"] = target + "_vs_" + reference
            for k in fields:
                multiplier = 1 if ("coverage" in k and "target_distance" not in k) or k in ("gain5_pp", "break_best_rate", "champion_top5_hit_fraction", "spearman") else -1
                p[k] = None if row[k] is None or baseline[k] is None else multiplier * (row[k] - baseline[k])
            pairs.append(p)
    pf = average_group(pairs, ("comparison", "fold"), fields); ps = average_group(pairs, ("comparison", "sample_seed"), fields)
    pm = {name: {k: mean([r[k] for r in pf if r["comparison"] == name]) for k in fields} for name in {r["comparison"] for r in pairs}}
    flags = derive_flags(means, pm, pf, ps, diagnostics, not case_summary["confirmed_data_inconsistency"])
    opt_counts = {m: {k: sum(bool(r[k]) for r in diagnostics if r["method"] == m) for k in ("KNOWN_FEASIBLE_DOMINANCE_PASS", "LOCAL_STATIONARITY_PASS", "LOCAL_PROBE_PASS", "joint_local_pass")} for m in NEW}
    rooms = {r["subset_id"] for r in subsets if r["headroom_pp"] > EPS}
    headroom = {m: {k: mean([r[k] for r in subsets if r["method"] == m and r["subset_id"] in rooms]) for k in fields} for m in ALL}
    case_contributions = []
    for target, reference in COMPARISONS:
        if reference in ("HEAD", "RANDOM5"): continue
        for cid in case_ids:
            own = [r for r in case_parts if r["method"] == target and r["candidate_id"] == cid]
            ref = [r for r in case_parts if r["method"] == reference and r["candidate_id"] == cid]
            case_contributions.append({"comparison": target + "_vs_" + reference, "candidate_id": cid, "target_mean_case_nlpd": mean([r["nlpd_accuracy_units"] for r in own]),
                "reference_mean_case_nlpd": mean([r["nlpd_accuracy_units"] for r in ref]),
                "global_nlpd_improvement_contribution": math.fsum(r["global_NLPD_contribution"] for r in ref) - math.fsum(r["global_NLPD_contribution"] for r in own)})
    ensure(len(units) == 150 and candidate_count == 5370 and new_count == 1074 and len(subsets) == 90, "Independent count mismatch")
    # Independent calculations above use no official aggregates; compare only now.
    official = read(root / "scoring/summary.json"); maximum = 0.
    ensure(official["flags"] == flags and official["optimization_counts_out_of10"] == opt_counts, "Independent optimization/scientific flags differ")
    ensure(official["C_exact_boundary_count"] == sum(r["method"] == "C_REFIT" and r["architecture_disabled"] for r in diagnostics), "Exact zero-boundary count differs")
    ensure(set(official["headroom_subset_ids"]) == rooms and official["case_trace_status"] == case_summary["trace_status"], "Headroom or trace status differs")
    for m in ALL:
        for k in fields:
            maximum = max(maximum, within(official["method_means"][m][k], means[m][k], m + ":" + k))
            maximum = max(maximum, within(official["headroom_conditional_means"][m][k], headroom[m][k], "headroom:" + m + ":" + k))
    for name, values in pm.items():
        for k, value in values.items(): maximum = max(maximum, within(official["paired_means"][name][k], value, name + ":" + k))
    tables = [(units, "unit_metrics", ("unit_id",), fields), (subsets, "subset_metrics", ("method", "subset_id"), fields),
              (folds, "fold_metrics", ("method", "fold"), fields), (samples, "sample_seed_metrics", ("method", "sample_seed"), fields),
              (pairs, "paired_subset", ("comparison", "subset_id"), fields), (pf, "paired_fold", ("comparison", "fold"), fields),
              (ps, "paired_sample_seed", ("comparison", "sample_seed"), fields), (regions, "region_unit_metrics", ("unit_id", "region"), list(stats)),
              (mask_rows, "head_mask_metrics", ("unit_id", "head_mask_seed"), list(stats)),
              (random_rows, "random5_reference", ("subset_id",), [k for k in random_rows[0] if k != "subset_id"]),
              (case_parts, "case_details", ("unit_id", "candidate_id"), ["nlpd_accuracy_units", "global_NLPD_contribution", "observed_variance_raw", "mean_raw", "prediction_rank"]),
              (case_contributions, "case_nlpd_contributions", ("comparison", "candidate_id"), ["target_mean_case_nlpd", "reference_mean_case_nlpd", "global_nlpd_improvement_contribution"])]
    for own, name, keys, metrics in tables: maximum = max(maximum, compare_table(own, root / "scoring" / (name + ".csv"), keys, metrics))
    saved = {r["unit_id"]: r for r in read_csv(root / "scoring/unit_metrics.csv")}
    for row in units:
        for key in ("ranking_ids", "selected_top5_ids", "champion_ids"): ensure(row[key] == saved[row["unit_id"]][key], "Exact candidate selections differ")
    result = {"pass": True, "no_model_calls": True, "independent_no_official_scorer_imports": True, "raw_scoring_units": 150, "new_selected_models": 30,
        "parent_references": 30, "historical_seed_references": 90, "all_candidate_rows": candidate_count, "new_candidate_rows": new_count,
        "subset_rows": len(subsets), "fold_rows": len(folds), "sample_rows": len(samples), "mask_rows": len(mask_rows), "region_unit_rows": len(regions),
        "paired_subset_rows": len(pairs), "max_absolute_statistic_difference": maximum, "flags": flags, "method_means": means, "paired_means": pm,
        "optimization_counts_out_of10": opt_counts, "C_exact_boundary_count": sum(r["method"] == "C_REFIT" and r["architecture_disabled"] for r in diagnostics),
        "case_contribution_checks": len(case_contributions), "case_trace_status": case_summary["trace_status"], "budget": budget,
        "statistical_seconds": time.perf_counter() - start, "predictions_manifest_sha256": digest(root / "predictions_manifest.json"), "seal_sha256": digest(root / "seal.json"),
        "official_summary_sha256": digest(root / "scoring/summary.json"), "auditor_sha256": digest(__file__),
        "scope": "sealed raw prediction arrays, saved independent gradients/probes and known-feasible objectives, file identities; no model or optimizer replay"}
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/independent_optcheck_scoring.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(); print(json.dumps(run(args.root.resolve()), ensure_ascii=False, sort_keys=True, allow_nan=False))
